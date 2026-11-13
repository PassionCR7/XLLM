# -*- coding: utf-8 -*-
"""
draft_model.py —— 草稿模型(Draft Model)定义

推测解码需要一个“又快又小”的草稿模型去批量提议候选 token，再由主模型
(目标模型)一次并行验证。本文件把主模型的基座 XLLMForCausalLM 复用成一个
更小/更浅的草稿模型，因此：

    - 词表 / 词嵌入 / lm_head 与主模型完全一致（同一套 tokenizer）；
    - 只堆叠更少的 Transformer 层（或更小的 hidden_size），单步前向更便宜；
    - 由 DraftModel.propose_n 连续做 gamma 步自回归，同时把每一步的提议
      分布 q_i 保存下来，供 _verification.verify_and_sample 做接受判断。

三种构建草稿模型的方式（对应不同训练预算）：
    1. DraftModel.from_main_model(main, num_draft_layers)
        零训练“截断式草稿”：直接复用主模型的 嵌入层 + 前 num_draft_layers 层
        + 末层归一化 + lm_head。不需要单独训练，作为起步/基线最实用。
    2. DraftModel(config)
        用一个更小 hidden_size / 更少层的独立配置新建，权重随机；之后可用
        from_pretrained 加载一个单独训练过的草稿 checkpoint（推荐，训练后
        draft 与主模型分布更接近，接受率更高，如 Qwen/DeepSeek/Kimi 的做法）。
    3. DraftModel.from_pretrained(ckpt_path, config)
        加载独立草稿 checkpoint。
"""

from __future__ import annotations

import copy
import os
from typing import Optional

import torch
import torch.nn as nn

from ..XLLMModel import XLLMConfig, XLLMForCausalLM
from ._verification import prepare_distribution, sample_token


class DraftModel(nn.Module):
    """
    草稿模型：对 XLLMForCausalLM 的轻量封装，提供“多步自回归提议”能力。

    self.model 是一个 XLLMForCausalLM 实例（更浅/更小的配置），前向接口与
    主模型一致，方便外部直接用同样的 past_key_values 结构做 KV 同步。
    """

    def __init__(self, config: XLLMConfig):
        super().__init__()
        self.config = config
        self.model = XLLMForCausalLM(config)

    # ------------------------------------------------------------------
    # 前向透传：与主模型 XLLMForCausalLM 接口一致
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
        **kwargs,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # 核心：连续提议 n 个候选 token（自回归，同时记录每步提议分布 q）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def propose_n(
        self,
        last_token: torch.Tensor,          # [B, 1] 当前最后已接受 token
        past_key_values,                   # 草稿的 KV cache，止于 last_token 前
        n: int,                            # 提议多少个候选（= gamma）
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
    ):
        """
        从 last_token 出发，逐 token 自回归提议 n 个候选。

        返回
        ----
        draft_ids   : [B, n]       提议的候选 token id
        draft_logits: [B, n, V]    每步提议时使用的原始 logits（还原分布 q 用）
        past_key_values: 提议 n 步后的草稿 KV cache（止于最后一个候选）
        """
        B = last_token.shape[0]
        ids_list = []
        logits_list = []
        cur = last_token
        kv = past_key_values
        for _ in range(n):
            out = self.model(
                input_ids=cur,
                past_key_values=kv,
                use_cache=True,
                logits_to_keep=1,
            )
            logits = out.logits[:, -1, :]          # [B, V] 预测下一个 token
            next_id = sample_token(
                logits,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                rng=rng,
            )                                      # [B]
            ids_list.append(next_id)
            logits_list.append(logits)
            kv = out.past_key_values
            cur = next_id.unsqueeze(-1)            # [B, 1]

        draft_ids = torch.stack(ids_list, dim=1)          # [B, n]
        draft_logits = torch.stack(logits_list, dim=1)    # [B, n, V]
        return draft_ids, draft_logits, kv

    # ------------------------------------------------------------------
    # 零训练构建：从主模型截断前 num_draft_layers 层作为草稿
    # ------------------------------------------------------------------
    @classmethod
    def from_main_model(
        cls,
        main_model: XLLMForCausalLM,
        num_draft_layers: int,
    ) -> "DraftModel":
        """
        复用主模型的 嵌入层 + 前 num_draft_layers 层 + 末层归一化 + lm_head，
        构建一个更浅的草稿模型。不需要单独训练，权重与主模型共享。

        说明：草稿模型由于更浅，单步前向更快，且与主模型在浅层语义上更接近，
        因而接受率通常不低。若想要更小 hidden_size 的草稿，请改用独立配置
        DraftModel(config) 并训练/加载。
        """
        c = main_model.config
        assert 0 < num_draft_layers <= c.num_hidden_layers, (
            f"num_draft_layers 必须在 (0, {c.num_hidden_layers}] 之间"
        )

        draft_cfg = XLLMConfig(
            dropout=c.dropout,
            hidden_size=c.hidden_size,
            intermediate_size=c.intermediate_size,
            num_attention_heads=c.num_attention_heads,
            num_hidden_layers=num_draft_layers,
            num_key_value_heads=c.num_key_value_heads,
            vocab_size=c.vocab_size,
            max_position_embeddings=c.max_position_embeddings,
            rms_norm_eps=c.rms_norm_eps,
            rope_theta=c.rope_theta,
            inference_rope_scaling=c.inference_rope_scaling,
            flash_attention=c.flash_attention,
            use_moe=c.use_moe,
            num_experts_per_tok=c.num_experts_per_tok,
            n_routed_experts=c.n_routed_experts,
            n_shared_experts=c.n_shared_experts,
            scoring_func=c.scoring_func,
            aux_loss_alpha=c.aux_loss_alpha,
            seq_aux=c.seq_aux,
            norm_topk_prob=c.norm_topk_prob,
        )

        draft = cls(draft_cfg)
        msd = main_model.state_dict()
        dsd = draft.state_dict()
        # 注意键名前缀差异：DraftModel 把 XLLMForCausalLM 包在 self.model 下，
        # 因此草稿的 state_dict 键是 "model.model.layers.N..."，而主模型是
        # "model.layers.N..."。复制时把草稿键开头的 "model." 前缀剥掉再匹配。
        prefix = "model."
        assign = {}
        for key, val in dsd.items():
            mkey = key[len(prefix):] if key.startswith(prefix) else key
            if mkey in msd and msd[mkey].shape == val.shape:
                assign[key] = msd[mkey].detach().clone()
        if assign:
            dsd.update(assign)
            draft.load_state_dict(dsd)
        return draft

    # ------------------------------------------------------------------
    # 加载独立草稿 checkpoint（单独训练过的小模型）
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, ckpt_path: str, config: XLLMConfig) -> "DraftModel":
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"草稿 checkpoint 不存在: {ckpt_path}")
        draft = cls(config)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # 兼容两种保存方式：直接 state_dict，或包裹在别的结构里
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        draft.load_state_dict(state, strict=False)
        return draft

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
