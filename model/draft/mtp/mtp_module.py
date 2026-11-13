# -*- coding: utf-8 -*-
"""
mtp_module.py —— MTP（Multi-Token Prediction）草稿头，用于与主干**联合预训练**

对齐 DeepSeek-V3 / V4 的 MTP 设计（V4 报告：沿用 V3 的 MTP 配置，深度为 1）：

    "在架构层面，每个 MTP 模块与主模型共享输入和输出嵌入，但维护独立的完整
     Transformer 块。"  —— DeepSeek-V4 技术报告 MTP 章节

    L_total = L_next_token + (λ/D) · Σ_k L_MTP^(k)

与 model/draft 里另两套实现的根本区别：

    DSparkDrafter（dspark_draft.py）
        事后训练，target **冻结**，梯度不回传。草稿只能逼近已有的 p。

    MTPHead（本文件）
        预训练期与主干**联合训练**，梯度回传到主干。MTP 损失既是草稿的训练目标，
        也是主干的辅助正则——迫使每个隐状态同时支撑"近处"和"远处"的预测，
        densify 监督信号。这也是论文强调 MTP "顺带提升主干本身性能"的原因。

结构（第 k 个模块，k=1..D）：

    x = proj( concat( RMSNorm(h_{k-1}), RMSNorm(Emb(token_{t+k})) ) )
    h_k = TransformerBlock(x)
    logits_k = lm_head(h_k)            # 与主干共享 lm_head 权重

    h_0 就是主干最后一层的隐状态。h_k[t] 预测 label[t+k+1]。

权重共享：Emb 与 lm_head 直接引用主干的 nn.Embedding / nn.Linear 对象，
不做复制，因此天然同源，也省参数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...XLLMModel import RMSNorm, XLLMBlock, XLLMConfig, XLLMForCausalLM


@dataclass
class MTPConfig:
    """MTP 头超参。hidden_size 必须与主干一致（要与主干隐状态拼接）。"""

    hidden_size: int = 512
    num_modules: int = 1            # D：链式深度。V3/V4 官方均为 1
    num_layers_per_module: int = 1  # 每个模块含几个 Transformer 块。V3 为 1
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    intermediate_size: Optional[int] = None
    dropout: float = 0.0
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    rope_theta: int = 1e6
    max_position_embeddings: int = 32768
    use_moe: bool = False           # MTP 块是否也用 MoE（论文未明确，默认关闭以省算力）
    num_experts_per_tok: int = 2
    n_routed_experts: int = 4
    n_shared_experts: int = 1

    @classmethod
    def from_backbone(cls, bc: XLLMConfig, **overrides) -> "MTPConfig":
        base = dict(
            hidden_size=bc.hidden_size,
            num_attention_heads=bc.num_attention_heads,
            num_key_value_heads=bc.num_key_value_heads,
            intermediate_size=bc.intermediate_size,
            dropout=bc.dropout,
            hidden_act=bc.hidden_act,
            rms_norm_eps=bc.rms_norm_eps,
            rope_theta=bc.rope_theta,
            max_position_embeddings=bc.max_position_embeddings,
        )
        base.update(overrides)
        return cls(**base)


class MTPHead(nn.Module):
    """
    单个 MTP 模块。参数里**不含** embedding 与 lm_head（通过 forward 由外部传入）。
    """

    def __init__(self, cfg: MTPConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_size

        # 论文：concat(RMSNorm(h), RMSNorm(Emb(tok))) 再投影回 hidden_size
        self.h_norm = RMSNorm(h, eps=cfg.rms_norm_eps)
        self.e_norm = RMSNorm(h, eps=cfg.rms_norm_eps)
        self.proj = nn.Linear(2 * h, h, bias=False)

        block_cfg = XLLMConfig(
            dropout=cfg.dropout,
            hidden_size=h,
            intermediate_size=cfg.intermediate_size,
            num_attention_heads=cfg.num_attention_heads,
            num_hidden_layers=cfg.num_layers_per_module,
            num_key_value_heads=cfg.num_key_value_heads,
            vocab_size=1,                    # 块内不使用词表，占位
            max_position_embeddings=cfg.max_position_embeddings,
            rms_norm_eps=cfg.rms_norm_eps,
            rope_theta=cfg.rope_theta,
            hidden_act=cfg.hidden_act,
            flash_attention=True,
            use_moe=cfg.use_moe,
            num_experts_per_tok=cfg.num_experts_per_tok,
            n_routed_experts=cfg.n_routed_experts,
            n_shared_experts=cfg.n_shared_experts,
        )
        self.blocks = nn.ModuleList(
            [XLLMBlock(i, block_cfg) for i in range(cfg.num_layers_per_module)]
        )

    def forward(
        self,
        h_prev: torch.Tensor,              # [B, L, H] 上一级隐状态
        prev_ids: torch.Tensor,            # [B, L]    条件 token（已按偏移切片）
        embedding: nn.Embedding,           # 主干 embed_tokens（共享）
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        e = embedding(prev_ids)                                    # [B,L,H]
        x = torch.cat([self.h_norm(h_prev), self.e_norm(e)], dim=-1)
        x = self.proj(x)
        for blk in self.blocks:
            # XLLMBlock 的签名：(hidden, position_embeddings, past_kv, use_cache, mask)
            x, _ = blk(
                x,
                position_embeddings,
                past_key_value=None,
                use_cache=False,
                attention_mask=attention_mask,
            )
        return x


class XLLMWithMTP(nn.Module):
    """
    主干 + D 个 MTP 头的联合训练包装。

    对外保持与 XLLMForCausalLM 一致的调用方式：
        out = model(input_ids=..., attention_mask=..., labels=...)
        out.loss        总损失（主干 CE + λ/D · Σ MTP CE）
        out.main_loss   主干 CE（单独回传，便于日志与 λ=0 对照）
        out.mtp_losses  各 MTP 模块的 CE 列表

    保存约定（重要）：
        backbone 权重用 self.backbone.state_dict() 单独存，键名与 XLLMForCausalLM
        完全一致，因此 eval.py 的 strict=True 加载不受影响；
        MTP 头另存一个文件，不污染主干 checkpoint。
    """

    def __init__(self, backbone: XLLMForCausalLM, cfg: MTPConfig, mtp_weight: float = 0.3):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        self.mtp_weight = float(mtp_weight)
        self.heads = nn.ModuleList([MTPHead(cfg) for _ in range(cfg.num_modules)])

    # ------------------------------------------------------------------
    @property
    def config(self) -> XLLMConfig:
        return self.backbone.config

    def num_mtp_parameters(self) -> int:
        return sum(p.numel() for p in self.heads.parameters())

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        **kwargs,
    ):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        main_loss = out.loss
        aux_loss = getattr(out, "aux_loss", None)   # MoE 负载均衡损失，dense 时是 0
        h0 = out.hidden_states                      # [B,T,H] 主干末层隐状态
        B, T, H = h0.shape
        D = self.cfg.num_modules

        mtp_losses: List[torch.Tensor] = []
        mtp_logits: List[torch.Tensor] = []

        # 所有模块共用同一段对齐长度，避免逐模块裁剪造成位置错位
        L = T - D - 1
        if labels is not None and L > 0:
            freqs_cos = self.backbone.model.freqs_cos[:L]
            freqs_sin = self.backbone.model.freqs_sin[:L]
            pos_emb = (freqs_cos, freqs_sin)
            emb = self.backbone.model.embed_tokens
            lm_head = self.backbone.lm_head

            # labels 可能是 -100（padding），转成 mask 供后续屏蔽
            mask = None
            if attention_mask is not None:
                mask = attention_mask[:, :L]

            h_prev = h0[:, :L, :]
            for k, head in enumerate(self.heads, start=1):
                prev_ids = input_ids[:, k: k + L]              # token_{t+k}
                h_k = head(h_prev, prev_ids, emb, pos_emb, attention_mask=mask)
                logits_k = lm_head(h_k)                        # [B,L,V]
                target_k = labels[:, k + 1: k + 1 + L]         # 预测 token_{t+k+1}

                ce = F.cross_entropy(
                    logits_k.reshape(-1, logits_k.size(-1)),
                    target_k.reshape(-1),
                    ignore_index=-100,
                )
                mtp_losses.append(ce)
                mtp_logits.append(logits_k)
                h_prev = h_k                                   # 链式：下一个模块接这个输出

        if mtp_losses:
            mtp_sum = torch.stack(mtp_losses).sum() / len(mtp_losses)
            total = main_loss + self.mtp_weight * mtp_sum
        else:
            mtp_sum = torch.zeros((), device=input_ids.device)
            total = main_loss

        # aux_loss 只累加到主干，不参与 MTP 头（MTP 头默认不用 MoE）
        if aux_loss is not None:
            total = total + aux_loss

        result = {
            "loss": total,
            "main_loss": main_loss,
            "mtp_loss": mtp_sum,
            "aux_loss": aux_loss if aux_loss is not None else torch.zeros(()),
            "mtp_losses": mtp_losses,
        }
        if return_logits:
            result["mtp_logits"] = mtp_logits
        return result

    # ------------------------------------------------------------------
    @torch.no_grad()
    def draft_block(
        self,
        hidden_states: torch.Tensor,     # [B,L,H] 主干隐状态（末位是当前位置 t）
        last_logits: torch.Tensor,       # [B,V] 主头在位置 t 的 logits
        start_pos: int,
        n_draft: Optional[int] = None,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
    ):
        """
        用 主头 + MTP 链 产出一整块草稿。

        **偏移语义（关键，容易写错）**

        训练时模块 k 的输入是 (h0[t], token_{t+k})，目标是 token_{t+k+1}。注意
        它**不是**"给 h0[t] 和 token_t，预测 token_{t+1}"——那样的配对在训练中
        从未出现过。所以模块 k 只能承接"已经知道 token_{t+k}"的情况：

            主头 lm_head(h0[t])      -> token_{t+1}   （上轮验证时已算过，零额外成本）
            模块 1 (h0[t], tok_{t+1}) -> token_{t+2}
            模块 2 (h1[t], tok_{t+2}) -> token_{t+3}
            ...

        因此 D 个模块实际给出 **1 + D** 个草稿 token，且第一个是"白拿"的。

        链式连接沿用训练时的约定：模块 k 的输出 h_k 位置索引仍为 t，不推进；
        位置编码因此全程用 t 处的 RoPE 相位（与训练既 forward 里
        freqs[:L] 的用法一致）。
        """
        from ..common._verification import sample_token

        n_mod = self.cfg.num_modules
        total = 1 + n_mod
        if n_draft is not None:
            total = min(total, max(1, n_draft))
        n_mod_used = total - 1

        emb = self.backbone.model.embed_tokens
        lm_head = self.backbone.lm_head

        L = hidden_states.shape[1]
        freqs_cos = self.backbone.model.freqs_cos[start_pos: start_pos + L]
        freqs_sin = self.backbone.model.freqs_sin[start_pos: start_pos + L]
        # 只保留位置 t = start_pos + L - 1 的相位
        pe = (freqs_cos[-1:], freqs_sin[-1:])

        # ---- 第 0 个草稿：主头直接给出，无需额外前向 ----
        ids = [sample_token(last_logits, do_sample=do_sample, temperature=temperature,
                            top_k=top_k, top_p=top_p, rng=rng)]
        logits_list = [last_logits]

        # ---- 后续草稿：MTP 链 ----
        h_prev = hidden_states[:, -1:, :]
        cur = ids[0]
        for k in range(n_mod_used):
            head = self.heads[k]
            h_prev = head(h_prev, cur.unsqueeze(-1), emb, pe)
            lg = lm_head(h_prev)[:, -1, :]                # [B,V]
            logits_list.append(lg)
            cur = sample_token(lg, do_sample=do_sample, temperature=temperature,
                               top_k=top_k, top_p=top_p, rng=rng)
            ids.append(cur)

        draft_ids = torch.stack(ids, dim=1)               # [B, 1+D]
        draft_logits = torch.stack(logits_list, dim=1)    # [B, 1+D, V]
        return draft_ids, draft_logits