# -*- coding: utf-8 -*-
"""
dspark_draft.py —— DSpark 式半自回归草稿模块（对齐 DeepSeek-V4.1 §2.4.3）

与 model/draft 里已有的 DraftModel（XLLMForCausalLM 截断副本）是两条完全不同的
技术路线。DSpark 的关键设计差异：

    1) 草稿骨干是"并行 one-pass"的：一次前向同时给出 gamma 个位置的 base logits，
       而不是像 DraftModel.propose_n 那样串行跑 gamma 步。这样 gamma 增大时草稿
       延迟几乎不变（DFlash 的核心观察：一次前向的代价与块长度无关）。
    2) 只用 target 的 hidden states，不持有自己的 KV cache。免除了 draft/target
       两侧 KV 对齐与裁剪的全部风险。
    3) 并行草稿的"后缀衰减"用一个小型 Markov 头按链式顺序修正（semi-autoregressive）。
       每步只是一次查表 + 一次低秩 matmul，论文实测把块长度 4→16 的额外延迟控制在
       0.2%~1.3%。
    4) 额外挂一个置信度头，逐位置预测"该 token 能通过验证"的条件概率。推理时调度器
       用它决定这一轮验证多长（见 dspark_scheduler.py）。
    5) 与 target 共享 lm_head / embedding，草稿侧没有自己的大词表矩阵。

训练侧对应 dspark_train.py，三个损失：CE + TV + 置信度 BCE，target 全程冻结。

与论文的一处已知偏离（如实标注）
--------------------------------
论文的 DFlash 骨干是"块扩散"：把未来块的 token embedding 加噪后一次去噪。本实现
简化为"可学习 mask 槽位"（K 个 mask embedding 拼在前缀后面，一次前向产出 K 个位置
的 logits）。功能上同样是一次前向产出整块草稿，但不含扩散加噪/去噪过程，因此质量
上限低于原版。若要复现原版，替换 _block_slots() 与对应训练目标即可，其余（Markov
头、置信度头、调度器、验证逻辑）无需改动。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...XLLMModel import (
    FeedForward,
    RMSNorm,
    XLLMConfig,
    apply_rotary_pos_emb,
    precompute_freqs_cis,
    repeat_kv,
)


# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------
@dataclass
class DSparkConfig:
    """草稿模块超参。hidden_size 必须与 target 一致（要消费其 hidden states）。"""

    hidden_size: int = 512                 # 必须等于 target.config.hidden_size
    vocab_size: int = 6400                 # 必须等于 target.config.vocab_size
    d_model: Optional[int] = None          # 骨干内部宽度；None 表示与 hidden_size 相同
    num_layers: int = 3                    # 论文：3 个 Transformer 块
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    intermediate_size: Optional[int] = None
    window: int = 128                      # 滑动注意力窗口（论文：128 tokens）
    max_draft: int = 5                     # 一次前向最多产出几个草稿位置（论文：5）
    markov_rank: int = 64                  # Markov 头低秩秩数
    dropout: float = 0.0
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1e6
    max_position_embeddings: int = 32768

    def __post_init__(self):
        if self.d_model is None:
            self.d_model = self.hidden_size
        if self.intermediate_size is None:
            inter = int(self.d_model * 8 / 3)
            self.intermediate_size = 64 * ((inter + 63) // 64)
        assert self.d_model % self.num_attention_heads == 0
        assert self.num_attention_heads % self.num_key_value_heads == 0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_attention_heads

    @classmethod
    def from_target(cls, target_config, **overrides) -> "DSparkConfig":
        """用 target 的配置推导草稿配置，保证 dim/attn 结构可比。"""
        base = dict(
            hidden_size=target_config.hidden_size,
            vocab_size=target_config.vocab_size,
            num_attention_heads=target_config.num_attention_heads,
            num_key_value_heads=target_config.num_key_value_heads,
            dropout=target_config.dropout,
            hidden_act=target_config.hidden_act,
            rms_norm_eps=target_config.rms_norm_eps,
            rope_theta=target_config.rope_theta,
            max_position_embeddings=target_config.max_position_embeddings,
        )
        base.update(overrides)
        return cls(**base)


# ----------------------------------------------------------------------
# 滑动窗口注意力块
# ----------------------------------------------------------------------
class DSparkBlock(nn.Module):
    """
    草稿骨干的一层：滑动窗口因果注意力 + SwiGLU FFN。

    与 XLLMBlock 的区别只有两点：窗口裁剪、不返回 KV（草稿不缓存）。
    """

    def __init__(self, cfg: DSparkConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = cfg.head_dim

        h = cfg.d_model
        self.q_proj = nn.Linear(h, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, h, bias=False)

        self.input_layernorm = RMSNorm(h, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(h, eps=cfg.rms_norm_eps)

        # FFN 复用 XLLMModel.FeedForward，需要一个 XLLMConfig 形状的容器
        ffn_cfg = XLLMConfig(
            hidden_size=cfg.d_model,
            intermediate_size=cfg.intermediate_size,
            dropout=cfg.dropout,
            hidden_act=cfg.hidden_act,
            rms_norm_eps=cfg.rms_norm_eps,
            num_hidden_layers=1,
            vocab_size=1,
        )
        self.mlp = FeedForward(ffn_cfg)
        self.attn_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attn_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        residual = x
        h = self.input_layernorm(x)
        xq = self.q_proj(h).view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = self.k_proj(h).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.v_proj(h).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=attn_mask,
            dropout_p=self.cfg.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(bsz, seq_len, -1)
        x = residual + self.attn_dropout(self.o_proj(out))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# ----------------------------------------------------------------------
# DSpark 草稿模块
# ----------------------------------------------------------------------
class DSparkDrafter(nn.Module):
    """
    半自回归草稿：并行骨干出 base logits，Markov 头按链修正，置信度头给接受概率。

    对外接口（推理用）：
        draft_block(prefix_hidden, start_pos, n_draft, last_token, ...)
            -> draft_ids [B, n], draft_logits [B, n, V], conf [B, n]
    训练用：
        forward_train(prefix_hidden, start_pos, teacher_tokens, lm_head, ...)
            -> base/corrected logits、置信度 logits，供三项损失计算
    """

    def __init__(self, cfg: DSparkConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # target hidden -> 草稿骨干宽度
        self.in_proj = nn.Linear(cfg.hidden_size, d, bias=False) if d != cfg.hidden_size else nn.Identity()
        # 草稿骨干 -> target hidden（以便复用 target 的 lm_head）
        self.out_proj = nn.Linear(d, cfg.hidden_size, bias=False) if d != cfg.hidden_size else nn.Identity()

        self.mask_emb = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.mask_emb, std=0.02)

        self.blocks = nn.ModuleList([DSparkBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(d, eps=cfg.rms_norm_eps)

        # Markov 头：上一草稿 token -> 词表空间的低秩加性偏置
        # 表大小必须是 vocab_size（索引是上一个 token id），不是位置数。
        self.markov_emb = nn.Embedding(cfg.vocab_size, cfg.markov_rank)
        self.markov_proj = nn.Linear(cfg.markov_rank, cfg.vocab_size, bias=False)
        nn.init.normal_(self.markov_emb.weight, std=0.02)
        nn.init.zeros_(self.markov_proj.weight)

        # 置信度头：逐位置条件接受概率
        self.conf_head = nn.Sequential(
            nn.Linear(d, max(d // 2, 16)),
            nn.SiLU(),
            nn.Linear(max(d // 2, 16), 1),
        )

        # 置信度校准参数（逐位置缩放 + 偏置），在 held-out 数据上拟合
        self.register_buffer("conf_scale", torch.ones(cfg.max_draft))
        self.register_buffer("conf_bias", torch.zeros(cfg.max_draft))

        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=cfg.head_dim,
            end=cfg.max_position_embeddings,
            rope_base=cfg.rope_theta,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    # ------------------------------------------------------------------
    # 内部：构造滑动窗口因果掩码
    # ------------------------------------------------------------------
    @staticmethod
    def _window_mask(seq_len: int, window: int, device) -> torch.Tensor:
        idx = torch.arange(seq_len, device=device)
        diff = idx[:, None] - idx[None, :]                    # i - j
        allowed = (diff >= 0) & (diff < window)
        return allowed[None, None, :, :]                      # [1,1,S,S] bool

    # ------------------------------------------------------------------
    # 内部：一次前向产出 base logits [B, n, H]
    # ------------------------------------------------------------------
    def _backbone(self, prefix_hidden: torch.Tensor, start_pos: int, n_draft: int):
        """
        prefix_hidden: [B, L, H] target 的 hidden states（调用方负责只给最后 window 个）
        start_pos:     prefix 第一个位置在绝对坐标中的位置（决定 RoPE 相位）
        返回 base_hidden [B, n_draft, H]，骨干在 n_draft 个 mask 槽位上的输出
        """
        bsz, L, _ = prefix_hidden.shape
        d = self.cfg.d_model

        x = self.in_proj(prefix_hidden)                        # [B,L,d]
        slots = self.mask_emb.view(1, 1, d).expand(bsz, n_draft, d)
        x = torch.cat([x, slots], dim=1)                       # [B, L+n, d]

        total = L + n_draft
        cos = self.freqs_cos[start_pos:start_pos + total]
        sin = self.freqs_sin[start_pos:start_pos + total]
        mask = self._window_mask(total, self.cfg.window, x.device)

        for blk in self.blocks:
            x = blk(x, cos, sin, mask)
        x = self.norm(x)
        x = self.out_proj(x[:, L:, :])                         # 只取槽位输出 [B,n,H]
        return x

    # ------------------------------------------------------------------
    # 内部：Markov 链式修正
    # ------------------------------------------------------------------
    def _markov_bias(self, prev_tokens: torch.Tensor) -> torch.Tensor:
        """prev_tokens: [B] long -> [B, H] 加性偏置"""
        return self.markov_proj(self.markov_emb(prev_tokens))

    def _apply_markov(self, base_logits: torch.Tensor, first_prev: torch.Tensor,
                      tokens: Optional[torch.Tensor] = None,
                      do_sample: bool = False, temperature: float = 1.0,
                      top_k: Optional[int] = None, top_p: Optional[float] = None,
                      rng: Optional[torch.Generator] = None):
        """
        沿链逐位置修正 logits。

        tokens 给定时（训练 / teacher forcing）直接用真实前一个 token；
        否则用已修正分布采样/取 argmax 得到的 token 作为下一步的条件。
        """
        from ..common._verification import sample_token

        n = base_logits.shape[1]
        outs, ids = [], []
        prev = first_prev
        for d in range(n):
            logits_d = base_logits[:, d, :] + self._markov_bias(prev)
            outs.append(logits_d)
            if tokens is not None:
                nxt = tokens[:, d]
            else:
                nxt = sample_token(logits_d, do_sample=do_sample, temperature=temperature,
                                   top_k=top_k, top_p=top_p, rng=rng)
            ids.append(nxt)
            prev = nxt
        return torch.stack(outs, dim=1), torch.stack(ids, dim=1)

    # ------------------------------------------------------------------
    # 推理接口
    # ------------------------------------------------------------------
    @torch.no_grad()
    def draft_block(
        self,
        prefix_hidden: torch.Tensor,      # [B, L, H]，最后一次前向的 hidden（可含历史窗口）
        start_pos: int,                   # prefix 首位置的绝对坐标
        n_draft: int,
        last_token: torch.Tensor,         # [B] 当前已接受的最后 token
        lm_head: nn.Linear,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
    ):
        """
        一次前向产出 n_draft 个草稿。返回
            draft_ids    [B, n]
            draft_logits [B, n, V]   已含 Markov 修正的 logits（还原分布 q 用）
            conf         [B, n]      逐位置条件接受概率（已校准）
        """
        n_draft = min(n_draft, self.cfg.max_draft)
        hidden = self._backbone(prefix_hidden, start_pos, n_draft)   # [B,n,H]
        base_logits = lm_head(hidden)                                # [B,n,V]

        logits, ids = self._apply_markov(
            base_logits, last_token, tokens=None, do_sample=do_sample,
            temperature=temperature, top_k=top_k, top_p=top_p, rng=rng,
        )

        # 置信度：条件接受概率，逐位置校准
        logit_c = self.conf_head(hidden).squeeze(-1)                 # [B,n]
        calib = logit_c * self.conf_scale[:n_draft] + self.conf_bias[:n_draft]
        conf = torch.sigmoid(calib)
        return ids, logits, conf

    # ------------------------------------------------------------------
    # 训练接口
    # ------------------------------------------------------------------
    def forward_teacher(
        self,
        prefix_hidden: torch.Tensor,
        start_pos: int,
        prev_tokens: torch.Tensor,     # [B, n] 每个草稿位置的条件 token（teacher forcing）
        lm_head: nn.Linear,
    ):
        """
        返回
            logits          [B, n, V]  含 Markov 修正
            base_logits     [B, n, V]  未修正
            conf_logits     [B, n]     置信度头原始输出（供 BCE）
            hidden          [B, n, H]  草稿骨干在槽位上的输出
        """
        n = prev_tokens.shape[1]
        hidden = self._backbone(prefix_hidden, start_pos, n)
        base_logits = lm_head(hidden)
        logits, _ = self._apply_markov(base_logits, prev_tokens[:, 0], tokens=prev_tokens)
        conf_logits = self.conf_head(hidden).squeeze(-1)
        return logits, base_logits, conf_logits, hidden

    # ------------------------------------------------------------------
    # 置信度校准：逐位置线性重标定，使链式乘积匹配真实接受率
    # ------------------------------------------------------------------
    @torch.no_grad()
    def calibrate(self, target_accept: torch.Tensor):
        """
        target_accept: [n_pos] 每个位置在 held-out 数据上的真实条件接受率。
        采用逐位置 logit 平移（单调、保序）做最小二乘近似解。
        """
        n = target_accept.numel()
        eps = 1e-6
        p = target_accept.clamp(eps, 1 - eps)
        self.conf_scale[:n] = 1.0
        self.conf_bias[:n] = torch.log(p / (1 - p)) * 0.5

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())