# -*- coding: utf-8 -*-
"""
speculative_decoding.py —— 推测解码(Speculative Decoding)推理引擎

把 草稿模型(DraftModel) 和 主模型(XLLMForCausalLM) 串成一个端到端的
加速解码循环。整体流程与 Qwen / DeepSeek / Kimi 等主流实现的推测解码
一致：

    每个迭代回合：
      1) 提议  ：草稿模型自回归提出 gamma 个候选 token（便宜、串行）；
      2) 验证  ：主模型把这 gamma 个候选拼成一次前向（并行验证，只花一次
                  主模型前向，等价于普通解码的 1 步，却能换取多个 token）；
      3) 接受  ：逐个位置用贪心/随机规则判断接受与否，取回压哨 token；
      4) 同步  ：把主/草稿的 KV cache 修剪到“已接受前缀 + 压哨 token”，
                  保证下一回合状态一致。

关键点：主模型每个回合只做 1 次前向（验证 gamma 个候选）而不是 gamma 次，
因此当草稿足够接近主模型（接受率高）时，吞吐大幅提升；而由于验证用的是
主模型自身分布，输出分布与直接解码完全一致（贪心等价 / 采样无偏）。

参考实现思路：
  - Qwen 系列：eager speculative decoding，草稿+目标并行验证；
  - DeepSeek-V3 的 MTP(Multi-Token Prediction)：单模块一次预测多个未来 token，
    等价于“草稿提议→并行验证”的变体，约 1.8x 加速；
  - Kimi(Moonshot)：同样采用轻量草稿模型 + 联合/并行解码。

本引擎默认 batch=1（推测解码的典型使用场景），主模型与草稿共用同一套
past_key_values（列表，每层一个 (k,v) 元组）结构。
"""

from __future__ import annotations

import time
from typing import Optional

import torch

from ...XLLMModel import XLLMForCausalLM
from ..common._verification import acceptance_stats, verify_and_sample
from .draft_model import DraftModel


class SpeculativeDecodingEngine:
    """
    草稿模型辅助主模型的推测解码引擎。

    参数
    ----
    target       : XLLMForCausalLM  主（目标）模型，验证阶段调用。
    draft        : DraftModel       草稿模型，提议阶段调用。
    gamma        : int>0            每回合提议的候选 token 数。越大吞吐潜力越高，
                                     但草稿更可能出错、被拒时浪费越多计算。常用 3~8。
    do_sample    : bool             是否随机采样（无偏）。False 为贪心解码。
    temperature/top_k/top_p : 采样参数，作用于草稿提议与目标验证分布。
    eos_token_id : 遇到该 token 即停止。
    seed         : 随机种子。
    """

    def __init__(
        self,
        target: XLLMForCausalLM,
        draft: DraftModel,
        gamma: int = 4,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: int = 2,
        seed: Optional[int] = None,
    ):
        self.target = target
        self.draft = draft
        assert gamma >= 1, "gamma 必须 >= 1"
        self.gamma = int(gamma)
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.eos_token_id = eos_token_id

        self.rng = torch.Generator(device=self._infer_device(target))
        if seed is not None:
            self.rng.manual_seed(seed)

        # 统计字段
        self._n_iter = 0
        self._draft_proposed = 0
        self._accepted = 0
        self._new_tokens = 0
        self._acc_per_iter: list[int] = []
        self._verify_time = 0.0

    # ------------------------------------------------------------------
    # 工具：KV cache 修剪（按绝对序列长度 keep_len 截断每层 (k, v)）
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_device(target) -> torch.device:
        """
        torch.multinomial 的 generator 必须与采样张量同设备，否则 CUDA 上报
        "Expected a 'cuda' device type for generator but found 'cpu'"。
        """
        try:
            return next(target.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _trim_kv(kv, keep_len: int):
        # 注意：主模型(XLLMModel)缓存的 KV 形状为 [B, seq, n_kv_heads, head_dim]，
        # 序列维度是 dim=1（在 transpose 之前就已缓存），所以按 dim=1 裁剪。
        return [
            (k[:, :keep_len, :, :].contiguous(), v[:, :keep_len, :, :].contiguous())
            for (k, v) in kv
        ]

    # ------------------------------------------------------------------
    # 单条序列的推测解码主循环
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate_sequence(self, input_ids: torch.Tensor, max_new_tokens: int) -> list[int]:
        """
        input_ids : [1, L] 已编码的 prompt（需左对齐、无 padding）。
        max_new_tokens : 最多生成的 token 数。
        返回生成的 token id 列表（不含 prompt）。
        """
        device = input_ids.device

        # ---- Prefill：主/草稿分别跑一遍完整 prompt，建立 KV cache ----
        out = self.target(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        target_kv = out.past_key_values
        target_last_logits = out.logits[:, -1, :].clone()   # 预测第一个生成 token 的分布

        out_d = self.draft(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        draft_kv = out_d.past_key_values
        # 修正：草稿 prefill 已把 last_token 写入 KV（长度 = L），而 propose_n 会再把
        # last_token 前向一次，导致草稿 KV 比主模型多 1、此后 RoPE 相位整体错开一位。
        # 处理办法：草稿 prefill 只吃 input_ids[:, :-1]，让两侧 KV 长度一致 = L；
        # propose_n 从 last_token 起步正好补齐最后一个位置。
        if input_ids.shape[1] > 1:
            out_d = self.draft(
                input_ids=input_ids[:, :-1], use_cache=True, logits_to_keep=1
            )
            draft_kv = out_d.past_key_values
        draft_kv_len = draft_kv[0][0].shape[1]

        last_token = input_ids[:, -1:]                       # [1, 1]
        ctx_len = input_ids.shape[1]                         # KV 当前长度（=已接受前缀长度）
        generated: list[int] = []

        while len(generated) < max_new_tokens:
            # 1) 提议：草稿串行提出 gamma 个候选，返回其分布 q
            draft_ids, draft_logits, draft_kv = self.draft.propose_n(
                last_token, draft_kv, self.gamma,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                rng=self.rng,
            )  # draft_ids:[1,gamma] draft_logits:[1,gamma,V]

            # 2) 验证：主模型一次前向并行处理 gamma 个候选
            t0 = time.perf_counter()
            out = self.target(
                input_ids=draft_ids,
                past_key_values=target_kv,
                use_cache=True,
                logits_to_keep=self.gamma,
            )
            self._verify_time += time.perf_counter() - t0
            target_logits = out.logits                      # [1, gamma, V]
            full_kv = out.past_key_values                   # 长度 = ctx_len + gamma

            # 3) 接受判断：返回接受的草稿数 num_acc 与压哨 token
            num_acc, next_token = verify_and_sample(
                target_logits, target_last_logits, draft_logits, draft_ids,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                rng=self.rng,
            )
            num_acc = int(num_acc.item())
            next_token = int(next_token.item())

            # 4) 组装本轮新 token：接受的前 num_acc 个草稿 + 压哨 token
            new_toks = draft_ids[0, :num_acc].tolist() + [next_token]
            remaining = max_new_tokens - len(generated)
            new_toks = new_toks[:remaining]
            generated.extend(new_toks)

            # 5) 同步主模型 KV：保留到“已接受前缀 + 压哨 token”
            keep = ctx_len + num_acc
            target_kv = self._trim_kv(full_kv, keep)
            nxt = torch.tensor([[next_token]], dtype=torch.long, device=device)
            out2 = self.target(
                input_ids=nxt, past_key_values=target_kv, use_cache=True, logits_to_keep=1
            )
            target_kv = out2.past_key_values
            target_last_logits = out2.logits[:, -1, :].clone()

            # 6) 同步草稿 KV：修剪到已接受前缀，再追加压哨 token
            #    修正：草稿 KV 恒定比主模型短 1（prefill 少喂了 last_token 那一位），
            #    因此必须用 keep-1 裁剪，两侧才对得上同一个绝对位置。
            draft_keep = keep - (full_kv[0][0].shape[1] - draft_kv[0][0].shape[1])
            draft_kv = self._trim_kv(draft_kv, draft_keep)
            out_d = self.draft(
                input_ids=nxt, past_key_values=draft_kv, use_cache=True, logits_to_keep=1
            )
            draft_kv = out_d.past_key_values

            last_token = nxt
            ctx_len = keep + 1
            assert ctx_len == target_kv[0][0].shape[1], (
                f"主模型 KV 长度失配: ctx_len={ctx_len}, "
                f"kv={target_kv[0][0].shape[1]}"
            )

            # 统计
            self._n_iter += 1
            self._draft_proposed += self.gamma
            self._accepted += num_acc
            self._new_tokens += len(new_toks)
            self._acc_per_iter.append(num_acc)

            if next_token == self.eos_token_id:
                break

        return generated

    # ------------------------------------------------------------------
    # 对外接口：生成
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens: int = 64,
    ) -> dict:
        """
        input_ids : [L] 或 [B, L] 的 token id（长整型）。
        max_new_tokens : 每个序列最多生成的新 token 数。

        返回 dict:
            sequences : 每个输入序列生成的 token id 列表（不含 prompt）。
            stats     : 汇总统计（回合数、接受率、近似加速比等）。
        """
        single = input_ids.dim() == 1
        if single:
            input_ids = input_ids.unsqueeze(0)

        # 每个 batch 元素独立跑（batch>1 时串行，保持正确性）
        all_seqs = []
        for b in range(input_ids.shape[0]):
            self._acc_per_iter = []
            seq = self._generate_sequence(input_ids[b : b + 1], max_new_tokens)
            all_seqs.append(seq)

        stats = acceptance_stats(self._acc_per_iter, self.gamma)
        stats.update({
            "new_tokens": sum(len(s) for s in all_seqs),
            "verify_time_s": round(self._verify_time, 4),
        })

        if single:
            return {"sequences": all_seqs[0], "stats": stats}
        return {"sequences": all_seqs, "stats": stats}

    # ------------------------------------------------------------------
    # 状态与统计
    # ------------------------------------------------------------------
    def reset_stats(self):
        self._n_iter = 0
        self._draft_proposed = 0
        self._accepted = 0
        self._new_tokens = 0
        self._acc_per_iter = []
        self._verify_time = 0.0
