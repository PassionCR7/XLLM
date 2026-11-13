# -*- coding: utf-8 -*-
"""
dspark_engine.py —— DSpark 式推测解码引擎（target-only KV）

与已有 SpeculativeDecodingEngine 的结构性区别：

    已有实现      草稿和 target 各自持有一套 KV，每轮要同步、要按不同长度裁剪。
                  -> 上轮排查出的 [4,4,4,0,0,0] 接受率塌陷就源于两侧长度差 1。

    本实现        只有 target 有 KV。草稿每轮拿 target 最近 window 个位置的 hidden
                  states 重新算一遍（一次前向），不缓存、不带自己的 KV。
                  -> 结构上不可能出现 KV 错位；草稿的"状态"就是 (hidden, start_pos)。

代价与权衡：
    * 草稿每轮重算 window 长度的注意力，比"带 KV 的增量"多花一点算力；
    * 但换来的是绝对的状态正确性，以及一次前向产出整块草稿（gamma 增大不加延迟）。
    这正是 DSpark 相对串行草稿的核心取舍。

无损性：验证阶段仍然走 _verification.verify_and_sample 的接受/拒采规则，
贪心模式与 target 贪心解码逐 token 一致，随机模式与 target 采样无偏。
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Union

import torch

from ...XLLMModel import XLLMForCausalLM
from ..common._verification import acceptance_stats, verify_and_sample
from .dspark_draft import DSparkDrafter
from .dspark_scheduler import BaseScheduler, build_scheduler


class DSparkEngine:
    """
    参数
    ----
    target      : XLLMForCausalLM，冻结，只做验证与提供 hidden states
    drafter     : DSparkDrafter
    scheduler   : 置信度调度器；None 时用 ThroughputAwareScheduler
    max_draft   : 一次草稿的最大长度（与 drafter.cfg.max_draft 一致）
    window      : 草稿滑动窗口长度（取 target hidden 的最近 window 个位置）
    do_sample   : True 走无偏随机接受；False 走贪心等价接受
    """

    def __init__(
        self,
        target: XLLMForCausalLM,
        drafter: DSparkDrafter,
        scheduler: Optional[BaseScheduler] = None,
        max_draft: Optional[int] = None,
        window: int = 128,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: int = 2,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.target = target
        self.drafter = drafter
        self.max_draft = int(max_draft or drafter.cfg.max_draft)
        assert self.max_draft <= drafter.cfg.max_draft, (
            f"max_draft({self.max_draft}) 不能超过 drafter 训练时的宽度({drafter.cfg.max_draft})"
        )
        self.window = int(window)
        self.scheduler = scheduler or build_scheduler("throughput", max_draft=self.max_draft)
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.eos_token_id = eos_token_id

        self.device = torch.device(device) if device is not None else next(target.parameters()).device
        # 修正：generator 必须与采样张量同设备，否则 CUDA 上报
        # "Expected a 'cuda' device type for generator but found 'cpu'"
        self.rng = torch.Generator(device=self.device)
        if seed is not None:
            self.rng.manual_seed(seed)

        self._reset_counters()

    # ------------------------------------------------------------------
    def _reset_counters(self):
        self._n_iter = 0
        self._accepted = 0
        self._new_tokens = 0
        self._acc_per_iter: List[int] = []
        self._verify_time = 0.0
        self._draft_time = 0.0
        self._k_history: List[int] = []

    # ------------------------------------------------------------------
    # KV 裁剪（只有 target 一侧，长度即绝对序列长度）
    # ------------------------------------------------------------------
    @staticmethod
    def _trim_kv(kv, keep_len: int):
        return [
            (k[:, :keep_len].contiguous(), v[:, :keep_len].contiguous())
            for (k, v) in kv
        ]

    # ------------------------------------------------------------------
    # 单序列主循环
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate_sequence(self, input_ids: torch.Tensor, max_new_tokens: int) -> List[int]:
        device = input_ids.device
        L = input_ids.shape[1]

        # ---- Prefill：target 跑完整 prompt，拿到 hidden 与 KV ----
        out = self.target(input_ids=input_ids, use_cache=True, logits_to_keep=L)
        target_kv = out.past_key_values
        target_hidden = out.hidden_states            # [1, L, H]，已过 final norm
        target_last_logits = out.logits[:, -1, :].clone()

        ctx_len = L                                   # target KV 当前长度
        last_token = input_ids[:, -1]                 # [1]
        generated: List[int] = []

        while len(generated) < max_new_tokens:
            # ---- 1) 草稿：取最近 window 个 hidden，一次前向出整块 ----
            win_start = max(0, ctx_len - self.window)
            h_win = target_hidden[:, win_start:, :]
            n_want = min(self.max_draft, max_new_tokens - len(generated))

            t0 = time.perf_counter()
            draft_ids, draft_logits, conf = self.drafter.draft_block(
                prefix_hidden=h_win,
                start_pos=win_start,
                n_draft=n_want,
                last_token=last_token,
                lm_head=self.target.lm_head,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                rng=self.rng,
            )
            self._draft_time += time.perf_counter() - t0

            # ---- 2) 调度：按置信度决定这轮验证多长 ----
            decision = self.scheduler.decide(conf[0], current_batch=1)
            k = max(1, min(decision.k, n_want))
            self._k_history.append(k)

            if k < draft_ids.shape[1]:
                draft_ids = draft_ids[:, :k]
                draft_logits = draft_logits[:, :k]

            # ---- 3) target 一次前向并行验证 k 个候选 ----
            t0 = time.perf_counter()
            out = self.target(
                input_ids=draft_ids,
                past_key_values=target_kv,
                use_cache=True,
                logits_to_keep=k,
            )
            self._verify_time += time.perf_counter() - t0
            full_kv = out.past_key_values
            # hidden_states 的长度随调用方式变化：prefill（无 past）返回完整序列，
            # 带 past 的增量调用只返回本次新增的 k 个位置。这里统一按"取最后 k 个"处理。
            verify_hidden = out.hidden_states[:, -k:, :]           # [1, k, H]

            # ---- 4) 接受判断（与主模型解码等价/无偏） ----
            num_acc, next_token = verify_and_sample(
                out.logits, target_last_logits, draft_logits, draft_ids,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                rng=self.rng,
            )
            num_acc = int(num_acc.item())
            next_token = int(next_token.item())

            new_toks = draft_ids[0, :num_acc].tolist() + [next_token]
            remaining = max_new_tokens - len(generated)
            new_toks = new_toks[:remaining]
            generated.extend(new_toks)

            # ---- 5) 同步 target KV 与 hidden：保留已接受前缀 + 地板 token ----
            keep = ctx_len + num_acc
            target_kv = self._trim_kv(full_kv, keep)
            # hidden 同步：prefix_hidden 必须与 target_kv 长度一致
            kept_hidden = torch.cat([target_hidden, verify_hidden], dim=1)[:, :keep, :]

            nxt = torch.tensor([[next_token]], dtype=torch.long, device=device)
            out2 = self.target(
                input_ids=nxt, past_key_values=target_kv, use_cache=True, logits_to_keep=1
            )
            target_kv = out2.past_key_values
            target_last_logits = out2.logits[:, -1, :].clone()
            # out2.hidden_states 同样是完整长度（keep+1），只取最后一个位置
            target_hidden = torch.cat([kept_hidden, out2.hidden_states[:, -1:, :]], dim=1)

            last_token = nxt[:, 0]
            ctx_len = target_hidden.shape[1]
            assert ctx_len == target_kv[0][0].shape[1], (
                f"hidden 与 KV 长度不一致: hidden={ctx_len}, kv={target_kv[0][0].shape[1]}"
            )

            # ---- 统计 ----
            self._n_iter += 1
            self._accepted += num_acc
            self._new_tokens += len(new_toks)
            self._acc_per_iter.append(num_acc)

            if next_token == self.eos_token_id:
                break

        return generated

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens: int = 64, eos_token_id: Optional[int] = None) -> Dict:
        """
        input_ids: [L] 或 [B, L]。batch>1 时逐条串行（与已有引擎一致）。
        返回 {"sequences": ..., "stats": {...}}
        """
        if eos_token_id is not None:
            self.eos_token_id = eos_token_id

        single = input_ids.dim() == 1
        if single:
            input_ids = input_ids.unsqueeze(0)

        self._reset_counters()
        all_seqs: List[List[int]] = []
        for b in range(input_ids.shape[0]):
            self._acc_per_iter = []
            all_seqs.append(self._generate_sequence(input_ids[b: b + 1], max_new_tokens))

        stats = acceptance_stats(self._acc_per_iter, self.max_draft)
        stats.update({
            "new_tokens": sum(len(s) for s in all_seqs),
            "avg_verify_len": round(sum(self._k_history) / max(len(self._k_history), 1), 3),
            "verify_time_s": round(self._verify_time, 4),
            "draft_time_s": round(self._draft_time, 4),
            "draft_overhead_ratio": round(
                self._draft_time / max(self._verify_time, 1e-9), 3
            ),
        })

        if single:
            return {"sequences": all_seqs[0], "stats": stats}
        return {"sequences": all_seqs, "stats": stats}