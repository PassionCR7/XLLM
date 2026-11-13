# -*- coding: utf-8 -*-
"""
mtp_engine.py —— MTP 草稿头的推测解码引擎

用途：把 mtp_pretrain.py 训出来的 MTP 头接进生成流程，验证它到底有没有用。
没有这个引擎，训练只能在 loss 曲线上看数字，无法回答"接受率多少、加速比多少"。

与其他两个引擎的结构差异
------------------------

    SpeculativeDecodingEngine（A）
        草稿是截断主干，**自带一套 KV**，需要与 target 做逐轮同步与裁剪，
        极易出错（本项目历史上就因为少喂/多喂一个 token 导致接受率腰斩）。

    DSparkEngine（B）
        草稿是并行骨干，**不持 KV**，只吃 target 的 hidden。

    MTPEngine（本文件，C）
        草稿完全复用主干：主头给第 1 个 token，MTP 链给后续 D 个。
        同样不持 KV——MTP 头训练时就没有 cache 概念（每次都是全序列一次前向）。

为什么 MTP 引擎天然不容易出现 KV 错位
-------------------------------------
因为草稿侧根本没有 KV。每轮只做两件事：
    1) 一次主干前向（验证 k 个候选，同时产出新的 hidden 与 KV）
    2) D 次 MTP 头前向（每次都是"单步 + 无 cache"，输入是上一级输出）

但有一个必须小心的地方：**裁剪后的 KV 与 hidden 必须严格等长**。这里沿用
DSparkEngine 的做法，每轮都会断言 ctx_len == kv_len，任何错位立刻暴露，
而不是悄悄降低接受率。

无损性
------
接受/拒绝完全走 _verification.verify_and_sample，与 A/B 共用同一套规则，
因此：贪心模式下与主干直接贪心解码逐 token 相同；随机模式下与主干采样无偏。
这一点由 _mtp_engine_check.py 实测验证，而不是仅凭推理断言。
"""

from __future__ import annotations

import time
from typing import List, Optional, Union

import torch

from ..common._verification import verify_and_sample


class MTPEngine:
    """
    MTP 推测解码引擎。

    参数
    ----
    model : XLLMWithMTP
        联合训练得到的模型（主干 + D 个 MTP 头）。
    max_draft : int | None
        每轮最多验证多少个 token。None 表示用 1 + num_modules
        （MTP 的天然宽度：主头 1 个 + D 个模块各 1 个）。
        注意不能超过 1 + num_modules，超过没有意义（MTP 链没这么深）。
    """

    def __init__(
        self,
        model,
        max_draft: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model = model
        self.backbone = model.backbone
        self.heads = model.heads
        self.max_draft = int(max_draft or (1 + model.cfg.num_modules))
        assert self.max_draft <= 1 + model.cfg.num_modules, (
            f"max_draft({self.max_draft}) 超过 MTP 链能提供的宽度 "
            f"(1 + {model.cfg.num_modules})"
        )
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.eos_token_id = eos_token_id

        self.device = torch.device(device) if device is not None else next(
            model.parameters()).device
        # generator 必须与采样张量同设备，否则 CUDA 上报
        # "Expected a 'cuda' device type for generator but found 'cpu'"
        self.rng = torch.Generator(device=self.device)
        if seed is not None:
            self.rng.manual_seed(seed)

        self._reset_counters()

    # ------------------------------------------------------------------
    def _reset_counters(self):
        self._n_iter = 0                 # 推理轮数
        self._n_accepted = 0             # 接受的草稿 token 总数
        self._n_tokens = 0               # 实际产出 token 总数
        self._acc_per_iter: List[int] = []
        self._draft_time = 0.0
        self._verify_time = 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def _trim_kv(kv, keep_len: int):
        return [(k[:, :keep_len].contiguous(), v[:, :keep_len].contiguous())
                for (k, v) in kv]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate_sequence(self, input_ids: torch.Tensor,
                           max_new_tokens: int) -> List[int]:
        device = input_ids.device
        L = input_ids.shape[1]

        # ---- Prefill：主干跑完整 prompt ----
        out = self.backbone(input_ids=input_ids, use_cache=True, logits_to_keep=L)
        target_kv = out.past_key_values
        prev_logits = out.logits[:, -1, :].clone()      # 位置 L-1 预测第 L 个 token
        last_hidden = out.hidden_states[:, -1:, :].clone()
        ctx_len = L

        generated: List[int] = []
        while len(generated) < max_new_tokens:
            n_want = min(self.max_draft, max_new_tokens - len(generated))

            # ---- 1) 草稿：主头 1 个 + MTP 链 D 个 ----
            t0 = time.perf_counter()
            draft_ids, draft_logits = self.model.draft_block(
                hidden_states=last_hidden,
                last_logits=prev_logits,
                start_pos=ctx_len - 1,
                n_draft=n_want,
                temperature=self.temperature,
                do_sample=self.do_sample,
                top_k=self.top_k,
                top_p=self.top_p,
                rng=self.rng,
            )
            self._draft_time += time.perf_counter() - t0
            k = draft_ids.shape[1]

            # ---- 2) 主干一次并行验证 k 个候选 ----
            # 这里的 k 个前向同时完成了两件事：验证补齐 + 产生后续要用的 hidden。
            t0 = time.perf_counter()
            out = self.backbone(
                input_ids=draft_ids,
                past_key_values=target_kv,
                use_cache=True,
                logits_to_keep=k,
            )
            self._verify_time += time.perf_counter() - t0
            full_kv = out.past_key_values
            verify_logits = out.logits                        # [1,k,V]

            # ---- 3) 接受判断（与主干解码等价 / 无偏） ----
            num_acc, next_token = verify_and_sample(
                verify_logits, prev_logits, draft_logits, draft_ids,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                rng=self.rng,
            )
            m = int(num_acc.item())
            next_token = int(next_token.item())

            generated.extend(draft_ids[0, :m].tolist())
            generated.append(next_token)
            self._n_iter += 1
            self._n_accepted += m
            self._acc_per_iter.append(m)
            self._n_tokens += m + 1

            # ---- 4) 状态推进 ----
            # 这里刻意采用"保留已接受前缀 + 单独补一次 next_token 前向"的做法，
            # 而不是试图省掉那次前向。原因：
            #   next_token 是在位置 ctx_len+m 上**被拒绝**后采出的压哨 token，
            #   它与 draft_ids[m] 不是同一个 token，因此 out.hidden_states 里
            #   第 m 个位置的 hidden 是"错误 token 的隐状态"，不能拿来做草稿输入。
            #   图上看起来省了一次前向，实际会把位置语义搞错，贪心等价性立刻崩。
            # 对 512 维 8 层的小模型，这次前向约占总时间的一小部分，不值得冒险省。
            keep = ctx_len + m
            target_kv = self._trim_kv(full_kv, keep)

            nt = torch.tensor([[next_token]], device=device, dtype=torch.long)
            t0 = time.perf_counter()
            out2 = self.backbone(input_ids=nt, past_key_values=target_kv,
                                 use_cache=True, logits_to_keep=1)
            self._verify_time += time.perf_counter() - t0
            target_kv = out2.past_key_values
            prev_logits = out2.logits[:, -1, :].clone()      # 预测 next_token 之后
            last_hidden = out2.hidden_states[:, -1:, :]
            ctx_len = keep + 1

            # 状态一致性断言：任何错位都会让接受率悄悄下降而不报错，必须显式检查
            assert ctx_len == target_kv[0][0].shape[1], (
                f"KV 与上下文长度不一致: ctx_len={ctx_len} "
                f"kv_len={target_kv[0][0].shape[1]}")

            if self.eos_token_id is not None and next_token == self.eos_token_id:
                break

        return generated[:max_new_tokens]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
    ):
        """
        生成。输入 [1, L] 或 [L]。

        返回
        ----
        tokens : List[int] 或 List[List[int]]
            与输入形状对应：1D 输入返回 1D，2D 输入返回 2D。
        """
        squeeze = input_ids.dim() == 1
        if squeeze:
            input_ids = input_ids.unsqueeze(0)
        assert input_ids.shape[0] == 1, "MTPEngine 目前只支持 batch=1"

        self._reset_counters()
        gen = self._generate_sequence(input_ids.to(self.device), max_new_tokens)
        return gen if squeeze else [gen]

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """
        本轮统计。

        avg_tokens_per_iter 的理论上限是 2+D：草稿宽度 k=1+D（主头 1 个 + D 个
        模块），若 k 个全部接受再加压哨 token，共 k+1=2+D 个。
        """
        n_iter = max(self._n_iter, 1)
        avg_acc = self._n_accepted / n_iter
        avg_verify_len = self._n_tokens / n_iter
        total_time = self._draft_time + self._verify_time
        return {
            "iters": self._n_iter,
            "tokens": self._n_tokens,
            "accepted": self._n_accepted,
            "accept_rate": self._n_accepted / max(self._n_tokens, 1),
            "avg_accepted_per_iter": avg_acc,
            "avg_tokens_per_iter": avg_verify_len,
            "acc_per_iter": list(self._acc_per_iter),
            "draft_time_s": self._draft_time,
            "verify_time_s": self._verify_time,
            "total_time_s": total_time,
            "ms_per_token": 1000.0 * total_time / max(self._n_tokens, 1),
            "draft_overhead_ratio": self._draft_time / max(total_time, 1e-9),
        }