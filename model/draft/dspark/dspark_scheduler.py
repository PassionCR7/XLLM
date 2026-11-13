# -*- coding: utf-8 -*-
"""
dspark_scheduler.py —— 置信度调度验证（对齐 DeepSeek-V4.1 DSpark 论文）

论文的核心贡献之一：把"这轮验证几个 token"从固定超参变成按负载动态决策。做法是

    Θ = τ · SPS(B)

其中 τ 是本轮期望接受的 token 数、SPS(B) 是当前 batch 规模下硬件每秒能跑的步数。
调度器对所有候选 token 按存活概率从高到低贪心纳入验证批次，逐步更新 Θ，直到边际
收益不再覆盖边际成本（batch 变大导致单步变慢）。

本实现把 SPS 曲线做成可注入的 profile（启动时测一次），并提供三者：
    ThroughputAwareScheduler —— 论文主方案：按 Θ 最大化选长度
    ConfidenceScheduler      —— 简化方案：按累计存活概率阈值截断
    FixedLengthScheduler     —— 基线：固定 gamma，用于对照实验
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch


@dataclass
class ThroughputProfile:
    """
    SPS(batch_size)：给定验证批次规模，硬件每秒能完成多少次验证前向。

    实测获取：固定长度跑一轮，记录不同 total_batch 下的 steps/sec。
    缺省值按"batch 越大越慢、且边际递减"的通用形状给一个占位曲线，
    真实部署必须用实测数据替换（论文强调这是 profile 出来的，不是假设的）。
    """

    batch_points: Sequence[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128])
    sps_points: Sequence[float] = field(
        default_factory=lambda: [10.0, 9.3, 8.2, 6.8, 5.2, 3.6, 2.2, 1.2]
    )

    def sps(self, batch_size: float) -> float:
        """
        线性插值；超出范围时按端点值外推。尾部用 1/sqrt 衰减，
        保证"batch 越大单步越慢"这一单调性在任何负载下都成立（否则调度器
        会误判为"验证越长越划算"，与论文描述的行为相反）。
        """
        bs = float(max(batch_size, 1.0))
        pts, sps = self.batch_points, self.sps_points
        if bs <= pts[0]:
            return float(sps[0])
        if bs >= pts[-1]:
            tail = float(sps[-1]) * (float(pts[-1]) / bs) ** 0.5
            return max(tail, float(sps[-1]) * 0.05)
        for i in range(1, len(pts)):
            if bs <= pts[i]:
                x0, x1 = float(pts[i - 1]), float(pts[i])
                y0, y1 = float(sps[i - 1]), float(sps[i])
                t = (bs - x0) / max(x1 - x0, 1e-9)
                return y0 + t * (y1 - y0)
        return float(sps[-1])


@dataclass
class ScheduleDecision:
    k: int                    # 本轮验证长度
    expected_accepted: float  # 期望接受数（含 +1 的地板 token）
    expected_throughput: float


class BaseScheduler:
    def __init__(self, max_draft: int = 5):
        self.max_draft = int(max_draft)

    def decide(self, conf: torch.Tensor, current_batch: int = 1) -> ScheduleDecision:
        raise NotImplementedError

    @staticmethod
    def survival(conf: torch.Tensor) -> List[float]:
        """
        置信度 -> 前缀存活概率。

        conf 是"条件接受概率"（给定前面全部被接受）。第 j 个草稿被接受的前提是
        前 j 个全部存活，因此存活概率是链式累乘。
        """
        c = conf.detach().float().flatten().tolist()
        out, acc = [], 1.0
        for v in c:
            acc *= max(0.0, min(1.0, float(v)))
            out.append(acc)
        return out


class ThroughputAwareScheduler(BaseScheduler):
    """
    论文主方案：在"验证长度 k"上最大化 Θ = τ·SPS(B)。

    τ(k) = 1 + Σ_{j<k} survival[j]，其中 +1 来自 k 个全接受后额外产出的地板 token。
    批次规模 B ≈ 当前并发请求数 × k（每个请求贡献 k 个验证位置）。
    """

    def __init__(self, max_draft: int = 5, profile: Optional[ThroughputProfile] = None):
        super().__init__(max_draft)
        self.profile = profile or ThroughputProfile()

    def decide(self, conf: torch.Tensor, current_batch: int = 1) -> ScheduleDecision:
        surv = self.survival(conf)[: self.max_draft]
        if not surv:
            return ScheduleDecision(1, 1.0, 0.0)

        best = None
        cum = 0.0
        for k in range(1, len(surv) + 1):
            cum += surv[k - 1]
            tau = 1.0 + cum
            batch = max(1.0, float(current_batch) * k)
            theta = tau * self.profile.sps(batch)
            if best is None or theta > best.expected_throughput:
                best = ScheduleDecision(k, tau, theta)
        return best


class ConfidenceScheduler(BaseScheduler):
    """简化方案：截断到累计存活概率低于阈值的位置。无需 SPS profile。"""

    def __init__(self, max_draft: int = 5, threshold: float = 0.35, min_len: int = 1):
        super().__init__(max_draft)
        self.threshold = float(threshold)
        self.min_len = int(min_len)

    def decide(self, conf: torch.Tensor, current_batch: int = 1) -> ScheduleDecision:
        surv = self.survival(conf)[: self.max_draft]
        k = 0
        for s in surv:
            if s < self.threshold:
                break
            k += 1
        k = max(self.min_len, min(k, len(surv)))
        tau = 1.0 + sum(surv[:k])
        return ScheduleDecision(k, tau, tau)


class FixedLengthScheduler(BaseScheduler):
    """基线：固定验证长度（等价于传统 speculative decoding 的固定 gamma）。"""

    def __init__(self, max_draft: int = 5, k: Optional[int] = None):
        super().__init__(max_draft)
        self.k = int(k if k is not None else max_draft)

    def decide(self, conf: torch.Tensor, current_batch: int = 1) -> ScheduleDecision:
        k = max(1, min(self.k, conf.numel(), self.max_draft))
        surv = self.survival(conf)[:k]
        return ScheduleDecision(k, 1.0 + sum(surv), 1.0 + sum(surv))


def build_scheduler(kind: str = "throughput", max_draft: int = 5, **kwargs):
    """kind: throughput | confidence | fixed"""
    kind = str(kind).lower()
    if kind == "throughput":
        return ThroughputAwareScheduler(max_draft=max_draft, **kwargs)
    if kind == "confidence":
        return ConfidenceScheduler(max_draft=max_draft, **kwargs)
    if kind == "fixed":
        return FixedLengthScheduler(max_draft=max_draft, **kwargs)
    raise ValueError(f"未知调度器类型: {kind}")