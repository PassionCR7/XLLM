# -*- coding: utf-8 -*-
"""
common —— 三套实现共用的推测解码核心规则

只做一件事：给定 target 分布与草稿 token，判定接受/拒绝并采出 token。
三套 engine（serial / dspark / mtp）都调用这里，所以它们的**无损性保证完全
一致**：贪心模式与 target 贪心解码逐 token 相同；随机模式与 target 采样无偏。

单独抽出来是为了防止一种常见错误——各实现在自己的 engine 里各写一份接受
逻辑，改了一处忘了另一处，导致"某条路线输出正确、另一条悄悄有偏"。
"""

from ._verification import (
    acceptance_stats,
    prepare_distribution,
    sample_token,
    verify_and_sample,
)

__all__ = [
    "verify_and_sample",
    "prepare_distribution",
    "sample_token",
    "acceptance_stats",
]