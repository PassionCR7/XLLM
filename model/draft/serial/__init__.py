# -*- coding: utf-8 -*-
"""
serial —— A：串行草稿 + KV 同步（基线实现）

思路最朴素的一种：把主模型的前 N 层截断出来当草稿模型，草稿和 target 各自
持有自己的 KV cache，每轮由 engine 负责把它们对齐、裁剪、同步。

    DraftModel                 截断式草稿模型（复用 XLLMForCausalLM 的前 N 层）
    SpeculativeDecodingEngine  推测解码引擎，负责 KV 同步与裁剪

特点
----
* **零训练可用**：`DraftModel.from_main_model` 直接截断已有权重就能跑，
  适合验证链路正确性。
* **加速比最弱**：草稿本身是一次真实的前向，成本不低，且草稿与 target 的
  分布差距大，接受率低。
* **KV 同步是主要复杂度来源**：接受长度不定，两个 cache 都要截到同一位置，
  这里最容易出 off-by-one。

想跑通流程、做正确性对照实验时用这条。要真实加速请看 dspark 或 mtp。
"""

from .draft_model import DraftModel
from .speculative_decoding import SpeculativeDecodingEngine

__all__ = ["DraftModel", "SpeculativeDecodingEngine"]