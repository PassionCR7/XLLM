# -*- coding: utf-8 -*-
"""
model.draft —— 草稿模型辅助主模型推理（推测解码 / Speculative Decoding）

对外暴露的核心接口：
    DraftModel              : 草稿模型（复用 XLLMForCausalLM 基座）
    SpeculativeDecodingEngine : 推测解码引擎（draft 提议 + 主模型并行验证）
    verify_and_sample       : 单回合验证/接受算法（纯 torch）
    prepare_distribution    : logits -> 概率分布（temperature/top-k/top-p）
    acceptance_stats        : 接受率与近似加速比统计
"""

from ._verification import (
    acceptance_stats,
    prepare_distribution,
    sample_token,
    verify_and_sample,
)
from .draft_model import DraftModel
from .speculative_decoding import SpeculativeDecodingEngine

__all__ = [
    "DraftModel",
    "SpeculativeDecodingEngine",
    "verify_and_sample",
    "prepare_distribution",
    "sample_token",
    "acceptance_stats",
]
