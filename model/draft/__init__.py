# -*- coding: utf-8 -*-
"""
model.draft —— 草稿模型辅助主模型推理（推测解码）

三套并行实现，技术路线不同，可分别使用也可做对照实验。按目录分开存放：

    common/     共用核心：接受/验证算法（三套 engine 都调它）
    serial/     A：串行草稿 + KV 同步（基线实现，零训练可用）
    dspark/     B：DSpark 式半自回归草稿（对齐 DeepSeek-V4.1 §2.4.3）
    mtp/        C：MTP 联合预训练（对齐 DeepSeek-V3 / V4）

快速选型
--------
    * 只需跑通链路、验证正确性  -> A 的 from_main_model（零训练）
    * 要接近论文的加速比        -> B，且必须真正训练草稿（dspark/dspark_train.py）
    * 从零开始训主模型          -> C，联合预训练白拿草稿头，且密集监督使主干变好

B 与 C 的本质区别（最容易混淆）
------------------------------
    B: target 冻结，梯度不回传，草稿只能**逼近**已有的 p。
    C: 预训练期梯度穿过主干，草稿与主干**共同塑造** p，是真正的联合训练。

三套共用 common._verification 的接受规则，无损性保证一致：
贪心模式与 target 贪心解码逐 token 相同；随机模式与 target 采样无偏。

兼容性说明
----------
本文件仍把三套的公开符号全部平铺导出，因此原有的
    from model.draft import DraftModel, SpeculativeDecodingEngine
    from model.draft import DSparkConfig, DSparkDrafter, DSparkEngine, build_scheduler
    from model.draft import MTPEngine
等写法无需修改。也可以按实现方式显式导入，语义更清晰：
    from model.draft.mtp import MTPEngine
"""

from .common import (
    acceptance_stats,
    prepare_distribution,
    sample_token,
    verify_and_sample,
)

from .serial import DraftModel, SpeculativeDecodingEngine

from .dspark import (
    BaseScheduler,
    ConfidenceScheduler,
    DSparkConfig,
    DSparkDrafter,
    DSparkEngine,
    FixedLengthScheduler,
    ScheduleDecision,
    ThroughputAwareScheduler,
    ThroughputProfile,
    build_scheduler,
)

from .mtp import MTPConfig, MTPEngine, MTPHead, XLLMWithMTP

__all__ = [
    # 共用
    "verify_and_sample",
    "prepare_distribution",
    "sample_token",
    "acceptance_stats",
    # A：串行草稿
    "DraftModel",
    "SpeculativeDecodingEngine",
    # B：DSpark
    "DSparkDrafter",
    "DSparkConfig",
    "DSparkEngine",
    "BaseScheduler",
    "ThroughputAwareScheduler",
    "ConfidenceScheduler",
    "FixedLengthScheduler",
    "ThroughputProfile",
    "ScheduleDecision",
    "build_scheduler",
    # C：MTP 联合预训练
    "MTPHead",
    "MTPConfig",
    "XLLMWithMTP",
    "MTPEngine",
]