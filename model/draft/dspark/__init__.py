# -*- coding: utf-8 -*-
"""
dspark —— B：DSpark 式半自回归草稿（对齐 DeepSeek-V4.1 §2.4.3）

与 serial 的关键差别：草稿**不需要自己的 KV cache**。并行骨干一次吐出整块
候选，再用一个轻量 Markov 链做局部修正，最后由置信度头决定每轮提交几个。

    DSparkDrafter       并行骨干 + Markov 修正链 + 置信度头
    DSparkEngine        只维护 target 的 KV（结构上消除错位与同步问题）
    dspark_scheduler    置信度调度器：throughput / confidence / fixed 三种策略

特点
----
* **结构上无错位**：草稿没有 cache，也就没有"两个 cache 怎么对齐"的问题，
  这是它相对 serial 最大的工程优势。
* **必须真正训练**（`dspark_train.py`）才有加速比；未训练时 Markov 链和
  置信度头都是随机值，接受率接近随机基线。
* **target 冻结**：梯度不回传主干，草稿只能逼近已有的 p。
  这一点是它与 mtp 的分水岭 —— mtp 是预训练期联合训练。

变长提交由调度器决定，这是 DSpark 相对固定长度推测解码的另一处优势。
"""

from .dspark_draft import DSparkConfig, DSparkDrafter
from .dspark_engine import DSparkEngine
from .dspark_scheduler import (
    BaseScheduler,
    ConfidenceScheduler,
    FixedLengthScheduler,
    ScheduleDecision,
    ThroughputAwareScheduler,
    ThroughputProfile,
    build_scheduler,
)

__all__ = [
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
]