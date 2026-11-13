# -*- coding: utf-8 -*-
"""
mtp —— C：MTP 联合预训练（对齐 DeepSeek-V3 / V4 的 Multi-Token Prediction）

在主干每层之后挂 D 个轻量 MTP 模块，训练时同时预测 t+1..t+D+1 多个位置。
模块与主干**共享 embedding 与 lm_head**，参数量只为多出的 Transformer 块。

    MTPHead       单个 MTP 模块（共享 emb / lm_head，自带一层 XLLMBlock）
    XLLMWithMTP   主干 + D 个模块，损失 = L_main + λ/D · Σ L_MTP
    MTPConfig     超参，MTPConfig.from_backbone 从主干配置派生
    MTPEngine     推理引擎：把训好的头接进生成流程，测接受率

与 dspark(B) 的本质区别
-----------------------
    B: target 冻结，梯度不回传，草稿只能**逼近**已有的 p。
    C: 预训练期梯度穿过主干，草稿与主干**共同塑造** p —— 真正的联合训练，
       而且密集监督信号会让主干本身变好，这是 B 的事后训练拿不到的。

草稿宽度的约定（容易搞错）
--------------------------
主头给出第 1 个草稿（就是主干自己的 argmax，零额外成本），D 个模块负责后续
D 个。所以 D 个模块实际产出 **1+D** 个草稿。读接受率时务必记住这一点：
D=1 时"每轮平均接受 1.0"是**未训练基线**，不是好结果。

上手见同目录 MT_PRETRAIN_GUIDE.md。
"""

from .mtp_engine import MTPEngine
from .mtp_module import MTPConfig, MTPHead, XLLMWithMTP

__all__ = ["MTPHead", "MTPConfig", "XLLMWithMTP", "MTPEngine"]