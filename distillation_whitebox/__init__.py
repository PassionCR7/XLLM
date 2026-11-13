"""白盒蒸馏包。

教师为本地 Qwen3-0.6B 权重，可访问完整 logits，实现真正的分布蒸馏。

典型用法
--------
    python train_kd.py --epochs 3 --batch_size 4   # 训练（含教师 logits 对齐）
    python evaluate.py --weights full_sft distill_whitebox
"""

__all__ = [
    "config",
    "vocab_align",
    "dataset",
    "teacher_logits",
    "student_loader",
    "train_kd",
    "evaluate",
]