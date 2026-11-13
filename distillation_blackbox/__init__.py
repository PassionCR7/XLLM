"""黑盒蒸馏包。

教师为闭源 API（deepseek-v4.1-flash-1017），只能获得文本与 token logprob，
因此本包实现 response-level / logprob-level 蒸馏。

典型用法
--------
    python gen_data.py --num_samples 2000          # 1) 生成教师数据
    python train_kd.py --epochs 2                  # 2) 训练学生
    python evaluate.py --weights full_sft distill_blackbox
"""

__all__ = ["config", "teacher_api", "gen_data", "train_kd", "evaluate"]