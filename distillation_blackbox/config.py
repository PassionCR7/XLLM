"""黑盒蒸馏全局配置。

教师侧是闭源 API（仅有文本输出，无 logits），因此本目录实现的是
**response-level / logprob-level 黑盒蒸馏**，而不是分布蒸馏。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 项目根目录（xllm/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_api_credentials(key_file: str) -> tuple[str, str, list[str]]:
    """从 api key 文本中解析 base_url / api_key / 候选模型名。

    该文件格式不固定（中文标签 + 冒号），因此用宽松正则提取，
    避免任何硬编码密钥进仓库。
    """
    import re

    text = Path(key_file).read_text(encoding="utf-8")
    key_match = re.search(r"API\s*Key\s*[:：]\s*(\S+)", text)
    url_match = re.search(r"(https?://[^\s]+)", text)
    model_line = re.search(r"模型名字\s*[:：]\s*(.+)", text)

    if not key_match or not url_match:
        raise ValueError(f"无法从 {key_file} 解析出 API Key 或 base_url")

    models: list[str] = []
    if model_line:
        # 形如 "deepseek-v4.1-flash-1017  或者  deepseek-v4-pro"
        for part in re.split(r"或者|,|，|/|\s{2,}", model_line.group(1)):
            name = part.strip()
            if name and name not in models:
                models.append(name)

    return key_match.group(1), url_match.group(1).rstrip("/"), models


@dataclass
class TeacherAPIConfig:
    """教师 API 配置。"""

    key_file: str = str(Path.home() / "Desktop" / "mygithub" / "可使用的API-Key.txt")
    api_key: str = ""
    base_url: str = ""
    model: str = "deepseek-v4.1-flash-1017"
    candidate_models: list[str] = field(default_factory=list)

    # 采样参数：蒸馏必须低温，保证教师输出稳定可复现
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0

    # 该教师是 reasoning 模型，默认走思考通道导致 content 为空。
    # 关闭 thinking 后 logprob 才是真实概率值（实测 -0.0014 量级）。
    enable_thinking: bool = False

    # 是否请求 token 级 logprob（用于 logprob-level 蒸馏）
    request_logprobs: bool = True
    top_logprobs: int = 5

    # 网络健壮性
    timeout: int = 120
    max_retries: int = 5
    retry_backoff: float = 2.0
    # 并发请求数：API 侧通常有限流，保守起步
    concurrency: int = 4

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key, self.base_url, self.candidate_models = _read_api_credentials(
                self.key_file
            )
        if self.candidate_models and self.model not in self.candidate_models:
            self.model = self.candidate_models[0]

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"


@dataclass
class StudentConfig:
    """学生（xllm）配置。"""

    model_dir: str = str(PROJECT_ROOT / "model")
    weight_dir: str = str(PROJECT_ROOT / "out")
    from_weight: str = "full_sft"      # pretrain / full_sft / grpo ...
    hidden_size: int = 512
    num_hidden_layers: int = 8
    use_moe: bool = False
    max_seq_len: int = 512
    # 保存目录
    save_dir: str = str(PROJECT_ROOT / "distillation_blackbox" / "out")
    save_weight: str = "distill_blackbox"


@dataclass
class DataConfig:
    """数据配置。"""

    # 教师生成回答所用的种子问题集
    seed_file: str = str(PROJECT_ROOT / "dataset" / "sft_t2t_mini.jsonl")
    # 蒸馏产物（教师对话数据）
    out_dir: str = str(PROJECT_ROOT / "distillation_blackbox" / "data")
    # 生成多少条教师样本
    num_samples: int = 2000
    # 每条种子最多派生几个 prompt
    max_prompts_per_seed: int = 1
    # 是否保留教师 reasoning_content 作为 CoT
    keep_reasoning: bool = False


@dataclass
class TrainConfig:
    """训练配置。"""

    epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 5e-5
    accumulation_steps: int = 1
    grad_clip: float = 1.0
    log_interval: int = 20
    save_interval: int = 200
    num_workers: int = 0            # Windows 下多进程 DataLoader 易死锁，默认 0
    dtype: str = "bfloat16"
    # 损失构成：CE（硬标签） + KL（教师 logprob 软标签）
    ce_weight: float = 1.0
    kd_weight: float = 0.5
    # 软标签蒸馏温度。黑盒场景教师只给 top-k，软件标签本身已是
    # 绝对值概率（未重归一化），T 略大于 1 即可软化，取 1.0 保持原分布。
    temperature: float = 1.0
    use_wandb: bool = False


@dataclass
class DistillConfig:
    teacher: TeacherAPIConfig = field(default_factory=TeacherAPIConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(key_file: str | None = None) -> DistillConfig:
    cfg = DistillConfig()
    if key_file:
        cfg.teacher.key_file = key_file
        cfg.teacher.api_key = ""
        cfg.teacher.__post_init__()
    return cfg