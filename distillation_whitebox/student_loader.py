"""学生（xllm）模型加载工具（白盒目录自带，保持目录自包含）。

注意：项目中 `out/pretrain_512.pth` 存在键名拼写缺陷
`post_attention_layernotm`（应为 `post_attention_layernorm`）。
若直接 `load_state_dict(strict=False)`，会静默漏加载 8 层后注意力归一化权重，
导致加载"成功"但模型行为异常。本模块在加载前自动修正此类键名并强制校验。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.XLLMModel import XLLMConfig, XLLMForCausalLM  # noqa: E402

# 已知的历史键名拼写错误 -> 正确键名
_KEY_TYPOS = {
    "post_attention_layernotm": "post_attention_layernorm",
}


def fix_state_dict_keys(state_dict: dict) -> tuple[dict, int]:
    """修正 checkpoint 中的键名拼写问题，返回 (新state_dict, 修正数量)。"""
    fixed: dict = {}
    n_fixed = 0
    for k, v in state_dict.items():
        new_k = k
        for bad, good in _KEY_TYPOS.items():
            if bad in new_k:
                new_k = new_k.replace(bad, good)
                n_fixed += 1
                break
        fixed[new_k] = v
    return fixed, n_fixed


def build_config(
    hidden_size: int = 512,
    num_hidden_layers: int = 8,
    use_moe: bool = False,
) -> XLLMConfig:
    return XLLMConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        use_moe=use_moe,
    )


def load_student(
    cfg,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    strict_report: bool = True,
):
    """加载学生模型与分词器。

    参数
    ----
    cfg : StudentModelConfig / StudentConfig
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir)
    lm_config = build_config(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        use_moe=cfg.use_moe,
    )
    model = XLLMForCausalLM(lm_config)

    if cfg.from_weight and cfg.from_weight.lower() != "none":
        moe_suffix = "_moe" if cfg.use_moe else ""
        suffix = f"_{cfg.hidden_size}{moe_suffix}"
        stem = (
            cfg.from_weight
            if cfg.from_weight.endswith(suffix)
            else f"{cfg.from_weight}{suffix}"
        )
        weight_path = os.path.join(cfg.weight_dir, f"{stem}.pth")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"未找到学生权重: {weight_path}")

        sd = torch.load(weight_path, map_location="cpu", weights_only=True)
        sd, n_fixed = fix_state_dict_keys(sd)
        if n_fixed:
            print(f"[学生] 已修正 {n_fixed} 个键名拼写错误 (layernotm -> layernorm)")

        result = model.load_state_dict(sd, strict=False)
        if strict_report:
            missing = [k for k in result.missing_keys]
            unexpected = [k for k in result.unexpected_keys]
            if missing:
                print(f"[学生][警告] 缺失权重 {len(missing)} 个: {missing[:4]}")
            if unexpected:
                print(f"[学生][警告] 多余权重 {len(unexpected)} 个: {unexpected[:4]}")
            if not missing and not unexpected:
                print("[学生] 权重完整加载")
        del sd

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[学生] 参数量 {n_params / 1e6:.2f} M | 词表 {lm_config.vocab_size} | "
          f"hidden {lm_config.hidden_size} | layers {lm_config.num_hidden_layers}")

    model = model.to(device=device, dtype=dtype)
    return model, tokenizer


def save_student(model, cfg, tag: str | None = None) -> str:
    """保存学生权重（半精度，与项目其他 checkpoint 保持一致）。"""
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    name = tag or cfg.save_weight
    moe_suffix = "_moe" if cfg.use_moe else ""
    suffix = f"_{cfg.hidden_size}{moe_suffix}"
    # tag 可能自带后缀（续训时会传完整权重名），避免叠成 ..._512_512.pth
    stem = name if name.endswith(suffix) else f"{name}{suffix}"
    path = save_dir / f"{stem}.pth"

    raw = getattr(model, "_orig_mod", model)
    sd = raw.state_dict()
    tmp = str(path) + ".tmp"
    torch.save({k: v.half() for k, v in sd.items()}, tmp)
    os.replace(tmp, path)
    print(f"[学生] 权重已保存 -> {path}")
    return str(path)