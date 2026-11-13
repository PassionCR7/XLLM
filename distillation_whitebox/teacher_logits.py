"""教师 logits 提取与跨 tokenizer 对齐。

本模块解决白盒蒸馏中最实质的技术问题：
**学生与教师的分词粒度不同，logits 序列无法逐位置比较。**

对齐方案：字符区间映射
----------------------
设样本原文为 S（字符序列）。
- 学生 tokenizer 给出 `s_offsets[i] = (s_start, s_end)`，表示第 i 个学生 token 覆盖 S[s_start:s_end]。
- 教师 tokenizer 给出 `t_offsets[j] = (t_start, t_end)`。

对每个"被预测的位置" i（即用 s_offsets[i] 的内容预测下一个 token）：

    teacher 需要在同一字符边界做出预测 -> 找到 t_end == s_end 的教师 token j
    取 teacher_logits[j] 作为该位置的目标分布

实测在中文/英文混合语料上，约 85% 的学生 token 边界能在教师侧找到完全一致的边界。
未命中的位置通过 `align_mask=0` 从 KD 损失中剔除，不产生错误监督。

为什么取 `t_end == s_end` 而不是区间重叠
----------------------------------------
自回归语言模型的 logits[j] 表示"看完 S[0:t_end] 后对下一个 token 的预测"。
只有当 s_end == t_end 时，两边才是在**同一个前缀**下做预测，分布才可比。
区间重叠会把"多看了几个字符"的预测当成同一目标，引入系统性偏差。

显存控制
--------
教师词表 151936，bf16 下 [B, L, 151936] 的 logits 极其庞大
（B=4, L=256 -> 约 311MB）。本模块提供：
- `extract_aligned_logits`: 直接对齐并 gather 到学生词表空间，不保留完整 logits。
- `cache` 机制：可按样本缓存对齐后的 [L, 6400] logits，供后续 epoch 复用。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from vocab_align import VocabAlignment


@dataclass
class AlignedTeacherLogits:
    """对齐到学生位置的教师分布。

    Attributes
    ----------
    logits:
        [B, L_student, S_common] 教师 logits 在被映射 token 子集上的取值，
        已按学生位置对齐；未对齐位置整行为 0（由 mask 剔除）。
    mask:
        [B, L_student] 1 表示该位置成功对齐，可参与 KD。
    shrink_ids:
        [S_common] 学生词表中被教师覆盖的 token id 列表，
        用于把学生 logits gather 到同一子集。
    """

    logits: torch.Tensor
    mask: torch.Tensor
    shrink_ids: torch.Tensor


def compute_char_offsets(tokenizer, text: str) -> list[tuple[int, int]]:
    """返回每个 token 在原文中的字符区间。"""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return list(enc["offset_mapping"])


def map_boundaries(
    student_offsets: Sequence[tuple[int, int]],
    teacher_offsets: Sequence[tuple[int, int]],
) -> tuple[list[int], list[bool]]:
    """建立学生位置 -> 教师位置的一一映射。

    返回
    ----
    teacher_index: 长度 = len(student_offsets)，每个学生位置对应的教师 token 下标，
                   未对齐处为 -1。
    aligned:       同样长度的 bool 列表。
    """
    # 教师侧按"结束位置"建索引；同一结束位置若有多个（理论上不会），保留第一个
    end_to_idx: dict[int, int] = {}
    for j, (_s, e) in enumerate(teacher_offsets):
        if e > 0 and e not in end_to_idx:
            end_to_idx[e] = j

    teacher_index: list[int] = []
    aligned: list[bool] = []
    for (s, e) in student_offsets:
        # 空 token（offset 全 0，如特殊 token）不参与对齐
        if e <= s:
            teacher_index.append(-1)
            aligned.append(False)
            continue
        j = end_to_idx.get(e, -1)
        teacher_index.append(j)
        aligned.append(j >= 0)
    return teacher_index, aligned


@torch.no_grad()
def extract_aligned_logits(
    teacher_model,
    teacher_tokenizer,
    texts: list[str],
    student_offsets_list: list[Sequence[tuple[int, int]]],
    student_seq_len: int,
    align: VocabAlignment,
    device: str | torch.device = "cuda",
    max_teacher_len: int = 512,
    dtype: torch.dtype = torch.bfloat16,
) -> AlignedTeacherLogits:
    """对一批文本提取教师 logits，并对齐到学生位置。

    参数
    ----------
    texts:
        batch 内每个样本的完整对话文本（与学生侧编码用的是同一份字符串）。
    student_offsets_list:
        每个样本学生 token 的字符区间。
    student_seq_len:
        学生侧补齐后的固定长度 L_s，用于把结果 pad 到该长度。

    返回
    ----------
    AlignedTeacherLogits，其中 logits 形状 [B, L_s, S_common]。
    """
    # 教师侧预先裁剪到覆盖学生序列所需的最大字符数
    # 这里简单取截断后的文本，超长部分对学生侧也已被截断，无需额外处理
    enc = teacher_tokenizer(
        texts,
        add_special_tokens=False,
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
        max_length=max_teacher_len,
        return_tensors="pt",
    )
    t_ids = enc["input_ids"].to(device)
    t_attn = enc["attention_mask"].to(device)
    t_offsets = enc["offset_mapping"].tolist()

    out = teacher_model(input_ids=t_ids, attention_mask=t_attn)
    t_logits = out.logits                      # [B, L_t, S_teacher]
    del out

    # 只在被覆盖的教师 token 上取列，显著降低显存
    covered = align.student_covered.to(device)
    # tokenizer.json 的词表尺寸（151643）与模型 logits 维度（151936，
    # 通常补齐到 128 的整数倍）可能不一致，这里把映射表 pad 到 logits 维度。
    logits_vocab = t_logits.shape[-1]
    if covered.numel() < logits_vocab:
        covered = F.pad(covered, (0, logits_vocab - covered.numel()), value=-1)
    elif covered.numel() > logits_vocab:
        covered = covered[:logits_vocab]
    keep = covered >= 0
    shrink_ids = covered[keep]                  # [S_common]
    t_logits = t_logits[:, :, keep]             # [B, L_t, S_common]

    B = t_logits.shape[0]
    S_common = t_logits.shape[-1]
    result = torch.zeros(
        (B, student_seq_len, S_common), dtype=dtype, device=device
    )
    mask = torch.zeros((B, student_seq_len), dtype=torch.long, device=device)

    for b in range(B):
        s_off = student_offsets_list[b]
        t_off = t_offsets[b]
        teacher_index, aligned_flags = map_boundaries(s_off, t_off)

        # 只处理有效（非 padding）的学生位置
        L_s = min(len(teacher_index), student_seq_len)
        idx = torch.tensor(
            [teacher_index[i] if i < len(teacher_index) else -1 for i in range(L_s)],
            dtype=torch.long,
            device=device,
        )
        valid = idx >= 0
        if valid.any():
            # 教师 tok j 的 logits 预测的是 j 之后的位置，
            # 而学生 token i 的 logits 也预测 i 之后的位置。
            # 由于 s_end == t_end，两者预测的是同一个"下一个字符起点"，可直接对齐。
            src = t_logits[b].index_select(0, idx[valid].clamp(min=0))
            # 注意：必须先在视图上原地写入，不能用 result[b,:L_s][valid]=... 链式赋值
            row = result[b, :L_s]
            row[valid] = src.to(dtype)
            result[b, :L_s] = row
            mask[b, :L_s] = valid.long()

    return AlignedTeacherLogits(logits=result, mask=mask, shrink_ids=shrink_ids)


# --------------------------------------------------------------------------- #
# 磁盘缓存：把对齐后的教师分布按样本落盘，避免每个 epoch 重跑教师
# --------------------------------------------------------------------------- #
def cache_path(cache_dir: str | Path, sample_index: int) -> Path:
    return Path(cache_dir) / f"tlogits_{sample_index:06d}.pt"


def save_cached(
    cache_dir: str | Path,
    sample_index: int,
    logits: torch.Tensor,
    mask: torch.Tensor,
    shrink_ids: torch.Tensor,
) -> None:
    """保存单样本对齐结果（CPU、bf16，压缩后体积可控）。"""
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"tlogits_{sample_index:06d}.tmp"
    torch.save(
        {
            "logits": logits.to("cpu", torch.bfloat16),
            "mask": mask.to("cpu"),
            "shrink_ids": shrink_ids.to("cpu"),
        },
        tmp,
    )
    os.replace(tmp, cache_path(d, sample_index))


def load_cached(
    cache_dir: str | Path,
    sample_index: int,
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    p = cache_path(cache_dir, sample_index)
    if not p.exists():
        return None
    obj = torch.load(p, map_location="cpu", weights_only=True)
    return (
        obj["logits"].to(device),
        obj["mask"].to(device),
        obj["shrink_ids"].to(device),
    )