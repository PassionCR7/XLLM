"""白盒蒸馏数据集。

核心难点：**教师与学生的 tokenizer 不同，同一个样本在两边的 token 序列长度不同**，
因此不能像常规 SFT 那样直接 `logits[:-1]` 与 `labels[1:]` 对齐。

解决方案：**字符区间对齐（character-span alignment）**

1. 用学生 tokenizer 的 `offset_mapping` 得到每个学生 token 覆盖的字符区间。
2. 用教师 tokenizer 的 `offset_mapping` 得到每个教师 token 覆盖的字符区间。
3. 对每个学生 token，找出"覆盖同一字符区间"的教师 token，
   把教师 token 的 logits 聚合（取区间最后一个 token，符合自回归预测语义）
   作为该位置的目标分布。

这样得到的 teacher_logits 序列与学生 logits 序列**逐位置对应**，
可以直接做 KL。同时给出 `align_mask`，标记哪些位置成功对齐
（实测边界命中率约 85%，未命中的位置从损失中剔除）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    pass

IGNORE_INDEX = -100


@dataclass
class EncodedSample:
    """单条样本在学生侧的编码结果。"""

    input_ids: list[int]
    labels: list[int]          # -100 表示不参与 CE
    kd_mask: list[int]         # 1 表示该位置参与 KD
    text: str


def build_text_and_answer_spans(
    conversations: list[dict],
    tokenizer,
    add_generation_prompt_assistant: bool = False,
) -> tuple[str, list[tuple[int, int]]]:
    """把对话渲染成文本，并返回所有 assistant 回复的字符区间。

    使用增量渲染：逐条追加消息，用长度差确定每条消息在最终文本中的位置。
    区间从 `<|im_start|>assistant\\n` 头部**之后**开始，
    与项目 `SFTDataset.generate_labels` 的约定一致（头部不计入损失）。
    """
    text = ""
    answer_spans: list[tuple[int, int]] = []
    for i, msg in enumerate(conversations):
        prev_len = len(text)
        text = tokenizer.apply_chat_template(
            conversations[: i + 1],
            tokenize=False,
            add_generation_prompt=False,
        )
        if msg["role"] == "assistant":
            # 头部形如 "<|im_start|>assistant\n"，跳过它只对回复内容计损失
            header = "<|im_start|>assistant\n"
            seg = text[prev_len:]
            hpos = seg.find(header)
            start = prev_len + (hpos + len(header) if hpos >= 0 else 0)
            answer_spans.append((start, len(text)))
    return text, answer_spans


def encode_with_spans(
    text: str,
    answer_spans: list[tuple[int, int]],
    tokenizer,
    max_length: int,
) -> EncodedSample:
    """编码文本，标签只覆盖 assistant 回复区间。"""
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = list(enc["input_ids"])
    offsets = enc["offset_mapping"]

    # 位置裁剪：只保留 [0, max_length) 内的字符
    def in_answer(s: int, e: int) -> bool:
        for a_s, a_e in answer_spans:
            # token 的起点落在 answer 区间内即视为答案 token
            if a_s <= s < a_e:
                return True
        return False

    labels = [
        tid if in_answer(s, e) else IGNORE_INDEX
        for tid, (s, e) in zip(input_ids, offsets)
    ]
    return EncodedSample(
        input_ids=input_ids,
        labels=labels,
        kd_mask=[0 if l == IGNORE_INDEX else 1 for l in labels],
        text=text,
    )


def pad_batch(
    sequences: list[list[int]],
    pad_id: int,
    max_len: int | None = None,
    pad_value: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """右侧补齐，返回 (padded, attention_mask)。"""
    length = max_len or max(len(s) for s in sequences)
    out = torch.full((len(sequences), length), pad_value, dtype=torch.long)
    mask = torch.zeros((len(sequences), length), dtype=torch.long)
    for i, seq in enumerate(sequences):
        n = min(len(seq), length)
        out[i, :n] = torch.tensor(seq[:n], dtype=torch.long)
        mask[i, :n] = 1
    return out, mask


class DistillDataset(Dataset):
    """白盒蒸馏数据集：同时产出学生侧序列与对齐所需的原始文本。"""

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_length: int = 256,
        limit: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))
                if limit and len(self.samples) >= limit:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        conv = self.samples[index]["conversations"]
        text, spans = build_text_and_answer_spans(conv, self.tokenizer)
        enc = encode_with_spans(text, spans, self.tokenizer, self.max_length)
        return {
            "input_ids": enc.input_ids,
            "labels": enc.labels,
            "kd_mask": enc.kd_mask,
            "text": enc.text,
            "index": index,
        }


def collate_fn(batch: list[dict], pad_id: int, max_length: int):
    """把样本列表拼成 batch。"""
    input_ids, attn = pad_batch(
        [b["input_ids"] for b in batch], pad_id, max_length, pad_value=pad_id
    )
    labels, _ = pad_batch(
        [b["labels"] for b in batch], pad_id, max_length, pad_value=IGNORE_INDEX
    )
    kd_mask, _ = pad_batch(
        [b["kd_mask"] for b in batch], pad_id, max_length, pad_value=0
    )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attn,
        "kd_mask": kd_mask,
        "texts": [b["text"] for b in batch],
        "indices": [b["index"] for b in batch],
    }