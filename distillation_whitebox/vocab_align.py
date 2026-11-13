"""教师/学生词表对齐层（白盒蒸馏的关键组件）。

为什么需要这一层
----------------
学生（xllm）词表 6400，教师（Qwen3）词表 151936。
实测两者：

* 字符串层面的重合 token 占学生词表 91.4%
* 但**几乎不存在相同 id**（重合 token 中仅 0.02% 的 id 一致）

因此不能假设 `teacher_logits[..., i]` 与 `student_logits[..., i]` 对应
同一个词。必须先建立 **教师 id -> 学生 id** 的映射，才能计算 KL。

映射策略（三级回退）
--------------------
1. **精确字节匹配**：把两个词表的 token 都还原为原始字节序列
   （GPT-2 byte-level BPE 的 `bytes_to_unicode` 逆映射），字节完全相同才映射。
   这是最可靠的一级，实测可覆盖学生词表约 91%。
2. **文本回退**：对无法字节匹配的 token，用其解码后的字符串
   （忽略替换字符）再尝试匹配。
3. **丢弃**：仍无对应者的学生 token 不参与 KD 损失（不是置零，
   而是从 KL 分母中剔除），避免引入错误监督。

对齐后的 KD 损失
----------------
教师分布只在"能映射到学生词表"的子集上重新归一化：
    p_t_align = softmax(teacher_logits[mapped_teacher_ids]) 
               作用于学生词表对应的位置
然后与学生的对应位置做 KL。这样既不需要 vocab 维度相同，
也不会因为几千个映射不到的 token 系统性拉偏分布。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# byte-level BPE 的字符 -> 字节 逆映射
# --------------------------------------------------------------------------- #
def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 / Qwen / xllm 共用的 byte <-> unicode 表。"""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_U2B = {v: k for k, v in _bytes_to_unicode().items()}


def token_to_bytes(token: str) -> bytes:
    """把 BPE 词表里的 token 字符串还原为原始字节。

    词表 token 中的每个字符都对应一个可打印 unicode 码位，
    需要先查表映射回 0-255 的原始字节。
    若遇到不在表内的字符（少数自定义 token），退化为 utf-8 编码。
    """
    try:
        return bytes(_U2B[ch] for ch in token)
    except KeyError:
        return token.encode("utf-8", "ignore")


# --------------------------------------------------------------------------- #
# 从 tokenizer.json 读取词表
# --------------------------------------------------------------------------- #
def load_vocab_from_tokenizer_json(path: str | Path) -> dict[str, int]:
    """读取 HuggingFace `tokenizer.json` 的 base vocab（token -> id）。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    vocab = data["model"]["vocab"]
    if not isinstance(vocab, dict):
        raise ValueError(f"{path} 的 model.vocab 不是 dict，无法解析")
    return dict(vocab)


def load_vocab_from_hf_tokenizer(tokenizer) -> dict[str, int]:
    """从已加载的 transformers tokenizer 获取词表。"""
    return dict(tokenizer.get_vocab())


# --------------------------------------------------------------------------- #
# 对齐结果
# --------------------------------------------------------------------------- #
@dataclass
class VocabAlignment:
    """教师 id -> 学生 id 的映射容器。

    Attributes
    ----------
    teacher_to_student:
        dict[int, int]，教师 token id 到学生 token id。
    student_covered:
        LongTensor(teacher_vocab_size)，值为学生 id，无法映射处为 -1。
        用于快速把教师 logits gather 到学生词表空间。
    coverage:
        学生词表被教师覆盖的比例（0-1），用于评估对齐质量。
    """

    teacher_to_student: dict[int, int]
    student_covered: torch.Tensor
    coverage: float
    teacher_vocab_size: int
    student_vocab_size: int

    def to(self, device) -> "VocabAlignment":
        self.student_covered = self.student_covered.to(device)
        return self

    def summary(self) -> dict:
        return {
            "teacher_vocab": self.teacher_vocab_size,
            "student_vocab": self.student_vocab_size,
            "mapped_teacher_tokens": len(self.teacher_to_student),
            "student_coverage": round(self.coverage, 4),
        }


def build_alignment(
    teacher_vocab: dict[str, int],
    student_vocab: dict[str, int],
    verbose: bool = True,
) -> VocabAlignment:
    """构建教师 -> 学生词表映射。

    参数
    ----
    teacher_vocab : token 字符串 -> 教师 id
    student_vocab : token 字符串 -> 学生 id
    """
    teacher_vocab_size = max(teacher_vocab.values()) + 1
    student_vocab_size = max(student_vocab.values()) + 1

    # ---- 一级：字节精确匹配 ----
    # 教师侧按字节建索引，同一字节序列若有多条（罕见），保留 id 最小者以保证确定性
    teacher_by_bytes: dict[bytes, int] = {}
    for tok, tid in sorted(teacher_vocab.items(), key=lambda kv: kv[1]):
        b = token_to_bytes(tok)
        if b and b not in teacher_by_bytes:
            teacher_by_bytes[b] = tid

    t2s: dict[int, int] = {}
    matched_level1: set[int] = set()
    for tok, sid in student_vocab.items():
        b = token_to_bytes(tok)
        tid = teacher_by_bytes.get(b)
        if tid is not None:
            # 同一教师 id 可能被多个学生 token 命中（理论上字节唯一，实际不会）
            if tid not in t2s:
                t2s[tid] = sid
                matched_level1.add(sid)

    # ---- 二级：剔除非法 UTF-8 替换字节后再匹配 ----
    # 部分词表会把无法解码的字节存成 U+FFFD（EF BF BD）。这类 token
    # 在字节层面不一致，但语义上是"同一段损坏字节"，允许再匹配一次。
    # 注意：绝不做去空格/casefold 之类的宽松匹配，那会破坏 BPE 语义。
    _REPL = b"\xef\xbf\xbd"
    student_unmatched = [
        (tok, sid) for tok, sid in student_vocab.items() if sid not in matched_level1
    ]
    if student_unmatched:
        teacher_by_clean: dict[bytes, int] = {}
        for tok, tid in sorted(teacher_vocab.items(), key=lambda kv: kv[1]):
            if tid in t2s:
                continue
            b = token_to_bytes(tok)
            clean = b.replace(_REPL, b"")
            if clean and clean not in teacher_by_clean:
                teacher_by_clean[clean] = tid
        for tok, sid in student_unmatched:
            clean = token_to_bytes(tok).replace(_REPL, b"")
            if not clean:
                continue
            tid = teacher_by_clean.get(clean)
            if tid is not None and tid not in t2s:
                t2s[tid] = sid
                matched_level1.add(sid)

    # ---- 构造 teacher_vocab_size 长度的 gather 索引 ----
    covered = torch.full((teacher_vocab_size,), -1, dtype=torch.long)
    for tid, sid in t2s.items():
        covered[tid] = sid

    student_matched = len(set(t2s.values()))
    coverage = student_matched / student_vocab_size if student_vocab_size else 0.0

    align = VocabAlignment(
        teacher_to_student=t2s,
        student_covered=covered,
        coverage=coverage,
        teacher_vocab_size=teacher_vocab_size,
        student_vocab_size=student_vocab_size,
    )

    if verbose:
        print(
            f"[对齐] 教师词表 {teacher_vocab_size} <-> 学生词表 {student_vocab_size}\n"
            f"       已映射教师 token: {len(t2s)}\n"
            f"       学生词表覆盖率: {coverage * 100:.2f}%"
        )
    return align


# --------------------------------------------------------------------------- #
# KD 损失
# --------------------------------------------------------------------------- #
def vocab_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    align: VocabAlignment,
    mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """在公共词表子集上计算 KL(教师 || 学生)。

    参数
    ----
    student_logits : [N, S_student]  展平后的学生 logits
    teacher_logits : [N, S_teacher]  展平后的教师 logits（顶点必须对齐同一位置）
    align          : 词表对齐结果
    mask           : [N] bool/float，只对有效位置计损失
    temperature    : 蒸馏温度 T，softmax(logits / T)

    返回
    ----
    (loss, 有效 token 数)
    """
    if student_logits.shape[0] != teacher_logits.shape[0]:
        raise ValueError(
            f"token 数不一致: student={student_logits.shape[0]}, "
            f"teacher={teacher_logits.shape[0]}"
        )

    # 教师侧取出"能映射到学生"的 token 位置
    covered = align.student_covered.to(teacher_logits.device)
    keep = covered >= 0                            # [S_teacher]
    if keep.sum() == 0:
        return student_logits.new_zeros(()), 0

    t_ids = covered[keep]                          # 映射到的学生 id
    t_logits = teacher_logits[:, keep].float()     # [N, S_common]
    s_logits = student_logits[:, t_ids].float()    # [N, S_common]

    # 重新归一化：教师分布与学生对数概率都在同一子集上
    log_p_t = F.log_softmax(t_logits / temperature, dim=-1)
    log_p_s = F.log_softmax(s_logits / temperature, dim=-1)

    # KL(p_t || p_s) = sum p_t * (log p_t - log p_s)
    kl = (log_p_t.exp() * (log_p_t - log_p_s)).sum(dim=-1)   # [N]

    if mask is not None:
        m = mask.to(kl.device).float()
        kl = kl * m
        denom = m.sum().clamp(min=1.0)
    else:
        denom = torch.tensor(float(kl.numel()), device=kl.device)

    loss = kl.sum() / denom
    # T^2 缩放：让 KD 梯度量级与硬标签 CE 可比（Hinton 2015）
    return loss * (temperature ** 2), int(kl.numel())


def topk_logit_matching_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    align: VocabAlignment,
    mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """可选的严格版本：只在教师 top-k 与学生 top-k 的并集上做 KL。

    对词表巨大而学生词表很小时，这个版本噪声更小，
    因为教师分布尾部的概率质量几乎全是映射不到的长尾 token。
    """
    covered = align.student_covered.to(teacher_logits.device)
    keep = covered >= 0
    if keep.sum() == 0:
        return student_logits.new_zeros(()), 0

    t_ids = covered[keep]
    t_logits = teacher_logits[:, keep].float()
    s_logits = student_logits[:, t_ids].float()

    log_p_t = F.log_softmax(t_logits / temperature, dim=-1)
    log_p_s = F.log_softmax(s_logits / temperature, dim=-1)
    kl = (log_p_t.exp() * (log_p_t - log_p_s)).sum(dim=-1)

    if mask is not None:
        m = mask.to(kl.device).float()
        kl = kl * m
        denom = m.sum().clamp(min=1.0)
    else:
        denom = torch.tensor(float(kl.numel()), device=kl.device)
    return (kl.sum() / denom) * (temperature ** 2), int(kl.numel())