"""把采集到的原始文档构建成预训练语料。

输入  out/raw/*.jsonl   （RawDoc，含 url/title/text/score）
输出  out/corpus/corpus.jsonl
      每行 {"text": "..."}，与 dataset/pretrain_t2t_mini.jsonl 完全同构，
      可直接被 dataset/lm_dataset.py 读取。

处理步骤：
  1. 跨源去重：正文 SimHash 近似去重 + 精确文本去重（同一文章多栏目转载常见）
  2. 结构重组：标题 + 正文拼成连续自然语言段落，适合 next-token prediction
  3. 二次清洗：去掉残存图片说明、日期行、极短碎片行
  4. 质量过滤：中文占比、长度区间、重复行比例
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from typing import Dict, Iterable, List, Tuple

from .crawl import clean_body, quality_ok

CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# 图片说明常见形态：「xxx 摄」「xxx/摄」「图/xxx」
CAPTION_RE = re.compile(r"(摄$|摄\s|／摄|/摄|图/|图片来源|供图|资料图)")
DATE_LINE_RE = re.compile(r"^\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?$")

# 页脚备案/许可证信息。这类文本会被密度打分误当作正文尾巴收进来，
# 实测在杭州网系文章末尾稳定出现，必须显式剔除。
FOOTER_RE = re.compile(
    r"(ICP备|ICP证|公网安备|增值电信业务|信息网络传播视听节目|"
    r"互联网新闻信息服务许可|网络文化经营许可|浙网文|"
    r"Copyright|版权所有|举报电话|违法和不良信息)"
)


def _simhash(text: str, bits: int = 64) -> int:
    """字符级 3-gram SimHash，用于近似去重。

    不用分词器（本机无 jieba），字符 n-gram 对中文去重已经足够——
    转载文章的标题/正文几乎逐字相同。
    """
    if not text:
        return 0
    tokens = [text[i : i + 3] for i in range(max(len(text) - 2, 1))]
    vec = [0] * bits
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:16], 16)
        for b in range(bits):
            vec[b] += 1 if (h >> b) & 1 else -1
    out = 0
    for b in range(bits):
        if vec[b] > 0:
            out |= 1 << b
    return out


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _normalize_body(text: str) -> str:
    """行级清洗：去掉页脚备案、图片说明、纯日期行、极短碎片。"""
    lines: List[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        # 页脚/许可证：整行丢弃
        if FOOTER_RE.search(s):
            continue
        if len(CJK_RE.findall(s)) < 8:
            continue
        if DATE_LINE_RE.match(s):
            continue
        if CAPTION_RE.search(s) and len(s) < 45:
            continue
        lines.append(s)
    return "\n".join(lines)


def _compose(title: str, body: str, source: str) -> str:
    """把标题与正文拼成适合预训练的连续文本。

    不插入 "标题：" 之类的标记——预训练语料追求自然文本分布，
    标记会污染语言模型对行文结构的建模。用句号衔接标题与首段即可。
    """
    t = re.sub(r"\s*[-_|]\s*杭州网.*$", "", title or "").strip()
    t = re.sub(r"\s*[-_|]\s*今日生活.*$", "", t).strip()
    t = re.sub(r"\s*[-_|]\s*休闲旅游.*$", "", t).strip()

    parts = []
    if t and len(CJK_RE.findall(t)) >= 4:
        parts.append(t if t.endswith(("。", "！", "？")) else t + "。")
    parts.append(body)
    text = "\n".join(p for p in parts if p)
    # 折叠多余空白
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def load_raw(raw_dir: str = "data_gen/out/raw") -> List[Dict]:
    docs: List[Dict] = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("text"):
                    docs.append(d)
    return docs


def build(
    raw_dir: str = "data_gen/out/raw",
    out_path: str = "data_gen/out/corpus/corpus.jsonl",
    min_cjk: int = 150,
    max_cjk: int = 20000,
    dedup_threshold: int = 4,
) -> Dict[str, int]:
    docs = load_raw(raw_dir)
    stats = {"raw": len(docs), "exact_dup": 0, "near_dup": 0, "too_short": 0,
             "too_long": 0, "low_quality": 0, "written": 0}

    seen_hash: set = set()
    simhashes: List[int] = []
    kept: List[str] = []

    # 按 score 降序处理，去重时保留质量更高的一篇
    docs.sort(key=lambda d: d.get("score", 0), reverse=True)

    for d in docs:
        body = _normalize_body(clean_body(d.get("text", "")))
        text = _compose(d.get("title", ""), body, d.get("source", ""))

        cjk = len(CJK_RE.findall(text))
        if cjk < min_cjk:
            stats["too_short"] += 1
            continue
        if cjk > max_cjk:
            stats["too_long"] += 1
            continue

        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        if h in seen_hash:
            stats["exact_dup"] += 1
            continue

        sh = _simhash(text)
        if any(_hamming(sh, prev) <= dedup_threshold for prev in simhashes):
            stats["near_dup"] += 1
            continue

        if not quality_ok(text, min_cjk=min_cjk):
            stats["low_quality"] += 1
            continue

        seen_hash.add(h)
        simhashes.append(sh)
        kept.append(text)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for text in kept:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    stats["written"] = len(kept)

    total_cjk = sum(len(CJK_RE.findall(t)) for t in kept)
    stats["total_cjk"] = total_cjk
    stats["avg_cjk"] = int(total_cjk / max(len(kept), 1))
    return stats


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="构建预训练语料")
    ap.add_argument("--raw", default="data_gen/out/raw")
    ap.add_argument("--out", default="data_gen/out/corpus/corpus.jsonl")
    ap.add_argument("--min-cjk", type=int, default=150)
    args = ap.parse_args()

    s = build(raw_dir=args.raw, out_path=args.out, min_cjk=args.min_cjk)
    print("[build] 统计")
    for k, v in s.items():
        print(f"  {k:14s} {v}")
    print(f"\n输出: {args.out}")