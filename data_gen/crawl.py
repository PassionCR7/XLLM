"""采集编排：发现 -> 抓取 -> 提取 -> 主题过滤 -> 落盘。

设计要点：
1. 断点续爬：把已处理的 URL 记到 state 文件，中断后重跑不会重复抓。
2. 原始层与语料层分离：out/raw/ 存标题+正文+元数据的 JSONL（可复查来源），
   out/corpus/ 存最终 {"text": ...}（直接喂给 dataset/lm_dataset.py）。
3. 主题过滤在正文层做，而不是只按标题过滤——标题党页面的正文往往不相关。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional

from . import extract
from .fetch import HttpClient, normalize_url
from .sources import (
    CITY_TERMS,
    DEFAULT_SOURCES,
    NEGATIVE_TERMS,
    PLAY_TERMS,
    SOURCES,
    WEEKEND_TERMS,
)


# ---------------------------------------------------------------- 主题打分
def topic_score(title: str, text: str) -> float:
    """返回 0~1 的主题相关度。

    判据只用「标题 + 导语开头 80 字」，不扫正文其余部分。
    原因（实测）：正文顺带提到杭州的文章很多。曾收录到
    「冬日温婉水墨扬州」「玩转赣州」这类外地游记，仅因正文某处出现
    「杭州」。而本地游玩内容的导语开头必定交代地点，所以把窗口收到
    前 80 字即可分开，且不会误杀「文博之旅」这类标题不带地名的稿件。

    必须项（本地性）：标题或导语前 80 字出现杭州地名。
    意图项（主题性）：同一窗口内出现游玩意图词，或具体游玩对象词。
                     二者缺一即拒绝；泛词不能顶替。
    """
    title = title or ""
    t_title = re.sub(
        r"\s*[-_|]\s*(杭州网|今日生活|休闲旅游|新浪旅游|新浪网).*$", "", title
    )
    lede = (text or "")[:80]
    win = t_title + " " + lede
    full = title + " " + (text or "")[:1200]

    def hits(terms, hay) -> int:
        return sum(1 for w in terms if w in hay)

    # --- 必须项：本地性 ---
    if hits(CITY_TERMS, win) == 0:
        return 0.0

    # --- 意图项：主题性（只看同一窗口）---
    week_n = hits(WEEKEND_TERMS, win)
    play_n = hits(PLAY_TERMS, win)
    if week_n == 0 and play_n == 0:
        # 窗口内只有地名、没有任何游玩线索 -> 判为本地新闻，不是游玩内容
        return 0.20

    neg_n = hits(NEGATIVE_TERMS, t_title) + 0.5 * hits(NEGATIVE_TERMS, full)

    score = 0.55
    score += 0.10 * min(week_n, 2)
    score += 0.07 * min(play_n, 3)
    score += 0.05 * min(hits(WEEKEND_TERMS, full), 3)
    score += 0.04 * min(hits(PLAY_TERMS, full), 4)
    score -= 0.22 * min(neg_n, 3)
    return max(0.0, min(score, 1.0))


# ---------------------------------------------------------------- 语料清洗
def clean_body(text: str) -> str:
    """去掉正文里的常见噪声行（编辑署名、版权、往期推荐等）。"""
    drop_patterns = (
        "责任编辑", "编辑：", "来源：", "原标题", "版权声明", "未经授权",
        "扫码", "二维码", "点击查看", "更多精彩", "往期回顾", "相关阅读",
        "声明：", "本文仅代表", "免责声明", "广告",
    )
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if len(s) < 10:
            continue
        if any(p in s for p in drop_patterns) and len(s) < 60:
            continue
        out.append(s)
    return "\n".join(out)


def quality_ok(text: str, min_cjk: int = 150, max_dup_ratio: float = 0.55) -> bool:
    """基础质量门槛：中文量足够，且重复行不过多。"""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    if cjk < min_cjk:
        return False
    lines = [ln for ln in text.split("\n") if ln]
    if len(lines) >= 6:
        uniq = len(set(lines))
        if 1 - uniq / len(lines) > max_dup_ratio:
            return False
    return True


@dataclass
class RawDoc:
    url: str
    source: str
    title: str
    text: str
    score: float
    fetched_at: str


class Crawler:
    def __init__(
        self,
        out_dir: str = "data_gen/out",
        client: Optional[HttpClient] = None,
        min_score: float = 0.42,
        min_cjk: int = 150,
        max_docs_per_source: int = 60,
    ):
        self.out_dir = out_dir
        self.raw_dir = os.path.join(out_dir, "raw")
        self.state_path = os.path.join(out_dir, "state.json")
        os.makedirs(self.raw_dir, exist_ok=True)

        self.client = client or HttpClient(verify_ssl=False, min_interval=1.2)
        self.min_score = min_score
        self.min_cjk = min_cjk
        self.max_docs_per_source = max_docs_per_source

        self.visited: set = set()
        self._load_state()
        self.stats: Dict[str, Dict[str, int]] = {}

    # ---------------- state ----------------
    def _load_state(self) -> None:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.visited = set(data.get("visited", []))
            except Exception:
                self.visited = set()

    def save_state(self) -> None:
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump({"visited": sorted(self.visited)}, f, ensure_ascii=False, indent=1)

    # ---------------- 发现 ----------------
    def discover(self, source) -> List[str]:
        found: List[str] = []
        seen = set()
        for page_url in source.page_urls():
            res = self.client.get(page_url)
            if not res.ok:
                self._bump(source.name, "list_fail")
                continue
            for href, _anchor in extract.extract_links(res.html):
                u = normalize_url(href, page_url)
                if not u or not source.is_article(u):
                    continue
                # 去 query、去 fragment，避免同一文章多种 URL
                u = u.split("#")[0].split("?")[0]
                if u in seen or u in self.visited:
                    continue
                seen.add(u)
                found.append(u)
        return found

    # ---------------- 单篇处理 ----------------
    def process(self, url: str, source_name: str) -> Optional[RawDoc]:
        res = self.client.get(url)
        self.visited.add(url)
        if not res.ok:
            self._bump(source_name, "fetch_fail")
            return None

        title = extract.page_title(res.html)
        body = clean_body(extract.html_to_text(res.html))
        if not quality_ok(body, min_cjk=self.min_cjk):
            self._bump(source_name, "quality_reject")
            return None

        score = topic_score(title, body)
        if score < self.min_score:
            self._bump(source_name, "offtopic")
            return None

        self._bump(source_name, "accepted")
        return RawDoc(
            url=url,
            source=source_name,
            title=title,
            text=body,
            score=round(score, 3),
            fetched_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def _bump(self, name: str, key: str) -> None:
        d = self.stats.setdefault(name, {})
        d[key] = d.get(key, 0) + 1

    # ---------------- 主流程 ----------------
    def run(self, sources=None, sleep_between: float = 0.8) -> Dict[str, int]:
        if sources is None:
            names = set(DEFAULT_SOURCES)
            sources = [s for s in SOURCES if s.name in names]
        total_written = 0
        for src in sources:
            print(f"\n[source] {src.name}  {src.note}")
            urls = self.discover(src)
            print(f"  discovered {len(urls)} article links")
            out_path = os.path.join(self.raw_dir, f"{src.name}.jsonl")
            written = 0
            with open(out_path, "a", encoding="utf-8") as f:
                for i, u in enumerate(urls[: self.max_docs_per_source], 1):
                    doc = self.process(u, src.name)
                    if doc:
                        f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
                        f.flush()
                        written += 1
                        print(f"  [{i}] OK score={doc.score} cjk~{len(doc.text)} {doc.title[:32]}")
                    else:
                        print(f"  [{i}] skip {u[:70]}")
                    time.sleep(sleep_between)
            self.save_state()
            total_written += written
            print(f"  -> wrote {written} docs to {out_path}")
        print("\n[stats]")
        for k, v in self.stats.items():
            print(f"  {k}: {v}")
        return {"written": total_written}