"""data_gen 命令行入口。

用法：
    # 采集全部默认源
    python -m data_gen.cli crawl

    # 只采指定源
    python -m data_gen.cli crawl --source hangzhou_travel --source jrsh_food

    # 调整每源上限与请求间隔
    python -m data_gen.cli crawl --max-docs 40 --interval 1.0

    # 列出可用源
    python -m data_gen.cli list

    # 把已采集的原始数据构建成预训练语料
    python -m data_gen.cli build

    # 一步到位：采集 + 构建
    python -m data_gen.cli all

    # 查看语料统计
    python -m data_gen.cli stats
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from dataclasses import asdict

from . import build_corpus
from .crawl import Crawler
from .fetch import HttpClient
from .sources import DEFAULT_SOURCES, SOURCES

CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def cmd_list(_args) -> int:
    print(f"{'源名称':<20}{'默认启用':<10}{'说明'}")
    print("-" * 78)
    for s in SOURCES:
        on = "是" if s.name in DEFAULT_SOURCES else "否"
        print(f"{s.name:<20}{on:<10}{s.note}")
    return 0


def cmd_crawl(args) -> int:
    if args.source:
        names = set(args.source)
        unknown = names - {s.name for s in SOURCES}
        if unknown:
            print(f"未知源: {', '.join(sorted(unknown))}", file=sys.stderr)
            print("可用源见: python -m data_gen.cli list", file=sys.stderr)
            return 2
        sources = [s for s in SOURCES if s.name in names]
    else:
        names = set(DEFAULT_SOURCES)
        sources = [s for s in SOURCES if s.name in names]

    client = HttpClient(
        timeout=args.timeout,
        retries=args.retries,
        min_interval=args.interval,
        verify_ssl=not args.insecure,
        respect_robots=not args.ignore_robots,
    )
    crawler = Crawler(
        out_dir=args.out,
        client=client,
        min_score=args.min_score,
        min_cjk=args.min_cjk,
        max_docs_per_source=args.max_docs,
    )
    crawler.run(sources=sources, sleep_between=args.sleep)
    return 0


def cmd_build(args) -> int:
    stats = build_corpus.build(
        raw_dir=os.path.join(args.out, "raw"),
        out_path=os.path.join(args.out, "corpus", "corpus.jsonl"),
        min_cjk=args.min_cjk,
    )
    print("[build] 统计")
    for k, v in stats.items():
        print(f"  {k:14s} {v}")
    return 0


def cmd_all(args) -> int:
    rc = cmd_crawl(args)
    if rc != 0:
        return rc
    return cmd_build(args)


def cmd_post(args) -> int:
    """用 LLM 把已抓取的原文改写成打卡帖，并合并进语料。"""
    from .build_corpus import load_raw, merge_append
    from .llm import LLMClient, LLMError
    from .posting import rewrite_batch

    raw_dir = os.path.join(args.out, "raw")
    docs = load_raw(raw_dir)
    if not docs:
        print(f"没有原始数据: {raw_dir}\n请先运行: python -m data_gen.cli crawl",
              file=sys.stderr)
        return 1

    # 只处理主题相关度达标的，省 LLM 调用
    docs = [d for d in docs if d.get("score", 0) >= args.min_score]
    docs.sort(key=lambda d: d.get("score", 0), reverse=True)

    # 断点续跑：跳过已改写过的 url
    done_path = os.path.join(args.out, "postings", "done_urls.json")
    done: set = set()
    if os.path.exists(done_path) and not args.force:
        try:
            done = set(json.load(open(done_path, encoding="utf-8")))
        except Exception:
            done = set()
    before = len(docs)
    docs = [d for d in docs if d.get("url") not in done]
    if before != len(docs):
        print(f"跳过已改写 {before - len(docs)} 篇，待处理 {len(docs)} 篇")
    else:
        print(f"待改写 {len(docs)} 篇")

    docs = docs[: args.limit]
    if not docs:
        print("没有需要处理的文档（全部已改写）。用 --force 可重跑。")
        return 0

    try:
        client = LLMClient(model_id=args.model, max_tokens=args.max_tokens)
    except LLMError as e:
        print(f"LLM 初始化失败: {e}", file=sys.stderr)
        return 2
    print(f"模型: {client.model_id}")

    stats: dict = {}
    postings = rewrite_batch(client, docs, min_cjk=args.min_cjk, out_stats=stats)

    # 落盘：明细 + 语料文本
    pdir = os.path.join(args.out, "postings")
    os.makedirs(pdir, exist_ok=True)
    detail_path = os.path.join(pdir, "postings.jsonl")
    with open(detail_path, "a", encoding="utf-8") as f:
        for p in postings:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    # 记录已处理 url（含被拒的，避免重复浪费调用）
    done |= {d.get("url") for d in docs if d.get("url")}
    json.dump(sorted(done), open(done_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    texts = [p.as_corpus_text() for p in postings]
    corpus = os.path.join(args.out, "corpus", "corpus.jsonl")
    ms = merge_append(texts, corpus, min_cjk=args.min_cjk)

    print("\n[rewrite] 统计")
    for k, v in stats.items():
        print(f"  {k:16s} {v}")
    print("\n[llm] 调用")
    for k, v in client.stats.items():
        print(f"  {k:16s} {v}")
    print("\n[merge] 语料")
    for k, v in ms.items():
        print(f"  {k:16s} {v}")
    print(f"\n明细: {detail_path}")
    print(f"语料: {corpus}")
    return 0


def cmd_rebuild(args) -> int:
    """从 raw 原文 + 打卡帖明细重建语料。

    必须先从 raw 重新 build，不能直接拿现有 corpus.jsonl 当输入——
    现有语料可能已经脱敏过，再脱敏一遍会二次污染（实测出现「经营者他」
    这种病句，以及姓名池为空导致「顾大姐」清不掉）。
    """
    from .build_corpus import build, rebuild_from_postings

    pp = os.path.join(args.out, "postings", "postings.jsonl")
    if not os.path.exists(pp):
        print(f"没有打卡帖明细: {pp}\n请先运行: python -m data_gen.cli post",
              file=sys.stderr)
        return 1
    corpus = os.path.join(args.out, "corpus", "corpus.jsonl")

    # 第一步：raw -> 语料（干净底稿，同时完成脱敏）
    b = build(raw_dir=os.path.join(args.out, "raw"), out_path=corpus)
    print("[rebuild] 1/2 从 raw 重建")
    for k in ("raw", "written", "total_cjk"):
        print(f"  {k:16s} {b.get(k, 0)}")

    # 第二步：合并打卡帖（keep_original 保持 True，否则第一步白做）
    s = rebuild_from_postings(
        pp, corpus,
        keep_original=not args.drop_originals,
        raw_dir=os.path.join(args.out, "raw"),
    )
    print("[rebuild] 2/2 合并打卡帖")
    for k, v in s.items():
        print(f"  {k:16s} {v}")
    print(f"\n语料: {corpus}")
    return 0


def cmd_stats(args) -> int:
    corpus = os.path.join(args.out, "corpus", "corpus.jsonl")
    if not os.path.exists(corpus):
        print(f"语料不存在: {corpus}，请先运行 build", file=sys.stderr)
        return 1
    n = 0
    total = 0
    lens = []
    with open(corpus, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            c = len(CJK_RE.findall(d.get("text", "")))
            n += 1
            total += c
            lens.append(c)
    lens.sort()
    print(f"文档数     : {n}")
    print(f"中文总字数 : {total:,}")
    if lens:
        print(f"平均长度   : {total // n} 字")
        print(f"最短/中位/最长: {lens[0]} / {lens[n // 2]} / {lens[-1]} 字")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m data_gen.cli",
        description="杭州周末游玩主题语料采集与构建",
    )
    ap.add_argument("--out", default="data_gen/out", help="输出目录，默认 data_gen/out")

    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="列出所有可用源")
    p.set_defaults(func=cmd_list)

    def add_crawl_opts(p):
        p.add_argument("--source", action="append", help="指定源名称，可重复")
        p.add_argument("--max-docs", type=int, default=30, help="每个源最多抓取篇数")
        p.add_argument("--interval", type=float, default=1.2, help="同域名请求最小间隔(秒)")
        p.add_argument("--sleep", type=float, default=0.4, help="篇间休眠(秒)")
        p.add_argument("--timeout", type=float, default=15.0, help="单请求超时(秒)")
        p.add_argument("--retries", type=int, default=3, help="失败重试次数")
        p.add_argument("--min-score", type=float, default=0.42, help="主题相关度阈值")
        p.add_argument("--min-cjk", type=int, default=150, help="正文最少中文字数")
        p.add_argument("--insecure", action="store_true", help="跳过 SSL 证书校验")
        p.add_argument("--ignore-robots", action="store_true", help="忽略 robots.txt（不建议）")

    p = sub.add_parser("crawl", help="采集网页数据")
    add_crawl_opts(p)
    p.set_defaults(func=cmd_crawl)

    p = sub.add_parser("build", help="构建预训练语料")
    p.add_argument("--min-cjk", type=int, default=150)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("all", help="采集 + 构建")
    add_crawl_opts(p)
    p.set_defaults(func=cmd_all)

    p = sub.add_parser("post", help="用 LLM 把原文改写成打卡帖并并入语料")
    p.add_argument("--limit", type=int, default=20, help="本次最多处理篇数")
    p.add_argument("--min-score", type=float, default=0.42, help="只处理该分数以上的原文")
    p.add_argument("--min-cjk", type=int, default=150, help="原文最少中文字数")
    p.add_argument("--model", default=None, help="模型 id，默认取配置里第一个")
    p.add_argument("--max-tokens", type=int, default=3000, help="单次生成 token 上限")
    p.add_argument("--force", action="store_true", help="忽略断点记录，全部重跑")
    p.set_defaults(func=cmd_post)

    p = sub.add_parser("rebuild", help="从打卡帖明细重建语料并合并原文")
    p.add_argument("--drop-originals", action="store_true",
                   help="删除原文新闻稿，语料只留打卡帖（默认保留，原文是另一种文体）")
    p.set_defaults(func=cmd_rebuild)

    p = sub.add_parser("stats", help="查看语料统计")
    p.set_defaults(func=cmd_stats)

    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())