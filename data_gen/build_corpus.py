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

# ---------------------------------------------------------------------------
# 个人信息脱敏
# ---------------------------------------------------------------------------
# 语料来自公开新闻稿，人名本来就登在报上。但把它喂进预训练语料是另一回事：
# 模型会把这些“真实姓名 + 真实职务”记成可复现的知识，生成时凭空输出一个
# 真人名字去配某句话，等于替别人制造言论。所以入库前统一去标识。
#
# 只处理两类，其余不动：
#   1. 记者/通讯员/作者署名 —— 对语言建模零价值，整行删除
#   2. 职务 + 全名 —— 直接删掉姓名，只留职务：「副书记陈骏辰说」->「副书记说」
#      比替换成「副书记某说」通顺，也更符合新闻匿名惯例
# 「陈先生」「李女士」这种只给姓的写法本来就是新闻界的匿名惯例，不算可识别
# 信息，保留原样。
SIGNATURE_RE = re.compile(
    r"^\s*(作者|记者|通讯员|摄影|编辑|来源)\s*[：:].{0,60}$"
    r"|^\s*(记者|通讯员)\s+[\u4e00-\u9fff]{2,4}(\s|　|$)"
)
ROLE_NAME_RE = re.compile(
    r"([\u4e00-\u9fff]{0,6}?(?:副)?(?:书记|主任|经理|站长|馆长|董事长|总经理|"
    r"村长|院长|校长|队长|组长|所长|店长|负责人))([\u4e00-\u9fff]{2,3})"
    r"(?=(?:说|告诉|介绍|表示|回忆|感叹|坦言|透露|笑|称|解释道|向|是|今年|已|也|都|在|从))"
)
# 真实手机号（11 位、1 开头）与 18 位身份证号
PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
IDCARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")

# 试过但放弃了：用「百家姓 + 2~3 字 + 是/说」抓没有职务前缀的裸姓名。
# 实测误伤率太高——「服务人员是」「武林广场是」「茶山是龙井」「花园里种了
# 桂花」全被命中，语料会被改成残句。宁可漏掉零星几个名字，也不能把正常
# 中文句子打烂。漏网的靠人工核对后加进 MANUAL_NAMES。

# 人工核对过的裸姓名（正则抓不到的）。这里是兜底，不是主路径——
# 每轮采集后跑一遍扫描脚本，把新发现的补进来。
# 注意「顾妈」不在此列：那是店铺招牌（顾妈手工馒头），属于商户品牌不是人名。
MANUAL_NAMES = {
    # name -> 替换文本。空串表示直接删掉（后面跟逗号分句的情况），
    # "他" 表示需要占位的场合。
    "叶顺虎": "他",      # 后陡门「十个勤天」稿件，无职务前缀
    "顾大姐": "",        # 「经营者顾大姐，客人都称她顾妈」-> 「经营者，客人都称她顾妈」
    # 说明：rebuild 若在**已脱敏**的语料上重复执行，会出现二次污染
    # （「经营者他」这种病句）。规范做法是先 build() 从 raw 重建，
    # 再 rebuild_from_postings() 合并打卡帖。
}


def collect_names(texts: List[str]) -> set:
    """从原文里收集「职务+姓名」模式中出现过的真实姓名。

    为什么要先收集再替换：LLM 改写打卡帖时经常把职务删掉只留名字
    （原文「公园负责人庄寻说」-> 改写「庄寻说」），此时单文本正则已经
    没有职务可抓。先用整批原文建一个姓名池，再全局替换，才盖得住。
    """
    names = set()
    for t in texts:
        for m in ROLE_NAME_RE.finditer(t or ""):
            nm = m.group(2)
            if len(nm) >= 2:
                names.add(nm)
    return names


def scrub_pii(text: str, names: Optional[set] = None) -> str:
    """去掉署名与「职务+全名」，脱敏手机号/身份证号。

    注意 ROLE_NAME 只在名字后面紧跟「说/告诉/介绍…」时才动手，避免把
    「负责人介绍情况」里的「介绍」误当人名——实测这个误伤很容易发生。

    names 传入姓名池时，池中姓名一律替换为「他」。这会让少数女性指代失真
    （「她是90后姑娘」变「他是90后姑娘」），但相比把真人姓名喂给模型，
    这个代价可以接受；而且职务+姓名的主路径会先把姓名整段删掉，
    真正走到池替换的只是 LLM 改写留下的残渣。
    """
    if not text:
        return text
    out_lines: List[str] = []
    for line in text.split("\n"):
        if SIGNATURE_RE.search(line):
            continue
        out_lines.append(line)
    s = "\n".join(out_lines)
    s = ROLE_NAME_RE.sub(r"\1", s)
    # 人工黑名单：与姓名池合并处理
    for nm, repl in MANUAL_NAMES.items():
        if nm in s:
            s = s.replace(nm, repl)
    if names:
        # 长名字先替换，避免「李强」把「李强华」切坏。
        # 替换文本按后接字符定：后面是「，/。/、」这类断句标点就整段删掉
        # （「经营者李四，客人都称她某姐」-> 「经营者，客人都称她某姐」），
        # 否则用「他」占位保句子通顺（「李四说是77届」-> 「他说是77届」）。
        for nm in sorted(names, key=len, reverse=True):
            if not nm:
                continue
            while nm in s:
                i = s.find(nm)
                j = i + len(nm)
                nxt = s[j:j + 1]
                repl = "" if nxt in ("，", "。", "、", "；", "：", "！", "？") else "他"
                # 「他说」会变「他他」，防重
                if repl == "他" and s[j:j + 1] == "他":
                    repl = ""
                s = s[:i] + repl + s[j:]
    s = PHONE_RE.sub("【电话已脱敏】", s)
    s = IDCARD_RE.sub("【证件号已脱敏】", s)
    return s


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

    # 先扫全批原文建姓名池，再逐篇脱敏（LLM 改写常把职务删掉只留名字）
    name_pool = collect_names([d.get("text", "") for d in docs])

    for d in docs:
        body = _normalize_body(clean_body(d.get("text", "")))
        text = _compose(d.get("title", ""), body, d.get("source", ""))
        text = scrub_pii(text, name_pool)

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


def merge_append(
    new_texts: List[str],
    out_path: str = "data_gen/out/corpus/corpus.jsonl",
    dedup_threshold: int = 4,
    min_cjk: int = 120,
) -> Dict[str, int]:
    """把新文本合并进已有语料，全局去重。

    为什么不直接 open(..., 'a')：追加会和已有语料重复。同一篇满觉陇的文章
    既可能来自新闻稿构建，也可能来自 LLM 改写，内容会高度相似。
    因此读取全量 -> 统一去重 -> 整体重写。

    out_path 不存在时等价于新建。
    """
    stats = {"existing": 0, "incoming": len(new_texts), "exact_dup": 0,
             "near_dup": 0, "too_short": 0, "added": 0, "total": 0}

    existing: List[str] = []
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue
                if t:
                    existing.append(t)
    stats["existing"] = len(existing)

    seen: set = set()
    sims: List[int] = []
    merged: List[str] = []

    for t in existing:
        h = hashlib.md5(t.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        sims.append(_simhash(t))
        merged.append(t)

    for t in new_texts:
        t = (t or "").strip()
        if len(CJK_RE.findall(t)) < min_cjk:
            stats["too_short"] += 1
            continue
        h = hashlib.md5(t.encode("utf-8")).hexdigest()
        if h in seen:
            stats["exact_dup"] += 1
            continue
        sh = _simhash(t)
        if any(_hamming(sh, prev) <= dedup_threshold for prev in sims):
            stats["near_dup"] += 1
            continue
        seen.add(h)
        sims.append(sh)
        merged.append(t)
        stats["added"] += 1

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for t in merged:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    stats["total"] = len(merged)
    return stats


def remove_superseded(
    postings_path: str,
    out_path: str = "data_gen/out/corpus/corpus.jsonl",
) -> Dict[str, int]:
    """移除已被打卡帖取代的原始新闻稿，避免同主题重复。

    背景：一条内容会同时存在两种形态——原始通稿（"11月11日，由西湖岳庙
    管理处打造的活动正式开启"）和 LLM 改写后的打卡帖（"路过就拍了两张"）。
    两者讲同一件事，同时留在语料里等于给模型喂重复样本，还会拉低
    有效信息密度。这里保留打卡帖、删掉对应的原文。

    判定方式：用 origin_title 定位原文（语料里原文的第一行就是标题），
    再做 SimHash 兜底，防止标题被清洗过导致字面不完全一致。
    """
    from .posting import Posting  # 局部导入避免循环

    stats = {"corpus_before": 0, "postings": 0, "removed": 0, "corpus_after": 0}
    if not os.path.exists(out_path):
        return stats

    origin_titles: List[str] = []
    if os.path.exists(postings_path):
        with open(postings_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ot = d.get("origin_title") or ""
                if ot:
                    origin_titles.append(ot.strip())
    stats["postings"] = len(origin_titles)

    corpus: List[str] = []
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    corpus.append(json.loads(line).get("text", ""))
                except json.JSONDecodeError:
                    pass
    stats["corpus_before"] = len(corpus)

    def norm(s: str) -> str:
        # 去掉站点后缀与标点，只留中文和字母，便于比对标题
        s = re.sub(r"[-_|]\s*(杭州网|今日生活|休闲旅游|新浪旅游|新浪网).*$", "", s)
        return re.sub(r"[^\u4e00-\u9fffa-zA-Z0-9]", "", s)

    targets = {norm(t) for t in origin_titles if norm(t)}
    # 只保留足够长的目标标题，避免短标题误删
    targets = {t for t in targets if len(t) >= 8}

    kept: List[str] = []
    for text in corpus:
        first_line = text.split("\n")[0]
        n = norm(first_line)
        hit = any(t and (t in n or n.startswith(t[:12])) for t in targets)
        if hit:
            stats["removed"] += 1
            continue
        kept.append(text)

    with open(out_path, "w", encoding="utf-8") as f:
        for t in kept:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    stats["corpus_after"] = len(kept)
    return stats


def rebuild_from_postings(
    postings_path: str,
    out_path: str = "data_gen/out/corpus/corpus.jsonl",
    keep_original: bool = True,
    raw_dir: Optional[str] = "data_gen/out/raw",
) -> Dict[str, int]:
    """从打卡帖明细重建语料，并把原文新闻稿合并回来。

    做三件事：
      1. 按 origin_title 去重：同一篇原文可能被改写多次（--force 重跑、
         或多轮采集），只保留**最新**的一个版本。
      2. 合并原文新闻稿（keep_original=True，默认）：打卡帖和通稿是两种
         不同文体，同时保留能让模型学到「客观叙述」与「第一人称体验」两套
         表达。只有显式 drop_originals 时才让打卡帖取代原文。
      3. 输出前做 SimHash 兜底去重，防止不同原文改写出雷同内容。

    注意：早期版本默认 keep_original=False，会把 21,828 字的原文语料整个
    换成 7,294 字的打卡帖——那是「替换」不是「补充」，语料量掉了 2/3。
    现在的默认值保证只增不减。
    """
    stats = {"postings": 0, "uniq_origin": 0, "dup_origin": 0,
             "near_dup": 0, "kept": 0, "removed_origin": 0}

    if not os.path.exists(postings_path):
        return stats

    rows: List[Dict] = []
    with open(postings_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    stats["postings"] = len(rows)

    # 同一 origin_title 保留最后一条（后写入的更新）
    latest: Dict[str, Dict] = {}
    for r in rows:
        key = (r.get("origin_title") or r.get("source_url") or "").strip()
        latest[key] = r
    stats["uniq_origin"] = len(latest)
    stats["dup_origin"] = len(rows) - len(latest)

    texts: List[str] = []
    seen_sim: List[int] = []
    # 姓名池从 postings 全部文本里收（含 origin_title，原文标题常带职务+姓名）
    name_pool = collect_names([f"{r.get('origin_title','')} {r.get('text','')}"
                               for r in rows])
    # postings 里通常一个都收不到——LLM 改写时就把职务删了。真正的人名在
    # raw 原文里，必须回读，否则「庄寻说」这类残渣永远清不掉。
    if raw_dir and os.path.isdir(raw_dir):
        name_pool |= collect_names([d.get("text", "") for d in load_raw(raw_dir)])
    for r in latest.values():
        title = (r.get("title") or "").strip()
        body = (r.get("text") or "").strip()
        if not body:
            continue
        if title and not title.endswith(("。", "！", "？", "!", "?")):
            title += "。"
        text = f"{title}\n{body}" if title else body
        text = scrub_pii(text, name_pool)
        sh = _simhash(text)
        if any(_hamming(sh, p) <= 3 for p in seen_sim):
            stats["near_dup"] += 1
            continue
        seen_sim.append(sh)
        texts.append(text)

    if keep_original and os.path.exists(out_path):
        # 保留已有原文：把不在打卡帖里的旧条目追加回来
        post_firsts = {t.split("\n")[0] for t in texts}
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    old = json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue
                if old and old.split("\n")[0] not in post_firsts:
                    texts.append(scrub_pii(old, name_pool))
                    post_firsts.add(old.split("\n")[0])
    elif os.path.exists(out_path):
        # 统计被移除的原文数量，便于核对
        with open(out_path, "r", encoding="utf-8") as f:
            before = sum(1 for line in f if line.strip())
        stats["removed_origin"] = max(before - stats["postings"], 0)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    stats["kept"] = len(texts)
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