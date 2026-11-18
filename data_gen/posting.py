"""用 LLM 把新闻稿改写为「打卡帖」，并做语义筛选。

为什么不直接收原文
------------------
原文是通稿体：「11月11日，由西湖岳庙管理处打造的主题活动正式开启」。
这种文本对预训练有两个问题：
  1. 语料库里已有大量同类新闻体，边际价值低；
  2. 模型学到的是公文腔，而不是用户真正需要的「这里怎么去、多少钱、
     几点人少」这类实用表达。

打卡帖改写补的正是这一块：第一人称、口语、带具体细节。

三道质量闸门
------------
1. **usable=false 直接丢**：时政、楼市、事故类新闻无法改写，模型会拒绝。
2. **事实核查（防幻觉）**：模型偶尔会加"人均五十"这种原文没有的数字。
   这里做一次轻量校验：正文里出现的「阿拉伯数字/价格词」若在原文中
   找不到对应，判为疑似编造并丢弃。宁可少收，不能把假事实喂进预训练语料。
3. **长度与重复度**：过短无信息量，与原文过度重合说明没改写成功。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from .llm import LLMClient, LLMError

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
# 中文数字：模型极爱写「人均八十元」「步行二十分钟」这种形式，
# 只查阿拉伯数字会完全漏掉，等于没做核查（实测踩过这个坑）。
CN_NUM_CHARS = "零一二三四五六七八九十百千万两半几"
# 单位只保留「有事实含义且不易混淆」的：金额、时长、距离。
# 已刻意排除 个/家/人/位 —— 实测它们会把「一个人」「三个人吃饭」
# 「五个站点」这类模糊叙述当成事实数据，导致正常改写稿被大量误杀。
CN_NUM_RE = re.compile(
    rf"[{CN_NUM_CHARS}]{{1,6}}\s*(?:元|块|分钟|小时|公里|米|天|岁)"
)


def _is_fuzzy(core: str) -> bool:
    """判断中文数字短语是否属于不确定表达，不参与编造判定。

    「一」单独出现时几乎总是冠词（一个人、一条路、去了一次），
    不是数量事实，必须排除——实测这是误杀的主要来源，一度让
    9/20 的正常稿件被拒。
    「十几」「数十」「百余」「半个」同样是模糊量词。
    """
    if core in ("一", "半", "两"):
        return True
    if "几" in core or "数" in core:
        return True
    if core.endswith(("余", "多")):
        return True
    return False


def _numbers(text: str) -> set:
    """抽取文本里的数字线索：阿拉伯数字 + 确定性的中文数字短语。

    中文数字不做规范化（「八十」和「80」视为不同），因为这里判的是
    「原文有没有这个量级的信息」，不是精确比对。
    """
    text = text or ""
    out = set(NUM_RE.findall(text))
    for m in CN_NUM_RE.findall(text):
        core = re.match(rf"[{CN_NUM_CHARS}]+", m).group(0)
        if not _is_fuzzy(core):
            out.add(core)
    return out

SYSTEM_PROMPT = """你是杭州本地生活内容编辑。任务：把一篇新闻稿改写成一条真实的「周末打卡帖」。

规则：
1. 第一人称，像小红书/朋友圈那样写，口语化，不要播音腔。
2. 必须保留原文里的具体事实：地点、时间、价格、交通、注意事项。
   严禁编造原文没有的信息，尤其是数字（价格、距离、车程、分钟数）。
   原文没提价格就不要写价格。
3. 结构：开头一句最有吸引力的点，中间 3-6 句讲体验细节，结尾给一句实用建议。
4. 长度 180-400 字，不要分小标题，不要 emoji，不要话题标签，不要"打卡攻略"这类标题党套话。
5. 如果原文根本不是一个可以去的「地方/活动」（政策、事故、时政新闻、纯经济数据），
   把 usable 设为 false。
6. 不要写真实人名，也不要写记者的姓名或署名。原文里「副书记张三说」这类，
   改写时用「村干部说」「负责人说」代替，职务可以留，姓名一律不写。
   手机号、身份证号绝对不能出现在正文里。

只输出 JSON，不要任何解释：
{"usable": true或false, "reason": "一句话说明", "title": "打卡帖标题", "text": "打卡帖正文"}"""


@dataclass
class Posting:
    """一条打卡帖。"""
    source_url: str
    source: str
    title: str
    text: str
    origin_title: str
    llm_model: str
    created_at: str

    def as_corpus_text(self) -> str:
        """转成预训练用的 {"text": ...} 所需字符串。

        出口统一过一遍 scrub_pii：提示词已经要求模型不要写人名，但模型
        并不可靠（实测仍会照抄原文里的「副书记某某某」）。在出口兜一道，
        比在提示词里祈祷管用。
        """
        from .build_corpus import scrub_pii

        t = self.title.strip()
        if t and not t.endswith(("。", "！", "？", "!", "?")):
            t += "。"
        return scrub_pii(f"{t}\n{self.text.strip()}")


def fact_ok(origin: str, rewritten: str, max_missing: int = 0) -> Tuple[bool, str]:
    """轻量事实核查：改写稿里的数字线索必须能在原文找到。

    只查数字，不查语义——查语义得再调一次模型，成本翻倍。
    实测模型的主要幻觉形式就是"补一个价格/时长"，查数字能挡掉大部分。

    默认 max_missing=0（不容忍）。原文档写「三十元左右」、改写写
    「人均三十元」，两边都能抽到「三十」，正常情况不会误杀。
    编造价格这类假事实一旦进预训练语料，模型会当知识学走，
    这是不可逆的污染，代价远高于漏收几条。
    """
    o_nums = _numbers(origin)
    r_nums = _numbers(rewritten)
    missing = sorted(n for n in r_nums if n not in o_nums)
    if len(missing) > max_missing:
        return False, f"疑似编造数字: {missing[:4]}"
    return True, ""


def overlap_ratio(origin: str, rewritten: str, n: int = 12) -> float:
    """改写稿与原文字符 12-gram 的重合率。

    重合过高说明模型在抄原文而不是改写，那就失去了"打卡帖"的意义。
    """
    if not origin or not rewritten:
        return 0.0

    def grams(s: str) -> set:
        s = re.sub(r"\s+", "", s)
        return {s[i : i + n] for i in range(max(len(s) - n + 1, 0))}

    a, b = grams(origin), grams(rewritten)
    if not b:
        return 0.0
    return len(a & b) / len(b)


def rewrite_one(
    client: LLMClient,
    title: str,
    body: str,
    min_cjk: int = 150,
    max_overlap: float = 0.55,
) -> Tuple[Optional[dict], str]:
    """改写单篇。返回 (Posting 所需的字段 dict 或 None, 丢弃原因)。"""
    body = (body or "").strip()
    if len(CJK_RE.findall(body)) < min_cjk:
        return None, "too_short"

    prompt = f"标题：{title}\n\n正文：\n{body[:4000]}"
    obj = client.chat_json(SYSTEM_PROMPT, prompt, max_tokens=3000)
    if not obj:
        return None, "llm_no_json"

    if not obj.get("usable"):
        return None, "llm_reject"

    text = (obj.get("text") or "").strip()
    new_title = (obj.get("title") or "").strip()
    if len(CJK_RE.findall(text)) < 120:
        return None, "rewrite_too_short"

    ok, why = fact_ok(body, text + " " + new_title)
    if not ok:
        return None, why

    ov = overlap_ratio(body, text)
    if ov > max_overlap:
        return None, f"copy_original({ov:.2f})"

    return {
        "title": new_title or title,
        "text": text,
        "reason": obj.get("reason", ""),
        "overlap": round(ov, 3),
    }, ""


def rewrite_batch(
    client: LLMClient,
    docs: List[Dict],
    min_cjk: int = 150,
    out_stats: Optional[Dict[str, int]] = None,
    verbose: bool = True,
) -> List[Posting]:
    """批量改写。docs 为 RawDoc 字典列表（含 title/text/url/source）。"""
    from . import _now  # 延迟导入，避免循环

    stats = out_stats if out_stats is not None else {}
    postings: List[Posting] = []

    for i, d in enumerate(docs, 1):
        title = d.get("title", "")
        body = d.get("text", "")
        try:
            res, why = rewrite_one(client, title, body, min_cjk=min_cjk)
        except LLMError as e:
            why, res = f"llm_error({str(e)[:40]})", None
        except Exception as e:
            why, res = f"error({type(e).__name__})", None

        if not res:
            key = why.split("(")[0]
            stats[key] = stats.get(key, 0) + 1
            if verbose:
                print(f"  [{i}] DROP {why:22s} {title[:34]}")
            continue

        stats["accepted"] = stats.get("accepted", 0) + 1
        postings.append(
            Posting(
                source_url=d.get("url", ""),
                source=d.get("source", ""),
                title=res["title"],
                text=res["text"],
                origin_title=title,
                llm_model=client.model_id or "",
                created_at=_now(),
            )
        )
        if verbose:
            print(f"  [{i}] OK   overlap={res['overlap']:.2f} {res['title'][:34]}")

    return postings