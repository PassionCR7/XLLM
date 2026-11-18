"""回归测试：把实测踩过的坑固化成用例。

运行：python -m data_gen.test_data_gen
不依赖 pytest（本机装不上），直接断言。
"""

from __future__ import annotations

import os
import sys

from .crawl import clean_body, quality_ok, topic_score
from .extract import extract_links, html_to_text, page_title
from .posting import fact_ok, overlap_ratio
from .build_corpus import _normalize_body, _simhash, _hamming, rebuild_from_postings


def _check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  <- {detail}" if not cond and detail else ""))
    return cond


def test_extract_unclosed_form() -> bool:
    """坑 1：未闭合 <input> 会让整页正文被丢弃。

    实测来源：jrsh.hangzhou.com.cn 文章页。页面顶部搜索框写成未闭合的
    `<input type="text" name="q">`（没有 /> 也没有 </input>）。早期版本把
    input 当作「需配对的丢弃标签」压入 skip 栈，此后所有正文都被判为
    「仍在跳过中」，题干有 39 个 <p>，提取结果却是 0 字符。
    """
    html = """
    <html><body>
      <div class="search"><input type="text" name="q"></div>
      <div class="zhengwen">
        <p>杭州西溪湿地的芦花已经开了，周末人不多，适合带娃去看。</p>
        <p>秋雪庵是核心观赏点，坐摇橹船过去大概二十分钟，门票五十元。</p>
        <p>建议早上九点前到，光线好，人也少，拍照不用担心排队。</p>
      </div>
    </body></html>
    """
    text = html_to_text(html)
    ok = "秋雪庵" in text and "九点前" in text
    return _check("未闭合 input 不影响正文提取", ok, f"got={text[:60]!r}")


def test_extract_unclosed_form_tag() -> bool:
    """未闭合的 <form> 同样不能吞掉后续正文。

    form 是真正需要配对的丢弃标签，但遇到缺失 </form> 时必须能在
    合理边界恢复，否则整页丢失。
    """
    html = """
    <html><body><form action="/s"><input name="q">
      <div class="zhengwen">
        <p>杭州龙坞的共享厨房周末要提前一周预约，人均八十元左右。</p>
        <p>店主说来的年轻人更在意一起做饭聊天的氛围，不拼厨艺。</p>
      </div>
    </body></html>
    """
    text = html_to_text(html)
    ok = "龙坞" in text
    return _check("未闭合 form 仍能取到正文", ok, f"got={text[:60]!r}")


def test_extract_skips_nav() -> bool:
    """导航/页脚噪声应被密度打分压掉。"""
    html = """
    <html><body>
      <div class="nav"><a href="/">首页</a><a href="/a">新闻</a><a href="/b">财经</a></div>
      <div class="article">
        <p>杭州龙坞的这间共享厨房最近很火，周末要提前一周预约才能排上。</p>
        <p>店主说这里不拼厨艺，来的年轻人更在意一起做饭聊天的氛围。</p>
        <p>人均消费八十元左右，可以自带食材，也可以现场采购。</p>
      </div>
      <div class="footer">Copyright 杭州网 版权所有</div>
    </body></html>
    """
    text = html_to_text(html)
    ok = "龙坞" in text and "首页" not in text and "Copyright" not in text
    return _check("导航与页脚不混入正文", ok, f"got={text[:70]!r}")


def test_topic_reject_foreign_city() -> bool:
    """坑 2：外地游记仅因正文提到杭州而被误收。

    实测来源：新浪旅游「冬日温婉水墨扬州」「玩转赣州」。判据必须只看
    标题 + 导语开头，不能扫全文。
    """
    cases = [
        ("江南宋城，山美水秀，客家摇篮，玩转赣州！_新浪旅游_新浪网",
         "有朋友问，江西有什么好玩的地方。九江的庐山，鹰潭的龙虎山。", False),
        ("青山隐隐水迢迢 冬日温婉水墨扬州_新浪旅游_新浪网",
         "扬州瘦西湖，从杭州出发两小时高铁可达。", False),
    ]
    ok = True
    for title, text, expect in cases:
        s = topic_score(title, text)
        ok &= _check(f"外地游记被拒: {title[:18]}", (s >= 0.42) == expect, f"score={s:.2f}")
    return ok


def test_topic_accept_local() -> bool:
    """杭州本地游玩内容必须收，标题不带「杭州」二字也不能误杀。"""
    cases = [
        ("赏芦花绽放 赴“红楼”之约-杭州网休闲旅游",
         "风过西溪，芦花如雪。2026杭州西溪听芦主题活动已启幕。"),
        ("文博之旅 常看常新-杭州网休闲旅游",
         "“博物馆之城——杭州的博物馆临时展览推介案例展”开展。"),
        ("钱塘湿地 迎来越冬候鸟-杭州网休闲旅游",
         "杭州钱塘大湾区湿地公园，迎来了越冬候鸟大军。"),
        ("御街吹起怀旧风-杭州网休闲旅游",
         "11月11日，市民在杭州城市音乐客厅内参观黑胶唱片展陈。浙江省杭州市上城区"
         "与民间藏家合作打造的杭州城市音乐客厅开启试营业。该城市音乐客厅位于杭州"
         "南宋御街历史文化街区附近，馆藏黑胶唱片5000余张，集唱片展览、听音体验、"
         "公益讲座与小型演出于一体。"),
    ]
    ok = True
    for title, text in cases:
        s = topic_score(title, text)
        ok &= _check(f"本地游玩被收: {title[:16]}", s >= 0.42, f"score={s:.2f}")
    return ok


def test_topic_reject_local_news() -> bool:
    """本地新闻（有地名但无游玩线索）应拒绝。"""
    cases = [
        ("本月杭州二手房成交狂飙", "杭州二手房成交量环比上涨，房价出现松动。"),
        ("杭州一男子跷二郎腿左腿失灵", "久坐导致神经压迫，男子送急诊。"),
        ("杭州地铁3号线今日开通", "地铁开通运营，市民出行更便利。"),
    ]
    ok = True
    for title, text in cases:
        s = topic_score(title, text)
        ok &= _check(f"本地新闻被拒: {title[:16]}", s < 0.42, f"score={s:.2f}")
    return ok


def test_fact_check_catches_fabrication() -> bool:
    """坑 4：模型会编造价格，必须能抓出来。

    实测来源：LLM 改写满觉陇赏桂稿时，原文只写「沿路有茶室」，
    模型补出「人均八十元」。假价格进了预训练语料会被当知识学走。
    """
    origin = "满觉陇是杭州赏桂地，沿路有茶室，建议早上八点前到。"
    ok = True
    ok &= _check("编造价格被抓",
                 not fact_ok(origin, "茶室人均八十元，八点前到。")[0])
    ok &= _check("编造两个价格被抓",
                 not fact_ok(origin, "人均八十元，停车二十块。")[0])
    ok &= _check("保留原文数字放行",
                 fact_ok(origin, "沿路十几家茶室，八点前到。")[0])
    ok &= _check("数字都对放行",
                 fact_ok("活动到12月13日，24家单位参与，门票50元",
                         "办到12月13日，24家单位都来，门票50元")[0])
    return ok


def test_fact_check_no_false_positive() -> bool:
    """坑 5：事实核查误杀——「一」被当成数字。

    实测事故：首版把 个/家/人/位 也当单位，导致「一个人去」「三场大集」
    全被判编造数字，9/20 篇正常稿件被拒。修法：「一」单独出现一律不算，
    单位只保留金额/时长/距离。
    """
    origin = "杭州城市音乐客厅试营业，馆藏黑胶唱片5000余张。"
    rewritten = ("一个人去御街逛，正好碰上城市音乐客厅试营业，"
                 "5000多张黑胶唱片，能待一上午。")
    ok = _check("「一个人」不触发误杀", fact_ok(origin, rewritten)[0])
    ok &= _check("「一场」不触发误杀",
                 fact_ok("满觉陇桂花开了", "看了一场桂花雨，香得很。")[0])
    ok &= _check("「三场大集」不触发误杀",
                 fact_ok("农交会在杭州举办", "连逛三场大集，一次逛完。")[0])
    return ok


def test_overlap_detects_copy() -> bool:
    """坑 6：模型可能直接抄原文，失去改写意义。"""
    o = "杭州西溪湿地芦花开了，秋雪庵是核心观赏点，周末有游园会。"
    ok = _check("抄原文重合率高", overlap_ratio(o, o) > 0.9)
    ok &= _check("改写后重合率低",
                 overlap_ratio(o, "风一吹芦花像雪，我去看了，人不少。") < 0.2)
    return ok


def test_postings_dedup() -> bool:
    """坑 7：--force 重跑会产生同一原文的多个版本。

    实测：25 篇原文被改写成 39 条，语料里同一事件出现 2~3 次。
    rebuild_from_postings 要按 origin_title 合并。
    """
    import json as _json
    import tempfile

    rows = [
        {"origin_title": "A篇", "title": "打卡A-旧", "text": "第一版内容" * 8},
        {"origin_title": "A篇", "title": "打卡A-新", "text": "第二版内容" * 8},
        {"origin_title": "B篇", "title": "打卡B", "text": "另一个地方" * 8},
    ]
    with tempfile.TemporaryDirectory() as td:
        pp = os.path.join(td, "p.jsonl")
        out = os.path.join(td, "c.jsonl")
        with open(pp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(_json.dumps(r, ensure_ascii=False) + "\n")
        s = rebuild_from_postings(pp, out)
        n = sum(1 for line in open(out, encoding="utf-8") if line.strip())
        ok = _check("同原文只保留一条", s["uniq_origin"] == 2 and n == 2,
                    f"uniq={s['uniq_origin']} lines={n}")
        # 保留的是后写入的版本
        first = _json.loads(open(out, encoding="utf-8").readline())
        ok &= _check("保留最新版本", "打卡A-新" in first["text"],
                     f"got={first['text'][:20]}")
    return ok


def test_rebuild_keeps_originals() -> bool:
    """坑 8：rebuild 默认必须「补充」而不是「替换」。

    实测事故：首版 keep_original 默认 False，rebuild 之后语料从 21,828 字
    掉到 7,294 字——打卡帖把原文顶掉了，语料量缩水 2/3。用户要的是
    「补充 corpus.jsonl」，语料只应增加。打卡帖与通稿是两种文体，
    同时保留才能让模型学到叙述与体验两套表达。
    """
    import json as _json
    import tempfile

    posting = {"origin_title": "西溪芦花", "title": "打卡西溪",
               "text": "风过芦花像雪，我去看了。" * 8}
    original = {"text": "西溪听芦节启幕。\n" + "秋雪庵是核心观赏点。" * 10}
    with tempfile.TemporaryDirectory() as td:
        pp = os.path.join(td, "p.jsonl")
        out = os.path.join(td, "c.jsonl")
        with open(pp, "w", encoding="utf-8") as f:
            f.write(_json.dumps(posting, ensure_ascii=False) + "\n")
        with open(out, "w", encoding="utf-8") as f:
            f.write(_json.dumps(original, ensure_ascii=False) + "\n")
        s = rebuild_from_postings(pp, out)          # 用默认参数
        lines = [l for l in open(out, encoding="utf-8") if l.strip()]
        ok = _check("默认保留原文", len(lines) == 2, f"lines={len(lines)}")
        ok &= _check("原文确实还在",
                     any("秋雪庵是核心观赏点" in l for l in lines))
        ok &= _check("打卡帖也进来了",
                     any("风过芦花像雪" in l for l in lines))
    return ok


def test_no_key_leak() -> bool:
    """坑 11：API key 会经 repr / 错误信息泄露到终端。

    实测两处：
      1. LLMClient 是 dataclass，api_key 字段默认进 __repr__，
         print(client) 或 traceback 里的局部变量展示会打出
         api_key='sk-xxx'。只给 _cfg/_key 加 repr=False 不够，
         构造入参必须单独排除。
      2. chat() 把 HTTPError 响应体拼进 LLMError，而 cli.py 会直接
         print 到终端。网关出错时会回显请求头，key 就跟着出去了。
    终端会进 CI 日志、会被截图、会被贴进 issue，必须脱敏。
    """
    from .llm import LLMClient, redact

    secret = "sk-testsecret1234567890abcdef"
    c = LLMClient(api_base="https://x/v1", api_key=secret, model_id="m")
    ok = _check("repr 不含 key", secret not in repr(c), f"repr={c!r}")
    ok &= _check("str 不含 key", secret not in str(c))

    # 响应体回显 / Bearer 头回显都要打码
    leaked = f'HTTP 401 {{"error":"invalid key {secret}"}}'
    ok &= _check("错误响应体里的 key 被打码",
                 secret not in redact(leaked, secret), f"got={redact(leaked, secret)}")
    ok &= _check("Bearer 后的 key 被打码",
                 secret not in redact(f"Authorization: Bearer {secret}", secret))

    # 不能误伤：URL、普通错误、长数字串必须原样保留
    for s in ["https://api.example.com/v1/chat/completions",
              "HTTP 500 Internal Server Error",
              "订单号 20261119000123456789"]:
        ok &= _check(f"不误伤: {s[:22]}", redact(s, secret) == s, f"got={redact(s, secret)}")
    return ok


def test_scrub_idempotent() -> bool:
    """坑 10：脱敏必须幂等，重复跑不能把句子改坏。

    实测事故：先在已脱敏的语料上再 rebuild 一次，姓名池为空，
    「经营者顾大姐，客人都称她顾妈」被替换成「经营者他，客人都称她顾妈」——
    他/她混用，成了病句。原因是脱敏依赖姓名池，而姓名池只能从**原始**
    文本收集，从已脱敏文本收不到。规范流程是先 build(raw) 再 rebuild。
    """
    from .build_corpus import scrub_pii

    once = scrub_pii("满觉陇村党委副书记陈骏辰说，人很多。")
    twice = scrub_pii(once)
    ok = _check("二次脱敏不改动已脱敏文本", once == twice, f"{once!r} != {twice!r}")

    # 带姓名池时也不能产生「他他」「他，」这类残句
    pool = {"顾大姐"}
    s1 = scrub_pii("经营者顾大姐，客人都称她顾妈。", pool)
    ok &= _check("姓名后接标点时整段删除",
                 s1 == "经营者，客人都称她顾妈。", f"got={s1!r}")
    s2 = scrub_pii("叶顺虎是77届农校生。", {"叶顺虎"})
    ok &= _check("姓名后接动词时用「他」占位", s2 == "他是77届农校生。",
                 f"got={s2!r}")
    return ok


def test_pii_scrubbed() -> bool:
    """坑 9：语料里留着真实姓名 = 替别人扩散个人信息。

    新闻稿里「副书记陈骏辰说」本来是公开的，但喂进预训练语料是另一回事：
    模型会把「真实姓名 + 真实职务」记成知识，生成时凭空造一句真人言论。
    出口必须脱敏。同时手机号/身份证号绝不允许进语料。
    """
    from .build_corpus import scrub_pii

    ok = _check("职务+姓名被去掉",
                scrub_pii("满觉陇村党委副书记陈骏辰说，人很多。")
                == "满觉陇村党委副书记说，人很多。")
    ok &= _check("村书记+姓名被去掉",
                 "汪国建" not in scrub_pii("村书记汪国建从社区出发。"))
    ok &= _check("署名行整行删除",
                 "管光前" not in scrub_pii("记者 管光前 临安微融圈\n正文在此。"))
    ok &= _check("手机号被脱敏",
                 "13812345678" not in scrub_pii("电话13812345678"))
    ok &= _check("身份证号被脱敏",
                 "330102199001011234" not in scrub_pii("证号330102199001011234"))
    # 不能误伤：只给姓的匿名写法本来就是新闻惯例，注释和职务介绍也不能删
    ok &= _check("「陈先生」保留",
                 "陈先生" in scrub_pii("游客陈先生带着爱人来了。"))
    ok &= _check("「负责人介绍」不误伤",
                 scrub_pii("负责人介绍情况时说，活动到月底。")
                 == "负责人介绍情况时说，活动到月底。")
    return ok


def test_footer_stripped() -> bool:
    """坑 3：页脚 ICP 备案信息被当作正文收进来。

    实测来源：杭州网系文章末尾稳定出现「增值电信业务经营许可证：浙B2-...」。
    """
    raw = (
        "杭州西溪听芦节启幕，秋雪庵是核心观赏点，每逢周末有游园会。\n"
        "增值电信业务经营许可证：浙B2-20110366 信息网络传播视听节目许可证：1105105\n"
        "浙公网安备：33010002000058号\n"
        "Copyright 杭州网 版权所有"
    )
    out = _normalize_body(raw)
    ok = "浙B2" not in out and "Copyright" not in out and "西溪" in out
    return _check("页脚备案信息被清除", ok, f"got={out[:80]!r}")


def test_simhash_dedup() -> bool:
    """转载文章的近似重复应被识别。"""
    a = "杭州西溪湿地的芦花已经开了，周末人不多，适合带娃去看，秋雪庵是核心观赏点。"
    b = "杭州西溪湿地的芦花已经开了，周末人不多，适合带娃去看，秋雪庵是核心观赏点。"  # 完全相同
    c = "杭州西湖断桥边的月老花境限时展出，三米高的永生花装置吸引不少市民打卡拍照。"
    sh_a, sh_b, sh_c = _simhash(a), _simhash(b), _simhash(c)
    ok = _hamming(sh_a, sh_b) == 0 and _hamming(sh_a, sh_c) > 10
    return _check("SimHash 区分重复与不同文本", ok,
                  f"same={_hamming(sh_a, sh_b)} diff={_hamming(sh_a, sh_c)}")


def test_quality_gate() -> bool:
    """质量门槛：过短拒绝，正常长度通过。"""
    short = "杭州西湖很美。" * 3            # 18 字
    good = "杭州西溪湿地的芦花已经开了，周末人不多，适合带娃去看。" * 8  # 192 字
    ok = (not quality_ok(short, min_cjk=150)) and quality_ok(good, min_cjk=150)
    return _check("质量门槛按中文量生效", ok,
                  f"short={len(short)} good={len(good)}")


def test_extract_links() -> bool:
    html = '<a href="/a/1.html">文章一</a><a href="//x.com/b">文章二</a>'
    links = extract_links(html)
    ok = len(links) == 2 and links[0][0] == "/a/1.html" and links[1][1] == "文章二"
    return _check("链接提取与锚文本", ok, f"got={links}")


def test_title_clean() -> bool:
    html = "<html><head><title>赏芦花绽放 赴红楼之约-杭州网休闲旅游</title></head></html>"
    ok = page_title(html).startswith("赏芦花绽放")
    return _check("标题提取", ok)


def main() -> int:
    tests = [
        test_extract_unclosed_form,
        test_extract_unclosed_form_tag,
        test_extract_skips_nav,
        test_topic_reject_foreign_city,
        test_topic_accept_local,
        test_topic_reject_local_news,
        test_fact_check_catches_fabrication,
        test_fact_check_no_false_positive,
        test_overlap_detects_copy,
        test_postings_dedup,
        test_rebuild_keeps_originals,
        test_pii_scrubbed,
        test_no_key_leak,
        test_scrub_idempotent,
        test_footer_stripped,
        test_simhash_dedup,
        test_quality_gate,
        test_extract_links,
        test_title_clean,
    ]
    print("=" * 62)
    results = [t() for t in tests]
    passed = sum(results)
    print("=" * 62)
    print(f"{passed}/{len(results)} 组测试通过")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())