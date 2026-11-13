"""回归测试：把实测踩过的坑固化成用例。

运行：python -m data_gen.test_data_gen
不依赖 pytest（本机装不上），直接断言。
"""

from __future__ import annotations

import sys

from .crawl import clean_body, quality_ok, topic_score
from .extract import extract_links, html_to_text, page_title
from .build_corpus import _normalize_body, _simhash, _hamming


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