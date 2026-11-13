"""HTML 正文提取。

只用标准库 html.parser 构建轻量 DOM，再用「文本块密度打分」挑正文容器。
不用 BeautifulSoup 的原因：本机装不上。不用固定 CSS 选择器的原因：站点改版
频繁，硬编码选择器一周就失效，而密度打分对多站点通用。

打分思路：
    score = 中文/字符量 × (1 - 链接密度) × 段落数加成 − 噪声惩罚
候选容器是 div/article/section/td 等块级元素，取分数最高者。长文本优先，
链接密集的导航/列表页自然得分低。
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Dict, List, Optional

# 整棵子树直接丢弃的标签（可嵌套，需要配对结束标签）
DROP_TAGS = {
    "script", "style", "noscript", "iframe", "svg", "canvas", "form",
    "nav", "header", "footer", "aside", "button", "select",
}

# 结构性容器：遇到它们时，说明前一个未闭合的 form 等噪声块已经结束，
# 必须解除 skip 状态，否则整页正文会被吞掉。
STRUCTURAL_TAGS = {"div", "article", "section", "main", "body", "table"}

# 空元素类噪声标签：HTML 里常被写成未闭合形式（如 `<input ...>`），
# 因此不能进 skip 栈，否则后续所有内容都会被误判为“仍在跳过中”。
VOID_NOISE_TAGS = {"input", "textarea", "option", "optgroup", "embed", "object"}

# 正文容器候选
BLOCK_TAGS = {"div", "article", "section", "main", "td", "body"}

# class/id 命中即降权（导航、面包屑、相关推荐、版权等）
NOISE_PATTERN = re.compile(
    r"(nav|menu|breadcrumb|crumb|footer|header|sidebar|side-bar|aside|"
    r"comment|share|related|recommend|hot-list|rank|advert|\bad\b|ads|"
    r"banner|toolbar|pagination|pager|copyright|声明|copyright)",
    re.I,
)

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
SENT_SPLIT = re.compile(r"[。！？；!?;]")
WS_RE = re.compile(r"[ \t\u3000\xa0]+")


class _Node:
    __slots__ = ("tag", "attrs", "children", "parent", "text_parts")

    def __init__(self, tag: str, attrs: Dict[str, str], parent: Optional["_Node"]):
        self.tag = tag
        self.attrs = attrs
        self.children: List["_Node"] = []
        self.parent = parent
        self.text_parts: List[str] = []

    # ---- 便捷属性 ----
    @property
    def cls(self) -> str:
        return (self.attrs.get("class") or "") + " " + (self.attrs.get("id") or "")

    def raw_text(self) -> str:
        """递归拼接本子树所有文本（含子节点）。"""
        buf = list(self.text_parts)
        for c in self.children:
            buf.append(c.raw_text())
        return "".join(buf)

    def link_len(self) -> int:
        """本子树内 <a> 标签的文本字符数。"""
        total = len("".join(self.text_parts)) if self.tag == "a" else 0
        for c in self.children:
            total += c.link_len()
        return total

    def paragraphs(self, out: List[str]) -> None:
        if self.tag == "p":
            t = self.raw_text().strip()
            if t:
                out.append(t)
            return
        for c in self.children:
            c.paragraphs(out)

    def text_lines(self, out: List[str], depth: int = 0) -> None:
        """按块级边界收集文本行，用于没有 <p> 的站点。"""
        if self.tag in ("p", "li", "h1", "h2", "h3", "blockquote", "td"):
            t = self.raw_text().strip()
            if t:
                out.append(t)
            return
        for c in self.children:
            c.text_lines(out, depth + 1)


class _DomBuilder(HTMLParser):
    """容错的 DOM 构建器。

    真实页面存在未闭合的 <form>/<script>，若只用 skip 计数器会在遇到
    `</form>` 缺失时永久卡在 skip 状态，把整页正文丢掉。这里改为维护
    skip 标签栈：进入时压栈，遇到同名结束标签就出栈并立即恢复解析。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("root", {}, None)
        self.stack: List[_Node] = [self.root]
        self._skip_tags: List[str] = []

    @property
    def _skipping(self) -> bool:
        return bool(self._skip_tags)

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in VOID_NOISE_TAGS:
            return
        # 结构性容器出现时，解除可能悬挂的未闭合噪声块（如缺失 </form>）
        if tag in STRUCTURAL_TAGS and self._skip_tags:
            self._skip_tags.clear()
        if tag in DROP_TAGS:
            self._skip_tags.append(tag)
            return
        if self._skipping:
            return
        if tag in ("br", "img", "hr", "meta", "link", "source", "wbr"):
            if tag == "br":
                self.stack[-1].text_parts.append("\n")
            return
        node = _Node(tag, {k.lower(): (v or "") for k, v in attrs}, self.stack[-1])
        self.stack[-1].children.append(node)
        self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        tag = tag.lower()
        if tag in DROP_TAGS or self._skipping:
            return
        node = _Node(tag, {k.lower(): (v or "") for k, v in attrs}, self.stack[-1])
        self.stack[-1].children.append(node)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in VOID_NOISE_TAGS:
            return
        if tag in DROP_TAGS:
            # 只弹出最近的同名标签，未闭合的其它标记不影响后续解析
            if tag in self._skip_tags:
                self._skip_tags.remove(tag)
            return
        if self._skipping:
            return
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        if self._skipping:
            return
        if data.strip():
            self.stack[-1].text_parts.append(data)


def _clean_text(s: str) -> str:
    s = WS_RE.sub(" ", s)
    s = s.replace("\r", "")
    lines = [ln.strip() for ln in s.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()


def _score(node: _Node) -> float:
    text = node.raw_text()
    if not text.strip():
        return 0.0
    cjk = len(CJK_RE.findall(text))
    if cjk < 60:
        return 0.0
    total = len(text)
    link_len = node.link_len()

    ps: List[str] = []
    node.paragraphs(ps)
    if len(ps) < 2:  # 无 <p> 结构时退回行采集
        ps = []
        node.text_lines(ps)

    n_para = sum(1 for p in ps if len(p) >= 12)
    sent = len(SENT_SPLIT.findall(text))

    score = cjk * (1.0 - min(link_len / max(total, 1), 1.0)) ** 2
    score *= 1.0 + min(n_para, 40) * 0.03 + min(sent, 120) * 0.005
    if NOISE_PATTERN.search(node.cls):
        score *= 0.25
    if node.tag in ("article", "main"):
        score *= 1.15
    return score


def _walk(node: _Node, best: List):
    if node.tag in BLOCK_TAGS:
        s = _score(node)
        if s > best[0]:
            best[0], best[1] = s, node
    for c in node.children:
        _walk(c, best)


def html_to_text(html: str) -> str:
    """从整页 HTML 中抽正文纯文本，段落用换行分隔。"""
    if not html:
        return ""
    builder = _DomBuilder()
    try:
        builder.feed(html)
        builder.close()
    except Exception:
        pass

    best: List = [0.0, None]
    _walk(builder.root, best)
    node = best[1] or builder.root

    paras: List[str] = []
    node.paragraphs(paras)
    if len(paras) < 2:
        paras = []
        node.text_lines(paras)

    lines = []
    seen = set()
    for p in paras:
        t = _clean_text(p)
        if len(t) < 8:
            continue
        if t in seen:
            continue
        seen.add(t)
        lines.append(t.replace("\n", " "))
    return "\n".join(lines)


def page_title(html: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", html or "", re.S | re.I)
    if not m:
        return ""
    return re.sub(r"\s+", " ", m.group(1)).strip()


def extract_links(html: str) -> List[tuple]:
    """返回 [(href, anchor_text), ...]，用于列表页发现详情链接。"""
    out: List[tuple] = []
    for m in re.finditer(
        r'<a\b[^>]*href\s*=\s*["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        html or "",
        re.S | re.I,
    ):
        href = m.group(1).strip()
        anchor = re.sub(r"<[^>]+>", "", m.group(2))
        anchor = WS_RE.sub(" ", anchor).strip()
        if href:
            out.append((href, anchor))
    return out