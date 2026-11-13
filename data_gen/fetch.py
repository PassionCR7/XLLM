"""HTTP 抓取层。

只用标准库 urllib，因为本机 pip 源不可达，无法安装 requests。
核心职责：
1. 编码探测：优先响应头 charset -> HTML meta -> 回退 gb18030。中文站点大量
   使用 GBK，探测失败会直接产出乱码语料，这是数据质量的第一道关。
2. 重试：目标站点存在明显抖动（实测同一 URL 连续请求会超时），采用指数退避。
3. 限速：按域名维护最小请求间隔，避免给对方服务器造成压力。
4. robots.txt：抓取前检查，遵守 Disallow 规则。
"""

from __future__ import annotations

import gzip
import io
import random
import re
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass, field
from typing import Dict, Optional

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# 常见中文站点编码别名
_CHARSET_ALIAS = {
    "gbk": "gb18030",
    "gb2312": "gb18030",
    "gb18030": "gb18030",
    "utf8": "utf-8",
    "utf-8": "utf-8",
}


def normalize_url(url: str, base: str = "") -> str:
    """补全协议相对链接（如 //www.19lou.com/x）与站内相对路径。"""
    url = url.strip()
    if not url or url.startswith(("javascript:", "mailto:", "#")):
        return ""
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/") and base:
        p = urllib.parse.urlsplit(base)
        return f"{p.scheme}://{p.netloc}{url}"
    if not url.startswith("http"):
        return urllib.parse.urljoin(base, url) if base else ""
    return url


def _maybe_decompress(raw: bytes, encoding: str) -> bytes:
    encoding = (encoding or "").lower()
    try:
        if "gzip" in encoding:
            return gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
        if "deflate" in encoding:
            return zlib.decompress(raw, -zlib.MAX_WBITS)
    except Exception:
        pass
    return raw


def detect_and_decode(raw: bytes, content_type: str = "") -> str:
    """按 charset 优先级解码，全部失败时回退 gb18030 忽略错误。

    gb18030 是 GB2312/GBK 的超集，作为兜底能覆盖绝大多数简体中文老站点，
    且不会像 utf-8 那样把整段中文解成替换字符。
    """
    candidates = []
    m = re.search(r"charset=[\"']?([\w\-]+)", content_type or "", re.I)
    if m:
        candidates.append(m.group(1).lower())
    m2 = re.search(rb"charset=[\"']?([\w\-]+)", raw[:4096], re.I)
    if m2:
        candidates.append(m2.group(1).decode("ascii", "ignore").lower())
    candidates.extend(["utf-8", "gb18030"])

    for cs in candidates:
        cs = _CHARSET_ALIAS.get(cs, cs)
        try:
            return raw.decode(cs)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("gb18030", "ignore")


@dataclass
class FetchResult:
    url: str
    status: int = 0
    html: str = ""
    error: str = ""
    elapsed: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == 200 and bool(self.html) and not self.error


@dataclass
class HttpClient:
    """带重试与限速的抓取客户端。"""

    user_agent: str = DEFAULT_UA
    timeout: float = 15.0
    retries: int = 3
    backoff: float = 1.5
    min_interval: float = 1.0  # 同域名两次请求的最小间隔（秒）
    respect_robots: bool = True
    verify_ssl: bool = True
    verbose: bool = True

    _last_hit: Dict[str, float] = field(default_factory=dict)
    _robots: Dict[str, Optional[re.Pattern]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        socket.setdefaulttimeout(self.timeout)
        ctx = ssl.create_default_context()
        if not self.verify_ssl:
            # 部分政府/地方站点证书链不完整，仅在对可信源时才关闭校验
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        self._ctx = ctx

    # ---------------- robots ----------------
    def _robots_pattern(self, url: str) -> Optional[re.Pattern]:
        host = urllib.parse.urlsplit(url).netloc
        if host in self._robots:
            return self._robots[host]
        pattern = None
        try:
            txt = self._raw_get(f"https://{host}/robots.txt", timeout=8).decode(
                "utf-8", "ignore"
            )
            disallows = []
            applies = False
            for line in txt.splitlines():
                line = line.split("#")[0].strip()
                low = line.lower()
                if low.startswith("user-agent:"):
                    ua = line.split(":", 1)[1].strip()
                    applies = ua == "*"
                elif applies and low.startswith("disallow:"):
                    path = line.split(":", 1)[1].strip()
                    if path:
                        disallows.append(path)
            if disallows:
                pattern = re.compile(
                    "|".join(re.escape(p) for p in disallows if p != "/")
                )
                if any(p == "/" for p in disallows):
                    pattern = re.compile("^/")
        except Exception:
            pattern = None
        self._robots[host] = pattern
        return pattern

    def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        pat = self._robots_pattern(url)
        if pat is None:
            return True
        path = urllib.parse.urlsplit(url).path or "/"
        return not pat.search(path)

    # ---------------- 底层请求 ----------------
    def _throttle(self, url: str) -> None:
        host = urllib.parse.urlsplit(url).netloc
        last = self._last_hit.get(host, 0.0)
        wait = self.min_interval - (time.time() - last)
        if wait > 0:
            time.sleep(wait + random.uniform(0, 0.3))
        self._last_hit[host] = time.time()

    def _raw_get(self, url: str, timeout: Optional[float] = None) -> bytes:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "close",
            },
        )
        with urllib.request.urlopen(
            req, timeout=timeout or self.timeout, context=self._ctx
        ) as resp:
            raw = resp.read()
            return _maybe_decompress(raw, resp.headers.get("Content-Encoding", ""))

    def get(self, url: str, timeout: Optional[float] = None) -> FetchResult:
        """抓取单个 URL，失败按指数退避重试。"""
        if not self.allowed(url):
            return FetchResult(url=url, error="blocked-by-robots")

        last_err = ""
        for attempt in range(1, self.retries + 1):
            start = time.time()
            try:
                self._throttle(url)
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": self.user_agent,
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                        "Accept-Encoding": "gzip, deflate",
                        "Connection": "close",
                    },
                )
                with urllib.request.urlopen(
                    req, timeout=timeout or self.timeout, context=self._ctx
                ) as resp:
                    raw = _maybe_decompress(
                        resp.read(), resp.headers.get("Content-Encoding", "")
                    )
                    html = detect_and_decode(raw, resp.headers.get("Content-Type", ""))
                    return FetchResult(
                        url=url,
                        status=resp.status,
                        html=html,
                        elapsed=time.time() - start,
                    )
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code}"
                # 4xx 除 429 外重试无意义
                if 400 <= e.code < 500 and e.code != 429:
                    break
            except Exception as e:  # 超时、连接重置、SSL 等
                last_err = f"{type(e).__name__}: {e}"

            if attempt < self.retries:
                time.sleep(self.backoff * attempt + random.uniform(0, 0.5))

        return FetchResult(url=url, status=0, error=last_err)