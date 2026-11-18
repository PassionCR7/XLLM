"""LLM 调用层（OpenAI 兼容协议）。

为什么需要 LLM
--------------
纯规则打分有两个治不好的病，在真实数据上都出现过：

1. **误收**：上城网一篇「习近平出席APEC峰会」的时政新闻，因为正文里出现
   「杭州」「文化」等词，规则给了 0.70 分（阈值 0.42），直接进了语料。
   规则数的是词频，看不懂"这篇文章到底在讲什么"。
2. **误杀/形态单一**：规则只能判断"要不要"，没法把一篇公文体的通稿
   改写成「打卡帖」——而打卡帖的价值恰恰在于第一人称体验、口语化表达、
   具体到几点几分哪条路的细节。这些只能靠模型生成。

设计取舍
--------
* 只用标准库 urllib，与本项目其余部分保持一致（本机 pip 源不可达）。
* 模型端点从 ~/.workbuddy/models.json 读取，不在代码里硬编码密钥。
* **推理模型会先烧 reasoning token**：实测同一个请求，20 个 token 预算会被
  推理全部吃掉，返回空字符串。因此默认 max_tokens 给足（2048），
  并在解析时容忍空内容。
* 输出统一要求 JSON，解析失败会重试一次并降级为正则抽取，绝不抛异常
  中断整批任务——爬虫最怕的就是单条数据把整轮跑挂掉。
"""

from __future__ import annotations

import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


DEFAULT_MODELS_PATH = "~/.workbuddy/models.json"


class LLMError(RuntimeError):
    pass


def redact(text: str, secret: Optional[str] = None) -> str:
    """把字符串里的密钥打码。

    为什么需要：HTTPError 的响应体有时会回显请求头（部分网关配置出错时
    会这么干），而 cli.py 会把错误信息直接 print 到终端。终端会进 CI 日志、
    会被截图、会贴进 issue。所以出口统一脱敏，不赌服务端不回显。

    只做两件事，都是精确匹配，不误伤正常文本：
      1. 已知的真实密钥值，整体替换
      2. 形如 sk-xxxx 的串（OpenAI 系约定的前缀），只保留前 6 位

    刻意不用「任意长字母数字串」这种宽泛正则——那会把 URL、文件名、
    中文句子一起打码，日志就没法看了。
    """
    if not text:
        return text
    if secret and len(secret) >= 8:
        text = text.replace(secret, secret[:6] + "*" * 8)
    text = re.sub(r"(sk-[A-Za-z0-9_\-]{6})[A-Za-z0-9_\-]{8,}", r"\1********", text)
    return text


def load_model_config(model_id: Optional[str] = None,
                      path: str = DEFAULT_MODELS_PATH) -> Dict:
    """从 WorkBuddy 的模型配置里取第一个可用模型（或指定 id）。"""
    p = os.path.expanduser(path)
    if not os.path.exists(p):
        raise LLMError(
            f"找不到模型配置 {p}。\n"
            "请在其中写入模型列表（含 url 与 apiKey），或在代码里显式传入\n"
            "LLMClient(api_base=..., api_key=..., model_id=...)。\n"
            "注意：不要把密钥写进命令行参数——shell 历史会留痕，进程列表也看得见。"
        )
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, list) or not cfg:
        raise LLMError(f"{p} 格式异常，应为模型列表")
    if model_id:
        for m in cfg:
            if m.get("id") == model_id:
                return m
        raise LLMError(f"配置里没有模型 {model_id}，可用: "
                       + ", ".join(x.get("id", "?") for x in cfg))
    return cfg[0]


def _extract_json(text: str) -> Optional[dict]:
    """从模型输出里抠出 JSON。

    模型常见毛病：包 ```json 围栏、前后加解释、结尾多个逗号。
    这里按「先整体解析，再定位花括号」的顺序尝试，尽量不浪费一次重试。
    """
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    # 定位最外层花括号
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        frag = s[i : j + 1]
        try:
            return json.loads(frag)
        except Exception:
            frag2 = re.sub(r",\s*([}\]])", r"\1", frag)  # 去尾逗号
            try:
                return json.loads(frag2)
            except Exception:
                return None
    return None


@dataclass
class LLMClient:
    model_id: Optional[str] = None
    api_base: Optional[str] = None
    # repr=False：这个字段存的是明文密钥。dataclass 默认会把所有字段塞进
    # __repr__，于是 print(client)、logging、traceback 里的局部变量展示
    # 都会把 key 原样打印出来。实测确实会：
    #   LLMClient(..., api_key='sk-xxx', ...)
    # 只加在 _cfg/_key 上不够，api_key 是构造入参，必须单独排除。
    api_key: Optional[str] = field(default=None, repr=False)
    models_path: str = DEFAULT_MODELS_PATH
    max_tokens: int = 2048
    timeout: float = 120.0
    retries: int = 2
    temperature: Optional[float] = None
    min_interval: float = 0.4
    verify_ssl: bool = False

    _cfg: Dict = field(default_factory=dict, repr=False)
    _last_call: float = field(default=0.0, repr=False)
    stats: Dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.api_base and self.api_key:
            self._cfg = {"id": self.model_id or "default",
                         "url": self.api_base, "apiKey": self.api_key}
        else:
            self._cfg = load_model_config(self.model_id, self.models_path)
            if self.model_id is None:
                self.model_id = self._cfg.get("id")
        self._url = self._cfg["url"].rstrip("/")
        if not self._url.endswith("/chat/completions"):
            self._url += "/chat/completions"
        self._key = self._cfg["apiKey"]
        ctx = ssl.create_default_context()
        if not self.verify_ssl:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        self._ctx = ctx

    def _bump(self, k: str) -> None:
        self.stats[k] = self.stats.get(k, 0) + 1

    # ------------------------------------------------------------------
    def chat(self, system: str, user: str, max_tokens: Optional[int] = None,
             json_mode: bool = True) -> str:
        """发起一次对话，返回纯文本内容（可能为空串）。"""
        wait = self.min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens or self.max_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        # 部分推理模型拒绝 response_format，这里不强制，靠提示词约束

        last_err = ""
        for attempt in range(1, self.retries + 1):
            try:
                req = urllib.request.Request(
                    self._url,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {self._key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(
                    req, timeout=self.timeout, context=self._ctx
                ) as resp:
                    raw = resp.read().decode("utf-8", "ignore")
                d = json.loads(raw)
                msg = d["choices"][0]["message"]
                content = msg.get("content") or ""
                self._bump("ok")
                return content
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", "ignore")[:200]
                except Exception:
                    pass
                # 响应体可能回显请求头，先脱敏再拼进错误信息
                last_err = redact(f"HTTP {e.code} {body}", self._key)
                if e.code in (401, 403):
                    break  # 鉴权错误重试无意义
            except Exception as e:
                last_err = redact(f"{type(e).__name__}: {e}", self._key)

            if attempt < self.retries:
                time.sleep(1.5 * attempt)

        self._bump("fail")
        raise LLMError(last_err)

    def chat_json(self, system: str, user: str,
                  max_tokens: Optional[int] = None) -> Optional[dict]:
        """要求模型返回 JSON 并解析；解析失败重试一次更严厉的提示。"""
        for i in range(2):
            sys2 = system if i == 0 else (
                system + "\n\n【重要】上一次输出无法解析。只输出一个 JSON 对象，"
                "不要任何解释文字，不要 markdown 代码围栏。"
            )
            try:
                txt = self.chat(sys2, user, max_tokens=max_tokens)
            except LLMError:
                if i == 0:
                    continue
                return None
            obj = _extract_json(txt)
            if obj is not None:
                return obj
            self._bump("json_retry")
        self._bump("json_fail")
        return None