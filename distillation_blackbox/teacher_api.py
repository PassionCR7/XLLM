"""黑盒教师 API 客户端。

设计要点
--------
1. **纯文本黑盒**：教师只暴露输入输出（以及可选的 token logprob），
   不暴露 hidden states / attention，因此本目录不做特征级蒸馏。
2. **推理型教师适配**：实测该 API 默认走 thinking 通道，
   `message.content` 为空而 `reasoning_content` 有内容。
   蒸馏目标是"面向用户的回答"，因此必须关闭 thinking；
   同时提供 fallback 解析，防止不同网关实现差异导致空样本。
3. **logprob 有效性校验**：实测未关闭 thinking 时 logprob 会返回
   -1e38 哨兵值，属于无效概率。客户端对每个 token 做健康度统计，
   只有真实概率占比足够高时才把该样本用于 logprob 级蒸馏。
"""

from __future__ import annotations

import json
import math
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable

from config import TeacherAPIConfig

# 网关返回的无效 logprob 哨兵（-1e38 量级）
_INVALID_LOGP = -1e30


@dataclass
class TeacherSample:
    """一条教师响应。"""

    prompt_messages: list[dict]
    response_text: str
    reasoning_text: str = ""
    finish_reason: str = ""
    # [(token_str, logprob, top_logprobs[{token, logprob}])]
    token_logprobs: list[dict] | None = None
    # 有效概率 token 占比，用于质量门控
    logprob_valid_ratio: float = 0.0
    usage: dict | None = None
    raw: dict | None = None

    def to_sft_conversation(self, keep_reasoning: bool = False) -> dict:
        """转成 xllm SFTDataset 期望的 conversations 格式。"""
        conv = list(self.prompt_messages)
        assistant: dict[str, Any] = {"role": "assistant", "content": self.response_text}
        if keep_reasoning and self.reasoning_text.strip():
            assistant["reasoning_content"] = self.reasoning_text
        conv.append(assistant)
        return {"conversations": conv}


class TeacherClient:
    """OpenAI 兼容接口的黑盒教师客户端。"""

    def __init__(self, cfg: TeacherAPIConfig):
        self.cfg = cfg
        self._local = threading.local()
        self.stats = {
            "requests": 0,
            "success": 0,
            "failed": 0,
            "empty_content": 0,
            "logprob_invalid": 0,
        }
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # 底层 HTTP
    # ------------------------------------------------------------------ #
    def _post(self, payload: dict) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.cfg.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.cfg.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.cfg.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _build_payload(self, messages: list[dict]) -> dict:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_tokens,
        }
        if self.cfg.request_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = self.cfg.top_logprobs
        if not self.cfg.enable_thinking:
            # 不同网关对关闭思考的参数命名不一，同时给出常见写法，
            # 服务端通常会忽略不认识的字段。
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            payload["enable_thinking"] = False
        return payload

    # ------------------------------------------------------------------ #
    # 单条请求（含重试）
    # ------------------------------------------------------------------ #
    def complete(self, messages: list[dict]) -> TeacherSample | None:
        payload = self._build_payload(messages)
        last_err: Exception | None = None

        for attempt in range(self.cfg.max_retries):
            try:
                data = self._post(payload)
                sample = self._parse(data, messages)
                with self._lock:
                    self.stats["requests"] += 1
                    if sample and sample.response_text.strip():
                        self.stats["success"] += 1
                    else:
                        self.stats["empty_content"] += 1
                if sample and sample.response_text.strip():
                    return sample
                # 内容为空不作为致命错误，交由上层过滤
                return sample
            except urllib.error.HTTPError as e:
                last_err = e
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "ignore")[:300]
                except Exception:
                    pass
                # 4xx（除 429）通常不可重试
                if 400 <= e.code < 500 and e.code != 429:
                    with self._lock:
                        self.stats["failed"] += 1
                    raise RuntimeError(f"HTTP {e.code}: {detail}") from e
            except Exception as e:  # 网络超时、连接重置等
                last_err = e

            if attempt < self.cfg.max_retries - 1:
                # 指数退避 + 抖动，避免并发下的同步重试风暴
                delay = self.cfg.retry_backoff ** attempt + random.random()
                time.sleep(delay)

        with self._lock:
            self.stats["failed"] += 1
        raise RuntimeError(f"教师请求失败（已重试{self.cfg.max_retries}次）: {last_err}")

    # ------------------------------------------------------------------ #
    # 响应解析
    # ------------------------------------------------------------------ #
    def _parse(self, data: dict, messages: list[dict]) -> TeacherSample:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}

        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        # 兼容部分网关把思考内容放在 content 里以  thinking 包裹的形式
        if not content and reasoning:
            content = ""

        logprobs_obj = choice.get("logprobs") or {}
        token_lps = logprobs_obj.get("content") or []

        valid = 0
        for item in token_lps:
            if item.get("logprob", _INVALID_LOGP) > _INVALID_LOGP:
                valid += 1
        ratio = valid / len(token_lps) if token_lps else 0.0
        if token_lps and ratio < 0.99:
            with self._lock:
                self.stats["logprob_invalid"] += 1

        return TeacherSample(
            prompt_messages=list(messages),
            response_text=content.strip(),
            reasoning_text=reasoning,
            finish_reason=choice.get("finish_reason", ""),
            token_logprobs=token_lps or None,
            logprob_valid_ratio=ratio,
            usage=data.get("usage"),
            raw=data,
        )

    # ------------------------------------------------------------------ #
    # 批量并发
    # ------------------------------------------------------------------ #
    def complete_batch(
        self,
        batch: Iterable[list[dict]],
        concurrency: int | None = None,
    ) -> list[TeacherSample | None]:
        jobs = list(batch)
        workers = concurrency or self.cfg.concurrency
        results: list[TeacherSample | None] = [None] * len(jobs)

        def _run(idx: int, msgs: list[dict]):
            try:
                return idx, self.complete(msgs)
            except Exception as e:
                print(f"[warn] 样本 {idx} 失败: {e}")
                return idx, None

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run, i, m) for i, m in enumerate(jobs)]
            for fut in as_completed(futures):
                idx, sample = fut.result()
                results[idx] = sample
        return results

    # ------------------------------------------------------------------ #
    # 能力探测
    # ------------------------------------------------------------------ #
    def probe(self) -> dict:
        """探测教师能力：是否可用、是否支持真实 logprob。"""
        probe_msgs = [
            {"role": "user", "content": "用一句话说明什么是知识蒸馏。"}
        ]
        sample = self.complete(probe_msgs)
        info = {
            "model": self.cfg.model,
            "reachable": sample is not None,
            "content_nonempty": bool(sample and sample.response_text),
            "content_preview": (sample.response_text[:80] if sample else ""),
            "supports_logprobs": bool(sample and sample.token_logprobs),
            "logprob_valid_ratio": (sample.logprob_valid_ratio if sample else 0.0),
            "finish_reason": (sample.finish_reason if sample else ""),
            "usage": (sample.usage if sample else None),
        }
        return info


def estimate_cost(samples: list[TeacherSample]) -> dict:
    """统计 token 消耗，便于成本核算。"""
    total_prompt = total_completion = 0
    for s in samples:
        if not s or not s.usage:
            continue
        total_prompt += s.usage.get("prompt_tokens", 0)
        total_completion += s.usage.get("completion_tokens", 0)
    return {
        "samples": len(samples),
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
    }