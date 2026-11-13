"""黑盒蒸馏数据生成：用教师 API 生成高质量教师回复。

产出格式与项目 `dataset/lm_dataset.py::SFTDataset` 完全兼容：

    {"conversations": [{"role":"user",...},{"role":"assistant",...}],
     "meta": {"teacher":"...", "prompt_tokens":.., "completion_tokens":..}}

设计要点
--------
1. **种子采样**：从项目现有 SFT 数据中提取 user 提问，去重后作为种子，
   保证蒸馏数据与学生的目标分布一致（而不是泛泛的开放域生成）。
2. **质量门控**：过滤空回复、超短回复、以及 logprob 无效的样本
   （后者说明网关未按预期返回真实概率）。
3. **断点续生成**：每写 N 条就落盘一次，中断后可从已有条数继续，
   避免长任务因网络抖动全部重来。
4. **并发 + 限流**：教师 API 有限流，用线程池并发但保留重试退避。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import DistillConfig                    # noqa: E402
from teacher_api import TeacherClient, estimate_cost  # noqa: E402


def extract_seed_prompts(
    seed_file: str,
    limit: int,
    dedup: bool = True,
    min_len: int = 4,
    max_len: int = 400,
) -> list[str]:
    """从项目 SFT 数据中提取 user 提问作为种子。"""
    prompts: list[str] = []
    seen: set[str] = set()

    with open(seed_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for msg in obj.get("conversations", []):
                if msg.get("role") != "user":
                    continue
                text = (msg.get("content") or "").strip()
                if not (min_len <= len(text) <= max_len):
                    continue
                if dedup:
                    if text in seen:
                        continue
                    seen.add(text)
                prompts.append(text)
            if len(prompts) >= limit:
                break
    return prompts[:limit]


def load_existing(out_path: Path) -> list[dict]:
    """读取已生成的数据（用于断点续生成）。"""
    if not out_path.exists():
        return []
    out = []
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _safe_complete(client: TeacherClient, messages: list[dict]):
    """永不抛异常的包装，便于在 as_completed 中逐个处理。"""
    try:
        return client.complete(messages)
    except Exception as e:
        print(f"  [warn] 请求异常: {type(e).__name__}: {str(e)[:100]}")
        return None


def main():
    parser = argparse.ArgumentParser(description="黑盒蒸馏：教师 API 数据生成")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--keep_reasoning", action="store_true")
    parser.add_argument("--resume", action="store_true", help="从已有文件继续生成")
    parser.add_argument("--probe_only", action="store_true", help="只探测教师能力")
    args = parser.parse_args()

    cfg = DistillConfig()
    if args.num_samples is not None:
        cfg.data.num_samples = args.num_samples
    if args.concurrency is not None:
        cfg.teacher.concurrency = args.concurrency
    if args.max_tokens is not None:
        cfg.teacher.max_tokens = args.max_tokens
    if args.temperature is not None:
        cfg.teacher.temperature = args.temperature
    if args.keep_reasoning:
        cfg.data.keep_reasoning = True

    client = TeacherClient(cfg.teacher)

    # ---------- 能力探测 ----------
    print("=" * 62)
    print("教师能力探测")
    print("=" * 62)
    info = client.probe()
    for k, v in info.items():
        print(f"  {k:22s}: {v}")
    if not info["reachable"]:
        raise RuntimeError("教师 API 不可达，请检查网络或密钥")
    if not info["content_nonempty"]:
        raise RuntimeError(
            "教师返回空内容。该模型为 reasoning 模型，"
            "请确认已关闭 thinking（enable_thinking=False）"
        )
    if args.probe_only:
        return

    # ---------- 种子 ----------
    out_dir = Path(cfg.data.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / "teacher_sft.jsonl"

    existing = load_existing(out_path) if args.resume else []
    if existing:
        print(f"[续生成] 已有 {len(existing)} 条，继续补齐到 {cfg.data.num_samples}")

    prompts = extract_seed_prompts(
        cfg.data.seed_file,
        limit=cfg.data.num_samples + len(existing) + 50,
    )
    print(f"[种子] 提取到 {len(prompts)} 条唯一 user 提问")

    if len(prompts) < cfg.data.num_samples:
        print(
            f"[警告] 种子数({len(prompts)}) 少于目标样本数({cfg.data.num_samples})，"
            f"将以实际种子数为准"
        )

    targets = prompts[len(existing): cfg.data.num_samples]
    if not targets:
        print("[完成] 目标数量已满足，无需继续生成")
        return

    # ---------- 并发生成 ----------
    print(f"[生成] 目标 {len(targets)} 条 | 并发 {cfg.teacher.concurrency} | "
          f"max_tokens {cfg.teacher.max_tokens} | T={cfg.teacher.temperature}")

    messages_batch = [[{"role": "user", "content": p}] for p in targets]
    t0 = time.time()

    written = len(existing)
    kept = 0
    failed = 0

    def _write_one(sample, fh) -> bool:
        """把单条样本写入文件，返回是否成功保留。"""
        if sample is None:
            return False
        if not sample.response_text.strip():
            return False
        if len(sample.response_text.strip()) < 8:
            return False

        record = sample.to_sft_conversation(keep_reasoning=cfg.data.keep_reasoning)
        record["meta"] = {
            "teacher": cfg.teacher.model,
            "prompt_tokens": (sample.usage or {}).get("prompt_tokens"),
            "completion_tokens": (sample.usage or {}).get("completion_tokens"),
            "logprob_valid_ratio": round(sample.logprob_valid_ratio, 4),
            "finish_reason": sample.finish_reason,
        }
        if sample.token_logprobs:
            record["teacher_logprobs"] = [
                {
                    "token": t["token"],
                    "logprob": t["logprob"],
                    "top": [
                        {"token": x["token"], "logprob": x["logprob"]}
                        for x in (t.get("top_logprobs") or [])[: cfg.teacher.top_logprobs]
                    ],
                }
                for t in sample.token_logprobs
                if t["logprob"] > -1e30
            ]
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True

    with open(out_path, "a", encoding="utf-8") as f:
        # 关键改进：不使用 complete_batch 的"整批等待"，
        # 而是 as_completed 逐个落盘。这样单条慢请求不会阻塞整批写入，
        # 中断时已完成的样本也不会丢失。
        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutTimeout

        workers = cfg.teacher.concurrency
        chunk = max(workers * 3, 6)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(messages_batch), chunk):
                part = messages_batch[start: start + chunk]
                futures = {
                    pool.submit(_safe_complete, client, m): i
                    for i, m in enumerate(part)
                }
                for fut in as_completed(futures):
                    try:
                        sample = fut.result(timeout=cfg.teacher.timeout + 60)
                    except Exception as e:
                        failed += 1
                        print(f"  [warn] 样本失败: {type(e).__name__}: {str(e)[:80]}")
                        continue
                    if _write_one(sample, f):
                        written += 1
                        kept += 1

                f.flush()
                elapsed = time.time() - t0
                rate = kept / elapsed if elapsed > 0 else 0
                remain = cfg.data.num_samples - written
                eta = remain / rate / 60 if rate > 0 else 0
                print(
                    f"  进度 {written}/{cfg.data.num_samples} | 保留 {kept} 失败 {failed} | "
                    f"{elapsed:.0f}s | {rate:.2f} 条/s | 预计剩余 {eta:.1f} min",
                    flush=True,
                )

    print("=" * 62)
    print(f"[完成] 本次写入 {kept} 条 -> {out_path}")
    print(f"[统计] {client.stats}")
    print(f"[输出] 总条数 {written}")


if __name__ == "__main__":
    main()