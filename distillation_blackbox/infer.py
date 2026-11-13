"""黑盒蒸馏学生的推理脚本。

用途
----
加载黑盒蒸馏产出的学生权重，做交互式或批量对话推理。

与 `evaluate.py` 里那个 `generate()` 的区别
-------------------------------------------
`evaluate.py` 的 generate 是评估流程的内部函数：写死单轮、写死采样参数、
不支持多轮历史、没有 CLI。本脚本是独立可用的推理入口：

* 单条提问 / 交互式 / 批量文件三种模式
* 支持系统提示与多轮历史
* 支持贪心与采样两种解码，可调 top_p / repetition_penalty
* batch 推理（多条提问一次前向）
* 可选 `--show_think` 保留思维链段落
* 可选 `--logprobs` 查看每个生成 token 的概率

用法
----
    # 单条提问
    python infer.py --prompt "你是谁？"

    # 指定权重
    python infer.py --weight distill_blackbox --prompt "你是谁？"

    # 交互式（多轮，保留历史）
    python infer.py --chat

    # 批量：每行一个提问，输出 jsonl
    python infer.py --input questions.txt --output answers.jsonl

    # 贪心解码，结果可复现
    python infer.py --prompt "你是谁？" --greedy

    # 对比基线权重
    python infer.py --weight full_sft --prompt "你是谁？"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DistillConfig                     # noqa: E402
from train_kd import load_student                    # noqa: E402

# 思维链标记。用 chr() 构造：源码里的尖括号在部分写入链路会被吞掉，
# 直接写会得到一个空字符串，清理函数静默失效。
_THINK_OPEN = chr(60) + "think" + chr(62)
_THINK_CLOSE = chr(60) + "/think" + chr(62)


def strip_think(text: str) -> str:
    """清理思维链，只保留面向用户的回答。

    三种情况依次处理：
    1. 成对标记：整段删除；
    2. 只有开标签（生成被长度截断）：截断到开标签之前；
    3. 无标记：原样返回。
    """
    out = text
    while True:
        start = out.find(_THINK_OPEN)
        if start < 0:
            break
        end = out.find(_THINK_CLOSE, start)
        if end < 0:
            out = out[:start]
            break
        out = out[:start] + out[end + len(_THINK_CLOSE):]
    return out.strip()


# --------------------------------------------------------------------------- #
# 生成核心
# --------------------------------------------------------------------------- #
@torch.no_grad()
def chat_once(
    model,
    tok,
    messages: list[dict],
    device: str = "cuda",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
    greedy: bool = False,
    return_logprobs: bool = False,
    no_repeat_ngram: int = 0,
) -> dict:
    """单条对话生成。messages 为 [{"role","content"}, ...]。"""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        repetition_penalty=repetition_penalty,
    )
    if no_repeat_ngram > 0:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram
    if greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

    t0 = time.time()
    out = model.generate(**gen_kwargs)
    elapsed = time.time() - t0

    new_ids = out[0][prompt_len:]
    raw = tok.decode(new_ids, skip_special_tokens=True)
    answer = strip_think(raw)

    result = {
        "answer": answer,
        "raw": raw,
        "prompt_tokens": int(prompt_len),
        "new_tokens": int(new_ids.numel()),
        "elapsed_s": round(elapsed, 3),
    }

    if return_logprobs:
        # 每生成一个 token 的 logprob。贪心解码下这是"该选择有多确定"，
        # 采样解码下只是"采样到的那一个 token 的原概率"。
        with torch.no_grad():
            logits = model(input_ids=out, attention_mask=torch.ones_like(out)).logits
        lp = F.log_softmax(logits[0, prompt_len - 1: -1].float(), dim=-1)
        picked = torch.gather(lp, 1, new_ids.unsqueeze(1)).squeeze(1)
        result["token_logprobs"] = [
            {"token": tok.decode([int(t)]), "logprob": round(float(v), 4)}
            for t, v in zip(new_ids, picked)
        ]

    return result


@torch.no_grad()
def batch_infer(
    model,
    tok,
    prompts: list[str],
    device: str = "cuda",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    greedy: bool = False,
    batch_size: int = 4,
    system: str | None = None,
    repetition_penalty: float = 1.05,
    no_repeat_ngram: int = 0,
) -> list[dict]:
    """批量生成。左侧补齐以便统一长度（右侧补齐会干扰生成位置）。"""
    old_side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    results: list[dict] = []
    t_all = time.time()

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        texts = []
        for p in chunk:
            msgs = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": p}
            ]
            texts.append(
                tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            )

        enc = tok(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(device)

        kwargs = dict(
            **enc,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            repetition_penalty=repetition_penalty,
        )
        if no_repeat_ngram and no_repeat_ngram > 0:
            kwargs["no_repeat_ngram_size"] = no_repeat_ngram
        if greedy:
            kwargs["do_sample"] = False
        else:
            kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)

        out = model.generate(**kwargs)
        gen = out[:, enc["input_ids"].shape[1]:]

        for p, g in zip(chunk, gen):
            raw = tok.decode(g, skip_special_tokens=True)
            results.append(
                {
                    "prompt": p,
                    "answer": strip_think(raw),
                    "raw": raw,
                    "new_tokens": int((g != tok.pad_token_id).sum()),
                }
            )

    tok.padding_side = old_side
    for r in results:
        r["batch_elapsed_s"] = round(time.time() - t_all, 3)
    return results


# --------------------------------------------------------------------------- #
# 权重解析
# --------------------------------------------------------------------------- #
def resolve_weight(name: str, cfg: DistillConfig) -> str:
    """权重名 -> 完整路径。先找蒸馏产物目录，再找项目 out/。"""
    suffix = f"_{cfg.student.hidden_size}"
    stem = name if name.endswith(suffix) else f"{name}{suffix}"
    for d in (cfg.student.save_dir, cfg.student.weight_dir):
        p = Path(d) / f"{stem}.pth"
        if p.exists():
            return str(p)
    # 也接受直接给 .pth 路径
    if name.endswith(".pth") and Path(name).exists():
        return name
    raise FileNotFoundError(
        f"未找到权重 {stem}.pth\n  已查找: {cfg.student.save_dir}\n           {cfg.student.weight_dir}"
    )


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="黑盒蒸馏学生推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--weight", default="distill_blackbox",
                    help="权重名（默认 distill_blackbox，也可 full_sft 做对照）")
    ap.add_argument("--prompt", default=None, help="单条提问")
    ap.add_argument("--input", default=None, help="批量提问文件，每行一条")
    ap.add_argument("--output", default=None, help="结果输出 jsonl 路径")
    ap.add_argument("--chat", action="store_true", help="交互式多轮对话")
    ap.add_argument("--system", default=None, help="系统提示词")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--greedy", action="store_true", help="贪心解码（结果可复现）")
    ap.add_argument("--batch_size", type=int, default=4, help="批量模式的批大小")
    ap.add_argument("--no_repeat_ngram", type=int, default=0,
                    help="禁止重复的 n-gram 长度（0 关闭，建议 3 抑制复读）")
    ap.add_argument("--show_think", action="store_true", help="保留思维链段落")
    ap.add_argument("--logprobs", action="store_true", help="输出每 token 的 logprob")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if not (args.prompt or args.chat or args.input):
        ap.error("需要 --prompt / --chat / --input 之一")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DistillConfig()
    weight_path = resolve_weight(args.weight, cfg)

    print(f"[权重] {weight_path}")
    print(f"[设备] {device}")
    # 关键：必须把解析结果写回 config，否则 load_student 仍按默认的
    # from_weight="full_sft" + weight_dir=<项目 out/> 加载，导致
    # --weight 参数被静默忽略（曾因此让所有测试都在跑基线）。
    cfg.student.weight_dir = str(Path(weight_path).parent)
    cfg.student.from_weight = Path(weight_path).stem
    model, tok = load_student(cfg, device)
    model.eval()

    def render(d: dict) -> str:
        return d["raw"] if args.show_think else d["answer"]

    # ---------- 批量 ----------
    if args.input:
        prompts = [
            ln.strip()
            for ln in Path(args.input).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        print(f"[批量] {len(prompts)} 条提问，batch_size={args.batch_size}")
        results = batch_infer(
            model, tok, prompts, device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p,
            greedy=args.greedy, batch_size=args.batch_size,
            system=args.system,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram=args.no_repeat_ngram,
        )
        out_path = Path(args.output or "infer_results.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[输出] {out_path}")
        print(f"[耗时] {results[0]['batch_elapsed_s'] if results else 0}s "
              f"| 平均 {sum(r['new_tokens'] for r in results)/max(1,len(results)):.0f} token/条")
        return

    # ---------- 交互式 ----------
    if args.chat:
        print("交互式对话（输入 exit 退出，clear 清空历史）")
        history: list[dict] = []
        if args.system:
            history.append({"role": "system", "content": args.system})
        while True:
            try:
                q = input("\n你> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                continue
            if q.lower() in ("exit", "quit", "q"):
                break
            if q.lower() == "clear":
                history = [m for m in history if m["role"] == "system"]
                print("历史已清空")
                continue

            history.append({"role": "user", "content": q})
            r = chat_once(
                model, tok, history, device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                greedy=args.greedy, return_logprobs=args.logprobs,
                no_repeat_ngram=args.no_repeat_ngram,
            )
            print(f"AI> {render(r)}")
            print(f"    [{r['new_tokens']} tok, {r['elapsed_s']}s]")
            history.append({"role": "assistant", "content": r["answer"]})
        return

    # ---------- 单条 ----------
    msgs = ([{"role": "system", "content": args.system}] if args.system else []) + [
        {"role": "user", "content": args.prompt}
    ]
    r = chat_once(
        model, tok, msgs, device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        greedy=args.greedy, return_logprobs=args.logprobs,
        no_repeat_ngram=args.no_repeat_ngram,
    )
    print("=" * 62)
    print(f"提问: {args.prompt}")
    print("=" * 62)
    print(render(r))
    print("=" * 62)
    print(f"[{r['new_tokens']} tok, {r['elapsed_s']}s, "
          f"{r['new_tokens']/max(r['elapsed_s'],1e-6):.1f} tok/s]")

    if args.logprobs and "token_logprobs" in r:
        print("\n逐 token 概率:")
        for t in r["token_logprobs"]:
            print(f"  {t['token']!r:14s} logprob {t['logprob']:8.4f}  "
                  f"p = {pow(2.718281828, t['logprob']):.4f}")


if __name__ == "__main__":
    main()