"""白盒蒸馏学生的推理脚本。

用途
----
加载白盒蒸馏产出的学生权重做对话推理，并可选择性地输出教师的对比结果。

与 `evaluate.py` 里那个 `generate()` 的区别
-------------------------------------------
`evaluate.py` 的 generate 是评估内部函数：写死单轮、写死采样参数、无 CLI。
本脚本是独立推理入口：

* 单条 / 交互式 / 批量三种模式
* 支持系统提示与多轮历史
* 贪心与采样可切换，top_p / repetition_penalty 可调
* `--compare_teacher` 同时跑 Qwen3-0.6B 教师，输出并排对比
* 可选 `--logprobs` 查看生成 token 概率

用法
----
    # 单条
    python infer.py --prompt "你是谁？"

    # 指定权重（默认 distill_whitebox）
    python infer.py --weight distill_whitebox --prompt "你是谁？"

    # 与基线对照
    python infer.py --weight full_sft --prompt "你是谁？"

    # 交互式多轮
    python infer.py --chat

    # 批量
    python infer.py --input questions.txt --output answers.jsonl

    # 同时输出教师回答做对比（额外占约 1.5GB 显存）
    python infer.py --prompt "你是谁？" --compare_teacher
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
# 顺序要紧：先插父目录（为了 model.XLLMModel），再把本地目录插到最前面，
# 否则本地的 dataset.py 会被项目根目录的 dataset/ 包遮蔽。
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from config import WhiteboxConfig                       # noqa: E402
from student_loader import load_student                 # noqa: E402
from train_kd import build_teacher                      # noqa: E402

# 思维链标记，用 chr() 构造避免源码尖括号被吞
_THINK_OPEN = chr(60) + "think" + chr(62)
_THINK_CLOSE = chr(60) + "/think" + chr(62)


def strip_think(text: str) -> str:
    """清理思维链，只保留最终回答。三种情况：成对/未闭合/无标记。"""
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
    """单条对话生成。"""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        repetition_penalty=repetition_penalty,
    )
    if no_repeat_ngram > 0:
        kwargs["no_repeat_ngram_size"] = no_repeat_ngram
    if greedy:
        kwargs["do_sample"] = False
    else:
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)

    t0 = time.time()
    out = model.generate(**kwargs)
    elapsed = time.time() - t0

    new_ids = out[0][prompt_len:]
    raw = tok.decode(new_ids, skip_special_tokens=True)
    result = {
        "answer": strip_think(raw),
        "raw": raw,
        "prompt_tokens": int(prompt_len),
        "new_tokens": int(new_ids.numel()),
        "elapsed_s": round(elapsed, 3),
    }

    if return_logprobs and new_ids.numel() > 0:
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
    """批量生成。必须左侧补齐，右侧补齐会让生成立即撞到 pad。"""
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

        enc = tok(texts, return_tensors="pt", add_special_tokens=False, padding=True).to(device)
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


def resolve_weight(name: str, cfg: WhiteboxConfig) -> str:
    """权重名 -> 完整路径。先蒸馏产物目录，再项目 out/；也接受 .pth 直给。"""
    suffix = f"_{cfg.student.hidden_size}"
    stem = name if name.endswith(suffix) else f"{name}{suffix}"
    for d in (cfg.student.save_dir, cfg.student.weight_dir):
        p = Path(d) / f"{stem}.pth"
        if p.exists():
            return str(p)
    if name.endswith(".pth") and Path(name).exists():
        return name
    raise FileNotFoundError(
        f"未找到权重 {stem}.pth\n  已查找: {cfg.student.save_dir}\n           {cfg.student.weight_dir}"
    )


def main():
    ap = argparse.ArgumentParser(
        description="白盒蒸馏学生推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--weight", default="distill_whitebox",
                    help="权重名（默认 distill_whitebox，也可 full_sft 做对照）")
    ap.add_argument("--prompt", default=None, help="单条提问")
    ap.add_argument("--input", default=None, help="批量提问文件，每行一条")
    ap.add_argument("--output", default=None, help="结果输出 jsonl")
    ap.add_argument("--chat", action="store_true", help="交互式多轮对话")
    ap.add_argument("--system", default=None, help="系统提示词")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--no_repeat_ngram", type=int, default=0,
                    help="禁止重复的 n-gram 长度（0 关闭，建议 3 抑制复读）")
    ap.add_argument("--greedy", action="store_true", help="贪心解码")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--show_think", action="store_true", help="保留思维链")
    ap.add_argument("--logprobs", action="store_true")
    ap.add_argument("--compare_teacher", action="store_true",
                    help="同时加载 Qwen3-0.6B 教师做并排对比")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if not (args.prompt or args.chat or args.input):
        ap.error("需要 --prompt / --chat / --input 之一")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = WhiteboxConfig()
    weight_path = resolve_weight(args.weight, cfg)

    print(f"[权重] {weight_path}")
    print(f"[设备] {device}")
    # 必须把解析结果写回 config，否则 load_student 仍按默认
    # from_weight="full_sft" 加载，--weight 会被静默忽略。
    cfg.student.weight_dir = str(Path(weight_path).parent)
    cfg.student.from_weight = Path(weight_path).stem
    student, s_tok = load_student(cfg.student, device)
    student.eval()

    teacher = t_tok = None
    if args.compare_teacher:
        print("[教师] 加载中（约需 1.5GB 显存）...")
        teacher, t_tok = build_teacher(cfg, device)

    def render(d: dict) -> str:
        return d["raw"] if args.show_think else d["answer"]

    # ---------- 批量 ----------
    if args.input:
        prompts = [
            ln.strip()
            for ln in Path(args.input).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        print(f"[批量] {len(prompts)} 条，batch_size={args.batch_size}")
        results = batch_infer(
            student, s_tok, prompts, device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p,
            greedy=args.greedy, batch_size=args.batch_size, system=args.system,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram=args.no_repeat_ngram,
        )
        out_path = Path(args.output or "infer_results.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[输出] {out_path}")
        return

    # ---------- 交互式 ----------
    if args.chat:
        print("交互式对话（exit 退出，clear 清空历史）")
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
                student, s_tok, history, device=device,
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
        student, s_tok, msgs, device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        greedy=args.greedy, return_logprobs=args.logprobs,
        no_repeat_ngram=args.no_repeat_ngram,
    )

    print("=" * 66)
    print(f"提问: {args.prompt}")
    print("-" * 66)
    print(f"学生: {render(r)}")
    print(f"      [{r['new_tokens']} tok, {r['elapsed_s']}s]")

    if teacher is not None:
        # Qwen3 默认会输出 think 段，strip_think 会把整段删掉导致答案为空。
        # 这里显式关闭思考通道，让教师直接给出面向用户的回答。
        t_msgs = ([{"role": "system", "content": args.system}] if args.system else []) + [
            {"role": "user", "content": args.prompt}
        ]
        try:
            t_text = t_tok.apply_chat_template(
                t_msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # 旧版模板不支持 enable_thinking 参数，退回默认渲染
            t_text = t_tok.apply_chat_template(
                t_msgs, tokenize=False, add_generation_prompt=True
            )
        t_enc = t_tok(t_text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            t_out = teacher.generate(
                **t_enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=not args.greedy,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=t_tok.pad_token_id,
                eos_token_id=t_tok.eos_token_id,
            )
        t_new = t_out[0][t_enc["input_ids"].shape[1]:]
        tr_answer = strip_think(t_tok.decode(t_new, skip_special_tokens=True))
        print("-" * 66)
        print(f"教师: {tr_answer}")
        print(f"      [{int(t_new.numel())} tok]")
    print("=" * 66)

    if args.logprobs and "token_logprobs" in r:
        print("\n逐 token 概率:")
        for t in r["token_logprobs"]:
            import math
            print(f"  {t['token']!r:14s} logprob {t['logprob']:8.4f}  "
                  f"p = {math.exp(t['logprob']):.4f}")


if __name__ == "__main__":
    main()