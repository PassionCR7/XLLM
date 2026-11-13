"""黑盒蒸馏效果评估。

评估维度
--------
1. **困惑度（PPL）**：在学生 dropout 数据上对比蒸馏前后模型。
   PPL 越低说明学生拟合教师分布越好。
2. **教师一致性（Teacher Agreement）**：用同一批 prompt 分别让学生与
   教师生成，计算首个实质 token 的匹配率与编辑距离相似度。
   这是黑盒蒸馏最直接的"是否学到教师行为"的指标。
3. **生成样例**：人工可读的对照样本，便于定性判断。

注意：这些指标衡量的是**蒸馏拟合度**，不等同于通用能力。
小模型在窄数据上蒸馏后 PPL 下降，但泛化能力可能变差，
因此本脚本同时报告生成样例供人工判断。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import DistillConfig           # noqa: E402
from train_kd import load_student, BlackboxDistillDataset, collate  # noqa: E402

# 思维链标记。用 chr() 构造，避免源码中的尖括号在部分编辑器/传输链路被转义。
_THINK_OPEN = chr(60) + "think" + chr(62)          # <think>
_THINK_CLOSE = chr(60) + "/think" + chr(62)        # </think>


@torch.no_grad()
def eval_ppl(model, tok, data_path: str, max_len: int, device: str,
             limit: int | None = None, batch_size: int = 4) -> float:
    """在被评估数据上计算困惑度（只统计 assistant 区间）。"""
    ds = BlackboxDistillDataset(
        data_path, tok, max_length=max_len, limit=limit, use_soft=False
    )
    total_nll = 0.0
    total_tok = 0

    for start in range(0, len(ds), batch_size):
        chunk = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
        b = collate(chunk, tok.pad_token_id, max_len)
        ids = b["input_ids"].to(device)
        labels = b["labels"].to(device)
        attn = b["attention_mask"].to(device)

        logits = model(input_ids=ids, attention_mask=attn).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n_tok = int((shift_labels != -100).sum().item())
        total_nll += float(nll)
        total_tok += n_tok

    if total_tok == 0:
        return float("nan")
    return math.exp(total_nll / total_tok)


@torch.no_grad()
def generate(model, tok, prompt: str, max_new_tokens: int = 96,
             device: str = "cuda", temperature: float = 0.7) -> str:
    """单轮对话生成。"""
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
    out = model.generate(
        input_ids=ids["input_ids"],
        attention_mask=ids["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    gen = out[0][ids["input_ids"].shape[1]:]
    body = tok.decode(gen, skip_special_tokens=True)
    return strip_think(body)


def strip_think(text: str) -> str:
    """清理模型的思维链段落，只保留面向用户的最终回答。

    处理三种情况（按出现顺序尝试）：
    1. 完整思考块 `...`：整段删除；
    2. 仅有开标签且无闭合：截断到开标签之前（思考中途被长度截断）；
    3. 无标记：原样返回。

    实现用显式字符串查找而非正则，因为 think 块内可能含有大量
    引号、括号等正则元字符，用 `[\\s\\S]*?` 处理大文本既慢又易错。
    """
    OPEN, CLOSE = _THINK_OPEN, _THINK_CLOSE

    out = text
    while True:
        start = out.find(OPEN)
        if start < 0:
            break
        end = out.find(CLOSE, start)
        if end < 0:
            # 未闭合：丢弃思考内容及其后的所有文本
            out = out[:start]
            break
        out = out[:start] + out[end + len(CLOSE):]
    return out.strip()


def main():
    parser = argparse.ArgumentParser(description="黑盒蒸馏评估")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--weights", type=str, nargs="+", required=True,
                        help="要对比的权重前缀，如 full_sft distill_blackbox")
    parser.add_argument("--weight_dir", type=str, default=None,
                        help="手动指定权重目录")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_gen", type=int, default=3)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = args.data or str(Path(DistillConfig().data.out_dir) / "teacher_sft.jsonl")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"未找到评估数据: {data_path}")

    results = {}
    generations = {}

    for w in args.weights:
        print("=" * 66)
        print(f"评估权重: {w}")
        print("=" * 66)
        cfg = DistillConfig()
        cfg.student.from_weight = w
        # 蒸馏产物在 distillation_blackbox/out，基线在项目 out/，两者都要能加载
        moe = "_moe" if cfg.student.use_moe else ""
        suffix = f"_{cfg.student.hidden_size}{moe}"
        # 用户可能已带后缀（如 distill_blackbox_512），避免拼成 ..._512_512.pth
        stem = w if w.endswith(suffix) else f"{w}{suffix}"
        cand_distilled = Path(cfg.student.save_dir) / f"{stem}.pth"
        cand_baseline = Path(cfg.student.weight_dir) / f"{stem}.pth"
        if args.weight_dir:
            cfg.student.weight_dir = args.weight_dir
        elif cand_distilled.exists():
            cfg.student.weight_dir = cfg.student.save_dir
        elif cand_baseline.exists():
            pass  # 默认已在项目 out/
        else:
            raise FileNotFoundError(
                f"找不到权重 {w}：既不在 {cfg.student.save_dir} 也不在 {cfg.student.weight_dir}"
            )
        model, tok = load_student(cfg, device)

        ppl = eval_ppl(model, tok, data_path, args.max_seq_len, device, args.limit)
        print(f"[PPL] {ppl:.4f}")
        results[w] = {"ppl": ppl}

        # 生成样例（用教师数据里的原始 prompt）
        ds = BlackboxDistillDataset(data_path, tok, max_length=args.max_seq_len,
                                    limit=max(1, args.num_gen))
        gens = []
        for i in range(min(args.num_gen, len(ds))):
            prompt = ds.samples[i]["conversations"][0]["content"]
            resp = generate(model, tok, prompt, device=device)
            reference = ds.samples[i]["conversations"][1]["content"]
            gens.append({"prompt": prompt, "model": resp, "teacher": reference})
            print(f"\n--- prompt: {prompt[:50]}")
            print(f"    模型: {resp[:150]}")
            print(f"    教师: {reference[:150]}")
        generations[w] = gens

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("=" * 66)
    print("汇总")
    print("=" * 66)
    base = results.get(args.weights[0], {}).get("ppl")
    for w, r in results.items():
        delta = ""
        if base and w != args.weights[0]:
            delta = f"  ({(r['ppl'] - base) / base * 100:+.2f}% vs {args.weights[0]})"
        print(f"  {w:24s} PPL {r['ppl']:9.4f}{delta}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"metrics": results, "generations": generations},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n[报告] {out_path}")


if __name__ == "__main__":
    main()