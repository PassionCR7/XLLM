"""白盒蒸馏效果评估。

相比黑盒，白盒能直接衡量**分布拟合度**，这是最有说服力的指标：

1. **KL 散度**：学生分布与教师分布（对齐子集上）的平均 KL。
   蒸馏的目标就是最小化它，因此它是最直接的训练目标指标。
2. **Top-1 一致率**：学生 argmax 是否等于教师 argmax（仅对齐位置）。
   注意跨分词器场景下该值天然偏低（粒度不同），仅作参考。
3. **PPL**：学生自身在数据上的困惑度，衡量语言建模能力是否保持。
4. **生成样例**：定性对照。

重要说明
--------
跨分词器蒸馏中，Top-1 一致率不可能达到很高：
学生 tokenizer 可能把 "，请" 合成一个 token，而教师倾向拆成 "，" + "请"。
两者在同一字符前缀下的 argmax 落在不同粒度上，属于**正常现象**。
真正的判据是 KL 是否下降、PPL 是否改善。
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

from config import WhiteboxConfig                        # noqa: E402
from dataset import DistillDataset                       # noqa: E402
from student_loader import load_student                  # noqa: E402
from teacher_logits import extract_aligned_logits        # noqa: E402
from vocab_align import (                                # noqa: E402
    load_vocab_from_tokenizer_json,
    build_alignment,
)


def build_teacher(cfg: WhiteboxConfig, device: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    p = cfg.teacher.model_path
    tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, cfg.teacher.dtype)
    m = AutoModelForCausalLM.from_pretrained(p, dtype=dtype, trust_remote_code=True)
    m = m.to(device).eval()
    for x in m.parameters():
        x.requires_grad_(False)
    return m, tok


@torch.no_grad()
def evaluate(
    student,
    s_tok,
    teacher,
    t_tok,
    align,
    data_path: str,
    max_len: int,
    device: str,
    limit: int | None = None,
    batch_size: int = 2,
    temperature: float = 1.0,
) -> dict:
    ds = DistillDataset(data_path, s_tok, max_length=max_len, limit=limit)

    total_kl = 0.0
    total_pos = 0
    total_nll = 0.0
    total_tok = 0
    top1_hit = 0

    for start in range(0, len(ds), batch_size):
        idxs = list(range(start, min(start + batch_size, len(ds))))
        items = [ds[i] for i in idxs]

        # 学生侧补齐
        B = len(items)
        input_ids = torch.full((B, max_len), s_tok.pad_token_id, dtype=torch.long)
        labels = torch.full((B, max_len), -100, dtype=torch.long)
        attn = torch.zeros((B, max_len), dtype=torch.long)
        for b, it in enumerate(items):
            n = min(len(it["input_ids"]), max_len)
            input_ids[b, :n] = torch.tensor(it["input_ids"][:n])
            labels[b, :n] = torch.tensor(it["labels"][:n])
            attn[b, :n] = 1

        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attn = attn.to(device)

        s_logits = student(input_ids=input_ids, attention_mask=attn).logits

        # 教师对齐
        texts = [it["text"] for it in items]
        offsets = [
            s_tok(t, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            for t in texts
        ]
        t_al = extract_aligned_logits(
            teacher, t_tok, texts, offsets, max_len, align,
            device=device, max_teacher_len=max_len * 2, dtype=torch.bfloat16,
        )

        s_shift = s_logits[:, :-1, :][:, :, t_al.shrink_ids].float()
        t_shift = t_al.logits[:, :-1, :].float()
        # mask 必须与 t_shift 同步用 [:-1]，否则错位一位，
        # 会把教师 logits 未对齐（全零）的位置也计入 KL。
        mask = t_al.mask[:, :-1].float()

        log_pt = F.log_softmax(t_shift / temperature, dim=-1)
        log_ps = F.log_softmax(s_shift / temperature, dim=-1)
        kl = (log_pt.exp() * (log_pt - log_ps)).sum(dim=-1)
        total_kl += float((kl * mask).sum())
        total_pos += int(mask.sum())

        # top-1 一致率
        agree = (t_shift.argmax(-1) == s_shift.argmax(-1)).float()
        top1_hit += float((agree * mask).sum())

        # PPL
        sh_logits = s_logits[..., :-1, :].contiguous()
        sh_labels = labels[..., 1:].contiguous()
        nll = F.cross_entropy(
            sh_logits.view(-1, sh_logits.size(-1)).float(),
            sh_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_nll += float(nll)
        total_tok += int((sh_labels != -100).sum())

    avg_kl = total_kl / max(total_pos, 1)
    return {
        "kl": avg_kl,
        "top1_agreement": top1_hit / max(total_pos, 1),
        "ppl": math.exp(total_nll / total_tok) if total_tok else float("nan"),
        "aligned_positions": total_pos,
        "samples": len(ds),
    }


@torch.no_grad()
def generate(student, tok, prompt: str, device: str, max_new_tokens: int = 96) -> str:
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
    out = student.generate(
        input_ids=ids["input_ids"],
        attention_mask=ids["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    body = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return strip_think(body)


# 思维链标记，用 chr() 构造避免源码尖括号被转义
_THINK_OPEN = chr(60) + "think" + chr(62)
_THINK_CLOSE = chr(60) + "/think" + chr(62)


def strip_think(text: str) -> str:
    """清理思维链段落，只保留面向用户的最终回答。"""
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


def main():
    parser = argparse.ArgumentParser(description="白盒蒸馏评估")
    parser.add_argument("--weights", type=str, nargs="+", required=True)
    parser.add_argument("--weight_dir", type=str, default=None,
                        help="权重目录，默认取 config 的 save_dir（蒸馏产物）")
    parser.add_argument("--from_dir", type=str, default=None,
                        help="若指定，则从项目 out/ 目录加载基线权重")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=192)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--num_gen", type=int, default=2)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = WhiteboxConfig()
    data_path = args.data or cfg.data.seed_file

    print("加载教师...")
    teacher, t_tok = build_teacher(cfg, device)
    t_vocab = load_vocab_from_tokenizer_json(
        os.path.join(cfg.teacher.model_path, "tokenizer.json")
    )
    s_vocab = load_vocab_from_tokenizer_json(
        os.path.join(cfg.student.model_dir, "tokenizer.json")
    )
    align = build_alignment(t_vocab, s_vocab)

    results, gens = {}, {}
    for w in args.weights:
        print("=" * 66)
        print(f"评估权重: {w}")
        print("=" * 66)
        c = WhiteboxConfig()
        c.student.from_weight = w
        # 权重路径解析：蒸馏产物在 distillation_whitebox/out，
        # 基线权重在项目 out/。两种都要能加载。
        base_cfg = WhiteboxConfig()
        distilled_dir = c.student.save_dir
        baseline_dir = base_cfg.student.weight_dir
        moe = "_moe" if c.student.use_moe else ""
        suffix = f"_{c.student.hidden_size}{moe}"
        # 用户可能已带后缀（如 distill_whitebox_512），避免拼成 ..._512_512.pth
        stem = w if w.endswith(suffix) else f"{w}{suffix}"
        cand_distilled = Path(distilled_dir) / f"{stem}.pth"
        cand_baseline = Path(baseline_dir) / f"{stem}.pth"
        if args.weight_dir:
            c.student.weight_dir = args.weight_dir
        elif cand_distilled.exists():
            c.student.weight_dir = distilled_dir
        elif cand_baseline.exists():
            c.student.weight_dir = baseline_dir
        student, s_tok = load_student(c.student, device)

        r = evaluate(
            student, s_tok, teacher, t_tok, align, data_path,
            args.max_seq_len, device, limit=args.limit,
            temperature=args.temperature,
        )
        results[w] = r
        print(f"  KL                : {r['kl']:.4f}")
        print(f"  Top-1 一致率      : {r['top1_agreement']*100:.2f}%")
        print(f"  PPL               : {r['ppl']:.4f}")
        print(f"  对齐位置数        : {r['aligned_positions']}")

        ds = DistillDataset(data_path, s_tok, max_length=args.max_seq_len,
                            limit=max(1, args.num_gen))
        samples = []
        for i in range(min(args.num_gen, len(ds))):
            conv = ds.samples[i]["conversations"]
            prompt = next(m["content"] for m in conv if m["role"] == "user")
            resp = generate(student, s_tok, prompt, device)
            samples.append({"prompt": prompt, "model": resp})
            print(f"\n  --- {prompt[:46]}")
            print(f"      模型: {resp[:130]}")
        gens[w] = samples

        del student
        if device == "cuda":
            torch.cuda.empty_cache()

    print("=" * 66)
    print("汇总（KL 越低越好）")
    print("=" * 66)
    for w, r in results.items():
        print(f"  {w:26s} KL {r['kl']:8.4f}  Top1 {r['top1_agreement']*100:5.1f}%  "
              f"PPL {r['ppl']:8.4f}")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"metrics": results, "generations": gens},
                                ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[报告] {p}")


if __name__ == "__main__":
    main()