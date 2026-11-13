# -*- coding: utf-8 -*-
"""
dspark_train.py —— DSpark 草稿模块训练（target 全程冻结）

对齐 DeepSeek-V4.1 §2.4.3 的训练设定：

    "we train only DSpark while keeping the backbone frozen"

三项损失（论文原文）：

    L_ce    标准语言建模交叉熵，教草稿预测下一个 token
    L_tv    草稿分布与 target 分布的总变差距离，把 q 拉向 p
    L_conf  置信度头的二元交叉熵，标签是软接受度 c* = 1 - 0.5·||p_d - q_d||_1

后训练阶段（SFT/RL 之后）的用法在本文件里同等适用：把 --target-ckpt 换成对齐后的
checkpoint 再跑一次，草稿就会重新对齐到漂移后的策略。这正是论文强调的
"keeps DSpark aligned with the evolving policy"。

用法
----
    # 冒烟测试（随机权重，不需要数据）
    python model/draft/dspark_train.py --smoke

    # 真实训练
    python model/draft/dspark_train.py \
        --target-ckpt checkpoints/full_sft_512.pth \
        --data dataset/sft_t2t_mini.jsonl \
        --steps 2000 --batch-size 2 --seq-len 256 \
        --out checkpoints/dspark_512.pth
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time
from typing import List, Optional, Tuple

def _find_repo_root(start=None):
    """向上定位仓库根目录（含 model/ 与 trainer/ 的那一层）。

    用标志目录而不是固定层数：本文件曾被移动过目录层级，硬编码 dirname 次数
    会在下次挪动时静默指错路径，进而把 ckpt / dataset 全解析到错误位置。
    """
    p = pathlib.Path(start or __file__).resolve().parent
    for cand in [p, *p.parents]:
        if (cand / "model").is_dir() and (cand / "trainer").is_dir():
            return str(cand)
    return str(p.parents[1])


_REPO_ROOT = _find_repo_root()
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F

from model.XLLMModel import XLLMConfig, XLLMForCausalLM
from model.draft.dspark.dspark_draft import DSparkConfig, DSparkDrafter


# ----------------------------------------------------------------------
# 权重加载
# ----------------------------------------------------------------------
def load_target(ckpt: Optional[str], hidden_size: int, num_layers: int,
                vocab_size: int, use_moe: bool, device) -> XLLMForCausalLM:
    cfg = XLLMConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=8,
        num_key_value_heads=2,
        vocab_size=vocab_size,
        max_position_embeddings=1024,
        use_moe=use_moe,
    )
    model = XLLMForCausalLM(cfg).to(device).eval()
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        # 项目历史 checkpoint 存在 post_attention_layernotm 这类拼写缺陷，
        # 这里做一次键名修正，否则 8 层归一化权重会被静默漏加载。
        fixed = {}
        for k, v in state.items():
            k2 = k.replace("post_attention_layernotm", "post_attention_layernorm")
            fixed[k2] = v
        missing, unexpected = model.load_state_dict(fixed, strict=False)
        print(f"[target] 加载 {ckpt}; missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("[target] 未加载 checkpoint，使用随机权重（仅用于跑通链路）")
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ----------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------
def iter_token_batches(path: Optional[str], tokenizer, batch_size: int, seq_len: int,
                       vocab_size: int, device, steps: int):
    """
    产出 [B, seq_len] 的 token 批次。

    支持 .jsonl（每行 {"text": ...} 或 {"chosen": ...}）。无数据时（冒烟测试）
    退化为随机 token，仅验证张量形状与损失可算通。
    """
    if not path or not os.path.exists(path):
        for _ in range(steps):
            yield torch.randint(3, vocab_size, (batch_size, seq_len), device=device)
        return

    buf: List[int] = []
    while True:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text") or obj.get("chosen") or obj.get("content") or ""
                if not text and "conversations" in obj:
                    # 本项目 sft/dpo 语料格式：conversations = [{role, content}, ...]
                    parts = []
                    for turn in obj["conversations"]:
                        if isinstance(turn, dict):
                            c = turn.get("content", "")
                            if isinstance(c, str) and c:
                                parts.append(c)
                    text = "\n".join(parts)
                if not isinstance(text, str) or not text:
                    continue
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                buf.extend(ids)
                while len(buf) >= batch_size * seq_len:
                    chunk = buf[: batch_size * seq_len]
                    buf = buf[batch_size * seq_len:]
                    yield torch.tensor(chunk, device=device).view(batch_size, seq_len)
        # 语料读完后循环（训练步数由外部控制）
        if not buf:
            continue


# ----------------------------------------------------------------------
# 单步损失
# ----------------------------------------------------------------------
def compute_losses(
    target: XLLMForCausalLM,
    drafter: DSparkDrafter,
    input_ids: torch.Tensor,
    n_draft: int,
    num_prefix_samples: int,
    tv_weight: float,
    ce_weight: float,
    conf_weight: float,
) -> Tuple[torch.Tensor, dict]:
    """
    在一条序列上采样多个前缀位置，对每个位置构造 n_draft 步草稿任务。

    位置约定：前缀结束于 p（含），草稿目标是 [p+1 .. p+n]，
    第 d 个草稿位置的条件 token 是 seq[p+d]（teacher forcing）。
    """
    B, L = input_ids.shape
    device = input_ids.device

    with torch.no_grad():
        out = target(input_ids=input_ids, use_cache=False, logits_to_keep=L)
        tgt_hidden = out.hidden_states            # [B,L,H]
        tgt_logits = out.logits                   # [B,L,V]

    if L <= n_draft + 2:
        raise ValueError("seq_len 太短，无法构造草稿任务")

    max_p = L - n_draft - 1
    ps = torch.randint(0, max_p, (num_prefix_samples,), device=device)

    total_ce = input_ids.new_zeros((), dtype=torch.float32)
    total_tv = input_ids.new_zeros((), dtype=torch.float32)
    total_conf = input_ids.new_zeros((), dtype=torch.float32)

    for p in ps.tolist():
        prefix_hidden = tgt_hidden[:, : p + 1, :]              # [B,p+1,H]
        teacher = input_ids[:, p + 1: p + 1 + n_draft]          # [B,n] 目标
        prev = input_ids[:, p: p + n_draft]                     # [B,n] 条件 token

        logits, base_logits, conf_logits, _ = drafter.forward_teacher(
            prefix_hidden=prefix_hidden,
            start_pos=0,
            prev_tokens=prev,
            lm_head=target.lm_head,
        )                                                       # [B,n,V]

        # ---- CE：草稿预测真实下几个 token ----
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            teacher.reshape(-1),
            reduction="mean",
        )
        total_ce = total_ce + ce

        # ---- TV：把草稿分布拉向 target 分布 ----
        q = F.softmax(logits.float(), dim=-1)
        with torch.no_grad():
            # target 在位置 p+d 对 token p+d+1 的分布：取 tgt_logits[:, p+d]
            t_pos = tgt_logits[:, p: p + n_draft, :].float()
            p_dist = F.softmax(t_pos, dim=-1)
        tv = 0.5 * (p_dist - q).abs().sum(dim=-1).mean()
        total_tv = total_tv + tv

        # ---- 置信度 BCE：软标签 c* = 1 - 0.5||p - q||_1 ----
        with torch.no_grad():
            soft_label = 1.0 - 0.5 * (p_dist - q).abs().sum(dim=-1)   # [B,n] in [0,1]
        bce = F.binary_cross_entropy_with_logits(
            conf_logits.float(), soft_label.float(), reduction="mean"
        )
        total_conf = total_conf + bce

    n = float(num_prefix_samples)
    ce, tv, conf = total_ce / n, total_tv / n, total_conf / n
    loss = ce_weight * ce + tv_weight * tv + conf_weight * conf
    return loss, {"ce": ce.item(), "tv": tv.item(), "conf": conf.item(), "loss": loss.item()}


# ----------------------------------------------------------------------
# 校准：在 held-out 数据上估计逐位置真实接受率，写入 conf_scale/bias
# ----------------------------------------------------------------------
@torch.no_grad()
def calibrate_confidence(target, drafter: XLLMForCausalLM, batches, n_draft: int) -> torch.Tensor:
    """返回 [n_draft] 的逐位置平均软接受度。"""
    acc = torch.zeros(n_draft)
    cnt = 0
    for input_ids in batches:
        B, L = input_ids.shape
        out = target(input_ids=input_ids, use_cache=False, logits_to_keep=L)
        h, tl = out.hidden_states, out.logits
        p = max(0, L - n_draft - 1)
        prefix_hidden = h[:, : p + 1, :]
        prev = input_ids[:, p: p + n_draft]
        logits, _, _, _ = drafter.forward_teacher(prefix_hidden, 0, prev, target.lm_head)
        q = F.softmax(logits.float(), dim=-1)
        pd = F.softmax(tl[:, p: p + n_draft, :].float(), dim=-1)
        soft = 1.0 - 0.5 * (pd - q).abs().sum(dim=-1)
        acc += soft.mean(dim=0).cpu()
        cnt += 1
    return acc / max(cnt, 1)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="DSpark 草稿模块训练")
    ap.add_argument("--target-ckpt", type=str, default=None)
    ap.add_argument("--data", type=str, default=None, help="jsonl 语料路径")
    ap.add_argument("--tokenizer", type=str, default=os.path.join(_REPO_ROOT, "model"))
    ap.add_argument("--hidden-size", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=8)
    ap.add_argument("--vocab-size", type=int, default=6400)
    ap.add_argument("--use-moe", action="store_true")
    ap.add_argument("--draft-steps", type=int, default=5, help="一次前向的草稿宽度")
    ap.add_argument("--draft-layers", type=int, default=3)
    ap.add_argument("--draft-window", type=int, default=128)
    ap.add_argument("--markov-rank", type=int, default=64)
    ap.add_argument("--prefix-samples", type=int, default=4)
    ap.add_argument("--ce-weight", type=float, default=1.0)
    ap.add_argument("--tv-weight", type=float, default=1.0)
    ap.add_argument("--conf-weight", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--calib-batches", type=int, default=8)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--smoke", action="store_true", help="小模型随机权重快速跑通")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    if args.smoke:
        args.hidden_size, args.num_layers = 128, 4
        args.vocab_size, args.seq_len = 512, 64
        args.draft_steps, args.batch_size, args.steps = 4, 2, 20
        args.data = None
        args.draft_window = 32

    target = load_target(args.target_ckpt, args.hidden_size, args.num_layers,
                         args.vocab_size, args.use_moe, device)

    cfg = DSparkConfig.from_target(
        target.config,
        num_layers=args.draft_layers,
        window=args.draft_window,
        max_draft=args.draft_steps,
        markov_rank=args.markov_rank,
    )
    drafter = DSparkDrafter(cfg).to(device)
    print(f"[draft] 参数量 {drafter.num_parameters()/1e6:.2f}M "
          f"(layers={cfg.num_layers}, d_model={cfg.d_model}, max_draft={cfg.max_draft})")

    tokenizer = None
    if args.data and os.path.exists(args.data):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        print(f"[data] {args.data}")

    gen = iter_token_batches(args.data, tokenizer, args.batch_size, args.seq_len,
                             args.vocab_size, device, args.steps)

    opt = torch.optim.AdamW([p for p in drafter.parameters() if p.requires_grad], lr=args.lr)
    t_start = time.time()

    drafter.train()
    for step in range(1, args.steps + 1):
        input_ids = next(gen)
        loss, parts = compute_losses(
            target, drafter, input_ids, args.draft_steps, args.prefix_samples,
            args.tv_weight, args.ce_weight, args.conf_weight,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(drafter.parameters(), 1.0)
        opt.step()

        if step % max(1, args.steps // 10) == 0 or step == 1:
            print(f"[{step}/{args.steps}] loss={parts['loss']:.4f} "
                  f"ce={parts['ce']:.4f} tv={parts['tv']:.4f} conf={parts['conf']:.4f} "
                  f"({time.time()-t_start:.1f}s)")

    # ---- 校准 ----
    drafter.eval()
    print("[calib] 在 held-out 批次上拟合置信度头...")
    calib_gen = iter_token_batches(args.data, tokenizer, args.batch_size, args.seq_len,
                                   args.vocab_size, device, args.calib_batches)
    calib_batches = [next(calib_gen) for _ in range(args.calib_batches)]
    acc_rates = calibrate_confidence(target, drafter, calib_batches, args.draft_steps)
    drafter.calibrate(acc_rates)
    print(f"[calib] 逐位置平均软接受度: {[round(v,3) for v in acc_rates.tolist()]}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        torch.save({
            "drafter": drafter.state_dict(),
            "cfg": cfg.__dict__,
            "hidden_size": args.hidden_size,
            "vocab_size": args.vocab_size,
            "calibrated": True,
        }, args.out)
        print(f"[save] {args.out}")

    print("DONE")


if __name__ == "__main__":
    main()