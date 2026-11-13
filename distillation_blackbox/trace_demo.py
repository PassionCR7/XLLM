"""对单条样本逐位置追踪 CE 与 KL 的计算过程（黑盒）。

只用于讲解，不参与训练。输出每个位置的真实数值。
"""

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from config import DistillConfig
from train_kd import (
    BlackboxDistillDataset,
    collate,
    load_student,
    sparse_soft_kd_loss,
    IGNORE_INDEX,
)


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 81
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = DistillConfig()
    cfg.student.max_seq_len = 512
    model, tok = load_student(cfg, device)
    model.eval()

    data_path = str(Path(cfg.data.out_dir) / "teacher_sft.jsonl")
    ds = BlackboxDistillDataset(data_path, tok, max_length=512, limit=idx + 1, use_soft=True)
    sample = ds[idx]

    print("=" * 78)
    print("【第 1 步】学生侧编码结果")
    print("=" * 78)
    ids = sample["input_ids"]
    labels = sample["labels"]
    print(f"序列长度 L = {len(ids)}")
    valid = [i for i, l in enumerate(labels) if l != IGNORE_INDEX]
    print(f"参与 CE 的 token 数 = {len(valid)}  (区间 {valid[0]}..{valid[-1]})")

    print("\n前 12 个 token（看 prompt 部分 labels 全为 -100）:")
    for i in range(min(12, len(ids))):
        piece = tok.decode([ids[i]])
        lab = "IGNORE" if labels[i] == IGNORE_INDEX else str(labels[i])
        print(f"  pos {i:3d}  id {ids[i]:5d}  label {lab:>7s}  {piece!r}")

    print("\nassistant 区间前 10 个 token（labels 有值，参与 CE）:")
    for i in valid[:10]:
        piece = tok.decode([ids[i]])
        print(f"  pos {i:3d}  id {ids[i]:5d}  label {labels[i]:5d}  {piece!r}")

    print("\n" + "=" * 78)
    print("【第 2 步】软标签：教师 top-5 -> 学生 id 的映射结果")
    print("=" * 78)
    soft_ids = sample["soft_ids"]
    soft_probs = sample["soft_probs"]
    soft_bg = sample["soft_bg"]
    n_soft = 0
    for i in valid:
        if any(s >= 0 for s in soft_ids[i]):
            n_soft += 1
    print(f"成功挂上软标签的位置数 = {n_soft} / {len(valid)} "
          f"({100*n_soft/max(1,len(valid)):.1f}%)")

    print("\n前 8 个有软标签的位置（展示概率构成）:")
    shown = 0
    for i in valid:
        if not any(s >= 0 for s in soft_ids[i]):
            continue
        toks = [(soft_ids[i][k], soft_probs[i][k]) for k in range(5) if soft_ids[i][k] >= 0]
        toks = [(tid, p) for tid, p in toks]
        ssum = sum(p for _, p in toks)
        readable = [(tok.decode([tid]), round(p, 4)) for tid, p in toks]
        print(f"  pos {i:3d}  预测目标={tok.decode([labels[i]])!r}")
        print(f"          教师top-k: {readable}")
        print(f"          Σp_topk = {ssum:.4f}   p_bg = {soft_bg[i]:.4f}")
        shown += 1
        if shown >= 8:
            break

    print("\n" + "=" * 78)
    print("【第 3 步】前向，取学生 logits，并做一位位移")
    print("=" * 78)
    b = collate([sample], tok.pad_token_id, 512)
    with torch.no_grad():
        out = model(
            input_ids=b["input_ids"].to(device),
            attention_mask=b["attention_mask"].to(device),
        )
    logits = out.logits                      # [1, L, V]
    print(f"学生 logits 形状 = {tuple(logits.shape)}  (V = 学生词表 {logits.size(-1)})")

    shift_logits = logits[..., :-1, :].contiguous()   # [1, L-1, V]
    shift_labels = b["labels"][..., 1:].contiguous().to(device)  # [1, L-1]
    print(f"位移后 shift_logits = {tuple(shift_logits.shape)}")
    print("含义：位置 i 的 logits 预测 token[i+1]，所以配 label[i+1]")

    print("\n" + "=" * 78)
    print("【第 4 步】CE 计算（逐位置 -> 平均）")
    print("=" * 78)
    flat_logits = shift_logits.view(-1, shift_logits.size(-1)).float()
    flat_labels = shift_labels.view(-1)
    per_tok = F.cross_entropy(flat_logits, flat_labels, ignore_index=IGNORE_INDEX,
                              reduction="none")
    mask_ce = flat_labels != IGNORE_INDEX
    nll_vals = per_tok[mask_ce]

    print(f"参与计算的 token 数 = {nll_vals.numel()}")
    print("\n前 12 个位置的 NLL 明细:")
    print(f"  {'位置':>6} {'目标token':>10} {'教师lp':>9} {'学生lp':>9} {'NLL':>8} {'学生p':>8}")
    # 教师 lp 与该位置对齐：教师 token 与 label 可能不同粒度，仅作参考
    k = 0
    for pos in range(shift_logits.size(1)):
        if flat_labels[pos] == IGNORE_INDEX:
            continue
        tid = int(flat_labels[pos])
        lp_s = float(F.log_softmax(flat_logits[pos], dim=-1)[tid])
        print(f"  {pos:6d} {tok.decode([tid])!r:>10} {'-':>9} {lp_s:9.4f} "
              f"{-lp_s:8.4f} {math.exp(lp_s):8.4f}")
        k += 1
        if k >= 12:
            break

    ce = F.cross_entropy(flat_logits, flat_labels, ignore_index=IGNORE_INDEX)
    print(f"\nCE = mean(NLL) = {float(ce):.4f}")
    print(f"对应 PPL = exp(CE) = {math.exp(float(ce)):.4f}")

    print("\n" + "=" * 78)
    print("【第 5 步】KL 计算（稀疏 + 背景兜底）")
    print("=" * 78)
    kd = sparse_soft_kd_loss(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        b["soft_ids"][:, 1:, :].reshape(-1, 5).to(device),
        b["soft_probs"][:, 1:, :].reshape(-1, 5).to(device),
        b["soft_bg"][:, 1:].reshape(-1).to(device),
        mask_ce.to(device),
        temperature=cfg.train.temperature,
    )
    print(f"KL（已乘 T^2，T={cfg.train.temperature}）= {float(kd):.4f}")

    print("\n逐位置 KL 明细（前 8 个有效位置）:")
    print(f"  {'位置':>6} {'Σp':>7} {'p_bg':>7} {'q_bg':>7} {'KL':>9}")
    sids = b["soft_ids"][:, 1:, :].reshape(-1, 5).to(device)
    sp = b["soft_probs"][:, 1:, :].reshape(-1, 5).to(device)
    sbg = b["soft_bg"][:, 1:].reshape(-1).to(device)
    log_q_all = F.log_softmax(flat_logits, dim=-1)
    k2 = 0
    for pos in range(shift_logits.size(1)):
        if not mask_ce[pos]:
            continue
        if not (sids[pos] >= 0).any():
            continue
        validk = (sids[pos] >= 0).float()
        qk = log_q_all[pos].gather(0, sids[pos].clamp(min=0))
        qsum = float((qk.exp() * validk).sum())
        psum = float(sp[pos].sum())
        bg = float(sbg[pos])
        T = cfg.train.temperature
        # 教师侧 T 变换
        lp_k = torch.log(sp[pos].clamp(min=1e-12)) / T
        lp_bg = torch.log(sbg[pos].clamp(min=1e-12)) / T
        allp = torch.cat([lp_k, lp_bg.unsqueeze(0)])
        pn = F.log_softmax(allp, dim=-1)
        klv = float((pn[:5].exp() * (pn[:5] - qk)).sum() + pn[5].exp() * (pn[5] - math.log(max(1e-12, 1 - qsum))))
        print(f"  {pos:6d} {psum:7.4f} {bg:7.4f} {1-qsum:7.4f} {klv:9.4f}")
        k2 += 1
        if k2 >= 8:
            break

    print("\n" + "=" * 78)
    print("【第 6 步】总损失")
    print("=" * 78)
    total = cfg.train.ce_weight * ce + cfg.train.kd_weight * kd
    print(f"L = {cfg.train.ce_weight} × CE + {cfg.train.kd_weight} × KL")
    print(f"  = {cfg.train.ce_weight} × {float(ce):.4f} + {cfg.train.kd_weight} × {float(kd):.4f}")
    print(f"  = {float(total):.4f}")
    print(f"\n梯度回传 -> 只更新学生参数（教师在此目录是 API，本就不在图中）")


if __name__ == "__main__":
    main()