"""对单条样本逐位置追踪 CE 与 KL 的计算过程（白盒）。

只用于讲解，不参与训练。
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
# 顺序要紧：先插父目录（为了 model.XLLMModel），再把本地目录插到最前面，
# 否则本地 dataset.py 会被根目录的 dataset/ 包遮蔽。
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from config import WhiteboxConfig
from dataset import DistillDataset
from student_loader import load_student
from vocab_align import load_vocab_from_tokenizer_json, build_alignment
from teacher_logits import extract_aligned_logits
from train_kd import build_teacher

IGNORE = -100


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = WhiteboxConfig()
    L = cfg.student.max_seq_len

    student, s_tok = load_student(cfg.student, device=device)
    student.eval()
    teacher, t_tok = build_teacher(cfg, device)

    t_vocab = load_vocab_from_tokenizer_json(
        os.path.join(cfg.teacher.model_path, "tokenizer.json"))
    s_vocab = load_vocab_from_tokenizer_json(
        os.path.join(cfg.student.model_dir, "tokenizer.json"))
    align = build_alignment(t_vocab, s_vocab, verbose=False)
    print(f"[对齐] 学生覆盖率 {align.coverage*100:.2f}%  "
          f"公共列数 {int((align.student_covered>=0).sum())}")

    ds = DistillDataset(cfg.data.seed_file, s_tok, max_length=L, limit=idx + 1)
    sample = ds[idx]
    ids, labels, text = sample["input_ids"], sample["labels"], sample["text"]

    print("=" * 78)
    print("【第 1 步】学生侧编码")
    print("=" * 78)
    valid = [i for i, l in enumerate(labels) if l != IGNORE]
    print(f"学生序列长度 L = {len(ids)}（已截断到 max_seq_len={L}）")
    print(f"assistant 区间 = [{valid[0]}, {valid[-1]}]，共 {len(valid)} 个 token")
    print(f"\n回答原文前 60 字: {text[valid[0]:valid[0]+60]!r}")

    enc = s_tok(text, add_special_tokens=False, return_offsets_mapping=True)
    s_off = enc["offset_mapping"]
    print("\nassistant 前 10 个学生 token（含字符区间）:")
    for i in valid[:10]:
        s, e = s_off[i]
        print(f"  pos {i:3d}  chars[{s:4d},{e:4d})  {s_tok.decode([ids[i]])!r}")

    print("\n" + "=" * 78)
    print("【第 2 步】教师侧编码 + 位置对齐")
    print("=" * 78)
    enc_t = t_tok(text, add_special_tokens=False, return_offsets_mapping=True,
                  truncation=True, max_length=L * 2)
    t_off = enc_t["offset_mapping"]
    print(f"教师 token 数 = {len(t_off)}（同一段文本，教师切得更碎）")
    print("\n教师前 10 个 token（对应同一段字符）:")
    for j in range(10):
        s, e = t_off[j]
        print(f"  tok {j:3d}  chars[{s:4d},{e:4d})  "
              f"{t_tok.decode([enc_t['input_ids'][j]])!r}")

    from teacher_logits import map_boundaries
    t_index, aligned = map_boundaries(s_off, t_off)
    n_ok = sum(aligned)
    print(f"\n对齐结果: 学生 {len(s_off)} 个位置中 {n_ok} 个找到 end 相同的教师 token "
          f"({100*n_ok/len(s_off):.1f}%)")
    print("\n前 12 个学生位置的映射（看 end 是否相等）:")
    for i in range(min(12, len(s_off))):
        s, e = s_off[i]
        j = t_index[i]
        if j >= 0:
            ts, te = t_off[j]
            print(f"  pos {i:3d} s_end={e:4d} -> tok {j:3d} t_end={te:4d}  "
                  f"{'✓' if e == te else '✗'}")
        else:
            print(f"  pos {i:3d} s_end={e:4d} -> 无对应（丢弃）")

    print("\n" + "=" * 78)
    print("【第 3 步】抽取对齐后的教师 logits（只取公共列）")
    print("=" * 78)
    with torch.no_grad():
        t_aligned = extract_aligned_logits(
            teacher, t_tok, [text], [s_off], student_seq_len=L,
            align=align, device=device, max_teacher_len=L * 2,
            dtype=torch.bfloat16,
        )
    t_logits, t_mask, shrink_ids = t_aligned.logits, t_aligned.mask, t_aligned.shrink_ids
    print(f"教师 logits 原始维度 = 151936")
    print(f"抽取后 = {tuple(t_logits.shape)}  (只保留 {len(shrink_ids)} 个公共列)")
    print(f"有效位置 mask 求和 = {int(t_mask.sum())}")

    print("\n" + "=" * 78)
    print("【第 4 步】学生前向")
    print("=" * 78)
    with torch.no_grad():
        out = student(input_ids=torch.tensor([ids]).to(device),
                      attention_mask=torch.ones(1, len(ids), dtype=torch.long).to(device))
    s_logits = out.logits
    print(f"学生 logits = {tuple(s_logits.shape)}")

    s_shift = s_logits[..., :-1, :]
    t_shift = t_logits[:, :-1, :]
    # mask 与 t_shift 同样取 [:-1]，保持与 logits 同步
    t_mask_shift = t_mask[:, :-1]
    lab = torch.tensor([labels]).to(device)

    print("\n" + "=" * 78)
    print("【第 5 步】CE 逐位置计算")
    print("=" * 78)
    sl = s_shift.reshape(-1, s_shift.size(-1)).float()
    yl = lab[..., 1:].reshape(-1)
    per = F.cross_entropy(sl, yl, ignore_index=IGNORE, reduction="none")
    m = yl != IGNORE
    ce = F.cross_entropy(sl, yl, ignore_index=IGNORE)
    print(f"{'位置':>6} {'目标':>8} {'学生p':>8} {'NLL':>8}")
    k = 0
    for pos in range(s_shift.size(1)):
        if not m[pos]:
            continue
        tid = int(yl[pos])
        lp = float(F.log_softmax(sl[pos], -1)[tid])
        print(f"{pos:6d} {s_tok.decode([tid])!r:>8} {math.exp(lp):8.4f} {-lp:8.4f}")
        k += 1
        if k >= 10:
            break
    print(f"\nCE = {float(ce):.4f}   （{int(m.sum())} 个 token 的 NLL 平均）")
    print(f"PPL = exp(CE) = {math.exp(float(ce)):.4f}")

    print("\n" + "=" * 78)
    print("【第 6 步】KL 逐位置计算（公共子集 + 双 mask）")
    print("=" * 78)
    T = cfg.kd.temperature
    kd_mask = torch.tensor([sample["kd_mask"]]).to(device)
    pos_mask = kd_mask[:, 1:].float() * t_mask_shift.float()
    s_shrink = s_shift[:, :, shrink_ids]
    log_p_t = F.log_softmax(t_shift.float() / T, dim=-1)
    log_p_s = F.log_softmax(s_shrink.float() / T, dim=-1)
    kl = (log_p_t.exp() * (log_p_t - log_p_s)).sum(dim=-1)
    kd = (kl * pos_mask).sum() / pos_mask.sum().clamp(min=1) * (T ** 2)

    print(f"T = {T}，公共列数 = {len(shrink_ids)}")
    print(f"两个 mask 相乘：kd_mask(assistant) × t_mask(对齐成功)")
    print(f"实际参与 KL 的位置数 = {int(pos_mask.sum())}")
    print(f"\n{'位置':>6} {'教师argmax':>12} {'p_t':>7} {'学生argmax':>12} {'p_s':>7} {'KL':>8}")
    cnt = 0
    for pos in range(pos_mask.size(1)):
        if pos_mask[0, pos] == 0:
            continue
        pt = log_p_t[0, pos].exp()
        ps = log_p_s[0, pos].exp()
        ti = int(pt.argmax())
        si = int(ps.argmax())
        t_tok_str = t_tok.decode([int(shrink_ids[ti])]) if False else f"id{ti}"
        s_str = s_tok.decode([int(shrink_ids[si])])
        print(f"{pos:6d} {ti:12d} {float(pt[ti]):7.4f} {s_str!r:>12} "
              f"{float(ps[si]):7.4f} {float(kl[0,pos]):8.4f}")
        cnt += 1
        if cnt >= 10:
            break
    print(f"\nKL（含 T²） = {float(kd):.4f}")

    print("\n" + "=" * 78)
    print("【第 7 步】总损失")
    print("=" * 78)
    total = cfg.kd.ce_weight * ce + cfg.kd.kd_weight * kd
    print(f"L = {cfg.kd.ce_weight} × {float(ce):.4f} + {cfg.kd.kd_weight} × {float(kd):.4f}"
          f" = {float(total):.4f}")


if __name__ == "__main__":
    main()