"""白盒蒸馏训练：学生 xllm + 教师 Qwen3-0.6B（本地 logits）。

损失设计
--------
    L = ce_weight * CE(student_logits, hard_labels)
      + kd_weight * KL( softmax(t_logits/T) || softmax(s_logits/T) ) * T^2

其中 KL 只在 **assistant 回复 token** 与 **成功对齐的位置** 的交集上计算，
双重 mask：
1. `kd_mask`：来自 dataset，只覆盖 assistant 回复区间；
2. `teacher_mask`：来自字符区间对齐，标记该位置教师侧是否有对应的预测。

关键实现细节
------------
* **教师不建图**：`torch.no_grad()` 前向，且只提取所需列（5850 而非 151936），
  显存占用从 ~300MB 降到 ~10MB/token 批次。
* **logits 位移对齐**：语言模型中 `logits[:, i]` 预测 `token[i+1]`。
  学生和教师都在同一字符前缀下预测同一后继，因此 `kd_mask` 与 `labels`
  都采用"错位一位"的写法：位置 i 的监督目标是 token i+1。
* **温度缩放**：KL 结果乘 T^2，使梯度量级与 CE 可比（Hinton et al. 2015）。
* **fp32 计算损失**：bf16 下 softmax/KL 会累积误差，损失计算统一转 float32。
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import WhiteboxConfig                     # noqa: E402
from dataset import DistillDataset, collate_fn        # noqa: E402
from student_loader import load_student, save_student  # noqa: E402
from teacher_logits import (                          # noqa: E402
    extract_aligned_logits,
    save_cached,
    load_cached,
)
from vocab_align import (                             # noqa: E402
    load_vocab_from_tokenizer_json,
    build_alignment,
)


def lr_lambda(step: int, total: int, base_lr: float) -> float:
    """余弦退火 + 预热 5%。"""
    warmup = max(1, int(total * 0.05))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))


def build_teacher(cfg: WhiteboxConfig, device: str):
    """加载教师模型与分词器。"""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    path = cfg.teacher.model_path
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到教师权重目录: {path}\n"
            f"请先下载 Qwen3-0.6B，或设置环境变量 XLLM_TEACHER_PATH"
        )

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=cfg.teacher.trust_remote_code)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = getattr(torch, cfg.teacher.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=dtype,
        trust_remote_code=cfg.teacher.trust_remote_code,
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n = sum(p.numel() for p in model.parameters())
    print(f"[教师] {cfg.teacher.model_name} 参数量 {n / 1e6:.1f} M | dtype {cfg.teacher.dtype}")
    return model, tok


def kd_step(
    student_model,
    teacher_model,
    teacher_tokenizer,
    batch: dict,
    align,
    student_tokenizer,
    cfg: WhiteboxConfig,
    device: str,
) -> dict:
    """单步前向 + 损失计算（不含 backward）。"""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attn = batch["attention_mask"].to(device)
    kd_mask = batch["kd_mask"].to(device)

    # ---------- 学生前向 ----------
    out = student_model(input_ids=input_ids, attention_mask=attn)
    s_logits = out.logits                       # [B, L, 6400]
    aux_loss = out.aux_loss

    # ---------- 硬标签 CE ----------
    shift_s = s_logits[..., :-1, :].contiguous()
    shift_y = labels[..., 1:].contiguous()
    ce = F.cross_entropy(
        shift_s.view(-1, shift_s.size(-1)).float(),
        shift_y.view(-1),
        ignore_index=-100,
    )

    kd_loss = torch.zeros((), device=device)
    align_positions = 0

    if cfg.kd.kd_weight > 0:
        # ---------- 教师 logits（带缓存） ----------
        batch_indices = batch["indices"]
        texts = batch["texts"]
        offsets = []
        for t in texts:
            enc = student_tokenizer(
                t, add_special_tokens=False, return_offsets_mapping=True
            )
            offsets.append(enc["offset_mapping"])

        need_compute = True
        if cfg.kd.cache_teacher_logits:
            loaded = [load_cached(cfg.kd.cache_dir, i, device) for i in batch_indices]
            if all(x is not None for x in loaded):
                t_logits = torch.stack([x[0] for x in loaded], dim=0)
                t_mask = torch.stack([x[1] for x in loaded], dim=0)
                shrink_ids = loaded[0][2]
                need_compute = False
            else:
                need_compute = True

        if need_compute:
            t_aligned = extract_aligned_logits(
                teacher_model,
                teacher_tokenizer,
                texts,
                offsets,
                student_seq_len=input_ids.size(1),
                align=align,
                device=device,
                max_teacher_len=cfg.teacher.max_seq_len
                if hasattr(cfg.teacher, "max_seq_len")
                else cfg.student.max_seq_len * 2,
                dtype=torch.bfloat16,
            )
            t_logits, t_mask = t_aligned.logits, t_aligned.mask
            shrink_ids = t_aligned.shrink_ids

            if cfg.kd.cache_teacher_logits:
                for k, idx in enumerate(batch_indices):
                    save_cached(
                        cfg.kd.cache_dir,
                        idx,
                        t_logits[k].cpu(),
                        t_mask[k].cpu(),
                        shrink_ids.cpu(),
                    )

        # ---------- 对齐到"预测下一个 token" ----------
        # 位置 i 的 logits 预测 i+1；教师 mask 同样右移一位
        t_shift = t_logits[:, :-1, :]
        t_mask_shift = t_mask[:, 1:]              # [B, L-1]
        s_shift = s_logits[..., :-1, :]

        # 学生在被覆盖子集上的 logits
        s_shrink = s_shift[:, :, shrink_ids]      # [B, L-1, S_common]

        # 组合 mask：assistant 区间 ∩ 对齐成功
        kd_shift = kd_mask[:, 1:].float()         # [B, L-1]
        pos_mask = kd_shift * t_mask_shift.float()

        T = cfg.kd.temperature
        log_p_t = F.log_softmax(t_shift.float() / T, dim=-1)
        log_p_s = F.log_softmax(s_shrink.float() / T, dim=-1)
        kl = (log_p_t.exp() * (log_p_t - log_p_s)).sum(dim=-1)   # [B, L-1]

        denom = pos_mask.sum().clamp(min=1.0)
        kd_loss = (kl * pos_mask).sum() / denom * (T ** 2)
        align_positions = int(pos_mask.sum().item())

    total = cfg.kd.ce_weight * ce + cfg.kd.kd_weight * kd_loss
    if aux_loss is not None:
        total = total + aux_loss

    return {
        "loss": total,
        "ce": ce.detach(),
        "kd": kd_loss.detach(),
        "aux": (aux_loss.detach() if aux_loss is not None else torch.zeros(())),
        "align_positions": align_positions,
    }


def main():
    parser = argparse.ArgumentParser(description="xllm 白盒蒸馏训练")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="只用前 N 条样本（调试）")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--kd_weight", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true", help="禁用教师 logits 缓存")
    parser.add_argument("--save_weight", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="从指定学生权重继续")
    args = parser.parse_args()

    cfg = WhiteboxConfig()
    if args.epochs is not None:
        cfg.kd.epochs = args.epochs
    if args.batch_size is not None:
        cfg.kd.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.kd.learning_rate = args.learning_rate
    if args.max_seq_len is not None:
        cfg.student.max_seq_len = args.max_seq_len
        cfg.data.max_seq_len = args.max_seq_len
    if args.temperature is not None:
        cfg.kd.temperature = args.temperature
    if args.kd_weight is not None:
        cfg.kd.kd_weight = args.kd_weight
    if args.no_cache:
        cfg.kd.cache_teacher_logits = False
    if args.save_weight:
        cfg.student.save_weight = args.save_weight
    if args.log_interval is not None:
        cfg.kd.log_interval = args.log_interval
    if args.resume:
        cfg.student.from_weight = args.resume

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    if device == "cuda":
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ---------- 1. 学生 ----------
    student, s_tok = load_student(cfg.student, device=device)

    # ---------- 2. 教师 ----------
    teacher, t_tok = build_teacher(cfg, device)

    # ---------- 3. 词表对齐 ----------
    t_vocab_path = os.path.join(cfg.teacher.model_path, "tokenizer.json")
    t_vocab = load_vocab_from_tokenizer_json(t_vocab_path)
    s_vocab = load_vocab_from_tokenizer_json(
        os.path.join(cfg.student.model_dir, "tokenizer.json")
    )
    align = build_alignment(t_vocab, s_vocab)

    # ---------- 4. 数据 ----------
    ds = DistillDataset(
        cfg.data.seed_file,
        s_tok,
        max_length=cfg.data.max_seq_len,
        limit=args.limit,
    )
    print(f"[数据] 样本数 {len(ds)} | max_len {cfg.data.max_seq_len}")

    loader = DataLoader(
        ds,
        batch_size=cfg.kd.batch_size,
        shuffle=True,
        num_workers=cfg.kd.num_workers,
        collate_fn=lambda b: collate_fn(b, s_tok.pad_token_id, cfg.data.max_seq_len),
        drop_last=False,
    )

    # ---------- 5. 优化器 ----------
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=cfg.kd.learning_rate, weight_decay=0.01
    )
    total_steps = len(loader) * cfg.kd.epochs
    print(f"[训练] epochs={cfg.kd.epochs} steps/epoch={len(loader)} total={total_steps}")
    print(f"[训练] T={cfg.kd.temperature} ce_w={cfg.kd.ce_weight} kd_w={cfg.kd.kd_weight}")

    student.train()
    best_loss = float("inf")
    global_step = 0
    t0 = time.time()

    for epoch in range(cfg.kd.epochs):
        epoch_loss = epoch_ce = epoch_kd = 0.0
        nb = 0
        for step, batch in enumerate(loader):
            for g in optimizer.param_groups:
                g["lr"] = lr_lambda(global_step, total_steps, cfg.kd.learning_rate)

            res = kd_step(
                student, teacher, t_tok, batch, align, s_tok, cfg, device
            )
            loss = res["loss"] / cfg.kd.accumulation_steps
            loss.backward()

            if (step + 1) % cfg.kd.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.kd.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            epoch_loss += float(res["loss"].detach())
            epoch_ce += float(res["ce"])
            epoch_kd += float(res["kd"])
            nb += 1

            if (step + 1) % cfg.kd.log_interval == 0:
                avg = epoch_loss / nb
                mem = (
                    torch.cuda.max_memory_allocated() / 1024**3
                    if device == "cuda" else 0
                )
                print(
                    f"Epoch[{epoch+1}/{cfg.kd.epochs}] step {step+1}/{len(loader)} | "
                    f"loss {float(res['loss'].detach()):.4f} "
                    f"(ce {float(res['ce']):.4f} kd {float(res['kd']):.4f}) | "
                    f"avg {avg:.4f} | align {res['align_positions']} | "
                    f"vram {mem:.2f}GB | {time.time()-t0:.0f}s"
                )

            if (step + 1) % cfg.kd.save_interval == 0:
                save_student(student, cfg.student,
                             tag=f"{cfg.student.save_weight}_ep{epoch+1}_s{step+1}")

        avg_loss = epoch_loss / max(nb, 1)
        print(
            f"== Epoch {epoch+1} 完成: loss {avg_loss:.4f} "
            f"(ce {epoch_ce/max(nb,1):.4f}, kd {epoch_kd/max(nb,1):.4f})"
        )
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_student(student, cfg.student, tag=cfg.student.save_weight)

    save_student(student, cfg.student)
    print(f"[完成] 总耗时 {time.time()-t0:.0f}s，最佳平均 loss {best_loss:.4f}")


if __name__ == "__main__":
    main()