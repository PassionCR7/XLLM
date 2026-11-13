"""黑盒蒸馏训练：学生 xllm + 教师 API 产出的数据。

由于教师只暴露文本（及 token logprob），本目录实现了三种层次的蒸馏：

1. **Sequence-level KD（序列级蒸馏）**
   直接用教师生成的回复做 SFT。最经典、最稳健，等价于软标签的"硬采样"形式。

2. **Logprob-level KD（概率级蒸馏）**
   利用 API 返回的 top-k logprob 构造稀疏软标签：
   在教师 top-k 支撑集上做 KL，其余位置由剩余概率质量兜底。
   这是黑盒条件下能拿到的最接近分布蒸馏的信号。

3. **混合模式**
   `ce_weight` 控制硬标签、`kd_weight` 控制软标签，可自由配比。

稀疏软标签的构造
----------------
API 只给 top-k（本实现 k=5）。要把它们变成合法分布，需要处理
"top-k 之外的概率质量"。若直接对 top-k 重归一化，会人为放大教师置信度。
本实现采用 **背景质量兜底**：
    p_i = exp(lp_i)                    for i in top-k
    p_bg = 1 - sum(p_i)                （剩余质量）
KL 计算时，学生侧对应位置同样并入这团"剩余质量"，
即  KL = sum_k p_k * (log p_k - log q_k) + p_bg * (log p_bg - log q_bg)
其中 q_bg 是学生在"非 top-k token"上的概率和。
这样既不丢失教师的不确定性，也不虚构 token 级别的细节。
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DistillConfig                        # noqa: E402
from model.XLLMModel import XLLMConfig, XLLMForCausalLM  # noqa: E402

# 复用白盒目录中已修正键名缺陷的加载逻辑，避免重复实现
_KEY_TYPOS = {"post_attention_layernotm": "post_attention_layernorm"}
IGNORE_INDEX = -100


# --------------------------------------------------------------------------- #
# 学生加载
# --------------------------------------------------------------------------- #
def fix_keys(sd: dict) -> tuple[dict, int]:
    out, n = {}, 0
    for k, v in sd.items():
        nk = k
        for bad, good in _KEY_TYPOS.items():
            if bad in nk:
                nk = nk.replace(bad, good)
                n += 1
                break
        out[nk] = v
    return out, n


def load_student(cfg: DistillConfig, device: str):
    from transformers import AutoTokenizer

    s = cfg.student
    tok = AutoTokenizer.from_pretrained(s.model_dir)
    lm_cfg = XLLMConfig(
        hidden_size=s.hidden_size,
        num_hidden_layers=s.num_hidden_layers,
        use_moe=s.use_moe,
    )
    model = XLLMForCausalLM(lm_cfg)

    if s.from_weight and s.from_weight.lower() != "none":
        moe = "_moe" if s.use_moe else ""
        suffix = f"_{s.hidden_size}{moe}"
        stem = s.from_weight if s.from_weight.endswith(suffix) else f"{s.from_weight}{suffix}"
        path = os.path.join(s.weight_dir, f"{stem}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到学生权重: {path}")
        sd = torch.load(path, map_location="cpu", weights_only=True)
        sd, nfix = fix_keys(sd)
        if nfix:
            print(f"[学生] 修正 {nfix} 个键名拼写错误")
        r = model.load_state_dict(sd, strict=False)
        if r.missing_keys:
            print(f"[学生][警告] 缺失 {len(r.missing_keys)} 个权重")
        else:
            print("[学生] 权重完整加载")
        del sd

    print(f"[学生] 参数量 {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    return model.to(device=device, dtype=torch.bfloat16), tok


# --------------------------------------------------------------------------- #
# 数据集：解析教师 JSONL，产出学生序列 + 稀疏软标签
# --------------------------------------------------------------------------- #
class BlackboxDistillDataset(Dataset):
    """读取教师生成的数据，编码为学生的训练样本。

    每条样本返回：
        input_ids  : 学生 token 序列
        labels     : 仅 assistant 区间有效，其余 -100
        soft_ids   : [L, K] 教师 top-k token 对应的学生 id（无法映射为 -1）
        soft_probs : [L, K] 对应概率（已扣除背景质量后的归一化前值）
        soft_bg    : [L]    剩余质量
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        limit: int | None = None,
        use_soft: bool = True,
    ):
        self.tok = tokenizer
        self.max_length = max_length
        self.use_soft = use_soft
        self.samples: list[dict] = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))
                if limit and len(self.samples) >= limit:
                    break

        # 教师 token 字符串 -> 学生 id 的映射辅助
        # 注意：不能用 tokenizer.get_vocab() 直接查！get_vocab() 返回的是
        # BPE 原始形如 "Ġhello"（含字节映射字符），而 API 返回的是解码后的
        # 可读文本 " hello"。两者字符串不相等，直接查会全部 MISSING。
        # 正确做法是用 tokenizer.encode() 反查真实 id。
        self._id_cache: dict[str, int] = {}
        # 字符对齐结果缓存（同一样本只计算一次 difflib 对齐）
        self._span_cache_key: int = -1
        self._span_map: list = []

    def text_to_id(self, piece: str) -> int:
        """把一段可读文本映射为学生词表中的**单个** token id。

        只有当该文本恰好被编码为 1 个 token 时才返回其 id，
        否则返回 -1（不做跨 token 的强行拼接，避免错误监督）。
        """
        if not piece:
            return -1
        cached = self._id_cache.get(piece)
        if cached is not None:
            return cached
        ids = self.tok.encode(piece, add_special_tokens=False)
        sid = ids[0] if len(ids) == 1 else -1
        self._id_cache[piece] = sid
        return sid

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _align_spans(joined: str, text: str) -> list[tuple[int, int] | None]:
        """把 `joined`（教师 token 拼接串）的每个字符映射到 `text` 中的字符下标。

        用 difflib 的 SequenceMatcher 做对齐，返回长度 = len(joined) 的列表，
        每项是该字符在 text 中的位置，无法对齐处为 None。

        这样即便 API 返回的 token 字符串与原文存在长度/转义差异，
        也能通过"匹配块"把大部分字符准确落到原文位置。
        """
        from difflib import SequenceMatcher

        mapping: list[int | None] = [None] * len(joined)
        sm = SequenceMatcher(None, joined, text, autojunk=False)
        for a, b, size in sm.get_matching_blocks():
            for k in range(size):
                mapping[a + k] = b + k
        return mapping

    def _token_char_span(
        self,
        joined: str,
        teacher_tokens: list[str],
        t_i: int,
        text: str,
    ) -> tuple[int, int] | None:
        """返回第 t_i 个教师 token 在原文中的 (start, end) 字符区间。"""
        # 延迟计算并缓存对齐结果（同一文本只算一次）
        key = id(joined)
        if self._span_cache_key != key:
            self._span_map = self._align_spans(joined, text)
            self._span_cache_key = key

        start = sum(len(t) for t in teacher_tokens[:t_i])
        end = start + len(teacher_tokens[t_i])
        if end <= start:
            return None
        mapping = self._span_map
        if end > len(mapping):
            return None

        # 取该 token 内所有成功对齐字符的位置，用首尾界定区间
        positions = [mapping[k] for k in range(start, end) if mapping[k] is not None]
        if not positions:
            return None
        # 若对齐点稀疏（少于一半字符命中），视为不可靠，丢弃该 token
        if len(positions) < max(1, (end - start) // 2):
            return None
        return positions[0], positions[-1] + 1

    def _encode(self, conv: list[dict], teacher_logprobs: list[dict]) -> dict:
        text = self.tok.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=False
        )
        enc = self.tok(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = list(enc["input_ids"])
        offsets = enc["offset_mapping"]

        # 定位 assistant 区间：跳过 "<|im_start|>assistant\n" 头部
        header = "<|im_start|>assistant\n"
        spans: list[tuple[int, int]] = []
        cursor = 0
        while True:
            pos = text.find(header, cursor)
            if pos < 0:
                break
            start = pos + len(header)
            end = text.find("<|im_end|>", start)
            end = end if end >= 0 else len(text)
            spans.append((start, end))
            cursor = end

        def in_answer(s: int) -> bool:
            return any(a <= s < b for a, b in spans)

        labels = [
            tid if in_answer(s) else IGNORE_INDEX
            for tid, (s, _e) in zip(input_ids, offsets)
        ]

        # ---- 软标签：把教师 top-k token 映射到学生 id，按字符区间定位 ----
        soft_ids = [[-1] * 5 for _ in input_ids]
        soft_probs = [[0.0] * 5 for _ in input_ids]
        soft_bg = [1.0 for _ in input_ids]

        if self.use_soft and teacher_logprobs:
            # ------------------------------------------------------------------
            # 软标签定位：教师 top-k token -> 学生 token
            #
            # 不能简单按字符长度累加！API 返回的 token 是解码后的近似字符串，
            # 实测拼接后 363 字符 vs 原文 343 字符（"\n\n" 等转义长度失真），
            # 长度累加会逐位漂移，导致软标签整体错位。
            #
            # 正确做法：把教师 token 拼接串与 assistant 原文做**序列对齐**，
            # 用对齐结果把每个教师 token 映射到它在原文中的精确字符区间。
            # ------------------------------------------------------------------
            end_to_idx = {}
            for i, (_s, e) in enumerate(offsets):
                if e > 0:
                    end_to_idx[e] = i

            teacher_tokens = [item.get("token", "") for item in teacher_logprobs]
            joined = "".join(teacher_tokens)

            for t_i, item in enumerate(teacher_logprobs):
                # 该教师 token 在原文中的字符区间
                span = self._token_char_span(joined, teacher_tokens, t_i, text)
                if span is None:
                    continue
                _cs, ce = span
                idx = end_to_idx.get(ce)
                if idx is None:
                    continue
                # 只在 assistant 区间启用软标签
                if labels[idx] == IGNORE_INDEX:
                    continue

                top = item.get("top") or []
                merged: dict[int, float] = {}
                for cand in top[:5]:
                    sid = self.text_to_id(cand.get("token", ""))
                    lp = cand.get("logprob", -1e30)
                    if sid >= 0 and lp > -1e30:
                        merged[sid] = merged.get(sid, 0.0) + math.exp(lp)
                if not merged:
                    continue
                items = sorted(merged.items(), key=lambda kv: -kv[1])[:5]
                ids_row = [sid for sid, _ in items] + [-1] * (5 - len(items))
                prb_row = [p for _, p in items] + [0.0] * (5 - len(items))
                soft_ids[idx] = ids_row
                # 保留绝对概率，背景 = 1 - top-k 总质量
                soft_probs[idx] = prb_row
                soft_bg[idx] = max(0.0, 1.0 - sum(p for _, p in items))

        return {
            "input_ids": input_ids,
            "labels": labels,
            "soft_ids": soft_ids,
            "soft_probs": soft_probs,
            "soft_bg": soft_bg,
        }

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        return self._encode(
            sample.get("conversations", []),
            sample.get("teacher_logprobs") or [],
        )


def collate(batch: list[dict], pad_id: int, max_len: int, K: int = 5):
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long)
    soft_ids = torch.full((B, max_len, K), -1, dtype=torch.long)
    soft_probs = torch.zeros((B, max_len, K), dtype=torch.float)
    soft_bg = torch.ones((B, max_len), dtype=torch.float)
    attn = torch.zeros((B, max_len), dtype=torch.long)

    for b, s in enumerate(batch):
        n = min(len(s["input_ids"]), max_len)
        input_ids[b, :n] = torch.tensor(s["input_ids"][:n])
        labels[b, :n] = torch.tensor(s["labels"][:n])
        attn[b, :n] = 1
        soft_ids[b, :n] = torch.tensor(s["soft_ids"][:n], dtype=torch.long)
        soft_probs[b, :n] = torch.tensor(s["soft_probs"][:n], dtype=torch.float)
        soft_bg[b, :n] = torch.tensor(s["soft_bg"][:n], dtype=torch.float)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attn,
        "soft_ids": soft_ids,
        "soft_probs": soft_probs,
        "soft_bg": soft_bg,
    }


# --------------------------------------------------------------------------- #
# 损失
# --------------------------------------------------------------------------- #
def sparse_soft_kd_loss(
    logits: torch.Tensor,       # [N, V] 学生 logits
    soft_ids: torch.Tensor,     # [N, K]
    soft_probs: torch.Tensor,   # [N, K]
    soft_bg: torch.Tensor,      # [N]
    mask: torch.Tensor,         # [N]
    temperature: float = 1.0,
) -> torch.Tensor:
    """稀疏软标签 KL，含背景质量兜底。"""
    valid_row = (soft_ids >= 0).any(dim=-1) & (mask > 0)
    if valid_row.sum() == 0:
        return logits.new_zeros(())

    lg = logits[valid_row].float()
    sids = soft_ids[valid_row].clamp(min=0)          # [n, K]
    sp = soft_probs[valid_row].float()               # [n, K]
    bg = soft_bg[valid_row].float()                  # [n]

    log_q = F.log_softmax(lg / temperature, dim=-1)  # [n, V]
    q_k = log_q.gather(1, sids)                      # [n, K]

    # 教师 top-k 的绝对概率（温度 T 下重新归一化到"教师分布"）
    # 教师概率本身对应 T=1；用 T 变换后需保持其余质量的比例
    p_k = sp
    # 温度作用于教师侧：对 log p 除以 T 后重归一化
    log_p_k = torch.log(p_k.clamp(min=1e-12)) / temperature
    log_bg = torch.log(bg.clamp(min=1e-12)) / temperature
    all_logp = torch.cat([log_p_k, log_bg.unsqueeze(1)], dim=1)   # [n, K+1]
    log_p_norm = F.log_softmax(all_logp, dim=-1)
    p_norm_k = log_p_norm[:, : sids.size(1)]
    p_norm_bg = log_p_norm[:, -1]

    # 学生在非 top-k token 上的概率质量
    q_k_sum = q_k.exp().sum(dim=-1)
    # 只统计有效的 K（soft_ids = -1 的槽位贡献 0，因为 gather 到 id 0 会污染）
    valid_k = (soft_ids[valid_row] >= 0).float()
    q_k_sum = (q_k.exp() * valid_k).sum(dim=-1)
    log_q_bg = torch.log((1.0 - q_k_sum).clamp(min=1e-12))

    kl = (p_norm_k.exp() * (p_norm_k - q_k)).sum(dim=-1) + \
         p_norm_bg.exp() * (p_norm_bg - log_q_bg)
    return kl.mean() * (temperature ** 2)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="xllm 黑盒蒸馏训练")
    parser.add_argument("--data", type=str, default=None, help="教师数据 jsonl")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--ce_weight", type=float, default=None)
    parser.add_argument("--kd_weight", type=float, default=None)
    parser.add_argument("--save_weight", type=str, default=None)
    parser.add_argument("--no_soft", action="store_true", help="只用硬标签（纯序列级蒸馏）")
    parser.add_argument("--log_interval", type=int, default=None)
    args = parser.parse_args()

    cfg = DistillConfig()
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.train.learning_rate = args.learning_rate
    if args.max_seq_len is not None:
        cfg.student.max_seq_len = args.max_seq_len
    if args.temperature is not None:
        cfg.train.temperature = args.temperature
    if args.ce_weight is not None:
        cfg.train.ce_weight = args.ce_weight
    if args.kd_weight is not None:
        cfg.train.kd_weight = args.kd_weight
    if args.save_weight is not None:
        cfg.student.save_weight = args.save_weight
    if args.no_soft:
        cfg.train.kd_weight = 0.0
    if args.log_interval is not None:
        cfg.train.log_interval = args.log_interval

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    data_path = args.data or str(Path(cfg.data.out_dir) / "teacher_sft.jsonl")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"未找到教师数据: {data_path}\n请先运行: python gen_data.py"
        )

    model, tok = load_student(cfg, device)
    ds = BlackboxDistillDataset(
        data_path, tok, max_length=cfg.student.max_seq_len,
        limit=args.limit, use_soft=cfg.train.kd_weight > 0,
    )
    print(f"[数据] {len(ds)} 条 | max_len {cfg.student.max_seq_len}")

    loader = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=lambda b: collate(b, tok.pad_token_id, cfg.student.max_seq_len),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate,
                            weight_decay=0.01)
    total = len(loader) * cfg.train.epochs
    print(f"[训练] epochs={cfg.train.epochs} steps={total} | "
          f"ce_w={cfg.train.ce_weight} kd_w={cfg.train.kd_weight} T={cfg.train.temperature}")

    model.train()
    gstep = 0
    t0 = time.time()

    for epoch in range(cfg.train.epochs):
        el = ec = ek = 0.0
        nb = 0
        for step, batch in enumerate(loader):
            for g in opt.param_groups:
                warm = max(1, int(total * 0.05))
                if gstep < warm:
                    g["lr"] = cfg.train.learning_rate * (gstep + 1) / warm
                else:
                    prog = (gstep - warm) / max(1, total - warm)
                    g["lr"] = cfg.train.learning_rate * (
                        0.1 + 0.45 * (1 + math.cos(math.pi * prog))
                    )

            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attn = batch["attention_mask"].to(device)
            mask = (labels != IGNORE_INDEX).float()

            out = model(input_ids=ids, attention_mask=attn)
            logits = out.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )

            kd = torch.zeros((), device=device)
            if cfg.train.kd_weight > 0:
                kd = sparse_soft_kd_loss(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    batch["soft_ids"][:, 1:, :].reshape(-1, 5).to(device),
                    batch["soft_probs"][:, 1:, :].reshape(-1, 5).to(device),
                    batch["soft_bg"][:, 1:].reshape(-1).to(device),
                    mask[:, 1:].reshape(-1).to(device),
                    temperature=cfg.train.temperature,
                )

            loss = cfg.train.ce_weight * ce + cfg.train.kd_weight * kd
            if out.aux_loss is not None:
                loss = loss + out.aux_loss

            (loss / cfg.train.accumulation_steps).backward()
            if (step + 1) % cfg.train.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)

            gstep += 1
            el += float(loss.detach()); ec += float(ce.detach()); ek += float(kd.detach())
            nb += 1

            if (step + 1) % cfg.train.log_interval == 0:
                print(
                    f"Epoch[{epoch+1}/{cfg.train.epochs}] {step+1}/{len(loader)} | "
                    f"loss {float(loss.detach()):.4f} (ce {float(ce.detach()):.4f} "
                    f"kd {float(kd.detach()):.4f}) | avg {el/nb:.4f} | "
                    f"{time.time()-t0:.0f}s"
                )

            if (step + 1) % cfg.train.save_interval == 0:
                _save(model, cfg, f"{cfg.student.save_weight}_ep{epoch+1}_s{step+1}")

        print(f"== Epoch {epoch+1}: loss {el/max(nb,1):.4f} "
              f"(ce {ec/max(nb,1):.4f}, kd {ek/max(nb,1):.4f})")
        _save(model, cfg, cfg.student.save_weight)

    _save(model, cfg, cfg.student.save_weight)
    print(f"[完成] 耗时 {time.time()-t0:.0f}s")


def _save(model, cfg: DistillConfig, tag: str) -> str:
    d = Path(cfg.student.save_dir)
    d.mkdir(parents=True, exist_ok=True)
    moe = "_moe" if cfg.student.use_moe else ""
    suffix = f"_{cfg.student.hidden_size}{moe}"
    stem = tag if tag.endswith(suffix) else f"{tag}{suffix}"
    p = d / f"{stem}.pth"
    raw = getattr(model, "_orig_mod", model)
    tmp = str(p) + ".tmp"
    torch.save({k: v.half() for k, v in raw.state_dict().items()}, tmp)
    os.replace(tmp, p)
    print(f"[保存] {p}")
    return str(p)


if __name__ == "__main__":
    main()