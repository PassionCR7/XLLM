# -*- coding: utf-8 -*-
"""
infer.py —— MTP 推测解码的推理入口（可直接跑，出文本）

mtp_engine.py 只提供张量级接口（吃 input_ids、吐 token id）。本文件补上它缺的
那一层：tokenizer 编解码、权重加载、命令行、以及与普通解码的对比基准。

为什么要有「对比基准」
----------------------
MTP 的加速比不能靠感觉。必须用同一 prompt、同一 max_new_tokens，分别跑
    baseline  主干直接贪心解码（每步 1 次前向）
    mtp       MTP 推测解码（每轮 1 次验证前向 + 1 次补位前向 + D 次草稿前向）
测两者的总耗时与每 token 延迟，才能算出真实加速比。

判据不是「每轮接受几个」，而是
    有效加速比 = baseline 每 token 耗时 / MTP 每 token 耗时
这个数小于 1 就说明 MTP 头白训了，哪怕接受率很好看。

用法
----
    # 基本推理（自动找 out/ 下的权重）
    python model/draft/mtp/infer.py --prompt "你好"

    # 指定权重与 MTP 头
    python model/draft/mtp/infer.py --prompt "你好" \\
        --weight pretrain --mtp_weight_file out/pretrain_mtp_mtp_512.pth

    # 同时跑基线与 MTP，输出加速比（推荐）
    python model/draft/mtp/infer.py --prompt "你好" --benchmark

    # 随机采样模式（验证无偏性时用）
    python model/draft/mtp/infer.py --prompt "你好" --do-sample --temperature 0.8 --top-p 0.9

    # 改 D（必须与训练时一致，否则 MTP 头对不上）
    python model/draft/mtp/infer.py --prompt "你好" --modules 1

注意
----
* MTP 头必须是训练过的。随机初始化的头会让接受率停在基线（每轮 1.0 个），
  此时 MTP 只是白白多花时间——`--benchmark` 会直接显示加速比 < 1。
* `--modules` 必须与训练时一致。MTP 头的权重形状绑定了 D，改了就加载不上。
"""

import argparse
import os
import pathlib
import sys
import time
from typing import List, Optional

# 保证可以从仓库根目录 import model（向上找标志目录，不硬编码层数）
def _find_repo_root(start=None):
    p = pathlib.Path(start or __file__).resolve().parent
    for cand in [p, *p.parents]:
        if (cand / "model").is_dir() and (cand / "trainer").is_dir():
            return str(cand)
    return str(p.parents[1])


_REPO_ROOT = _find_repo_root()
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from model.XLLMModel import XLLMConfig, XLLMForCausalLM
from model.draft.mtp.mtp_module import MTPConfig, XLLMWithMTP
from model.draft.mtp.mtp_engine import MTPEngine


# ----------------------------------------------------------------------
# 权重加载
# ----------------------------------------------------------------------
def _strip_wrappers(state):
    """兼容多种保存格式：裸 state_dict / {"model": ...} / {"state_dict": ...}。"""
    if isinstance(state, dict):
        for key in ("model", "state_dict", "module"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    # 历史拼写错误（本项目早期版本有这个词），加载前统一修正
    return {k.replace("post_attention_layernotm", "post_attention_layernorm"): v
            for k, v in state.items()}


def load_mtp_checkpoint(mtp_file: str, device):
    """
    读 MTP 头文件，返回 (heads_state, MTPConfig, 元信息)。

    MTP 头文件里存了 cfg.__dict__，因此 D 可以从文件自动推断，
    不需要用户手动传 --modules（但还是允许覆盖，便于对照实验）。
    """
    ck = torch.load(mtp_file, map_location="cpu", weights_only=True)
    if "heads" not in ck:
        raise KeyError(f"{mtp_file} 里没有 'heads' 字段，不是 MTP 头文件")
    cfg_dict = ck.get("cfg", {})
    return ck["heads"], cfg_dict, {
        "hidden_size": ck.get("hidden_size"),
        "vocab_size": ck.get("vocab_size"),
        "mtp_weight": ck.get("mtp_weight"),
    }


def _check_backbone_against_mtp(args, cfg_dict, heads_state):
    """
    用 MTP 头的 shape 反推主干结构并校验。

    为什么必须做：load_state_dict(strict=False) 会**静默跳过** shape 不匹配的
    权重，不报错。于是 GQA 头数配错时，程序照常跑完，只是加载了一半权重，
    输出全是垃圾——这种 bug 最难查，必须在加载前显式拦住。
    """
    problems = []

    # hidden_size 由 cfg 或 proj 权重直接给出
    h = cfg_dict.get("hidden_size")
    if h and h != args.hidden_size:
        problems.append(f"hidden_size: 传入 {args.hidden_size} vs 头文件 {h}")

    # proj.weight 形状是 [H, 2H]，可直接确认 H
    proj_key = next((k for k in heads_state if k.endswith("proj.weight")), None)
    if proj_key:
        out_dim, in_dim = heads_state[proj_key].shape
        if out_dim != args.hidden_size:
            problems.append(f"proj 输出维 {out_dim} != hidden_size {args.hidden_size}")
        if in_dim != 2 * args.hidden_size:
            problems.append(f"proj 输入维 {in_dim} != 2*hidden_size {2 * args.hidden_size}")

    # k_proj 形状是 [kv_heads*kv_head_dim, H]，其中 kv_head_dim = H // num_heads
    k_key = next((k for k in heads_state if k.endswith("k_proj.weight")), None)
    if k_key:
        k_out, _ = heads_state[k_key].shape
        expect = args.kv_heads * (args.hidden_size // args.heads)
        if k_out != expect:
            problems.append(
                f"k_proj 输出维 {k_out} != kv_heads*(H/heads) = "
                f"{args.kv_heads}*({args.hidden_size}/{args.heads}) = {expect}")

    return problems


def build_model(args, device):
    """构建主干 + MTP 头，返回 XLLMWithMTP。"""
    # ---- MTP 头（先读，因为 hidden_size / D 以它为准） ----
    mtp_file = args.mtp_weight_file
    heads_state = None
    cfg_dict = {}
    if mtp_file:
        if not os.path.exists(mtp_file):
            raise FileNotFoundError(f"MTP 头文件不存在: {mtp_file}")
        heads_state, cfg_dict, meta = load_mtp_checkpoint(mtp_file, device)
        print(f"[mtp] 头文件 {mtp_file}")
        if meta["mtp_weight"] is not None:
            print(f"[mtp] 训练时 λ={meta['mtp_weight']}  "
                  f"hidden={meta['hidden_size']}  vocab={meta['vocab_size']}")
        # 自动对齐 hidden_size（避免用户传错导致 shape mismatch）
        if meta["hidden_size"] and meta["hidden_size"] != args.hidden_size:
            print(f"[warn] --hidden_size={args.hidden_size} 与头文件 "
                  f"{meta['hidden_size']} 不符，以头文件为准")
            args.hidden_size = meta["hidden_size"]
        if "num_modules" in cfg_dict and cfg_dict["num_modules"] != args.modules:
            print(f"[warn] --modules={args.modules} 与头文件 D="
                  f"{cfg_dict['num_modules']} 不符，以头文件为准")
            args.modules = cfg_dict["num_modules"]
        # 用头的 shape 校验主干超参，避免 strict=False 静默跳过
        probs = _check_backbone_against_mtp(args, cfg_dict, heads_state)
        if probs:
            raise ValueError(
                "主干超参与 MTP 头不匹配，直接加载会静默跳过一半权重：\n  "
                + "\n  ".join(probs)
                + "\n请按训练时的配置传 --hidden-size/--heads/--kv-heads")

    # ---- 主干 ----
    cfg = XLLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        vocab_size=args.vocab_size,
        max_position_embeddings=max(args.max_seq_len, 1024),
        use_moe=bool(args.moe),
    )
    backbone = XLLMForCausalLM(cfg)

    bb_file = args.weight_file
    if bb_file and os.path.exists(bb_file):
        state = _strip_wrappers(torch.load(bb_file, map_location="cpu",
                                           weights_only=True))
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        print(f"[backbone] {bb_file}  missing={len(missing)} "
              f"unexpected={len(unexpected)}")
        if len(missing) > cfg.num_hidden_layers:
            print(f"[warn] missing 偏多（{len(missing)} 个），权重可能不匹配")
    else:
        print(f"[warn] 未找到主干权重 {bb_file}，使用随机初始化（输出无意义）")

    # ---- 组装 ----
    mtp_cfg = MTPConfig.from_backbone(cfg, num_modules=args.modules)
    model = XLLMWithMTP(backbone, mtp_cfg, mtp_weight=0.3)   # λ 推理时不影响

    if heads_state is not None:
        model.heads.load_state_dict(heads_state)
        print(f"[mtp] 已加载头权重（D={args.modules}）")
    else:
        print("[warn] 未加载 MTP 头 → 使用随机初始化。"
              "此时接受率会停在基线，MTP 只会更慢。")

    model = model.to(device).eval()
    n_bb = sum(p.numel() for p in backbone.parameters()) / 1e6
    n_mtp = model.num_mtp_parameters() / 1e6
    print(f"[param] 主干 {n_bb:.2f}M | MTP {n_mtp:.2f}M "
          f"({n_mtp / max(n_bb, 1e-9) * 100:.1f}%)")
    return model


# ----------------------------------------------------------------------
# 两种解码
# ----------------------------------------------------------------------
@torch.no_grad()
def baseline_greedy(model, input_ids: torch.Tensor, max_new: int,
                    eos_id: Optional[int]) -> tuple[List[int], float]:
    """
    主干直接贪心解码：每步 1 次前向，出 1 个 token。这是对比基准。
    返回 (token 列表, 总耗时秒)。
    """
    backbone = model.backbone
    t0 = time.perf_counter()
    out = backbone(input_ids=input_ids, use_cache=True,
                   logits_to_keep=input_ids.shape[1])
    kv = out.past_key_values
    cur = out.logits[:, -1, :]
    gen: List[int] = []
    for _ in range(max_new):
        tok = int(cur.argmax(-1).item())
        gen.append(tok)
        if eos_id is not None and tok == eos_id:
            break
        nt = torch.tensor([[tok]], dtype=torch.long, device=input_ids.device)
        o = backbone(input_ids=nt, past_key_values=kv, use_cache=True,
                     logits_to_keep=1)
        kv = o.past_key_values
        cur = o.logits[:, -1, :]
    return gen, time.perf_counter() - t0


def mtp_generate(model, input_ids: torch.Tensor, args, device):
    """走 MTP 引擎，返回 (token 列表, 引擎, 耗时)。"""
    engine = MTPEngine(
        model,
        max_draft=1 + model.cfg.num_modules,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=args.eos_id,
        seed=args.seed,
        device=device,
    )
    t0 = time.perf_counter()
    tokens = engine.generate(input_ids, max_new_tokens=args.max_new)
    elapsed = time.perf_counter() - t0
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    return tokens, engine, elapsed


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="MTP 推测解码推理",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 输入
    p.add_argument("--prompt", type=str, default="你好",
                   help="输入文本；为空串则只用 --token-ids")
    p.add_argument("--token-ids", type=str, default=None,
                   help="逗号分隔的 token id，给了就忽略 --prompt")

    # 权重
    p.add_argument("--weight", type=str, default="pretrain",
                   help="out/ 下的主干权重名前缀，自动拼 _{hidden}.pth")
    p.add_argument("--weight-file", type=str, default=None,
                   help="直接指定主干权重完整路径（优先于 --weight）")
    p.add_argument("--mtp-weight-file", type=str, default=None,
                   help="MTP 头文件路径。默认自动找 out/{weight}_mtp_{hidden}.pth")
    p.add_argument("--save-dir", type=str, default=os.path.join(_REPO_ROOT, "out"))

    # 结构（与训练一致）
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--vocab-size", type=int, default=6400)
    p.add_argument("--modules", type=int, default=1, help="D：MTP 链深度")
    p.add_argument("--moe", type=int, default=0, choices=[0, 1])

    # 生成
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--do-sample", action="store_true",
                   help="随机采样（默认贪心）；贪心结果应与主干逐 token 一致")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eos-id", type=int, default=None,
                   help="EOS token id，默认从 tokenizer 自动取")

    # 对比
    p.add_argument("--benchmark", action="store_true",
                   help="同时跑主干贪心基线，输出真实加速比")
    p.add_argument("--bench-repeat", type=int, default=1,
                   help="benchmark 重复次数，取平均（越小越省时）")
    p.add_argument("--bench-ignore-eos", action="store_true",
                   help="benchmark 时忽略 EOS，强制生成满 --max-new 个 token，"
                        "保证两侧长度一致、时间可比（推荐用于测速）")

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # ---- 自动补路径 ----
    moe = "_moe" if args.moe else ""
    if args.weight_file is None:
        args.weight_file = os.path.join(
            args.save_dir, f"{args.weight}_{args.hidden_size}{moe}.pth")
    if args.mtp_weight_file is None:
        cand = os.path.join(
            args.save_dir, f"{args.weight}_mtp_{args.hidden_size}{moe}.pth")
        if os.path.exists(cand):
            args.mtp_weight_file = cand

    # ---- tokenizer ----
    tokenizer = None
    if args.token_ids is None:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(_REPO_ROOT, "model"))
        except Exception as e:
            print(f"[warn] tokenizer 加载失败（{e}），改用 --token-ids")
    if args.eos_id is None and tokenizer is not None:
        eos = tokenizer.eos_token_id
        args.eos_id = eos if eos is not None else None

    # ---- 模型 ----
    print("=" * 60)
    model = build_model(args, device)

    # ---- 编码输入 ----
    if args.token_ids:
        ids = [int(x) for x in args.token_ids.split(",") if x.strip()]
        if not ids:
            raise ValueError("--token-ids 解析后为空")
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        prompt_text = None
    else:
        prompt_text = args.prompt
        if tokenizer is None:
            raise RuntimeError("没有 tokenizer，请改用 --token-ids")
        input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)

    if input_ids.shape[1] + args.max_new > args.max_seq_len:
        print(f"[warn] prompt({input_ids.shape[1]}) + max_new({args.max_new}) "
              f"超过 max_seq_len({args.max_seq_len})，可能越界")

    # ---- MTP 推理 ----
    print("-" * 60)
    # 测速模式下两边都不许提前 EOS 退出，否则长度不同、时间无法直接比较
    eff_eos = None if (args.benchmark and args.bench_ignore_eos) else args.eos_id
    args.eos_id = eff_eos
    t0 = time.perf_counter()
    tokens, engine, total_elapsed = mtp_generate(model, input_ids, args, device)
    full = torch.cat([input_ids, torch.tensor([tokens], device=device)], dim=1)
    text = tokenizer.decode(full[0].tolist(), skip_special_tokens=True) \
        if tokenizer is not None else None

    print(f"[MTP] 模式     : {'随机采样' if args.do_sample else '贪心'}")
    print(f"[MTP] D        : {model.cfg.num_modules}  "
          f"(每轮验证 {1 + model.cfg.num_modules} 个)")
    print(f"[MTP] 生成     : {len(tokens)} token, 耗时 {total_elapsed:.3f}s")
    print(f"[MTP] 输出     : {text if text is not None else tokens}")

    s = engine.stats()
    print("\n---- MTP 统计 ----")
    print(f"  轮数              : {s['iters']}")
    print(f"  产出 token        : {s['tokens']}")
    print(f"  接受草稿          : {s['accepted']}")
    print(f"  接受率            : {s['accept_rate']:.4f}")
    print(f"  每轮平均接受      : {s['avg_accepted_per_iter']:.2f} "
          f"(宽度 1+D={1 + model.cfg.num_modules})")
    print(f"  每轮平均产出      : {s['avg_tokens_per_iter']:.2f} "
          f"(上限 2+D={2 + model.cfg.num_modules})")
    print(f"  草稿耗时占比      : {s['draft_overhead_ratio']:.3f}")
    print(f"  每 token          : {s['ms_per_token']:.2f} ms")
    print(f"  acc_per_iter      : {s['acc_per_iter'][:24]}")
    if s["avg_accepted_per_iter"] < 1.15:
        print("  [提示] 每轮平均接受接近 1.0 → MTP 头基本没学到东西，"
              "加速会很有限")

    # ---- 基线对比 ----
    if args.benchmark:
        print("\n" + "=" * 60)
        print("基准对比：主干直接贪心解码")
        # 公平性：两边必须产出同样多的 token 才能比时间。
        # 若一边先撞 EOS 提前停，直接比总耗时是错的（长度不同）。
        # 因此基线按「MTP 实际产出的 token 数」固定长度跑，禁用 EOS 提前退出。
        n_target = max(len(tokens), 1)
        ref_times = []
        ref_tokens = []
        for _ in range(max(1, args.bench_repeat)):
            rt_tokens, rt = baseline_greedy(model, input_ids, n_target,
                                            eos_id=None)   # 固定长度，不提前停
            ref_times.append(rt)
            ref_tokens = rt_tokens
        base_elapsed = sum(ref_times) / len(ref_times)

        n_ref = len(ref_tokens)
        n_mtp = len(tokens)
        print(f"  基线     : {n_ref} token, {base_elapsed:.3f}s, "
              f"{1000 * base_elapsed / max(n_ref, 1):.2f} ms/token")
        print(f"  MTP      : {n_mtp} token, {total_elapsed:.3f}s, "
              f"{1000 * total_elapsed / max(n_mtp, 1):.2f} ms/token")

        ms_base = 1000 * base_elapsed / max(n_ref, 1)
        ms_mtp = 1000 * total_elapsed / max(n_mtp, 1)
        speedup = ms_base / max(ms_mtp, 1e-9)
        print(f"  有效加速比: {speedup:.2f}x  "
              f"({'加速' if speedup > 1.02 else '未加速' if speedup > 0.98 else '变慢'})")

        # 小样本下 MTP 的固定开销（首次 kernel 预热、计时抖动）占比很大，
        # 提示用户加长序列再测，否则结论不可靠。
        if n_mtp < 32:
            print(f"  [注意] 仅 {n_mtp} 个 token，"
                  f"首次前向的预热开销会严重拉低加速比，"
                  f"建议 --max-new 64 以上再判断")

        # 贪心模式下核对等价性——这是正确性的一票否决项
        if not args.do_sample:
            n = min(n_ref, n_mtp)
            same = ref_tokens[:n] == tokens[:n]
            print(f"\n  等价性检查: {'逐 token 一致 ✓' if same else '不一致 ✗'} "
                  f"(比对前 {n} 个 token)")
            if not same:
                for i, (a, b) in enumerate(zip(ref_tokens[:n], tokens[:n])):
                    if a != b:
                        print(f"    首个分歧位于第 {i} 个 token: "
                              f"基线={a} vs MTP={b}")
                        break
                print("    不一致说明引擎有 bug，不要继续测速度。")

    print("=" * 60)


if __name__ == "__main__":
    main()