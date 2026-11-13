# -*- coding: utf-8 -*-
"""
_mtp_engine_check.py —— MTP 引擎的验收脚本

验收顺序按"一票否决"排列，前一项不过后面都不用看：

    A. 贪心等价性（最关键）
       MTP 引擎贪心生成的 token 序列，必须与主干直接贪心解码**逐 token 相同**。
       任何 KV / 位置 / logits 对齐的错误都会在这里暴露。
       注意：只要求"与主干贪心相同"，不要求 MTP 草稿命中——即使 D 个模块全是
       乱猜，验证环节也会保证输出正确，只是没加速。所以这一项测的是**正确性**，
       与训练质量无关。

    B. 随机模式无偏性
       同一 prompt 采多次，MTP 引擎与主干直接采样的输出分布应一致。
       用卡方/偏离度做粗检，样本量有限，只作为 sanity check。

    C. 接受率
       这才是衡量"训练是否有效"的指标。MTP 草稿命中率越高，加速越多。
       未训练的 MTP 头命中率应接近随机基线；训练后应显著上升。

    D. 统计口径自检
       avg_tokens_per_iter 必须满足 1 <= x <= 1+D，且 acc_per_iter 的最大值
       不超过 max_draft。

用法
----
    # 用随机初始化的小模型快速验证引擎正确性
    python model/draft/_mtp_engine_check.py --quick

    # 用真实 512 权重 + 已训练的 MTP 头（若有）
    python model/draft/_mtp_engine_check.py --weight pretrain --mtp out/xxx_mtp_512.pth
"""

import argparse
import os
import pathlib
import sys

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
from collections import Counter

from model.XLLMModel import XLLMConfig, XLLMForCausalLM
from model.draft.mtp.mtp_module import MTPConfig, XLLMWithMTP
from model.draft.mtp.mtp_engine import MTPEngine


def baseline_greedy(backbone, input_ids, max_new_tokens):
    """主干直接贪心解码（对照组）。"""
    kv = None
    cur = input_ids
    out = backbone(input_ids=cur, use_cache=True, logits_to_keep=1)
    kv = out.past_key_values
    logits = out.logits[:, -1, :]
    gen = []
    for _ in range(max_new_tokens):
        nxt = logits.argmax(dim=-1)
        gen.append(int(nxt.item()))
        out = backbone(input_ids=nxt.unsqueeze(0), past_key_values=kv,
                       use_cache=True, logits_to_keep=1)
        kv = out.past_key_values
        logits = out.logits[:, -1, :]
    return gen


def build(args):
    if args.quick:
        cfg = XLLMConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=128,
                         max_position_embeddings=512, intermediate_size=128)
        bb = XLLMForCausalLM(cfg)
        mtp_cfg = MTPConfig.from_backbone(cfg, num_modules=args.modules)
        model = XLLMWithMTP(bb, mtp_cfg).eval()
        return model, cfg

    moe = "_moe" if args.use_moe else ""
    cfg = XLLMConfig(hidden_size=512, num_hidden_layers=args.layers,
                     use_moe=bool(args.use_moe), max_position_embeddings=1024)
    bb = XLLMForCausalLM(cfg)
    path = os.path.join(_REPO_ROOT, "out", f"{args.weight}_{args.hidden}{moe}.pth")
    if os.path.exists(path):
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        sd = {k.replace("post_attention_layernotm", "post_attention_layernorm"): v
              for k, v in sd.items()}
        miss, unexp = bb.load_state_dict(sd, strict=False)
        print(f"[load] {path}  missing={len(miss)} unexpected={len(unexp)}")

    mtp_cfg = MTPConfig.from_backbone(cfg, num_modules=args.modules)
    model = XLLMWithMTP(bb, mtp_cfg).eval()

    if args.mtp and os.path.exists(args.mtp):
        ck = torch.load(args.mtp, map_location="cpu", weights_only=True)
        model.heads.load_state_dict(ck["heads"])
        print(f"[load] MTP 头 {args.mtp}")
    else:
        print("[warn] MTP 头随机初始化（接受率会接近随机基线）")
    return model, cfg


def check_A_greedy_equiv(model, cfg, n_prompts=6, gen_len=20):
    print("\n[A] 贪心等价性 —— MTP 引擎必须与主干直接贪心逐 token 相同")
    torch.manual_seed(0)
    D = model.cfg.num_modules
    eng = MTPEngine(model, max_draft=1 + D, do_sample=False)
    all_ok = True
    for i in range(n_prompts):
        ids = torch.randint(0, cfg.vocab_size, (1, 12))
        ref = baseline_greedy(model.backbone, ids, gen_len)
        got = eng.generate(ids, max_new_tokens=gen_len)
        got = got[0] if got and isinstance(got[0], list) else got
        ok = ref == got
        all_ok &= ok
        mark = "OK " if ok else "FAIL"
        print(f"  [{mark}] prompt{i} 一致={ok}")
        if not ok:
            for j, (a, b) in enumerate(zip(ref, got)):
                if a != b:
                    print(f"        首个分歧 @{j}: ref={a} got={b}")
                    break
    assert all_ok, "贪心等价性失败 —— 引擎存在状态对齐错误"
    print("  -> 通过（输出与主干贪心完全一致，无损）")
    return eng


def check_B_sampling(model, cfg, n=2000, seed=1):
    """
    随机模式无偏性。

    判定方式要小心：本项目的快速测试用的是**随机初始化**的小模型，其输出分布
    接近均匀（128 词表上 p[argmax]≈0.03），此时任何有限样本的 TV 距离都会被
    统计噪声主导（n=300 时纯噪声就能到 0.28）。

    因此这里改用"与理论分布对比"的对称判据：
        引擎 -> 理论 的 TV  应约等于  直接采样 -> 理论 的 TV
    两者同量级即说明引擎无系统性偏差；若引擎 TV 显著更大才有问题。
    """
    print("\n[B] 随机模式无偏性")
    D = model.cfg.num_modules
    V = cfg.vocab_size
    ids = torch.randint(0, V, (1, 10))

    with torch.no_grad():
        out = model.backbone(input_ids=ids, use_cache=True, logits_to_keep=1)
        p = F.softmax(out.logits[:, -1, :].float(), dim=-1)[0]

    torch.manual_seed(seed)
    c_dir = Counter()
    with torch.no_grad():
        for _ in range(n):
            c_dir[int(torch.multinomial(p, 1))] += 1

    eng = MTPEngine(model, max_draft=1 + D, do_sample=True, temperature=1.0, seed=seed)
    torch.manual_seed(seed)
    c_eng = Counter()
    for _ in range(n):
        got = eng.generate(ids, max_new_tokens=1)
        got = got[0] if isinstance(got[0], list) else got
        c_eng[int(got[0])] += 1

    def tv(c):
        return 0.5 * sum(abs(c.get(k, 0) / n - p[k].item()) for k in range(V))

    td, te = tv(c_dir), tv(c_eng)
    print(f"  n={n}  vocab={V}  p[argmax]={p.max():.4f}")
    print(f"  直接采样 vs 理论 TV = {td:.4f}")
    print(f"  引擎采样 vs 理论 TV = {te:.4f}")
    print(f"  比值 = {te / max(td, 1e-9):.2f}")
    assert te < td * 1.6 + 0.02, (
        f"引擎分布偏离显著大于直接采样（{te:.4f} vs {td:.4f}），疑似有偏")
    print("  -> 通过（引擎偏差与直接采样的统计噪声同量级，无系统性偏差）")


def check_C_accept(model, cfg, n=40, gen_len=24):
    print("\n[C] 接受率（衡量训练效果的核心指标）")
    D = model.cfg.num_modules
    eng = MTPEngine(model, max_draft=1 + D, do_sample=False)
    torch.manual_seed(7)
    for _ in range(n):
        ids = torch.randint(0, cfg.vocab_size, (1, 12))
        eng.generate(ids, max_new_tokens=gen_len)
    s = eng.stats()
    print(f"  轮数={s['iters']}  产出={s['tokens']}  接受={s['accepted']}")
    print(f"  接受率={s['accept_rate']:.4f}  每轮平均接受={s['avg_accepted_per_iter']:.2f} "
          f"(草稿宽度 1+D={1+D})")
    print(f"  每轮平均产出 token={s['avg_tokens_per_iter']:.2f}  "
          f"(上限 2+D={2+D})")
    print(f"  acc_per_iter 分布={s['acc_per_iter'][:20]}")
    print(f"  草稿耗时占比={s['draft_overhead_ratio']:.3f}  "
          f"{s['ms_per_token']:.2f} ms/token")
    print("  判据：D=1 时每轮平均接受 ≈1.0 表示 MTP 头没学到东西；"
          ">1.3 说明有效；>1.8 表示训练充分")
    return s


def check_D_stats(s, D):
    print("\n[D] 统计口径自检")
    # 每轮草稿宽度 k = 1 + D（主头 1 个 + D 个模块）。
    # 最坏情况 k 个全部接受，再加 1 个压哨 token，产出 = k + 1 = 2 + D。
    # 所以 avg_tokens_per_iter 的合法区间是 [1, 2+D]，不是 [1, 1+D]。
    up = 2 + D
    assert 1.0 <= s["avg_tokens_per_iter"] <= up + 1e-9, (
        f"avg_tokens_per_iter={s['avg_tokens_per_iter']} 超出 [1, {up}]")
    assert s["avg_accepted_per_iter"] <= 1 + D + 1e-9, (
        f"avg_accepted_per_iter={s['avg_accepted_per_iter']} 超出宽度 {1+D}")
    assert max(s["acc_per_iter"]) <= 1 + D, (
        f"单轮接受 {max(s['acc_per_iter'])} 超过草稿宽度 {1+D}")
    assert s["accepted"] <= s["tokens"], "接受数不应超过产出数"
    print(f"  avg_tokens_per_iter={s['avg_tokens_per_iter']:.2f} ∈ [1, {up}]  OK")
    print(f"  avg_accepted_per_iter={s['avg_accepted_per_iter']:.2f} ∈ [0, {1+D}]  OK")
    print("  -> 通过")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="小模型随机权重，只验引擎正确性")
    p.add_argument("--weight", type=str, default="pretrain")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--use_moe", action="store_true")
    p.add_argument("--mtp", type=str, default=None, help="MTP 头 checkpoint 路径")
    p.add_argument("--modules", type=int, default=1, help="MTP 深度 D")
    args = p.parse_args()

    print("=" * 66)
    print("MTP 引擎验收")
    print("=" * 66)
    model, cfg = build(args)
    print(f"vocab={cfg.vocab_size} hidden={cfg.hidden_size} D={model.cfg.num_modules}")

    check_A_greedy_equiv(model, cfg)
    check_B_sampling(model, cfg)
    s = check_C_accept(model, cfg)
    check_D_stats(s, model.cfg.num_modules)

    print("\n" + "=" * 66)
    print("引擎验收通过（A/D 是一票否决项，C 用于观察训练效果）")


if __name__ == "__main__":
    main()