"""XLLM / MiniMind 参数量核算（纯 Python，不依赖 torch）。

支持两代架构，因为 minimind 上游改过 FFN 宽度公式与 GQA 配置：

  minimind2 世代（26M / 104M / 145M）
      intermediate_size = 64 * ceil( int(hidden * 8/3) / 64 )
      num_key_value_heads = 2（GQA 8:2）
      无 QK-Norm

  minimind3 世代（64M / 198M）
      intermediate_size = ceil(hidden * pi / 64) * 64
      num_key_value_heads = 4（GQA 8:4）
      带 QK-Norm（q_norm / k_norm，各 head_dim 个参数）

MoE 的两种接法也不同：
  minimind2-moe : 4 路由专家 + 1 共享专家，叠加在 dense FFN 之外
  minimind3-moe : 只有 4 路由专家，直接替换 dense FFN（无共享专家）

用法:
    python count_params.py                    # 本仓库 XLLMConfig 默认值
    python count_params.py --arch minimind3   # 对齐 minimind-3 的 64M 规格
    python count_params.py --arch minimind3 --use_moe 1
    python count_params.py --all              # 打印全部官方档位对照
"""

import argparse
import math

# 官方 README / HF 权重声明值，仅用于自检
OFFICIAL = {
    ("minimind2", 512, 8, 0): 25.83,
    ("minimind2", 640, 8, 1): 145.03,
    ("minimind2", 768, 16, 0): 104.03,
    ("minimind3", 768, 8, 0): 63.91,
    ("minimind3", 768, 8, 1): 198.42,
}


def ffn_intermediate(hidden_size: int, arch: str) -> int:
    if arch == "minimind3":
        return math.ceil(hidden_size * math.pi / 64) * 64
    inter = int(hidden_size * 8 / 3)
    return 64 * ((inter + 64 - 1) // 64)


def count(hidden_size=512, num_hidden_layers=8, vocab_size=6400,
          num_attention_heads=8, num_key_value_heads=None,
          use_moe=False, n_routed_experts=4, n_shared_experts=0,
          arch="minimind2"):
    # 两代默认 KV 头数不同：minimind2 用 2，minimind3 用 4
    if num_key_value_heads is None:
        num_key_value_heads = 4 if arch == "minimind3" else 2
    qk_norm = arch == "minimind3"

    head_dim = hidden_size // num_attention_heads

    # ---- Attention（GQA）----
    attn = (hidden_size * (num_attention_heads * head_dim)      # q
            + hidden_size * (num_key_value_heads * head_dim)    # k
            + hidden_size * (num_key_value_heads * head_dim)    # v
            + (num_attention_heads * head_dim) * hidden_size)   # o
    if qk_norm:
        attn += 2 * head_dim                                    # q_norm + k_norm

    # ---- FFN（SwiGLU: gate / up / down，均无 bias）----
    inter = ffn_intermediate(hidden_size, arch)
    ffn_one = 3 * hidden_size * inter

    # ---- 每层 MLP 总计 ----
    if use_moe:
        if arch == "minimind3":
            mlp = ffn_one * n_routed_experts                    # 替换 dense FFN
        else:
            mlp = ffn_one * (n_routed_experts + n_shared_experts)  # 叠加共享专家
        mlp += n_routed_experts * hidden_size                   # MoE gate，无 bias
    else:
        mlp = ffn_one

    norms = 2 * hidden_size                                     # 两个 LayerNorm
    per_layer = attn + mlp + norms

    embed = vocab_size * hidden_size                            # 与 lm_head 共享，只计一次
    total = embed + num_hidden_layers * per_layer + hidden_size

    return {
        "arch": arch,
        "head_dim": head_dim,
        "kv_heads": num_key_value_heads,
        "intermediate_size": inter,
        "qk_norm": qk_norm,
        "attn_per_layer": attn,
        "ffn_per_layer": ffn_one,
        "mlp_per_layer": mlp,
        "per_layer": per_layer,
        "ffn_total": num_hidden_layers * ffn_one,
        "embed": embed,
        "total": total,
        "ffn_share": num_hidden_layers * ffn_one / total,
        "layers": num_hidden_layers,
    }


def render(r, hs, nl, moe):
    print(f"arch                = {r['arch']}")
    print(f"hidden_size         = {hs}")
    print(f"num_hidden_layers   = {nl}")
    print(f"kv_heads / head_dim = {r['kv_heads']} / {r['head_dim']}")
    formula = "ceil(h*pi/64)*64" if r["arch"] == "minimind3" else "64*ceil(int(h*8/3)/64)"
    print(f"intermediate_size   = {r['intermediate_size']}   [{formula}]")
    print(f"QK-Norm             = {'有' if r['qk_norm'] else '无'}")
    print("-" * 48)
    print(f"Attention / 层       = {r['attn_per_layer']:>12,}")
    print(f"FFN(单个) / 层        = {r['ffn_per_layer']:>12,}")
    print(f"MLP 合计 / 层         = {r['mlp_per_layer']:>12,}")
    print(f"每个 Block            = {r['per_layer']:>12,}")
    print(f"Embedding(tied)     = {r['embed']:>12,}")
    print("-" * 48)
    print(f"总参数量             = {r['total']:>12,}  = {r['total'] / 1e6:.2f}M")
    if moe:
        act = r["total"] - (4 - 1) * nl * r["ffn_per_layer"]
        print(f"每 token 激活         ≈ {act / 1e6:.2f}M   (top-1 路由，只走 1 个专家)")


def check(r, hs, nl, moe):
    key = (r["arch"], hs, nl, moe)
    if key in OFFICIAL:
        exp = OFFICIAL[key]
        ok = abs(r["total"] / 1e6 - exp) < 0.05
        print(f"对照官方声明 {exp}M：{'一致 ✓' if ok else '不一致 ✗'}")


def main():
    p = argparse.ArgumentParser(description="XLLM / MiniMind 参数量核算")
    p.add_argument("--arch", choices=["minimind2", "minimind3", "auto"], default="auto",
                   help="minimind2=8/3公式+kv2+无QK-Norm；minimind3=pi公式+kv4+QK-Norm")
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--vocab_size", type=int, default=6400)
    p.add_argument("--use_moe", type=int, default=0)
    p.add_argument("--all", action="store_true", help="打印全部官方档位对照")
    args = p.parse_args()

    if args.all:
        print("MiniMind 官方谱系复算（与 README / HF 权重声明值比对）")
        print("=" * 78)
        rows = [
            ("minimind2", 512, 8, 0, "minimind2-small", "26M"),
            ("minimind2", 640, 8, 1, "minimind2-moe", "145M"),
            ("minimind2", 768, 16, 0, "minimind2", "104M"),
            ("minimind3", 768, 8, 0, "minimind-3", "64M"),
            ("minimind3", 768, 8, 1, "minimind-3-moe", "198M"),
        ]
        for arch, h, l, moe, name, label in rows:
            shared = 1 if (arch == "minimind2" and moe) else 0
            r = count(h, l, use_moe=bool(moe), arch=arch, n_shared_experts=shared)
            exp = OFFICIAL[(arch, h, l, moe)]
            flag = "✓" if abs(r["total"] / 1e6 - exp) < 0.05 else "✗"
            print(f"{name:17s} {label:>5s}  hidden={h:3d} L={l:2d} kv={r['kv_heads']} "
                  f"inter={r['intermediate_size']:5d} 实算={r['total'] / 1e6:7.2f}M {flag}")

        print("\n本仓库 XLLMConfig 默认值（hidden=512, 8 层, kv=2, 8/3 公式, 无 QK-Norm）")
        for moe in (0, 1):
            r = count(512, 8, use_moe=bool(moe), arch="minimind2",
                      n_routed_experts=4, n_shared_experts=1)
            tag = "MoE(4路由+1共享)" if moe else "Dense"
            print(f"  XLLM 512/8  {tag:16s} = {r['total'] / 1e6:7.2f}M")

        print("\n若把 XLLM 升级到 minimind-3 规格（768/8, kv=4, pi 公式, QK-Norm）")
        for moe in (0, 1):
            r = count(768, 8, use_moe=bool(moe), arch="minimind3", n_routed_experts=4)
            tag = "MoE(4路由)" if moe else "Dense"
            print(f"  XLLM 768/8  {tag:16s} = {r['total'] / 1e6:7.2f}M")
        return

    arch = "minimind2" if args.arch == "auto" else args.arch
    # 指定 arch 时，未显式覆盖的维度取该世代的官方默认值，
    # 否则 --arch minimind3 会沿用 minimind2 的 512 宽，得出无意义的数字。
    hidden = args.hidden_size
    if arch == "minimind3":
        hidden = 768 if args.hidden_size == 512 else args.hidden_size
    kv = 4 if arch == "minimind3" else 2
    shared = 0 if arch == "minimind3" else 1
    r = count(hidden, args.num_hidden_layers, args.vocab_size,
              num_key_value_heads=kv, use_moe=bool(args.use_moe),
              n_routed_experts=4, n_shared_experts=shared, arch=arch)
    render(r, hidden, args.num_hidden_layers, bool(args.use_moe))
    check(r, hidden, args.num_hidden_layers, args.use_moe)


if __name__ == "__main__":
    main()