"""参数量核算脚本：不依赖 torch，纯手工推演 XLLMModel 的结构。

用途是随时复核 README 中的参数量表述。默认配置与 trainer/train_pretrain.py 一致：
hidden_size=512, num_hidden_layers=8, vocab_size=6400。

用法：
    python count_params.py
    python count_params.py --hidden_size 768 --num_hidden_layers 16
"""

import argparse


def ffn_intermediate(hidden_size: int) -> int:
    """与 XLLMModel.FeedForward.__init__ 完全一致：先取 float 再上取整到 64 的倍数。"""
    inter = int(hidden_size * 8 / 3)
    return 64 * ((inter + 64 - 1) // 64)


def count(hidden_size=512, num_hidden_layers=8, vocab_size=6400,
          num_attention_heads=8, num_key_value_heads=2,
          use_moe=False, n_routed_experts=4, n_shared_experts=1):
    head_dim = hidden_size // num_attention_heads

    # ---- Attention（GQA）----
    q = hidden_size * (num_attention_heads * head_dim)
    k = hidden_size * (num_key_value_heads * head_dim)
    v = hidden_size * (num_key_value_heads * head_dim)
    o = (num_attention_heads * head_dim) * hidden_size
    attn = q + k + v + o

    # ---- FFN（SwiGLU: gate / up / down，均无 bias）----
    inter = ffn_intermediate(hidden_size)
    ffn_one = 3 * hidden_size * inter  # gate + up + down

    # ---- 每个 Block ----
    norms = 2 * hidden_size  # input_layernorm + post_attention_layernorm
    if use_moe:
        ffn = ffn_one * (n_routed_experts + n_shared_experts)
        gate = n_routed_experts * hidden_size  # MoEGate.weight，无 bias
    else:
        ffn = ffn_one
        gate = 0
    per_layer = attn + ffn + norms + gate

    # ---- 全局 ----
    embed = vocab_size * hidden_size  # 与 lm_head 共享，只计一次
    final_norm = hidden_size

    total = embed + num_hidden_layers * per_layer + final_norm
    return {
        "head_dim": head_dim,
        "intermediate_size": inter,
        "attn_per_layer": attn,
        "ffn_per_layer": ffn_one,
        "per_layer": per_layer,
        "ffn_total": num_hidden_layers * ffn_one,
        "embed": embed,
        "total": total,
        "ffn_share": num_hidden_layers * ffn_one / total,
    }


def main():
    p = argparse.ArgumentParser(description="XLLM 参数量核算")
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--vocab_size", type=int, default=6400)
    p.add_argument("--use_moe", type=int, default=0)
    args = p.parse_args()

    r = count(args.hidden_size, args.num_hidden_layers, args.vocab_size,
              use_moe=bool(args.use_moe))

    print(f"hidden_size       = {args.hidden_size}")
    print(f"num_hidden_layers = {args.num_hidden_layers}")
    print(f"intermediate_size = {r['intermediate_size']}")
    print(f"head_dim          = {r['head_dim']}")
    print("-" * 42)
    print(f"Attention / 层     = {r['attn_per_layer']:>12,}")
    print(f"FFN / 层           = {r['ffn_per_layer']:>12,}")
    print(f"每层合计           = {r['per_layer']:>12,}")
    print(f"Embedding(tied)   = {r['embed']:>12,}")
    print(f"FFN 合计           = {r['ffn_total']:>12,}  ({r['ffn_share']:.1%} of total)")
    print("-" * 42)
    print(f"总参数量           = {r['total']:>12,}  = {r['total']/1e6:.2f}M")

    ref = {"512/8/dense": 25.83, "512/8/moe": 95.05}
    key = f"{args.hidden_size}/{args.num_hidden_layers}/{'moe' if args.use_moe else 'dense'}"
    if key in ref:
        ok = abs(r["total"] / 1e6 - ref[key]) < 0.01
        print(f"与 bench_result.json 实测 {ref[key]}M 对比：{'一致 ✓' if ok else '不一致 ✗'}")


if __name__ == "__main__":
    main()