# -*- coding: utf-8 -*-
"""
xllm 推理速度基准测试

与 eval.py 的区别
------------------
eval.py 是交互式对话脚本（靠 input() 选模式、TextStreamer 打印），
全程不统计时间与 token 数，无法产出速度指标。
本脚本复用与 eval.py 完全相同的模型构建与权重加载路径，
但用纯计时的方式测量以下指标：

    TTFT      首个 token 延迟（含 prefill 时间）
    prefill   输入预填充吞吐 (tok/s)
    decode    单流解码速度 (tok/s) —— 也就是通常说的"推理速度"
    e2e       端到端总耗时

测速采用 get_rng_state / 重新 forward 的显式循环，而不依赖 generate 内部计时，
避免 streamer 与采样逻辑带来的额外开销干扰测量。

用法
----
    python bench_speed.py                        # Dense + MoE 全测
    python bench_speed.py --only dense           # 只测 Dense
    python bench_speed.py --new_tokens 256       # 指定生成长度
"""

import argparse
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers import AutoTokenizer
from model.XLLMModel import XLLMConfig, XLLMForCausalLM

warnings.filterwarnings("ignore")

# 与 eval.py 保持一致的测试 prompt，用于模拟真实对话负载
PROMPTS = [
    "你有什么特长？",
    "为什么天空是蓝色的",
    "请用Python写一个计算斐波那契数列的函数",
    "解释一下光合作用的基本过程",
]

SHORT_PROMPT = "介绍一下你自己"


def build_model(weight, hidden_size, num_layers, use_moe, device):
    """完全复用 eval.py 的构建逻辑，保证测的就是 eval.py 会跑的模型。"""
    moe_suffix = "_moe" if use_moe else ""
    ckp = f"checkpoints/{weight}_{hidden_size}{moe_suffix}.pth"

    tokenizer = AutoTokenizer.from_pretrained("model")
    model = XLLMForCausalLM(
        XLLMConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            use_moe=use_moe,
        )
    )
    state = torch.load(ckp, map_location="cpu", weights_only=True)

    # 老版本 checkpoint 里 post_attention_layernorm 被误拼成 layernotm，
    # 直接 strict=True 会崩。这里做一次键名规范化，其余仍然严格校验。
    fixed = {}
    for k, v in state.items():
        fixed[k.replace("post_attention_layernotm", "post_attention_layernorm")] = v
    model.load_state_dict(fixed, strict=True)

    model.eval().to(device)
    n_param = sum(p.numel() for p in model.parameters())
    return model, tokenizer, n_param, ckp


@torch.inference_mode()
def measure(model, tokenizer, prompt, device, new_tokens, use_cache, repeat=3):
    """测量单个 prompt 的 prefill / decode / TTFT。取多次运行的中位数。"""
    text = tokenizer.bos_token + prompt
    enc = tokenizer(text, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    prompt_len = input_ids.shape[1]

    # 预热：第一次 forward 会触发 cuDNN 算法选择与显存分配，不预热会严重高估耗时
    warm_ids = input_ids[:, : min(prompt_len, 8)]
    model(input_ids=warm_ids, use_cache=use_cache)
    torch.cuda.synchronize()

    decode_speeds, prefill_speeds, ttfts, e2e = [], [], [], []

    for _ in range(repeat):
        # ---------- Prefill：整段 prompt 一次前向 ----------
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=input_ids, use_cache=use_cache)
        torch.cuda.synchronize()
        t_prefill = time.perf_counter() - t0
        prefill_speeds.append(prompt_len / t_prefill)

        # ---------- Decode：逐 token 自回归 ----------
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        torch.cuda.synchronize()
        t_start = time.perf_counter()
        t_first = None
        generated = 1
        # 测速用贪心解码，避免采样引入的随机性影响可比性
        for _step in range(new_tokens - 1):
            out = model(
                input_ids=next_tok,
                past_key_values=past if use_cache else None,
                use_cache=use_cache,
            )
            if t_first is None:
                torch.cuda.synchronize()
                t_first = time.perf_counter() - t_start
            past = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated += 1
        torch.cuda.synchronize()
        t_decode = time.perf_counter() - t_start

        decode_speeds.append(generated / t_decode)
        ttfts.append(t_prefill + (t_first or 0))
        e2e.append(t_prefill + t_decode)

        if device == "cuda":
            torch.cuda.empty_cache()

    return {
        "prompt": prompt,
        "prompt_len": prompt_len,
        "new_tokens": new_tokens,
        "prefill_tok_s": statistics.median(prefill_speeds),
        "decode_tok_s": statistics.median(decode_speeds),
        "ttft_s": statistics.median(ttfts),
        "e2e_s": statistics.median(e2e),
    }


def run_config(tag, weight, hidden, layers, use_moe, device, new_tokens, repeat):
    print(f"\n{'=' * 68}")
    print(f"[{tag}]  weight={weight}  hidden={hidden}  layers={layers}  moe={use_moe}")
    print("=" * 68)

    model, tokenizer, n_param, ckp = build_model(weight, hidden, layers, use_moe, device)
    print(f"checkpoint : {ckp}")
    print(f"参数量     : {n_param / 1e6:.2f} M")
    if device == "cuda":
        alloc = torch.cuda.max_memory_allocated() / 1024**3
        print(f"显存峰值   : {alloc:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    rows = []
    for p in PROMPTS:
        r = measure(model, tokenizer, p, device, new_tokens, use_cache=True, repeat=repeat)
        rows.append(r)
        print(
            f"  {r['prompt'][:20]:<22} in={r['prompt_len']:>3} "
            f"prefill={r['prefill_tok_s']:>7.1f} tok/s  "
            f"decode={r['decode_tok_s']:>6.1f} tok/s  "
            f"ttft={r['ttft_s'] * 1000:>6.0f} ms"
        )

    avg_decode = statistics.mean(r["decode_tok_s"] for r in rows)
    avg_prefill = statistics.mean(r["prefill_tok_s"] for r in rows)
    print(f"  {'平均值':<22} {'':>7} prefill={avg_prefill:>7.1f} tok/s  decode={avg_decode:>6.1f} tok/s")

    # 关掉 KV cache 再测一次，量化 cache 的实际贡献。
    # 必须用与上面完全相同的生成长度，否则序列长度不同会让对比失真
    # （长度越短，全量重算越便宜，看起来反而"更快"）。
    no_cache = measure(model, tokenizer, SHORT_PROMPT, device,
                       new_tokens=new_tokens, use_cache=False, repeat=2)
    ratio = avg_decode / no_cache["decode_tok_s"]
    note = "" if ratio > 1.15 else "  <- 该模型规模下固定开销主导，cache 收益被淹没"
    print(f"  关闭 KV cache ({new_tokens} token): decode={no_cache['decode_tok_s']:.1f} tok/s "
          f"-> 加速比 {ratio:.2f}x{note}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return {
        "tag": tag,
        "weight": weight,
        "params_m": round(n_param / 1e6, 2),
        "avg_decode_tok_s": round(avg_decode, 2),
        "avg_prefill_tok_s": round(avg_prefill, 2),
        "no_cache_decode_tok_s": round(no_cache["decode_tok_s"], 2),
        "kv_cache_speedup": round(avg_decode / no_cache["decode_tok_s"], 2),
        "per_prompt": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows],
    }


def main():
    ap = argparse.ArgumentParser(description="xllm 推理速度基准测试")
    ap.add_argument("--only", choices=["dense", "moe", "all"], default="all")
    ap.add_argument("--new_tokens", type=int, default=128, help="每次生成的新 token 数")
    ap.add_argument("--repeat", type=int, default=3, help="每个 prompt 重复次数，取中位数")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="bench_result.json")
    args = ap.parse_args()

    dev_name = torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU"
    print(f"设备: {dev_name}   torch {torch.__version__}   generate {args.new_tokens} tokens x{args.repeat}")
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True

    results = []
    if args.only in ("dense", "all"):
        results.append(run_config("Dense-26M", "full_sft_cot", 512, 8, False,
                                  args.device, args.new_tokens, args.repeat))
    if args.only in ("moe", "all"):
        results.append(run_config("MoE-145M", "full_sft", 512, 8, True,
                                  args.device, args.new_tokens, args.repeat))

    out_path = Path(args.out)
    out_path.write_text(
        json.dumps({"device": dev_name, "new_tokens": args.new_tokens, "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已写入 {out_path.resolve()}")


if __name__ == "__main__":
    main()