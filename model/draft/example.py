# -*- coding: utf-8 -*-
"""
example.py —— 草稿模型辅助主模型推理的可运行示例

演示 SpeculativeDecodingEngine 的完整使用：
    1) 从主模型截断出零训练草稿；
    2) 构建推测解码引擎；
    3) 对比“普通解码”与“推测解码”的输出（贪心模式应逐 token 一致）；
    4) 打印接受率与近似加速比。

用法：
    python model/draft/example.py                  # 用随机权重的小模型快速跑通链路
    python model/draft/example.py --ckpt checkpoints/pretrain_512.pth
                                                  # 加载真实主模型权重
"""

import argparse
import os
import sys

# 保证可以从仓库根目录 import model
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from model.XLLMModel import XLLMConfig, XLLMForCausalLM
from model.draft import DraftModel, SpeculativeDecodingEngine


def build_main(ckpt: str | None, device: torch.device):
    """构建主模型：有 checkpoint 则加载，否则随机权重。"""
    cfg = XLLMConfig(
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        vocab_size=6400,
        max_position_embeddings=1024,
    )
    model = XLLMForCausalLM(cfg).to(device).eval()
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"[main] 已加载权重: {ckpt}")
    else:
        print("[main] 使用随机权重（未加载 checkpoint，仅用于跑通链路）")
    return model


def vanilla_greedy(model: XLLMForCausalLM, input_ids: torch.Tensor, max_new: int, eos: int):
    """普通贪心解码（对比基线）：手工自回归循环，逐 token 生成。"""
    kv = None
    ids = input_ids
    out = []
    for _ in range(max_new):
        o = model(input_ids=ids, past_key_values=kv, use_cache=True, logits_to_keep=1)
        tok = o.logits[:, -1].argmax(-1).item()
        kv = o.past_key_values
        out.append(tok)
        ids = torch.tensor([[tok]], dtype=torch.long, device=input_ids.device)
        if tok == eos:
            break
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="主模型 checkpoint 路径")
    parser.add_argument("--gamma", type=int, default=4, help="每回合提议的候选 token 数")
    parser.add_argument("--max-new", type=int, default=32, help="生成 token 数上限")
    parser.add_argument("--do-sample", action="store_true", help="随机无偏采样（默认贪心）")
    parser.add_argument("--draft-layers", type=int, default=4, help="截断草稿的层数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main_model = build_main(args.ckpt, device)

    # 零训练截断草稿：复用主模型前 draft_layers 层（需与主模型同设备）
    draft = DraftModel.from_main_model(main_model, args.draft_layers).to(device)
    print(f"[draft] 截断草稿: {args.draft_layers}/{main_model.config.num_hidden_layers} 层, "
          f"参数量 {draft.num_parameters()/1e6:.2f}M")

    engine = SpeculativeDecodingEngine(
        target=main_model,
        draft=draft,
        gamma=args.gamma,
        do_sample=args.do_sample,
        eos_token_id=2,
        seed=args.seed,
    )

    # 简单 prompt（已编码；随机权重模型下仅用于演示流程）
    prompt = torch.tensor([1, 17, 260, 512, 900, 88], dtype=torch.long, device=device)
    prompt = prompt.unsqueeze(0)   # [1, L]

    if not args.do_sample:
        print("\n===== 贪心对比：普通解码 vs 推测解码（应逐 token 一致）=====")
        ref = vanilla_greedy(main_model, prompt, args.max_new, 2)
        res = engine.generate(prompt, max_new_tokens=args.max_new)
        gen = res["sequences"][0]          # 二维输入 → sequences 是列表的列表，取第 0 条
        print(f"普通解码 : {ref}")
        print(f"推测解码 : {gen}")
        match = gen[: len(ref)] == ref[: len(gen)]
        print(f"一致性   : {'一致 ✓' if match else '不一致 ✗'}  (等价性验证)")
    else:
        print("\n===== 随机无偏采样：推测解码 =====")
        res = engine.generate(prompt, max_new_tokens=args.max_new)
        print(f"推测解码 : {res['sequences'][0]}")

    print(f"\n===== 统计 =====")
    stats = res["stats"]
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
