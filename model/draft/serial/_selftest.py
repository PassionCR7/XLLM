# -*- coding: utf-8 -*-
"""
_selftest.py —— 推测解码链路自测

不依赖 checkpoint，用随机权重的小模型验证两件事：
    1) 贪心模式下，推测解码输出 == 普通贪心解码输出（逐 token 一致，等价性验证）；
    2) 随机模式下，输出落在合法 token 范围内、统计字段完整。

用法：  python model/draft/_selftest.py
"""

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

from model.XLLMModel import XLLMConfig, XLLMForCausalLM
from model.draft.serial import DraftModel, SpeculativeDecodingEngine


def vanilla_greedy(model, input_ids, max_new, eos):
    """普通自回归贪心解码（基线），手工循环以避开 GenerationMixin 的额外约束。"""
    kv, out = None, []
    ids = input_ids
    for _ in range(max_new):
        o = model(input_ids=ids, past_key_values=kv, use_cache=True, logits_to_keep=1)
        tok = o.logits[:, -1].argmax(-1).item()
        kv = o.past_key_values
        out.append(tok)
        ids = torch.tensor([[tok]], dtype=torch.long)
        if tok == eos:
            break
    return out


def main():
    torch.manual_seed(0)
    device = torch.device("cpu")

    cfg = XLLMConfig(
        hidden_size=256, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=1000, max_position_embeddings=256,
    )
    main = XLLMForCausalLM(cfg).to(device).eval()
    draft = DraftModel.from_main_model(main, num_draft_layers=3)

    print(f"[selftest] main.layers={main.config.num_hidden_layers} "
          f"draft.layers={draft.config.num_hidden_layers} "
          f"vocab={cfg.vocab_size}")

    prompt = torch.tensor([[7, 42, 99, 313, 555]], dtype=torch.long)   # [1, L]，模型需 2D 输入

    # ---- 1) 贪心等价性 ----
    ref = vanilla_greedy(main, prompt.clone(), 24, 2)
    for gamma in (3, 5):
        engine = SpeculativeDecodingEngine(
            main, draft, gamma=gamma, do_sample=False, eos_token_id=2, seed=0
        )
        res = engine.generate(prompt, max_new_tokens=24)
        gen = res["sequences"][0]        # 2D 输入 → sequences 是列表的列表，取第 0 条
        eq = gen[: len(ref)] == ref[: len(gen)]
        print(f"  [gamma={gamma}] 推测={gen}")
        print(f"  [gamma={gamma}] 普通={ref}")
        print(f"  [gamma={gamma}] 逐token一致={eq}")
        assert eq, "贪心模式下推测解码与普通解码不一致！"
        print(f"  [gamma={gamma}] stats={res['stats']}")
    print("PASS: 贪心等价性 ✓\n")

    # ---- 1b) 草稿=全模型层数：应高接受率，验证“全部接受”路径与加速比 ----
    draft_full = DraftModel.from_main_model(main, num_draft_layers=main.config.num_hidden_layers)
    engine_full = SpeculativeDecodingEngine(
        main, draft_full, gamma=4, do_sample=False, eos_token_id=2, seed=0
    )
    res_full = engine_full.generate(prompt, max_new_tokens=24)
    gen_full = res_full["sequences"][0]
    eq_full = gen_full[: len(ref)] == ref[: len(gen_full)]
    print(f"  [草稿=全层] 逐token一致={eq_full}")
    print(f"  [草稿=全层] stats={res_full['stats']}")
    assert eq_full, "草稿=全层时推测解码应与普通解码一致！"
    assert res_full["stats"]["avg_accepted"] > 0, "草稿=全层时应有正接受率！"
    print("PASS: 全接受路径 / 加速比统计 ✓\n")

    # ---- 2) 随机模式 ----
    engine = SpeculativeDecodingEngine(
        main, draft, gamma=4, do_sample=True, temperature=0.8, top_k=50,
        eos_token_id=2, seed=42,
    )
    res = engine.generate(prompt, max_new_tokens=16)
    gen = res["sequences"][0]
    assert all(0 <= t < cfg.vocab_size for t in gen), "随机模式输出 token 越界！"
    assert "stats" in res and "approx_speedup" in res["stats"]
    print(f"  随机推测={gen}")
    print(f"  stats={res['stats']}")
    print("PASS: 随机采样模式 ✓\n")

    print("ALL SELFTEST PASSED")


if __name__ == "__main__":
    main()
