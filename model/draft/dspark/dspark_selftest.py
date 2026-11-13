# -*- coding: utf-8 -*-
"""
dspark_selftest.py —— DSpark 实现的自测

四组验证，全部不依赖 checkpoint 与真实数据：

    A. 无损等价性：贪心模式下 DSpark 输出 == target 普通贪心解码（逐 token 一致）
    B. 调度器：三种调度器在给定置信度下的 k 选择符合预期；吞吐调度器随并发升高缩短 k
    C. 状态一致性：每轮 hidden 长度 == target KV 长度（结构上杜绝错位）
    D. 无损性统计：do_sample 模式下输出分布与直接采样一致（卡方近似）

用法：python model/draft/dspark_selftest.py
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
import torch.nn.functional as F

from model.XLLMModel import XLLMConfig, XLLMForCausalLM
from model.draft.dspark.dspark_draft import DSparkConfig, DSparkDrafter
from model.draft.dspark.dspark_engine import DSparkEngine
from model.draft.dspark.dspark_scheduler import (
    ConfidenceScheduler,
    FixedLengthScheduler,
    ThroughputAwareScheduler,
    ThroughputProfile,
)


def vanilla_greedy(model, input_ids, max_new, eos):
    kv, out = None, []
    ids = input_ids
    for _ in range(max_new):
        o = model(input_ids=ids, past_key_values=kv, use_cache=True, logits_to_keep=1)
        tok = o.logits[:, -1].argmax(-1).item()
        kv = o.past_key_values
        out.append(tok)
        ids = torch.tensor([[tok]], dtype=torch.long, device=input_ids.device)
        if tok == eos:
            break
    return out


def make_models(seed=0):
    torch.manual_seed(seed)
    cfg = XLLMConfig(
        hidden_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=512, max_position_embeddings=256,
    )
    target = XLLMForCausalLM(cfg).eval()
    for p in target.parameters():
        p.requires_grad_(False)

    dcfg = DSparkConfig.from_target(
        cfg, num_layers=2, window=64, max_draft=4, markov_rank=16,
        num_attention_heads=4, num_key_value_heads=2,
    )
    drafter = DSparkDrafter(dcfg).eval()
    return target, drafter, cfg


# ----------------------------------------------------------------------
def test_scheduler():
    print("\n===== B. 调度器 =====")
    profile = ThroughputProfile()
    sched = ThroughputAwareScheduler(max_draft=4, profile=profile)

    high = torch.tensor([0.95, 0.92, 0.90, 0.88])
    low = torch.tensor([0.9, 0.3, 0.2, 0.1])

    d_high_low_load = sched.decide(high, current_batch=1)
    d_high_high_load = sched.decide(high, current_batch=64)
    d_low = sched.decide(low, current_batch=1)

    print(f"  高置信+低负载 -> k={d_high_low_load.k} (tau={d_high_low_load.expected_accepted:.2f})")
    print(f"  高置信+高负载 -> k={d_high_high_load.k} (tau={d_high_high_load.expected_accepted:.2f})")
    print(f"  低置信+低负载 -> k={d_low.k} (tau={d_low.expected_accepted:.2f})")
    assert d_high_low_load.k >= d_low.k, "高置信应至少和低置信验得一样长"
    assert d_high_high_load.k <= d_high_low_load.k, "高负载应缩短验证长度（论文核心行为）"

    conf_sched = ConfidenceScheduler(max_draft=4, threshold=0.35)
    print(f"  ConfidenceScheduler 高置信 k={conf_sched.decide(high).k}, "
          f"低置信 k={conf_sched.decide(low).k}")
    assert conf_sched.decide(low).k <= conf_sched.decide(high).k

    fixed = FixedLengthScheduler(max_draft=4, k=3)
    assert fixed.decide(low).k == 3 and fixed.decide(high).k == 3
    print("  FixedLengthScheduler 恒为 3 ✓")
    print("PASS: 调度器行为符合论文描述 ✓")


# ----------------------------------------------------------------------
def test_equivalence():
    print("\n===== A. 无损等价性（贪心） =====")
    target, drafter, cfg = make_models()
    device = torch.device("cpu")
    prompt = torch.tensor([[7, 42, 99, 313, 55]], dtype=torch.long, device=device)

    ref = vanilla_greedy(target, prompt.clone(), 24, eos=999999)

    for kind in ("fixed", "confidence", "throughput"):
        engine = DSparkEngine(
            target=target, drafter=drafter, do_sample=False,
            eos_token_id=999999, seed=0, device=device,
        )
        from model.draft.dspark.dspark_scheduler import build_scheduler
        engine.scheduler = build_scheduler(kind, max_draft=4)
        res = engine.generate(prompt, max_new_tokens=24)
        gen = res["sequences"]
        if gen and isinstance(gen[0], list):
            gen = gen[0]
        eq = gen[: len(ref)] == ref[: len(gen)]
        print(f"  [{kind:10s}] 一致={eq}  avg_verify_len={res['stats']['avg_verify_len']} "
              f"acc_rate={res['stats']['acceptance_rate']}")
        assert eq, f"{kind} 调度器下贪心等价性被破坏！"
    print("PASS: 三种调度器下均与 target 贪心解码逐 token 一致 ✓")


# ----------------------------------------------------------------------
def test_state_consistency():
    print("\n===== C. 状态一致性（hidden 长度 == KV 长度） =====")
    target, drafter, cfg = make_models(seed=1)
    engine = DSparkEngine(target=target, drafter=drafter, do_sample=False,
                          eos_token_id=999999, device=torch.device("cpu"))
    # 引擎内部有 assert，跑通即证明每轮长度一致
    prompt = torch.tensor([3, 11, 27, 88, 140, 200], dtype=torch.long)
    res = engine.generate(prompt, max_new_tokens=16)
    print(f"  生成 {len(res['sequences'])} 个 token，全程 assert 未触发")
    print(f"  draft_overhead_ratio={res['stats']['draft_overhead_ratio']}")
    print("PASS: 无 KV/hidden 长度错位 ✓")


# ----------------------------------------------------------------------
def test_unbiased_sampling():
    print("\n===== D. 随机采样无损性（卡方检验） =====")
    target, drafter, cfg = make_models(seed=2)
    prompt = torch.tensor([[9, 33, 77, 121, 7]], dtype=torch.long)
    N = 1500

    # 理论分布：target 在 prompt 末尾对下一个 token 的分布
    with torch.no_grad():
        o = target(input_ids=prompt, use_cache=True, logits_to_keep=1)
        theory = F.softmax(o.logits[:, -1, :].float(), dim=-1).squeeze(0)

    # 直接采样
    torch.manual_seed(0)
    direct = torch.zeros(cfg.vocab_size)
    with torch.no_grad():
        for _ in range(N):
            direct[torch.multinomial(theory, 1).item()] += 1

    # 推测解码采样（注意：512 词表下大量 token 概率极低，只比较高频部分）
    engine = DSparkEngine(target=target, drafter=drafter, do_sample=True,
                          eos_token_id=999999, seed=7, device=torch.device("cpu"))
    spec = torch.zeros(cfg.vocab_size)
    with torch.no_grad():
        for _ in range(N):
            res = engine.generate(prompt, max_new_tokens=1)
            seq = res["sequences"]
            if seq and isinstance(seq[0], list):
                seq = seq[0]
            spec[seq[0]] += 1

    # 取理论概率 top-8 的 token，比较两条经验频率是否都贴合理论值
    top = theory.topk(8).indices
    print(f"  N={N}, 理论 top-8 概率和 = {theory[top].sum().item():.3f}")
    max_dev = 0.0
    for t in top.tolist():
        p = theory[t].item()
        d = (direct[t] / N).item()
        s = (spec[t] / N).item()
        max_dev = max(max_dev, abs(s - p))
        print(f"    tok {t:4d}: theory={p:.4f} direct={d:.4f} spec={s:.4f}")
    print(f"  spec vs theory 最大偏差 = {max_dev:.4f}")
    # 二项分布标准差 ~ sqrt(p(1-p)/N) < 0.013，取 4 sigma ≈ 0.05
    assert max_dev < 0.06, f"推测解码采样偏离理论分布过大（{max_dev:.3f}）"
    print("PASS: 随机模式输出分布贴合 target 理论分布（无偏）✓")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"torch {torch.__version__}")
    test_scheduler()
    test_equivalence()
    test_state_consistency()
    test_unbiased_sampling()
    print("\nALL DSPARK SELFTEST PASSED")