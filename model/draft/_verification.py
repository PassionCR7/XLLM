# -*- coding: utf-8 -*-
"""
_verification.py —— 推测解码(Speculative Decoding)的“验证与接受”算法核心

本模块只依赖 PyTorch，不依赖主模型 / transformers，因此可以脱离
XLLM 主模型独立做单元测试，是整个草稿模型辅助推理的关键数学部分。

它实现了推测解码里最核心的一步：**用目标模型(主模型)的一次并行前向，
同时验证草稿模型提出的 gamma 个候选 token，决定接受哪几个、被拒后采样
哪个 token 作为输出**。这一套“提议-并行验证-接受/拒采”机制正是
Qwen / DeepSeek / Kimi 以及所有主流推测解码方案共享的骨架。

两种接受规则（对应两种解码目标）：
    1. greedy 贪心接受 —— 结果与“主模型贪心解码”完全一致，只是跑得更快；
    2. stochastic 随机接受 —— 结果与“主模型采样解码”在分布上无偏(unbiased)，
       保证不因草稿模型而破坏采样质量。参考 Chen et al. 2023 “Accelerating
       Large Language Model Decoding with Speculative Sampling”。

数学背景（一次验证回合）：
    设当前位置已知的上下文为前缀，草稿模型依次提出 d_1..d_gamma 共 gamma 个
    token（每个来自分布 q_i）。主模型把 [d_1..d_gamma] 整体拼进一个前向，得到
    目标分布 p_i（p_i 表示“在已接受 d_1..d_{i-1} 后，主模型认为下一个 token
    的分布”）。逐个位置判断：
        - 贪心：若 argmax p_i == d_i 则接受 d_i，否则在该位置停止并输出 argmax p_i；
        - 随机：以 min(1, p_i(d_i)/q_i(d_i)) 的概率接受 d_i；若被拒，从
                (p_i - q_i)_+ 归一化后的分布里采样作为输出，从而保证无偏。
    若 gamma 个全部接受，则再额外采样/取主模型对 d_gamma 之后一个位置的分布
    p_{gamma+1}，作为本轮“压哨”的最后一个 token。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def prepare_distribution(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """
    把原始 logits 转成可采样的概率分布：temperature 缩放 → top-k 截断 → top-p
    (nucleus) 截断 → softmax。

    参数
    ----
    logits : [..., vocab]
    temperature : 采样温度，>0。
    top_k : 只保留 logits 最高的前 top_k 个（其余置 -inf）。
    top_p : 只保留累积概率不超过 top_p 的最小 token 集合（nucleus）。

    返回
    ----
    probs : [..., vocab]，和为 1 的概率分布。
    """
    logits = logits.float().div(max(float(temperature), 1e-6))

    if top_k is not None and int(top_k) > 0:
        k = min(int(top_k), logits.size(-1))
        topk_val, _ = logits.topk(k, dim=-1)
        # 低于第 k 大值的 token 全部屏蔽
        logits = torch.where(
            logits >= topk_val[..., -1:],
            logits,
            torch.full_like(logits, float("-inf")),
        )

    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # 累积概率已经超过 top_p 的那部分 token 屏蔽（保留第一个刚好越过边界的 token）
        remove = (cum_probs - F.softmax(sorted_logits, dim=-1)) > float(top_p)
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        # scatter 把裁剪后的 logits 放回原始索引位置
        logits = sorted_logits.scatter(dim=-1, index=sorted_idx, src=sorted_logits)

    return F.softmax(logits, dim=-1)


def sample_token(
    logits: torch.Tensor,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """
    从一个 logits 张量中选出一个 token id。
    do_sample=False 时取 argmax（贪心）；do_sample=True 时按概率采样。
    """
    probs = prepare_distribution(logits, temperature, top_k, top_p)
    if do_sample:
        return torch.multinomial(probs, 1, generator=rng).squeeze(-1)
    return probs.argmax(dim=-1)


@torch.no_grad()
def verify_and_sample(
    target_logits: torch.Tensor,
    target_last_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    draft_ids: torch.Tensor,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    一次并行验证：决定草稿提出的 gamma 个 token 中接受几个，并给出“压哨”token。

    参数
    ----
    target_logits   : [B, gamma, vocab]  主模型一次前向得到的 logits。位置 i 预测的是
                      接受完 d_{i+1} 之后“下一个 token”的分布。
    target_last_logits : [B, vocab]      主模型在“当前最后已接受 token”位置上的 logits，
                      即预测第一个草稿 token d_1 的分布（由上轮前向携带）。
    draft_logits    : [B, gamma, vocab]  草稿模型每一步提议时采用的原始 logits，
                      用来还原提议分布 q_i。
    draft_ids       : [B, gamma]         草稿模型提出的候选 token id。

    返回
    ----
    num_accepted : [B] 长整型，每个 batch 元素接受的草稿 token 个数（0..gamma）。
    next_token   : [B] 长整型，本轮“压哨”输出 token（被接受草稿后紧跟的那一个，
                  由主模型分布决定）。
    """
    B, gamma, V = target_logits.shape
    device = target_logits.device
    gen = rng if rng is not None else torch.Generator(device=device)

    # p[:, 0] 预测第一个草稿 token d_1；p[:, i] 预测 d_{i+1}（i=1..gamma-1）；
    # p[:, gamma] 是“gamma 个全部接受后”的下一个 token 分布。
    p_full = torch.cat([target_last_logits.unsqueeze(1), target_logits], dim=1)
    p = prepare_distribution(p_full, temperature, top_k, top_p)     # [B, gamma+1, V]
    q = prepare_distribution(draft_logits, temperature, top_k, top_p)  # [B, gamma, V]

    count = torch.zeros(B, dtype=torch.long, device=device)   # 已接受的草稿 token 个数
    active = torch.ones(B, dtype=torch.bool, device=device)   # 该元素是否仍处于“逐位接受”状态
    rejected_next = torch.zeros(B, dtype=torch.long, device=device)  # 被拒时压哨 token
    any_reject = torch.zeros(B, dtype=torch.bool, device=device)

    for i in range(gamma):
        if not active.any():
            break
        p_i = p[:, i]              # 预测 d_{i+1}
        q_i = q[:, i]              # 草稿提出 d_{i+1} 时的分布
        d_i = draft_ids[:, i]

        p_d = p_i.gather(-1, d_i.unsqueeze(-1)).squeeze(-1)   # p_i(d_{i+1})
        q_d = q_i.gather(-1, d_i.unsqueeze(-1)).squeeze(-1)   # q_i(d_{i+1})

        if do_sample:
            # 随机接受：以 min(1, p/q) 的概率接受草稿 token
            accept_prob = torch.clamp(p_d / (q_d + 1e-20), max=1.0)
            is_accept = torch.rand(B, device=device, generator=gen) < accept_prob
        else:
            # 贪心接受：主模型 argmax 恰好等于草稿 token 才接受
            is_accept = p_i.argmax(dim=-1) == d_i

        is_accept = is_accept & active
        reject = active & ~is_accept

        if reject.any():
            if do_sample:
                # 随机模式：从 (p - q)_+ 归一化后的分布里采样，保证与“直接采样”无偏
                adjusted = (p_i - q_i).clamp_min(0.0)
                adjusted = adjusted / adjusted.sum(dim=-1, keepdim=True).clamp_min(1e-20)
                sampled = torch.multinomial(adjusted, 1, generator=gen).squeeze(-1)
            else:
                # 贪心模式：主模型在该位置上的 argmax 就是输出（Leviathan 2023）
                sampled = p_i.argmax(dim=-1)
            rejected_next[reject] = sampled[reject]
            any_reject[reject] = True

        count[is_accept] = i + 1          # 接受了 d_{i+1} 的元素，计数更新为 i+1
        active = active & is_accept       # 被拒元素变为不活跃

    # 压哨 token：
    #   - 所有 gamma 个草稿 token 都被接受(active=True) → 从 p[:, gamma] 采下一个 token；
    #   - 中间被拒 → 用拒采时采到的 rejected_next。
    p_next = p[:, gamma]
    if do_sample:
        sampled_next = torch.multinomial(p_next, 1, generator=gen).squeeze(-1)
    else:
        sampled_next = p_next.argmax(dim=-1)
    next_token = torch.where(active, sampled_next, rejected_next)

    return count, next_token


def acceptance_stats(accepted_per_iter: list[int], gamma: int) -> dict:
    """
    根据每轮实际接受的草稿 token 个数，汇总接受率与理论加速比。

    说明：一次验证回合主模型只需 1 次前向（并行验证 gamma 个 token），
    而普通解码要 gamma 步前向。因此“每轮实际产生的新 token 数”
    ≈ 1 + 平均接受数，就近似等于相对普通解码的加速比（未计草稿模型开销）。
    """
    if not accepted_per_iter:
        return {
            "n_iter": 0,
            "avg_accepted": 0.0,
            "acceptance_rate": 0.0,
            "tokens_per_iter": 1.0,
            "approx_speedup": 1.0,
        }
    n = len(accepted_per_iter)
    total_acc = sum(accepted_per_iter)
    avg_acc = total_acc / n
    # 接受率 = 接受的草稿 token / 提出的草稿 token（分母含 1 保底）
    denom = total_acc + (n * gamma - total_acc)
    return {
        "n_iter": n,
        "avg_accepted": round(avg_acc, 3),
        "acceptance_rate": round(total_acc / max(denom, 1), 3),
        "tokens_per_iter": round(1.0 + avg_acc, 3),
        "approx_speedup": round(1.0 + avg_acc, 3),
    }
