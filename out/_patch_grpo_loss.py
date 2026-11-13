# -*- coding: utf-8 -*-
path = r"D:\workspace\xllm\trainer\train_grpo.py"
with open(path, encoding="utf-8") as f:
    src = f.read()

# ========== A) 在 rollout 后捕获旧策略 logps（GRPO 重要性比率用） ==========
old_a = """        per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))
        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(
                ref_model, outputs, completion_ids.size(1)
            )"""
new_a = """        # 记录 rollout 时的行为策略（旧策略）logps，用于 GRPO 重要性比率 exp(new - old)
        with torch.no_grad():
            old_per_token_logps = get_per_token_logps(
                model, outputs, completion_ids.size(1)
            )
        per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))
        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(
                ref_model, outputs, completion_ids.size(1)
            )"""
assert src.count(old_a) == 1, f"A found {src.count(old_a)}"
src = src.replace(old_a, new_a)

# ========== B) 替换损失为 minimind 风格的 GRPO/CISPO 目标 ==========
old_b = """        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        per_token_loss = -(
            torch.exp(per_token_logps - per_token_logps.detach())
            * advantages.unsqueeze(1)
            - args.beta * per_token_kl
        )

        loss = (
            (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        ).mean() / args.accumulation_steps
        loss.backward()"""
new_b = """        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        # 重要性比率：exp(新策略logp - 旧策略logp)，旧策略logp在rollout时捕获
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        if args.loss_type == "cispo":
            # CISPO：对比率上界截断（detach），loss = -(clamped_ratio * adv * logp - beta*kl)
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(
                clamped_ratio * advantages.unsqueeze(1) * per_token_logps
                - args.beta * per_token_kl
            )
        else:
            # GRPO（PPO-clip 风格）：min(ratio, clipped(ratio)) * adv
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(
                torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl
            )

        loss = (
            (per_token_loss * completion_mask).sum(dim=1)
            / completion_mask.sum(dim=1).clamp(min=1)
        ).mean() / args.accumulation_steps
        loss.backward()"""
assert src.count(old_b) == 1, f"B found {src.count(old_b)}"
src = src.replace(old_b, new_b)

# ========== C) 日志：policy loss 乘以 accumulation_steps（与 minimind 一致） ==========
old_c = """            policy_loss_val = loss.item()"""
new_c = """            policy_loss_val = loss.item() * args.accumulation_steps"""
assert src.count(old_c) == 1, f"C found {src.count(old_c)}"
src = src.replace(old_c, new_c)

# ========== D) 释放 old_per_token_logps ==========
old_d = """        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps"""
new_d = """        del (
            prompt_inputs,
            outputs,
            completion_ids,
            per_token_logps,
            ref_per_token_logps,
            old_per_token_logps,
        )"""
assert src.count(old_d) == 1, f"D found {src.count(old_d)}"
src = src.replace(old_d, new_d)

# ========== E) 新增 --loss_type / --epsilon / --epsilon_high 参数 ==========
old_e = """    parser.add_argument("--beta", type=float, default=0.02, help="KL惩罚系数")"""
new_e = """    parser.add_argument("--beta", type=float, default=0.02, help="KL惩罚系数")
    parser.add_argument(
        "--loss_type",
        type=str,
        default="cispo",
        choices=["grpo", "cispo"],
        help="loss类型（cispo=minimind默认，grpo=PPO-clip风格）",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon"
    )
    parser.add_argument(
        "--epsilon_high", type=float, default=5.0, help="epsilon上界"
    )"""
assert src.count(old_e) == 1, f"E found {src.count(old_e)}"
src = src.replace(old_e, new_e)

with open(path, "w", encoding="utf-8") as f:
    f.write(src)
print("OK, all GRPO loss patches applied")
