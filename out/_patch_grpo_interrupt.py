# -*- coding: utf-8 -*-
import textwrap

path = r"D:\workspace\xllm\trainer\train_grpo.py"
with open(path, encoding="utf-8") as f:
    src = f.read()

# ========== 1) 模块级：训练进度记录（Ctrl+C 优雅保存用） ==========
old_1 = 'warnings.filterwarnings("ignore")\n\n\ndef calculate_rewards'
new_1 = 'warnings.filterwarnings("ignore")\n\n\n# 记录当前训练进度，供 Ctrl+C 优雅退出时保存\n_rl_state = {"epoch": 0, "step": 0}\n\n\ndef calculate_rewards'
assert src.count(old_1) == 1, f"1 found {src.count(old_1)}"
src = src.replace(old_1, new_1)

# ========== 2) grpo_train_epoch 循环体：记录当前 step/epoch ==========
old_2 = """    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]"""
new_2 = """    for step, batch in enumerate(loader, start=start_step + 1):
        _rl_state["step"] = step
        _rl_state["epoch"] = epoch
        prompts = batch["prompt"]"""
assert src.count(old_2) == 1, f"2 found {src.count(old_2)}"
src = src.replace(old_2, new_2)

# ========== 3) 主循环外包 try/except KeyboardInterrupt，中断时保存当前权重 ==========
old_3 = """    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)

        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), args.batch_size, start_step
            )
            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始"
            )
            grpo_train_epoch(
                epoch,
                loader,
                len(loader) + start_step,
                ref_model,
                reward_model,
                reward_tokenizer,
                start_step,
                wandb,
            )
        else:
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                pin_memory=True,
                drop_last=False,
                shuffle=(train_sampler is None),
                num_workers=args.num_workers,
                sampler=train_sampler,
            )
            grpo_train_epoch(
                epoch,
                loader,
                len(loader),
                ref_model,
                reward_model,
                reward_tokenizer,
                0,
                wandb,
            )"""

block = old_3
indented = textwrap.indent(block, "    ")  # 整个 for-epoch 块整体缩进 4 格放进 try:
new_3 = "    try:\n" + indented + """    except KeyboardInterrupt:
        # Ctrl+C 优雅退出：把当前权重与续训点完整保存后再退出
        Logger(
            f"KeyboardInterrupt 收到，正在保存当前权重（Epoch {_rl_state['epoch'] + 1}, Step {_rl_state['step']}）..."
        )
        model.eval()
        moe_suffix = "_moe" if lm_config.use_moe else ""
        ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
        state_dict = (
            model.module.state_dict()
            if isinstance(model, DistributedDataParallel)
            else model.state_dict()
        )
        torch.save({k: v.half() for k, v in state_dict.items()}, ckp + ".tmp")
        os.replace(ckp + ".tmp", ckp)
        lm_checkpoint(
            lm_config,
            weight=args.save_weight,
            model=model,
            optimizer=optimizer,
            epoch=_rl_state["epoch"],
            step=_rl_state["step"],
            wandb=wandb,
            save_dir="../checkpoints",
            scheduler=scheduler,
        )
        model.train()
        Logger(
            f"已保存 {ckp} 与 checkpoints/{args.save_weight}_{lm_config.hidden_size}_resume.pth，"
            "之后可用 --from_resume=1 从该点续训"
        )
        raise SystemExit(0)"""
assert src.count(old_3) == 1, f"3 found {src.count(old_3)}"
src = src.replace(old_3, new_3)

with open(path, "w", encoding="utf-8") as f:
    f.write(src)
print("OK, graceful Ctrl+C save added")
