# -*- coding: utf-8 -*-
"""
mtp_pretrain.py —— 主干 + MTP 草稿头的**联合预训练**

这不是"训练完主模型再补草稿"（那是 dspark_train.py 的事后路线），而是从一开始就
让 MTP 损失参与主干的反向传播。对齐 DeepSeek-V3/V4 的做法：

    L_total = L_next_token + (λ/D) · Σ_{k=1..D} L_MTP^(k)

两个收益（论文均明确指出，且缺一不可）：
    1) 推理侧白拿一个草稿头，无需额外训练阶段；
    2) 训练侧 densify 监督信号——每个隐状态被迫同时支撑近处与远处预测，
       主干本身质量提升。这一点是事后训练拿不到的。

与 trainer/train_pretrain.py 的关系
-----------------------------------
本脚本是它的超集：数据管线（PretrainDataset）、学习率调度（get_lr）、分布式、
混合精度、日志格式全部复用；差异只有三处：
    - 模型换成 XLLMWithMTP（主干 + D 个 MTP 头）
    - 损失换成 loss = main_loss + λ·mtp_loss
    - checkpoint 拆成两个文件（主干 / MTP 头），主干文件与原有命名完全兼容

用法
----
    # 从头联合预训练（推荐）
    python model/draft/mtp_pretrain.py --epochs 2 --batch_size 16

    # 在已有主干上补 MTP 头（主干继续参与训练，λ 可小一些）
    python model/draft/mtp_pretrain.py --from_weight pretrain --mtp_weight 0.3

    # 小模型冒烟
    python model/draft/mtp_pretrain.py --smoke
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
import warnings

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
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.XLLMModel import XLLMConfig, XLLMForCausalLM
from model.draft.mtp.mtp_module import MTPConfig, XLLMWithMTP
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    init_distributed_mode,
    setup_seed,
    SkipBatchSampler,
)

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# 权重加载
# ----------------------------------------------------------------------
def build_backbone(cfg: XLLMConfig, from_weight: str, save_dir: str, device,
                   mtp_path: str | None):
    """
    构造主干并尽量加载已有权重。

    注意 lm_checkpoint 的 naming：{save_dir}/{weight}_{hidden}{moe}.pth
    历史 checkpoint 里存在 post_attention_layernotm 拼写缺陷，这里统一修正。
    """
    model = XLLMForCausalLM(cfg).to(device)
    moe = "_moe" if cfg.use_moe else ""
    path = None
    if from_weight and from_weight.lower() != "none":
        cand = os.path.join(save_dir, f"{from_weight}_{cfg.hidden_size}{moe}.pth")
        if os.path.exists(cand):
            path = cand
        else:
            Logger(f"[warn] 未找到 {cand}，改用随机初始化")

    if path:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        state = {k.replace("post_attention_layernotm", "post_attention_layernorm"): v
                 for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        Logger(f"[backbone] 加载 {path} (missing={len(missing)}, unexpected={len(unexpected)})")

    # MTP 头单独加载（若存在）
    if mtp_path and os.path.exists(mtp_path):
        Logger(f"[mtp] 加载 {mtp_path}")
    else:
        Logger("[mtp] MTP 头随机初始化")
    return model


# ----------------------------------------------------------------------
def train_epoch(epoch, loader, steps_per_epoch, args, model, optimizer, scaler,
                autocast_ctx, lm_config, mtp_cfg, start_step=0, wandb=None,
                global_step_offset=0):
    """
    steps_per_epoch : 本 epoch 实际跑的步数（已按 max_steps 裁剪）
    global_step_offset : 已完成的总步数，resume 时用于学习率接续
    """
    start_time = time.time()
    total_steps = args.plan_total_steps        # 学习率调度的分母（整个训练计划）
    last_step = start_step

    for step, (input_ids, labels, attention_mask) in enumerate(loader, start=start_step + 1):
        if step > steps_per_epoch:
            break
        last_step = step
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        attention_mask = attention_mask.to(args.device)

        gstep = global_step_offset + (step - start_step)
        lr = get_lr(gstep, total_steps, args.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with autocast_ctx:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            # out["loss"] 已含 主干 CE + λ·ΣMTP CE + MoE aux_loss
            loss = out["loss"] / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == steps_per_epoch:
            spend = time.time() - start_time
            done = step - start_step
            cur_loss = loss.item() * args.accumulation_steps
            main_l = float(out["main_loss"].item())
            mtp_l = float(out["mtp_loss"].item())
            cur_lr = optimizer.param_groups[-1]["lr"]
            remain = steps_per_epoch - step
            eta_min = (spend / max(done, 1)) * remain / 60
            Logger(
                f"Epoch:[{epoch+1}/{args.epochs}]({step}/{steps_per_epoch}) "
                f"loss:{cur_loss:.6f} main:{main_l:.6f} mtp:{mtp_l:.6f} "
                f"lr:{cur_lr:.10f} eta:{eta_min:.1f}min"
            )
            if wandb:
                wandb.log({"loss": cur_loss, "main_loss": main_l,
                           "mtp_loss": mtp_l, "lr": cur_lr, "step": gstep})

        if (step % args.save_interval == 0 or step == steps_per_epoch) and is_main_process():
            save_checkpoint(model, args, lm_config, mtp_cfg, weight=args.save_weight,
                            optimizer=optimizer, scaler=scaler,
                            epoch=epoch, step=step, global_step=global_step_offset + (step - start_step))

    return last_step


def save_checkpoint(model, args, lm_config, mtp_cfg, weight="pretrain_mtp",
                    optimizer=None, scaler=None, epoch=0, step=0, global_step=0):
    """
    拆分保存，共三个文件：

        {weight}_{hidden}{moe}.pth        纯主干，键名与 XLLMForCausalLM 一致
                                          -> eval.py 的 strict=True 可直接加载
        {weight}_mtp_{hidden}{moe}.pth    MTP 头 + 自描述 cfg
        {weight}_train_{hidden}{moe}.pth  训练状态（optimizer/scaler/进度），仅续训用

    训练状态单列一个文件，是为了让主干文件保持"纯权重"语义——否则 eval.py
    用 strict=True 加载时会撞到 optimizer 之类的意外键。
    """
    moe = "_moe" if lm_config.use_moe else ""
    raw = model.module if isinstance(model, DistributedDataParallel) else model

    backbone_sd = {k: v.half() for k, v in raw.backbone.state_dict().items()}
    bb_path = f"{args.save_dir}/{weight}_{lm_config.hidden_size}{moe}.pth"
    torch.save(backbone_sd, bb_path)

    mtp_sd = {k: v.half() for k, v in raw.heads.state_dict().items()}
    mtp_path = f"{args.save_dir}/{weight}_mtp_{lm_config.hidden_size}{moe}.pth"
    torch.save({
        "heads": mtp_sd,
        "cfg": mtp_cfg.__dict__,
        "hidden_size": lm_config.hidden_size,
        "vocab_size": lm_config.vocab_size,
        "mtp_weight": getattr(raw, "mtp_weight", None),
    }, mtp_path)

    train_path = f"{args.save_dir}/{weight}_train_{lm_config.hidden_size}{moe}.pth"
    state = {
        "epoch": epoch, "step": step, "global_step": global_step,
        "mtp_weight": getattr(raw, "mtp_weight", None),
        "plan_total_steps": args.plan_total_steps,
        "args": {k: v for k, v in vars(args).items()
                 if isinstance(v, (int, float, str, bool, type(None)))},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, train_path)

    Logger(f"[save] {bb_path}\n[save] {mtp_path}\n[save] {train_path}")


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="XLLM + MTP 联合预训练")
    # 保存
    p.add_argument("--save_dir", type=str, default=os.path.join(_REPO_ROOT, "out"))
    p.add_argument("--save_weight", type=str, default="pretrain_mtp")
    # 训练
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--device", type=str,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--accumulation_steps", type=int, default=8)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--save_interval", type=int, default=100)
    # 架构
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    # MTP
    p.add_argument("--mtp_modules", type=int, default=1,
                   help="MTP 链式深度 D。V3/V4 官方为 1；2~5 可试，收益递减")
    p.add_argument("--mtp_layers", type=int, default=1, help="每个 MTP 模块的 Transformer 块数")
    p.add_argument("--mtp_weight", type=float, default=0.3,
                   help="λ，MTP 辅助损失权重。V3 报告在 0.1~0.3 区间")
    p.add_argument("--mtp_use_moe", type=int, default=0, choices=[0, 1],
                   help="MTP 块是否也用 MoE。论文未明确，默认关以省算力")
    # 数据
    p.add_argument("--data_path", type=str,
                   default=os.path.join(_REPO_ROOT, "dataset", "pretrain_t2t_mini.jsonl"))
    p.add_argument("--from_weight", type=str, default="none")
    p.add_argument("--mtp_ckpt", type=str, default=None)
    p.add_argument("--from_resume", type=int, default=0, choices=[0, 1])
    # 跟踪
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="XLLM-MTP-Pretrain")
    p.add_argument("--max_steps", type=int, default=0,
                   help="单个 epoch 的步数上限，0 表示用完整数据集。"
                        "注意：它同时定义学习率余弦周期的长度（见 --plan_total_steps）")
    p.add_argument("--plan_total_steps", type=int, default=0,
                   help="整个训练计划的总步数（学习率余弦周期的分母）。"
                        "0 表示自动推算 = epochs × min(len(loader), max_steps)。"
                        "续训时应显式传入与原计划一致的值，否则学习率会跳变")
    p.add_argument("--smoke", action="store_true", help="小模型 + 少步数快速跑通")
    args = p.parse_args()

    if args.smoke:
        args.hidden_size, args.num_hidden_layers = 128, 4
        args.max_seq_len, args.batch_size = 64, 4
        args.log_interval, args.save_interval = 5, 100
        args.accumulation_steps = 4
        args.epochs = 1
        args.max_steps = args.max_steps or 40
        args.plan_total_steps = args.plan_total_steps or args.max_steps
        # 冒烟用独立权重名，避免误加载正式训练的 checkpoint 或覆盖它
        args.save_weight = "smoke_mtp"
        args.save_dir = os.path.join(_REPO_ROOT, "out", "_smoke")

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = XLLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        max_position_embeddings=max(args.max_seq_len, 1024),
    )

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (nullcontext() if device_type == "cpu"
                    else torch.cuda.amp.autocast(dtype=dtype))

    # ---- 主干 ----
    moe = "_moe" if lm_config.use_moe else ""

    # 续训时主干必须回到上次训练结束的状态，而不是 --from_weight 指定的旧权重。
    # 若已保存过 {save_weight}_{hidden}{moe}.pth，就把它当作主干来源。
    bb_source = args.from_weight
    if args.from_resume:
        own_bb = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe}.pth"
        if not os.path.exists(own_bb):
            raise FileNotFoundError(
                f"--from_resume 1 但找不到上次训练保存的主干: {own_bb}\n"
                f"提示：续训需要 {args.save_weight} 这套 checkpoint 完整存在")
        bb_source = args.save_weight
        Logger(f"[resume] 主干从 {own_bb} 恢复")

    mtp_path = args.mtp_ckpt
    if mtp_path is None:
        cand = os.path.join(args.save_dir, f"{args.save_weight}_mtp_{lm_config.hidden_size}{moe}.pth")
        mtp_path = cand if os.path.exists(cand) else None

    backbone = build_backbone(lm_config, bb_source, args.save_dir, args.device, mtp_path)

    mtp_cfg = MTPConfig.from_backbone(
        lm_config,
        num_modules=args.mtp_modules,
        num_layers_per_module=args.mtp_layers,
        use_moe=bool(args.mtp_use_moe),
    )
    model = XLLMWithMTP(backbone, mtp_cfg, mtp_weight=args.mtp_weight).to(args.device)

    if mtp_path and os.path.exists(mtp_path):
        ck = torch.load(mtp_path, map_location="cpu", weights_only=True)
        model.heads.load_state_dict(ck["heads"])
        Logger("[mtp] 已恢复 MTP 头权重")

    Logger(
        f"主干参数 {sum(p.numel() for p in backbone.parameters())/1e6:.2f}M | "
        f"MTP 参数 {model.num_mtp_parameters()/1e6:.2f}M | "
        f"D={mtp_cfg.num_modules} λ={args.mtp_weight}"
    )

    # ---- 数据 ----
    from transformers import AutoTokenizer
    from dataset.lm_dataset import PretrainDataset
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(_REPO_ROOT, "model"))
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ---- 训练计划（学习率周期的分母，必须在建 loader 前定下来） ----
    # 数据集有 127 万条，len(loader) 可能是 31 万步。若直接拿它当分母，学习率
    # 在整个训练过程中几乎不动（cos 的斜率约等于 0），等于没有调度。
    # 因此语义明确为：
    #   steps_per_epoch = min(len(loader), max_steps)   本 epoch 实际跑多少步
    #   plan_total_steps = epochs × steps_per_epoch     余弦周期长度
    n_batches = len(train_ds) // max(args.batch_size, 1)
    if dist.is_initialized():
        n_batches = n_batches // dist.get_world_size()
    steps_per_epoch = args.max_steps if args.max_steps else n_batches
    steps_per_epoch = min(steps_per_epoch, n_batches)

    if args.plan_total_steps:
        total_steps = args.plan_total_steps
    else:
        total_steps = steps_per_epoch * args.epochs
    args.plan_total_steps = total_steps

    Logger(
        f"训练计划: {args.epochs} epoch × {steps_per_epoch} step = {total_steps} step "
        f"(数据集 {len(train_ds)} 条, batch {args.batch_size})"
    )

    # ---- 续训 ----
    resume_state = None
    resume_epoch, resume_step, resume_gstep = 0, 0, 0
    if args.from_resume:
        moe = "_moe" if lm_config.use_moe else ""
        tr_path = f"{args.save_dir}/{args.save_weight}_train_{lm_config.hidden_size}{moe}.pth"
        if not os.path.exists(tr_path):
            raise FileNotFoundError(
                f"--from_resume 1 但找不到训练状态文件: {tr_path}\n"
                f"提示：只有本脚本保存过 checkpoint 后才能续训。"
                f"若只想加载主干权重继续训练，请用 --from_weight 并保持 --from_resume 0")
        resume_state = torch.load(tr_path, map_location="cpu", weights_only=False)
        resume_epoch = int(resume_state.get("epoch", 0))
        resume_step = int(resume_state.get("step", 0))
        resume_gstep = int(resume_state.get("global_step", 0))
        if resume_state.get("optimizer") is not None:
            optimizer.load_state_dict(resume_state["optimizer"])
        if resume_state.get("scaler") is not None:
            scaler.load_state_dict(resume_state["scaler"])
        Logger(
            f"[resume] epoch={resume_epoch} step={resume_step} "
            f"global_step={resume_gstep}/{total_steps}"
        )

    if dist.is_initialized():
        model.backbone._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb.init(project=args.wandb_project,
                   name=f"MTP-D{args.mtp_modules}-lam{args.mtp_weight}")

    for epoch in range(resume_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        # 续训时跳过本 epoch 已跑过的步
        start_step = resume_step if epoch == resume_epoch else 0
        gstep_offset = resume_gstep if epoch == resume_epoch else epoch * steps_per_epoch

        train_epoch(epoch, loader, steps_per_epoch, args, model, optimizer, scaler,
                    autocast_ctx, lm_config, mtp_cfg, start_step, wandb,
                    global_step_offset=gstep_offset)

    save_checkpoint(model, args, lm_config, mtp_cfg, weight=args.save_weight,
                    optimizer=optimizer, scaler=scaler,
                    epoch=args.epochs - 1, step=steps_per_epoch,
                    global_step=total_steps)
    Logger("DONE")


if __name__ == "__main__":
    main()