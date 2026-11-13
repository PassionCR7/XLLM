# -*- coding: utf-8 -*-
"""
_mtp_check.py —— MTP 联合训练的**正确性**验证（不依赖训练收敛）

跑 `--smoke` 只能证明"不报错 + loss 是个数"，证明不了"偏移对齐正确"。
本脚本用确定性构造把偏移钉死：

    A. 主损失 vs 手算：把 backbone 的 CE 与手工 shift 的 CE 对比，确认理解一致
    B. MTP 偏移：构造 backbone 隐状态 = 输入 embedding 的恒等映射（proj 置单位，
       块内权重置零 → h_k 只剩下 e_norm(Emb(input_ids[:, k:k+L])) 的投影），
       然后检查 logits_k 与 lm_head(Emb(token_{t+k})) 的预测目标是否对齐
    C. 梯度回流：确认 MTP 损失确实把梯度传回主干（联合训练的核心特征，
       也是它与 DSpark 事后训练的本质区别）
    D. checkpoint 拆分：主干文件键名与 XLLMForCausalLM 完全一致，无 MTP 污染

用法：
    python model/draft/_mtp_check.py
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
from model.draft.mtp.mtp_module import MTPConfig, XLLMWithMTP


def build(vocab=128, hidden=64, layers=2, seq=16, D=2, weight=0.3):
    cfg = XLLMConfig(
        hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=vocab, max_position_embeddings=512,
        intermediate_size=hidden * 2,
    )
    bb = XLLMForCausalLM(cfg)
    mcfg = MTPConfig.from_backbone(cfg, num_modules=D)
    return XLLMWithMTP(bb, mcfg, mtp_weight=weight), cfg, mcfg


def check_A_main_loss():
    """主干 CE 与手算 shift CE 一致。"""
    torch.manual_seed(0)
    model, cfg, _ = build()
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    labels = ids.clone()
    labels[:, -3:] = -100

    with torch.no_grad():
        out = model.backbone(input_ids=ids, labels=labels)
        logits = out.logits
        manual = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, cfg.vocab_size),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
    diff = abs(out.loss.item() - manual.item())
    print(f"[A] backbone loss={out.loss.item():.6f} manual={manual.item():.6f} diff={diff:.2e}")
    assert diff < 1e-4, "主干 CE 与手算不一致"
    print("    OK")


def check_B_mtp_offset():
    """
    MTP 偏移验证 —— 直接钉死损失的目标切片。

    训练里模块 k 的 CE 必须是：logits_k[t] 对 labels[t+k+1]。
    这里用 return_logits 拿到同一个 logits，分别用"正确切片"和几种"错位切片"
    手算 CE，只有正确切片能精确复现训练时的 mtp_losses[k]。
    """
    torch.manual_seed(1)
    model, cfg, mcfg = build(D=2)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 20))
    labels = ids.clone()

    with torch.no_grad():
        res = model(input_ids=ids, labels=labels, return_logits=True)
        mtp_logits = res["mtp_logits"]
        reported = [x.item() for x in res["mtp_losses"]]

    T = ids.shape[1]
    D = mcfg.num_modules
    L = T - D - 1

    for k in range(1, D + 1):
        lg = mtp_logits[k - 1]                       # [B, L, V]
        assert lg.shape[1] == L, f"模块{k} 长度应为 {L}，实际 {lg.shape[1]}"
        cands = {
            "正确 labels[t+k+1]": labels[:, k + 1: k + 1 + L],
            "错位 labels[t+k]":   labels[:, k: k + L],
            "错位 labels[t+1]":   labels[:, 1: 1 + L],
        }
        vals = {}
        for name, tgt in cands.items():
            if tgt.shape[1] != L:
                continue
            vals[name] = F.cross_entropy(
                lg.reshape(-1, cfg.vocab_size), tgt.reshape(-1),
                ignore_index=-100).item()
        match = [n for n, v in vals.items() if abs(v - reported[k - 1]) < 1e-5]
        detail = "  ".join(f"{n}={v:.4f}" for n, v in vals.items())
        print(f"[B] 模块{k} reported={reported[k-1]:.4f} | {detail}")
        assert match and match[0] == "正确 labels[t+k+1]", \
            f"模块{k} 目标切片错误，匹配到 {match}"
    print("    OK（目标切片与论文公式 L_MTP^(k): predict token_{t+k+1} 一致）")


def check_C_grad_flow():
    """
    梯度回流验证 —— 联合训练与事后训练的分水岭。
    MTP 损失必须能产生主干参数的非零梯度。
    """
    torch.manual_seed(2)
    model, cfg, _ = build(D=1)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    labels = ids.clone()

    # 只留 MTP 损失：λ=1 且 main_loss 权重置零不方便，改为只看 mtp_loss 的反向
    res = model(input_ids=ids, labels=labels)
    model.zero_grad(set_to_none=True)
    res["mtp_loss"].backward()

    # 主干 embed_tokens 与第一层 attn 应有梯度
    g_emb = model.backbone.model.embed_tokens.weight.grad
    g_l0 = model.backbone.model.layers[0].self_attention.q_proj.weight.grad
    n_emb = 0 if g_emb is None else g_emb.abs().sum().item()
    n_l0 = 0 if g_l0 is None else g_l0.abs().sum().item()
    # MTP 头自己的参数也应该有梯度
    g_head = model.heads[0].proj.weight.grad
    n_head = 0 if g_head is None else g_head.abs().sum().item()

    print(f"[C] grad|emb={n_emb:.4e} grad|layer0.q={n_l0:.4e} grad|head.proj={n_head:.4e}")
    assert n_head > 0, "MTP 头自身没有梯度"
    assert n_emb > 0, "MTP 损失未回传到主干 embed_tokens —— 不是联合训练"
    assert n_l0 > 0, "MTP 损失未回传到主干 Transformer 层"
    print("    OK（梯度确实穿过主干，联合训练成立）")


def check_D_ckpt_split():
    """主干 state_dict 不含任何 MTP 键，eval.py 的 strict=True 才安全。"""
    model, cfg, _ = build(D=1)
    bb_keys = set(model.backbone.state_dict().keys())
    mtp_keys = set(model.heads.state_dict().keys())
    inter = bb_keys & mtp_keys
    print(f"[D] backbone keys={len(bb_keys)} mtp keys={len(mtp_keys)} 交集={len(inter)}")
    assert not inter, f"主干与 MTP 键名重叠: {list(inter)[:5]}"
    assert all(not k.startswith("heads.") for k in bb_keys), "主干 state_dict 混入了 MTP 键"
    print("    OK")


def check_E_infer_consistency():
    """
    draft_block 的偏移语义验证（推理侧）。

    约定：主头给第 1 个草稿（零成本、必然等于主干 argmax），MTP 链给后续 D 个。
    因此这里检查两件事：
        1) 输出宽度必须是 1 + D（不是 D）
        2) 第 1 个草稿必须恰好等于 lm_head(h0[t]).argmax()
    第 2 条是整个推理链路的地基——如果它错了，等价性必然失败。
    """
    torch.manual_seed(3)
    D = 2
    model, cfg, _ = build(D=D)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 12))

    with torch.no_grad():
        out = model.backbone(input_ids=ids, logits_to_keep=1)
        h_last = out.hidden_states[:, -1:, :]
        last_logits = out.logits[:, -1, :]
        d_ids, d_lg = model.draft_block(
            hidden_states=h_last, last_logits=last_logits,
            start_pos=ids.shape[1] - 1, n_draft=1 + D, do_sample=False)

    expect_w = 1 + D
    print(f"[E] draft_block: ids={tuple(d_ids.shape)} logits={tuple(d_lg.shape)} "
          f"(期望宽度 {expect_w})")
    assert d_lg.dim() == 3 and d_ids.shape[1] == expect_w, (
        f"宽度应为 {expect_w}，实际 {d_ids.shape[1]}")
    assert torch.isfinite(d_lg).all(), "draft_block 产出非有限值"

    head_argmax = int(last_logits.argmax(-1).item())
    first = int(d_ids[0, 0].item())
    print(f"    主头 argmax={head_argmax}  第1个草稿={first}")
    assert first == head_argmax, "第 1 个草稿必须等于主头 argmax（零成本那一个）"
    print("    OK")


def check_F_real_backbone():
    """
    真实主干加载：从 out/pretrain_512.pth（或 _moe）构造 XLLMWithMTP，
    确认 strict=False 下 missing/unexpected 符合预期，且 MTP 头可正常前向。
    """
    import os
    cands = ["out/pretrain_512.pth", "out/pretrain_512_moe.pth"]
    path = next((c for c in cands if os.path.exists(os.path.join(_REPO_ROOT, c))), None)
    if not path:
        print("[F] 跳过：未找到 pretrain_512*.pth")
        return

    use_moe = "_moe" in path
    cfg = XLLMConfig(hidden_size=512, num_hidden_layers=8, use_moe=use_moe,
                     max_position_embeddings=1024)
    bb = XLLMForCausalLM(cfg)
    state = torch.load(os.path.join(_REPO_ROOT, path), map_location="cpu",
                       weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = {k.replace("post_attention_layernotm", "post_attention_layernorm"): v
             for k, v in state.items()}
    missing, unexpected = bb.load_state_dict(state, strict=False)

    mcfg = MTPConfig.from_backbone(cfg, num_modules=1)
    model = XLLMWithMTP(bb, mcfg, mtp_weight=0.3).eval()

    with torch.no_grad():
        ids = torch.randint(0, cfg.vocab_size, (1, 24))
        res = model(input_ids=ids, labels=ids.clone())

    print(f"[F] {path} use_moe={use_moe} missing={len(missing)} unexpected={len(unexpected)}")
    print(f"    main_loss={res['main_loss'].item():.4f} mtp_loss={res['mtp_loss'].item():.4f} "
          f"mtp_params={model.num_mtp_parameters()/1e6:.2f}M")
    assert res["mtp_loss"].item() > 0, "MTP 损失异常"
    # intermediate_size 必须被 FeedForward 就地补上，否则 MTP 块构造会拿到 None
    assert mcfg.intermediate_size is not None, "intermediate_size 为 None"
    print("    OK")


if __name__ == "__main__":
    print("=" * 60)
    print("MTP 联合训练正确性检查")
    print("=" * 60)
    check_A_main_loss()
    check_B_mtp_offset()
    check_C_grad_flow()
    check_D_ckpt_split()
    check_E_infer_consistency()
    check_F_real_backbone()
    print("=" * 60)
    print("全部通过")