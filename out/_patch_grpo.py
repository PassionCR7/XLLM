# -*- coding: utf-8 -*-
path = r"D:\workspace\xllm\trainer\train_grpo.py"
with open(path, encoding="utf-8") as f:
    src = f.read()

old = """        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1) :]

        def get_per_token_logps(mdl, input_ids, n_keep):
            input_ids = (
                input_ids.detach().clone() if input_ids.is_inference() else input_ids
            )
            logits = mdl(input_ids=input_ids, logits_to_keep=n_keep + 1).logits[
                :, :-1, :
            ]
            per_token_logps = []
            for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
                ids_row = (
                    ids_row.detach().clone() if ids_row.is_inference() else ids_row
                )
                token_logps = torch.gather(
                    logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)
                ).squeeze(1)
                per_token_logps.append(token_logps)
            return torch.stack(per_token_logps)"""

new = """        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1) :]
        # 生成结束后及时回收 KV cache 占用的显存，为后续分块前向留出空间
        torch.cuda.empty_cache()

        def get_per_token_logps(mdl, input_ids, n_keep, chunk_size=4):
            # 分块前向以降低峰值显存：注意力分数矩阵大小为 O(B * seq_len^2)，
            # GRPO 的 batch 被 num_generations 放大（如 2*8=16 条序列），一次性全量前向
            # 在 8GB 显存上会 OOM（16*1602^2*8层≈5GB，另有常驻的 1.8B reward 模型）。
            # 按 chunk_size 分批计算每行 logps 后拼接，逐行语义与原实现完全一致。
            input_ids = (
                input_ids.detach().clone() if input_ids.is_inference() else input_ids
            )
            all_per_token_logps = []
            n_rows = input_ids.size(0)
            for start in range(0, n_rows, chunk_size):
                chunk_ids = input_ids[start : start + chunk_size]
                logits = mdl(
                    input_ids=chunk_ids, logits_to_keep=n_keep + 1
                ).logits[:, :-1, :]
                for logits_row, ids_row in zip(logits, chunk_ids[:, -n_keep:]):
                    ids_row = (
                        ids_row.detach().clone() if ids_row.is_inference() else ids_row
                    )
                    token_logps = torch.gather(
                        logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)
                    ).squeeze(1)
                    all_per_token_logps.append(token_logps)
            return torch.stack(all_per_token_logps)"""

assert src.count(old) == 1, f"old block found {src.count(old)} times"
src = src.replace(old, new)

# 2) padding_side fix (mirror of train_ppo.py)
old2 = """    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token"""
new2 = """    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    tokenizer.padding_side = "left"  # GRPO 在线生成需要左侧 padding，否则右 padding 会导致生成错位
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token"""
assert src.count(old2) == 1, f"old2 block found {src.count(old2)} times"
src = src.replace(old2, new2)

with open(path, "w", encoding="utf-8") as f:
    f.write(src)
print("OK, edits applied")
