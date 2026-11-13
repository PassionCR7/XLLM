# model/draft —— 草稿模型辅助主模型推理（推测解码）

本目录实现 LLM **推理环节的草稿模型对主模型的辅助**，即 **推测解码
(Speculative Decoding)**：用一个更小、更快的草稿模型批量“猜测”多个下一个
token，再由主模型一次并行验证。**输出与主模型直接解码完全等价**（贪心一致 /
采样无偏），却大幅减少主模型前向次数，从而提升生成吞吐、降低显存占用。

本实现整体对齐 Qwen / DeepSeek / Kimi 等主流方案的推测解码骨架。

---

## 1. 为什么需要草稿模型？（背景）

自回归 LLM 生成是**逐 token 串行**的：每生成一个 token 都要完整跑一遍整
个模型（所有层）。生成的 100 个 token 就要跑 100 次主模型前向，瓶颈在
“每个 token 都要跑一遍大模型”。

**推测解码的思想**（Leviathan 2023 / Chen et al. 2023）：

> 用一个又快又小的草稿模型先把 `gamma` 个候选 token 猜出来；主模型把
> 这 `gamma` 个候选**拼进一次前向**一次性验证。接受率越高，一次前向换取
> 的 token 越多，整体就越快。

- 主模型每回合只跑 **1 次** 前向（并行验证 `gamma` 个位置），而非 `gamma` 次。
- 只要草稿足够接近主模型（接受率高），总前向次数大幅下降，`≈ 1 + 平均接受数` 倍的加速。

### 与各家方案的对应

| 方案 | 关键思路 | 与本实现的关系 |
|---|---|---|
| **Qwen** | eager speculative decoding：草稿提议 + 目标并行验证 | 本引擎的“提议→并行验证→接受”即此骨架 |
| **DeepSeek-V3 (MTP)** | 单个模块一次预测多个未来 token，等价于草稿提议+验证，~1.8x | `gamma` 个候选并行验证正是 MTP 的验证形式 |
| **Kimi (Moonshot)** | 轻量草稿模型 + 联合/并行解码 | `DraftModel`（浅层草稿）+ 并行验证同理 |

> 注：各家差异主要在“草稿从哪来”（独立小模型 / 截断共享层 / 多 token 预测头）。
> 本目录两种草稿构建方式分别对应“截断共享层（零训练）”和“独立草稿 checkpoint”。

---

## 2. 目录与模块

```
model/draft/
├── __init__.py               # 包导出
├── _verification.py          # 纯 torch 的验证/接受算法（可独立单测）
├── draft_model.py            # DraftModel：草稿模型（复用 XLLMForCausalLM）
├── speculative_decoding.py   # SpeculativeDecodingEngine：端到端解码引擎
└── example.py                # 可运行示例
```

### 2.1 算法核心 `_verification.py`（纯 torch，不依赖主模型）
- `prepare_distribution(logits, temperature, top_k, top_p)`：logits → 概率分布。
- `verify_and_sample(...)`：一次并行验证，决定接受几个草稿 token 并给出压哨 token。
- `acceptance_stats(...)`：接受率与近似加速比。

**接受规则**（两种解码目标）：
- **greedy 贪心**：`argmax(p_i) == d_i` 则接受；被拒处输出 `argmax(p_i)`。
  结果与主模型贪心解码逐 token 一致。
- **stochastic 随机**：以 `min(1, p_i(d_i)/q_i(d_i))` 接受；被拒则从
  `(p_i - q_i)_+` 归一化后采样。结果与主模型采样解码**无偏**（Chen et al. 2023）。

### 2.2 草稿模型 `draft_model.py`
`DraftModel` 复用主模型基座 `XLLMForCausalLM`，三种构建方式：
1. `DraftModel.from_main_model(main, num_draft_layers)` —— **零训练截断草稿**
   （复用主模型 嵌入 + 前 N 层 + 归一化 + lm_head），最适合起步/基线；
2. `DraftModel(config)` —— 独立更小配置，权重随机（可训练/加载）；
3. `DraftModel.from_pretrained(ckpt, config)` —— 加载独立草稿 checkpoint。
`propose_n(...)` 连续自回归提议 `gamma` 个候选，同时返回每步提议分布 `q`。

### 2.3 推理引擎 `speculative_decoding.py`
`SpeculativeDecodingEngine(target, draft, gamma, ...)`，每个回合：
1. **提议**：草稿提出 `gamma` 个候选；
2. **验证**：主模型一次前向并行验证；
3. **接受**：逐位接受/拒采，得到压哨 token；
4. **同步**：主/草稿 KV cache 修剪到“已接受前缀+压哨”，保证下一回合一致。

---

## 3. 数学与索引约定（重要）

主模型验证时输入 `[d_1..d_gamma]`，`past_key_values` 已含“已接受前缀”，因此
主模型 logits 各位置预测关系为：

```
target_logits[:, i] 预测“接受完 d_{i+1} 后”的下一个 token   （i=0..gamma-1）
target_last_logits    预测第一个草稿 token d_1（由上一回合携带）
```

由此构造完整目标分布 `p = [target_last_logits, target_logits]`（`gamma+1` 个位置），
逐位与草稿分布 `q_i` 做接受判断；`gamma` 个全接受时，从 `p[:, gamma]` 采“压哨”。

---

## 4. 使用方法

```python
import torch
from model.XLLMModel import XLLMConfig, XLLMForCausalLM
from model.draft import DraftModel, SpeculativeDecodingEngine

# 0) 准备主模型（示例：从 checkpoint 加载）
cfg = XLLMConfig(hidden_size=512, num_hidden_layers=8, num_attention_heads=8,
                 num_key_value_heads=2, vocab_size=6400)
main = XLLMForCausalLM(cfg)
# main.load_state_dict(torch.load("checkpoints/pretrain_512.pth", map_location="cpu"))

# 1) 零训练截断草稿：复用主模型前 4 层（需与主模型同 device）
draft = DraftModel.from_main_model(main, num_draft_layers=4).to(main.device)

# 2) 构建推测解码引擎（贪心 / 每回合提议 4 个）
engine = SpeculativeDecodingEngine(target=main, draft=draft,
                                   gamma=4, do_sample=False, eos_token_id=2)

# 3) 生成
prompt = torch.tensor([[1, 5, 9, 120, 300]])   # 已编码 prompt，需左对齐
result = engine.generate(prompt, max_new_tokens=64)
print(result["sequences"])     # 生成的 token id
print(result["stats"])         # 接受率 / 近似加速比
```

### 参数建议
- `gamma`：常用 3~8。越大加速潜力越高，但草稿越易错、被拒浪费越多，需按接受率调优。
- `do_sample=False` 贪心确定性解码；`do_sample=True` 无偏随机采样。
- 推荐先用 `from_main_model` 零训练草稿验证整条链路，再训练独立草稿提升接受率。

---

## 5. 加速比来源与注意点

- **收益**：每回合主模型只做 1 次前向（验证 `gamma` 个候选）≈ 普通解码 1 步，
  而普通解码要 `gamma` 步。`近似加速比 ≈ 1 + 平均接受数`（见 `acceptance_stats`）。
- **前提**：草稿必须明显更便宜（更浅/更小），否则草稿开销会抵消收益。
- **等价性**：贪心模式与主模型贪心解码一致；随机模式与主模型采样无偏，
  不会因草稿破坏质量。
- **局限**：本引擎默认 batch=1（推测解码典型场景），支持多序列但为串行处理。

---

## 6. 扩展方向（进阶）

- 用**独立更小 hidden_size 的草稿**并训练，提升接受率（更接近 Qwen/DeepSeek 实践）。
- 实现 DeepSeek 的 **MTP 多 token 预测头**：把“草稿”内嵌到主模型尾部一次预测。
- 引入 **EAGLE** 式草稿（基于主模型特征 + 门控），进一步逼近主模型分布。
- 与 XLLM 的分页 KV cache / 算子融合结合，获得叠加收益。
