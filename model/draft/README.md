# model/draft —— 草稿模型辅助主模型推理（推测解码）

本目录实现 LLM **推理环节的草稿模型对主模型的辅助**，即 **推测解码
(Speculative Decoding)**：用一个更小、更快的草稿模型批量“猜测”多个下一个
token，再由主模型一次并行验证。**输出与主模型直接解码完全等价**（贪心一致 /
采样无偏），却大幅减少主模型前向次数，从而提升生成吞吐、降低显存占用。

本实现整体对齐 Qwen / DeepSeek / Kimi 等主流方案的推测解码骨架。

---

## 0. 三套实现（重要）

本目录现在有三条技术路线，**互不依赖，共用验证算法**：

| | A. 串行草稿（原实现） | B. DSpark 式（对齐 V4.1） | C. MTP 联合预训练（对齐 V3/V4） |
|---|---|---|---|
| 草稿结构 | `XLLMForCausalLM` 截断副本 | 并行骨干 + Markov 头 + 置信度头 | 共享 emb/lm_head 的 D 个 Transformer 块 |
| 草稿提议 | `propose_n` 串行 γ 步 | 一次前向出整块 | 链式 D 步，每步一个 token |
| KV cache | 草稿与 target **各持一套**，需同步裁剪 | **只有 target 有** | 只在主干（MTP 头训练时无 cache） |
| 验证长度 | 固定 γ | **按置信度动态调度** | D（通常 =1） |
| target 是否训练 | 否 | **冻结**，事后训练草稿 | **一起训练**，梯度穿过主干 |
| 入口 | `SpeculativeDecodingEngine` | `DSparkEngine` | `XLLMWithMTP` + `mtp_pretrain.py` |

**选型建议**：
- 只想跑通链路 / 验证正确性 → A 的 `from_main_model`（零训练）
- 已有主模型、要补草稿 → B，并真正训练草稿（`dspark_train.py`）
- **从零开始训主模型 → C**，草稿头白拿，且 MTP 损失 densify 监督信号、主干本身也变好

**B 和 C 最容易混淆，关键区别只有一条**：B 的 target 冻结，草稿只能**逼近**已有的
p；C 在预训练期就让梯度穿过主干，草稿与主干**共同塑造** p。论文强调的"MTP 顺带
提升主干性能"只在 C 里存在。

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

三套实现按技术路线分目录存放，共用部分单独抽出：

```
model/draft/
├── __init__.py                # 平铺导出（保持向后兼容）+ 选型说明
├── README.md                  # 本文档
│
├── common/                    # 三套共用
│   ├── __init__.py
│   └── _verification.py       # 纯 torch 的验证/接受算法（可独立单测）
│
├── serial/                    # 【A】串行草稿 + KV 同步（基线，零训练可用）
│   ├── __init__.py
│   ├── draft_model.py         # DraftModel：截断式草稿模型
│   ├── speculative_decoding.py# SpeculativeDecodingEngine：端到端解码
│   ├── example.py             # 可运行示例
│   └── _selftest.py           # 自测（贪心等价 / 随机无偏）
│
├── dspark/                    # 【B】DSpark 式半自回归草稿
│   ├── __init__.py
│   ├── dspark_draft.py        # DSparkDrafter：并行骨干 + Markov 头 + 置信度头
│   ├── dspark_scheduler.py    # 三种调度器（throughput / confidence / fixed）
│   ├── dspark_engine.py       # DSparkEngine：target-only KV 的解码循环
│   ├── dspark_train.py        # 训练脚本（CE + TV + 置信度 BCE）
│   └── dspark_selftest.py     # 自测（等价性 / 调度器 / 状态一致 / 无偏性）
│
└── mtp/                       # 【C】MTP 联合预训练
    ├── __init__.py
    ├── mtp_module.py          # MTPHead + XLLMWithMTP
    ├── mtp_engine.py          # MTPEngine：推理引擎，测接受率
    ├── mtp_pretrain.py        # 联合预训练脚本
    ├── _mtp_check.py          # 六项正确性验证
    ├── _mtp_engine_check.py   # 引擎验收（等价性 / 无偏 / 接受率）
    └── MT_PRETRAIN_GUIDE.md   # MTP 上手手册
```

**导入方式**：两种都支持，指向同一对象。

```python
from model.draft import MTPEngine                    # 平铺，向后兼容
from model.draft.mtp import MTPEngine                # 按实现方式，语义更清晰
```

### 2.0 算法核心 `common/_verification.py`（纯 torch，不依赖主模型）
- `prepare_distribution(logits, temperature, top_k, top_p)`：logits → 概率分布。
- `verify_and_sample(...)`：一次并行验证，决定接受几个草稿 token 并给出压哨 token。
- `acceptance_stats(...)`：接受率与近似加速比。

三套 engine 都调用这一份接受逻辑，所以**无损性保证完全一致**。单独抽出来是为
防止"各实现各写一份接受规则、改了一处忘了另一处"导致某条路线悄悄有偏。

**接受规则**（两种解码目标）：
- **greedy 贪心**：`argmax(p_i) == d_i` 则接受；被拒处输出 `argmax(p_i)`。
  结果与主模型贪心解码逐 token 一致。
- **stochastic 随机**：以 `min(1, p_i(d_i)/q_i(d_i))` 接受；被拒则从
  `(p_i - q_i)_+` 归一化后采样。结果与主模型采样解码**无偏**（Chen et al. 2023）。

### 2.2 草稿模型 `serial/draft_model.py`
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
from model.draft.serial import DraftModel, SpeculativeDecodingEngine

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

---

# 第二部分：DSpark 式半自回归草稿

对齐 **DeepSeek-V4.1-Flash 技术报告 §2.4.3**（DSpark: Confidence-Scheduled
Speculative Decoding with Semi-Autoregressive Generation）。论文原文关键设定：

> "DSpark is introduced in a dedicated stage after pre-training. In this stage, we
> train only DSpark while keeping the backbone frozen. During post-training, we
> continue to train DSpark alongside the backbone, without propagating gradients
> from the DSpark objective into the backbone."

## 7. 文件

位于 `model/draft/dspark/`：

```
├── dspark_draft.py       # DSparkDrafter：并行骨干 + Markov 头 + 置信度头
├── dspark_scheduler.py   # 三种调度器（throughput / confidence / fixed）
├── dspark_engine.py      # DSparkEngine：target-only KV 的解码循环
├── dspark_train.py       # 训练脚本（CE + TV + 置信度 BCE）
└── dspark_selftest.py    # 自测（等价性 / 调度器 / 状态一致 / 无偏性）
```

## 8. 三个核心设计

**并行骨干（一次前向出整块）**
`DSparkDrafter._backbone` 把 target 的 hidden states 与 K 个可学习 mask 槽位拼接，
一次前向产出 K 个位置的 base logits。草稿长度从 1 增到 5，延迟基本不变——这是
DFlash/DSpark 相对串行草稿的根本收益来源。

**Markov 头（修正后缀衰减）**
并行草稿各位置独立预测，块尾准确率会明显下降。Markov 头对第 d 个位置施加一个
只依赖"上一个草稿 token"的低秩加性偏置：

```
logits_d = base_logits_d + markov_proj(markov_emb(prev_token))
```

每步只是一次查表 + 一次 `[rank] → [vocab]` 变换，成本远低于一个完整 Transformer 块。
实测 `draft_overhead_ratio ≈ 0.25`（草稿耗时约为验证耗时的 1/4）。

**置信度调度（论文主贡献）**
置信度头逐位置预测条件接受概率，调度器按 `Θ = τ·SPS(B)` 选验证长度：

| 场景 | k 的选择 | 实测 |
|---|---|---|
| 高置信 + 低负载 | 尽量长 | k=4, τ=4.30 |
| 高置信 + **高负载** | **主动缩短** | k=1, τ=1.95 |
| 低置信 + 低负载 | 中等 | k=2, τ=2.17 |

这与论文描述一致：高并发时主动丢弃低置信草稿，把 target 算力留给更可能成功的候选。

## 9. 用法

**训练**（target 冻结，只训草稿）

```bash
python model/draft/dspark/dspark_train.py \
    --target-ckpt checkpoints/full_sft_512_moe.pth --use-moe \
    --data dataset/_subset_pretrain.jsonl \
    --steps 2000 --batch-size 4 --seq-len 128 --lr 4e-4 \
    --out checkpoints/dspark_512.pth
```

**推理**

```python
import torch
from model.XLLMModel import XLLMConfig, XLLMForCausalLM
from model.draft.dspark import DSparkConfig, DSparkDrafter, DSparkEngine, build_scheduler

target = XLLMForCausalLM(cfg).eval()          # 加载主模型权重
ckpt = torch.load("checkpoints/dspark_512.pth", map_location="cpu", weights_only=True)
drafter = DSparkDrafter(DSparkConfig(**ckpt["cfg"])).eval()
drafter.load_state_dict(ckpt["drafter"])

engine = DSparkEngine(
    target, drafter,
    scheduler=build_scheduler("throughput", max_draft=5),
    do_sample=False, eos_token_id=2,
)
res = engine.generate(torch.tensor([[1, 17, 260, 512]]), max_new_tokens=64)
print(res["sequences"], res["stats"])
```

## 10. 实测结果

**无损性（保证成立，已自测）**

| 项目 | 结果 |
|---|---|
| 贪心等价性（三种调度器） | 与 target 贪心解码**逐 token 一致** |
| 随机无偏性（N=1500） | 最大频率偏差 **0.003**（理论分布对比） |
| hidden 与 KV 长度一致性 | 每轮 assert 通过，无错位 |

在真实权重（`full_sft_512_moe.pth`）上也验证了等价性成立。

**接受率（诚实记录，未达论文水平）**

600 步短训后的逐步 top-1 命中率：`[0.175, 0.075, 0.069]`。这远低于论文的
90%+ 二 token 接受率。原因有三，按影响排序：

1. **骨干简化**：论文 DFlash 骨干是块扩散（加噪/去噪），本实现简化为可学习 mask
   槽位，不含扩散过程，这是最主要的差距来源；
2. **训练量**：600 步 × batch 4 × 128 token 远不足以拟合 6400 词表；
3. **Markov 秩偏小**（64），对 6400 词表的高秩依赖拟合能力有限。

另外 target 本身只有 25.8M 参数，其下一 token 分布熵较高，也会压低绝对接受率。

## 11. 与论文的已知偏离

| 项目 | 论文 | 本实现 |
|---|---|---|
| 草稿骨干 | DFlash 块扩散 | 可学习 mask 槽位（无扩散） |
| 草稿宽度 | 5 | 5（可配） |
| Markov 头 | 链式查表 | 低秩加性偏置（`rank=64`） |
| 置信度校准 | 逐位置迭代重标定 | 逐位置 logit 平移（单调近似） |
| SPS 曲线 | 启动时实测 profile | 缺省占位曲线，需替换 |
| 批量并发 | 全 batch 统一调度 | batch=1 逐条串行 |

**要让加速比接近论文，优先做这两件事**：把 SPS 曲线换成实测值（`ThroughputProfile`），
以及把骨干换成真正的块扩散。其余部分（Markov、置信度、调度、验证）无需改动。

---
---

# 第三部分：MTP 联合预训练

前两部分的草稿都是**事后**挂上去的。这一部分不同：MTP 草稿头与主干在预训练阶段
**一起训练**，梯度穿过主干。这是 DeepSeek-V3 / V4 的做法。

## 12. 文件

位于 `model/draft/mtp/`：

| 文件 | 作用 |
|---|---|
| `mtp_module.py` | `MTPHead`（单个模块）、`XLLMWithMTP`（主干 + D 个头 + 联合损失） |
| `mtp_pretrain.py` | 联合预训练脚本，复用 `trainer/train_pretrain.py` 的数据与调度 |
| `mtp_engine.py` | **推理引擎**：把训好的 MTP 头接进生成流程，测接受率与加速比 |
| `_mtp_check.py` | 六项正确性验证（偏移、梯度回流、ckpt 拆分等） |
| `_mtp_engine_check.py` | 引擎验收（贪心等价、采样无偏、接受率、统计口径） |
| `MT_PRETRAIN_GUIDE.md` | **上手手册：怎么跑、参数怎么定、结果怎么读** |

## 13. 损失函数

论文原文形式：

```
L_total = L_next_token + (λ/D) · Σ_{k=1..D} L_MTP^(k)
```

本实现对应 `mtp_module.py: XLLMWithMTP.forward`：

```python
L  = T - D - 1                 # 所有模块共用同一段对齐长度，避免逐模块裁剪错位
h_prev = h0[:, :L, :]          # h0 是主干末层隐状态
for k, head in enumerate(self.heads, start=1):
    prev_ids = input_ids[:, k : k + L]        # token_{t+k}
    h_k      = head(h_prev, prev_ids, emb, pos_emb, mask)
    logits_k = lm_head(h_k)                   # 与主干共享 lm_head，省参数
    target_k = labels[:, k+1 : k+1+L]         # 预测 token_{t+k+1}
    ce       = F.cross_entropy(logits_k.reshape(-1, V), target_k.reshape(-1),
                               ignore_index=-100)
    h_prev   = h_k                            # 链式：模块 k 的输出喂给 k+1
total = main_loss + λ · mean(ce_list) + aux_loss
```

单个模块内部：

```
x    = proj( concat( RMSNorm(h_{k-1}), RMSNorm(Emb(token_{t+k})) ) )
h_k  = TransformerBlock(x)
```

**索引约定（关键，容易错）**：`h0[t]` 预测 `labels[t+1]`。所以模块 k 用
`input_ids[t+k]` 作条件、预测 `labels[t+k+1]`。长度统一取 `T-D-1`。

## 14. 用法

```bash
# 冒烟（40 步，小模型，独立目录不污染正式 ckpt）
python model/draft/mtp/mtp_pretrain.py --smoke

# 在已有主干上继续联合训练（最常用）
python model/draft/mtp/mtp_pretrain.py --from_weight pretrain --save_weight pretrain_mtp \
  --hidden_size 512 --num_hidden_layers 8 --batch_size 8 --accumulation_steps 8 \
  --mtp_modules 1 --mtp_weight 0.3 --learning_rate 1e-4 \
  --epochs 1 --max_steps 2000 --plan_total_steps 2000

# 续训（恢复主干 + MTP 头 + AdamW 状态 + 进度）
python model/draft/mtp/mtp_pretrain.py --from_resume 1 --save_weight pretrain_mtp \
  --hidden_size 512 --num_hidden_layers 8 --epochs 2 \
  --max_steps 2000 --plan_total_steps 4000
```

详细参数说明与结果判读见 [`MT_PRETRAIN_GUIDE.md`](./MT_PRETRAIN_GUIDE.md)。

### 学习率周期的语义（容易踩的坑）

数据集有 127 万条，`len(loader)` 是 31 万步。若直接拿它当余弦周期的分母，
`get_lr` 在整个训练中几乎不动，等于没有调度。所以：

- `--max_steps`：本 epoch 实际跑多少步
- `--plan_total_steps`：整个训练计划的步数（余弦周期）

**续训时必须显式传与原计划一致的 `--plan_total_steps`**，否则学习率跳变。

### 取草稿与推理

```python
from model.draft.mtp import MTPEngine

eng = MTPEngine(model, max_draft=1 + model.cfg.num_modules, do_sample=False)
tokens = eng.generate(input_ids, max_new_tokens=32)
print(eng.stats())     # accept_rate / avg_accepted_per_iter / ms_per_token
```

**偏移语义（`draft_block` 的关键约定）**：训练时模块 k 的输入是
`(h0[t], token_{t+k})`、目标是 `token_{t+k+1}`。所以推理时不能把 `token_t`
喂给模块 1——那个配对训练中从未出现过。正确链路是：

```
主头 lm_head(h0[t])         -> token_{t+1}   （上轮已验证，零额外成本）
模块 1 (h0[t], tok_{t+1})   -> token_{t+2}
模块 2 (h1[t], tok_{t+2})   -> token_{t+3}
```

因此 D 个模块实际给出 **1 + D** 个草稿，第一个是"白拿"的。这也解释了下文
接受率的基线。

### checkpoint 约定（重要）

联合训练产生**三个**文件：

```
out/pretrain_mtp_512.pth          # 纯主干，键名与 XLLMForCausalLM 一致
out/pretrain_mtp_mtp_512.pth      # {"heads":..., "cfg":..., "hidden_size":..., "vocab_size":...}
out/pretrain_mtp_train_512.pth    # optimizer/scaler/进度，仅续训用
```

主干单独一个文件，是为了让 `eval.py` 的 `strict=True` 能直接加载——不会撞到
MTP 头或 optimizer 的意外键。已在真实 512 维权重上验证 `missing=0 unexpected=0`。

## 15. 实测结果

### smoke（hidden=128, 4 层, seq=64, batch=4, 40 步, λ=0.3）

```
[mtp] MTP 头随机初始化
主干参数 1.57M | MTP 参数 0.22M | D=1 λ=0.3
Epoch:[1/1](5/40)  loss:11.455625 main:8.805183 mtp:8.834804 lr:0.000482872895
Epoch:[1/1](20/40) loss:11.385837 main:8.751987 mtp:8.779499 lr:0.000275000000
Epoch:[1/1](40/40) loss:11.222671 main:8.625519 mtp:8.657173 lr:0.000050000000
DONE
```

`ln(6400)=8.76`，40 步几乎没离开随机水平，符合预期（仅为跑通验证），但 loss
确实单调下行、学习率余弦衰减正常、两个 ckpt 均正确落盘。

### 正确性验证 `_mtp_check.py`（六项全过）

```
[A] backbone loss=5.035396 manual=5.035396 diff=0.00e+00
[B] 模块1 reported=4.9129 | 正确 labels[t+k+1]=4.9129  错位 labels[t+k]=4.9041
[B] 模块2 reported=4.9585 | 正确 labels[t+k+1]=4.9585  错位 labels[t+k]=4.9015
[C] grad|emb=5.6482e+01 grad|layer0.q=7.8073e+00 grad|head.proj=9.2208e+01
[D] backbone keys=21 mtp keys=12 交集=0
[E] draft_block: ids=torch.Size([1, 1]) logits=torch.Size([1, 1, 128])
[F] out/pretrain_512.pth use_moe=False missing=0 unexpected=0
    main_loss=13.0560 mtp_loss=8.8461 mtp_params=3.34M
全部通过
```

三项要点：

- **[B] 偏移**：三种候选切片的 CE 只差 0.01 量级，只有正确切片能精确复现训练时的
  `reported` 值（误差 `<1e-5`）。这说明该实现即使用阈值判断也不可靠，必须精确匹配。
- **[C] 梯度回流**：只对 `mtp_loss` 反传后，主干 `embed_tokens` 梯度 56.5、
  `layer0.q_proj` 梯度 7.8，均非零 —— 这才是"联合训练"，是 B 拿不到的。
- **[F] 真实主干**：512 维官方权重上 `missing=0 unexpected=0`，MTP 头 3.34M 参数
  （约占主干 57M 的 6%）。

### 引擎验收 `_mtp_engine_check.py`

在真实 512 权重上（`pretrain_512.pth` + 随机 MTP 头）：

```
[A] prompt0..5 一致=True                                   <- 贪心完全等价
[B] 直接采样 vs 理论 TV = 0.2587
    引擎采样 vs 理论 TV = 0.2660   比值 1.03               <- 无系统性偏差
[C] 轮数=12 产出=24 接受=12
    接受率=0.5000  每轮平均接受=1.00
    acc_per_iter=[1,1,1,1,1,1,1,1,1,1,1,1]
    草稿耗时占比=0.107   27.89 ms/token
[D] avg_tokens_per_iter=2.00 ∈ [1,3]  OK
```

**怎么读 [C]**：MTP 的草稿宽度是 1+D，其中**第 1 个来自主头**（主干自己的
argmax，必然命中），后续 D 个才来自 MTP 头。所以 D=1 时"接受率 0.5、每轮 1.00"
是**未训练基线**，不是好结果；`acc_per_iter` 全是 1 正是"MTP 头全错"的指纹。
判据：每轮平均接受 >1.3 说明有效，>1.8 说明充分。

### 端到端小规模训练（150 步，512 权重 + λ=0.3 + lr=3e-4）

```
Epoch:[1/1](50/150)  loss:5.0046 main:3.0264 mtp:6.5938 lr:0.0002325
Epoch:[1/1](150/150) loss:4.4707 main:2.5640 mtp:6.3556 lr:0.0000300
```

训练后验收：

```
[C] 轮数=11 产出=24 接受=13
    接受率=0.5417  每轮平均接受=1.18          <- 从 1.00 上升到 1.18
    acc_per_iter=[1,1,2,2,1,1,1,1,1,1,1]      <- 开始出现 2
```

方向正确，150 步远不足以收敛，但足够证明训练-推理链路闭环有效。

### 一个曾经的真实 bug（记录备查）

`mtp_engine.py` 最初为省一次前向，在"草稿未被全部接受"时复用
`verify_hidden[:, m]` 作为下一轮草稿输入。贪心等价性测试立刻暴露（6 个 prompt
中 4 个失败，分歧点在第 2~7 个 token）。

根因：`next_token` 是在位置 `ctx_len+m` 上**被拒绝**后采出的压哨 token，它与
`draft_ids[m]` 不是同一个 token，所以那个位置的 hidden 是"错误 token 的隐状态"。

正确做法是「保留已接受前缀 + 对 `next_token` 单独补一次前向」，位置严格连续。
代价是每轮多一次单 token 前向（约占总时间 10%~12%），这次前向**不能省**。

## 16. 与其他两套方案的关系

| 场景 | 用哪个 |
|---|---|
| 训练预算充足、从零训主模型 | **C**，一次训练同时得到主模型和草稿头 |
| 已经有主模型，只想补草稿 | **B**（`dspark_train.py`），target 冻结、训练快 |
| 只想验证推测解码链路 | **A**，零训练 |

三者共用 `_verification.py` 的接受/验证规则，推测解码的**无损性保证完全一致**：
贪心模式下与 target 贪心解码逐 token 相同，随机模式下与 target 采样无偏。

## 17. 已知限制

1. **`--smoke` 不会收敛**，它只验证链路。要看真实效果需在 `pretrain_t2t_mini.jsonl`
   （127 万条）上跑足够步数。
2. **D>1 时收益递减**：按 V3 报告，D=1 已拿到大部分收益，D=3 主要用于密集监督。
   本实现支持任意 D（引擎与验收脚本已测 D=1 和 D=2），但未做大规模对照实验。
3. **MoE 主干**的 `aux_loss` 已正确接入（`out.aux_loss` → `total`），但 MTP 块默认
   不用 MoE（`--mtp_use_moe 0`），论文未明确此点，默认关以省算力。
4. **引擎只支持 batch=1**：与 B 的 `DSparkEngine` 一致，逐条串行。批量并发需要
   按 batch 内最大 `m` 做对齐，涉及额外的 padding 与 mask 逻辑，目前未实现。
5. **每轮有一次"补位前向"开销**：压哨 token 必须单独过一遍主干才能拿到它的
   hidden 与 next logits（原因见 §15 的 bug 记录）。这使加速比低于理论值
   `1 + 平均接受数`。对这个小模型约占 10%~12%；模型越大、每轮接受数越多，
   这项占比越低。
6. **未做真实加速比测量**：当前 `ms_per_token` 是单次测量的粗值，未与主干
   直接解码做同条件对照。要给出可信加速比，需要用同一 prompt 集合分别跑
   `baseline_greedy` 与 MTP 引擎并对比总耗时（脚本里已有 `baseline_greedy`
   可复用）。
