# XLLM

**基于轻量小模型从零复现的大语言模型项目，低成本、全流程可落地、适合 LLM 入门实战学习**

⭐ 本项目为个人学习复现项目，基于 [minimind](https://github.com/jingyaogong/minimind) 开源项目二次学习重构实现，完整复刻小参数 LLM 全链路训练、微调、强化学习、工具调用、模型蒸馏与推理加速能力。

## 📖 项目介绍

XLLM 是一个**从零基于 PyTorch 原生实现的超轻量大语言模型项目**，专为 AI 入门学习者设计，以极低的算力成本、极简的代码结构，完整复现工业级 LLM 的全生命周期技术链路。

所有核心算法、模型结构、训练逻辑均手写实现，无黑盒依赖，适配个人开发者学习、调试、二次开发。

核心优势：

- **超低成本**：单卡 3090 即可完成全流程训练，SFT 阶段仅需 2 小时、GPU 租用成本低至 3 元左右
- **全链路开源**：覆盖预训练、监督微调、LoRA 微调、DPO/PPO/GRPO 强化学习、Agent 工具调用、模型蒸馏（黑盒/白盒）、推测解码加速
- **纯原生实现**：不依赖 transformers 的 Trainer/trl/peft 等高层封装，每一行核心代码均可追溯
- **高兼容性**：支持 llama.cpp、vllm、ollama 等主流推理框架，兼容 OpenAI API 协议
- **轻量化高性能**：Dense 25.83M / MoE 145.03M 两档可选，结构对齐 Qwen3 生态（详见「模型参数规格」）

## ✨ 核心功能特性

### 1. 模型结构
- 支持 Dense 基础模型与 MoE 混合专家模型双架构
- Pre-Norm + RMSNorm、SwiGLU 激活、RoPE 旋转位置编码，支持 YaRN 长文本外推
- GQA 分组查询注意力（8 头 / 2 KV 头，降低 KV cache 占用）
- 自定义 Tokenizer（6400 词表），适配思考标签与工具调用标记
- Flash Attention 加速

### 2. 完整训练链路
- **预训练（Pretrain）**：通用文本语义学习
- **监督微调（SFT）**：多轮对话、指令跟随
- **LoRA 微调**：低成本垂直领域适配
- **偏好对齐（DPO）**：基于人类偏好的回复优化
- **强化学习**：原生实现 PPO / GRPO，含 Agentic RL 多轮工具交互
- **模型蒸馏**：黑盒（API logits）+ 白盒（teacher logits）双路线

### 3. 推理加速（model/draft）
推测解码（Speculative Decoding）三套独立实现，**输出与主模型直接解码完全等价**，但大幅减少主模型前向次数：

| 路线 | 草稿结构 | 适用场景 |
|---|---|---|
| **A. 串行草稿**（serial） | 主模型截断副本 | 只跑通链路、验证正确性，零训练 |
| **B. DSpark 式**（dspark） | 并行骨干 + Markov 头 + 置信度头 | 已有主模型，事后训练草稿逼近 |
| **C. MTP 联合预训练**（mtp） | 共享 emb/lm_head 的多步 Transformer 块 | 从零训主模型，草稿头白拿且主干同步变好 |

三条路线共用验证算法，技术细节见 [`model/draft/README.md`](model/draft/README.md)。

### 4. 数据生产（data_gen）
面向中文垂直领域（本地生活/游玩）的**网页语料采集与改写流水线**，产出与 `dataset/*.jsonl` 完全同构的 `{"text": ...}` 格式：

- 多源站点适配器 + 文本密度正文提取（不依赖固定 CSS 选择器）
- 字符编码自动探测（UTF-8 / GBK / gb18030）
- 主题过滤（本地性 + 游玩意图双判据）、SimHash 近似去重
- **LLM 改写**：把通稿改写成第一人称「打卡帖」，含三道质量闸门（数字事实核查、原文重合度、可游性判定）
- **隐私脱敏**：删除记者署名、职务+姓名降级、手机号/身份证打码

```bash
python -m data_gen.cli list                    # 查看数据源
python -m data_gen.cli all --max-docs 30       # 采集 + 构建语料
python -m data_gen.cli post                    # LLM 改写并合并
python -m data_gen.cli stats                   # 语料统计
```

详见 [`data_gen/README.md`](data_gen/README.md)。

### 5. 拓展能力
- 自适应思考：动态开关显式思维链输出
- 原生 Tool Call：数学计算、时间查询、随机数生成
- 断点续训：中断恢复、跨设备续训
- 可视化训练：SwanLab / Wandb
- OpenAI API 协议部署

## 📊 模型参数规格

### 参数配置 → 参数量对照表

改 `XLLMConfig` 里几个参数，就能得到不同的模型规模。下表是**实算值**（非估算），
每一档都可用 `count_params.py` 复现（命令见表格下方）：

| hidden_size | num_hidden_layers | kv_heads | FFN 公式 | use_moe | 参数量 | 备注 |
|:-----------:|:-----------------:|:--------:|----------|:-------:|-------:|------|
| **512** | 8 | 2 | 8/3 | 0 | **25.83M** | ⬅ 本仓库默认 |
| 512 | 8 | 2 | 8/3 | 1 | **95.05M** | 本仓库默认开 MoE（4 路由 + 1 共享） |
| 640 | 8 | 2 | 8/3 | 0 | 38.84M | |
| **640** | 8 | 2 | 8/3 | 1 | **145.03M** | 官方 minimind2-moe 档（145M） |
| 768 | 8 | 2 | 8/3 | 0 | 54.47M | 只加宽、不换公式 |
| 768 | 8 | 2 | 8/3 | 1 | 205.49M | |
| 768 | 16 | 2 | 8/3 | 0 | 104.03M | 官方 minimind2 档（104M） |
| 768 | 8 | 4 | **π/64** | 0 | **63.91M** | 官方 minimind-3 档（64M） |
| 768 | 8 | 4 | **π/64** | 1 | **198.42M** | 官方 minimind-3-moe（198M-A64M） |

对应的复现命令：

```bash
python count_params.py                                    # 25.83M  本仓库默认
python count_params.py --use_moe 1                        # 95.05M
python count_params.py --hidden_size 640                  # 38.84M
python count_params.py --hidden_size 640 --use_moe 1      # 145.03M
python count_params.py --hidden_size 768                  # 54.47M
python count_params.py --hidden_size 768 --use_moe 1      # 205.49M
python count_params.py --hidden_size 768 --num_hidden_layers 16   # 104.03M
python count_params.py --arch minimind3                   # 63.91M
python count_params.py --arch minimind3 --use_moe 1       # 198.42M
python count_params.py --all                              # 官方五档汇总自查
```

> `--arch minimind3` 会自动带上该世代的默认规格（hidden=768、kv=4、π 公式、QK-Norm），
> 不用手动逐个指定；`--arch minimind2`（默认）则用 8/3 公式与 kv=2。

几个容易踩的点：

- **只加宽不够**：`hidden_size` 512→768 只得 54.47M，离 64M 还差 9.4M，因为 FFN 公式没跟着换。
- **8/3 与 π/64 是两代公式**，同样是 `hidden=768` 时 `intermediate_size` 分别是 `2048` 和 `2432`，
  差值全部落在 FFN 上（FFN 占每层的 76.7%，对总参数量影响最大）。
- **MoE 有两种接法**：minimind2 是「4 路由 + 1 共享专家」（叠加在 dense FFN 之外），
  minimind3 是「4 路由专家」（直接替换 dense FFN），所以同样 768 宽，一个 205.49M、一个 198.42M。
- **各因素的实际贡献**（均按 768 宽 8 层测算，可叠加）：

  | 改动 | 参数量变化 |
  |------|-----------|
  | `hidden_size 512 → 768`（8/3 公式） | **+28.64M**（25.83 → 54.47） |
  | FFN 公式 `8/3 → π/64`（inter 2048 → 2432） | **+7.08M**（54.47 → 61.55） |
  | `kv_heads 2 → 4` | **+2.36M**（61.55 → 63.91） |
  | QK-Norm（q_norm / k_norm） | +1536 个参数（约 +0.002M，可忽略） |

  合计 +38.08M，与 25.83 → 63.91M 吻合。**可见从 512 升级到 768 时，宽度是主要变量，
  但 FFN 公式不跟着换，会少算 7.08M**，这正是「只改 hidden_size 得 54.47M 而非 63.91M」的原因。

### 本仓库默认配置

以 `XLLMConfig` 默认值为准（`model/XLLMModel.py`）：

| 配置项 | 值 | 说明 |
|---|---|---|
| hidden_size | 512 | 隐藏层维度 |
| num_hidden_layers | 8 | Transformer 层数 |
| num_attention_heads | 8 | 查询头数 |
| num_key_value_heads | 2 | KV 头数（GQA 4:1） |
| vocab_size | 6400 | 词表大小 |
| max_position_embeddings | 32768 | 上下文长度 |
| intermediate_size | 1408 | 未显式指定时由 `64*ceil(⌈hidden*8/3⌉/64)` 推导 |
| 参数量 | **25.83M** | Dense，embedding 与 lm_head 共享（tied） |
| MoE 总参数量 | **145.03M** | `--use_moe 1` + `hidden=640`：4 路由 + 1 共享专家 |

#### 参数量拆解（Dense，可逐项复核）

| 组成 | 计算式 | 参数量 | 占比 |
|------|--------|--------|------|
| Embedding（与 lm_head 共享） | 6400 × 512 | 3.28M | 12.7% |
| 单层 Attention（GQA 4:1） | 512² + 2×(512×128) + 512² | 0.655M | — |
| 单层 FFN（SwiGLU） | 3 × 512 × 1408 | **2.163M** | — |
| 单层 RMSNorm ×2 | 2 × 512 | 0.001M | — |
| 8 层合计 | 8 × 2.819M | 22.55M | 87.3% |
| **总计** | 3.28M + 22.55M + 512 | **25.83M** | 100% |

FFN 是参数主体，共 17.30M，占全模型 **67.0%**、占每个 Transformer 层的 76.7%。

#### 与上游 MiniMind 的架构差异（重要）

MiniMind 改过两代架构，`64M` 属于**新一代**，不是本仓库当前代码的规模。
把两代的关键差异列出来，是为了说明「为什么同样叫 XLLM，参数量会差这么多」：

| 项目 | minimind2 世代（本仓库代码） | minimind3 世代（官方口径 64M） |
|------|------------------------------|-------------------------------|
| hidden_size | **512** | **768** |
| FFN 宽度公式 | `64*ceil(int(h*8/3)/64)` → 1408 | `ceil(h*π/64)*64` → 2432 |
| num_key_value_heads | 2（GQA 8:2） | 4（GQA 8:4） |
| QK-Norm | 无 | **有**（q_norm / k_norm） |
| MoE 结构 | 4 路由 + 1 共享专家（叠加） | 4 路由（替换 dense FFN） |

各档对应的具体参数量见本节开头的「参数配置 → 参数量对照表」，此处不再重复。

**结论**：64M / 198M 是 MiniMind-3 的规格，25.83M / 145.03M 是本仓库当前 minimind2 世代代码的规格。
旧 README 直接照搬上游新版数字，才造成「文档写 64M、代码只有 25.83M」的错位。

如需升级到 64M 档，需同时改四项（只改 hidden_size 不够）：

```python
# model/XLLMModel.py
hidden_size: int = 768,           # 512 -> 768
num_key_value_heads: int = 4,     # 2 -> 4
# FeedForward 内改为: intermediate_size = math.ceil(hidden_size * math.pi / 64) * 64
# Attention 内补 q_norm / k_norm 两个 RMSNorm(head_dim)
```

四项齐改后实算 **63.91M**，与官方 64M 一致；只改 `hidden_size=768` 只能得到 54.47M。

> 以上数字全部可用 `python count_params.py --all` 复核（纯 Python，不依赖 torch），
> 该脚本同时内置 minimind 五个官方档位（26M / 104M / 145M / 64M / 198M）的自动比对。
> 本仓库 `bench_result.json` 的实测记录（Dense 25.83M / MoE 95.05M）与脚本对当前默认配置的输出逐位一致。

可通过 `--hidden_size` 与 `--num_hidden_layers` 调整规模，例如 `hidden_size=640` 配合 `--use_moe 1` 得到 MoE 变体。

注意 `eval.py` 中 `--hidden_size` 帮助文本的「640=MoE-145M」指的是**上游 640 档**（4 路由 + 1 共享），
实算 145.03M 与之一致；但本仓库 `XLLMConfig` 的 MoE 默认 `hidden=512`，实算 95.05M，
两者不同源，取数时需确认 `hidden_size` 参数。

## 💻 环境依赖配置

### 1. 基础软硬件环境（推荐）
- 系统：Ubuntu 20.04+ / Windows 10+ / macOS
- GPU：NVIDIA 3090/4090（24G 显存及以上，单卡即可）
- CUDA：12.2
- Python：3.10+
- 内存：≥32G

### 2. 依赖安装

项目使用 `pyproject.toml` 管理依赖（**没有** `requirements.txt`）：

```bash
# 克隆仓库
git clone https://github.com/PassionCR7/XLLM.git
cd XLLM

# 方式一：uv（推荐）
uv sync

# 方式二：pip（阿里云镜像加速）
pip install numpy pandas torch transformers -i https://mirrors.aliyun.com/pypi/simple
```

核心依赖：`numpy>=2.3`、`pandas>=2.3`、`torch>=2.9`、`transformers>=4.57`。

蒸馏子模块另有独立 `requirements.txt`，见各自目录。

## 📂 项目目录结构

```
XLLM/
├── model/                        # 模型核心结构
│   ├── XLLMModel.py              # 主模型（XLLMConfig / XLLMForCausalLM / MoE）
│   ├── model_lora.py             # LoRA 适配层
│   ├── tokenizer.json            # 6400 词表
│   ├── tokenizer_config.json
│   └── draft/                    # 推测解码三套实现
│       ├── common/               # 共用验证算法
│       ├── serial/               # A. 串行草稿（零训练）
│       ├── dspark/               # B. DSpark 式（并行骨干 + 置信度调度）
│       └── mtp/                  # C. MTP 联合预训练（含架构图与踩坑记录）
│
├── trainer/                      # 全流程训练脚本
│   ├── train_pretrain.py         # 预训练
│   ├── train_full_sft.py         # 全参 SFT
│   ├── train_lora.py             # LoRA 微调
│   ├── train_dpo.py              # DPO 偏好对齐
│   ├── train_ppo.py              # PPO 强化学习
│   ├── train_grpo.py             # GRPO 强化学习
│   ├── train_agent.py            # Agentic RL（多轮工具调用）
│   ├── rollout_engine.py         # RL rollout 引擎
│   └── trainer_utils.py          # 公共工具（种子、学习率、日志）
│
├── method/                       # 底层算法单元（独立可测）
│   ├── rmsnorm.py                # RMSNorm
│   ├── rope.py                   # RoPE 旋转位置编码
│   └── GQA.py                    # 分组查询注意力
│
├── distillation_blackbox/        # 黑盒蒸馏（教师为 API，仅用 logits）
│   ├── teacher_api.py            # 教师 API 封装
│   ├── gen_data.py               # 教师生成 SFT 数据
│   ├── train_kd.py               # 学生训练
│   ├── evaluate.py / infer.py / trace_demo.py
│   └── reports/                  # 评估结果
│
├── distillation_whitebox/        # 白盒蒸馏（可读教师权重，对齐 logits 分布）
│   ├── teacher_logits.py         # 教师 logits 缓存
│   ├── student_loader.py         # 学生模型加载
│   ├── vocab_align.py            # 师生词表对齐
│   ├── train_kd.py / dataset.py / evaluate.py / infer.py
│   └── reports/
│
├── data_gen/                     # 中文语料采集与改写流水线
│   ├── fetch.py                  # HTTP 层（编码探测/退避重试/限速/robots）
│   ├── extract.py                # 正文提取（文本密度打分）
│   ├── sources.py                # 站点适配器与主题词表
│   ├── crawl.py                  # 采集编排与主题过滤
│   ├── build_corpus.py           # 去重/重组/清洗 + 隐私脱敏
│   ├── llm.py                    # OpenAI 兼容客户端
│   ├── posting.py                # LLM 改写成打卡帖 + 事实核查
│   ├── cli.py                    # 命令行入口
│   └── test_data_gen.py          # 19 组回归测试
│
├── dataset/                      # 数据集与奖励模型
│   ├── *.jsonl                   # 训练数据（见下节）
│   └── internlm2-1_8b-reward/    # 奖励模型（PPO/GRPO 用）
│
├── eval.py                       # 推理与对话入口
├── bench_speed.py                # 推理速度基准（TTFT / TPOT / 吞吐）
├── bench_result.json             # 基准测试结果
├── count_params.py               # 参数量核算（纯 Python，不依赖 torch）
├── XLLMModel.py                  # 早期单文件版本（保留兼容，新代码用 model/）
├── main.py                       # 占位入口
├── pyproject.toml                # 依赖与构建配置
└── README.md
```

> `checkpoints/`、`out/` 为本地训练产物目录，不入库。

## 📥 数据集下载与配置

本项目沿用 minimind 开源标准化数据集，无需自行预处理，分为**轻量化快速复现版**和**完整版**。

### 1. 数据集清单

**核心必备（新手推荐）**
- `pretrain_t2t_mini.jsonl`（1.2GB）：轻量化预训练数据
- `sft_t2t_mini.jsonl`（1.6GB）：轻量化 SFT 对话数据
- `rlaif.jsonl`（24MB）：RL 强化学习数据

**进阶完整版**
- `pretrain_t2t.jsonl`（10GB）：全量预训练数据
- `sft_t2t.jsonl`（14GB）：全量 SFT 对话 & 工具调用数据
- `dpo.jsonl`（53MB）：DPO 偏好对齐数据
- `agent_rl.jsonl`（86MB）：Agent 多轮工具交互数据
- `agent_rl_math.jsonl`（18MB）：数学推理强化学习数据

**奖励模型**：`dataset/internlm2-1_8b-reward/` 需从 ModelScope / HuggingFace 拉取完整权重（`model-00001/00002-of-00002.safetensors`）。

### 2. 下载方式

- ModelScope（国内高速）：`gongjy/minimind-3`
- HuggingFace：`jingyaogong/minimind-3`

### 3. 配置

所有 `.jsonl` 放入项目根目录的 `./dataset/` 下，训练脚本直接读取。

## 🚀 快速运行教程

> 本项目**不自带权重**，预训练权重均从 minimind 官方下载

### 一、模型推理

#### 1. 下载官方预训练权重
```bash
# ModelScope
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3

# 或 HuggingFace
git clone https://huggingface.co/jingyaogong/minimind-3
```

#### 2. 命令行推理
```bash
# transformers 格式权重
python eval.py --load_from ./minimind-3

# 原生 PyTorch 权重（默认读 checkpoints/ 下的 .pth）
python eval.py --load_from model --weight full_sft

# 推理 Reason 模型（自动启用 enable_thinking，输出显式思维链）
python eval.py --load_from model --weight reason

# 加载 LoRA 权重
python eval.py --load_from model --weight full_sft --lora_weight lora_identity
```

`--weight` 可选：`pretrain` / `full_sft` / `rlhf` / `reason` / `ppo_actor` / `grpo` / `spo`

#### 3. 推理速度基准
```bash
python bench_speed.py
```
输出 TTFT（首 token 延迟）、TPOT（每 token 耗时）、吞吐等指标，结果写入 `bench_result.json`。

### 二、从零训练全流程

所有训练脚本在 `./trainer` 目录下执行，支持单卡/多卡、断点续训。

#### 1. 预训练（必备）
```bash
cd trainer
python train_pretrain.py
python train_pretrain.py --from_resume 1        # 断点续训
```

#### 2. 监督微调 SFT（必备）
```bash
python train_full_sft.py
python train_full_sft.py --from_resume 1
```

#### 3. 进阶训练
```bash
python train_lora.py      # LoRA 微调
python train_dpo.py       # DPO 偏好对齐
python train_ppo.py       # PPO
python train_grpo.py      # GRPO
python train_agent.py     # Agentic RL（多轮工具调用）
```

#### 4. 模型蒸馏

```bash
# 白盒（可读教师权重）
cd distillation_whitebox
python train_kd.py --epochs 2 --batch_size 4 --max_seq_len 192
python evaluate.py --weights full_sft distill_whitebox --limit 32

# 黑盒（教师为 API）
cd distillation_blackbox
python gen_data.py           # 教师生成数据
python train_kd.py
```

> `teacher_logits.py` 是库模块（字符区间对齐 + logits 缓存），由 `train_kd.py` 内部调用，不单独执行。

各目录的 `README.md` 有完整参数说明。

#### 5. 推测解码

```bash
# A. 串行草稿（零训练，直接验证）
cd model/draft/serial
python example.py

# B. DSpark（先训练草稿）
cd model/draft/dspark
python dspark_train.py
python dspark_selftest.py

# C. MTP 联合预训练
cd model/draft/mtp
python mtp_pretrain.py
python _mtp_check.py
```

## ⚙️ 核心特性使用说明

### 1. 自适应思考开关
思维链输出由权重类型控制：加载 `--weight reason` 时，`eval.py` 会自动置 `enable_thinking=True`，模板中插入思考标签；加载其他权重则为直接回答。

```bash
python eval.py --load_from model --weight reason
```

### 2. 权重合并与导出
LoRA 权重与基础模型合并、导出完整部署权重，逻辑见 `model/model_lora.py`（`apply_lora` / `merge_lora` / `save_lora`）。

### 3. OpenAI API 部署
兼容标准 OpenAI 协议，可接入任意第三方对话 UI。`model/XLLMModel.py` 中的 `XLLMForCausalLM` 继承 `PreTrainedModel`，可直接被 vllm / transformers 的 serving 层加载。

## 📈 训练成本与效率

基于单卡 NVIDIA 3090 实测：
- Dense 25.83M（`hidden=512`）预训练 + SFT 全流程：耗时 ≈ 2.3 小时，总成本 ≈ 3 元
- MoE 145.03M（`hidden=640`，4 路由 + 1 共享专家）全流程：耗时 ≈ 3.2 小时，总成本 ≈ 4.2 元

> 若改用 `hidden=768` 的 64M 档或 198M MoE 档，参数约为前者的 2.5 倍与 1.4 倍，成本需相应上浮，
> 上述耗时/费用仅对 512 / 640 两档实测有效。

## 🧪 测试

```bash
# 数据流水线回归测试（19 组，覆盖实测踩过的坑）
python -m data_gen.test_data_gen
```

## 📝 项目声明

- **学习复现说明**：本项目基于 **jingyaogong/minimind**开源项目学习、理解、重构实现，仅用于技术学习与交流。
- **开源协议**：沿用 Apache 2.0，免费开源，可自由学习、二次开发。
- **资源说明**：本仓库**不存放任何权重与数据集**，均来自原 minimind 官方地址。
- **数据合规**：`data_gen` 采集的语料会做隐私脱敏（署名、真实姓名、手机号、身份证号），产物目录不入库。

## 🌟 欢迎 Star & Fork

如果本项目对你的 LLM 学习、实战开发有帮助，欢迎 **Star** 收藏、**Fork** 二次开发！
