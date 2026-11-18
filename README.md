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
- **轻量化高性能**：核心模型为 26M 参数量级，结构对齐 Qwen3 生态

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

以 `XLLMConfig` 默认值为准（`model/XLLMModel.py`）：

| 配置项 | 值 | 说明 |
|---|---|---|
| hidden_size | 512 | 隐藏层维度 |
| num_hidden_layers | 8 | Transformer 层数 |
| num_attention_heads | 8 | 查询头数 |
| num_key_value_heads | 2 | KV 头数（GQA 4:1） |
| vocab_size | 6400 | 词表大小 |
| max_position_embeddings | 32768 | 上下文长度 |
| 参数量 | **≈26M** | hidden=512 / 8 层时的实际规模 |

可通过 `--hidden_size` 与 `--num_hidden_layers` 调整规模，例如 `hidden_size=640` 配合 `--use_moe 1` 得到 MoE 变体。

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
- 26M 模型预训练 + SFT 全流程：耗时 ≈ 2.3 小时，总成本 ≈ 3 元
- MoE 变体全流程：耗时 ≈ 3.2 小时，总成本 ≈ 4.2 元

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
