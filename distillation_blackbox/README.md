# 知识蒸馏：黑盒与白盒两套实现

本目录是 xllm 项目的知识蒸馏扩展，两个子目录各自独立、互不依赖：

| 目录 | 蒸馏范式 | 教师 | 可获得信号 |
|------|---------|------|-----------|
| `distillation_blackbox/` | 黑盒（response / logprob 级） | `deepseek-v4.1-flash-1017`（API） | 文本 + token logprob（top-5） |
| `distillation_whitebox/`  | 白盒（logits 级分布蒸馏） | `Qwen3-0.6B`（本地权重） | 完整 logits（151936 维） |

学生模型统一为 xllm（25.83M 参数，词表 6400，hidden 512，8 层）。

---

## 一、黑盒蒸馏 `distillation_blackbox/`

### 为什么教师默认关闭 thinking

实测该 API 是推理型模型：默认把内容写入 `reasoning_content`，
`message.content` 为空。若直接蒸馏，会得到大量空样本。

关闭 thinking 后有两个关键收益：

1. `content` 有正常回答，可用于序列级蒸馏；
2. `logprob` 变为**真实概率**。未关闭时返回的是 `-1e38` 哨兵值
   （无效占位），关闭后实测为 `-0.0014` 量级。

```python
payload["chat_template_kwargs"] = {"enable_thinking": False}
```

### 三种蒸馏层次

| 层次 | 信号 | 实现 |
|------|------|------|
| Sequence-level | 教师生成的完整回复 | 直接做 SFT（`--no_soft`） |
| Logprob-level | top-5 token 概率 | 稀疏软标签 KL（默认启用） |
| 混合 | 两者加权 | `--ce_weight / --kd_weight` |

### 稀疏软标签的构造（关键技术点）

API 只返回 top-5，直接重归一化会**人为放大教师置信度**。
本实现采用"背景质量兜底"：保留 top-k 的绝对概率，
剩余质量 `1 - sum(p_topk)` 作为背景项参与 KL：

```
p_i  = exp(logprob_i)          # top-k
p_bg = 1 - Σ p_i               # 剩余质量
KL = Σ p_i·(log p_i − log q_i) + p_bg·(log p_bg − log q_bg)
```

`q_bg` 是学生在非 top-k token 上的概率和，因此教师的不确定性被完整保留。

### 两个已修复的隐蔽缺陷

**1. 教师 token 与原文长度不相等**

API 返回的 token 是解码后的近似字符串。实测拼接后 363 字符，
而原文 343 字符（`\n\n` 等转义长度失真）。按字符长度累加会**逐位漂移**，
导致软标签整体错位，产生静默的错误监督。

解决：用 `difflib.SequenceMatcher` 把教师 token 拼接串与原文做序列对齐，
取匹配块把每个 token 映射到精确字符区间；对齐点不足一半的 token 直接丢弃。

**2. `get_vocab()` 不能用于文本反查**

`tokenizer.get_vocab()` 返回的是 BPE 原始形如 `Ġhello`（含字节映射字符），
而 API 返回的是可读文本 `" hello"`。直接查表全部 MISSING（实测 0% 命中）。

解决：用 `tokenizer.encode(piece)` 反查，且仅当编码结果为**单 token** 时采用。
覆盖率从 0% 提升到 55.8%，配合对齐后软标签实际命中 assistant token 的 68~78%。

### 用法

```bash
cd distillation_blackbox

# 1) 探测教师能力（先跑这个，确认 logprob 有效）
python gen_data.py --probe_only

# 2) 生成教师数据（支持断点续生成）
python gen_data.py --num_samples 2000 --concurrency 8

# 3) 训练
python train_kd.py --epochs 2 --batch_size 8 --max_seq_len 512

# 纯序列级蒸馏（不用软标签）
python train_kd.py --no_soft

# 4) 评估
python evaluate.py --weights full_sft distill_blackbox
```

---

## 二、白盒蒸馏 `distillation_whitebox/`

### 教师选型：为什么是 Qwen3-0.6B

在 8GB 显存（RTX 4060 Laptop）约束下比较了 Qwen 系列：

| 模型 | 参数量 | bf16 显存 | 结论 |
|------|--------|----------|------|
| **Qwen3-0.6B** | 0.6B | ~1.2GB | **采用**：可全精度前向，batch 到 8 |
| Qwen3-1.7B | 1.7B | ~3.4GB | 显存可容但 batch 需降到 2；且 HF 下载需约 2 小时 |
| Qwen3-4B+ | 4B+ | >8GB | 8GB 无法前向，量化会破坏 logits 精度 |

**决定性理由**：白盒蒸馏需要精确 logits，量化教师会引入系统偏差，
必须全精度加载，因此参数量被 8GB 显存硬约束。
0.6B 教师相对 25.83M 学生仍有 **23 倍**参数优势，容量差足以产生有意义的软标签。

实测教师前向显存：batch 8 × 长度 256 峰值 **1.92GB**，剩余 6GB 可支撑学生反传。

> 如需更强教师，把 `config.py` 的 `model_path` 指向 Qwen3-1.7B 并调低
> `batch_size` 即可，实现不依赖具体规模。

### 词表对齐（本实现最核心的技术点）

学生词表 6400，教师词表 151936。实测：

- 字符串层面重合 token 占学生词表 **91.4%**
- 但**几乎不存在相同 id**（重合 token 中仅 0.02% id 一致）

因此绝不能假设 `teacher_logits[..., i]` 与 `student_logits[..., i]` 对应同一个词。

采用**三级映射**：

1. **字节精确匹配**：把两侧 token 都还原为原始字节
   （GPT-2 byte-level BPE 的 `bytes_to_unicode` 逆映射），字节完全相同才映射。
   这是最可靠的一级，覆盖学生词表 91.41%。
2. **剔除替换字节后再匹配**：处理词表中存成 `U+FFFD` 的损坏字节。
3. **丢弃**：仍无对应者不参与 KD（从 KL 分母剔除，而非置零）。

### 跨分词器位置对齐

两套 tokenizer 对同一句话的切分粒度不同，logits 序列长度不一致，
无法逐位置比较。采用**字符区间对齐**：

```
学生 token i 的字符区间 (s_start, s_end)
教师 token j 的字符区间 (t_start, t_end)

若 s_end == t_end  →  teacher_logits[j] 对齐到 student_logits[i]
```

**为什么要求边界严格相等**：自回归模型的 `logits[j]` 表示"看完 `S[0:t_end]`
后对下一个 token 的预测"。只有当 `s_end == t_end` 时，两边才是在**同一个前缀**
下做预测，分布才可比。若允许区间重叠，会把"多看了几个字符"的预测当成同一目标，
引入系统性偏差。

实测中文语料边界命中率约 **76~85%**。未命中位置通过 `align_mask=0` 从损失剔除。

### 温度选择（实测校正）

教材默认 T=2，但实测该教师输出分布已相当尖锐：

| T | 平均熵 | 说明 |
|---|--------|------|
| 1.0 | 2.34 | 尖锐，教师判断明确 |
| 1.5 | 3.30 | 适度软化 |
| **2.0** | **4.52** | 超过均匀分布上限(8.67)的 50%，**过度平滑** |

T=2 会把分布推向"什么词都行"的均匀状态，反而伤害蒸馏。
本实现默认 **T=1.2**，并保留实测数据供复现。

### 损失

```
L = ce_weight · CE(student, hard_labels)
  + kd_weight · KL( softmax(t_logits/T) ‖ softmax(s_logits/T) ) · T²
```

`T²` 缩放使 KD 梯度量级与 CE 可比（Hinton et al. 2015）。
KL 计算统一转 float32（bf16 下 softmax/KL 会累积误差）。

### 显存控制

教师词表 151936，`[B, L, 151936]` 的 logits 极其庞大
（B=4, L=256 约 311MB）。本实现只提取映射到的 5850 列，
并对齐结果做**磁盘缓存**，第二个 epoch 起无需重跑教师。

### 用法

```bash
cd distillation_whitebox

# 训练（教师 logits 首次前向后自动缓存）
python train_kd.py --epochs 3 --batch_size 4 --max_seq_len 192

# 快速验证（小样本）
python train_kd.py --limit 16 --epochs 3 --batch_size 4

# 评估：KL / Top-1 一致率 / PPL
python evaluate.py --weights full_sft distill_whitebox --limit 24
```

---

## 三、关键约定与注意事项

### 1. checkpoint 键名拼写缺陷（已自动修正）

项目中 `out/pretrain_512.pth` 存在键名拼写错误：

```
model.layers.N.post_attention_layernotm.weight   ← 错误（应为 layernorm）
```

若直接 `load_state_dict(strict=False)`，会**静默漏加载 8 层后注意力归一化权重**
（`missing_keys` 非空但被忽略），模型"加载成功"却行为异常。

两个目录的加载器都会在加载前自动修正，并报告修正数量：

```
[学生] 已修正 8 个键名拼写错误 (layernotm -> layernorm)
[学生] 权重完整加载
```

### 2. 跨分词器 Top-1 一致率天然偏低

白盒评估中 Top-1 一致率约 40~50% 属**正常**。原因：学生把 `"，请"` 合成一个
token，教师倾向拆成 `"，"` + `"请"`，同一前缀下 argmax 落在不同粒度。

判据应看 **KL 是否下降** 与 **PPL 是否改善**，而非 Top-1。

### 3. 训练数据规模

黑盒冒烟测试用 24 条数据，CE 停在 3.19 高位，属预期（数据量过小）。
正式训练建议 ≥ 2000 条北京时间：
按实测 0.7 条/s（并发 8 时更高），2000 条约需 30~50 分钟。

### 4. 环境

```bash
# 本项目可用的 Python 环境（已装 torch 2.5.1 + CUDA 12.1）
D:\Anaconda2025\envs\py11\python.exe

# 白盒教师权重（ModelScope 下载，约 1.5GB）
D:\models\models\Qwen--Qwen3-0.6B\snapshots\master

# 可用环境变量覆盖
set XLLM_TEACHER_PATH=<你的教师权重目录>
```

### 5. 教师模型下载

HF 直连实测仅约 50~176 KB/s，ModelScope 可达 **15~19 MB/s**：

```python
from modelscope import snapshot_download
snapshot_download("Qwen/Qwen3-0.6B", cache_dir=r"D:\models")
```

### 6. 推理

训练完成后用 `infer.py` 加载蒸馏权重做对话。它独立于 `evaluate.py`，
支持三种模式：

```bash
cd distillation_blackbox

# 单条提问
python infer.py --prompt "你是谁？"

# 与基线对照（验证蒸馏确实改变了行为）
python infer.py --weight full_sft --prompt "你是谁？"

# 交互式多轮（保留历史，exit 退出 / clear 清空）
python infer.py --chat

# 批量：每行一个提问
python infer.py --input questions.txt --output answers.jsonl --batch_size 8

# 贪心解码（结果可复现）+ 查看每 token 概率
python infer.py --prompt "你是谁？" --greedy --logprobs
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--weight` | 权重名，`distill_blackbox`（默认）或 `full_sft` 做对照 |
| `--max_new_tokens` | 最大生成长度，默认 256 |
| `--temperature` / `--top_p` | 采样参数，默认 0.7 / 0.9 |
| `--greedy` | 关闭采样，走 argmax |
| `--no_repeat_ngram 3` | 抑制复读（小模型常见问题） |
| `--show_think` | 保留思维链段落 |
| `--logprobs` | 输出每个生成 token 的 logprob |

注意：`--weight` 的路径解析结果会写回 config 后再加载。早期版本只解析不写回，
导致 `--weight` 被静默忽略、所有请求实际都在加载 `full_sft`——这个坑已修。

**关于复读**：这个 25.83M 学生只用 300 条教师数据蒸馏，贪心解码下容易出现
段落级自我重复（把同一句话连着输出三遍）。实测各方案效果：

| 做法 | 效果 | 副作用 |
|------|------|--------|
| 默认（采样 T=0.7, top_p=0.9） | 不出现整段复读 | 输出随机，后半段偶尔跑题 |
| `--repetition_penalty 1.15` | **仍复读**，只是每遍内容略变 | 无 |
| `--repetition_penalty 1.3` | **完全消除**，行文连贯 | 语气变浮夸，夹带 emoji |
| `--no_repeat_ngram 3` | 消除逐字重复 | 贪心+强抑制下会编造事实 |

实测同一提问「你是谁？」，贪心解码：

```
无抑制            → "我是…minimind，由中国的个人开发者独立开发。" 重复 3 遍
penalty=1.15      → 仍然重复 3 遍（每遍措辞微调），等于无效
penalty=1.3       → 单段完整回答，不复读，但出现 😊🌟🎉
ngram=3           → 不复读，但后半段编造"基于阿里云的智能助手"
```

**推荐**：日常对话用默认采样即可；要可复现的结果用
`--greedy --repetition_penalty 1.3`。注意 `penalty=1.15` 这个常见推荐值
在本模型上不起作用，需要到 1.3 才生效。

根因是数据量太少（300 条）而非解码配置，调参只能缓解表象。
要提高生成质量需要扩充教师数据或增大训练轮数。

---

## 四、文件清单

### `distillation_blackbox/`

| 文件 | 作用 |
|------|------|
| `config.py` | 配置（从 api key 文件解析密钥，不硬编码） |
| `teacher_api.py` | 教师 API 客户端（重试 / 并发 / logprob 健康度校验） |
| `gen_data.py` | 生成教师数据（质量门控 + 断点续生成） |
| `train_kd.py` | 训练（序列级 + 稀疏 logprob 软标签） |
| `evaluate.py` | 评估（PPL + 生成样例） |
| `infer.py` | **推理**（单条 / 交互 / 批量，支持 logprob 输出） |
| `trace_demo.py` | 单样本 CE/KL 逐位置追踪（讲解用，不参与训练） |

### `distillation_whitebox/`

| 文件 | 作用 |
|------|------|
| `config.py` | 配置（含教师选型的量化对比依据） |
| `vocab_align.py` | 词表对齐（字节级三级映射） |
| `dataset.py` | 数据集（assistant 区间标注） |
| `teacher_logits.py` | 教师 logits 提取 + 跨分词器字符对齐 + 缓存 |
| `student_loader.py` | 学生加载（自动修正键名缺陷） |
| `train_kd.py` | 训练（KL + CE 联合损失） |
| `evaluate.py` | 评估（KL / Top-1 / PPL） |
| `infer.py` | **推理**（单条 / 交互 / 批量，可与教师并排对比） |
| `trace_demo.py` | 单样本 CE/KL 逐位置追踪（讲解用，不参与训练） |

---

## 五、验证结果（实测）

### 黑盒（300 条教师数据 × 3 epoch，评估 32 样本）

| 指标 | full_sft（基线） | distill_blackbox | 变化 |
|------|-----------------|------------------|------|
| **PPL ↓** | 33.6185 | **21.0894** | **−37.3%** |

教师能力探测：

```
reachable=True, content_nonempty=True,
supports_logprobs=True, logprob_valid_ratio=1.0
```

数据生成：300/300 成功，149 次请求 **0 失败、0 空回复**，速率约 1.1 条/s。

软标签质量：
- top-k → 学生单 token 映射率 **55.8%**
- assistant token 实际命中 **68~78%**
- 概率分布合理（如"我" p=0.946，背景质量 0.053）
- 质量门控：`Σp_topk < 0.01` 的位置直接舍弃（实测占教师 token 的 **17.3%**）。
  这类位置通常是教师 top-1 本身就是多 token（如 `'哈哈'` → `'哈','哈'`），
  剩余质量 `p_bg ≈ 1`，KL 退化为常数、对参数无梯度贡献，却会稀释分母。

### 白盒（对照参考）

| 指标 | 基线 | distill_whitebox | 变化 |
|------|------|------------------|------|
| KL ↓ | 1.8642 | 1.3470 | −27.8% |
| PPL ↓ | 6.4100 | 4.0649 | −36.6% |

> 两套 PPL 不可直接比较：黑盒在教师生成的 300 条数据上评估，
> 白盒在项目 SFT 数据上评估，分布不同。各自对比本目录基线即可。