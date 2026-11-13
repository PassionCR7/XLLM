# 白盒蒸馏（logits 级分布蒸馏）

学生：xllm（25.83M，词表 6400）　教师：Qwen3-0.6B（596M，词表 151936）

与同级的 `distillation_blackbox/` 完全独立，两者可分别运行。

---

## 一、教师选型依据

在 8GB 显存（RTX 4060 Laptop）下实测比较 Qwen 系列：

| 模型 | 参数量 | bf16 显存 | 结论 |
|------|--------|----------|------|
| **Qwen3-0.6B** | 0.6B | ~1.2GB | **采用**：全精度前向，batch 可到 8 |
| Qwen3-1.7B | 1.7B | ~3.4GB | 显存可容但 batch 需降到 2；HF 下载约 2 小时 |
| Qwen3-4B+ | 4B+ | >8GB | 8GB 无法前向 |

**决定因素**：白盒蒸馏需要精确 logits，量化教师会引入系统偏差，
必须全精度加载，因此参数量受 8GB 显存硬约束。
0.6B 教师相对 25.83M 学生仍有 **23 倍**参数优势。

实测教师前向峰值显存（bf16）：

```
batch=2, len=256 → 1.32GB
batch=4, len=256 → 1.52GB
batch=8, len=256 → 1.92GB
```

> 换更强教师只需改 `config.py` 的 `model_path` 并降低 `batch_size`，
> 代码不依赖具体规模。

---

## 二、三个核心技术问题

### 1. 词表不对齐

学生词表 6400，教师 151936。实测：

- 字符串层面重合 token 占学生词表 **91.4%**
- 但**几乎不存在相同 id**：重合 token 中仅 0.02% 的 id 一致

所以 `teacher_logits[..., i]` 与 `student_logits[..., i]` **不是同一个词**，
直接算 KL 是错误的。

采用三级映射（`vocab_align.py`）：

1. **字节精确匹配**：用 GPT-2 byte-level BPE 的 `bytes_to_unicode` 逆映射，
   把两侧 token 都还原为原始字节，字节完全相同才映射 → 覆盖 91.41%
2. **剔除替换字节后再匹配**：处理存成 `U+FFFD` 的损坏字节
3. **丢弃**：仍无对应者不参与 KD（从 KL 分母剔除，不置零）

### 2. 位置不对齐

两套 tokenizer 切分粒度不同，logits 序列长度不一致。

采用**字符区间对齐**（`teacher_logits.py`）：

```
学生 token i 的字符区间 (s_start, s_end)
教师 token j 的字符区间 (t_start, t_end)
若 s_end == t_end  →  teacher_logits[j] 对齐到 student_logits[i]
```

**为什么必须边界严格相等**：`logits[j]` 表示"看完 `S[0:t_end]` 后预测下一个
token"。只有 `s_end == t_end` 时，两边才在**同一前缀**下预测，分布才可比。
若允许区间重叠，会把"多看几个字符"的预测当同一目标，引入系统偏差。

实测中文语料边界命中率 **76%~85%**，未命中位置用 `align_mask=0` 剔除。

### 3. 温度过度平滑

教材默认 T=2，但实测该教师分布本已尖锐：

| T | 对齐位置平均熵 | 说明 |
|---|--------------|------|
| 1.0 | 2.34 | 尖锐 |
| 1.5 | 3.30 | 适度软化 |
| 2.0 | **4.52** | 超过均匀分布上限(8.67)的 50%，**过度平滑** |
| 3.0 | 6.76 | 接近均匀，已失去判别信息 |

T=2 会把学生推向"什么词都行"的状态。本实现默认 **T=1.2**。

---

## 三、损失与显存

```
L = ce_weight · CE(student, hard_labels)
  + kd_weight · KL( softmax(t_logits/T) ‖ softmax(s_logits/T) ) · T²
```

- `T²` 缩放使 KD 梯度与 CE 可比（Hinton et al. 2015）
- KL 统一 float32 计算（bf16 下 softmax/KL 会累积误差）
- 教师 `no_grad` 前向，且只取映射到的 5850 列（非 151936）

**显存实测**：batch=4、len=192 时，教师前向 + 学生反传峰值 **2.07GB**。

**教师 logits 磁盘缓存**：首次前向后按样本落盘，第二个 epoch 起直接读取。
2000 样本 × 192 长度的缓存约 4.3GB（已加入 `.gitignore`）。

---

## 四、用法

```bash
cd distillation_whitebox

# 训练（默认 full_sft 起步，T=1.2，batch=4）
python train_kd.py --epochs 2 --batch_size 4 --max_seq_len 192

# 快速验证链路
python train_kd.py --limit 16 --epochs 3 --batch_size 4 --max_seq_len 192

# 评估（KL / Top-1 / PPL）
python evaluate.py --weights full_sft distill_whitebox --limit 32
```

教师权重路径默认：

```
D:\models\models\Qwen--Qwen3-0.6B\snapshots\master
```

可用环境变量 `XLLM_TEACHER_PATH` 覆盖。

---

## 五、实测结果

### 2000 样本 × 2 epoch（最终交付权重）

| 指标 | full_sft（基线） | distill_whitebox | 变化 |
|------|-----------------|------------------|------|
| **KL ↓** | 2.2579 | **1.2855** | **−43.1%** |
| **PPL ↓** | 6.4100 | **4.0649** | **−36.6%** |
| Top-1 一致率 | 36.74% | 44.73% | +8.0pp |

### 关于 Top-1 一致率偏低

跨分词器场景下 Top-1 一致率**不可能很高**，属正常现象：

学生 tokenizer 把 `"，请"` 合成一个 token，教师倾向拆成 `"，"` + `"请"`。
同一字符前缀下两边的 argmax 落在不同粒度上。

**判据应看 KL 是否下降、PPL 是否改善**，Top-1 仅作参考。

### 训练日志（收敛性）

```
Epoch 1: loss 4.94 → 4.28（CE 2.01 → 1.75，KD 2.97 → 2.53）
Epoch 2: loss 4.28 → 3.93（CE 1.75 → 1.60，KD 2.53 → 2.33）
```

---

## 六、注意事项

### checkpoint 键名缺陷（已自动修正）

项目 `out/pretrain_512.pth` 存在拼写错误：

```
model.layers.N.post_attention_layernotm.weight   ← 错误
model.layers.N.post_attention_layernorm.weight   ← 正确
```

直接 `load_state_dict(strict=False)` 会**静默漏加载 8 层归一化权重**，
模型"加载成功"但行为异常。`student_loader.py` 会自动修正并报告：

```
[学生] 已修正 8 个键名拼写错误 (layernotm -> layernorm)
[学生] 权重完整加载
```

### 环境

```bash
D:\Anaconda2025\envs\py11\python.exe   # torch 2.5.1 + CUDA 12.1
```

### 教师下载

HF 直连实测 50~176 KB/s，ModelScope 可达 15~19 MB/s：

```python
from modelscope import snapshot_download
snapshot_download("Qwen/Qwen3-0.6B", cache_dir=r"D:\models")
```

---

## 七、文件清单

| 文件 | 作用 |
|------|------|
| `config.py` | 配置 + 教师选型的量化依据 |
| `vocab_align.py` | 词表对齐（字节级三级映射）+ KL 损失 |
| `dataset.py` | 数据集（assistant 区间标注） |
| `teacher_logits.py` | 教师 logits 提取 + 字符对齐 + 磁盘缓存 |
| `student_loader.py` | 学生加载（自动修正键名缺陷） |
| `train_kd.py` | 训练主流程 |
| `evaluate.py` | 评估（KL / Top-1 / PPL / 生成样例） |