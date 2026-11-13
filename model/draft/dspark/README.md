# dspark —— B：DSpark 式半自回归草稿

对齐 DeepSeek-V4.1 §2.4.3。与 `serial` 的根本差别：**草稿不需要自己的 KV
cache**。并行骨干一次吐出整块候选，再用轻量 Markov 链修正，最后由置信度头
决定每轮提交几个。

## 文件

| 文件 | 作用 |
|---|---|
| `dspark_draft.py` | `DSparkDrafter`：并行骨干 + Markov 头 + 置信度头 |
| `dspark_scheduler.py` | 三种调度器（throughput / confidence / fixed） |
| `dspark_engine.py` | `DSparkEngine`：target-only KV 的解码循环 |
| `dspark_train.py` | 训练脚本（CE + TV + 置信度 BCE） |
| `dspark_selftest.py` | 自测（等价性 / 调度器 / 状态一致 / 无偏性） |

## 三个核心设计

**并行骨干**：把 target 的 hidden states 与 K 个可学习 mask 槽位拼接，一次
前向产出 K 个位置的 base logits。草稿长度从 1 增到 5，延迟基本不变——这是
相对串行草稿的根本收益来源。

**Markov 头**：并行草稿各位置独立预测，块尾准确率会明显下降。Markov 头对第
d 个位置施加只依赖"上一个草稿 token"的低秩加性偏置：

```
logits_d = base_logits_d + markov_proj(markov_emb(prev_token))
```

**置信度调度**：不是每轮都固定提交 K 个，而是由置信度头估计"前几个能中"，
按策略决定提交长度。变长提交是相对固定长度推测解码的另一处优势。

## 特点

**优点**
- **结构上无错位**：草稿没有 KV cache，也就没有"两个 cache 怎么对齐"的问题
- 并行出块，延迟与草稿长度基本无关
- 变长提交，避免长尾浪费

**必须注意**
- **必须真正训练**才有加速比。未训练时 Markov 链与置信度头都是随机值，
  接受率接近随机基线。跑 `dspark_train.py`。
- **target 冻结**：梯度不回传主干，草稿只能**逼近**已有的 p。
  这是它与 `../mtp` 的分水岭——mtp 是预训练期联合训练，草稿与主干共同塑造 p。

## 跑

```bash
cd /d/workspace/xllm
PY=/d/Anaconda2025/envs/py11/python.exe

$PY model/draft/dspark/dspark_selftest.py        # 自测
$PY model/draft/dspark/dspark_train.py --help    # 训练参数
```