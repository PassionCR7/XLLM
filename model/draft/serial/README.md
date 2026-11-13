# serial —— A：串行草稿 + KV 同步（基线实现）

最朴素的一种推测解码：把主模型的前 N 层截断出来当草稿模型，草稿和 target
各自持有 KV cache，每轮由 engine 负责对齐、裁剪、同步。

## 文件

| 文件 | 作用 |
|---|---|
| `draft_model.py` | `DraftModel` 截断式草稿模型（复用 `XLLMForCausalLM`） |
| `speculative_decoding.py` | `SpeculativeDecodingEngine` 解码引擎，负责 KV 同步 |
| `example.py` | 可运行示例 |
| `_selftest.py` | 自测（贪心等价 / 随机无偏） |

## 三种构建方式

1. `from_main_model(main, num_draft_layers)` —— **零训练**，截断已有权重即可跑
2. `from_pretrained(path)` —— 独立训练的草稿 checkpoint
3. 直接构造 —— 从零训练草稿

## 特点

**优点**
- 零训练可用，`from_main_model` 直接就能跑通链路
- 结构最简单，适合做正确性对照实验、验证 `common` 的接受逻辑

**缺点（这是它只作基线的原因）**
- **加速比最弱**：草稿本身是一次真实前向，成本不低；且截断模型与完整模型的
  分布差距大，接受率低。
- **KV 同步是主要复杂度**：接受长度不定，两个 cache 都要截到同一位置，
  这里最容易出 off-by-one。`_selftest.py` 覆盖了这部分。

要真实加速请看 `../dspark` 或 `../mtp`。

## 跑

```bash
cd /d/workspace/xllm
PY=/d/Anaconda2025/envs/py11/python.exe

$PY model/draft/serial/_selftest.py          # 自测
$PY model/draft/serial/example.py --help     # 示例
```