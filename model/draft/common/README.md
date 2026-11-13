# common —— 三套实现共用的接受/验证核心

`_verification.py` 只有三个函数，但三套引擎（`serial` / `dspark` / `mtp`）全部
依赖它。这是**推测解码正确性的唯一来源**。

## 为什么不各写一份

接受逻辑看起来只有十几行，很容易想"顺手在 engine 里写一下"。但历史上这种
复制粘贴会立刻出问题：改了一处的 top_p 归一化，忘了另一处，结果就是某条路线
输出正确、另一条悄悄有偏，而且**很难在 loss 曲线上发现**——因为 loss 照常下降。

所以这里强制单一实现，三套 engine 共用，它们的无损性保证天然一致。

## 接口

| 函数 | 作用 |
|---|---|
| `prepare_distribution(logits, temperature, top_k, top_p)` | logits → 概率分布 |
| `verify_and_sample(...)` | 一次并行验证，返回接受的 token 数与压哨 token |
| `acceptance_stats(...)` | 接受率与近似加速比 |

## 接受规则

- **贪心**：`argmax(p_i) == d_i` 则接受；被拒处输出 `argmax(p_i)`。
  结果与 target 贪心解码**逐 token 相同**。
- **随机**：以 `min(1, p_i(d_i)/q_i(d_i))` 接受；被拒则从 `(p_i - q_i)_+`
  归一化后采样。结果与 target 采样解码**无偏**（Chen et al. 2023）。

## 注意 top_k / top_p 的处理

`top_p` 的 nucleus 截断必须在**归一化之后**做，且 `draft` 与 `verify` 两侧要用
**同一套**截断参数。如果草稿侧用了 top_p=0.9 而验证侧用了 0.95，无偏性就破了。
参数通过 `verify_and_sample` 统一传入，避免两侧不一致。