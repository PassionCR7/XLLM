# mtp —— C：MTP 联合预训练

对齐 DeepSeek-V3 / V4 的 Multi-Token Prediction。在主干每层之后挂 D 个轻量
MTP 模块，训练时同时预测 t+1..t+D+1 多个位置。模块与主干**共享 embedding 与
lm_head**，额外参数只为多出的 Transformer 块。

## 文件

| 文件 | 作用 |
|---|---|
| `mtp_module.py` | `MTPHead` + `XLLMWithMTP`（模型定义与联合损失） |
| `mtp_engine.py` | 引擎：张量级接口，吃 input_ids、吐 token id |
| `infer.py` | **推理入口：tokenizer + 命令行 + 加速比基准，可直接出文本** |
| `mtp_pretrain.py` | 联合预训练脚本 |
| `_mtp_check.py` | 六项正确性验证（偏移 / 梯度回流 / ckpt 拆分等） |
| `_mtp_engine_check.py` | 引擎验收（贪心等价 / 采样无偏 / 接受率 / 统计口径） |
| `MT_PRETRAIN_GUIDE.md` | **上手手册：怎么跑、参数怎么定、结果怎么读** |

**先看 [`MT_PRETRAIN_GUIDE.md`](./MT_PRETRAIN_GUIDE.md)**，那份是操作手册，
本文只讲设计定位。

## 快速推理

```bash
PY=/d/Anaconda2025/envs/py11/python.exe

# 最简：自动找 out/ 下的权重
$PY model/draft/mtp/infer.py --prompt "人工智能是"

# 测真实加速比（推荐；会同时跑主干基线并核对等价性）
$PY model/draft/mtp/infer.py --prompt "人工智能是" --max-new 64 \
    --benchmark --bench-ignore-eos

# 带训过的 MTP 头
$PY model/draft/mtp/infer.py --prompt "人工智能是" \
    --weight pretrain --mtp-weight-file out/pretrain_mtp_mtp_512.pth \
    --max-new 64 --benchmark --bench-ignore-eos
```

`infer.py` 做三件 `mtp_engine.py` 做不到的事：

1. **tokenizer 编解码**——引擎只认 token id，出不了文字。
2. **`--benchmark` 真实加速比**——同 prompt 跑主干直接贪心作基准，算出
   `ms/token` 比值。这个数小于 1 就说明 MTP 白训了，哪怕接受率好看。
3. **加载前校验主干超参**——`load_state_dict(strict=False)` 会静默跳过 shape
   不匹配的权重，GQA 头数配错时程序照跑但输出全是垃圾。`infer.py` 会用
   MTP 头的 `k_proj` / `proj` 形状反推并直接报错。

## 与 dspark(B) 的本质区别

```
B: target 冻结，梯度不回传，草稿只能**逼近**已有的 p。
C: 预训练期梯度穿过主干，草稿与主干**共同塑造** p。
```

C 是真正的联合训练，而且密集监督信号（一次预测多个未来 token）会让主干本身
变好——这是 B 的事后训练拿不到的。代价是必须在预训练阶段就规划好，不能事后补。

## 草稿宽度的约定（最容易搞错的地方）

主头给出**第 1 个**草稿（就是主干自己的 argmax，零额外成本），D 个模块负责
后续 D 个。所以 D 个模块实际产出 **1+D** 个草稿。

读接受率时务必记住这一点：

| D=1 时每轮平均接受 | 含义 |
|---|---|
| ≈ 1.00 | MTP 头完全没学到东西（随机初始化就是这个值） |
| 1.3 ~ 1.6 | 学到了，但还弱 |
| > 1.8 | 训练充分，加速明显 |

D=1 时"接受率 0.5、每轮 2.0 token"是**未训练基线**，不是好结果。
`acc_per_iter` 全是 1 正是"MTP 头全错"的指纹。

## 索引约定（实现时容易错）

`h0[t]` 预测 `labels[t+1]`。所以模块 k 用 `input_ids[t+k]` 作条件、预测
`labels[t+k+1]`，长度统一取 `T-D-1`。

`_mtp_check.py` 的 [B] 项专门验证这个偏移——三种候选切片（`labels[t+k+1]` /
`labels[t+k]` / `labels[t+1]`）的 CE 只差 0.01 量级，**只有正确切片能精确复现
训练时的 reported 值**（误差 <1e-5）。即使用阈值判断都不可靠，必须精确匹配。

## 跑

```bash
cd /d/workspace/xllm
PY=/d/Anaconda2025/envs/py11/python.exe

$PY model/draft/mtp/mtp_pretrain.py --smoke              # 链路自检
$PY model/draft/mtp/_mtp_check.py                        # 六项正确性
$PY model/draft/mtp/_mtp_engine_check.py --quick --modules 1   # 引擎验收
```

完整训练与验收流程见 [`MT_PRETRAIN_GUIDE.md`](./MT_PRETRAIN_GUIDE.md)。