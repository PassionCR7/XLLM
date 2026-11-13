# MTP 联合预训练 —— 上手手册

这份文档只讲**怎么跑**，设计原理见上一级目录的 `../README.md` 第三部分。

本目录（`model/draft/mtp/`）内容：

| 文件 | 作用 |
|---|---|
| `mtp_module.py` | `MTPHead` + `XLLMWithMTP`（模型定义） |
| `mtp_engine.py` | 推理引擎，测接受率 |
| `mtp_pretrain.py` | 联合预训练脚本 |
| `_mtp_check.py` | 六项正确性验证 |
| `_mtp_engine_check.py` | 引擎验收 |
| `MT_PRETRAIN_GUIDE.md` | 本文档 |

---

## 0. 环境

```bash
cd /d/workspace/xllm
PY=/d/Anaconda2025/envs/py11/python.exe   # torch 2.5.1
```

注意：项目自带的 `.venv` 已损坏（`ModuleNotFoundError: No module named 'encodings'`），
请始终用上面这个解释器。

> Windows 下如果 `python ... > x.log` 报 `ENOSPC`，是 C 盘临时目录满了。
> 本手册的命令都把日志写到 `_logs/`（在 D 盘）。

---

## 1. 五分钟跑通（不构成训练，只验证链路）

```bash
$PY model/draft/mtp/mtp_pretrain.py --smoke
```

预期输出：

```
主干参数 1.57M | MTP 参数 0.22M | D=1 λ=0.3
训练计划: 1 epoch × 40 step = 40 step (数据集 1270238 条, batch 4)
Epoch:[1/1](40/40) loss:11.22 main:8.63 mtp:8.66 lr:0.0000500000
[save] out/_smoke/smoke_mtp_128.pth
[save] out/_smoke/smoke_mtp_mtp_128.pth
[save] out/_smoke/smoke_mtp_train_128.pth
DONE
```

`ln(6400)=8.76`，所以 loss 停在 8.6 附近说明只学到了很少——**这是正常的**，
40 步不可能收敛。smoke 只证明"不报错、三个文件都能存"。

冒烟产物在 `out/_smoke/`，不会污染正式训练的 checkpoint。

---

## 2. 在你的 pretrain_512 基础上继续联合训练

这是最常用的路径。主干已有权重，加上 MTP 头一起训练：

```bash
mkdir -p _logs
$PY model/draft/mtp/mtp_pretrain.py \
  --from_weight pretrain \
  --save_weight pretrain_mtp \
  --hidden_size 512 --num_hidden_layers 8 \
  --max_seq_len 512 --batch_size 8 --accumulation_steps 8 \
  --mtp_modules 1 --mtp_weight 0.3 \
  --learning_rate 1e-4 \
  --epochs 1 --max_steps 2000 --plan_total_steps 2000 \
  --log_interval 20 --save_interval 500 \
  > _logs/mtp_train.log 2>&1
```

### 参数怎么定

| 参数 | 建议 | 说明 |
|---|---|---|
| `--mtp_modules` | 1 | V3/V4 官方就是 1。设 3 只增加密集监督，加速比几乎不变 |
| `--mtp_weight` | 0.1 ~ 0.3 | λ。太大伤主干，太小草稿学不好。先 0.3 |
| `--learning_rate` | 1e-4 | **比从头训练小**。主干已经是训练好的，用小学习率微调 |
| `--max_steps` | 2000 起 | 单个 epoch 的步数上限 |
| `--plan_total_steps` | 与 max_steps 一致 | 学习率余弦周期的分母，见下方"两个容易踩的坑" |

显存不够就降 `--batch_size`、升 `--accumulation_steps`，有效 batch 保持
`batch_size × accumulation_steps` 不变即可。

---

## 3. 两个容易踩的坑（我都已经处理，但你得知道为什么）

### 3.1 `--max_steps` 和学习率的耦合

数据集有 127 万条，`len(loader)` 是 31 万步。如果直接拿它当学习率余弦周期的
分母，`get_lr` 在整个训练过程中几乎不动（cos 的斜率接近 0），等于**没有调度**。

所以现在语义是：

- `--max_steps`：本 epoch 实际跑多少步
- `--plan_total_steps`：整个训练计划的步数 = 余弦周期

**续训时必须显式传 `--plan_total_steps` 且与原计划一致**，否则学习率会跳变。

### 3.2 三个 checkpoint 文件

```
out/pretrain_mtp_512.pth         纯主干权重，键名与 XLLMForCausalLM 一致
out/pretrain_mtp_mtp_512.pth     MTP 头 + 自描述 cfg
out/pretrain_mtp_train_512.pth   optimizer/scaler/进度，仅续训用
```

主干单独一个文件，是为了让 `eval.py` 的 `strict=True` 能直接加载——不会撞到
MTP 或 optimizer 的意外键。已实测 `missing=0 unexpected=0`。

---

## 4. 续训

```bash
$PY model/draft/mtp/mtp_pretrain.py \
  --from_weight pretrain --from_resume 1 \
  --save_weight pretrain_mtp \
  --hidden_size 512 --num_hidden_layers 8 \
  --epochs 2 --max_steps 2000 --plan_total_steps 4000 \
  > _logs/mtp_resume.log 2>&1
```

`--from_resume 1` 会做四件事：恢复主干、恢复 MTP 头、恢复 AdamW 状态、
恢复 epoch/step 进度。日志里能看到：

```
[resume] 主干从 out/pretrain_mtp_512.pth 恢复
[resume] epoch=0 step=2000 global_step=2000/4000
```

如果只想加载主干继续训（不恢复进度），用 `--from_weight` 并保持
`--from_resume 0`。

---

## 5. 训完必须验收，否则等于没训

loss 下降**不代表**草稿有用。唯一有意义的指标是接受率，必须用引擎测：

```bash
$PY model/draft/mtp/_mtp_engine_check.py \
  --weight pretrain_mtp --hidden 512 --layers 8 \
  --mtp out/pretrain_mtp_mtp_512.pth \
  --modules 1 > _logs/mtp_eval.log 2>&1
```

看 `[C] 接受率` 这一段：

```
接受率=0.5000  每轮平均接受=1.00
每轮平均产出 token=2.00  (上限 1+D=2)
```

### 怎么读这些数字（重要）

MTP 的草稿结构是「主头 1 个 + D 个模块各 1 个」，所以：

- **第 1 个草稿来自主头**，就是主干自己的 argmax → **必然 100% 命中**
- 后续 D 个来自 MTP 头，这才是真正考验训练效果的部分

因此 `D=1` 时，**接受率 0.5000 / 每轮 2.00 token 是"未训练"的基线**，
不是好结果。有意义的判据是：

| D=1 时每轮平均接受 | 含义 |
|---|---|
| ≈ 1.00 | MTP 头完全没学到东西（随机初始化就是这个值） |
| 1.3 ~ 1.6 | 学到了，但还弱 |
| > 1.8 | 训练充分，加速明显 |

未训练的实测结果（512 权重 + 随机 MTP 头）：

```
轮数=12  产出=24  接受=12
接受率=0.5000  每轮平均接受=1.00
acc_per_iter=[1,1,1,1,1,1,1,1,1,1,1,1]   ← 稳定为 1，第 2 个从未命中
草稿耗时占比=0.107   27.89 ms/token
```

`acc_per_iter` 全是 1 这个模式，正是"主头命中、MTP 头全错"的指纹。

---

## 6. 引擎正确性（改代码后必跑）

```bash
$PY model/draft/mtp/_mtp_engine_check.py --quick --modules 1
```

四项检查，其中 A 和 D 是**一票否决**：

- **A. 贪心等价性**：MTP 引擎的输出必须与主干直接贪心**逐 token 相同**。
  这一项测的是正确性，与训练质量无关——即使 MTP 全猜错，验证环节也保证输出正确。
- **B. 采样无偏性**：引擎采样与直接采样的偏差应在统计噪声量级。
- **C. 接受率**：观察训练效果。
- **D. 统计口径自检**：`avg_tokens_per_iter ∈ [1, 1+D]`。

本机实测（hidden=64 随机小模型 / 真实 512 权重，D=1 和 D=2 都跑过）：

```
[A] prompt0..5 一致=True  -> 通过
[B] 直接采样 vs 理论 TV = 0.0914
    引擎采样 vs 理论 TV = 0.1037   比值 1.13   -> 通过
[D] -> 通过
```

---

## 7. 一个曾经的真实 bug（供参考）

`mtp_engine.py` 最初为了省一次前向，在"草稿未被全部接受"时复用了
`verify_hidden[:, m]` 作为下一轮的草稿输入。这在贪心等价性测试里立刻暴露
（6 个 prompt 里 4 个失败，分歧点在第 2~7 个 token）。

根因：`next_token` 是在位置 `ctx_len+m` 上**被拒绝**后采出的压哨 token，
它与 `draft_ids[m]` 不是同一个 token，所以那个位置的 hidden 是"错误 token 的
隐状态"，不能拿来用。

正确做法是「保留已接受前缀 + 对 `next_token` 单独补一次前向」，位置严格连续。
代价是每轮多一次单 token 前向（约占总时间 10%），但这是**必须付的**。

如果你以后想优化这里，立刻跑 `_mtp_engine_check.py --quick`，A 项会直接告诉你
有没有改坏。

---

## 8. 完整流程清单

```bash
# 1. 链路自检
$PY model/draft/mtp/mtp_pretrain.py --smoke

# 2. 引擎正确性（首次或改代码后）
$PY model/draft/mtp/_mtp_engine_check.py --quick --modules 1

# 3. 真实训练
$PY model/draft/mtp/mtp_pretrain.py --from_weight pretrain --save_weight pretrain_mtp \
  --hidden_size 512 --num_hidden_layers 8 --max_seq_len 512 \
  --batch_size 8 --accumulation_steps 8 --mtp_modules 1 --mtp_weight 0.3 \
  --learning_rate 1e-4 --epochs 1 --max_steps 2000 --plan_total_steps 2000 \
  > _logs/mtp_train.log 2>&1

# 4. 验收（看每轮平均接受是否 > 1.0）
$PY model/draft/mtp/_mtp_engine_check.py --weight pretrain_mtp --hidden 512 --layers 8 \
  --mtp out/pretrain_mtp_mtp_512.pth --modules 1 > _logs/mtp_eval.log 2>&1

# 5. 实际推理（出文本 + 测真实加速比）
$PY model/draft/mtp/infer.py --prompt "人工智能是" \
  --weight pretrain_mtp --max-new 64 --benchmark --bench-ignore-eos

# 6. 若继续训
$PY model/draft/mtp/mtp_pretrain.py --from_resume 1 --save_weight pretrain_mtp \
  --hidden_size 512 --num_hidden_layers 8 --epochs 2 \
  --max_steps 2000 --plan_total_steps 4000 > _logs/mtp_resume.log 2>&1
```

---

## 9. 推理与加速比

```bash
$PY model/draft/mtp/infer.py --prompt "人工智能是" --max-new 64 \
    --benchmark --bench-ignore-eos --bench-repeat 3
```

输出解读：

```
每轮平均接受      : 1.95 (宽度 1+D=2)          ← 训练效果
每轮平均产出      : 2.95 (上限 2+D=3)
acc_per_iter      : [2, 2, 2, ..., 1]          ← 几乎每轮都吃满
基线     : 64 token, 0.619s, 9.68 ms/token
MTP      : 64 token, 0.762s, 11.90 ms/token
有效加速比: 0.81x  (变慢)
等价性检查: 逐 token 一致 ✓ (比对前 64 个 token)
```

**关键教训：接受率高 ≠ 加速。** 上面这组实测里每轮接受 1.95（几乎满编），
但加速比 0.81x 仍是变慢。原因是每轮有三项固定开销：

```
每轮耗时 = D × MTP前向 + 1 × 主干验证前向 + 1 × 主干补位前向
```

补位前向来自压哨 token 必须单独过一次主干（不能复用验证阶段的 hidden，
否则位置错位）。D=1 时理论最优加速是 `(1+1)/1 = 2x`，但实测只到
`(2+1)/2 = 1.5x` 上限，再被草稿开销吃掉就成了 0.81x。

**`--bench-ignore-eos` 必须加。** 否则两侧可能因 EOS 提前退出而生成长度
不同，比总耗时是错的。加了这个参数两边都强制生成满 `--max-new`。

**`--bench-repeat` 取 3 次平均**，单次测量在小模型上抖动很大。