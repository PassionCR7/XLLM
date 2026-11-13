# 黑盒/白盒蒸馏推理怎么跑

两种蒸馏的学生都是同一套 xllm 权重（25.83M），推理脚本各自独立。

---

## 0. 环境（唯一前提）

用 Anaconda 的 py11 环境，**不要用系统 python**（缺 torch）：

```bash
D:/Anaconda2025/envs/py11/python.exe
```

显存需求约 1GB，8GB 卡足够。

---

## 1. 黑盒蒸馏推理

```bash
cd D:/workspace/xllm

# 单条提问（推荐默认采样）
D:/Anaconda2025/envs/py11/python.exe distillation_blackbox/infer.py --prompt "你是谁？"

# 贪心解码 + 抑制复读（结果可复现）
D:/Anaconda2025/envs/py11/python.exe distillation_blackbox/infer.py \
    --prompt "你是谁？" --greedy --repetition_penalty 1.3

# 与基线对照，验证蒸馏确实改变了行为
D:/Anaconda2025/envs/py11/python.exe distillation_blackbox/infer.py \
    --weight full_sft --prompt "你是谁？" --greedy

# 交互式多轮（exit 退出 / clear 清空历史）
D:/Anaconda2025/envs/py11/python.exe distillation_blackbox/infer.py --chat

# 批量：每行一条提问
D:/Anaconda2025/envs/py11/python.exe distillation_blackbox/infer.py \
    --input questions.txt --output answers.jsonl --batch_size 8

# 查看每个 token 的概率
D:/Anaconda2025/envs/py11/python.exe distillation_blackbox/infer.py \
    --prompt "你好" --greedy --max_new_tokens 16 --logprobs
```

默认权重：`distillation_blackbox/out/distill_blackbox_512.pth`

---

## 2. 白盒蒸馏推理

```bash
cd D:/workspace/xllm

# 单条提问
D:/Anaconda2025/envs/py11/python.exe distillation_whitebox/infer.py --prompt "你是谁？"

# 贪心解码
D:/Anaconda2025/envs/py11/python.exe distillation_whitebox/infer.py \
    --prompt "你是谁？" --greedy

# 交互式多轮
D:/Anaconda2025/envs/py11/python.exe distillation_whitebox/infer.py --chat

# 批量
D:/Anaconda2025/envs/py11/python.exe distillation_whitebox/infer.py \
    --input questions.txt --output answers.jsonl --batch_size 4

# 与教师 Qwen3-0.6B 并排对比（额外占约 1.5GB 显存）
D:/Anaconda2025/envs/py11/python.exe distillation_whitebox/infer.py \
    --prompt "你是谁？" --compare_teacher
```

默认权重：`distillation_whitebox/out/distill_whitebox_512.pth`

`--compare_teacher` 是白盒独有的，黑盒教师是闭源 API，需联网和密钥，不适合实时对比。

---

## 3. 参数速查

两个脚本参数基本一致（白盒多一个 `--compare_teacher`）：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--weight` | 权重名，不带 `.pth` | 各自蒸馏权重 |
| `--prompt` | 单条提问 | — |
| `--chat` | 交互多轮 | 关 |
| `--input` / `--output` | 批量输入 / jsonl 输出 | — |
| `--batch_size` | 批量并发数 | 4 |
| `--max_new_tokens` | 最大生成长度 | 256 |
| `--temperature` | 采样温度 | 0.7 |
| `--top_p` | 核采样 | 0.9 |
| `--repetition_penalty` | 重复惩罚 | 1.05 |
| `--no_repeat_ngram` | 禁止重复的 n-gram 长度 | 0（关） |
| `--greedy` | 贪心解码，结果可复现 | 关 |
| `--system` | 系统提示词 | 无 |
| `--show_think` | 保留思维链 | 关 |
| `--logprobs` | 逐 token 输出 logprob | 关 |
| `--device` | cuda / cpu | 自动 |

---

## 4. 已知坑

### 黑盒复读

黑盒学生只在 300 条教师数据上蒸馏，贪心解码容易整段重复。实测：

| 设置 | 效果 |
|------|------|
| 默认采样 | 不整段复读，偶尔跑题 |
| `--repetition_penalty 1.15` | **仍复读，无效** |
| `--repetition_penalty 1.3` | 消除复读，但语气浮夸带 emoji |
| `--no_repeat_ngram 3` | 消除逐字重复，但会编造事实 |

`1.15` 是网上的通用推荐值，在你这模型上不起作用，要 1.3。

### 批量模式参数曾失效（已修）

`batch_infer()` 原来没有 `repetition_penalty` / `no_repeat_ngram` 参数，
批量运行时这两个参数被**静默丢弃**，导致批量结果和单条结果不一致。
现已补上参数并下发到 `generate()`，批量与单条行为一致。

### 权重名写错不会报错

`--weight` 传了不存在的名字会直接抛 `FileNotFoundError`，这个会报。
但早期版本解析出路径后没写回 config，导致传什么都加载默认权重——
现已修复，传 `--weight full_sft` 能真正切到基线。

### 白盒显存碎片

白盒评估/推理偶发 OOM 但显存其实够用，加环境变量：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 5. 预期输出

黑盒（`--greedy --repetition_penalty 1.3`）：

```
我是一个由中国的个人开发者独立开发的智能助手，专注于提供高效、精准的服务…
```

白盒（`--greedy`）：

```
我是由jingyaogong创建的高效小参数AI模型，专注于提供精准、快速的信息和帮助。
```

白盒输出带 `jingyaogong`（xllm 作者）是 SFT 阶段继承的自我认知，正常现象。