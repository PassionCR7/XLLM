# data_gen —— 杭州「周末去哪玩」语料采集

给 `dataset/pretrain_t2t*.jsonl` 补充本地生活类中文语料。产出的 `{"text": ...}`
与现有预训练文件**完全同构**，可直接被 `dataset/lm_dataset.py` 读取。

## 快速开始

```bash
cd D:/workspace/xllm

# 看有哪些数据源
python -m data_gen.cli list

# 采集默认源（约 5~8 分钟）
python -m data_gen.cli crawl --max-docs 30

# 采集 + 构建语料，一步到位
python -m data_gen.cli all --max-docs 30

# 只看构建结果统计
python -m data_gen.cli stats
```

产出：

```
data_gen/out/
├── raw/                      原始文档（含 url，可复查来源）
│   ├── hangzhou_travel.jsonl
│   └── ...
├── corpus/corpus.jsonl       <- 直接喂给训练脚本
└── state.json                断点续爬记录
```

## 当前实测产出

```
python -m data_gen.cli all --max-docs 40 --interval 0.8 --sleep 0.2 --insecure
```

| 指标 | 数值 |
|---|---|
| 原始采集 | 26 篇 |
| 入语料 | **19 篇** |
| 中文总字数 | 21,828 |
| 平均长度 | 1,148 字（最短 273 / 中位 1,014 / 最长 3,517） |
| 脏数据 | 0（页脚/备案/版权信息全部清除） |

抽样标题，确认主题对口：

> 遛狗、喂羊、拍秋景 杭州这座矿山上的宠物公园藏不住了
> 天还没亮，杭州满觉陇已经开始打桂花了
> 西湖断桥边新浪漫打卡点 3米永生花"月老"限时展出
> 不拼厨艺拼"胡闹"！杭州龙坞这间厨房让年轻人抢着做饭
> 第十一届茶奥会开幕 四条漫游线路，带你玩遍茶山

格式与 `dataset/pretrain_t2t_mini.jsonl` 逐字段一致（均为 `{"text": ...}`），
已实测可被 `dataset/lm_dataset.py` 直接读取。

## 前情：为什么不用 requests / BeautifulSoup

本机 `pip` 源不可达（`No matching distribution found`），`.venv` 也已损坏
（`ModuleNotFoundError: No module named 'encodings'`）。因此**全部代码只用
Python 标准库**：`urllib` 抓取、`html.parser` 解析、`re` 清洗。

想加依赖时先修好虚拟环境：

```bash
rm -rf .venv && uv sync
```

## 数据源现状（均为实测结论，非推测）

| 渠道 | 结果 |
|---|---|
| bing / baidu / sogou / so.com | **不可用**，本机网络经代理，全部 502 |
| www.19lou.com 帖子页 | **不可用**，返回 1.3KB JS 跳转壳，静态抓不到正文 |
| hz.bendibao.com | **不可用**，SSL 握手超时 |
| mafengwo / travel.qunar.com | **不可用**，202 + 异步渲染 |
| fashion.hangzhou.com.cn | **可用，主力源**，杭州网-休闲旅游，实测 13 篇全相关 |
| jrsh.hangzhou.com.cn/node_5387 | **可用，补充源**，杭州网-今日生活-资讯，约 1/3 相关 |
| travel.sina.com.cn/zhejiang | **实测无效**，2 篇全是外地游记（扬州/赣州），保留作备用 |
| jrsh node_9760 / 9765 | **实测无效**，36 篇仅 2 篇相关，全是饮食/居家科普 |

关键限制：**搜索引擎走不通，所以做不到"按关键词全网搜"**，只能靠站点栏目
列表页翻页发现文章。要扩源，就在 `sources.py` 里加 `Source`。

默认只启用 `hangzhou_travel` + `jrsh_news` 两个源——这是按实测产出率定的，
不是拍脑袋。其余源写进配置但默认关闭，需要时用 `--source xxx` 单独开启。

## 架构

```
fetch.py          HTTP 层：编码探测、指数退避重试、按域名限速、robots 检查
extract.py        正文提取：自建轻量 DOM + 文本块密度打分（不依赖固定选择器）
sources.py        站点配置 + 主题词表
crawl.py          编排：发现 -> 抓取 -> 提取 -> 主题过滤 -> 落盘
build_corpus.py   去重（SimHash）+ 重组 + 二次清洗 -> corpus.jsonl
cli.py            命令行
test_data_gen.py  回归测试（11 组，覆盖实测踩过的坑）
```

### 三个必须知道的实现细节（都是实测踩出来的）

**1. 编码探测是数据质量的第一道关。** 杭州网主站 `hznews` 是 UTF-8，但
`19lou` 等老站是 GBK。探测顺序：响应头 `charset` → HTML `meta` → 回退
`gb18030`。直接用 UTF-8 解 GBK 会产出满屏乱码却"看起来抓成功了"。

**2. 脏 HTML 会吃掉整页正文。** 实测 `jrsh` 页面存在未闭合的 `<form>` 和
`<input>`。若用 skip 计数器，遇到缺失的结束标签后会永久卡在"跳过中"状态，
把整页正文丢光（表现为 `cjk=0`）。修法两条：
- `input`/`textarea` 等空元素移出 skip 逻辑（它们常被写成未闭合形式）
- 遇到 `div`/`article` 等结构性容器时，清空挂起的 skip 栈

**3. 页脚备案信息会被密度打分误收。** 杭州网系文章末尾稳定出现
「增值电信业务经营许可证：浙B2-...」，长度够、链接少，密度打分认它是正文。
`build_corpus.FOOTER_RE` 里按关键词行级剔除。

### 主题过滤：只看「标题 + 导语前 80 字」

```
必须项（本地性）：标题或导语前 80 字出现杭州地名   -> 否则判 0
意图项（主题性）：同一窗口内出现游玩意图词，或具体游玩对象词
泛词：「公园/景点/路线」只能加成，不能顶替意图项
```

为什么窗口要收到 80 字：曾收录到「冬日温婉水墨扬州」「玩转赣州」两篇，
它们正文某处顺带提到「杭州」，用「标题+正文前 200 字」就漏过去了。本地
游玩内容的导语开头必定交代地点，收到 80 字即可分开。

另一个坑是**歧义地名**：`CITY_TERMS` 里原本有「宋城」，而赣州自称
「江南宋城」，直接误命中。这类词必须移除，只留专指杭州的地名。

`python -m data_gen.test_data_gen` 里有 11 组回归用例，把上述坑都固化了，
改代码后先跑它。

## 调参

```bash
# 放宽主题阈值，收更多（噪声也会变多）
python -m data_gen.cli crawl --min-score 0.35

# 只要长文，适合学行文风格
python -m data_gen.cli crawl --min-cjk 500 --max-docs 50
```

礼貌抓取默认开启：遵守 robots、同域名 1.2s 间隔、最多重试 3 次。
**别关掉 `--ignore-robots`，也别把 `--interval` 调到 0.3 以下。**

## 已知局限

- 语料规模小：单轮全源约 40~60 篇、2 万字级别，只够做**领域微调补充**，
  撑不起从零预训练。
- 时间偏移：新浪源是 2022-2023 年内容；杭州网源是近期内容但随日期滚动，
  重跑结果会变。
- 无摘要/问答对：目前只产出纯文本。若要 SFT 数据，需在 `build_corpus.py`
  里加模板生成（把正文首段当 answer、标题当 query）。
- 图片说明残留：`_normalize_body` 已过滤大部分「xxx 摄」短行，但长图注仍可能
  混入，量少可接受。