"""data_gen: 面向预训练语料的网页采集工具包。

模块划分：
    fetch        — HTTP 抓取层（重试、编码探测、限速、robots 遵守）
    extract      — HTML 正文提取（基于文本块密度打分，自适应各站点）
    sources      — 站点适配器配置（列表页规则、正文选择器、主题过滤词）
    crawl        — 采集编排（发现链接 -> 抓取 -> 提取 -> 主题过滤 -> 落盘）
    llm          — LLM 调用层（OpenAI 兼容，读 ~/.workbuddy/models.json）
    posting      — 打卡帖改写（LLM 改写 + 事实核查 + 语义筛选）
    build_corpus — 语料构建（去重、去噪、长度过滤，输出 {"text": ...}）
"""

import time

__all__ = [
    "fetch", "extract", "sources", "crawl",
    "llm", "posting", "build_corpus",
]


def _now() -> str:
    """统一的时间戳格式，避免各模块各写一套。"""
    return time.strftime("%Y-%m-%d %H:%M:%S")