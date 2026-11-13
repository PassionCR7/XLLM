"""data_gen: 面向预训练语料的网页采集工具包。

模块划分：
    fetch      — HTTP 抓取层（重试、编码探测、限速、robots 遵守）
    extract    — HTML 正文提取（基于文本块密度打分，自适应各站点）
    sources    — 站点适配器配置（列表页规则、正文选择器、主题过滤词）
    crawl      — 采集编排（发现链接 -> 抓取 -> 提取 -> 主题过滤 -> 落盘）
    build_corpus — 语料构建（去重、去噪、长度过滤、切分为 {"text": ...}）
"""

__all__ = ["fetch", "extract", "sources", "crawl", "build_corpus"]