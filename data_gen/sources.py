"""站点适配器配置。

每个 Source 描述一个可抓取的站点/频道：
    list_urls      列表页，用于发现详情页链接
    link_pattern   详情页 URL 的正则特征，用来从列表页筛掉导航链接
    paginate       分页 URL 模板，{page} 从 2 开始
    max_pages      最多翻页数

**这些配置全部是在本机实测通过后才写进来的**，不是猜的。已被排除的渠道：
  - bing / baidu / sogou / so.com 搜索：本机网络经代理，全部返回 502，不可用
      -> 因此无法用「关键词搜索」发现页面，只能靠站点栏目列表
  - www.19lou.com 帖子页：返回 1.3KB 的 JS 跳转壳，静态抓不到正文
  - hz.bendibao.com：SSL 握手超时，不可达
  - www.mafengwo.cn / travel.qunar.com：返回 202/空壳，正文异步加载
  - hznews 城市/经济/科教频道：实测全为政务/民生新闻，主题通过率 0/12，
      保留配置但不作为主力源

主力源（实测有稳定相关产出）：
  fashion.hangzhou.com.cn  杭州网-休闲旅游   <- 最对口
  jrsh.hangzhou.com.cn     杭州网-今日生活   <- 饮食/居家等生活类
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Source:
    name: str
    list_urls: List[str]
    link_pattern: str
    paginate: Optional[str] = None
    max_pages: int = 1
    note: str = ""

    def is_article(self, url: str) -> bool:
        return bool(re.search(self.link_pattern, url))

    def page_urls(self) -> List[str]:
        urls = list(self.list_urls)
        if self.paginate:
            for p in range(2, self.max_pages + 1):
                urls.append(self.paginate.format(page=p))
        return urls


SOURCES: List[Source] = [
    # ---------------- 主力源 ----------------
    Source(
        name="hangzhou_travel",
        list_urls=["https://fashion.hangzhou.com.cn/index.htm"],
        link_pattern=r"fashion\.hangzhou\.com\.cn/content/\d{4}-\d{2}/\d{2}/content_\d+\.html?$",
        max_pages=1,
        note="杭州网-休闲旅游，最对口的游玩内容源",
    ),
    Source(
        name="jrsh_food",
        list_urls=["https://jrsh.hangzhou.com.cn/node_9760.html"],
        link_pattern=r"jrsh\.hangzhou\.com\.cn/content/\d{4}-\d{2}/\d{2}/content_\d+\.html?$",
        paginate="https://jrsh.hangzhou.com.cn/node_9760_{page}.html",
        max_pages=3,
        note="杭州网-今日生活-饮食",
    ),
    Source(
        name="jrsh_home",
        list_urls=["https://jrsh.hangzhou.com.cn/node_9765.html"],
        link_pattern=r"jrsh\.hangzhou\.com\.cn/content/\d{4}-\d{2}/\d{2}/content_\d+\.html?$",
        paginate="https://jrsh.hangzhou.com.cn/node_9765_{page}.html",
        max_pages=2,
        note="杭州网-今日生活-居家",
    ),
    Source(
        name="jrsh_news",
        list_urls=["https://jrsh.hangzhou.com.cn/node_5387.html"],
        link_pattern=r"jrsh\.hangzhou\.com\.cn/content/\d{4}-\d{2}/\d{2}/content_\d+\.html?$",
        paginate="https://jrsh.hangzhou.com.cn/node_5387_{page}.html",
        max_pages=2,
        note="杭州网-今日生活-资讯（泛生活，靠主题过滤筛选）",
    ),
    # ---------------- 补充源：游记长文 ----------------
    Source(
        name="sina_travel",
        list_urls=["https://travel.sina.com.cn/zhejiang/"],
        link_pattern=r"travel\.sina\.com\.cn/(?:wbarticle|domestic|outbound)/\d{4}-\d{2}-\d{2}/detail-\w+\.shtml$",
        max_pages=1,
        note="新浪旅游-浙江，游记长文，正文密度高",
    ),
    # ---------------- 备用源：政务新闻（通过率低，默认不启用） ----------------
    Source(
        name="hznews_chengshi",
        list_urls=["https://hznews.hangzhou.com.cn/chengshi/"],
        link_pattern=r"/chengshi/content/\d{4}-\d{2}/\d{2}/content_\d+\.html?$",
        paginate="https://hznews.hangzhou.com.cn/chengshi/index_{page}.htm",
        max_pages=2,
        note="杭州网-城市频道（政务/民生为主，主题通过率低）",
    ),
]

# 默认启用的源。
# 取舍依据是本机实测产出率，不是主观判断：
#   hangzhou_travel  13/13 通过，主力源
#   jrsh_news         1/3 通过，补充
# 实测被剔除的源：
#   sina_travel  2/2 全是外地游记（扬州/赣州），本地性不满足，保留作备用
#   jrsh_food    18 篇仅 1 篇相关，全是饮食科普，与「去哪玩」无关
#   jrsh_home    18 篇仅 1 篇相关，全是居家科普
DEFAULT_SOURCES = [
    "hangzhou_travel", "jrsh_news",
]

# ---------------------------------------------------------------- 主题过滤
# 目标主题：杭州本地 + 周末休闲游玩。
# 注意：这里的词用于判断「文章是否讲杭州」，必须选**专指杭州**的地名。
# 「宋城」已移除——赣州自称「江南宋城」，会误命中；杭州宋城用「杭州宋城」
# 这种组合词表达更安全，「杭州」本身已能覆盖。
CITY_TERMS = [
    "杭州", "西湖", "余杭", "萧山", "临平", "钱塘", "富阳", "临安", "桐庐",
    "建德", "淳安", "千岛湖", "滨江", "拱墅", "上城", "下城", "良渚",
    "西溪", "湘湖", "运河", "钱江", "之江", "武林", "河坊街",
    "龙井", "九溪", "灵隐", "径山", "塘栖", "超山", "大明山", "莫干山",
    "淳杨线", "富春江", "千岛", "天目山", "浙大", "南山路", "北山街",
]

WEEKEND_TERMS = [
    "周末", "去哪", "遛娃", "打卡", "露营", "徒步", "赏花", "踏青", "秋游",
    "春游", "一日游", "二日游", "citywalk", "city walk", "逛吃", "周边游",
    "自驾", "骑行", "野餐", "采摘", "市集", "夜市", "看展", "亲子",
    "郊游", "出游", "游园", "游玩", "玩水", "赏秋", "赏桂", "赏梅",
]

PLAY_TERMS = [
    "景点", "景区", "公园", "乐园", "博物馆", "美术馆", "展览", "演出",
    "剧院", "美食", "餐厅", "小吃", "路线", "攻略", "推荐", "门票",
    "风景", "湿地", "绿道", "古镇", "书店", "咖啡馆", "民宿", "温泉",
    "芦苇", "芦花", "桂花", "荷花", "梅花", "枫叶", "银杏",
    # 本地生活/游玩活动类高频词（实测补入，此前缺失导致正常游玩内容被误杀）
    "展陈", "展会", "市集", "集市", "活动", "文旅", "文创", "艺术",
    "音乐节", "演唱会", "庙会", "灯会", "赛事", "开幕", "启幕",
    "街区", "老街", "夜游", "游船", "观鸟", "采摘园", "农庄",
    "乐园", "水乐园", "亲子游", "研学", "逛",
]

# 明显不属于游玩语料的负面词，命中即降权
NEGATIVE_TERMS = [
    "房价", "楼市", "二手房", "招聘", "股价", "涨停", "基金", "疫苗",
    "诈骗", "判决", "事故", "车祸", "火灾", "纪委", "通报", "问责",
    "退费", "维权", "投诉", "物业费", "社保", "公积金", "限行",
]

__all__ = [
    "Source", "SOURCES", "DEFAULT_SOURCES", "CITY_TERMS", "WEEKEND_TERMS",
    "PLAY_TERMS", "NEGATIVE_TERMS",
]