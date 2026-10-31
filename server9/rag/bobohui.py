# -*- coding: utf-8 -*-
"""第十一届博博会。数字人此刻就在现场的宽创展厅里。"""
from __future__ import annotations
import re

_ASK = re.compile(r"博博会|中国博物馆及相关产品与技术博览会")

TEXT = (
    "第十一届中国博物馆及相关产品与技术博览会，简称博博会，定于二零二六年十一月十三日至十六日"
    "在海南国际会展中心举办，主题是「高质量发展的博物馆：科技赋能 联结未来」。"
    "二零二六年十月二十日上午，新闻发布会在海口举行。中国博物馆协会理事长关强、秘书长李金光，"
    "海南省旅游和文化广电体育厅副厅长李海钢，以及海口市人民政府副市长何霁锋出席，"
    "公布了筹备进展、展览布局、特色活动和公众参观安排。"
    "这是十五五开局之年的文博行业盛会，集中展示我国博物馆事业建设成果，推动文博科技深度融合，"
    "深化中外文明交流互鉴。"
    "我此刻就在这场博博会现场，人就在宽创国际的展厅里。"
    "宽创展厅用科技把博物馆和观众联结起来。"
)

WHEN_WHERE = (
    "我就在博博会现场的宽创展厅里。"
    "第十一届博博会定于二零二六年十一月十三日至十六日，在海南国际会展中心举办。"
)
NO_OLD_TICKET = "原先的门票价格我不介绍。"
PRESENCE_RULE = (
    "你人已经在博博会现场，就在宽创国际的展厅里。"
    "介绍时说我就在这里，不要说在那里，不要说我听说。"
)
HOOK_TOPIC = {
    "weather": "天气",
    "news": "新闻",
    "exhibit": "我的作品",
    "knowledge": "我的作品",
    "technique": "我的作品",
    "opening": "我的作品",
    "interactive": "我的作品",
    "deep": "我的作品",
    "person": "我的经历",
    "visit": "展会安排",
    "chitchat": "我的介绍",
}


def hall_hook(topic: str) -> str:
    """了解完刚讲的内容，再问还想了解博博会或宽创展台的什么。"""
    topic = re.sub(r"\s+", "", topic or "") or "这些"
    return "了解完%s，您还想了解博博会或宽创展台的哪些内容呢？" % topic


def is_bobohui_question(question: str) -> bool:
    return bool(_ASK.search(question or ""))


def bobohui_hit(question: str):
    """问到本届博博会时，给检索一条可直接引用的资料。"""
    if not is_bobohui_question(question):
        return None
    return {
        "id": "bobohui_2026_11",
        "text": TEXT,
        "source": "第十一届博博会新闻发布会",
        "category": "exhibition_bobohui",
        "title": "第十一届博博会",
        "score": 1.0,
        "recall": "hall",
    }


def merge_bobohui(question: str, hits, threshold: float):
    expo = bobohui_hit(question)
    if not expo:
        return list(hits or [])
    hits = list(hits or [])
    if not hits or float(hits[0].get("score") or 0) < float(threshold):
        return [expo]
    rest = [h for h in hits if h.get("id") != expo["id"]]
    return [expo] + rest
