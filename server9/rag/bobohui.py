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
    "你人就在博博会现场的宽创展厅里，这是事实，但不要主动报位置。"
    "不要说在那里，不要说我听说。"
    "只有观众问你在哪里、展厅在哪、人在不在现场时，才说我就在这里。"
    "讲作品、经历、器物或其他问题时，开口就是答案本身，不要先说我在这里。\n"
)
_WHERE = re.compile(
    r"(?:你|人|咱们|我们).{0,4}在哪|"
    r"展[厅台].{0,6}在哪|"
    r"在哪里举办|哪里举办|哪个馆|什么地方办|"
    r"人在不在|你在不在|是不是在现场"
)
# 先匹配带展厅的整段位置，再匹配「我此刻就在」这种开口。只削开头，留下后面的事。
_BOOTH_LEAD = re.compile(
    r"^(?:"
    r"我此刻就在(?:这场)?博博会(?:现场)?(?:的)?(?:宽创(?:国际)?)?(?:的)?(?:展[厅台]|站台)?[里中]?[，,]?"
    r"|我(?:此刻)?就在这里[，,]?"
    r"|我(?:此刻)?在这里[，,]?"
    r"|我此刻就在[，,]?"
    r"|我就在(?:这场)?博博会(?:现场)?(?:的)?(?:宽创(?:国际)?)?(?:的)?(?:展[厅台]|站台)?[里中]?[，,]?"
    r"|人就在宽创(?:国际)?(?:的)?(?:展[厅台]|站台)?[里中]?[，,]?"
    r"|就在(?:这场)?博博会(?:现场)?(?:的)?(?:宽创(?:国际)?)?(?:的)?(?:展[厅台]|站台)?[里中]?[，,]?"
    r"|就在宽创(?:国际)?(?:的)?(?:展[厅台]|站台)?[里中]?[，,]?"
    r"|就在这场[，,]?"
    r")"
)
_PRESENCE_ONLY = re.compile(
    r"^(?:就在|人就在|我(?:此刻)?就在)"
    r"(?:这场)?博博会(?:现场)?(?:的)?(?:宽创(?:国际)?)?(?:的)?"
    r"(?:展厅|展台|站台)?[里中]?[。！？]?$"
)
_LOC_FILL = re.compile(
    r"我(?:此刻|就)?在这里|我(?:此刻)?就在|人就在|这场|博博会|现场|"
    r"宽创国际|宽创|展厅|展台|站台|第十一届|"
    r"中国博物馆及相关产品与技术博览会|正在|举行|里|的"
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
    """了解完刚讲的内容，再问还想了解博博会或宽创展台的什么。屏幕按钮用这一句。"""
    topic = re.sub(r"\s+", "", topic or "") or "这些"
    return "了解完%s，您还想了解博博会或宽创展台的哪些内容呢？" % topic


def spoken_closings(topic: str) -> list:
    """口播句尾的几种问法。同一话题不要每次都念同一句。"""
    topic = re.sub(r"\s+", "", topic or "") or "这些"
    return [
        hall_hook(topic),
        "关于%s，您还想再听一段，还是换个博博会的话题？" % topic,
        "%s先讲到这儿。您要是想问博博会或宽创展台，直接说就行。" % topic,
        "我还能接着讲。您是想继续听%s，还是看看这届博博会？" % topic,
        "还有想了解的，尽管问。博博会和宽创展台的事我都可以说。",
        "您再挑一个问题吧。可以接着问%s，也可以问这届博博会。" % topic,
    ]


def pick_spoken_closing(topic: str, seed: str = "") -> str:
    """按刚讲完的正文挑一句，同一段话每次挑到的问法相同。"""
    import zlib
    pool = spoken_closings(topic)
    if not pool:
        return ""
    key = (seed or topic or "这些").strip()
    return pool[zlib.adler32(key.encode("utf-8")) % len(pool)]


def spoken_statements(topic: str) -> list:
    """正文已经在发问时，只接不带问号的邀请，避免连着两句问话。"""
    topic = re.sub(r"\s+", "", topic or "") or "这些"
    return [
        "%s先讲到这儿。您要是想问博博会或宽创展台，直接说就行。" % topic,
        "还有想了解的，尽管问。博博会和宽创展台的事我都可以说。",
    ]


def pick_spoken_statement(topic: str, seed: str = "") -> str:
    import zlib
    pool = spoken_statements(topic)
    key = (seed or topic or "这些").strip()
    return pool[zlib.adler32(key.encode("utf-8")) % len(pool)]


def is_bobohui_question(question: str) -> bool:
    return bool(_ASK.search(question or ""))


def asks_where(question: str) -> bool:
    """观众在问人在哪、展厅在哪。讲画、讲经历时的「在哪里画的」不算。"""
    return bool(_WHERE.search(question or ""))


def _peel_location(sentence: str) -> str:
    """削掉句首的展厅位置。后面还有要讲的事就留下，纯位置则削空。"""
    cur = (sentence or "").strip()
    for _ in range(4):
        matched = _BOOTH_LEAD.match(cur)
        if not matched:
            break
        nxt = cur[matched.end():].lstrip("，, ")
        if nxt == cur:
            break
        cur = nxt
    return cur.strip()


def _only_location(sentence: str) -> bool:
    text = (sentence or "").strip()
    if _PRESENCE_ONLY.match(text):
        return True
    bare = re.sub(r"[\s，,。！？、；：]", "", text)
    if not bare:
        return True
    return _LOC_FILL.sub("", bare) == ""


def _presence_sentence(sentence: str) -> bool:
    """整句只在报位置。短内容如「画睡莲」「作画」不算位置。"""
    return _only_location(_peel_location(sentence))


def _drop_presence_lead(sentence: str) -> str:
    peeled = _peel_location(sentence)
    if _only_location(peeled):
        return ""
    return peeled


def strip_unasked_here(question: str, text: str) -> str:
    """没问位置时，去掉开头的「我在这里，就在宽创展厅」。问到哪里举办则保留。"""
    raw = (text or "").strip()
    if not raw or asks_where(question) or is_bobohui_question(question):
        return raw
    from rag.visit import is_visit_query
    if is_visit_query(question):
        return raw
    parts = re.findall(r"[^。！？]+[。！？]?", raw) or [raw]
    kept = []
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        if _presence_sentence(piece):
            continue
        piece = _drop_presence_lead(piece)
        if piece:
            kept.append(piece if piece[-1] in "。！？" else piece)
    cleaned = "".join(kept).strip()
    return cleaned or raw


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
