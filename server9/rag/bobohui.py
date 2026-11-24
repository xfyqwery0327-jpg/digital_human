# -*- coding: utf-8 -*-
"""当前场地的口播。文案在 rag.scene，由 deploy/scene.id 决定是宽创大厅还是博博会。"""
from __future__ import annotations
import re


def _pack():
    from rag.scene import active
    return active()


def __getattr__(name):
    pack = _pack()
    key = {
        "TEXT": "text",
        "WHEN_WHERE": "when_where",
        "NO_OLD_TICKET": "no_old_ticket",
        "PRESENCE_RULE": "presence_rule",
        "ANSWER_RULE": "answer_rule",
    }.get(name)
    if key:
        return pack[key]
    raise AttributeError(name)


_WHERE = re.compile(
    r"(?:你|人|咱们|我们).{0,4}在哪|"
    r"展[厅台].{0,6}在哪|"
    r"在哪里举办|哪里举办|哪个馆|什么地方办|"
    r"人在不在|你在不在|是不是在现场"
)
# 先匹配带展厅的整段位置，再匹配「我此刻就在」这种开口。只削开头，留下后面的事。
_PLACE = r"(?:这场)?(?:博博会(?:现场)?|宽创大厅)"
_BOOTH_LEAD = re.compile(
    r"^(?:"
    r"我此刻就在" + _PLACE + r"(?:的)?(?:宽创(?:国际)?)?(?:的)?(?:展[厅台]|站台|大厅)?[里中]?[，,]?"
    r"|我(?:此刻)?就在这里[，,]?"
    r"|我(?:此刻)?在这里[，,]?"
    r"|我此刻就在[，,]?"
    r"|我就在" + _PLACE + r"(?:的)?(?:宽创(?:国际)?)?(?:的)?(?:展[厅台]|站台|大厅)?[里中]?[，,]?"
    r"|人就在宽创(?:国际|大厅)?(?:的)?(?:展[厅台]|站台)?[里中]?[，,]?"
    r"|就在" + _PLACE + r"(?:的)?(?:宽创(?:国际)?)?(?:的)?(?:展[厅台]|站台|大厅)?[里中]?[，,]?"
    r"|就在宽创(?:国际|大厅)?(?:的)?(?:展[厅台]|站台)?[里中]?[，,]?"
    r"|就在这场[，,]?"
    r")"
)
_PRESENCE_ONLY = re.compile(
    r"^(?:就在|人就在|我(?:此刻)?就在)"
    + _PLACE +
    r"(?:的)?(?:宽创(?:国际)?)?(?:的)?"
    r"(?:展厅|展台|站台|大厅)?[里中]?[。！？]?$"
)
_LOC_FILL = re.compile(
    r"我(?:此刻|就)?在这里|我(?:此刻)?就在|人就在|这场|博博会|宽创大厅|宽创发展|现场|"
    r"宽创国际|宽创|展厅|展台|站台|大厅|第十一届|"
    r"中国博物馆及相关产品与技术博览会|正在|举行|里|的"
)
_KEEP_PLACE = re.compile(
    r"博博会|中国博物馆及相关产品与技术博览会|宽创大厅|宽创发展|宽创国际"
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
    """了解完刚讲的内容，再问还想了解当前场地的什么。屏幕按钮用这一句。"""
    topic = re.sub(r"\s+", "", topic or "") or "这些"
    return "了解完%s，%s" % (topic, _pack()["hook_tail"])


def spoken_closings(topic: str) -> list:
    """口播句尾的几种问法。同一话题不要每次都念同一句。"""
    topic = re.sub(r"\s+", "", topic or "") or "这些"
    pack = _pack()
    place = pack["short_name"]
    booth = pack["booth_name"]
    look = pack["look_name"]
    return [
        hall_hook(topic),
        "关于%s，您还想再听一段，还是换个%s的话题？" % (topic, place),
        "%s先讲到这儿。您要是想问%s或%s，直接说就行。" % (topic, place, booth),
        "我还能接着讲。您是想继续听%s，还是看看%s？" % (topic, look),
        "还有想了解的，尽管问。%s和%s的事我都可以说。" % (place, booth),
        "您再挑一个问题吧。可以接着问%s，也可以问%s。" % (topic, look),
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
    pack = _pack()
    return [
        "%s先讲到这儿。您要是想问%s或%s，直接说就行。" % (topic, pack["short_name"], pack["booth_name"]),
        "还有想了解的，尽管问。%s和%s的事我都可以说。" % (pack["short_name"], pack["booth_name"]),
    ]


def pick_spoken_statement(topic: str, seed: str = "") -> str:
    import zlib
    pool = spoken_statements(topic)
    key = (seed or topic or "这些").strip()
    return pool[zlib.adler32(key.encode("utf-8")) % len(pool)]


def is_bobohui_question(question: str) -> bool:
    """当前场地要接的问题。大厅认宽创发展，博博会认本届展会。"""
    return bool(_pack()["ask"].search(question or ""))


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
    if not raw or asks_where(question) or _KEEP_PLACE.search(question or "") or is_bobohui_question(question):
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
    """问到当前场地时，给检索一条可直接引用的资料。"""
    from rag.scene import place_hit
    return place_hit(question)


def merge_bobohui(question: str, hits, threshold: float):
    from rag.scene import merge_hits
    return merge_hits(question, hits, threshold)
