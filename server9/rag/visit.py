# -*- coding: utf-8 -*-
"""展会只介绍博博会。不把吉维尼或原先展的时间、票价说成现场安排。"""
from __future__ import annotations
import re

_VISIT = re.compile(
    r"开放时间|几点开门|几点闭|几点开馆|闭馆|入馆|展期|门票|票价|学生票|优惠票|免票|早鸟|"
    r"哪里举办|展览在哪|展厅在哪|什么时候开|什么时候闭"
)
_FOREIGN_PLACE = re.compile(r"吉维尼|法国花园|印象派特展|诺曼底园|二楼西侧")
_SCHEDULE = re.compile(r"开放|闭馆|展期|入馆|几点|票价|门票|周一")
_FAKE_LINE = re.compile(
    r"印象派特展|二楼西侧|十二月三十一|每周三免费|明信片一套|成人票八十|成人票80"
)
_SENT = re.compile(r"[^。！？]+[。！？]?")

CANONICAL_HOURS = (
    "我就在博博会现场的宽创展厅里。"
    "第十一届博博会定于二零二六年十一月十三日至十六日，在海南国际会展中心举办。"
    "原先的门票价格不介绍。"
)

_VISIT_HINT = (
    "这个问题在问展会时间或地点。只介绍第十一届博博会，不要介绍原先的展，也不要报原先的门票价格。"
    "不要把吉维尼或别的馆的开放时间说成博博会的时间。\n"
)


def _monet(persona) -> bool:
    return getattr(persona, "id", "") == "monet"


def is_visit_query(question: str) -> bool:
    return bool(_VISIT.search(question or ""))


def visit_hint(question: str, persona=None) -> str:
    del persona
    if is_visit_query(question):
        return _VISIT_HINT
    return ""


def leaks_foreign_schedule(sentence: str, persona=None) -> bool:
    """吉维尼可以作为画画的地方。一旦和开放时间、展期、票价绑在一起，就是串场。"""
    if persona is not None and not _monet(persona):
        return False
    text = sentence or ""
    if _FAKE_LINE.search(text):
        return True
    return bool(_FOREIGN_PLACE.search(text) and _SCHEDULE.search(text))


def _sentences(text: str):
    parts = [p.strip() for p in _SENT.findall(text or "") if p.strip()]
    return parts or ([text.strip()] if (text or "").strip() else [])


def guard_hall_clock(question: str, answer: str, persona=None) -> str:
    """问时间、票价、地址时，整段改回博博会。讲画时只删掉串进来的展期句。"""
    if not _monet(persona):
        return answer or ""
    parts = _sentences(answer)
    if not any(leaks_foreign_schedule(p, persona) for p in parts):
        return answer or ""
    if is_visit_query(question):
        return CANONICAL_HOURS
    kept = [p for p in parts if not leaks_foreign_schedule(p, persona)]
    if kept:
        return "".join(kept)
    return CANONICAL_HOURS


def filter_spoken_sentence(sentence: str, question: str, persona=None,
                           state: dict = None) -> str:
    """流式口播：串场的展期句不送去合成。问时间时只改口一次。"""
    sent = (sentence or "").strip()
    if not sent or not leaks_foreign_schedule(sent, persona):
        return sent
    box = state if state is not None else {}
    if is_visit_query(question) and not box.get("replaced"):
        box["replaced"] = True
        return CANONICAL_HOURS
    return ""
