# -*- coding: utf-8 -*-
"""本场展是「浪漫与星辰」。开放时间以这份资料为准，不采用吉维尼或印象派特展的时间。"""
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
    "这场展览是浪漫与星辰。"
    "从二零二六年七月十日至十月十日。"
    "每天九点到十八点开放，周六到二十一点。"
    "平日十七点停止入馆，周六二十点停止入馆。"
    "周一闭馆，法定节假日除外。"
    "地点在贵州省博物馆负一楼六号厅。"
)

_VISIT_HINT = (
    "这个问题在问本场展览「浪漫与星辰」的开放时间、票价或地址。"
    "只许使用资料里浪漫与星辰的句子。"
    "不要把吉维尼、法国花园或印象派特展的时间说成本场展的时间。\n"
)


def _monet(persona) -> bool:
    return getattr(persona, "id", "") == "monet"


def is_visit_query(question: str) -> bool:
    return bool(_VISIT.search(question or ""))


def visit_hint(question: str, persona=None) -> str:
    if _monet(persona) and is_visit_query(question):
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
    """问时间、票价、地址时，整段改回浪漫与星辰。讲画时只删掉串进来的展期句。"""
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
    return "开放时间以浪漫与星辰这场展为准。"


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
