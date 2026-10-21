# -*- coding: utf-8 -*-
"""本届展范围：规则跟人设走，禁止用莫奈文案/正则给妇好兜底。"""
from __future__ import annotations
import re
from functools import lru_cache
from typing import List, Optional

from persona.models import PersonaConfig

_GENERIC_OTHER = re.compile(
    r"(?:其他|其它|别的)(?:展览|展会|特展)|"
    r"(?:其他|其它|别的)展(?!品)|"
    r"上一[届场]展|下一[届场]展|上一次展|下一次展|"
    r"巡回展|双年展|三年展|博览会",
    re.IGNORECASE,
)
_GENERIC_LEAK = re.compile(
    r"(?:其他|其它|别的)(?:展览|展会|特展)|"
    r"(?:其他|其它|别的)展(?!品)|"
    r"巡回展|双年展|三年展|"
    r"上一[届场]展|下一[届场]展",
    re.IGNORECASE,
)


def _persona(explicit: Optional[PersonaConfig] = None) -> PersonaConfig:
    if explicit is not None:
        return explicit
    from persona.runtime import active
    return active()


def refuse_text(persona: Optional[PersonaConfig] = None) -> str:
    p = _persona(persona)
    return (p.refuse_text or "").strip() or "这个我还不太清楚。我只讲解眼前这场展。"


@lru_cache(maxsize=32)
def _compiled(patterns: tuple) -> List[re.Pattern]:
    out = []
    for pat in patterns:
        try:
            out.append(re.compile(pat, re.IGNORECASE))
        except re.error:
            out.append(re.compile(re.escape(pat), re.IGNORECASE))
    return out


def is_cross_persona_query(question: str, persona: Optional[PersonaConfig] = None) -> bool:
    """问的是另一个数字人的身份/展陈，必须拒答，不能去查别人的库。"""
    p = _persona(persona)
    q = (question or "").strip()
    if not q:
        return False
    for term in p.forbidden_terms:
        if term and term in q:
            return True
    for rx in _compiled(tuple(p.cross_persona_patterns)):
        if rx.search(q):
            return True
    return False


def is_other_exhibition_query(question: str, persona: Optional[PersonaConfig] = None) -> bool:
    """问的是别的展会/别的馆，而不是本厅展品。"""
    q = (question or "").strip()
    if not q:
        return False
    if is_cross_persona_query(q, persona):
        return True
    p = _persona(persona)
    patterns = p.other_exhibition_patterns
    if patterns:
        for rx in _compiled(tuple(patterns)):
            if rx.search(q):
                return True
        return False
    return bool(_GENERIC_OTHER.search(q))


def locked_category(_ignored=None, persona: Optional[PersonaConfig] = None):
    """忽略客户端 category，只用当前请求人设的知识类目。"""
    p = _persona(persona)
    return list(p.categories) if p.categories else []


def refuse_payload(reason: str = "low_score", persona: Optional[PersonaConfig] = None) -> dict:
    return {
        "text": refuse_text(persona),
        "sources": [],
        "refused": True,
        "reason": reason,
    }


def answer_leaves_exhibition(text: str, persona: Optional[PersonaConfig] = None) -> bool:
    """模型改写稿是否讲到了别的展会或另一个人设。"""
    t = text or ""
    p = _persona(persona)
    if p.leak_patterns:
        for rx in _compiled(tuple(p.leak_patterns)):
            if rx.search(t):
                return True
        return False
    return bool(_GENERIC_LEAK.search(t))


def clamp_hall_answer(text: str, persona: Optional[PersonaConfig] = None) -> str:
    if answer_leaves_exhibition(text, persona):
        return refuse_text(persona)
    return text or refuse_text(persona)


def hall_safe_sentence(sent: str, persona: Optional[PersonaConfig] = None):
    """流式切句：越界则改口拒答，并应停止后续句。返回 (口播文本, 是否停止)。"""
    if not sent:
        return sent, False
    if answer_leaves_exhibition(sent, persona):
        return refuse_text(persona), True
    return sent, False


def script_leaves_hall(text: str, persona: Optional[PersonaConfig] = None) -> bool:
    """自定义欢迎/再见等口播稿是否在介绍别的展或别的馆。"""
    t = (text or "").strip()
    if not t:
        return False
    return is_other_exhibition_query(t, persona) or answer_leaves_exhibition(t, persona)


def iter_hall_history(turns, persona: Optional[PersonaConfig] = None):
    """给 LLM 的对话历史：丢掉问其他展、或答稿已越界的轮次，避免带偏下一轮。"""
    for turn in turns or []:
        q = getattr(turn, "question", "") or ""
        a = getattr(turn, "answer", "") or ""
        if is_other_exhibition_query(q, persona) or answer_leaves_exhibition(a, persona):
            continue
        yield q, a
