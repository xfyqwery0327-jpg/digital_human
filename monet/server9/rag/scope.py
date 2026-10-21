# -*- coding: utf-8 -*-
"""本届展范围：只讲眼前这场展，不介绍其他展会。"""
from __future__ import annotations
import re

from config import EXHIBITION_CATEGORY, REFUSE_TEXT

# 「其他展品」指向本厅作品，不能当成其他展会
_OTHER_SHOW = re.compile(
    r"(?:其他|其它|别的)(?:展览|展会|特展)|"
    r"(?:其他|其它|别的)展(?!品)|"
    r"上一[届场]展|下一[届场]展|上一次展|下一次展|"
    r"巡回展|双年展|三年展|博览会|"
    r"卢浮宫|奥赛|奥塞|大都会博物馆|大英博物馆|泰特|"
    r"故宫博物院|国家博物馆|上海博物馆|"
    r"其他馆|别的博物馆|另一个展|别的馆",
    re.IGNORECASE,
)


def is_other_exhibition_query(question: str) -> bool:
    """问的是别的展会/别的馆，而不是本厅展品。"""
    q = (question or "").strip()
    if not q:
        return False
    return bool(_OTHER_SHOW.search(q))


def locked_category(_ignored=None):
    """机器人/游客请求里的 category 一律忽略，只查本届展。"""
    return EXHIBITION_CATEGORY


def refuse_payload(reason: str = "low_score") -> dict:
    return {
        "text": REFUSE_TEXT,
        "sources": [],
        "refused": True,
        "reason": reason,
    }


_LEAK_IN_ANSWER = re.compile(
    r"(?:其他|其它|别的)(?:展览|展会|特展)|"
    r"(?:其他|其它|别的)展(?!品)|"
    r"巡回展|双年展|三年展|"
    r"上一[届场]展|下一[届场]展",
    re.IGNORECASE,
)


def answer_leaves_exhibition(text: str) -> bool:
    """模型改写稿是否讲到了别的展会。"""
    return bool(_LEAK_IN_ANSWER.search(text or ""))


def clamp_hall_answer(text: str) -> str:
    if answer_leaves_exhibition(text):
        return REFUSE_TEXT
    return text or REFUSE_TEXT


def hall_safe_sentence(sent: str):
    """流式切句：越界则改口拒答，并应停止后续句。返回 (口播文本, 是否停止)。"""
    if not sent:
        return sent, False
    if answer_leaves_exhibition(sent):
        return REFUSE_TEXT, True
    return sent, False


def script_leaves_hall(text: str) -> bool:
    """自定义欢迎/再见等口播稿是否在介绍别的展或别的馆。"""
    t = (text or "").strip()
    if not t:
        return False
    return is_other_exhibition_query(t) or answer_leaves_exhibition(t)


def iter_hall_history(turns):
    """给 LLM 的对话历史：丢掉问其他展、或答稿已越界的轮次，避免带偏下一轮。"""
    for turn in turns or []:
        q = getattr(turn, "question", "") or ""
        a = getattr(turn, "answer", "") or ""
        if is_other_exhibition_query(q) or answer_leaves_exhibition(a):
            continue
        yield q, a
