# -*- coding: utf-8 -*-
"""检索命中按原文送进回答。上下文 4096，不再按 500 字浓缩。"""
from __future__ import annotations

from config import PASSAGE_BUDGET

_SENT_END = "。！？；\n"


def read_hits(hits, budget: int = PASSAGE_BUDGET) -> list:
    """按检索顺序附上原文。总长超出窗口预算时，先保留前面的命中，并在句末截断。"""
    left = max(0, int(budget))
    out = []
    for hit in hits or []:
        if left <= 0:
            break
        text = ((hit or {}).get("text") or "").strip()
        if not text:
            continue
        if len(text) <= left:
            out.append(text)
            left -= len(text)
            continue
        piece = _cut(text, left)
        if piece:
            out.append(piece)
        break
    return out


def _cut(text: str, keep: int) -> str:
    text = (text or "").strip()
    if keep <= 0 or not text:
        return ""
    if len(text) <= keep:
        return text
    cut = max(text.rfind(ch, 0, keep) for ch in _SENT_END)
    if cut >= int(keep * 0.6):
        return text[:cut + 1].strip()
    return text[:keep].strip()
