# -*- coding: utf-8 -*-
"""口播分句。单段限制在 30~40 个汉字，禁止把整段长文一次送进 CosyVoice3。

长文本会让 Qwen2 的 KV 和每步 (seq, seq) mask 按句长膨胀，Flow / HiFT 的激活也会跟着变大。
机器人一轮回答先切成小段，逐段合成。
"""
from __future__ import annotations

DEFAULT_MAX_CHARS = 36
MIN_MAX_CHARS = 30
MAX_MAX_CHARS = 40

_STRONG = set("。！？；!?")
_WEAK_MARK = set("，、,")


def clamp_max_chars(value=None) -> int:
    """接口层夹到 30~40。工具函数本身仍接受调用方传入的上限，方便单测。"""
    if value is None or value == "":
        return DEFAULT_MAX_CHARS
    try:
        n = int(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_CHARS
    if n < MIN_MAX_CHARS:
        return MIN_MAX_CHARS
    if n > MAX_MAX_CHARS:
        return MAX_MAX_CHARS
    return n


def _clauses(text: str) -> list:
    """按句读切开，标点留在本段。小数点两侧都是数字时不切。"""
    parts = []
    buf = []
    n = len(text)
    for i, ch in enumerate(text):
        buf.append(ch)
        if ch in "，。！？；、!?":
            piece = "".join(buf).strip()
            if piece:
                parts.append(piece)
            buf = []
            continue
        if ch in ".,":
            prev_d = i > 0 and text[i - 1].isdigit()
            next_d = i + 1 < n and text[i + 1].isdigit()
            if prev_d and next_d:
                continue
            piece = "".join(buf).strip()
            if piece:
                parts.append(piece)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _hard_cut(text: str, max_chars: int) -> list:
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars) if text[i:i + max_chars].strip()]


def split_tts_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list:
    """返回每段长度不超过 max_chars 的口播片段。空文本返回空列表。"""
    max_chars = int(max_chars or DEFAULT_MAX_CHARS)
    if max_chars < 8:
        max_chars = 8
    raw = (text or "").replace("\r", "").replace("\n", "").strip()
    if not raw:
        return []

    clauses = _clauses(raw) or [raw]
    pieces = []
    for clause in clauses:
        if len(clause) <= max_chars:
            pieces.append(clause)
        else:
            pieces.extend(_hard_cut(clause, max_chars))

    # 只合并「对了，」这种很短的弱标点。句号后的下一句永远单独成段。
    merged = []
    buf = ""
    for piece in pieces:
        if not buf:
            buf = piece
            continue
        prev_strong = buf[-1] in _STRONG
        can_merge = (
            not prev_strong
            and buf[-1] in _WEAK_MARK
            and len(buf) < 8
            and len(buf) + len(piece) <= max_chars
        )
        if can_merge:
            buf = buf + piece
        else:
            merged.append(buf)
            buf = piece
    if buf:
        merged.append(buf)

    out = []
    for piece in merged:
        piece = piece.strip()
        if not piece or not piece.strip("，。！？；、,.!?… "):
            continue
        if len(piece) <= max_chars:
            out.append(piece)
        else:
            out.extend(_hard_cut(piece, max_chars))
    return out
