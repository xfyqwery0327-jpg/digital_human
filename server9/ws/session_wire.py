# -*- coding: utf-8 -*-
"""游客 TTS 连接上只允许最新一轮把口播帧推到前端。"""
from __future__ import annotations

from typing import Optional, Sequence, Union

# 这些帧带 session_id。旧会话被新一轮打断后禁止再下发，否则 UE 会把播放器关掉。
TTS_SESSION_WIRE = frozenset({
    "tts_start", "tts_end", "tts_chunk", "tts_sentence", "llm_answer",
})


def tts_wire_allowed(obj: dict, session_id: str,
                     latest_sid: Optional[Union[Sequence[str], str]] = None) -> bool:
    """只允许当前连接上最新一轮 TTS 把口播帧推到前端。"""
    if not obj:
        return False
    kind = obj.get("type")
    if kind not in TTS_SESSION_WIRE:
        return True
    current = ""
    if isinstance(latest_sid, (list, tuple)):
        current = (latest_sid[0] if latest_sid else "") or ""
    elif latest_sid:
        current = str(latest_sid)
    if not current:
        return True
    return current == session_id
