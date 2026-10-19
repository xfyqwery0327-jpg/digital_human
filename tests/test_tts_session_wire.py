# -*- coding: utf-8 -*-
"""旧 TTS 会话帧不得覆盖新一轮，否则 UE 收不到 tts_chunk。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER9 = ROOT / "server9"
if str(SERVER9) not in sys.path:
    sys.path.insert(0, str(SERVER9))

from ws.session_wire import tts_wire_allowed  # noqa: E402


def test_stale_cancelled_end_is_dropped():
    old = "f02e6e07-aaaa-bbbb-cccc-dddddddddddd"
    new = "f786da30-aaaa-bbbb-cccc-dddddddddddd"
    latest = [new]
    end = {"type": "tts_end", "session_id": old, "cancelled": True,
           "audio": False, "seq_count": 0}
    assert tts_wire_allowed(end, old, latest) is False
    start = {"type": "tts_start", "session_id": new, "text": "我是克洛德·莫奈"}
    assert tts_wire_allowed(start, new, latest) is True
    chunk = {"type": "tts_chunk", "session_id": new, "seq": 0, "data": "AAA="}
    assert tts_wire_allowed(chunk, new, latest) is True


def test_current_session_can_still_end():
    sid = "current-session"
    latest = [sid]
    end = {"type": "tts_end", "session_id": sid, "seq_count": 3}
    assert tts_wire_allowed(end, sid, latest) is True


def test_control_frames_not_gated():
    latest = ["new-sid"]
    assert tts_wire_allowed({"type": "pong"}, "old-sid", latest) is True
    assert tts_wire_allowed({"type": "ready"}, "old-sid", latest) is True
    assert tts_wire_allowed({"type": "cancel_ack"}, "old-sid", latest) is True
    assert tts_wire_allowed({"type": "error", "message": "x"}, "old-sid", latest) is True


def test_empty_latest_allows_all():
    sid = "only-one"
    assert tts_wire_allowed({"type": "tts_end"}, sid, None) is True
    assert tts_wire_allowed({"type": "tts_end"}, sid, [""]) is True


def test_visitor_tts_start_waits_for_speakable_text():
    src = (SERVER9 / "ws" / "tts.py").read_text(encoding="utf-8")
    assert "await ensure_started(user_question)" not in src
    assert "tts_wire_allowed" in src
    assert "spawn_tts" in src
    assert "begin_tts_session" in src
    begin = src.find("def begin_tts_session")
    assert begin != -1
    block = src[begin:begin + 500]
    assert block.find("latest_sid[0] = session_id") < block.find("cancel_all()")
