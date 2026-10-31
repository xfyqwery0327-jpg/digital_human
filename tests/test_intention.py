# -*- coding: utf-8 -*-
"""tts_start 前的屏幕意图：导览 / 预约 / 展品 / talk。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER9 = ROOT / "server9"
if str(SERVER9) not in sys.path:
    sys.path.insert(0, str(SERVER9))

from rag.dialogue import button_screen_intent, first_exhibit_name, intention_event  # noqa: E402
from ws.session_wire import tts_wire_allowed  # noqa: E402

TITLES = ["千里江山图", "清明上河图", "睡莲池", "睡莲", "日出·印象"]
ALIASES = {"日出印象": "日出·印象"}


def test_frontend_button_scripts():
    exhibit = intention_event("介绍一下你的展品")
    assert button_screen_intent("介绍一下你的展品") == "exhibit"
    assert exhibit["message"] == {"intentionType": "exhibit", "targetName": ""}
    reserve = intention_event("预约二维码展示。")
    assert reserve["message"] == {"intentionType": "reserve", "targetName": ""}
    guide = intention_event("展馆导览图展示")
    assert guide["message"] == {"intentionType": "guide", "targetName": ""}


def test_button_script_is_spoken_before_booking_rewrite():
    src = (SERVER9 / "ws" / "tts.py").read_text(encoding="utf-8")
    start = src.find("audio_task = asyncio.create_task")
    booking = src.find("elif is_ticket_booking", start)
    block = src[start:booking]
    assert "button_screen_intent" in block
    assert block.find("button_screen_intent") < block.find("is_foreign_question")


def test_exhibit_uses_earliest_name_in_question():
    event = intention_event(
        "介绍一下《千里江山图》和《清明上河图》",
        "先说清明上河图，再说千里江山图。",
        titles=TITLES, aliases=ALIASES)
    assert event["type"] == "intention"
    assert event["message"]["intentionType"] == "exhibit"
    assert event["message"]["targetName"] == "千里江山图"


def test_reply_fills_name_when_question_has_none():
    event = intention_event(
        "介绍一下这些展品",
        "先看《清明上河图》，再看《千里江山图》。",
        titles=TITLES, aliases=ALIASES)
    assert event["message"]["intentionType"] == "exhibit"
    assert event["message"]["targetName"] == "清明上河图"


def test_longer_title_wins_when_it_starts_first():
    assert first_exhibit_name("先看睡莲池，再看睡莲", TITLES) == "睡莲池"
    assert first_exhibit_name("先看睡莲，再看睡莲池", TITLES) == "睡莲"


def test_alias_and_book_title_without_catalog():
    assert first_exhibit_name("日出印象怎么画的", TITLES, ALIASES) == "日出·印象"
    assert first_exhibit_name("介绍《千里江山图》", titles=[]) == "千里江山图"


def test_reserve_guide_and_talk_have_empty_target():
    reserve = intention_event("帮我预约明天门票", "好，预约请在屏幕上填写。")
    assert reserve["message"] == {"intentionType": "reserve", "targetName": ""}
    guide = intention_event("带我导览一下", "从这边开始。")
    assert guide["message"]["intentionType"] == "guide"
    assert guide["message"]["targetName"] == ""
    hours = intention_event("周一闭馆吗", "周一闭馆。")
    assert hours["message"]["intentionType"] == "guide"
    talk = intention_event("你好", "你好。", titles=TITLES)
    assert talk["message"] == {"intentionType": "talk", "targetName": ""}


def test_named_exhibit_beats_look_verb():
    event = intention_event("带我看看《睡莲》", titles=TITLES)
    assert event["message"]["intentionType"] == "exhibit"
    assert event["message"]["targetName"] == "睡莲"


def test_classified_exhibit_without_name_uses_reply():
    event = intention_event(
        "讲讲那个", "这是《千里江山图》。旁边还有清明上河图。",
        classified_intent="exhibit", titles=TITLES)
    assert event["message"]["intentionType"] == "exhibit"
    assert event["message"]["targetName"] == "千里江山图"


def test_booking_beats_named_exhibit():
    event = intention_event("帮我预约《睡莲》的票", titles=TITLES)
    assert event["message"]["intentionType"] == "reserve"
    assert event["message"]["targetName"] == ""


def test_stale_intention_is_dropped():
    latest = ["new-sid"]
    frame = {"type": "intention", "message": {"intentionType": "talk", "targetName": ""}}
    assert tts_wire_allowed(frame, "old-sid", latest) is False
    assert tts_wire_allowed(frame, "new-sid", latest) is True


def test_intention_is_emitted_before_tts_start():
    tts = (SERVER9 / "ws" / "tts.py").read_text(encoding="utf-8")
    start = tts.find("async def ensure_started")
    block = tts[start:tts.find("async def gated_send", start)]
    assert block.find("intention_event") < block.find('"tts_start"')
    stream = (SERVER9 / "robot" / "stream.py").read_text(encoding="utf-8")
    send = stream.find("async def _send_intention")
    fan = stream.find("async def fanout", send)
    body = stream[send:fan]
    assert "_send_intention" in body
    gate = stream.find('if kind == "tts_start"')
    assert gate != -1
    assert stream.find("await _send_intention()", gate) < stream.find(
        "last_pushed = await tts_fanout.send", gate)
