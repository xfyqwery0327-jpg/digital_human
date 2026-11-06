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


def test_polyphone_places_and_relics_fill_target():
    from config import PERSONAS_DIR
    from persona.registry import _parse_persona, _yaml_load

    fuhao = _parse_persona(_yaml_load(PERSONAS_DIR / "fuhao.yaml"), "fuhao")
    event = intention_event(
        "你怎么占卜", "我用龟甲占卜，在殷墟灼烧。", persona=fuhao, titles=[])
    assert event["message"]["targetName"] == "甲骨"
    kuma = _parse_persona(_yaml_load(PERSONAS_DIR / "kumarajiva.yaml"), "kumarajiva")
    event = intention_event(
        "讲讲你的经历",
        "我曾在龟兹的克孜尔石窟和库木吐喇石窟驻留。",
        persona=kuma, titles=[])
    assert event["message"]["targetName"] == "克孜尔千佛洞"
    event = intention_event(
        "金刚经讲什么",
        "《金刚般若波罗蜜经》是我在长安译的。",
        persona=kuma, titles=[])
    assert event["message"]["intentionType"] == "exhibit"
    assert event["message"]["targetName"] == "金刚经"


def test_intention_waits_for_full_reply():
    tts = (SERVER9 / "ws" / "tts.py").read_text(encoding="utf-8")
    start = tts.find("async def ensure_started")
    block = tts[start:tts.find("async def _open_screen", start)]
    assert block.find("reply_final") < block.find("started = True")
    assert block.find("_intention_for") < block.find('"tts_start"')


def test_spoken_work_fills_target_for_three_personas():
    from config import PERSONAS_DIR
    from persona.registry import _parse_persona, _yaml_load

    cases = [
        ("monet", "介绍一下你自己",
         "我是克劳德·莫奈。其中《睡莲》我前后画了很多幅。", "talk", "睡莲"),
        ("monet", "介绍一下你的作品",
         "我一生追逐光。《印象·日出》捕捉清晨，而《睡莲》是系列。", "exhibit", "日出·印象"),
        ("fuhao", "介绍一下你自己",
         "我的事刻在甲骨上，墓里还有青铜钺。", "talk", "甲骨"),
        ("kumarajiva", "讲讲你的生平",
         "我到长安后译了《金刚经》，后来又译《法华经》。", "talk", "金刚经"),
    ]
    for pid, question, reply, kind, name in cases:
        persona = _parse_persona(_yaml_load(PERSONAS_DIR / ("%s.yaml" % pid)), pid)
        event = intention_event(question, reply, persona=persona, titles=[])
        assert event["message"]["intentionType"] == kind, pid
        assert event["message"]["targetName"] == name, (pid, event["message"])


def test_guide_reply_carries_named_work():
    event = intention_event(
        "带我导览一下", "从这边开始，先看《睡莲》。",
        titles=TITLES, aliases=ALIASES)
    assert event["message"]["intentionType"] == "guide"
    assert event["message"]["targetName"] == "睡莲"


def test_reserve_and_refuse_do_not_carry_work():
    reserve = intention_event(
        "帮我预约明天门票", "好，预约请在屏幕上填写。睡莲也在这边。",
        titles=TITLES, aliases=ALIASES)
    assert reserve["message"]["targetName"] == ""
    refused = intention_event(
        "妇好是谁",
        "这个我还不太清楚。展会我只介绍博博会。你可以继续问我的画，比如睡莲或《日出·印象》。",
        titles=TITLES, aliases=ALIASES)
    assert refused["message"]["targetName"] == ""
    plain = intention_event("你好", "我读过《高僧传》。", titles=TITLES, aliases=ALIASES)
    assert plain["message"]["targetName"] == ""


def test_stale_intention_is_dropped():
    latest = ["new-sid"]
    frame = {"type": "intention", "message": {"intentionType": "talk", "targetName": ""}}
    assert tts_wire_allowed(frame, "old-sid", latest) is False
    assert tts_wire_allowed(frame, "new-sid", latest) is True


def test_intention_is_emitted_before_tts_start():
    tts = (SERVER9 / "ws" / "tts.py").read_text(encoding="utf-8")
    start = tts.find("async def ensure_started")
    block = tts[start:tts.find("async def gated_send", start)]
    assert block.find("_intention_for") < block.find('"tts_start"')
    answer = tts.find("async def send_llm_answer")
    answer_block = tts[answer:tts.find("async def _emit_tts_end", answer)]
    assert answer_block.find("refresh_work_target") < answer_block.find('"llm_answer"')
    stream = (SERVER9 / "robot" / "stream.py").read_text(encoding="utf-8")
    send = stream.find("async def _send_intention")
    fan = stream.find("async def fanout", send)
    body = stream[send:fan]
    assert "_send_intention" in body
    gate = stream.find('if kind == "tts_start"')
    assert gate != -1
    assert stream.find("await _send_intention()", gate) < stream.find(
        "last_pushed = await tts_fanout.send", gate)
