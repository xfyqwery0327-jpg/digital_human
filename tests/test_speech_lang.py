# -*- coding: utf-8 -*-
"""口播切到中英日韩。问题仍是中文，TTS 请求带上 language。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server9"))

from rag.dialogue import RealtimeMemory, rule_intent
from rag.speech_lang import (
    confirm_text, detect_lang, needs_translation, output_rule, parse_switch,
    qwen_language_of, set_reply_lang, split_switch,
)
from tts.remote import tts_payload


def test_switch_intent_from_chinese():
    assert parse_switch("切换英语") == "en"
    assert parse_switch("换成日语") == "ja"
    assert parse_switch("说韩语") == "ko"
    assert parse_switch("用中文") == "zh"
    assert parse_switch("今天天气怎么样") == ""
    assert rule_intent("切换英语") == "lang_switch"
    assert rule_intent("今天天气怎么样") == "weather"
    code, rest = split_switch("用英语介绍一下博博会")
    assert code == "en" and "博博会" in rest
    mem = RealtimeMemory()
    set_reply_lang(mem, "ja")
    assert mem.reply_lang == "ja"
    assert output_rule("zh") == ""
    assert "日语" in output_rule("ja")
    assert "China Museum Expo" in output_rule("en")
    assert "中国博物館エキスポ" in output_rule("ja")
    assert "중국 박물관 엑스포" in output_rule("ko")
    assert "Bobohui" not in output_rule("en")


def test_short_clause_does_not_ask_for_translation():
    assert needs_translation("今、", "ja") is False
    assert needs_translation("我站在展厅里。", "ja") is True
    assert needs_translation("I am right here in the hall.", "en") is False
    assert needs_translation("我在这里。", "en") is True


def test_qwen_language_field():
    samples = {
        "zh": confirm_text("zh"),
        "en": confirm_text("en"),
        "ja": confirm_text("ja"),
        "ko": confirm_text("ko"),
    }
    expect = {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean"}
    for code, text in samples.items():
        assert detect_lang(text) == code
        body = tts_payload(text, "monet")
        assert body["language"] == expect[code]
        assert body["speaker_id"] == "monet"
        assert "http" not in body["text"]
        assert qwen_language_of(text) == expect[code]
