# -*- coding: utf-8 -*-
"""口播切到中英日韩。问题仍是中文，TTS 请求带上 language。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server9"))

from rag.dialogue import RealtimeMemory, rule_intent
from rag.speech_lang import (
    confirm_text, detect_lang, needs_translation, output_rule, parse_switch,
    polish_spoken, qwen_language_of, set_reply_lang, split_switch,
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


def test_polish_report_failures_without_model():
    han = __import__("re").compile(r"[\u4e00-\u9fff]")
    weather = (
        "Haikou is partly cloudy today, with temperatures ranging from 十四 to 二十六 degrees Celsius. "
        "After learning about the weather, what aspects of China Museum Expo would you like to know?"
    )
    got = polish_spoken(weather, "en")
    assert "14" in got and "26" in got and han.search(got) is None

    ko_weather = (
        "기온은 十四도에서 二十六도입니다. 날씨를 알아보셨以后, "
        "더 알고 싶은 것은 박물관 엑스포의 어떤 내용인가요?"
    )
    got = polish_spoken(ko_weather, "ko")
    assert "14" in got and "26" in got and "이후" in got
    assert "중국 박물관 엑스포" in got
    assert len(han.findall(got)) < 2

    visit = (
        "The 十一th China Museum Expo will take place from November 十三 to 十六, "
        "二千零二十六, at the Hainan International Convention and Exhibition Center."
    )
    got = polish_spoken(visit, "en")
    assert "11th" in got and "13" in got and "16" in got and "2026" in got
    assert han.search(got) is None

    ko_visit = "제十一회는 二千零二十六년 十一월 十三일부터 十六일까지 열립니다."
    got = polish_spoken(ko_visit, "ko")
    assert "제11회" in got and "2026" in got and "11월" in got
    assert han.search(got) is None

    ja = "展示会では、ただ博博会を紹介します。"
    got = polish_spoken(ja, "ja")
    assert "博博会" not in got and "中国博物館エキスポ" in got and "紹介" in got

    booth = "我在这里，就在宽创国际的展厅里。"
    assert polish_spoken(booth, "zh") == booth
    assert "/no_think" not in polish_spoken("OK.\n/no_think", "en")
    assert polish_spoken("예를 들어, 金刚경입니다.", "ko") == "예를 들어, 금강경입니다."


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
