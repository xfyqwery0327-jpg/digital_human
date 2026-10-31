# -*- coding: utf-8 -*-
"""新闻不连着重复，天气认地名，博博会资料和专名读音。"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server9"))

from persona.registry import _parse_persona, _yaml_load
from config import PERSONAS_DIR
from rag.bobohui import is_bobohui_question, merge_bobohui
from rag.dialogue import RealtimeMemory, rule_intent
from rag.scope import is_other_exhibition_query
from services.local_feed import (
    anchor_script, extract_weather_city, news_card, next_beijing_clock, select_news,
)
from services.screen_hooks import headline_event


def _persona(pid):
    return _parse_persona(_yaml_load(PERSONAS_DIR / ("%s.yaml" % pid)), pid)


def test_kumarajiva_prompt_marks_qiuci():
    text = _persona("kumarajiva").system_prompt
    assert "龟兹读作 qiū cí" in text
    assert "专有名词读音" in text
    fuhao = _persona("fuhao").system_prompt
    assert "妇好读作 fù hǎo" in fuhao


def test_weather_city_and_default_cases():
    assert extract_weather_city("今天天气怎么样") == ""
    assert extract_weather_city("今天上海天气") == "上海"
    assert extract_weather_city("上海今天天气") == "上海"
    assert extract_weather_city("北京天气怎么样") == "北京"
    assert extract_weather_city("会下雨吗") == ""
    assert rule_intent("今天上海天气") == "weather"
    assert rule_intent("上海今天天气") == "weather"


def test_news_followup_uses_last_turn():
    assert rule_intent("最近有什么新闻") == "news"
    assert rule_intent("再来一条") == "news"
    mem = RealtimeMemory()
    mem.last_realtime_kind = "news"
    assert rule_intent("了解更多", mem) == "news"
    mem.last_realtime_kind = ""
    assert rule_intent("了解更多", mem) is None
    assert rule_intent("《睡莲》今天天气怎么样") is None


def test_news_pick_skips_recent_keys():
    items = [
        {"uniquekey": "a", "title": "甲", "plain": "甲的正文。"},
        {"uniquekey": "b", "title": "乙", "plain": "乙的正文。"},
        {"uniquekey": "c", "title": "丙", "plain": "丙的正文。"},
    ]
    first, cursor = select_news(items, set(), 0)
    second, cursor = select_news(items, {first["uniquekey"]}, cursor)
    assert first["uniquekey"] != second["uniquekey"]
    third, _cursor = select_news(items, {"a", "b", "c"}, 0)
    assert third["uniquekey"] in {"a", "b", "c"}


def test_anchor_script_stays_between_200_and_250():
    sentences = [
        "湖人本场第一节还能跟上节奏，之后攻守两端都出现了明显波动",
        "东契奇独自砍下高分，但后三节受到了针对性限制",
        "内线篮板和防守还算合格，外线投射却连续打铁",
        "替补席失误增多，轮换球员找不到稳定的进攻点",
        "全场最终被对手完成逆转，分差拉大到两位数",
        "公开报道认为问题出在球权过于集中，队友没有接上",
    ]
    body = "。".join(sentences) + "。"
    text = anchor_script("湖人遭勇士逆转", "<p>%s</p>" % body)
    assert text.startswith("那现在为你播报下今日头条新闻。")
    assert 200 <= len(text) <= 250
    assert "http" not in text and "<" not in text
    assert text.endswith("。")


def test_news_speech_keeps_link_off_the_mouth():
    html = "<p>详情见 https://news.example.com/a123 。湖人这一场输了，分差拉到两位数。</p>"
    text = anchor_script("今日头条", html)
    assert text.startswith("那现在为你播报下今日头条新闻。")
    assert "http" not in text and "example.com" not in text
    card = news_card({
        "text": text,
        "url": "https://news.example.com/a123",
        "thumbnail_pic_s": "http://img.example.com/a.jpg",
    })
    assert card["url"] == "https://news.example.com/a123"
    assert card["thumbnail_pic_s"].startswith("http://img.example.com/")


def test_cache_clock_is_beijing():
    sh = timezone(timedelta(hours=8))
    morning = datetime(2026, 10, 31, 5, 30, tzinfo=sh)
    assert next_beijing_clock(6, morning) == datetime(2026, 10, 31, 6, 0, tzinfo=sh)
    evening = datetime(2026, 10, 31, 13, 0, tzinfo=sh)
    assert next_beijing_clock(6, evening) == datetime(2026, 11, 1, 6, 0, tzinfo=sh)
    assert next_beijing_clock(0, evening) == datetime(2026, 11, 1, 0, 0, tzinfo=sh)


def test_bobohui_is_this_expo_not_another_show():
    persona = _persona("monet")
    assert is_bobohui_question("第十一届博博会什么时候开")
    assert not is_other_exhibition_query("第十一届博博会什么时候开", persona)
    assert is_other_exhibition_query("介绍一下别的博览会", persona)
    hits = merge_bobohui("博博会在哪里举办", [], 0.35)
    assert hits and "宽创国际" in hits[0]["text"]
    assert hits[0]["score"] >= 0.35


def test_headline_event_uses_list_field_names():
    event = headline_event("标题", "http://img/a.jpg", "https://news.example/a", "s1", "monet")
    assert event["url"] == "https://news.example/a"
    assert event["thumbnail_pic_s"] == "http://img/a.jpg"
    assert event["image_url"] == event["thumbnail_pic_s"]
