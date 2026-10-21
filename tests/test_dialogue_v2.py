# -*- coding: utf-8 -*-
"""第一期：口播与屏幕字段分开，天气不编数字，展品名只从标题和别名来。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server9"))

from persona.models import PersonaConfig
from persona.registry import _parse_persona, _yaml_load
from rag.dialogue import (
    FOREIGN_REPLY, RealtimeMemory, exhibit_names, is_foreign_question,
    note_hall, note_realtime, pack_done, realtime_reply, rule_intent,
    suggest_hooks, titles_from_items, visit_fact_reply,
)
from rag.scope import is_other_exhibition_query, refuse_text
from config import PERSONAS_DIR


def _persona(pid):
    return _parse_persona(_yaml_load(PERSONAS_DIR / ("%s.yaml" % pid)), pid)


def test_weather_second_turn_has_no_digits():
    persona = _persona("monet")
    mem = RealtimeMemory()
    note_realtime(mem)
    first = realtime_reply(mem, persona)
    note_realtime(mem)
    second = realtime_reply(mem, persona)
    assert "二十二" not in first and "22" not in first
    assert "就到这里" in second
    assert "{" not in first and "reply_text" not in second
    hooks = suggest_hooks("weather", mem, persona)
    assert len(hooks) == 1
    assert hooks[0] not in first and hooks[0] not in second


def test_weather_fact_then_back_to_hall():
    persona = _persona("monet")
    mem = RealtimeMemory()
    note_realtime(mem)
    fact = "上海今天阴，十八到二十四摄氏度。"
    first = realtime_reply(mem, persona, fact)
    assert first.startswith(fact)
    assert "今天更适合在厅里看光和颜色" in first
    assert "场馆里不播报" not in first
    assert "{" not in first
    note_realtime(mem)
    second = realtime_reply(mem, persona, fact)
    assert "就到这里" in second
    assert "十八" not in second
    fuhao = _persona("fuhao")
    mem2 = RealtimeMemory()
    note_realtime(mem2)
    pulled = realtime_reply(mem2, fuhao, "贵阳今天有雨，十六到二十摄氏度。")
    assert "商代" in pulled
    assert "睡莲" not in pulled


def test_fuhao_and_kumarajiva_hooks_match_their_halls():
    banned = ("睡莲", "莫奈", "印象派", "日出")
    expect = {
        "fuhao": {
            "exhibit": "甲骨",
            "person": "领兵",
            "visit": "开馆",
            "chitchat": "甲骨",
            "weather": "甲骨",
            "name": "青铜钺",
        },
        "kumarajiva": {
            "exhibit": "舍利盒",
            "person": "长安",
            "visit": "在哪里",
            "chitchat": "译经",
            "weather": "译经",
            "name": "舍利盒",
        },
    }
    for pid, tokens in expect.items():
        persona = _persona(pid)
        for intent, token in tokens.items():
            if intent == "name":
                continue
            hooks = suggest_hooks(intent, RealtimeMemory(), persona)
            assert hooks
            assert len(hooks) <= 2
            assert all(word not in hook for hook in hooks for word in banned)
            assert any(token in hook for hook in hooks)
        named = suggest_hooks("exhibit", RealtimeMemory(), persona, [tokens["name"]])
        assert tokens["name"] in named[0]
        assert named[0].startswith("想再")
        assert all(word not in named[0] for word in banned)


def test_feed_lines_skip_missing_numbers():
    from services.local_feed import news_line, weather_line
    line = weather_line("上海市", 3, 18.2, 24.4)
    assert line.startswith("上海今天阴")
    assert "十八" in line and "二十四" in line
    assert "18" not in line
    assert weather_line("上海", 3, None, 24) == ""
    news = news_line("上海", "地铁十号线北延有了官方回应")
    assert news.startswith("上海这边有一条消息")
    assert news.endswith("。")
    assert "{" not in news
    clipped = news_line("上海", "上海深挖城市文旅资源，拓展新场景新体验——处处有精彩，时时有 ...")
    assert "时时有" not in clipped
    assert clipped.endswith("。")


def test_streak_resets_only_on_hall():
    persona = _persona("monet")
    mem = RealtimeMemory()
    note_realtime(mem)
    note_realtime(mem)
    note_hall(mem)
    note_realtime(mem)
    assert "就到这里" not in realtime_reply(mem, persona)


def test_exhibit_alias_and_title():
    persona = _persona("monet")
    names = exhibit_names(
        "我想听睡莲，还有日出印象。",
        titles_from_items([{"title": "《日出·印象》是哪一年画的？", "text": "提到《星月夜》"}]),
        persona.exhibit_aliases)
    assert "睡莲" in names
    assert "日出·印象" in names
    assert "星月夜" not in names


def test_visit_fact_uses_hall_clock():
    persona = _persona("monet")
    text = visit_fact_reply("这个展会几点开门", persona)
    assert "浪漫与星辰" in text
    assert "十二月三十一" not in text
    assert visit_fact_reply("睡莲在哪画的", persona) == ""


def test_fuhao_does_not_recommend_monet():
    persona = _persona("fuhao")
    assert is_other_exhibition_query("介绍一下莫奈", persona)
    spoken = refuse_text(persona)
    assert "睡莲" not in spoken
    mem = RealtimeMemory()
    note_realtime(mem)
    reply = realtime_reply(mem, persona)
    hooks = suggest_hooks("weather", mem, persona)
    assert "睡莲" not in reply
    assert all("睡莲" not in h for h in hooks)


def test_headline_jpg_goes_to_screen_not_speech():
    from services.local_feed import headline_from_item
    from services.screen_hooks import headline_event, request_headline_image
    pack = headline_from_item({
        "title": "西北枢纽集体起飞",
        "img": {"u": "https://n.sinaimg.cn/news/a.jpg"},
        "url": "https://news.sina.com.cn/a",
    })
    assert pack["image_url"].endswith(".jpg")
    assert pack["text"].startswith("今天的头条是")
    assert "jpg" not in pack["text"] and "http" not in pack["text"]
    event = headline_event(pack["title"], pack["image_url"], pack["link"], "s1", "monet")
    assert event["type"] == "headline" and event["action"] == "show"
    assert event["image_url"].endswith(".jpg")
    skipped = headline_from_item({"title": "这条头条没有jpg原图", "img": {"u": "https://x/a.png"}})
    assert skipped["image_url"] == "" and skipped["text"].startswith("今天的头条是")
    reserved = request_headline_image(pack["title"], pack["image_url"], pack["link"], "monet", "s1")
    assert reserved["code"] == "not_configured"


def test_v1_chinese_scenes_stay_plain_speech():
    """v1 的六个中文场景：口播无字段名，按钮和展品名分开。"""
    persona = _persona("monet")
    mem = RealtimeMemory()
    note_realtime(mem)
    weather = realtime_reply(mem, persona, "上海今天晴，二十二到二十八摄氏度。")
    weather_hooks = suggest_hooks("weather", mem, persona)
    assert "摄氏度" in weather and "今天更适合在厅里看光和颜色" in weather
    assert len(weather_hooks) == 1 and "睡莲" in weather_hooks[0]
    assert weather_hooks[0] not in weather and "{" not in weather

    mem2 = RealtimeMemory()
    note_realtime(mem2)
    news = realtime_reply(mem2, persona, "上海这边有一条消息，博物馆新展览开幕。")
    news_done = pack_done("news", suggest_hooks("news", mem2, persona), [], False, [], [], None)
    assert "博物馆" in news and "今天更适合" in news
    assert news_done["lang"] == "zh" and len(news_done["hook_suggest"]) == 1
    assert news_done["exhibit_names"] == []

    spoken = "睡莲是我在吉维尼画的，着重表现水面光影。"
    names = exhibit_names(spoken, [], persona.exhibit_aliases)
    exhibit_hooks = suggest_hooks("exhibit", RealtimeMemory(), persona, names)
    assert "睡莲" in names and len(exhibit_hooks) <= 2
    assert all(hook not in spoken for hook in exhibit_hooks)

    life = "我一生都在捕捉光线的变化。"
    life_hooks = suggest_hooks("person", RealtimeMemory(), persona)
    assert any("生平" in hook or "鲁昂" in hook for hook in life_hooks)
    assert all(hook not in life for hook in life_hooks)

    hours = visit_fact_reply("这个展会几点开门", persona)
    visit_hooks = suggest_hooks("visit", RealtimeMemory(), persona)
    assert "浪漫与星辰" in hours and "十二月三十一" not in hours
    assert visit_hooks and len(visit_hooks) <= 2

    hello_hooks = suggest_hooks("chitchat", RealtimeMemory(), persona)
    hello_done = pack_done("chitchat", hello_hooks, [], False, [], [], None)
    assert len(hello_done["hook_suggest"]) == 1
    assert hello_done["exhibit_names"] == [] and hello_done["lang"] == "zh"

    for question in (
        "Tell me about Water Lilies",
        "モネの生涯を教えてください",
        "인상, 해돋이에 대해 알려주세요",
    ):
        assert is_foreign_question(question)
    assert "Water" not in FOREIGN_REPLY and "モ" not in FOREIGN_REPLY


def test_foreign_and_hook_limit():
    assert is_foreign_question("Tell me about Water Lilies")
    assert is_foreign_question("モネについて教えて")
    assert not is_foreign_question("介绍一下睡莲")
    assert "{" not in FOREIGN_REPLY
    done = pack_done("weather", ["甲", "乙"], ["睡莲"], False, [], [], None)
    assert done["lang"] == "zh"
    assert done["hook_suggest"] == ["甲"]
    assert rule_intent("今天天气怎么样") == "weather"
    assert rule_intent("《睡莲》今天天气怎么样") is None
