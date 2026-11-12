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
    append_spoken_followup, spoken_followup, suggest_hooks, titles_from_items, visit_fact_reply,
    is_noise_question, is_relation_question,     is_family_question, family_reply,
    is_life_question, life_reply, relation_reply,
)
from rag.scope import is_other_exhibition_query, refuse_text
from config import PERSONAS_DIR


def _persona(pid):
    return _parse_persona(_yaml_load(PERSONAS_DIR / ("%s.yaml" % pid)), pid)


def test_noise_relation_and_family_do_not_invent():
    assert is_noise_question("六")
    assert is_noise_question("12")
    assert not is_noise_question("今天什么天气啊")
    assert is_relation_question("呃你和泪落案是什么关系啊怎么认识的")
    assert not is_relation_question("你和《睡莲》是什么关系")
    assert is_family_question("你有几个孩子")
    assert is_family_question("你有几个老婆")
    assert is_family_question("你结婚了吗")
    assert is_family_question("你结过几次婚")
    assert is_family_question("莫奈结婚")
    assert not is_family_question("《卡美尔》画的是谁")
    hits = [{
        "text": (
            "结识卡美尔。1867年7月，卡美尔生下他们的长子——简。"
            "1870年6月28日，与卡美尔结婚。3月，卡美尔生下第二个儿子米契尔后病情恶化。"
            "莫奈有几个儿子？1878年3月17日，他们有了另一个儿子，迈克尔（Michael）。"
            "1870年，莫奈与卡米尔结婚。1879年，莫奈夫人死于盆腔癌。"
            "1892年，同年莫奈与爱丽丝结婚。婚姻\n"
            "欧希德宣告破产，莫奈与欧希德的妻子爱丽丝互通款曲。"
        ),
    }]
    child = family_reply("你有几个孩子", hits)
    wife = family_reply("你有几个老婆", hits)
    assert "让" in child and "简" not in child and "米契尔" in child
    assert "迈克尔" not in child and "病情" not in child and "？" not in child
    assert "没有女儿" not in child
    household_hits = hits + [{
        "text": "1881年12月，莫奈与爱丽丝及他们共同抚养的8个孩子搬离维特伊。",
    }]
    girl = family_reply("你有女儿吗", household_hits)
    assert girl.startswith("我没有女儿。")
    assert "让" in girl and "米契尔" in girl and "八个" in girl
    assert "8个" not in girl
    raised = family_reply("你有几个孩子", household_hits)
    assert "没有女儿" not in raised and "八个" in raised
    assert "卡美尔" in wife and "爱丽丝" in wife and "结了婚" in wife
    assert "莫奈我" not in wife and "癌" not in wife and "死于" not in wife
    married = family_reply("你结过几次婚", hits)
    assert married == wife
    long_bio = ("父亲" * 180) + "1870年，莫奈与卡米尔结婚。后来莫奈和爱丽丝在1892年结婚。"
    twice = family_reply("莫奈结婚", [{"text": long_bio}])
    assert "卡美尔" in twice and "爱丽丝" in twice and "癌" not in twice
    assert family_reply("你有几个孩子", [{"text": "今天展出睡莲。"}]) == ""


def test_childhood_and_life_stay_on_monet():
    assert is_life_question("你小时候在哪里长大")
    assert is_life_question("讲讲你的生平")
    assert is_life_question("你这一生有哪些经历")
    assert not is_life_question("《日出·印象》的创作经历")
    hits = [{
        "text": (
            "1840年11月14日出生于巴黎拉菲特街。"
            "1845年，全家移居勒阿弗尔，莫奈在这里的度过了童年。"
            "1867年7月，卡美尔生下他们的长子——简。"
            "1870年6月28日，与卡美尔结婚。"
            "1892年，同年莫奈与爱丽丝结婚。"
            "1878年3月17日，他们有了另一个儿子，迈克尔（Michael）。"
            "晚年我定居吉维尼。"
            "皮埃尔·杜蒙的生平。巴布罗·毕加索有哪些重要经历。"
        ),
    }]
    grown = life_reply("你小时候在哪里长大", hits)
    assert "勒阿弗尔" in grown and "巴黎长大" not in grown
    assert "卡美尔" not in grown and "杜蒙" not in grown
    life = life_reply("讲讲你的生平", hits)
    assert "勒阿弗尔" in life and "卡美尔" in life and "爱丽丝" in life
    assert "让" in life and "简" not in life and "米契尔" in life
    assert "吉维尼" in life
    assert "杜蒙" not in life and "毕加索" not in life


def test_birth_death_parents_and_specific_wives():
    assert not is_relation_question("你和卡米尔是什么关系")
    assert is_relation_question("你和马奈是什么关系")
    assert is_family_question("卡米尔是谁")
    assert is_family_question("你太太是谁")
    assert is_family_question("你父亲是谁")
    assert not is_family_question("《卡美尔》画的是谁")
    assert is_life_question("你是在哪长大的")
    assert is_life_question("你出生在哪里")
    assert is_life_question("你什么时候去世的")
    assert is_life_question("介绍一下你的人生")
    hits = [{
        "text": (
            "克劳德·莫奈（Oscar-Claude Monet，1840年11月14日－1926年12月5日），全名奥斯卡-克劳德·莫奈，出生于巴黎。"
            "1845年，全家移居勒阿弗尔，莫奈在这里的度过了童年。"
            "1867年7月，卡美尔生下他们的长子——简。"
            "1870年6月28日，与卡美尔结婚。"
            "1892年，同年莫奈与爱丽丝结婚。"
            "1878年3月17日，他们有了另一个儿子，迈克尔（Michael）。"
            "1881年12月，莫奈与爱丽丝及他们共同抚养的8个孩子搬离维特伊。"
            "父亲库路多·阿多洛夫·莫奈是一位商人，在阿弗尔与他的姐夫合伙经营着一家仪器店。莫奈5岁时来到父亲的身边。"
            "1861年，应召赴阿尔及利亚服兵役。1862年，因病退役。"
            "普法战争爆发，莫奈只身到伦敦。"
            "1926年，莫奈已几乎完全失明。"
            "1926年，莫奈12月6日病逝于吉维尼家中，享年86岁。"
            "晚年我定居吉维尼。"
            "莱热《母亲》的生平资料有哪些？"
        ),
    }]
    born = life_reply("你出生在哪里", hits)
    assert "十一月十四日" in born and "巴黎" in born and "勒阿弗尔" not in born
    grown = life_reply("你是在哪长大的", hits)
    assert "勒阿弗尔" in grown and "巴黎长大" not in grown
    died = life_reply("你什么时候去世的", hits)
    assert "十二月五日" in died and "六日" not in died
    assert "吉维尼" in died and "八十六" in died
    assert family_reply("你和卡米尔是什么关系", hits) == "卡美尔是我的妻子。"
    assert family_reply("卡米尔是谁", hits) == "卡美尔是我的妻子。"
    assert "爱丽丝是我的妻子" in family_reply("爱丽丝是谁", hits)
    wives = family_reply("你太太是谁", hits)
    assert "卡美尔" in wives and "爱丽丝" in wives
    father = family_reply("你父亲是谁", hits)
    assert "商人" in father and "勒阿弗尔" in father and "五岁" in father
    assert "库路多" not in father
    assert "没有写我母亲" in family_reply("你母亲是谁", hits)
    assert "阿尔及利亚" in life_reply("你当过兵吗", hits)
    assert "一八六一年" in life_reply("你当过兵吗", hits)
    assert "伦敦" in life_reply("你去过英国吗", hits)
    assert "失明" in life_reply("你眼睛后来怎么了", hits)
    assert relation_reply("你和马奈是什么关系") == "我和马奈不是兄弟，是朋友。他比我大，画得比我实。"
    assert relation_reply("你和卡米尔是什么关系") == ""
    own = family_reply("八个孩子都是你亲生的吗", hits)
    assert own.startswith("亲生的是长子让和米契尔。")
    assert "八个" in own
    life = life_reply("介绍一下你的人生", hits)
    assert "勒阿弗尔" in life and "十二月五日" in life and "让" in life and "吉维尼" in life


def test_weather_second_turn_has_no_digits():
    persona = _persona("monet")
    mem = RealtimeMemory()
    note_realtime(mem)
    first = realtime_reply(mem, persona)
    note_realtime(mem)
    second = realtime_reply(mem, persona)
    assert "二十二" not in first and "22" not in first
    assert "博博会" in first and "博博会" in second
    assert "就到这里" not in second
    assert "{" not in first and "reply_text" not in second
    hooks = suggest_hooks("weather", mem, persona)
    assert hooks == ["了解完天气，您还想了解博博会或宽创展台的哪些内容呢？"]
    from rag.bobohui import spoken_closings
    assert any(line in first for line in spoken_closings("天气"))
    assert any(line in second for line in spoken_closings("天气"))


def test_weather_fact_then_back_to_hall():
    persona = _persona("monet")
    mem = RealtimeMemory()
    note_realtime(mem)
    fact = "上海今天阴，十八到二十四摄氏度。"
    first = realtime_reply(mem, persona, fact)
    assert first.startswith(fact)
    assert "博博会" in first
    assert "场馆里不播报" not in first
    assert "{" not in first
    note_realtime(mem)
    second = realtime_reply(mem, persona, fact)
    assert "博博会" in second
    assert "十八" in second
    fuhao = _persona("fuhao")
    mem2 = RealtimeMemory()
    note_realtime(mem2)
    pulled = realtime_reply(mem2, fuhao, "贵阳今天有雨，十六到二十摄氏度。")
    assert "博博会" in pulled
    assert "睡莲" not in pulled


def test_fuhao_and_kumarajiva_hooks_match_their_halls():
    banned = ("睡莲", "莫奈", "印象派", "日出", "甲骨", "舍利盒")
    expect = {
        "person": "了解完我的经历",
        "visit": "了解完展会安排",
        "chitchat": "了解完我的介绍",
        "weather": "了解完天气",
        "news": "了解完新闻",
    }
    exhibit = {"monet": "了解完我的作品", "fuhao": "了解完我的器物", "kumarajiva": "了解完我的译经"}
    knowledge = {"monet": "了解完我的作品", "fuhao": "了解完我的经历", "kumarajiva": "了解完我的经历"}
    tail = "您还想了解博博会或宽创展台的哪些内容呢？"
    for pid in ("fuhao", "kumarajiva", "monet"):
        persona = _persona(pid)
        heads = dict(expect)
        heads["exhibit"] = exhibit[pid]
        heads["knowledge"] = knowledge[pid]
        for intent, head in heads.items():
            hooks = suggest_hooks(intent, RealtimeMemory(), persona, ["青铜钺"])
            assert hooks == ["%s，%s" % (head, tail)], (pid, intent, hooks)
            assert all(word not in hook_text for hook_text in hooks for word in banned)


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
    assert "我就在博博会现场的宽创展厅里" in text and "海南国际会展中心" in text
    assert "在那里" not in text
    assert "浪漫与星辰" not in text and "七十八" not in text
    assert visit_fact_reply("睡莲在哪画的", persona) == ""
    assert "元" not in visit_fact_reply("门票多少钱", persona)


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
    assert "摄氏度" in weather and "博博会" in weather
    assert weather_hooks == ["了解完天气，您还想了解博博会或宽创展台的哪些内容呢？"]
    from rag.bobohui import spoken_closings
    assert any(weather.endswith(line) for line in spoken_closings("天气"))
    assert "{" not in weather

    mem2 = RealtimeMemory()
    note_realtime(mem2)
    news = realtime_reply(mem2, persona, "上海这边有一条消息，博物馆新展览开幕。", kind="news")
    news_done = pack_done("news", suggest_hooks("news", mem2, persona), [], False, [], [], None)
    assert "博物馆" in news and "博博会" in news
    assert news_done["lang"] == "zh" and len(news_done["hook_suggest"]) == 1
    assert news_done["exhibit_names"] == []

    spoken = "睡莲是我在吉维尼画的，着重表现水面光影。"
    names = exhibit_names(spoken, [], persona.exhibit_aliases)
    exhibit_hooks = suggest_hooks("exhibit", RealtimeMemory(), persona, names)
    assert "睡莲" in names and exhibit_hooks == ["了解完我的作品，您还想了解博博会或宽创展台的哪些内容呢？"]
    assert "这场展有我的作品" not in spoken

    life = "我一生都在捕捉光线的变化。"
    life_hooks = suggest_hooks("person", RealtimeMemory(), persona)
    assert life_hooks == ["了解完我的经历，您还想了解博博会或宽创展台的哪些内容呢？"]
    assert all(hook not in life for hook in life_hooks)

    hours = visit_fact_reply("这个展会几点开门", persona)
    visit_hooks = suggest_hooks("visit", RealtimeMemory(), persona)
    assert "博博会" in hours and "浪漫与星辰" not in hours and "十二月三十一" not in hours
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


def test_spoken_followup_after_answer():
    from rag.bobohui import spoken_closings
    body = "睡莲是我在吉维尼画的。"
    spoken = append_spoken_followup(body, "exhibit")
    assert spoken == body + spoken_followup("exhibit", body)
    assert spoken_followup("exhibit", body) in spoken_closings("我的作品")
    seen = {spoken_followup("exhibit", "口播%d。" % i) for i in range(24)}
    assert len(seen) >= 4
    weather = "海口今天多云。" + spoken_followup("weather", "海口今天多云。")
    assert append_spoken_followup(weather, "weather") == weather
    ticket = "好的，预约窗口已经打开。"
    assert append_spoken_followup(ticket, "ticket_service") == ticket
    assert append_spoken_followup("这个我还不太清楚，您可以换个问法。", "knowledge") == "这个我还不太清楚，您可以换个问法。"
    visit = visit_fact_reply("展览在哪里举办", _persona("monet"))
    asked = append_spoken_followup(visit, "visit")
    assert asked.startswith(visit)
    assert any(asked.endswith(line) for line in spoken_closings("展会安排"))


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
