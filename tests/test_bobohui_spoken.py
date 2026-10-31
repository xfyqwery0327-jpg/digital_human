# -*- coding: utf-8 -*-
"""三个人物的固定回句：只介绍博博会，不报原先展和门票，钩子只有展台这一句。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server9"))

from persona.registry import _parse_persona, _yaml_load
from config import PERSONAS_DIR
from rag.bobohui import HOOK_TOPIC, hall_hook
from rag.dialogue import RealtimeMemory, note_realtime, realtime_reply, suggest_hooks, visit_fact_reply

OLD_SHOW = ("浪漫与星辰", "贵州省博物馆", "七十八元", "四十九元", "这场商代的展", "龟兹博物馆", "这场展有我的")
INTENTS = ("exhibit", "person", "visit", "chitchat", "weather", "news")
QUESTIONS = (
    ("天气", "今天天气怎么样"),
    ("新闻", "再来一条新闻"),
    ("作品", "介绍一下我的作品"),
    ("门票", "门票多少钱"),
    ("时间", "这个展会几点开门"),
    ("地点", "展览在哪里举办"),
)


def _persona(pid):
    return _parse_persona(_yaml_load(PERSONAS_DIR / ("%s.yaml" % pid)), pid)


def spoken_lines(persona):
    """固定回句，不调用大模型。作品问题不走门票和展期。"""
    mem = RealtimeMemory()
    note_realtime(mem)
    weather = realtime_reply(mem, persona, "海口今天多云，二十六到三十一摄氏度。", kind="weather")
    news = realtime_reply(mem, persona, "那现在为你播报下今日头条新闻。有一条消息。", kind="news")
    hooks = {intent: suggest_hooks(intent, RealtimeMemory(), persona) for intent in INTENTS}
    return {
        "天气口播": weather,
        "新闻口播": news,
        "门票": visit_fact_reply("门票多少钱", persona),
        "开放时间": visit_fact_reply("这个展会几点开门", persona),
        "地点": visit_fact_reply("展览在哪里举办", persona),
        "作品问题": visit_fact_reply("介绍一下我的作品", persona),
        "钩子": hooks,
    }


def test_three_personas_only_speak_bobohui():
    samples = {}
    for pid in ("monet", "fuhao", "kumarajiva"):
        persona = _persona(pid)
        lines = spoken_lines(persona)
        samples[pid] = lines
        prompt = persona.system_prompt
        assert "不要说这场展有我的" in prompt or "不要说这场展有我的作品" in prompt or "不要说这场展有我的器物" in prompt
        assert "原先" in prompt and "博博会" in prompt
        assert "浪漫与星辰" not in (persona.ticket_info.get("price") or "")
        for label in ("天气口播", "新闻口播", "开放时间", "地点"):
            text = lines[label]
            assert "博博会" in text, (pid, label, text)
            assert all(word not in text for word in OLD_SHOW), (pid, label, text)
        assert all(word not in lines["门票"] for word in OLD_SHOW)
        assert lines["作品问题"] == ""
        assert lines["门票"] == "原先的门票价格我不介绍。"
        assert "我就在博博会现场的宽创展厅里" in lines["开放时间"]
        assert "在那里" not in lines["开放时间"] and "我听说" not in lines["开放时间"]
        assert "不要说在那里" in prompt and "不要说我听说" in prompt
        assert lines["天气口播"].endswith(hall_hook("天气"))
        assert lines["新闻口播"].endswith(hall_hook("新闻"))
        assert "http" not in lines["新闻口播"]
        for intent, hooks in lines["钩子"].items():
            assert hooks == [hall_hook(HOOK_TOPIC[intent])], (pid, intent, hooks)
            assert all(word not in hooks[0] for word in ("睡莲", "甲骨", "舍利盒", "开馆"))
    return samples


if __name__ == "__main__":
    got = test_three_personas_only_speak_bobohui()
    for pid, lines in got.items():
        print("==", pid)
        for key, value in lines.items():
            if key == "钩子":
                print("  钩子:", value["weather"][0])
            else:
                print("  %s: %s" % (key, value))
    print("ok")
