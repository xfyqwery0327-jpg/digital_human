# -*- coding: utf-8 -*-
"""现场看天气、新闻，以及三个人物介绍博博会。需要外网；介绍博博会还需要本机大模型。"""
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server9"))

from persona.registry import _parse_persona, _yaml_load
from persona.runtime import bind
from config import LLM_URL, PERSONAS_DIR
from llm.client import clean_full_answer, llm_collect
from llm.persona import system_prompt
from rag.bobohui import TEXT, merge_bobohui
from rag.dialogue import RealtimeMemory, prepare_realtime
from rag.search import works_answer_hint
from rag.visit import visit_hint

OLD = ("浪漫与星辰", "贵州省博物馆", "这场商代", "龟兹博物馆", "七十八元", "四十九元", "这场展有我的")
ASK = "介绍一下第十一届博博会"


def _persona(pid):
    return _parse_persona(_yaml_load(PERSONAS_DIR / ("%s.yaml" % pid)), pid)


def _llm_up() -> bool:
    try:
        host, port = LLM_URL.split("://", 1)[1].split("/", 1)[0].split(":")
        socket.create_connection((host, int(port)), 2).close()
        return True
    except Exception:
        return False


def _print_turn(title, turn):
    bundle = turn.get("bundle") or {}
    text = turn.get("reply") or ""
    print("【%s】字数=%s" % (title, len(text)))
    print(text)
    if turn.get("kind") == "news":
        print("标题:", bundle.get("title") or "")
        print("url:", bundle.get("url") or "")
        print("thumbnail_pic_s:", bundle.get("thumbnail_pic_s") or "")
    bad = [word for word in OLD if word in text]
    print("旧展会残留:", "无" if not bad else "、".join(bad))
    if turn.get("kind") == "news":
        print("口播含链接:", "有" if "http" in text.lower() else "无")
    print()


def bobohui_answer(persona):
    hits = merge_bobohui(ASK, [], 0.2)
    ref = "\n".join("- %s" % h["text"] for h in hits)
    user_prompt = "【参考资料】\n%s\n\n【问题】%s" % (ref, ASK)
    user_prompt += (
        "\n\n展会只介绍博博会，不要介绍原先的展，也不要报原先的门票价格。"
        "问到本人作品或经历时只讲事情本身，不要说这场展有我的作品。"
        "你人就在博博会现场的宽创展厅里，这是事实，但不要主动报位置。不要说在那里，不要说我听说。只有观众问你在哪里、展厅在哪时，才说我就在这里。讲作品时不要先说我在这里。\n"
    )
    user_prompt += works_answer_hint(ASK, persona)
    user_prompt += visit_hint(ASK, persona)
    user_prompt += "/no_think"
    messages = [
        {"role": "system", "content": system_prompt(persona)},
        {"role": "user", "content": user_prompt},
    ]
    with bind(persona):
        raw = llm_collect(messages, max_tokens=300)
        return clean_full_answer(raw)


def main():
    persona = _persona("monet")
    mem = RealtimeMemory()
    _print_turn("默认天气", prepare_realtime("今天天气怎么样", mem, persona))
    mem2 = RealtimeMemory()
    _print_turn("上海天气", prepare_realtime("今天上海天气", mem2, persona))
    mem3 = RealtimeMemory()
    first = prepare_realtime("最近有什么新闻", mem3, persona)
    _print_turn("新闻第一条", first)
    second = prepare_realtime("再来一条", mem3, persona)
    _print_turn("新闻再来一条", second)
    same = (first.get("bundle") or {}).get("uniquekey") == (second.get("bundle") or {}).get("uniquekey")
    print("两条新闻是否同一条:", "是" if same else "否")
    print()
    if not _llm_up():
        print("本机大模型未开，三个人物的博博会介绍没有口播。")
        print("会注入的资料是：")
        print(TEXT)
        return 1
    for pid in ("monet", "fuhao", "kumarajiva"):
        p = _persona(pid)
        text = bobohui_answer(p)
        print("【%s 介绍博博会】字数=%s" % (p.display_name, len(text)))
        print(text)
        bad = [word for word in OLD if word in text]
        print("旧展会残留:", "无" if not bad else "、".join(bad))
        print("提到博博会:", "是" if "博博会" in text else "否")
        print("说了在那里:", "是" if "在那里" in text else "否")
        print("说了我听说:", "是" if "我听说" in text else "否")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
