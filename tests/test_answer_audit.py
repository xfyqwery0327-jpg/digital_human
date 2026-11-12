# -*- coding: utf-8 -*-
"""检查固定口播和句尾问话：串答、不该报的位置、妇好被说成作品。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server9"))

from persona.registry import _parse_persona, _yaml_load
from rag.bobohui import HOOK_TOPIC, spoken_closings, strip_unasked_here
from rag.dialogue import (
    FOREIGN_REPLY, RealtimeMemory, append_spoken_followup, note_realtime,
    realtime_reply, spoken_followup, visit_fact_reply,
)
from rag.scope import refuse_text
from rag.transcript import match_transcript, question_score
from robot.scripted import CAPABILITY_REPLY, capability_reply
from config import PERSONAS_DIR

BANNED = (
    "在那里", "我听说", "这场展有我的", "浪漫与星辰", "贵州省博物馆",
    "龟兹博物馆", "七十八元", "四十九元",
)
PRESENCE_OPEN = ("我在这里", "我就在这里", "我此刻就在", "我就在博博会", "人就在宽创", "就在博博会", "就在宽创")


def _persona(pid):
    return _parse_persona(_yaml_load(PERSONAS_DIR / ("%s.yaml" % pid)), pid)


def _rows(pid):
    path = ROOT / "kb" / pid / "transcript.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _hook_count(text, intent, persona=None):
    from rag.dialogue import spoken_topic
    pool = spoken_closings(spoken_topic(intent, persona))
    return sum(1 for line in pool if line and line in (text or ""))


def audit():
    issues = []

    def add(kind, who, detail):
        issues.append((kind, who, detail))

    for pid in ("monet", "fuhao", "kumarajiva"):
        persona = _persona(pid)
        rows = _rows(pid)
        for row in rows:
            question = row["question"]
            answer = row["answer"]
            hit = match_transcript(question, rows)
            if hit != answer:
                add("串答", pid, "问「%s」对上了别的回答：%s" % (question, hit[:36]))
            for word in BANNED:
                if word in answer:
                    add("禁词", pid, "「%s」里出现「%s」" % (question[:24], word))
            cleaned = strip_unasked_here(question, answer)
            if any(cleaned.startswith(prefix) for prefix in PRESENCE_OPEN):
                add("报位置", pid, "没问位置仍以位置开头：%s" % cleaned[:40])
            spoken = append_spoken_followup(cleaned, "knowledge", persona)
            already_asking = answer.rstrip().endswith("？") or answer.rstrip().endswith("?")
            hooks = _hook_count(spoken, "knowledge", persona)
            if already_asking:
                extra = spoken[len(cleaned):] if spoken.startswith(cleaned) else spoken
                if "？" in extra or "?" in extra:
                    add("两问", pid, "原文已是问句，又接了问句：%s" % question[:24])
            elif hooks != 1:
                add("句尾", pid, "「%s」句尾问话 %s 句" % (question[:24], hooks))
            if pid != "monet" and "我的作品" in spoken[len(cleaned):]:
                add("人称", pid, "讲完「%s」却邀请继续问我的作品" % question[:24])

        for i, left in enumerate(rows):
            for right in rows[i + 1:]:
                score = question_score(left["question"], right["question"])
                if score >= 0.72:
                    add("易混", pid, "%.2f 「%s」≈「%s」" % (
                        score, left["question"][:22], right["question"][:22]))

        samples = (
            ("睡莲是在哪里画的", "我在这里，就在博博会宽创站台中。睡莲是我在吉维尼画的。"),
            ("介绍一下妇好鸮尊", "我在这里。就在博博会宽创展厅里。鸮尊是酒器。"),
            ("你打过哪些仗", "我就在这里，第十一届博博会的宽创国际展厅里。我统兵出征。"),
        )
        for question, answer in samples:
            cleaned = strip_unasked_here(question, answer)
            if any(token in cleaned for token in ("我在这里", "就在博博会", "就在宽创", "宽创站台", "宽创展厅")):
                add("报位置", pid, "问「%s」后仍剩：%s" % (question, cleaned))

    persona = _persona("monet")
    mem = RealtimeMemory()
    note_realtime(mem)
    weather = realtime_reply(mem, persona, "海口今天多云，二十六到三十一摄氏度。", kind="weather")
    again = append_spoken_followup(weather, "weather")
    if again != weather:
        add("句尾", "天气", "事实后面已经问过，又接了一句")
    if _hook_count(weather, "weather") != 1:
        add("句尾", "天气", "天气口播里的问话不是一句")
    if any(word in weather for word in BANNED):
        add("禁词", "天气", weather)

    for question, intent in (
        ("帮我预约门票", "ticket_service"),
        ("停止播放", "command"),
    ):
        body = "好的。"
        spoken = append_spoken_followup(body, intent)
        if spoken != body:
            add("句尾", intent, "不该加问话，结果是：%s" % spoken)
    refused = append_spoken_followup(refuse_text(persona), "knowledge")
    if refused != refuse_text(persona):
        add("句尾", "拒答", "拒答后又接了问话")
    foreign = append_spoken_followup(FOREIGN_REPLY, "chitchat")
    if _hook_count(foreign, "chitchat") != 1:
        add("句尾", "外文", "外文说明的问话不是一句：%s" % foreign)

    functions = capability_reply("莫奈，你还有什么功能？")
    if functions != CAPABILITY_REPLY:
        add("功能", "monet", "五项功能没有按原稿")
    for word in ("作品", "睡莲", "日出", "印象"):
        if word in functions:
            add("功能", "monet", "五项功能里出现「%s」" % word)
    doubled = append_spoken_followup(functions, "chitchat")
    if doubled != functions:
        add("两问", "功能", "五项功能末尾已有「您想先试哪一项呢？」，通用追问还会再接一句")

    visit = visit_fact_reply("展览在哪里举办", persona)
    asked = append_spoken_followup(visit, "visit")
    if not asked.startswith(visit) or _hook_count(asked, "visit") != 1:
        add("句尾", "参观", asked)
    if "我在这里" not in visit and "我就在博博会" not in visit:
        add("报位置", "参观", "问展览在哪时没有说明人在现场")

    return issues


def test_answer_audit_lists_remaining_problems():
    """固定回答不再串人称、不再残留单独的位置句、不再连着两句问话。"""
    issues = audit()
    for kind, who, detail in issues:
        print("[%s] %s %s" % (kind, who, detail))
    assert not issues, "问题 %s 条" % len(issues)


if __name__ == "__main__":
    test_answer_audit_lists_remaining_problems()
