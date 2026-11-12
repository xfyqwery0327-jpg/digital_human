# -*- coding: utf-8 -*-
"""观众实录问句对上时照搬回答，短问候不能误撞。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER9 = ROOT / "server9"
if str(SERVER9) not in sys.path:
    sys.path.insert(0, str(SERVER9))

from rag.transcript import (  # noqa: E402
    match_transcript, paragraphs_of_docx, parse_labeled_qa, parse_narrative, parse_transcript,
    question_score,
)

SRC = Path(r"C:\Users\BroadMesse\Desktop\全息柜人物数据导入")
FUHAO_Q = "“妇好”这两个字是什么意思呀？是你的名字吗？"
FUHAO_A = (
    "“妇”是我们那时候对王室夫人的称呼，“好”读“子”，是我出身的族名。"
    "合起来就是“来自子族的妇”。我死后庙号叫“辛”，所以我的孩子们也叫我“母辛”"
    "——你看那方大铜鼎上写的“司母辛”，就是他们为祭祀我而铸的。"
)


def _qa(name: str):
    folder = next(p for p in SRC.iterdir() if p.is_dir() and name in p.name)
    doc = next(p for p in folder.glob("*.docx") if "问答" in p.name)
    return parse_transcript(paragraphs_of_docx(doc))


def test_fuhao_name_question_is_copied():
    rows = _qa("妇好")
    assert len(rows) == 20
    hit = match_transcript("观众2：“妇好”这两个字是什么意思呀？是你的名字吗？", rows)
    assert hit == FUHAO_A
    assert match_transcript("嗨，妇好，这两个字是什么意思呀？是你的名字吗？", rows) == FUHAO_A
    assert "司母辛" in match_transcript("妇好这两个字是什么意思，是你的名字吗", rows)


def test_short_greeting_does_not_copy_a_long_answer():
    rows = _qa("妇好") + _qa("莫奈")
    assert match_transcript("你好", rows) == ""
    assert match_transcript("嗨", rows) == ""
    assert question_score("你好", FUHAO_Q) == 0.0


def test_each_persona_keeps_its_own_answer():
    monet = _qa("莫奈")
    luo = _qa("鸠摩罗什")
    assert len(monet) == 20 and len(luo) == 20
    impression = match_transcript("“印象派”这个名字是什么意思呀？是你们自己起的门派吗？", monet)
    assert "1874" in impression and "日出" in impression
    tongue = match_transcript("那个“舌灿莲花”是真的吗？你火化后舌头真的没烂？", luo)
    assert tongue.startswith("是真的")
    assert "鸠摩罗什：" not in tongue
    assert match_transcript("“印象派”这个名字是什么意思呀？", luo) == ""


def test_fuhao_booth_questions_are_copied():
    text = (ROOT / "kb" / "fuhao" / "sources" / "问答对.txt").read_text(encoding="utf-8")
    rows = parse_labeled_qa(text)
    assert len(rows) == 4
    oracle = match_transcript("听说您会占卜？", rows)
    assert oracle.startswith("对。古时凡事问天") and "三个关键词" in oracle
    battle = match_transcript("您作为将军，哪场战役是你印象最深刻的？", rows)
    assert battle.startswith("印象最深的是伐羌方") and "一万三千" in battle
    wine = match_transcript("听说您还会制酒？", rows)
    assert "鬯酒" in wine and "鸮尊盛酒" in wine
    owl = match_transcript("那可以详细讲讲这件妇好鸮尊吗？", rows)
    assert owl.startswith("鸮尊是我墓中代表性青铜酒礼器") and "妇好" in owl
    assert match_transcript("你打过最厉害的一仗是哪场？", rows) == ""
    assert "三个关键词" in match_transcript("你会占卜吗", rows)
    assert "一万三千" in match_transcript("哪场战役印象最深", rows)
    assert "鬯酒" in match_transcript("你还会酿酒吗", rows)
    assert match_transcript("详细讲讲鸮尊", rows).startswith("鸮尊是我墓中代表性青铜酒礼器")
    assert match_transcript("妇好鸮尊是什么", rows).startswith("鸮尊是我墓中代表性青铜酒礼器")
    full = json.loads((ROOT / "kb" / "fuhao" / "transcript.json").read_text(encoding="utf-8"))
    assert "热血沸腾" in match_transcript("你打过最厉害的一仗是哪场？", full)
    assert match_transcript("哪场战役印象最深", full).startswith("印象最深的是伐羌方")
    assert "三个关键词" in match_transcript("你会占卜吗", full)


def test_narrative_keeps_ten_pieces():
    folder = next(p for p in SRC.iterdir() if p.is_dir() and "妇好" in p.name)
    doc = next(p for p in folder.glob("*.docx") if "自述" in p.name)
    sections = parse_narrative(paragraphs_of_docx(doc))
    titles = [row["title"] for row in sections]
    assert any("鸮尊" in title for title in titles)
    assert any("司母辛鼎" in title for title in titles)
    assert all("图1" not in row["text"] for row in sections)
    assert len(sections) >= 10
