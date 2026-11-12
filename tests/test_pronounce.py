# -*- coding: utf-8 -*-
"""多音字只改合成文本，字幕用的原句不动。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server9"))

from tts.pronounce import pronounce_for_tts  # noqa: E402


def test_qiuci_and_other_polyphones_become_homophones():
    spoken = pronounce_for_tts("我出生在龟兹，师父是鸠摩罗什，母亲耆婆来自罽宾。")
    assert "丘慈" not in spoken
    assert "龟兹(qiu1 ci2)" in spoken
    assert "鸠摩罗石" in spoken
    assert "奇婆" in spoken
    assert "记宾" in spoken
    assert pronounce_for_tts("妇好在殷墟见过司母戊，也见过单于。") == "妇浩在阴虚见过司母务，也见过蝉于。"
    assert pronounce_for_tts("勒阿弗尔的光") == "叻阿弗尔的光"


def test_plain_sentence_stays():
    text = "我在这里，第十一届博博会。"
    assert pronounce_for_tts(text) == text
    assert pronounce_for_tts("") == ""
