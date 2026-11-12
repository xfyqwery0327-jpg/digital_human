# -*- coding: utf-8 -*-
"""莫奈三问对稿：命中口播稿，其他人设和无关问题不拦截。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER9 = ROOT / "server9"
if str(SERVER9) not in sys.path:
    sys.path.insert(0, str(SERVER9))

from robot.scripted import (  # noqa: E402
    capability_reply, match_fixed_script, fixed_script_wav, needs_long_spoken,
    normalize_ask_mode, scripted_turn,
)


def test_three_robot_questions_hit_distinct_scripts():
    self_text = match_fixed_script("monet", "介绍一下你自己")
    works_text = match_fixed_script("monet", "介绍一下你的作品")
    pond_text = match_fixed_script("monet", "介绍一下睡莲池这幅作品吧")
    assert self_text and self_text.startswith("我是克劳德·莫奈")
    assert "两千多幅" in self_text and "180多幅" in self_text
    assert works_text and "印象·日出" in works_text and "光在空气里的呼吸" in works_text
    assert pond_text and "1900年" in pond_text and "芝加哥艺术学院" in pond_text
    assert len({self_text, works_text, pond_text}) == 3


def test_pond_wins_over_works_when_both_keywords_present():
    text = match_fixed_script("monet", "那你能不能给我们介绍下睡莲这个作品呢")
    assert text and "芝加哥艺术学院" in text
    assert "追逐光与瞬间" not in text


def test_loose_robot_wording_still_hits():
    assert match_fixed_script("monet", "莫奈介绍下自己吧").startswith("我是克劳德·莫奈")
    works = match_fixed_script("monet", "好的，你一生都有哪些作品啊")
    assert works and "印象·日出" in works


def test_fixed_scripts_map_to_06_12_wavs():
    self_wav = fixed_script_wav("monet", "介绍一下你自己")
    works_wav = fixed_script_wav("monet", "介绍一下你的作品")
    pond_wav = fixed_script_wav("monet", "介绍一下睡莲池这幅作品吧")
    assert self_wav.name == "s1.wav" and self_wav.is_file()
    assert works_wav.name == "s2.wav" and works_wav.is_file()
    assert pond_wav.name == "s3.wav" and pond_wav.is_file()
    assert "06_12" in str(self_wav)


def test_scripted_turn_switches_audio_source():
    query = "莫奈介绍下自己吧"
    wav_turn = scripted_turn("monet", query, 1)
    cosy_turn = scripted_turn("monet", query, 2)
    assert wav_turn["mode"] == 1
    assert wav_turn["text"].startswith("我是克劳德·莫奈")
    assert wav_turn["wav"].name == "s1.wav"
    assert cosy_turn["text"] == wav_turn["text"]
    assert cosy_turn["wav"] is None
    assert cosy_turn["mode"] == 2
    assert scripted_turn("monet", query, 3) is None
    assert scripted_turn("monet", "你好", 2) is None
    assert scripted_turn("fuhao", query, 1) is None


def test_normalize_ask_mode_defaults_to_wav():
    assert normalize_ask_mode(None) == 1
    assert normalize_ask_mode("2") == 2
    assert normalize_ask_mode(3) == 3
    assert normalize_ask_mode(9) == 1
    assert normalize_ask_mode("llm") == 1


def test_other_persona_and_unrelated_query_skip_script():
    assert match_fixed_script("fuhao", "介绍一下你自己") is None
    assert match_fixed_script("monet", "你好") is None
    assert match_fixed_script("monet", "睡莲是在哪里画的") is None
    assert fixed_script_wav("fuhao", "介绍一下你自己") is None


def test_what_else_can_you_do_lists_five_functions():
    text = capability_reply("莫奈，你还有什么功能？")
    assert text.startswith("我本事可不少，给您数五样：")
    for name in ("陪您聊", "带您看展品", "帮您约门票", "给您带路", "扫码进手机", "您想先试哪一项呢？"):
        assert name in text
    for banned in ("作品", "睡莲", "日出", "印象", "吉维尼"):
        assert banned not in text
    assert capability_reply("你还有什么功能") == text
    assert capability_reply("莫奈，你还能干什么呢？") == text
    assert capability_reply("你还能做些什么") == text
    assert capability_reply("好的，你一生都有哪些作品啊") == ""
    assert capability_reply("听说你眼睛后来不好了？那怎么还能画画？") == ""
    assert capability_reply("你好") == ""
    assert capability_reply("没什么功能啊") == ""
    assert capability_reply("这个是什么功能") == ""
    assert capability_reply("这个展品有什么功能") == ""
    assert capability_reply("你的作品还有什么功能") == ""
    assert capability_reply("你有哪些功能") == text
    assert capability_reply("妇好，你还有什么功能") == text
    assert needs_long_spoken("莫奈，你还能干什么呢？") is False
