# -*- coding: utf-8 -*-
"""检索原文直接送入，只在超出窗口预算时按句截断。不加载模型。"""
from __future__ import annotations
import sys
from pathlib import Path

SERVER9 = Path(__file__).resolve().parents[1] / "server9"
if str(SERVER9) not in sys.path:
    sys.path.insert(0, str(SERVER9))

from rag.passages import read_hits  # noqa: E402


def test_short_text_is_kept_whole():
    text = "睡莲是在吉维尼的池塘里画的。后来又画了鲁昂大教堂。"
    assert read_hits([{"text": text}]) == [text]


def test_later_sentences_stay_without_keyword_filter():
    text = "甲" * 40 + "。这段没有问到的内容也要留下。睡莲是在吉维尼画的。"
    assert read_hits([{"text": text}]) == [text]


def test_overflow_keeps_the_first_hit_and_cuts_on_a_sentence():
    first = "睡莲是在吉维尼的池塘里画的。" * 40
    second = "鲁昂大教堂在不同光线下画了很多幅。"
    outs = read_hits([{"text": first}, {"text": second}], budget=80)
    assert len(outs) == 1
    assert outs[0].startswith("睡莲是在吉维尼的池塘里画的。")
    assert outs[0].endswith("。")
    assert len(outs[0]) <= 80
    assert "鲁昂" not in outs[0]


def test_two_hits_both_pass_when_they_fit():
    a = "睡莲是在吉维尼的池塘里画的。"
    b = "鲁昂大教堂在不同光线下画了很多幅。"
    assert read_hits([{"text": a}, {"text": b}], budget=200) == [a, b]


def test_empty():
    assert read_hits([]) == []
    assert read_hits([{"text": "  "}]) == []
