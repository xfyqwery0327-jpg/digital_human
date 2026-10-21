# -*- coding: utf-8 -*-
"""浪漫与星辰展期锁定、门票预约、画作截取。不加载模型。"""
from __future__ import annotations
import sys
from pathlib import Path

SERVER9 = Path(__file__).resolve().parents[1] / "server9"
if str(SERVER9) not in sys.path:
    sys.path.insert(0, str(SERVER9))

from rag.visit import (  # noqa: E402
    CANONICAL_HOURS, filter_spoken_sentence, guard_hall_clock, is_visit_query, leaks_foreign_schedule,
)
from services.screen_hooks import (  # noqa: E402
    artworks_in_text, catalog_from_items, is_ticket_booking, request_ticket_form,
)


class _Persona:
    def __init__(self, pid):
        self.id = pid


MONET = _Persona("monet")
FUHAO = _Persona("fuhao")


def test_booking_is_not_a_price_question():
    assert is_ticket_booking("帮我预约明天门票")
    assert is_ticket_booking("我要订一张票")
    assert not is_ticket_booking("浪漫与星辰门票多少钱")
    assert not is_ticket_booking("今天几点闭馆")
    assert is_visit_query("今天几点闭馆")
    assert is_visit_query("门票多少钱")


def test_giverny_painting_stays_but_hours_do_not():
    painted = "睡莲是我在吉维尼的池塘里画的。"
    assert not leaks_foreign_schedule(painted, MONET)
    assert guard_hall_clock("睡莲在哪画的", painted, MONET) == painted

    leaked = "印象派特展展期到十二月三十一日，地点在二楼西侧。"
    assert leaks_foreign_schedule(leaked, MONET)
    fixed = guard_hall_clock("展览什么时候开门", leaked, MONET)
    assert fixed == CANONICAL_HOURS
    assert "浪漫与星辰" in fixed
    assert "吉维尼" not in fixed

    mixed = "睡莲画的是水面的光。吉维尼花园每天九点开放。"
    kept = guard_hall_clock("介绍睡莲", mixed, MONET)
    assert "水面的光" in kept
    assert "九点开放" not in kept


def test_spoken_filter_replaces_visit_once():
    state = {}
    first = filter_spoken_sentence(
        "吉维尼花园的展期到十二月三十一日。", "几点开门", MONET, state)
    second = filter_spoken_sentence(
        "印象派特展在二楼西侧。", "几点开门", MONET, state)
    assert "浪漫与星辰" in first
    assert second == ""
    assert filter_spoken_sentence("我在吉维尼画睡莲。", "介绍睡莲", MONET, {}) == "我在吉维尼画睡莲。"
    assert not leaks_foreign_schedule("印象派特展展期到十二月。", FUHAO)


def test_artwork_extract_longest_name():
    items = [
        {"title": "请介绍《卡美尔(穿绿裙的女人)》。", "text": "代表作有《日出·印象》和《睡莲》系列，还有《睡莲池》。"},
    ]
    catalog = catalog_from_items(items)
    names = artworks_in_text("我想看《日出印象》，也想听睡莲池和睡莲。", catalog)
    assert "日出·印象" in names
    assert "睡莲池" in names
    assert "睡莲" in names
    assert "卡美尔(穿绿裙的女人)" not in names


def test_ticket_api_is_reserved():
    result = request_ticket_form("帮我预约明天门票", "monet", "s1")
    assert result["code"] == "not_configured"
    assert result["payload"]["query"] == "帮我预约明天门票"
