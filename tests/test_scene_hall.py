# -*- coding: utf-8 -*-
"""宽创大厅是当前场地。知识、钩子、提示词由 deploy/scene.id 一起换。"""
from __future__ import annotations
import os
import sys
from pathlib import Path

SERVER9 = Path(__file__).resolve().parents[1] / "server9"
if str(SERVER9) not in sys.path:
    sys.path.insert(0, str(SERVER9))

from rag.bobohui import hall_hook, merge_bobohui  # noqa: E402
from rag.scene import rewrite, scene_id  # noqa: E402


def test_default_scene_is_the_hall():
    os.environ.pop("DH_SCENE", None)
    assert scene_id() == "broadmesse_hall"
    assert "宽创大厅" in hall_hook("天气")
    assert "博博会" not in hall_hook("天气")


def test_company_history_comes_back_from_the_shared_index():
    hits = merge_bobohui("介绍一下宽创", [], 0.35)
    assert hits
    assert "2005" in hits[0]["text"] or "张东" in hits[0]["text"]
    founded = merge_bobohui("宽创是什么时候成立的", [], 0.35)
    assert founded and "2005" in founded[0]["text"]


def test_scene_id_switches_hook_and_prompt():
    expo = "展会只介绍第十一届博博会。您还想了解博博会或宽创展台的哪些内容呢？"
    hall = rewrite(expo, "broadmesse_hall")
    assert "宽创大厅" in hall and "第十一届博博会" not in hall
    back = rewrite(hall, "bobohui")
    assert "第十一届博博会" in back
    assert "博博会或宽创展台" in back
