# -*- coding: utf-8 -*-
"""合成前把多音字换成模型读得准的同音字。字幕和 targetName 仍用原字。"""
from __future__ import annotations

# 长词在前，避免「罗什」抢在「鸠摩罗什」前面被换掉。
_READINGS = (
    ("鸠摩罗什", "鸠摩罗石"),
    ("勒阿弗尔", "叻阿弗尔"),
    ("龟兹", "丘慈"),
    ("耆婆", "奇婆"),
    ("罽宾", "记宾"),
    ("苏巴什", "苏巴石"),
    ("妇好", "妇浩"),
    ("殷墟", "阴虚"),
    ("司母戊", "司母务"),
    ("武丁", "武钉"),
    ("罗什", "罗石"),
    ("解忧", "谢忧"),
    ("单于", "蝉于"),
)


def pronounce_for_tts(text: str) -> str:
    """只改送进合成器的字符串。龟兹读丘慈，什读石，妇好的好读郝。"""
    raw = text or ""
    if not raw:
        return raw
    out = raw
    for src, dst in _READINGS:
        if src in out:
            out = out.replace(src, dst)
    return out
