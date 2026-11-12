# -*- coding: utf-8 -*-
"""合成前改口播字符串。字幕和 targetName 仍用原字。

龟兹写成 tokenizer 认的拼音后缀 `龟兹(qiu1 ci2)`。语音服务把后缀收成 qiū cí，
不再把汉字和拼音各念一遍。其余多音字仍换成读得准的同音字。
"""
from __future__ import annotations

# 长词在前，避免「罗什」抢在「鸠摩罗什」前面被换掉。
_READINGS = (
    ("鸠摩罗什", "鸠摩罗石"),
    ("勒阿弗尔", "叻阿弗尔"),
    ("龟兹", "龟兹(qiu1 ci2)"),
    ("耆婆", "奇婆"),
    ("罽宾", "记宾"),
    ("苏巴什", "苏巴石"),
    ("妇好", "妇浩"),
    ("殷墟", "阴虚"),
    ("司母戊", "司母务"),
    ("武丁", "武钉"),
    ("罗什", "罗石"),
    ("单于", "蝉于"),
)


def pronounce_for_tts(text: str) -> str:
    """只改送进合成器的字符串。龟兹写成拼音后缀，什读石，妇好的好读四声浩。"""
    raw = text or ""
    if not raw:
        return raw
    out = raw
    for src, dst in _READINGS:
        if src in out:
            out = out.replace(src, dst)
    return out
