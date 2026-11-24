# -*- coding: utf-8 -*-
"""场地包。改 deploy/scene.id 后重启，一起换知识、钩子和提示词。

broadmesse_hall：人在宽创大厅，宽创发展的问答从 monet/index3 读回。
bobohui：人在第十一届博博会现场的宽创展厅。
"""
from __future__ import annotations
import json
import os
import re
from functools import lru_cache

from config import PROJECT_ROOT

SCENE_FILE = PROJECT_ROOT / "deploy" / "scene.id"
DEFAULT_SCENE = "broadmesse_hall"

# 博博会说法在前，宽创大厅说法在后。按原文长度从长到短替换，避免短句截断长句。
PAIRS = (
    (
        "你此刻就在第十一届博博会现场，人就在宽创国际的展厅里。",
        "你此刻就在宽创大厅里。",
    ),
    (
        "现在只介绍第十一届博博会，不要介绍原先的展，包括浪漫与星辰、贵州省博物馆",
        "人在宽创大厅。不要介绍原先的展，包括浪漫与星辰、贵州省博物馆，也不要把博博会说成现在的所在",
    ),
    (
        "现在只介绍第十一届博博会，不要介绍原先的商代特展",
        "人在宽创大厅。不要介绍原先的商代特展，也不要把博博会说成现在的所在",
    ),
    (
        "吉维尼是我画画的花园，不是博博会的场馆。禁止把吉维尼的开放时间说成博博会的时间",
        "吉维尼是我画画的花园，不是宽创大厅。禁止把吉维尼的开放时间说成宽创大厅的时间",
    ),
    (
        "有人问第十一届博博会，或中国博物馆及相关产品与技术博览会时，按参考资料介绍。你人就在现场的宽创展厅里，不要把展台说成在那里",
        "有人问宽创的发展、荣誉、业务或展区时，按参考资料介绍。你人就在宽创大厅里，不要把大厅说成在那里",
    ),
    (
        "问展会时间和地点时，只说第十一届博博会",
        "问地点时说宽创大厅，不要把博博会说成现在的所在",
    ),
    (
        "展会只介绍第十一届博博会。",
        "你此刻在宽创大厅。可以讲宽创的发展，也可以讲你自己。",
    ),
    (
        "现在只介绍第十一届博博会。",
        "你此刻在宽创大厅，可以讲宽创的发展。",
    ),
    (
        "您还想了解博博会或宽创展台的哪些内容呢？",
        "您还想了解宽创大厅或宽创发展的哪些内容呢？",
    ),
    ("今日在博博会上由我陪你说话。", "今日在宽创大厅由我陪你说话。"),
    ("今日由我在博博会上讲。", "今日由我在宽创大厅讲。"),
    ("展会我只介绍博博会。", "我现在在宽创大厅。"),
    ("展会只介绍博博会。", "现在人在宽创大厅。"),
    ("展会只讲博博会。", "人在宽创大厅。"),
    ("只说你只介绍博博会", "只说你在宽创大厅"),
)

_COMPANY = re.compile(
    r"宽创|发展历程|张东|Broadmesse|broadmesse|荣誉|出海|产业园|"
    r"邦德|智联|万里江河|博物馆\s*4|EPCO|宽小志"
)
_COMPANY_FILE = PROJECT_ROOT / "monet" / "index3" / "items.json"

_BOBOHUI = {
    "id": "bobohui",
    "short_name": "博博会",
    "booth_name": "宽创展台",
    "look_name": "这届博博会",
    "hook_tail": "您还想了解博博会或宽创展台的哪些内容呢？",
    "this_place": re.compile(r"博博会|中国博物馆及相关产品与技术博览会"),
    "ask": re.compile(r"博博会|中国博物馆及相关产品与技术博览会"),
    "text": (
        "第十一届中国博物馆及相关产品与技术博览会，简称博博会，定于二零二六年十一月十三日至十六日"
        "在海南国际会展中心举办，主题是「高质量发展的博物馆：科技赋能 联结未来」。"
        "二零二六年十月二十日上午，新闻发布会在海口举行。中国博物馆协会理事长关强、秘书长李金光，"
        "海南省旅游和文化广电体育厅副厅长李海钢，以及海口市人民政府副市长何霁锋出席，"
        "公布了筹备进展、展览布局、特色活动和公众参观安排。"
        "这是十五五开局之年的文博行业盛会，集中展示我国博物馆事业建设成果，推动文博科技深度融合，"
        "深化中外文明交流互鉴。"
        "我此刻就在这场博博会现场，人就在宽创国际的展厅里。"
        "宽创展厅用科技把博物馆和观众联结起来。"
    ),
    "when_where": (
        "我就在博博会现场的宽创展厅里。"
        "第十一届博博会定于二零二六年十一月十三日至十六日，在海南国际会展中心举办。"
    ),
    "no_old_ticket": "原先的门票价格我不介绍。",
    "presence_rule": (
        "你人就在博博会现场的宽创展厅里，这是事实，但不要主动报位置。"
        "不要说在那里，不要说我听说。"
        "只有观众问你在哪里、展厅在哪、人在不在现场时，才说我就在这里。"
        "讲作品、经历、器物或其他问题时，开口就是答案本身，不要先说我在这里。\n"
    ),
    "answer_rule": "展会只介绍博博会，不要介绍原先的展，也不要报原先的门票价格。",
    "visit_hint": (
        "这个问题在问展会时间或地点。只介绍第十一届博博会，不要介绍原先的展，也不要报原先的门票价格。"
        "不要把吉维尼或别的馆的开放时间说成博博会的时间。\n"
    ),
    "source": "第十一届博博会新闻发布会",
    "title": "第十一届博博会",
    "category": "exhibition_bobohui",
}

_HALL = {
    "id": "broadmesse_hall",
    "short_name": "宽创大厅",
    "booth_name": "宽创发展",
    "look_name": "宽创大厅",
    "hook_tail": "您还想了解宽创大厅或宽创发展的哪些内容呢？",
    "this_place": re.compile(r"宽创大厅|宽创发展|宽创国际|宽创"),
    "ask": _COMPANY,
    "text": (
        "我此刻就在宽创大厅里。宽创大厅在上海张江的宽创国际总部。"
        "大厅可以看荣誉墙、公司介绍和发展历程，再往下是人工智能展示馆、万里江河和智联万象。"
        "问到宽创的发展，按资料从二零零五年三地创立讲起。"
    ),
    "when_where": (
        "我就在宽创大厅里。"
        "大厅在上海张江的宽创国际总部，从一楼荣誉墙走到魔墙区的发展历程，"
        "再下到人工智能展示馆、万里江河和智联万象。"
    ),
    "no_old_ticket": "原先的门票价格我不介绍。",
    "presence_rule": (
        "你人就在宽创大厅里，这是事实，但不要主动报位置。"
        "不要说在那里，不要说我听说。"
        "只有观众问你在哪里、大厅在哪、人在不在现场时，才说我就在这里。"
        "讲作品、经历、器物、宽创发展或其他问题时，开口就是答案本身，不要先说我在这里。\n"
    ),
    "answer_rule": (
        "人在宽创大厅。问到宽创的发展、荣誉、业务或展区时按参考资料讲。"
        "不要介绍原先的展，也不要报原先的门票价格，不要把博博会说成你现在所在的地方。"
    ),
    "visit_hint": (
        "这个问题在问时间或地点。人在宽创大厅。不要介绍原先的展，也不要报原先的门票价格。"
        "不要把吉维尼或博博会的时间说成宽创大厅的时间。\n"
    ),
    "source": "宽创高频QA",
    "title": "宽创大厅",
    "category": "exhibition_broadmesse",
}

PACKS = {
    "bobohui": _BOBOHUI,
    "broadmesse_hall": _HALL,
}


def scene_id() -> str:
    """环境变量 DH_SCENE 优先，否则读 deploy/scene.id。"""
    picked = (os.environ.get("DH_SCENE") or "").strip()
    if not picked and SCENE_FILE.exists():
        picked = SCENE_FILE.read_text(encoding="utf-8").strip()
    if picked not in PACKS:
        return DEFAULT_SCENE
    return picked


def active() -> dict:
    return PACKS[scene_id()]


def rewrite(text: str, scene: str = "") -> str:
    """把一段文案改成当前场地的说法。两种写法来回换，短句不会截断长句。"""
    name = scene or scene_id()
    out = text or ""
    pairs = sorted(PAIRS, key=lambda item: len(item[0]), reverse=True)
    if name == "broadmesse_hall":
        for src, dst in pairs:
            out = out.replace(src, dst)
        return out
    for src, dst in sorted(PAIRS, key=lambda item: len(item[1]), reverse=True):
        out = out.replace(dst, src)
    return out


def _rewrite_list(values, scene: str) -> list:
    return [rewrite(str(item), scene) for item in (values or [])]


def apply_scene(persona, scene: str = ""):
    """人设读入后，把场地相关的提示词、招呼、拒答和钩子换成当前场景。"""
    name = scene or scene_id()
    persona.system_prompt = rewrite(persona.system_prompt or "", name)
    persona.refuse_text = rewrite(persona.refuse_text or "", name)
    persona.warmup_text = rewrite(persona.warmup_text or "", name)
    persona.hello_texts = _rewrite_list(persona.hello_texts, name)
    persona.bye_texts = _rewrite_list(persona.bye_texts, name)
    persona.realtime_first = rewrite(persona.realtime_first or "", name)
    persona.realtime_again = rewrite(persona.realtime_again or "", name)
    persona.realtime_hooks = _rewrite_list(persona.realtime_hooks, name)
    persona.hooks = {
        str(key): _rewrite_list(lines, name)
        for key, lines in (persona.hooks or {}).items()
    }
    persona.chitchat = {
        str(key): rewrite(str(value), name)
        for key, value in (persona.chitchat or {}).items()
    }
    return persona


def _grams(text: str) -> set:
    raw = re.sub(r"[\s？?，,。！!、]", "", text or "")
    if len(raw) < 2:
        return set()
    return {raw[i:i + 2] for i in range(len(raw) - 1)}


@lru_cache(maxsize=1)
def company_qa() -> tuple:
    """宽创发展问答。文件与主分支的 monet/index3 是同一份，只取带问句的条目。"""
    if not _COMPANY_FILE.exists():
        return tuple()
    data = json.loads(_COMPANY_FILE.read_text(encoding="utf-8"))
    rows = []
    for item in data:
        ask = (item.get("question") or "").strip()
        text = (item.get("text") or "").strip()
        if ask and text:
            rows.append((ask, text, item.get("id") or ask, item.get("title") or ask, item.get("source") or "宽创高频QA"))
    return tuple(rows)


def company_hits(question: str, limit: int = 2) -> list:
    """问到宽创、发展、荣誉、业务时，把原来的问答排到检索前面。"""
    if scene_id() != "broadmesse_hall":
        return []
    if not _COMPANY.search(question or ""):
        return []
    qg = _grams(question)
    ranked = []
    for ask, text, item_id, title, source in company_qa():
        overlap = len(qg & _grams(ask))
        if overlap:
            ranked.append((overlap, len(ask), ask, text, item_id, title, source))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    picked = [row for row in ranked if row[0] >= 2][:limit]
    if not picked and ranked:
        picked = ranked[:1]
    hits = []
    for _overlap, _length, _ask, text, item_id, title, source in picked:
        hits.append({
            "id": item_id,
            "text": text,
            "source": source,
            "category": "exhibition_broadmesse",
            "title": title,
            "score": 1.0,
            "recall": "hall",
        })
    return hits


def place_hit(question: str):
    """问到当前场地时给一条可引用的介绍。博博会场景仍是发布会通稿，大厅场景是宽创大厅。"""
    pack = active()
    if not pack["ask"].search(question or ""):
        return None
    if pack["id"] == "broadmesse_hall":
        return None
    return {
        "id": "bobohui_2026_11",
        "text": pack["text"],
        "source": pack["source"],
        "category": pack["category"],
        "title": pack["title"],
        "score": 1.0,
        "recall": "hall",
    }


def merge_hits(question: str, hits, threshold: float) -> list:
    hits = list(hits or [])
    if scene_id() == "broadmesse_hall":
        extra = company_hits(question)
        if not extra:
            return hits
        ids = {item.get("id") for item in extra}
        rest = [item for item in hits if item.get("id") not in ids]
        weak = not rest or float(rest[0].get("score") or 0) < float(threshold)
        return extra if weak else extra + rest
    expo = place_hit(question)
    if not expo:
        return hits
    if not hits or float(hits[0].get("score") or 0) < float(threshold):
        return [expo]
    rest = [item for item in hits if item.get("id") != expo["id"]]
    return [expo] + rest
