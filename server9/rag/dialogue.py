# -*- coding: utf-8 -*-
"""游客讲解的控制通道：意图短句、按钮钩子、展品名。口播里不放这些字段。"""
from __future__ import annotations
import re
from typing import Dict, Iterable, List

_WEATHER = re.compile(r"天气|气温|下雨|降温|升温|预报")
_NEWS = re.compile(r"新闻|头条|时事")
_BOOK = re.compile(r"《([^》]{1,40})》")
_FOREIGN_SCRIPT = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\uac00-\ud7af]")
_LATIN = re.compile(r"[A-Za-z]")
_HAN = re.compile(r"[\u4e00-\u9fff]")
_NORM = re.compile(r"[\s·・•]+")

_REALTIME = {
    "monet": (
        "场馆里不播报天气和新闻。今天更适合在厅里看光和颜色。",
        "天气和新闻就到这里。厅里还有作品可以看。",
        ["想了解本次展出的睡莲吗？", "要不要看看《日出·印象》？"],
    ),
    "fuhao": (
        "场馆里不播报天气和新闻。今日只讲这一场商代的展。",
        "这件事就到这里。厅里的甲骨和青铜可以继续看。",
        ["想看看厅里的甲骨吗？", "要不要听一件青铜？"],
    ),
    "kumarajiva": (
        "场馆里不播报天气和新闻。今日只讲这一场龟兹的展。",
        "这件事就到这里。厅里的译经和造像可以继续看。",
        ["想听听译经的事吗？", "要不要看看舍利盒？"],
    ),
}

FOREIGN_REPLY = "现在只用中文讲解。"
HALL_INTENTS = ("exhibit", "person", "visit")
_NAME_HOOK = {
    "monet": "想再看看%s吗？",
    "fuhao": "想再看看%s吗？",
    "kumarajiva": "想再听听%s吗？",
}


class RealtimeMemory:
    """没有会话对象的连接（游客 TTS）用来记连续天气/新闻。"""

    def __init__(self):
        self.realtime_streak = 0
        self.hook_used = set()
        self.last_answer = ""

    def bump_realtime(self) -> int:
        self.realtime_streak += 1
        return self.realtime_streak

    def clear_realtime(self) -> None:
        self.realtime_streak = 0


def rule_intent(question: str):
    """天气、新闻走词表。问题里已经点了书名号时，留给展品检索。"""
    text = question or ""
    if _BOOK.search(text):
        return None
    if _WEATHER.search(text):
        return "weather"
    if _NEWS.search(text):
        return "news"
    return None


def is_foreign_question(question: str) -> bool:
    """有日文假名、韩文，或整句是英文。夹了几个拉丁字母的中文问题仍当中文。"""
    text = question or ""
    if _FOREIGN_SCRIPT.search(text):
        return True
    return bool(_LATIN.search(text)) and not bool(_HAN.search(text))


def _lines(persona):
    pid = getattr(persona, "id", "") or ""
    first, again, hooks = _REALTIME.get(pid, _REALTIME["monet"])
    custom_first = (getattr(persona, "realtime_first", "") or "").strip()
    custom_again = (getattr(persona, "realtime_again", "") or "").strip()
    custom_hooks = [h for h in (getattr(persona, "realtime_hooks", None) or []) if h]
    return custom_first or first, custom_again or again, custom_hooks or list(hooks)


def _return_clause(first: str, again: str) -> str:
    """第一句事实说完，用原收敛句后半拉回本厅。"""
    parts = [p.strip() for p in re.split(r"(?<=[。！？])", first or "") if p.strip()]
    if len(parts) >= 2:
        return "".join(parts[1:])
    return again or ""


def realtime_reply(session, persona, fact: str = "") -> str:
    """第 1 轮可带一句查到的事实，说完立刻拉回本厅。第 2 轮起只说收敛句。"""
    first, again, _hooks = _lines(persona)
    streak = int(getattr(session, "realtime_streak", 1) or 1)
    if streak >= 2:
        return again
    fact = (fact or "").strip()
    if not fact:
        return first
    if fact[-1] not in "。！？":
        fact += "。"
    return fact + _return_clause(first, again)


def _named_hooks(persona, names, intent: str) -> List[str]:
    """本轮口播里出现的展品，优先变成一条对应按钮。"""
    if intent not in ("exhibit", "person", "knowledge"):
        return []
    pid = getattr(persona, "id", "") or ""
    template = _NAME_HOOK.get(pid) or "想再了解一下%s吗？"
    found = []
    for name in names or []:
        name = (name or "").strip()
        if not name:
            continue
        hook = template % name
        if hook not in found:
            found.append(hook)
        if len(found) >= 2:
            break
    return found


def suggest_hooks(intent: str, session, persona, names: List[str] = None) -> List[str]:
    """按钮文案。实时类和寒暄只给 1 条，其余最多 2 条。不写进口播。"""
    limit = 1 if intent in ("weather", "news", "chitchat") else 2
    if intent in ("command", "ticket_service"):
        return []
    _first, _again, realtime_hooks = _lines(persona)
    hooks = dict(getattr(persona, "hooks", None) or {})
    if intent in ("weather", "news"):
        pool = list(realtime_hooks)
    elif intent == "chitchat":
        pool = list(hooks.get("opening") or hooks.get("knowledge") or [])
    elif intent == "exhibit":
        pool = list(hooks.get("technique") or []) + list(hooks.get("knowledge") or [])
    elif intent == "person":
        pool = list(hooks.get("deep") or []) + list(hooks.get("knowledge") or [])
    elif intent == "visit":
        pool = list(hooks.get("interactive") or []) + list(hooks.get("opening") or [])
    else:
        pool = list(hooks.get("opening") or [])
    pool = _named_hooks(persona, names, intent) + pool
    used = getattr(session, "hook_used", None)
    if used is None:
        used = set()
        try:
            session.hook_used = used
        except Exception:
            pass
    picked = []
    fresh = [h for h in pool if h and h not in used]
    if len(fresh) < limit:
        used.clear()
        fresh = [h for h in pool if h]
    for hook in fresh:
        if hook in picked:
            continue
        picked.append(hook)
        used.add(hook)
        if len(picked) >= limit:
            break
    return picked


def visit_fact_reply(question: str, persona) -> str:
    """开放时间、票价、地址优先用人设里写好的句子，不够再交给检索。"""
    q = question or ""
    if _BOOK.search(q):
        return ""
    info = dict(getattr(persona, "ticket_info", None) or {})
    if not info:
        return ""
    if any(k in q for k in ("多少钱", "票价", "价格")) and any(k in q for k in ("票", "门票")):
        return (info.get("price") or "").strip()
    if any(k in q for k in ("学生票", "优惠票", "免票", "免费")):
        return (info.get("discount") or "").strip()
    if any(k in q for k in (
            "开放时间", "几点开门", "几点闭", "几点开馆", "闭馆", "停止入馆",
            "展期", "什么时候开", "什么时候闭")):
        return (info.get("duration") or "").strip()
    if any(k in q for k in ("哪里举办", "展览在哪", "展厅在哪", "地址")):
        return (info.get("location") or "").strip()
    return ""


def note_hall(session) -> None:
    clear = getattr(session, "clear_realtime", None)
    if clear:
        clear()


def note_realtime(session) -> int:
    bump = getattr(session, "bump_realtime", None)
    if bump:
        return int(bump())
    session.realtime_streak = int(getattr(session, "realtime_streak", 0) or 0) + 1
    return session.realtime_streak


def titles_from_items(items: Iterable[dict]) -> List[str]:
    """只取条目标题里的书名号，不扫传记正文。"""
    found = []
    seen = set()
    for it in items or []:
        if not isinstance(it, dict):
            continue
        for name in _BOOK.findall(it.get("title") or ""):
            name = name.strip()
            key = _NORM.sub("", name)
            if len(key) < 2 or key in seen:
                continue
            seen.add(key)
            found.append(name)
    found.sort(key=lambda s: len(_NORM.sub("", s)), reverse=True)
    return found


def exhibit_names(reply: str, titles: Iterable[str] = (),
                  aliases: Dict[str, str] = None) -> List[str]:
    """口播里出现的名字对齐到标准标题。对不上的不收。"""
    raw = reply or ""
    if not raw:
        return []
    pairs = []
    for title in titles or []:
        title = (title or "").strip()
        if title:
            pairs.append((title, title))
    for key, target in (aliases or {}).items():
        key = (key or "").strip()
        target = (target or "").strip()
        if key and target:
            pairs.append((key, target))
    pairs.sort(key=lambda item: len(_NORM.sub("", item[0])), reverse=True)
    occupied = [False] * len(raw)
    picked = []
    seen = set()

    def take(start: int, end: int, canonical: str):
        if end <= start or canonical in seen:
            return
        if any(occupied[start:end]):
            return
        for i in range(start, end):
            occupied[i] = True
        seen.add(canonical)
        picked.append(canonical)

    for surface, canonical in pairs:
        variants = [surface]
        bare = _NORM.sub("", surface)
        if bare and bare not in variants:
            variants.append(bare)
        for variant in variants:
            start = 0
            while True:
                idx = raw.find(variant, start)
                if idx < 0:
                    break
                take(idx, idx + len(variant), canonical)
                if canonical in seen:
                    break
                start = idx + len(variant)
    return picked


def pack_done(intent: str, hook_suggest: List[str], exhibit_names_found: List[str],
              refused: bool, sources, suggestions, sentiment) -> dict:
    limit = 1 if intent in ("weather", "news", "chitchat") else 2
    payload = {
        "type": "done",
        "refused": bool(refused),
        "intent": intent or "knowledge",
        "lang": "zh",
        "hook_suggest": list(hook_suggest or [])[:limit],
        "exhibit_names": [] if refused else list(exhibit_names_found or []),
        "sources": list(sources or []),
        "suggestions": suggestions if suggestions is not None else [],
        "sentiment": sentiment,
    }
    return payload
