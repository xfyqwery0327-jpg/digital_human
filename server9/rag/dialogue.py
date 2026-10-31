# -*- coding: utf-8 -*-
"""游客讲解的控制通道：意图短句、按钮钩子、展品名。口播里不放这些字段。"""
from __future__ import annotations
import re
from typing import Dict, Iterable, List, Optional, Tuple

_WEATHER = re.compile(r"天气|气温|下雨|降温|升温|预报")
_NEWS = re.compile(r"新闻|头条|时事|播报新闻")
_NEWS_MORE = re.compile(r"再来一条|再来一则|再报一条|再讲一条|换一条|下一条|另一条|还有一条|还想再来")
_NEWS_MORE_SOFT = re.compile(r"了解更多|多讲一点|还有呢|还有吗|继续播")
_BOOK = re.compile(r"《([^》]{1,40})》")
_FOREIGN_SCRIPT = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\uac00-\ud7af]")
_LATIN = re.compile(r"[A-Za-z]")
_HAN = re.compile(r"[\u4e00-\u9fff]")
_NORM = re.compile(r"[\s·・•]+")

def _hook_pair():
    from rag.bobohui import hall_hook
    return hall_hook("天气"), hall_hook("新闻"), [hall_hook("我的作品")]


_REALTIME = {pid: _hook_pair() for pid in ("monet", "fuhao", "kumarajiva")}

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
        self.news_played = []
        self.last_realtime_kind = ""
        self.reply_lang = "zh"

    def bump_realtime(self) -> int:
        self.realtime_streak += 1
        return self.realtime_streak

    def clear_realtime(self) -> None:
        self.realtime_streak = 0


_NOISE = re.compile(r"^[0-9零一二三四五六七八九十两〇○]+[啊呀呢吧吗]?$")
_RELATION = re.compile(r"什么关系|怎么认识|如何认识|怎样认识|咋认识")
_SPOUSE = re.compile(r"老婆|妻子|夫人|爱人")
_CHILD = re.compile(r"孩子|儿子|女儿")
_SKIP_FAMILY = re.compile(
    r"？|吗|何时|有几个|是谁|是什么|请介绍|如何|请说明|请根据|值得关注|"
    r"《|欧希德|凡尔纳|费加罗|谢鲁班|垂钓|去世|死于|病情|卧床|癌"
)
_NAME_ALIAS = {
    "卡米尔": "卡美尔",
    "卡米耶": "卡美尔",
    "迈克尔": "米契尔",
    "米歇尔": "米契尔",
    "让": "简",
}


def rule_intent(question: str, session=None):
    """切换语言、天气、新闻走词表。上一条刚播过新闻时，再来一条也算新闻。"""
    text = question or ""
    from rag.speech_lang import split_switch
    code, rest = split_switch(text)
    if code and len(rest) < 2:
        return "lang_switch"
    if _BOOK.search(text):
        return None
    if _WEATHER.search(text):
        return "weather"
    if _NEWS.search(text) or _NEWS_MORE.search(text):
        return "news"
    last = getattr(session, "last_realtime_kind", "") if session is not None else ""
    if last == "news" and _NEWS_MORE_SOFT.search(text):
        return "news"
    return None


def is_noise_question(question: str) -> bool:
    """单个字或纯数字，没有可检索的问题。"""
    text = re.sub(r"[\s，。！？、…,.!?]", "", question or "")
    if not text:
        return False
    if len(text) <= 1:
        return True
    return bool(_NOISE.fullmatch(text))


def is_relation_question(question: str) -> bool:
    """问和某人什么关系、怎么认识。点了书名号的仍交给展品检索。"""
    text = question or ""
    if _BOOK.search(text):
        return False
    return bool(_RELATION.search(text))


def is_family_question(question: str) -> bool:
    """问孩子或配偶。点了书名号的仍交给展品检索。"""
    text = question or ""
    if _BOOK.search(text):
        return False
    return bool(_SPOUSE.search(text) or _CHILD.search(text))


def _family_name(raw: str) -> str:
    name = re.sub(r"（[^）]*）|\([^)]*\)", "", raw or "")
    name = name.strip("，、 ")
    return _NAME_ALIAS.get(name, name)


def _family_sentences(blob: str):
    parts = re.split(r"[\n。！？]", blob or "")
    out = []
    for part in parts:
        sent = re.sub(r"《[^》]*》", "", part or "")
        sent = re.sub(r"\s+", "", sent)
        if len(sent) < 4 or _SKIP_FAMILY.search(sent):
            continue
        out.append(sent)
    return out


def _persona_blobs(persona):
    from rag import index as rag
    bundle = rag.bundle_for(persona) if persona is not None else None
    if not bundle:
        return []
    return [{"text": it.get("text") or ""} for it in bundle.get("items") or []]


def family_reply(question: str, hits=None, persona=None) -> str:
    """只根据资料里的结婚对象和子女名字开口。问句、病故和其他人的家事不念。"""
    if hits is None:
        hits = _persona_blobs(persona)
    blob = "\n".join((h.get("text") or "") for h in hits or [])
    parts = _family_sentences(blob)
    ask = question or ""
    lines = []
    if _SPOUSE.search(ask):
        names = []
        for sent in parts:
            if "结婚" not in sent:
                continue
            for matched in re.finditer(r"(?:与|和)([\u4e00-\u9fff]{2,4})(?:在[\u4e00-\u9fff0-9]{0,12})?结婚", sent):
                name = _family_name(matched.group(1))
                if name in {"莫奈", "妻子", "夫人"} or name in names:
                    continue
                names.append(name)
        if len(names) == 1:
            lines.append("我和%s结了婚。" % names[0])
        elif names:
            rest = "、".join(names[1:])
            lines.append("我和%s结了婚，后来又和%s结了婚。" % (names[0], rest))
    if _CHILD.search(ask):
        eldest = []
        later = []

        def remember(bucket, name: str) -> None:
            if not name or name in {"奥拉斯"} or name in eldest or name in later:
                return
            bucket.append(name)

        for sent in parts:
            if not re.search(r"长子|儿子|女儿", sent):
                continue
            for matched in re.finditer(r"长子[——\-]*([\u4e00-\u9fff]{1,4})", sent):
                remember(eldest, _family_name(matched.group(1)))
            for pattern in (
                r"第[一二三四五]个儿子([\u4e00-\u9fff]{1,4})",
                r"另一个儿子[，,]?([\u4e00-\u9fff]{1,4})",
                r"儿子([\u4e00-\u9fff]{1,3})",
            ):
                for matched in re.finditer(pattern, sent):
                    remember(later, _family_name(matched.group(1)))
        later = [name for name in later if name not in eldest]
        if eldest and later:
            lines.append("我们的长子是%s，后来又有了儿子%s。" % (eldest[0], "、".join(later)))
        elif eldest:
            lines.append("我们的长子是%s。" % "、".join(eldest))
        elif len(later) == 1:
            lines.append("资料里写到儿子%s。" % later[0])
        elif later:
            lines.append("资料里写到儿子%s。" % "、".join(later))
    if not lines:
        return ""
    from tts.speaktext import to_spoken_zh
    return to_spoken_zh("".join(lines))


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


def _hall_hook(persona, kind: str = "") -> str:
    """天气说完问了解完天气，新闻说完问了解完新闻。"""
    del persona
    from rag.bobohui import hall_hook
    topic = "新闻" if kind == "news" else "天气"
    return hall_hook(topic)


def realtime_reply(session, persona, fact: str = "", kind: str = "") -> str:
    """播完查到的事实，再按刚讲的内容加一句钩子。没查到就只说这句钩子。"""
    kind = kind or getattr(session, "last_realtime_kind", "") or "weather"
    hook = _hall_hook(persona, kind)
    fact = (fact or "").strip()
    if not fact:
        return hook
    if fact[-1] not in "。！？":
        fact += "。"
    return fact + hook


def remember_news(session, key: str) -> None:
    if session is None or not key:
        return
    played = getattr(session, "news_played", None)
    if played is None:
        played = []
        try:
            session.news_played = played
        except Exception:
            return
    if key not in played:
        played.append(key)


def prepare_realtime(question: str, session, persona) -> dict:
    """天气或新闻：取缓存或现查，口播后带回博博会。不是这两类时返回空。"""
    kind = rule_intent(question, session)
    if kind not in ("weather", "news"):
        return {}
    note_realtime(session)
    avoid = list(getattr(session, "news_played", None) or [])
    from services.local_feed import feed_bundle
    bundle = feed_bundle(kind, question or "", avoid if kind == "news" else None)
    if kind == "news":
        remember_news(session, (bundle or {}).get("uniquekey") or "")
    try:
        session.last_realtime_kind = kind
    except Exception:
        pass
    reply = realtime_reply(session, persona, (bundle or {}).get("text") or "", kind)
    return {"kind": kind, "reply": reply, "bundle": bundle or {}}


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
    """按钮跟着刚讲的内容走：了解完天气、新闻或我的作品。"""
    del session, persona, names
    if intent in ("command", "ticket_service"):
        return []
    from rag.bobohui import HOOK_TOPIC, hall_hook
    return [hall_hook(HOOK_TOPIC.get(intent, "这些"))]


def visit_fact_reply(question: str, persona) -> str:
    """问展会时间和地点只答博博会。原先门票价格不报。问作品不走这里。"""
    del persona
    q = question or ""
    if _BOOK.search(q):
        return ""
    from rag.bobohui import NO_OLD_TICKET, WHEN_WHERE
    if any(k in q for k in ("多少钱", "票价", "价格", "学生票", "优惠票", "免票", "免费")) and any(
            k in q for k in ("票", "门票")):
        return NO_OLD_TICKET
    if any(k in q for k in ("学生票", "优惠票", "免票")):
        return NO_OLD_TICKET
    if any(k in q for k in (
            "开放时间", "几点开门", "几点闭", "几点开馆", "闭馆", "停止入馆",
            "展期", "什么时候开", "什么时候闭",
            "哪里举办", "展览在哪", "展厅在哪", "地址")):
        return WHEN_WHERE
    return ""


def note_hall(session) -> None:
    clear = getattr(session, "clear_realtime", None)
    if clear:
        clear()
    try:
        session.last_realtime_kind = ""
    except Exception:
        pass


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


_GUIDE = re.compile(
    r"导览|带我逛|带我参观|带我看看|带我讲解|"
    r"参观路线|游览路线|讲解路线|推荐路线|"
    r"怎么逛|如何参观|怎么参观|"
    r"从哪开始|从哪里开始|先看哪|先看什么|"
    r"开始导览|帮我导览|给我导览"
)
_EXHIBIT_ASK = re.compile(
    r"展品|这幅画|那幅画|这个画|这件作品|那件作品|"
    r"这件文物|那件文物|这件器物|名画|画作|这个展品|那个展品"
)
_BTN_PUNCT = re.compile(r"[\s，。！？、；：,.!?;:\"“”‘’'（）()【】\[\]《》·・]+")
# 前端按钮直接把这三句当 type=tts 的 text 发来，不再交给大模型改写。
_BUTTON_SCRIPTS = (
    ("介绍一下你的展品", "exhibit"),
    ("预约二维码展示", "reserve"),
    ("展馆导览图展示", "guide"),
)


def _name_pairs(titles: Iterable[str] = (), aliases: Dict[str, str] = None) -> List[Tuple[str, str]]:
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
    return pairs


def _canon_from_pairs(inner: str, pairs: List[Tuple[str, str]]) -> str:
    key = _NORM.sub("", inner or "")
    if len(key) < 2:
        return ""
    for surface, canonical in pairs:
        if _NORM.sub("", surface) == key:
            return canonical
    return ""


def _catalog_for(persona, titles: Optional[Iterable[str]], aliases: Optional[Dict[str, str]]):
    resolved_titles = list(titles) if titles is not None else None
    resolved_aliases = dict(aliases) if aliases is not None else None
    if resolved_titles is None:
        resolved_titles = []
        if persona is not None:
            try:
                from rag import index as rag
                resolved_titles = titles_from_items(rag.active_items(persona))
            except Exception:
                resolved_titles = []
    if resolved_aliases is None:
        resolved_aliases = {}
        if persona is not None:
            resolved_aliases = dict(getattr(persona, "exhibit_aliases", None) or {})
    return resolved_titles, resolved_aliases


def first_exhibit_name(text: str, titles: Iterable[str] = (),
                       aliases: Dict[str, str] = None) -> str:
    """文本里位置最靠前的展品。重叠时留更长的名字。书名号对不上目录时仍收原文。"""
    raw = text or ""
    if not raw:
        return ""
    pairs = _name_pairs(titles, aliases)
    found = []

    def add(start: int, end: int, canonical: str) -> None:
        if end <= start or not canonical:
            return
        found.append((start, end, canonical))

    for matched in _BOOK.finditer(raw):
        inner = (matched.group(1) or "").strip()
        canon = _canon_from_pairs(inner, pairs)
        if not canon and len(_NORM.sub("", inner)) >= 2:
            canon = inner
        add(matched.start(), matched.end(), canon)

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
                add(idx, idx + len(variant), canonical)
                start = idx + len(variant)

    if not found:
        return ""
    found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    kept = []
    for start, end, canonical in found:
        if any(start < prev_end and end > prev_start for prev_start, prev_end, _name in kept):
            continue
        kept.append((start, end, canonical))
    kept.sort(key=lambda item: item[0])
    return kept[0][2]


def _map_screen_intent(intent: str) -> str:
    label = (intent or "").strip()
    if label in ("ticket_service", "reserve"):
        return "reserve"
    if label in ("visit", "guide"):
        return "guide"
    if label == "exhibit":
        return "exhibit"
    return ""


def button_screen_intent(question: str) -> str:
    """前端三个按钮的固定话术。对上才返回 guide / reserve / exhibit。"""
    key = _BTN_PUNCT.sub("", question or "")
    if not key:
        return ""
    for phrase, kind in _BUTTON_SCRIPTS:
        if key == _BTN_PUNCT.sub("", phrase):
            return kind
    return ""


def screen_intention_kind(question: str, persona=None, classified_intent: str = "",
                          titles: Iterable[str] = None, aliases: Dict[str, str] = None) -> str:
    """导览 guide、预约 reserve、展品 exhibit，其余 talk。只看用户问题。"""
    from rag.visit import is_visit_query
    from services.screen_hooks import is_ticket_booking

    text = question or ""
    button = button_screen_intent(text)
    if button:
        return button
    if is_ticket_booking(text):
        return "reserve"
    catalog, alias_map = _catalog_for(persona, titles, aliases)
    # 点了名的展品优先于「带我看看」这类导览词，屏幕要落到这一件上。
    if first_exhibit_name(text, catalog, alias_map):
        return "exhibit"
    if _GUIDE.search(text) or is_visit_query(text):
        return "guide"
    if _EXHIBIT_ASK.search(text):
        return "exhibit"
    return _map_screen_intent(classified_intent) or "talk"


def intention_event(question: str, reply: str = "", persona=None,
                    classified_intent: str = "", titles: Iterable[str] = None,
                    aliases: Dict[str, str] = None) -> dict:
    """tts_start 前发给前端。展品只带最早出现的一件，其他类型 targetName 为空。"""
    catalog, alias_map = _catalog_for(persona, titles, aliases)
    kind = screen_intention_kind(
        question, persona, classified_intent, catalog, alias_map)
    target = ""
    if kind == "exhibit":
        target = (
            first_exhibit_name(question or "", catalog, alias_map)
            or first_exhibit_name(reply or "", catalog, alias_map)
        )
    return {
        "type": "intention",
        "message": {
            "intentionType": kind,
            "targetName": target or "",
        },
    }


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
