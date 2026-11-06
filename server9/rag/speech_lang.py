# -*- coding: utf-8 -*-
"""口播语言。观众问题暂时仍是中文，回答可以切到中英日韩。"""
from __future__ import annotations
import re

LANGS = {
    "zh": {"label": "中文", "qwen": "Chinese", "confirm": "好的，接下来我用中文讲。"},
    "en": {"label": "英语", "qwen": "English", "confirm": "Okay, I will speak English from now on."},
    "ja": {"label": "日语", "qwen": "Japanese", "confirm": "はい、これからは日本語で話します。"},
    "ko": {"label": "韩语", "qwen": "Korean", "confirm": "네, 이제부터 한국어로 말씀드리겠습니다."},
}

_NEAR = re.compile(
    r"(?:切换|换成|改成|改用|用|说|讲)(?:到|成|为|下)?\s*(英语|英文|日语|日文|韩语|韩文|中文|汉语)"
)
_WORD = {
    "英语": "en", "英文": "en",
    "日语": "ja", "日文": "ja",
    "韩语": "ko", "韩文": "ko",
    "中文": "zh", "汉语": "zh",
}
_HAN = re.compile(r"[\u4e00-\u9fff]")
_LAT = re.compile(r"[A-Za-z]")
_KANA = re.compile(r"[\u3040-\u30ff]")
_HANG = re.compile(r"[\uac00-\ud7af]")
_LEAD = re.compile(r"^(请|帮我|给我|麻烦|一下)+")


def parse_switch(question: str) -> str:
    """切换英语、用日语、说韩语。没有语种词时返回空。"""
    code, _rest = split_switch(question)
    return code


def split_switch(question: str):
    """返回 (语种, 去掉切换说法后剩下的问题)。不是切换时语种为空。"""
    text = question or ""
    found = _NEAR.search(text)
    if not found:
        return "", text
    code = _WORD.get(found.group(1)) or ""
    if not code:
        return "", text
    rest = (text[:found.start()] + text[found.end():]).strip(" ，,。！？")
    rest = _LEAD.sub("", rest).strip(" ，,。！？")
    return code, rest


def reply_lang_of(session) -> str:
    code = getattr(session, "reply_lang", "") if session is not None else ""
    return code if code in LANGS else "zh"


def set_reply_lang(session, code: str) -> str:
    code = code if code in LANGS else "zh"
    if session is not None:
        try:
            session.reply_lang = code
        except Exception:
            pass
    return code


def confirm_text(code: str) -> str:
    return LANGS.get(code, LANGS["zh"])["confirm"]


_NAME = {
    "zh": "博博会",
    "en": "China Museum Expo",
    "ja": "中国博物館エキスポ",
    "ko": "중국 박물관 엑스포",
}
_PROPER = {
    "en": "博博会是专名，必须写成 China Museum Expo，不要写成博博会。宽创国际写成 Kuanchuang。数字用阿拉伯数字。句号用英文句点，逗号用英文逗号。",
    "ja": "博博会是专名，必须写成中国博物館エキスポ，不要写成博博会。句号用。，停顿用、。",
    "ko": "博博会是专名，必须写成중국 박물관 엑스포，不要写成博博会，整段不要出现汉字。数字用阿拉伯数字。句号用英文句点，逗号用英文逗号。",
}
_GLUE = {
    "en": {"以后": "after"},
    "ko": {"以后": "이후", "金刚经": "금강경", "金刚": "금강", "法华经": "법화경"},
    "ja": {"介绍": "紹介"},
}
_DIG = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3,
    "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_NUM = set("零〇一二两三四五六七八九十百千万亿")
_THINK = re.compile(r"<think>.*?</think>|<reasoning>.*?</reasoning>|/no_think\b|</?think>|</?reasoning>", re.I | re.S)


def output_rule(code: str) -> str:
    """加在大模型用户提示末尾。中文不加。"""
    item = LANGS.get(code or "")
    if not item or code == "zh":
        return ""
    return (
        "观众的问题是中文。这一段口播必须全部用%s，不要夹中文。%s\n"
        % (item["label"], _PROPER.get(code, ""))
    )


def detect_lang(text: str) -> str:
    """看口播实际写成了哪一国。认不出返回空。"""
    raw = text or ""
    han = len(_HAN.findall(raw))
    lat = len(_LAT.findall(raw))
    kana = len(_KANA.findall(raw))
    hang = len(_HANG.findall(raw))
    if hang >= 2 and hang >= han:
        return "ko"
    if kana >= 2:
        return "ja"
    if lat >= 8 and lat > han * 2:
        return "en"
    if han >= 2 and kana == 0 and hang == 0:
        return "zh"
    return ""


def matches_lang(text: str, code: str) -> bool:
    """英语口播里不能再夹汉字。日语靠假名，韩语靠谚文。"""
    got = detect_lang(text)
    if code == "en":
        return got == "en" and not _HAN.search(text or "")
    if code == "ko":
        return got == "ko" and len(_HAN.findall(text or "")) < 2
    return got == code


def qwen_language_of(text: str, code: str = "") -> str:
    """qwen-tts 的 language 字段。认不出就交给模型 Auto。"""
    picked = code if code in LANGS else detect_lang(text)
    item = LANGS.get(picked or "")
    return item["qwen"] if item else "Auto"


def _cn_to_int(token: str) -> int:
    """十四 → 14，二零二六 → 2026，二千零二十六 → 2026。"""
    if not any(ch in "十百千万亿" for ch in token):
        return int("".join(str(_DIG[ch]) for ch in token))
    total = 0
    section = 0
    number = 0
    for ch in token:
        if ch in _DIG:
            number = _DIG[ch]
            continue
        unit = {"十": 10, "百": 100, "千": 1000}.get(ch)
        if unit:
            section += (number or 1) * unit
            number = 0
            continue
        scale = 10000 if ch == "万" else 100000000
        total += (section + number) * scale
        section = 0
        number = 0
    return total + section + number


def _convert_cn_numbers(text: str) -> str:
    """只改成段的汉字数字。单独一个「一」不动。"""
    out = []
    i = 0
    raw = text or ""
    while i < len(raw):
        if raw[i] not in _NUM:
            out.append(raw[i])
            i += 1
            continue
        j = i + 1
        while j < len(raw) and raw[j] in _NUM:
            j += 1
        token = raw[i:j]
        use = any(ch in "十百千万亿" for ch in token) or (
            len(token) >= 2 and all(ch in _DIG for ch in token)
        )
        out.append(str(_cn_to_int(token)) if use else token)
        i = j
    return "".join(out)


def _ordinals(text: str, code: str) -> str:
    def _en(match):
        n = match.group(1)
        if n.endswith("1") and not n.endswith("11"):
            suf = "st"
        elif n.endswith("2") and not n.endswith("12"):
            suf = "nd"
        elif n.endswith("3") and not n.endswith("13"):
            suf = "rd"
        else:
            suf = "th"
        return n + suf

    if code == "en":
        return re.sub(r"第(\d+)(?:届|回)", _en, text)
    if code == "ko":
        return re.sub(r"第(\d+)(?:届|回)", r"제\1회", text)
    return text


def _normalize_expo(text: str, code: str) -> str:
    name = _NAME.get(code) or ""
    if not name or name in (text or ""):
        return text or ""
    if code == "ko":
        return re.sub(r"(?<!중국 )박물관\s*엑스포", name, text)
    if code == "en":
        return re.sub(r"\b[Bb]obohui\b", name, text)
    return text


def polish_spoken(text: str, code: str = "zh") -> str:
    """去掉思考标记，换上专名，英语和韩语里的汉字数字改成阿拉伯数字。"""
    code = code if code in LANGS else "zh"
    spoken = _THINK.sub("", text or "")
    spoken = spoken.replace("博博会", _NAME.get(code) or "博博会")
    for old, new in _GLUE.get(code, {}).items():
        spoken = spoken.replace(old, new)
    if code in ("en", "ko"):
        spoken = _convert_cn_numbers(spoken)
        spoken = _ordinals(spoken, code)
    spoken = _normalize_expo(spoken, code)
    spoken = re.sub(r"[ \t]*\n+[ \t]*", " ", spoken)
    spoken = re.sub(r"[ \t]{2,}", " ", spoken)
    spoken = re.sub(r"\s+([。！？，、,.!?])", r"\1", spoken)
    return spoken.strip()


def _still_dirty(text: str, code: str) -> bool:
    if code == "en":
        return _HAN.search(text or "") is not None
    if code == "ko":
        return len(_HAN.findall(text or "")) >= 2
    if code == "ja":
        return "博博会" in (text or "")
    return False


def _translate(text: str, code: str) -> str:
    from llm.client import llm_collect
    label = LANGS[code]["label"]
    raw = llm_collect([
        {"role": "user", "content": (
            "把下面的口播译成%s。只输出译文，不要解释，不要加引号，不要输出思考过程。"
            "数字用阿拉伯数字，不要写汉字数字。%s\n%s"
            % (label, _PROPER.get(code, ""), text)
        )},
    ], max_tokens=400)
    return _THINK.sub("", raw or "").strip()


def _scrub(text: str, code: str) -> str:
    """译文里还夹着汉字时再改一版。专名先收成 EXPO，避免被改掉。"""
    from llm.client import llm_collect
    name = _NAME.get(code) or ""
    masked = (text or "").replace(name, "EXPO") if name else (text or "")
    label = LANGS[code]["label"]
    raw = llm_collect([
        {"role": "user", "content": (
            "把下面口播改成纯%s。不要留下任何汉字。数字用阿拉伯数字。"
            "EXPO 原样保留。只输出正文。\n%s" % (label, masked)
        )},
    ], max_tokens=400)
    out = _THINK.sub("", raw or "").strip()
    if name:
        out = out.replace("EXPO", name)
    return out


def needs_translation(text: str, code: str = "zh") -> bool:
    """短分句还看不出语种时不翻译。翻译会再打一次大模型，把正在生成的长句掐断。"""
    spoken = (text or "").strip()
    code = code if code in LANGS else "zh"
    if not spoken or matches_lang(spoken, code):
        return False
    if detect_lang(spoken) == "":
        return code == "en" and _HAN.search(spoken) is not None
    return True


def prepare_spoken(text: str, code: str = "zh") -> str:
    """换成目标语，并清掉汉字数字、思考标记和没译完的专名。"""
    code = code if code in LANGS else "zh"
    spoken = polish_spoken(text or "", code)
    if not spoken or code == "zh":
        return spoken
    if needs_translation(spoken, code):
        try:
            spoken = polish_spoken(_translate(spoken, code) or spoken, code)
        except Exception:
            return spoken
    if _still_dirty(spoken, code):
        try:
            spoken = polish_spoken(_scrub(spoken, code) or spoken, code)
        except Exception:
            pass
    return spoken
