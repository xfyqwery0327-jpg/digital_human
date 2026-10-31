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


_PROPER = {
    "en": "博博会是专名，必须写成 China Museum Expo，不要写成博博会。宽创国际写成 Kuanchuang。句号用英文句点，逗号用英文逗号。",
    "ja": "博博会是专名，必须写成中国博物館エキスポ，不要写成博博会。句号用。，停顿用、。",
    "ko": "博博会是专名，必须写成중국 박물관 엑스포，不要写成博博会，整段不要出现汉字。句号用英文句点，逗号用英文逗号。",
}


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


def _translate(text: str, code: str) -> str:
    from llm.client import clean_full_answer, llm_collect
    label = LANGS[code]["label"]
    raw = llm_collect([
        {"role": "user", "content": (
            "把下面的口播译成%s。只输出译文，不要解释，不要加引号。%s\n%s\n/no_think"
            % (label, _PROPER.get(code, ""), text)
        )},
    ], max_tokens=400)
    return (clean_full_answer(raw) or "").strip()


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
    """已经是目标语就原样用。不是的话再译一版，然后交给 TTS。"""
    spoken = (text or "").strip()
    code = code if code in LANGS else "zh"
    if not spoken or not needs_translation(spoken, code):
        return spoken
    try:
        translated = _translate(spoken, code)
    except Exception:
        return spoken
    return translated or spoken
