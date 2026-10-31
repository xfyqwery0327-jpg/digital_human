# -*- coding: utf-8 -*-
"""口播文本：合并短句、年代/数量转中文数字，避免结巴和「一八九七年」被念成洋数字。"""
from __future__ import annotations
import re

_DIGITS = "零一二三四五六七八九"

_YEAR_RE = re.compile(r"(?<![\d.])(\d{3,4})年")
# 区间至少一侧带小数：1-1.4米 → 一到一点四米。纯整数区间留给 _NUM_RE，避免 2024-09-21 被误读。
_RANGE_RE = re.compile(
    r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*[-~～—–]\s*(\d{1,3}(?:\.\d+)?)(?![\d.])"
)
_DEC_RE = re.compile(r"(?<![\d.])(\d{1,4})\.(\d+)(?![\d.])")
_NUM_RE = re.compile(r"(?<![\d.])(\d{1,4})(?![\d.])")


def _year_to_zh(num: str) -> str:
    """年份逐位读：1897年 → 一八九七年。"""
    return "".join(_DIGITS[int(c)] for c in num) + "年"


def _frac_to_zh(frac: str) -> str:
    """小数部分逐位读：4 → 四，04 → 零四。"""
    return "".join(_DIGITS[int(c)] for c in frac if c.isdigit()) or "零"


def _num_token_to_zh(token: str) -> str:
    token = (token or "").strip()
    if "." in token:
        a, b = token.split(".", 1)
        a = a or "0"
        return _int_to_zh(int(a)) + "点" + _frac_to_zh(b)
    return _int_to_zh(int(token))


def _int_to_zh(n: int) -> str:
    if n < 0:
        return "负" + _int_to_zh(-n)
    if n < 10:
        return _DIGITS[n]
    if n < 20:
        return "十" + (_DIGITS[n - 10] if n > 10 else "")
    if n < 100:
        a, b = divmod(n, 10)
        return _DIGITS[a] + "十" + (_DIGITS[b] if b else "")
    if n < 1000:
        a, rest = divmod(n, 100)
        s = _DIGITS[a] + "百"
        if rest == 0:
            return s
        if rest < 10:
            return s + "零" + _DIGITS[rest]
        return s + _int_to_zh(rest)
    if n < 10000:
        a, rest = divmod(n, 1000)
        s = _DIGITS[a] + "千"
        if rest == 0:
            return s
        if rest < 100:
            return s + "零" + _int_to_zh(rest)
        return s + _int_to_zh(rest)
    return str(n)


def to_spoken_zh(text: str) -> str:
    """字幕和口播共用：年代逐位、小数/区间、其余四位以内整数转中文读法。"""
    if not text:
        return text

    def _year(m):
        return _year_to_zh(m.group(1))

    out = _YEAR_RE.sub(_year, text)

    def _range(m):
        a, b = m.group(1), m.group(2)
        if "." not in a and "." not in b:
            return m.group(0)
        return _num_token_to_zh(a) + "到" + _num_token_to_zh(b)

    out = _RANGE_RE.sub(_range, out)

    def _dec(m):
        return _int_to_zh(int(m.group(1))) + "点" + _frac_to_zh(m.group(2))

    out = _DEC_RE.sub(_dec, out)

    def _num(m):
        return _int_to_zh(int(m.group(1)))

    out = _NUM_RE.sub(_num, out)
    # CosyVoice 把「〇」读错（如二〇二六年）。口播前直接换成「零」。
    return out.replace("〇", "零")


PUNCT_SPLIT = r"[，。！？；、,.!?]"
_STRONG_END = "。！？；.!?"


# 中文用全角句读。英语、韩语用半角。日语用句号和顿号，不把英文句点当句界。
_MARKS = {
    "zh": "。！？；，、",
    "en": ".!?;,",
    "ja": "。！？、",
    "ko": ".!?,",
}
_ABBREV = re.compile(
    r"(?:Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|No|vs|etc|U\.S|U\.K|a\.m|p\.m)\.$",
    re.I,
)


def sentence_marks(lang: str = "zh") -> str:
    return _MARKS.get(lang or "zh", _MARKS["zh"])


def is_sentence_break(text: str, index: int, lang: str = "zh") -> bool:
    """这个位置是不是该国的句读。小数、英文缩写和省略号不当句界。"""
    raw = text or ""
    if index < 0 or index >= len(raw):
        return False
    ch = raw[index]
    if ch not in sentence_marks(lang):
        return False
    if ch in ".,":
        prev_d = index > 0 and raw[index - 1].isdigit()
        next_d = index + 1 < len(raw) and raw[index + 1].isdigit()
        if prev_d and next_d:
            return False
    if ch == "." and index + 1 < len(raw) and raw[index + 1] == ".":
        return False
    if (lang or "zh") == "en" and ch == ".":
        if _ABBREV.search(raw[max(0, index - 8):index + 1]):
            return False
        if index > 0 and raw[index - 1].isupper() and (index == 1 or not raw[index - 2].isalpha()):
            nxt = raw[index + 1] if index + 1 < len(raw) else ""
            if nxt in ("", " "):
                return False
    return True


def find_sentence_end(buf: str, lang: str = "zh") -> int:
    """最早的句读位置。书名号里的标点不算句界。"""
    depth = 0
    for i, ch in enumerate(buf or ""):
        if ch == "《":
            depth += 1
            continue
        if ch == "》":
            if depth > 0:
                depth -= 1
            continue
        if depth == 0 and is_sentence_break(buf, i, lang):
            return i
    return -1


def _split_punct_keep_decimal(text: str) -> list:
    """按句读切开，但保留 1.4 这类小数点，禁止切成「1.」+「4米」。"""
    parts = []
    buf = []
    n = len(text)
    for i, ch in enumerate(text):
        buf.append(ch)
        if ch in "，。！？；、!?":
            piece = "".join(buf).strip()
            if piece:
                parts.append(piece)
            buf = []
            continue
        if ch in ".,":
            prev_d = i > 0 and text[i - 1].isdigit()
            next_d = i + 1 < n and text[i + 1].isdigit()
            if prev_d and next_d:
                continue
            piece = "".join(buf).strip()
            if piece:
                parts.append(piece)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def split_by_marks(text: str, lang: str = "zh") -> list:
    """只在该国标点处断开。书名号里面不切，小数和英文缩写不切。

    中文用。！？；，、。英语、韩语用 .!?,。日语用。！？、。
    「我画的《睡莲》是晚年最重要的作品。」整句保留。
    """
    raw = (text or "").strip()
    if not raw:
        return []
    parts = []
    buf = []
    depth = 0
    for i, ch in enumerate(raw):
        buf.append(ch)
        if ch == "《":
            depth += 1
            continue
        if ch == "》":
            if depth > 0:
                depth -= 1
            continue
        if depth != 0 or not is_sentence_break(raw, i, lang):
            continue
        piece = "".join(buf).strip()
        if piece:
            parts.append(piece)
        buf = []
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts or [raw]


_JIEBA_PUNCT = set("，。！？；、,.!?:：…·．“”\"'‘’（）()【】[]《》「」『』—–~～")
_jieba_ready = False


def _jieba_cut(text: str) -> list:
    """按词切开。专名词典和检索共用一份，加载失败就用 jieba 默认词典。"""
    global _jieba_ready
    import jieba
    if not _jieba_ready:
        try:
            from config import DICT_PATH
            if DICT_PATH.exists():
                jieba.load_userdict(str(DICT_PATH))
        except Exception:
            pass
        _jieba_ready = True
    return [w for w in jieba.lcut(text or "") if w and not w.isspace()]


def _glue_jieba_punct(words: list) -> list:
    """标点不是批次边界，贴在前一个词上。书名号里的词合成一条。"""
    out = []
    for word in words:
        if not word or word.isspace():
            continue
        if out and all(ch in _JIEBA_PUNCT for ch in word):
            out[-1] += word
            continue
        out.append(word)
    merged = []
    buf = ""
    depth = 0
    for word in out:
        if depth or "《" in word or "「" in word or "『" in word:
            buf += word
            depth += word.count("《") + word.count("「") + word.count("『")
            depth -= word.count("》") + word.count("」") + word.count("』")
            if depth <= 0:
                merged.append(buf)
                buf = ""
                depth = 0
            continue
        merged.append(word)
    if buf:
        merged.append(buf)
    return merged


def jieba_pieces(text: str) -> list:
    """整段按 jieba 拆成批次条目。逗号、句号留在前一个词上，不单独成批。"""
    raw = (text or "").strip()
    if not raw:
        return []
    pieces = _glue_jieba_punct(_jieba_cut(raw))
    return pieces or [raw]


def take_jieba_ready(buf: str, flush: bool = False) -> tuple:
    """流式缓冲里已经能确定的词先交出去，最后一个词留着，等后面的字再切。

    flush=True 时把剩余整段交出去，用于 LLM 结束。
    返回 (可进 TTS 批次的词, 还要继续攒的尾巴)。
    """
    raw = buf or ""
    if not raw.strip():
        return [], ""
    pieces = jieba_pieces(raw)
    if not pieces:
        return [], ""
    if flush:
        return pieces, ""
    if len(pieces) == 1:
        return [], raw
    return pieces[:-1], pieces[-1]


def split_speak_chunks(text: str, max_chars: int = 40) -> list:
    """按 ，。！？；、 切开。完整句不粘下一段；尾句单独成段，禁止粘到上一句。"""
    raw = (text or "").strip()
    if not raw:
        return []
    parts = _split_punct_keep_decimal(raw)
    if not parts:
        return [raw]
    out = []
    buf = ""
    n = len(parts)
    for i, p in enumerate(parts):
        is_last = i == n - 1
        if not buf:
            buf = p
            continue
        # 尾段必须单独成项，禁止粘到上一句（CosyVoice 截长句会把最后一句吃掉）。
        if is_last:
            out.append(buf)
            buf = p
            continue
        # 句号类完整句、或已够长：单独成段。只把「对了，」这种弱标点碎片并进下一截。
        prev_complete = buf[-1] in _STRONG_END or len(buf) >= 8
        if prev_complete or len(buf) + len(p) > max_chars:
            out.append(buf)
            buf = p
        else:
            buf = f"{buf}{p}"
    if buf:
        out.append(buf)
    return out


def group_speak_chunks(chunks, max_chars: int = 40) -> list:
    """把已按标点切好的小句顺序拼成不超过 max_chars 的合成单元。

    给 CosyVoice3 / Qwen3-TTS 用：它们一次整句出声，调用次数越多句间空档越长。
    单个小句本身超长时原样保留，不在句中硬切。max_chars <= 0 时不拼。
    """
    items = [c.strip() for c in (chunks or []) if c and c.strip()]
    if max_chars <= 0:
        return items
    out = []
    buf = ""
    for c in items:
        if buf and len(buf) + len(c) > max_chars:
            out.append(buf)
            buf = c
        else:
            buf = f"{buf}{c}"
    if buf:
        out.append(buf)
    return out
