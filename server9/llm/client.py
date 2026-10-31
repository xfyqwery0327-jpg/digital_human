# -*- coding: utf-8 -*-
"""打 llama.cpp、流式增量、洗 think。"""
import re
from typing import Tuple

THINK_RE = re.compile(r"<think>.*?</think>|<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE)
OPEN_TAG_RE = re.compile(r"<(?:think|reasoning)\b", re.IGNORECASE)
# 结尾疑似半截标签（如 "<th"、"</reas"），暂扣不发，等下一个 delta 再判定
PARTIAL_TAG_RE = re.compile(r"<[a-zA-Z/]{0,10}$")
# 硬断句符号
SENT_END = "。！？；，、"

_OPENING_PATTERNS = [
    r"^根据.*?资料[,，：:]",
    r"^让我.*?[。！]",
    r"^这个问题.*?[。！]",
    r"^首先.*?然后.*?[。！]",
    r"^好的[，,].*?[。！]",
    r"^接下来.*?[。！]",
]

def _fix_persona(text: str) -> str:
    """只使用当前请求人设自己的替换表，不做莫奈全局兜底。"""
    try:
        from persona.runtime import active
        p = active()
    except Exception:
        return text
    fixes = list(p.persona_fix or [])
    patterns = []
    for pat, repl in (p.persona_patterns or []):
        try:
            patterns.append((re.compile(pat), repl))
        except re.error:
            continue
    for wrong, right in fixes:
        if wrong and wrong in text:
            print(f"⚠️ 检测到人格串台词'{wrong}'，已替换为'{right}'")
            text = text.replace(wrong, right)
    for pat, repl in patterns:
        text = pat.sub(repl, text)
    return text

def _strip_think(text: str) -> str:
    text = THINK_RE.sub("", text)
    text = re.sub(r"</?think>|</?reasoning>", "", text, flags=re.IGNORECASE)
    return text

def clean_stream_sentence(text: str) -> str:
    """逐句清洗（流式 TTS 用）：只做 think 标签清理 + 串台修复 + 空白收敛。
    开场白删除规则只针对全文开头，逐句应用会误删正文，故不在此使用。"""
    if not text:
        return text
    text = _strip_think(text)
    text = _fix_persona(text)
    from tts.speaktext import to_spoken_zh
    text = to_spoken_zh(text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def strip_identity_opening(text: str, persona=None) -> str:
    """去掉每轮自我介绍套话（如妇好开口必说女将军）。只剥全文/首句开头。"""
    if not text:
        return text
    try:
        from persona.runtime import active
        p = persona if persona is not None else active()
        patterns = list(getattr(p, "opening_strip_patterns", None) or [])
    except Exception:
        patterns = []
    out = text.strip()
    for _ in range(3):
        changed = False
        for pat in patterns:
            try:
                out2 = re.sub(pat, "", out, count=1)
            except re.error:
                continue
            if out2 != out:
                out = out2.strip()
                changed = True
                break
        if not changed:
            break
    return out


_ASK_SENT = re.compile(r"[^。！？]+[。！？]?")


def shorten_ask_reply(text: str, max_sentences: int = 2) -> str:
    """ask 口播只留前两句。没有句号时整段保留。"""
    t = (text or "").strip()
    if not t:
        return t
    parts = [p.strip() for p in _ASK_SENT.findall(t) if p.strip()]
    if len(parts) <= max_sentences:
        return t
    return "".join(parts[:max_sentences]).strip()


def spoken_len(text: str) -> int:
    """口播字数：去掉空白，标点计入。"""
    return len(re.sub(r"\s+", "", text or ""))


def fit_spoken_chars(text: str, lo: int = 150, hi: int = 200) -> str:
    """超长时收到 hi 字以内，并尽量停在句末。不足 lo 字时原样返回，交给调用方重写。"""
    t = re.sub(r"\s+", "", (text or "").strip())
    if len(t) <= hi:
        return t
    acc = ""
    for part in _ASK_SENT.findall(t):
        part = part.strip()
        if not part:
            continue
        if len(acc) + len(part) > hi:
            break
        acc += part
    if lo <= len(acc) <= hi:
        return acc
    cut = t[:hi]
    for i in range(len(cut) - 1, lo - 2, -1):
        if cut[i] in "。！？":
            return cut[:i + 1]
    body = cut[:hi - 1].rstrip("，、；：,;:")
    return (body or cut[:hi - 1]) + "。"


def clean_full_answer(text: str) -> str:
    """全文清洗（存档/整段回答用）：额外删除模型口癖开场白。"""
    if not text:
        return text
    text = _strip_think(text)
    for pattern in _OPENING_PATTERNS:
        text = re.sub(pattern, "", text, count=1)
    text = strip_identity_opening(text)
    text = _fix_persona(text)
    from tts.speaktext import to_spoken_zh
    text = to_spoken_zh(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def extract_clean_delta(full_raw: str, clean_pos: int) -> Tuple[str, int]:
    """流式安全增量：删除完整 think 块，扣住未闭合标签及结尾半截标签。
    返回 (可安全发送的新文本, 新的绝对游标)。"""
    clean_text = THINK_RE.sub("", full_raw)
    m = OPEN_TAG_RE.search(clean_text)
    safe_end = m.start() if m else len(clean_text)
    pm = PARTIAL_TAG_RE.search(clean_text, 0, safe_end)
    if pm and pm.start() < safe_end:
        safe_end = pm.start()
    if safe_end <= clean_pos:
        return "", clean_pos
    return clean_text[clean_pos:safe_end], safe_end

def _find_sentence_end(buf: str, chars: str = SENT_END, lang: str = "") -> int:
    """最早的句读位置。书名号里的标点不算句界。lang 为空时仍按传入的符号表。"""
    if lang:
        from tts.speaktext import find_sentence_end
        return find_sentence_end(buf, lang)
    depth = 0
    for i, ch in enumerate(buf or ""):
        if ch == "《":
            depth += 1
            continue
        if ch == "》":
            if depth > 0:
                depth -= 1
            continue
        if depth == 0 and ch in chars:
            return i
    return -1


import json
from typing import List, Dict, Iterator
import requests
from config import LLM_URL


def iter_llm_sse(resp) -> Iterator[str]:
    """按 UTF-8 读 llama.cpp 的 SSE 行。

    Content-Type 为 text/event-stream 且没有 charset 时，requests 按 RFC
    把 text/* 标成 ISO-8859-1。模型吐出的 UTF-8 中文会被拆成拉丁字符和控制符，
    前端字幕就是菱形、问号。application/json 会按 UTF-8 解，所以另一路 answer 仍是中文。
    """
    resp.encoding = "utf-8"
    for line in resp.iter_lines(decode_unicode=True):
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        yield line


def _llm_collect(messages: List[Dict], max_tokens: int = 300) -> str:
    payload = {"messages": messages, "temperature": 0.05,
               "max_tokens": int(max_tokens), "stream": True}
    resp = requests.post(LLM_URL, json=payload, timeout=60, stream=True)
    resp.raise_for_status()
    full_raw = ""
    try:
        for line in iter_llm_sse(resp):
            if not line or not line.startswith("data: "):
                continue
            chunk = line[6:]
            if chunk == "[DONE]":
                break
            try:
                delta = json.loads(chunk)["choices"][0].get("delta", {})
            except Exception:
                continue
            content = delta.get("content", "")
            if content:
                full_raw += content
    finally:
        try:
            resp.close()
        except Exception:
            pass
    return full_raw


llm_collect = _llm_collect
