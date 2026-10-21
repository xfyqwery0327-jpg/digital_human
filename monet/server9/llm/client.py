# -*- coding: utf-8 -*-
"""打 llama.cpp、流式增量、洗 think。"""
import re
from typing import Tuple

THINK_RE = re.compile(r"<think>.*?</think>|<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE)
OPEN_TAG_RE = re.compile(r"<(?:think|reasoning)\b", re.IGNORECASE)
# 结尾疑似半截标签（如 "<th"、"</reas"），暂扣不发，等下一个 delta 再判定
PARTIAL_TAG_RE = re.compile(r"<[a-zA-Z/]{0,10}$")
# 硬断句符号
SENT_END = "。！？；"

# 串台修复收窄——"莫奈"不做全局替换，避免误伤"我就是莫奈"类正常回答
PERSONA_FIX = {
    "本王": "我",
    "商王武丁": "",
    "讲解员": "我",
}
PERSONA_PATTERNS = [
    (re.compile(r"莫奈(说|认为|表示|曾|的看法是)"), r"我\1"),
    (re.compile(r"(?:克洛德[·・]?|克劳德[·・]?)?莫奈是"), "我是"),
    (re.compile(r"(?:克洛德[·・]?|克劳德[·・]?)?莫奈为"), "我为"),
    (re.compile(r"他主张"), "我主张"),
    (re.compile(r"他(?:晚年)?定居"), "我定居"),
]

_OPENING_PATTERNS = [
    r"^根据.*?资料[,，：:]",
    r"^让我.*?[。！]",
    r"^这个问题.*?[。！]",
    r"^首先.*?然后.*?[。！]",
    r"^好的[，,].*?[。！]",
    r"^接下来.*?[。！]",
]

def _fix_persona(text: str) -> str:
    for wrong, right in PERSONA_FIX.items():
        if wrong in text:
            print(f"⚠️ 检测到人格串台词'{wrong}'，已替换为'{right}'")
            text = text.replace(wrong, right)
    for pat, repl in PERSONA_PATTERNS:
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
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def clean_full_answer(text: str) -> str:
    """全文清洗（存档/整段回答用）：额外删除模型口癖开场白。"""
    if not text:
        return text
    text = _strip_think(text)
    for pattern in _OPENING_PATTERNS:
        text = re.sub(pattern, "", text, count=1)
    text = _fix_persona(text)
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

def _find_sentence_end(buf: str, chars: str = SENT_END) -> int:
    earliest = -1
    for ch in chars:
        pos = buf.find(ch)
        if pos != -1 and (earliest == -1 or pos < earliest):
            earliest = pos
    return earliest


import json
from typing import List, Dict
import requests
from config import LLM_URL

def _llm_collect(messages: List[Dict]) -> str:
    payload = {"messages": messages, "temperature": 0.05, "max_tokens": 300, "stream": True}
    resp = requests.post(LLM_URL, json=payload, timeout=60, stream=True)
    resp.raise_for_status()
    full_raw = ""
    try:
        for line in resp.iter_lines(decode_unicode=True):
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
