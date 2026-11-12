# -*- coding: utf-8 -*-
"""全息柜观众实录：问句对上就照搬回答，不经大模型改写。"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

_PUNCT = re.compile(r"[\s，。！？、；：,.!?;:\"“”‘’'（）()【】\[\]《》·・…—\-、]+")
_ADDR = re.compile(
    r"^(?:嗨|哈喽|你好呀|你好|您好)[，,\s]*(?:莫奈|克劳德|妇好|鸠摩罗什|罗什)?[，,\s]*"
)
_SPEAKER = re.compile(r"^(?:鸠摩罗什|莫奈|妇好)[：:]\s*")
_SECTION = re.compile(r"^[一二三四五六七八九十百]+、")
_AUDIENCE = re.compile(r"^观众\d+[：:]\s*(.*)$")
_FIGURE = re.compile(r"^图\d+")
_EDITOR = re.compile(r"精简至原文|核心信息点完整保留|回复内容精简")
_CLOSING = re.compile(r"讲到这儿|拜拜|阿弥陀佛|善哉善哉")

# 问句对上实录才照搬。短问候、天气这类不够像某一问。
HIT_SCORE = 0.72
# 妇好展台四条原稿的换种说法。模糊分不够时才用，避免压过更像的旧问句。
_LOOSE_BOOTH = (
    (re.compile(r"(?:你|您|妇好).{0,6}占卜"), "听说您会占卜"),
    (re.compile(r"(?:你|您).{0,8}(?:酿酒|制酒)|酿酒吗|制酒吗"), "听说您还会制酒"),
    (re.compile(r"战役.{0,10}(?:印象|深刻)|(?:印象最深|最深刻).{0,8}战役"), "印象最深刻"),
    (re.compile(
        r"(?:详细|仔细|具体).{0,4}(?:讲讲|讲一讲|说说|介绍).{0,8}鸮尊"
        r"|妇好鸮尊是什么"
        r"|介绍(?:一下|下)?.{0,4}妇好鸮尊"
        r"|讲讲(?:这件)?(?:妇好)?鸮尊"
    ), "详细讲讲这件妇好鸮尊"),
)


def kb_transcript_path(persona_id: str) -> Path:
    from config import KB_DIR
    return Path(KB_DIR) / (persona_id or "") / "transcript.json"


def normalize_question(text: str) -> str:
    raw = (text or "").strip()
    raw = _ADDR.sub("", raw)
    raw = re.sub(r"^观众\d+[：:]\s*", "", raw)
    return _PUNCT.sub("", raw)


def question_score(query: str, recorded: str) -> float:
    """1 是同一问。太短的话不记分，避免「你好」撞上长问句。"""
    left = normalize_question(query)
    right = normalize_question(recorded)
    if len(left) < 6 or len(right) < 6:
        return 0.0
    if left == right:
        return 1.0
    ratio = SequenceMatcher(None, left, right).ratio()
    if left in right or right in left:
        cover = min(len(left), len(right)) / max(len(left), len(right))
        if cover >= 0.45 or min(len(left), len(right)) >= 8:
            ratio = max(ratio, 0.74 + 0.26 * cover)
    return ratio


def _loose_booth_answer(query: str, rows) -> str:
    """换种说法仍念妇好那四条原稿。问句太短、模糊分不够时才走到这里。"""
    q = normalize_question(query)
    if len(q) < 4:
        return ""
    needle = ""
    for pattern, anchor in _LOOSE_BOOTH:
        if pattern.search(q):
            needle = anchor
            break
    if not needle:
        return ""
    for row in rows or []:
        recorded = normalize_question(row.get("question") or "")
        answer = (row.get("answer") or "").strip()
        if answer and needle in recorded:
            return answer
    return ""


def match_transcript(query: str, rows) -> str:
    best = ""
    best_score = 0.0
    for row in rows or []:
        recorded = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not recorded or not answer:
            continue
        score = question_score(query, recorded)
        if score > best_score:
            best_score = score
            best = answer
    if best_score >= HIT_SCORE:
        return best
    return _loose_booth_answer(query, rows)


_cache = {}


def transcript_reply(query: str, persona=None) -> str:
    pid = getattr(persona, "id", None) or ""
    if not pid:
        try:
            from persona.runtime import active
            pid = active().id
        except Exception:
            pid = ""
    path = kb_transcript_path(pid)
    if not path.is_file():
        return ""
    stamp = path.stat().st_mtime
    cached = _cache.get(pid)
    if not cached or cached[0] != stamp:
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        cached = (stamp, rows if isinstance(rows, list) else [])
        _cache[pid] = cached
    return match_transcript(query, cached[1])


def _drop_banner(paragraphs) -> list:
    """文稿标题和编辑说明不进口播。"""
    kept = []
    for para in paragraphs or []:
        if len(para) <= 40 and re.search(r"自述|面对面|十件珍宝|十幅光影|十部经典", para):
            continue
        kept.append(para)
    return kept


def paragraphs_of_docx(path: Path) -> list:
    import zipfile
    from xml.etree import ElementTree as ET
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))
    rows = []
    for para in root.findall(".//w:p", ns):
        text = "".join(t.text or "" for t in para.findall(".//w:t", ns)).strip()
        if text and not _EDITOR.search(text):
            rows.append(text)
    return rows


def clean_answer(parts) -> str:
    text = "".join(p.strip() for p in parts if p and p.strip())
    return _SPEAKER.sub("", text).strip()


_LABELED_Q = re.compile(r"^观众提问[：:]\s*(.*)$")
_LABELED_A = re.compile(r"^(?:妇好|莫奈|鸠摩罗什)?台词[：:]\s*(.*)$")


def parse_labeled_qa(text: str) -> list:
    """「观众提问 / 台词」对照稿。一问一句，返回 [{question, answer}]。"""
    pairs = []
    question = None
    answer_parts = []

    def flush():
        nonlocal question, answer_parts
        answer = "".join(part.strip() for part in answer_parts if part and part.strip())
        if question and answer:
            pairs.append({"question": question, "answer": answer})
        question = None
        answer_parts = []

    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        asked = _LABELED_Q.match(line)
        if asked:
            flush()
            question = asked.group(1).strip()
            continue
        spoken = _LABELED_A.match(line)
        if spoken and question:
            answer_parts.append(spoken.group(1))
            continue
        if question and answer_parts:
            answer_parts.append(line)
    flush()
    return [row for row in pairs if row["question"] and row["answer"]]


def parse_transcript(paragraphs) -> list:
    """观众问答。返回 [{question, answer}]。开场和结束语不放进可照搬的问句。"""
    paragraphs = _drop_banner(paragraphs)
    pairs = []
    current_q = None
    current_a = []
    started = False
    for para in paragraphs or []:
        matched = _AUDIENCE.match(para)
        if matched:
            if current_q is not None:
                pairs.append({"question": current_q, "answer": clean_answer(current_a)})
            current_q = matched.group(1).strip()
            current_a = []
            started = True
            continue
        if not started or _SECTION.match(para):
            continue
        current_a.append(para)
    if current_q is not None:
        if len(current_a) >= 2 and _CLOSING.search(current_a[-1]):
            current_a = current_a[:-1]
        answer = clean_answer(current_a)
        if answer:
            pairs.append({"question": current_q, "answer": answer})
    return [row for row in pairs if row["question"] and row["answer"]]


def parse_transcript_aside(paragraphs) -> list:
    """问答实录里的开场和结束语。不是某一问的原句，仍放进检索。"""
    paragraphs = _drop_banner(paragraphs)
    intro = []
    after = []
    seen = False
    for para in paragraphs or []:
        if _AUDIENCE.match(para):
            seen = True
            after = []
            continue
        if _SECTION.match(para):
            continue
        if not seen:
            intro.append(para)
        else:
            after.append(para)
    frames = []
    intro_text = "".join(p.strip() for p in intro if p.strip())
    if intro_text:
        frames.append({"title": "见面", "text": intro_text})
    if len(after) >= 2 and _CLOSING.search(after[-1]):
        frames.append({"title": "结束语", "text": after[-1].strip()})
    return frames


def parse_narrative(paragraphs) -> list:
    """自述按「一、二、三」成段。图注和编辑说明不进正文。"""
    paragraphs = _drop_banner(paragraphs)
    sections = []
    intro = []
    title = None
    body = []

    def flush():
        nonlocal title, body
        text = "".join(p.strip() for p in body if p and p.strip())
        if title and text:
            sections.append({"title": title, "text": text})
        elif text and title is None:
            sections.append({"title": "开场", "text": text})
        title = None
        body = []

    for para in paragraphs or []:
        if _FIGURE.match(para):
            continue
        if _SECTION.match(para):
            flush()
            title = re.sub(r"^[一二三四五六七八九十百]+、", "", para).strip() or para
            continue
        if title is None:
            intro.append(para)
        else:
            body.append(para)
    flush()
    intro_text = "".join(p.strip() for p in intro if p and not _FIGURE.match(p)).strip()
    if intro_text:
        sections.insert(0, {"title": "开场", "text": intro_text})
    return [row for row in sections if row.get("text")]
