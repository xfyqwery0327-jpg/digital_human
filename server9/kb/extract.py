# -*- coding: utf-8 -*-
"""原文抽取：txt/md 必支持，docx 走 zip，pdf 有库才读。"""
from __future__ import annotations
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


def extract_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md", ".csv"):
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".docx":
        return _extract_docx(path)
    if suffix == ".pdf":
        return _extract_pdf(path)
    raise ValueError(f"不支持的文件类型: {suffix}，请用 txt / md / docx / pdf")


def _extract_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    parts = [t.text for t in root.findall(".//w:t", ns) if t.text]
    return "\n".join(parts)


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ValueError("读取 PDF 需要安装 pypdf：pip install pypdf") from e
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


_EDITOR_NOTE = re.compile(
    r"[^。\n]{0,40}没查到资料[，,。]?后续找到了再加|（?图片待补充）?"
)


def chunk_text(text: str, title: str, source: str, category: str,
               persona_id: str, size: int = 500, overlap: int = 80) -> list:
    raw = re.sub(r"\r\n?", "\n", text or "")
    raw = _EDITOR_NOTE.sub("", raw).strip()
    if not raw:
        return []
    parts = re.split(r"(?<=[。！？；\n])", raw)
    buf = ""
    blocks = []
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        if buf and len(buf) + len(piece) > size:
            blocks.append(buf.strip())
            buf = (buf[-overlap:] if overlap and len(buf) > overlap else "") + piece
        else:
            buf += piece
    if buf.strip():
        blocks.append(buf.strip())
    chunks = []
    for i, body in enumerate(blocks):
        chunks.append({
            "id": f"{persona_id}_{_slug(source)}_{i+1}",
            "title": title,
            "text": body,
            "source": source,
            "category": category,
            "dedup_key": f"{source}:{i+1}",
        })
    return chunks


def _slug(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE)[:40] or "doc"
