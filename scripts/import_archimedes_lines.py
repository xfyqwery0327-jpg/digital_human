# -*- coding: utf-8 -*-
"""把阿基米德口播台词库 docx 切成一问一块，写入 kb/archimedes/chunks.json。

动作、表情不入库，避免被念出来。建索引请再跑：

    python scripts/build_persona_index.py --persona archimedes --publish
"""
from __future__ import annotations
import json
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCX = Path(r"C:\Users\Nick Xiong\Downloads\阿基米德 数字人口播台词库1021.docx")
SOURCE_NAME = "阿基米德数字人口播台词库1021.docx"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def paragraphs(path: Path) -> list:
    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))
    lines = []
    for para in root.findall(".//w:p", NS):
        text = "".join(t.text or "" for t in para.findall(".//w:t", NS)).strip()
        if text:
            lines.append(text)
    return lines


def _level(line: str, current: str) -> str:
    if "L1" in line and "启蒙" in line:
        return "L1"
    if "L2" in line and "进阶" in line:
        return "L2"
    if "L3" in line and "深度" in line:
        return "L3"
    if line.startswith("五、"):
        return "guide"
    return current


def _take_until(lines, i, stop_prefix):
    buf = []
    while i < len(lines) and not lines[i].startswith(stop_prefix):
        buf.append(lines[i])
        i += 1
    return "".join(buf).strip(), i


def parse_chunks(lines: list) -> list:
    chunks = []
    level = ""
    i = 0
    seq = 0
    while i < len(lines):
        line = lines[i]
        level = _level(line, level)
        if line.startswith("问题："):
            question = line.split("：", 1)[1].strip()
            i += 1
            extra, i = _take_until(lines, i, "【台词】")
            if extra and not extra.startswith("【"):
                question = (question + extra).strip()
            script = ""
            if i < len(lines) and lines[i].startswith("【台词】"):
                script = lines[i][len("【台词】"):].strip()
                i += 1
                more, i = _take_until(lines, i, "【动作】")
                script = (script + more).strip()
            if script:
                seq += 1
                chunks.append(_item(seq, question, script, level or "qa"))
            continue
        if line.startswith(("待机引导语", "回答结束引导语", "通用拒答话术")):
            title = line.split("（", 1)[0].strip()
            i += 1
            script = ""
            if i < len(lines) and lines[i].startswith("【台词】"):
                script = lines[i][len("【台词】"):].strip()
                i += 1
                more, i = _take_until(lines, i, "【动作】")
                script = (script + more).strip()
            if script:
                seq += 1
                chunks.append(_item(seq, title, script, "guide"))
            continue
        if line.startswith("本文档为"):
            seq += 1
            chunks.append({
                "id": "archimedes_venue_%02d" % seq,
                "title": "展项说明",
                "text": "展项：合肥科协临展「与阿基米德对话——走进古典科学」。我在这场展里讲几何、杠杆、浮力、圆周率和叙拉古。",
                "source": SOURCE_NAME,
                "category": "exhibition_archimedes",
                "dedup_key": "%s:venue" % SOURCE_NAME,
            })
        i += 1
    return chunks


def _item(seq: int, question: str, script: str, level: str) -> dict:
    return {
        "id": "archimedes_%s_%02d" % (level.lower(), seq),
        "title": question,
        "text": "问题：%s\n台词：%s" % (question, script),
        "source": SOURCE_NAME,
        "category": "exhibition_archimedes",
        "dedup_key": "%s:%s" % (SOURCE_NAME, seq),
    }


def main():
    src = DEFAULT_DOCX
    if not src.exists():
        raise SystemExit("找不到台词库: %s" % src)
    dest_dir = ROOT / "kb" / "archimedes" / "sources"
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / SOURCE_NAME)
    chunks = parse_chunks(paragraphs(src))
    if len(chunks) < 15:
        raise SystemExit("切块过少: %s" % len(chunks))
    out = ROOT / "kb" / "archimedes" / "chunks.json"
    out.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print("chunks", len(chunks), "->", out)
    for c in chunks:
        print("-", c["id"], c["title"][:40])


if __name__ == "__main__":
    main()
