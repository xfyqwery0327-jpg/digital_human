# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

from config import INDEX_DIR, KB_DIR

INDEX_FILES = ("faiss.index", "items.json", "faiss_map.json", "bm25.pkl")


def kb_root(persona_id: str) -> Path:
    p = KB_DIR / persona_id
    for name in ("sources", "draft", "_prev"):
        (p / name).mkdir(parents=True, exist_ok=True)
    return p


def sources_dir(persona_id: str) -> Path:
    return kb_root(persona_id) / "sources"


def draft_dir(persona_id: str) -> Path:
    return kb_root(persona_id) / "draft"


def prev_dir(persona_id: str) -> Path:
    return kb_root(persona_id) / "_prev"


def chunks_path(persona_id: str) -> Path:
    return kb_root(persona_id) / "chunks.json"


def live_index_dir(persona_id: str) -> Path:
    dedicated = INDEX_DIR / persona_id
    dedicated.mkdir(parents=True, exist_ok=True)
    return dedicated


def list_sources(persona_id: str) -> List[dict]:
    d = sources_dir(persona_id)
    rows = []
    for f in sorted(d.iterdir()):
        if f.is_file() and not f.name.startswith("."):
            rows.append({
                "name": f.name,
                "bytes": f.stat().st_size,
                "mtime": int(f.stat().st_mtime),
            })
    return rows


def save_source(persona_id: str, filename: str, data: bytes) -> Path:
    safe = Path(filename).name.replace("..", "")
    dest = sources_dir(persona_id) / safe
    dest.write_bytes(data)
    return dest


def delete_source(persona_id: str, filename: str) -> bool:
    dest = sources_dir(persona_id) / Path(filename).name
    if dest.exists():
        dest.unlink()
        return True
    return False


def save_chunks(persona_id: str, chunks: List[dict]) -> None:
    chunks_path(persona_id).write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunks(persona_id: str) -> List[dict]:
    p = chunks_path(persona_id)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def index_ready(path: Path) -> bool:
    return all((path / n).exists() for n in INDEX_FILES)


def copy_index(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in INDEX_FILES:
        s = src / name
        if s.exists():
            shutil.copy2(s, dest / name)


def status_payload(persona, rag_ready: bool, embedder_ok: bool) -> dict:
    pid = persona.id
    live = Path(persona.index_dir) if persona.index_dir else live_index_dir(pid)
    if not (live / "faiss.index").exists():
        live = live_index_dir(pid)
    draft = draft_dir(pid)
    chunks = load_chunks(pid)
    preview = [{
        "id": c.get("id"), "title": c.get("title"), "source": c.get("source"),
        "category": c.get("category"), "chars": len(c.get("text") or ""),
        "text": (c.get("text") or "")[:180],
    } for c in chunks[:80]]
    live_items = 0
    items_file = live / "items.json"
    if items_file.exists():
        try:
            live_items = len(json.loads(items_file.read_text(encoding="utf-8")))
        except Exception:
            live_items = 0
    published_at = None
    if (live / "faiss.index").exists():
        published_at = int((live / "faiss.index").stat().st_mtime)
    draft_ready = index_ready(draft)
    live_ready = index_ready(live)
    if live_ready:
        state = "live"
    elif draft_ready:
        state = "draft"
    elif chunks:
        state = "preview"
    elif list_sources(pid):
        state = "sources"
    else:
        state = "empty"
    return {
        "persona_id": pid,
        "display_name": persona.display_name,
        "category": (persona.categories or [f"exhibition_{pid}"])[0],
        "state": state,
        "sources": list_sources(pid),
        "chunk_count": len(chunks),
        "chunks": preview,
        "live_ready": live_ready,
        "draft_ready": draft_ready,
        "live_items": live_items,
        "published_at": published_at,
        "can_rollback": index_ready(prev_dir(pid)),
        "rag_deps": rag_ready,
        "embedder": embedder_ok,
        "live_dir": str(live),
        "draft_dir": str(draft),
    }
