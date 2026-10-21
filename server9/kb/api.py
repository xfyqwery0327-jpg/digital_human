# -*- coding: utf-8 -*-
from __future__ import annotations
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from config import ROBOT_API_TOKEN
from persona.registry import registry
from persona.stage import stage
from rag import index as rag
from kb.store import (
    copy_index, delete_source, draft_dir, live_index_dir, load_chunks,
    prev_dir, save_chunks, save_source, status_payload, INDEX_FILES, index_ready,
)
from kb.extract import chunk_text, extract_file
from kb.build import build_draft, can_build, validate_pack

router = APIRouter(tags=["knowledge"])
WEB = Path(__file__).resolve().parent / "web" / "knowledge.html"


def _verify(x_robot_token: Optional[str] = Header(default=None, alias="X-Robot-Token")):
    if ROBOT_API_TOKEN and x_robot_token != ROBOT_API_TOKEN:
        raise HTTPException(status_code=401, detail="invalid X-Robot-Token")


def _persona(persona_id: str):
    p = registry.get(persona_id)
    if not p or not p.enabled:
        raise HTTPException(status_code=404, detail=f"未知数字人: {persona_id}")
    return p


def _embedder_ok() -> bool:
    return bool(getattr(rag, "HAS_RAG_DEPS", False) and rag.embedder is not None)


@router.get("/admin/knowledge", response_class=HTMLResponse)
def admin_knowledge_page():
    if not WEB.exists():
        raise HTTPException(status_code=500, detail="工作台页面文件缺失")
    return HTMLResponse(WEB.read_text(encoding="utf-8"))


@router.get("/admin/kb/meta")
def kb_meta(_: None = Depends(_verify)):
    ok, reason = can_build()
    return {
        "current_persona_id": stage.current_id(),
        "token_required": bool(ROBOT_API_TOKEN),
        "can_build": ok,
        "build_hint": reason,
        "personas": [p.snapshot() for p in registry.enabled()],
    }


@router.get("/admin/kb/{persona_id}/status")
def kb_status(persona_id: str, _: None = Depends(_verify)):
    p = _persona(persona_id)
    return status_payload(p, getattr(rag, "HAS_RAG_DEPS", False), _embedder_ok())


@router.post("/admin/kb/{persona_id}/sources")
async def kb_upload_sources(persona_id: str, files: List[UploadFile] = File(...),
                            _: None = Depends(_verify)):
    _persona(persona_id)
    saved = []
    for f in files:
        data = await f.read()
        if not data:
            continue
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"{f.filename} 超过 20MB")
        path = save_source(persona_id, f.filename or "untitled.txt", data)
        saved.append(path.name)
    return {"ok": True, "saved": saved}


class PasteReq(BaseModel):
    title: str = "现场粘贴"
    text: str


@router.post("/admin/kb/{persona_id}/paste")
def kb_paste(persona_id: str, req: PasteReq, _: None = Depends(_verify)):
    _persona(persona_id)
    if not (req.text or "").strip():
        raise HTTPException(status_code=422, detail="正文为空")
    name = (req.title or "paste").strip() + ".txt"
    path = save_source(persona_id, name, req.text.encode("utf-8"))
    return {"ok": True, "saved": path.name}


@router.delete("/admin/kb/{persona_id}/sources/{filename}")
def kb_delete_source(persona_id: str, filename: str, _: None = Depends(_verify)):
    _persona(persona_id)
    if not delete_source(persona_id, filename):
        raise HTTPException(status_code=404, detail="文件不存在")
    return {"ok": True}


@router.post("/admin/kb/{persona_id}/parse")
def kb_parse(persona_id: str, size: int = 500, overlap: int = 80,
             _: None = Depends(_verify)):
    p = _persona(persona_id)
    cat = (p.categories or [f"exhibition_{persona_id}"])[0]
    from kb.store import sources_dir
    chunks = []
    errors = []
    for f in sorted(sources_dir(persona_id).iterdir()):
        if not f.is_file() or f.name.startswith("."):
            continue
        try:
            text = extract_file(f)
            chunks.extend(chunk_text(text, f.stem, f.name, cat, persona_id, size, overlap))
        except Exception as e:
            errors.append({"file": f.name, "error": str(e)})
    save_chunks(persona_id, chunks)
    preview = [{
        "id": c["id"], "title": c["title"], "source": c["source"],
        "category": c["category"], "chars": len(c["text"]),
        "text": c["text"][:180],
    } for c in chunks]
    return {"ok": True, "count": len(chunks), "chunks": preview, "errors": errors}


@router.post("/admin/kb/{persona_id}/build")
def kb_build(persona_id: str, _: None = Depends(_verify)):
    _persona(persona_id)
    chunks = load_chunks(persona_id)
    if not chunks:
        raise HTTPException(status_code=422, detail="请先解析预览")
    try:
        return build_draft(persona_id, chunks)
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/admin/kb/{persona_id}/pack")
async def kb_pack(persona_id: str, file: UploadFile = File(...),
                  _: None = Depends(_verify)):
    p = _persona(persona_id)
    data = await file.read()
    tmp = Path(tempfile.mkdtemp(prefix="kbpack_"))
    try:
        zpath = tmp / (file.filename or "pack.zip")
        zpath.write_bytes(data)
        extract_to = tmp / "unzip"
        extract_to.mkdir()
        if zpath.suffix.lower() == ".zip":
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(extract_to)
            folder = extract_to
            if not (folder / "faiss.index").exists():
                subs = [d for d in folder.iterdir() if d.is_dir()]
                if len(subs) == 1:
                    folder = subs[0]
        else:
            raise HTTPException(status_code=422, detail="请上传包含四件套的 zip")
        try:
            validate_pack(folder)
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        dest = draft_dir(persona_id)
        copy_index(folder, dest)
        items = (dest / "items.json").read_text(encoding="utf-8")
        import json
        save_chunks(persona_id, json.loads(items))
        return {"ok": True, "count": len(json.loads(items)), "draft_dir": str(dest)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@router.post("/admin/kb/{persona_id}/publish")
def kb_publish(persona_id: str, _: None = Depends(_verify)):
    p = _persona(persona_id)
    draft = draft_dir(persona_id)
    if not index_ready(draft):
        raise HTTPException(status_code=422, detail="没有可发布的草稿索引")
    live = live_index_dir(persona_id)
    prev = prev_dir(persona_id)
    if index_ready(live):
        copy_index(live, prev)
    copy_index(draft, live)
    p.index_dir = str(live)
    loaded = rag.reload_persona_index(p)
    return {
        "ok": True,
        "loaded": loaded,
        "live_dir": str(live),
        "message": "已发布到舞台" if loaded else "文件已写入，但本机无法热加载向量（缺 FAISS）。现场环境重启后生效。",
    }


@router.post("/admin/kb/{persona_id}/rollback")
def kb_rollback(persona_id: str, _: None = Depends(_verify)):
    p = _persona(persona_id)
    prev = prev_dir(persona_id)
    if not index_ready(prev):
        raise HTTPException(status_code=422, detail="没有可回滚的上一版")
    live = live_index_dir(persona_id)
    copy_index(prev, live)
    p.index_dir = str(live)
    loaded = rag.reload_persona_index(p)
    return {"ok": True, "loaded": loaded}


class TryReq(BaseModel):
    query: str
    top_k: int = 3


@router.post("/admin/kb/{persona_id}/try-search")
def kb_try(persona_id: str, req: TryReq, _: None = Depends(_verify)):
    p = _persona(persona_id)
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=422, detail="请输入试检索问题")
    if rag.embedder is None or not rag._persona_indexes.get(persona_id):
        chunks = load_chunks(persona_id)
        hits = []
        for c in chunks:
            if q in c["text"] or q in c.get("title", ""):
                hits.append({"title": c["title"], "score": 1.0, "text": c["text"][:160], "recall": "text"})
            if len(hits) >= req.top_k:
                break
        return {"ok": True, "mode": "substring", "results": hits,
                "hint": "向量索引未加载，当前只做原文包含匹配"}
    from rag.search import search
    from persona.runtime import bind
    with bind(p):
        hits = search(q, req.top_k, p.categories, None, persona=p)
    return {"ok": True, "mode": "rag", "results": hits}
