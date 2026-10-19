# -*- coding: utf-8 -*-
"""启动、CORS、HTTP / WebSocket 路由。"""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import *
import session as session_mod
from session import persistence, active_sessions, cleanup_inactive_sessions
from rag.intent import memory_system
from rag.index import load_index
from rag import index as rag
from rag.search import search
from rag.scope import is_other_exhibition_query, locked_category, answer_leaves_exhibition, refuse_text
from tts import cosyvoice as tts
from tts.cosyvoice import load_cosyvoice
from tts.engine import tts_engine_name
from robot.hub import robot_hub
from robot.api import router as robot_router
from persona.api import router as persona_router
from kb.api import router as kb_router
from persona.stage import stage
from persona.runtime import resolve, bind, call_bound
from persona.registry import registry
from ws.tts import websocket_tts_stream
from ws.chat import websocket_chat
from ws.hub import tts_fanout, accept_websocket


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime_log("startup.log", "[BOOT] lifespan start root=%s" % PROJECT_ROOT)
    print("\n" + "=" * 60)
    print("🚀 正在启动服务...")
    print("=" * 60)
    try:
        load_index()
        runtime_log("startup.log", "[BOOT] load_index done tts=%s" % tts.tts_ready())
    except Exception as e:
        runtime_log("startup.log", "[BOOT] load_index FAIL %r" % (e,))
        print(f"⚠️ 知识库/TTS 加载失败，以降级模式继续: {e}")
        import traceback
        traceback.print_exc()
        from persona.stage import stage as _stage
        from persona.registry import registry as _reg
        if not _reg._items:
            _reg.load()
        _stage.bootstrap()
        if not tts.cosyvoice_model:
            try:
                load_cosyvoice()
            except Exception as e2:
                runtime_log("startup.log", "[BOOT] load_cosyvoice FAIL %r" % (e2,))
    session_mod.cleanup_task = asyncio.create_task(cleanup_inactive_sessions())
    session_mod.robot_idle_task = asyncio.create_task(robot_hub.idle_watch_loop())
    print(f"⏰ 启动会话清理任务(间隔 {SESSION_CLEANUP_INTERVAL}s,超时 {SESSION_TIMEOUT}s)")
    print(f"🤖 机器人互动已启用 idle={ROBOT_IDLE_TIMEOUT}s（只收 ask，不回调对方）")
    if SESSION_PERSISTENCE:
        print(f"📂 发现 {len(persistence.list_sessions())} 个持久化会话")
    runtime_log("startup.log", "[BOOT] ready tts=%s err=%s" % (
        tts.tts_ready(), tts._tts_last_error or "-"))
    print("✅ 启动完成\n")
    yield
    print("\n正在关闭服务...")
    if robot_hub.session_id:
        try:
            await robot_hub.release("server_shutdown")
        except Exception as e:
            print(f"[ROBOT] 关闭时释放会话失败: {e}")
    for task in (session_mod.cleanup_task, session_mod.robot_idle_task):
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    print(f"💾 保存 {len(active_sessions)} 个活跃会话...")
    for s in active_sessions.values():
        try:
            persistence.save_session(s)
            if ENABLE_MEMORY:
                await memory_system.save_session_summary(s)
        except Exception as e:
            print(f"⚠️ 保存会话 {s.session_id} 失败: {e}")
    if tts.tts_executor:
        tts.tts_executor.shutdown(wait=False, cancel_futures=True)
    print("✅ 关闭完成")


app = FastAPI(title="Digital Human Guide v8.5.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _ws_chat_default(websocket: WebSocket):
    await websocket_chat(websocket, None)


async def _ws_tts_default(websocket: WebSocket):
    await websocket_tts_stream(websocket, None)


async def _ws_catchall(websocket: WebSocket, rest: str):
    path = websocket.scope.get("path", "") or ""
    runtime_log("ws.log", "[WS] catchall path=%s rest=%s client=%s" % (
        path, rest, websocket.client))
    low = path.lower().rstrip("/")
    parts = [p for p in path.split("/") if p]
    if low.endswith("/tts") or low.endswith("tts"):
        pid = parts[-2] if len(parts) >= 3 and parts[-2] != "ws" else None
        await websocket_tts_stream(websocket, pid)
        return
    if low.endswith("/chat") or low.endswith("chat"):
        pid = parts[-2] if len(parts) >= 3 and parts[-2] != "ws" else None
        await websocket_chat(websocket, pid)
        return
    await accept_websocket(websocket)
    await websocket.close(code=1008, reason="unknown ws path")


app.websocket("/ws/chat")(_ws_chat_default)
app.websocket("/ws/chat/")(_ws_chat_default)
app.websocket("/ws/tts")(_ws_tts_default)
app.websocket("/ws/tts/")(_ws_tts_default)
app.websocket("/ws/{persona_id}/chat")(websocket_chat)
app.websocket("/ws/{persona_id}/chat/")(websocket_chat)
app.websocket("/ws/{persona_id}/tts")(websocket_tts_stream)
app.websocket("/ws/{persona_id}/tts/")(websocket_tts_stream)
app.websocket("/ws/{rest:path}")(_ws_catchall)
app.include_router(robot_router)
app.include_router(persona_router)
app.include_router(kb_router)


class SearchReq(BaseModel):
    query: str
    top_k: int = 2
    category: Optional[str] = None


@app.post("/search")
def api_search(req: SearchReq):
    return _search_for(registry.default_id, req)


@app.post("/api/{persona_id}/search")
def api_search_persona(persona_id: str, req: SearchReq):
    return _search_for(persona_id, req)


def _search_for(persona_id: str, req: SearchReq):
    p = resolve(persona_id, allow_default=True)
    with bind(p):
        if is_other_exhibition_query(req.query, p):
            return {"query": req.query, "results": [], "refused": True,
                    "reason": "other_exhibition", "persona_id": p.id}
        results = search(req.query, req.top_k, locked_category(req.category, p), persona=p)
        threshold = MIN_SCORE_RERANK if rag.HAS_RERANKER else MIN_SCORE_RRF
        kept = [r for r in results if r["score"] >= threshold]
        return {"query": req.query, "results": kept, "persona_id": p.id}


@app.post("/tts")
async def api_tts(req: dict):
    return await _tts_for(registry.default_id, req)


@app.post("/api/{persona_id}/tts")
async def api_tts_persona(persona_id: str, req: dict):
    return await _tts_for(persona_id, req)


async def _tts_for(persona_id: str, req: dict):
    p = resolve(persona_id, allow_default=True)
    with bind(p):
        text = req.get("text", "")
        speaker = req.get("speaker", "中文男")
        if not tts.cosyvoice_model:
            return {"error": "TTS 未启用"}
        if not text:
            return {"error": "text 不能为空"}
        if answer_leaves_exhibition(text, p):
            text = refuse_text(p)
        try:
            loop = asyncio.get_running_loop()
            audio_b64 = await loop.run_in_executor(
                tts.tts_executor,
                lambda t=text, s=speaker, per=p: call_bound(
                    tts.text_to_base64_audio, t, s, persona=per))
            return {"audio": audio_b64, "format": "wav", "persona_id": p.id}
        except Exception as e:
            return {"error": str(e)}


def _llm_ok() -> bool:
    try:
        import socket
        s = socket.create_connection(("127.0.0.1", 8080), 0.3)
        s.close()
        return True
    except Exception:
        return False


@app.get("/health")
def api_health():
    p = stage.current()
    idx = rag.index
    return {"status": "healthy", "chunks": len(rag.items),
            "vectors": idx.ntotal if idx else 0,
            "tts": tts.tts_ready(), "tts_spk_registered": tts._tts_use_spk_id,
            "tts_persona": tts._voice_persona_id, "tts_spk_id": tts._voice_spk_id,
            "tts_infer": tts.tts_infer_mode(),
            "tts_prompt_mode": tts._tts_prompt_mode, "tts_stream": tts._tts_support_stream,
            "tts_engine": tts_engine_name(),
            "tts_error": tts._tts_last_error or "",
            "llm": _llm_ok(),
            "intent": rag.HAS_INTENT, "sessions": len(active_sessions),
            "robot": robot_hub.snapshot(),
            "tts_ws_clients": tts_fanout.client_count(),
            "persona": p.snapshot(),
            "personas": [x.snapshot() for x in registry.all()],
            "cosyvoice3_model": COSYVOICE3_MODEL}


def run():
    import uvicorn
    print("\n" + "=" * 60)
    print("启动数字讲解员 v8.5.0")
    print("=" * 60)
    print(f"📍 服务地址: http://{HOST}:{PORT}")
    print(f"📍 对话 WebSocket: ws://{HOST}:{PORT}/ws/{{persona_id}}/chat")
    print(f"📍 TTS WebSocket: ws://{HOST}:{PORT}/ws/{{persona_id}}/tts")
    print(f"📍 机器人互动: POST http://{HOST}:{PORT}/api/{{persona_id}}/robot/ask")
    print(f"📍 知识库工作台: http://{HOST}:{PORT}/admin/knowledge")
    if not registry._items:
        registry.load()
    names = " / ".join(f"{p.id}({p.display_name})" for p in registry.enabled())
    print(f"🎭 可用人设: {names}")
    print(f"📍 接口文档: http://{HOST}:{PORT}/docs")
    print("=" * 60 + "\n")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    run()
