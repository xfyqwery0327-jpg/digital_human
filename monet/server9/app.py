# -*- coding: utf-8 -*-
"""启动、CORS、HTTP / WebSocket 路由。"""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import *
import session as session_mod
from session import persistence, active_sessions, cleanup_inactive_sessions
from rag.intent import memory_system
from rag.index import load_index
from rag import index as rag
from rag.search import search
from rag.scope import is_other_exhibition_query, locked_category, answer_leaves_exhibition
from tts import cosyvoice as tts
from robot.hub import robot_hub
from robot.api import router as robot_router
from ws.tts import websocket_tts_stream
from ws.chat import websocket_chat
from ws.hub import tts_fanout


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "=" * 60)
    print("🚀 正在启动服务...")
    print("=" * 60)
    load_index()
    session_mod.cleanup_task = asyncio.create_task(cleanup_inactive_sessions())
    session_mod.robot_idle_task = asyncio.create_task(robot_hub.idle_watch_loop())
    print(f"⏰ 启动会话清理任务(间隔 {SESSION_CLEANUP_INTERVAL}s,超时 {SESSION_TIMEOUT}s)")
    print(f"🤖 机器人互动已启用 idle={ROBOT_IDLE_TIMEOUT}s（只收 ask，不回调对方）")
    if SESSION_PERSISTENCE:
        print(f"📂 发现 {len(persistence.list_sessions())} 个持久化会话")
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


app = FastAPI(title="Monet Digital Guide v8.5.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.websocket("/ws/tts")(websocket_tts_stream)
app.websocket("/ws/chat")(websocket_chat)
app.include_router(robot_router)


class SearchReq(BaseModel):
    query: str
    top_k: int = 2
    category: Optional[str] = None


@app.post("/search")
def api_search(req: SearchReq):
    if is_other_exhibition_query(req.query):
        return {"query": req.query, "results": [], "refused": True,
                "reason": "other_exhibition"}
    results = search(req.query, req.top_k, locked_category(req.category))
    threshold = MIN_SCORE_RERANK if rag.HAS_RERANKER else MIN_SCORE_RRF
    kept = [r for r in results if r["score"] >= threshold]
    return {"query": req.query, "results": kept}


@app.post("/tts")
async def api_tts(req: dict):
    text = req.get("text", "")
    speaker = req.get("speaker", "中文男")
    if not tts.cosyvoice_model:
        return {"error": "TTS 未启用"}
    if not text:
        return {"error": "text 不能为空"}
    if answer_leaves_exhibition(text):
        text = REFUSE_TEXT
    try:
        loop = asyncio.get_running_loop()
        audio_b64 = await loop.run_in_executor(
            tts.tts_executor, tts.text_to_base64_audio, text, speaker)
        return {"audio": audio_b64, "format": "wav"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
def api_health():
    return {"status": "healthy", "chunks": len(rag.items),
            "vectors": rag.index.ntotal if rag.index else 0,
            "tts": bool(tts.cosyvoice_model), "tts_spk_registered": tts._tts_use_spk_id,
            "tts_prompt_mode": tts._tts_prompt_mode, "tts_stream": tts._tts_support_stream,
            "intent": rag.HAS_INTENT, "sessions": len(active_sessions),
            "robot": robot_hub.snapshot(),
            "tts_ws_clients": tts_fanout.client_count()}


def run():
    import uvicorn
    print("\n" + "=" * 60)
    print("启动莫奈数字讲解员 v8.5.0")
    print("=" * 60)
    print(f"📍 服务地址: http://{HOST}:{PORT}")
    print(f"📍 对话 WebSocket: ws://{HOST}:{PORT}/ws/chat")
    print(f"📍 TTS WebSocket: ws://{HOST}:{PORT}/ws/tts")
    print(f"📍 机器人互动: http://{HOST}:{PORT}/api/robot/ask")
    print(f"📍 接口文档: http://{HOST}:{PORT}/docs")
    print("=" * 60 + "\n")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    run()
