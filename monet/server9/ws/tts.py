# -*- coding: utf-8 -*-
"""TTS WebSocket：流式 PCM + 取消。"""
from __future__ import annotations
import asyncio
import json
import re
import time
import uuid
import threading
from typing import Dict

import requests
from fastapi import WebSocket, WebSocketDisconnect

from config import *
from llm.persona import SYSTEM_PROMPT
from llm.client import extract_clean_delta, clean_stream_sentence, SENT_END, clean_full_answer, _find_sentence_end
from rag import index as rag
from rag.search import search as search_fn
from tts.cosyvoice import push_sentence_pcm
from tts import cosyvoice as tts
from ws.hub import tts_fanout, attach_robot_id
from robot.hub import robot_hub
from rag.scope import is_other_exhibition_query, clamp_hall_answer, hall_safe_sentence

async def handle_tts_request(websocket: WebSocket, safe_send, user_question: str,
                             speaker: str, enable_llm: bool, session_id: str,
                             cancel_ev: threading.Event,
                             conn_sessions: Dict[str, threading.Event]):
    loop = asyncio.get_running_loop()
    t0 = time.time()
    seq = 0
    sent_idx = 0
    cancelled = False
    started = False
    audio_queue = asyncio.Queue()

    if cancel_ev.is_set():
        conn_sessions.pop(session_id, None)
        return

    async def ensure_started(text_hint: str = ""):
        nonlocal started
        if not started:
            started = True
            await safe_send({"type": "tts_start", "session_id": session_id,
                             "format": "pcm", "sample_rate": TTS_SAMPLE_RATE,
                             "channels": 1, "sample_width": 2, "text": text_hint})

    async def gated_send(obj: dict):
        if obj.get("type") in ("tts_chunk", "tts_sentence") and not started:
            await ensure_started(obj.get("text") or "")
        await safe_send(obj)

    async def _emit_tts_end():
        """游客被机器人接管时不发 tts_end，一轮 ask 只允许 robot stream 发一次 end。"""
        if cancel_ev.is_set() and robot_hub.busy:
            print(f"⏭ 跳过游客 tts_end（机器人本轮接管）session={session_id[:8]}")
            return
        if not started and (cancelled or cancel_ev.is_set()):
            return
        await ensure_started()
        await safe_send({"type": "tts_end", "session_id": session_id,
                         "seq_count": seq, "sentence_count": sent_idx,
                         "cancelled": bool(cancelled or cancel_ev.is_set())})

    async def send_subtitle(sent: str):
        """确保 tts_start 已发出，然后把 (文本, 字幕序号) 入队。
        字幕将由 audio_worker 在合成首包前同步发出，实现声画对齐。"""
        nonlocal sent_idx
        await ensure_started(sent)          # tts_start 尽早发，不等音频
        await audio_queue.put((sent, sent_idx))
        sent_idx += 1

    async def audio_worker():
        """后台协程：从队列取 (句子, 字幕序号)，串行合成推送音频"""
        nonlocal seq, cancelled
        while True:
            item = await audio_queue.get()
            if item is None:
                break
            sent, s_idx = item
            seq, cancelled = await push_sentence_pcm(
                websocket, gated_send, sent, speaker, session_id, seq, cancel_ev,
                subtitle_text=sent, subtitle_index=s_idx)
            if cancelled:
                break

    audio_task = asyncio.create_task(audio_worker())

    try:
        if is_other_exhibition_query(user_question):
            print(f"🚫 其他展会问题，拉回本届: {user_question[:40]}")
            llm_answer = REFUSE_TEXT
            await safe_send({"type": "llm_answer", "text": llm_answer})
            await send_subtitle(llm_answer)
        elif not enable_llm:
            llm_answer = user_question
            await safe_send({"type": "llm_answer", "text": llm_answer})
            sentences = [s.strip() for s in re.split(f"(?<=[{SENT_END}])", llm_answer) if s.strip()]
            for sent in sentences:
                await send_subtitle(sent)
                if cancelled:
                    break
        else:
            try:
                print(f"🤖 用户问题: {user_question}")
                hits = await loop.run_in_executor(
                    None, lambda: search_fn(user_question, 2, EXHIBITION_CATEGORY, None))
                threshold = MIN_SCORE_RERANK if rag.HAS_RERANKER else MIN_SCORE_RRF

                if hits:
                    print(f"🔍 Top1 分数: {hits[0]['score']} (阈值 {threshold}) | "
                          f"{hits[0]['title']} | 检索耗时 {time.time()-t0:.2f}s")

                if cancel_ev.is_set():
                    cancelled = True
                elif not hits or hits[0]["score"] < threshold:
                    llm_answer = REFUSE_TEXT
                    await safe_send({"type": "llm_answer", "text": llm_answer})
                    await send_subtitle(llm_answer)
                else:
                    contexts = [h["text"][:COMPRESS_MAX] for h in hits]
                    ref = "\n".join(f"- {c}" for c in contexts)
                    user_content = (
                        f"【参考资料】\n{ref}\n\n【问题】{user_question}\n\n"
                        "只根据参考资料讲这场展，不要提及其他展览。\n/no_think"
                    )
                    llm_payload = {"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                                {"role": "user", "content": user_content}],
                                   "temperature": 0.05, "max_tokens": 300, "stream": True}

                    sse_queue: asyncio.Queue = asyncio.Queue()
                    resp_holder = {}

                    def _read_sse():
                        resp = None
                        try:
                            resp = requests.post(LLM_URL, json=llm_payload, timeout=60, stream=True)
                            resp_holder['resp'] = resp
                            resp.raise_for_status()
                            for line in resp.iter_lines(decode_unicode=True):
                                if cancel_ev.is_set():
                                    break
                                loop.call_soon_threadsafe(sse_queue.put_nowait, line)
                        except Exception:
                            pass
                        finally:
                            try:
                                if resp is not None:
                                    resp.close()
                            except Exception:
                                pass
                            loop.call_soon_threadsafe(sse_queue.put_nowait, None)

                    loop.run_in_executor(None, _read_sse)

                    full_raw = ""
                    clean_pos = 0
                    buf = ""
                    first_logged = False
                    leaked = False

                    while True:
                        line = await sse_queue.get()
                        if line is None:
                            break
                        if cancel_ev.is_set():
                            cancelled = True
                            break
                        if not line or not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            break
                        try:
                            delta = json.loads(payload)["choices"][0].get("delta", {})
                        except Exception:
                            continue
                        content = delta.get("content", "")
                        if not content:
                            continue

                        full_raw += content
                        new_content, clean_pos = extract_clean_delta(full_raw, clean_pos)
                        if not new_content:
                            continue
                        buf += new_content

                        while True:
                            pos = _find_sentence_end(buf, SENT_END)
                            if pos == -1:
                                break
                            sent = clean_stream_sentence(buf[:pos + 1].strip())
                            buf = buf[pos + 1:]
                            if not sent:
                                continue
                            sent, stop = hall_safe_sentence(sent)
                            if not first_logged:
                                print(f"⏱️ 首句就绪: {time.time()-t0:.2f}s | {sent[:30]}")
                                first_logged = True
                            await send_subtitle(sent)
                            if stop:
                                leaked = True
                                break
                            if cancelled:
                                break
                        if cancelled or leaked:
                            break

                    if cancelled or leaked:
                        r = resp_holder.get('resp')
                        if r is not None:
                            try:
                                r.close()
                            except Exception:
                                pass

                    if not cancelled and not leaked and buf.strip():
                        tail = clean_stream_sentence(buf.strip())
                        if tail:
                            tail, stop = hall_safe_sentence(tail)
                            await send_subtitle(tail)
                            if stop:
                                leaked = True

                    if not cancelled:
                        llm_answer = REFUSE_TEXT if leaked else clamp_hall_answer(
                            clean_full_answer(full_raw) or REFUSE_TEXT)
                        print(f"✅ LLM 回答: {llm_answer[:50]}... | 总耗时 {time.time()-t0:.2f}s")
                        await safe_send({"type": "llm_answer", "text": llm_answer})

            except Exception as e:
                print(f"⚠️ LLM 调用失败: {e}")
                if not cancel_ev.is_set():
                    llm_answer = "抱歉，我现在有点不舒服。"
                    await safe_send({"type": "llm_answer", "text": llm_answer})
                    await send_subtitle(llm_answer)

        await audio_queue.put(None)
        await audio_task

        try:
            await _emit_tts_end()
        except Exception:
            pass

        print(f"✅ TTS 会话 {session_id[:8]} {'取消' if cancelled else '完成'}，"
              f"共 {seq} 个分片，耗时 {time.time()-t0:.2f}s")

    except Exception as e:
        print(f"❌ TTS 请求处理错误: {e}")
        import traceback
        traceback.print_exc()
        try:
            if not cancel_ev.is_set():
                await safe_send({"type": "error", "message": f"处理失败: {str(e)}"})
            await audio_queue.put(None)
            await audio_task
            cancelled = True
            await _emit_tts_end()
        except Exception:
            pass
    finally:
        conn_sessions.pop(session_id, None)

async def websocket_tts_stream(websocket: WebSocket):
    await websocket.accept()

    # 会话/取消状态按连接隔离，互不干扰
    conn_sessions: Dict[str, threading.Event] = {}
    conn_tasks: set = set()
    send_lock = asyncio.Lock()

    async def safe_send(obj: dict):
        payload = attach_robot_id(obj, "")
        async with send_lock:
            await websocket.send_json(payload)

    def cancel_all():
        for ev in conn_sessions.values():
            ev.set()

    await tts_fanout.register(websocket, send_lock, conn_sessions)
    await safe_send({"type": "ready", "service": TTS_SERVICE_NAME})
    print("🎤 TTS WebSocket 连接建立")
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "ping":
                await safe_send({"type": "pong"})
                continue
            elif msg_type == "tts":
                user_question = data.get("text", "")
                speaker = data.get("speaker", "中文男")
                enable_llm = data.get("enable_llm", True)

                if not user_question:
                    await safe_send({"type": "error", "message": "文本不能为空"})
                    continue
                # 机器人正在推这一轮时，前端常会：
                # 1) 收到 asr 后再发 type=tts（原问题）
                # 2) 收到 llm_answer 后再发 type=tts（把讲稿当新问题）
                # 3) 先 cancel 再 tts
                # 第二条会和 robot 流抢同一条 /ws/tts，听起来就是推流被切成两次。
                if robot_hub.busy:
                    print(f"⏭ 忽略游客 TTS（机器人正在推流）: {user_question[:40]}")
                    continue
                if robot_hub.is_robot_ask_echo(user_question):
                    print(f"⏭ 忽略游客 TTS（与当前机器人 ask 相同，避免 asr 回声）: {user_question[:40]}")
                    continue
                if not tts.cosyvoice_model:
                    await safe_send({"type": "error", "message": "TTS 服务未启用"})
                    continue

                cancel_all()

                session_id = str(uuid.uuid4())
                cancel_ev = threading.Event()
                conn_sessions[session_id] = cancel_ev
                task = asyncio.create_task(handle_tts_request(
                    websocket, safe_send, user_question, speaker,
                    enable_llm, session_id, cancel_ev, conn_sessions))
                conn_tasks.add(task)
                task.add_done_callback(conn_tasks.discard)

            elif msg_type == "cancel":
                if robot_hub.busy:
                    print("⏭ 忽略游客 cancel（机器人正在推流，本轮由 robot stream 收尾）")
                    continue
                print("收到取消指令，标记本连接所有活跃会话")
                cancel_all()
                await safe_send({"type": "cancel_ack"})
            else:
                await safe_send({"type": "error", "message": f"未知消息类型: {msg_type}"})

    except WebSocketDisconnect:
        print("🔌 TTS WebSocket 断开")
        cancel_all()
        for t in list(conn_tasks):
            t.cancel()
    except Exception as e:
        print(f"❌ TTS WebSocket 错误: {e}")
        import traceback
        traceback.print_exc()
        cancel_all()
        for t in list(conn_tasks):
            t.cancel()
    finally:
        await tts_fanout.unregister(websocket, conn_sessions)
