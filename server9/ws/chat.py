# -*- coding: utf-8 -*-
"""讲解对话 WebSocket。"""
from __future__ import annotations
import asyncio
import json
import time
from typing import Optional

import requests
from fastapi import WebSocket, WebSocketDisconnect

from config import *
from llm.client import extract_clean_delta, clean_stream_sentence, clean_full_answer, SENT_END, _find_sentence_end, strip_identity_opening
from session import SessionState, persistence, active_sessions
from rag.intent import (
    intent_classifier, sentiment_analyzer, ner_system,
    recommendation_engine, memory_system,
)
from llm.persona import hook_generator, ticket_promoter, system_prompt, chitchat_reply
from persona.runtime import resolve, bind, active, call_bound
from rag.scope import (
    is_other_exhibition_query, locked_category, clamp_hall_answer,
    hall_safe_sentence, answer_leaves_exhibition, iter_hall_history,
    refuse_text,
)
from rag import index as rag
from rag.search import search as search_fn, works_answer_hint
from tts import cosyvoice as tts
from robot.hub import robot_hub
from ws.hub import accept_websocket

def _match_command(question: str, keyword: str) -> bool:
    return keyword in question

async def _tts_b64(text: str, speaker: str, persona=None) -> str:
    loop = asyncio.get_running_loop()
    p = persona if persona is not None else active()
    return await loop.run_in_executor(
        tts.tts_executor,
        lambda t=text, s=speaker, per=p: call_bound(
            tts.text_to_base64_audio, t, s, persona=per))


async def _send_reply_with_audio(websocket, reply: str, speaker: str):
    await websocket.send_json({"type": "text", "content": reply})
    if tts.cosyvoice_model:
        try:
            audio_b64 = await _tts_b64(reply, speaker)
            await websocket.send_json({"type": "audio", "content": audio_b64,
                                       "sentence": reply, "index": 0})
        except Exception as e:
            print(f"⚠️ TTS 失败: {e}")

async def handle_command(websocket: WebSocket, question: str, session: SessionState, speaker: str):
    commands = {
        "暂停": "好的,已暂停。需要继续时告诉我。",
        "停": "好的,已暂停。",
        "别说了": "好的,已暂停。",
        "声音大": "好的,我会说大声一点。",
        "大声": "好的,我会说大声一点。",
        "声音小": "好的,我会轻声一些。",
        "小声": "好的,我会轻声一些。",
        "再说一遍": clamp_hall_answer(session.history[-1].answer) if session.history else "还没有对话记录哦。",
        "重复": clamp_hall_answer(session.history[-1].answer) if session.history else "还没有对话记录哦。"
    }
    reply = None
    for keyword in sorted(commands.keys(), key=len, reverse=True):
        if _match_command(question, keyword):
            reply = commands[keyword]
            break
    if not reply:
        reply = "不好意思,我没理解这个指令。"
    await _send_reply_with_audio(websocket, reply, speaker)
    session.add_turn(question, reply, [], None, [], "command")
    persistence.save_session(session)
    await websocket.send_json({"type": "done", "intent": "command", "refused": False,
                               "sources": [], "suggestions": [], "sentiment": None})

async def handle_chitchat(websocket: WebSocket, question: str, session: SessionState, speaker: str):
    reply = chitchat_reply(question)
    await _send_reply_with_audio(websocket, reply, speaker)
    session.add_turn(question, reply, [], None, [], "chitchat")
    persistence.save_session(session)
    await websocket.send_json({"type": "done", "intent": "chitchat", "refused": False,
                               "sources": [], "suggestions": [], "sentiment": None})

async def websocket_chat(websocket: WebSocket, persona_id: Optional[str] = None):
    from fastapi import HTTPException
    path = websocket.scope.get("path", "")
    print(f"🔌 Chat WS 握手 path={path} persona_arg={persona_id!r} client={websocket.client}")
    try:
        persona = resolve(persona_id, allow_default=True)
    except HTTPException as e:
        try:
            await accept_websocket(websocket)
            from persona.registry import registry
            await websocket.send_json({
                "type": "error", "code": "unknown_persona",
                "message": str(e.detail),
                "hint": registry.endpoint_hint("chat"),
            })
            await websocket.close(code=1008, reason=str(e.detail)[:120])
        except Exception:
            pass
        print(f"⚠️ Chat WS 拒绝 path={path}: {e.detail}")
        return
    with bind(persona):
        await _websocket_chat_loop(websocket, persona)


async def _websocket_chat_loop(websocket: WebSocket, persona):
    await accept_websocket(websocket)
    session: Optional[SessionState] = None
    session_id = None
    send_lock = asyncio.Lock()

    async def safe_send(obj: dict):
        async with send_lock:
            await websocket.send_json(obj)

    try:
        init_data = await websocket.receive_json()
        user_id = init_data.get("user_id", f"anonymous_{id(websocket)}")
        session_id = init_data.get("session_id")

        if session_id:
            if session_id in active_sessions:
                session = active_sessions[session_id]
                if getattr(session, "persona_id", "") and session.persona_id != persona.id:
                    session = None
            else:
                session = persistence.load_session(session_id)
                if session:
                    if getattr(session, "persona_id", "") and session.persona_id != persona.id:
                        session = None
                    else:
                        session.persona_id = persona.id
                        active_sessions[session_id] = session

        if not session:
            session_id = f"session_{int(time.time())}_{id(websocket)}"
            session = SessionState(user_id, session_id, persona.id)
            active_sessions[session_id] = session
            if ENABLE_MEMORY:
                session.preferences = await memory_system.load_user_preferences(user_id)
                session.user_profile = await memory_system.load_user_profile(user_id)

        await safe_send({"type": "init", "session_id": session_id, "user_id": user_id,
                         "persona_id": persona.id,
                         "display_name": persona.display_name,
                         "preferences": session.preferences,
                         "history_count": len(session.history),
                         "resumed": len(session.history) > 0})

        loop = asyncio.get_running_loop()

        while True:
            data = await websocket.receive_json()
            question = data.get("question", "")
            category = data.get("category")
            top_k = data.get("top_k", 2)
            enable_audio = data.get("enable_audio", True)
            speaker = data.get("speaker", "中文男")

            if not question:
                await safe_send({"type": "error", "content": "问题不能为空"})
                continue
            if robot_hub.is_robot_ask_echo(question):
                print(f"⏭ 忽略游客对话（与当前机器人 ask 相同，避免 asr 回声）: {question[:40]}")
                continue

            # 其他展会必须先于意图分流，避免被闲聊/指令绕过
            if is_other_exhibition_query(question):
                print(f"🚫 其他展会问题，拉回本届: {question[:40]}")
                refuse_text_val = refuse_text(persona)
                await safe_send({"type": "text", "content": refuse_text_val})
                if enable_audio and tts.cosyvoice_model:
                    try:
                        audio_b64 = await _tts_b64(refuse_text_val, speaker, persona)
                        await safe_send({"type": "audio", "content": audio_b64,
                                         "sentence": refuse_text_val, "index": 0})
                    except Exception as e:
                        print(f"⚠️ TTS 失败: {e}")
                session.add_turn(question, refuse_text_val, [], None, [], "knowledge")
                persistence.save_session(session)
                await safe_send({"type": "done", "refused": True, "intent": "knowledge",
                                 "sources": [], "suggestions": [], "sentiment": None})
                continue

            # 意图识别移入 executor，不阻塞事件循环
            intent, raw_score, _conf = await loop.run_in_executor(
                None, intent_classifier.classify, question)

            if intent == "command" and intent_classifier.should_route_to("command", raw_score):
                await handle_command(websocket, question, session, speaker)
                continue
            elif intent == "chitchat" and intent_classifier.should_route_to("chitchat", raw_score):
                await handle_chitchat(websocket, question, session, speaker)
                continue

            sentiment = sentiment_analyzer.analyze(question)
            entities = ner_system.extract_entities(question)

            category = locked_category(category)

            # 检索移入 executor
            _q, _cat, _tk, _sess = question, category, top_k, session
            hits = await loop.run_in_executor(None, lambda: search_fn(_q, _tk, _cat, _sess, persona))

            threshold = MIN_SCORE_RERANK if rag.HAS_RERANKER else MIN_SCORE_RRF
            if not hits or hits[0]["score"] < threshold:
                refuse_text_val = refuse_text(persona)
                await safe_send({"type": "text", "content": refuse_text_val})
                if enable_audio and tts.cosyvoice_model:
                    try:
                        audio_b64 = await _tts_b64(refuse_text_val, speaker, persona)
                        await safe_send({"type": "audio", "content": audio_b64,
                                         "sentence": refuse_text_val, "index": 0})
                    except Exception as e:
                        print(f"⚠️ TTS 失败: {e}")
                session.add_turn(question, refuse_text_val, [], sentiment, entities, "knowledge")
                persistence.save_session(session)
                await safe_send({"type": "done", "refused": True, "intent": "knowledge",
                                 "sources": [], "suggestions": [],
                                 "sentiment": sentiment.to_dict()})
                continue

            # 构建上下文
            contexts = [h["text"][:COMPRESS_MAX] for h in hits]
            ref = "\n".join(f"- {c}" for c in contexts)

            history_text = ""
            hall_turns = list(iter_hall_history(session.get_context_window()))
            for q, a in hall_turns:
                history_text += f"Q: {q}\nA: {a}\n\n"

            recent_entities = session.get_recent_entities()
            entity_context = "、".join(
                e.text for e in recent_entities[:5]
                if e.text and not is_other_exhibition_query(e.text)
            )

            messages = [{"role": "system", "content": system_prompt()}]
            for q, a in hall_turns:
                messages.append({"role": "user", "content": q})
                messages.append({"role": "assistant", "content": a})

            user_prompt = f"【参考资料】\n{ref}\n\n【问题】{question}"
            if history_text:
                user_prompt = f"【对话历史】\n{history_text}\n" + user_prompt
            if entity_context:
                user_prompt = f"【实体上下文】最近提到: {entity_context}\n\n" + user_prompt
            user_prompt += "\n\n只根据参考资料讲这场展，不要提及其他展览。\n"
            user_prompt += works_answer_hint(question, persona)
            user_prompt += "/no_think"
            messages.append({"role": "user", "content": user_prompt})

            llm_payload = {"messages": messages, "temperature": 0.05,
                           "max_tokens": 300, "stream": True}

            # TTS 后台 worker 队列，文本流不再被合成拖慢
            tts_queue: Optional[asyncio.Queue] = None
            worker_task: Optional[asyncio.Task] = None
            use_tts = enable_audio and tts.cosyvoice_model

            if use_tts:
                tts_queue = asyncio.Queue()

                async def tts_worker(q: asyncio.Queue):
                    idx = 0
                    while True:
                        sent = await q.get()
                        if sent is None:
                            break
                        try:
                            audio_b64 = await _tts_b64(sent, speaker, persona)
                            await safe_send({"type": "audio", "content": audio_b64,
                                             "sentence": sent, "index": idx})
                            idx += 1
                        except Exception as e:
                            print(f"⚠️ TTS 失败: {e}")

                worker_task = asyncio.create_task(tts_worker(tts_queue))

            async def enqueue_sentence(sent: str):
                if use_tts and sent:
                    await tts_queue.put(sent)

            try:
                sse_queue: asyncio.Queue = asyncio.Queue()

                def _read_sse():
                    resp = None
                    try:
                        resp = requests.post(LLM_URL, json=llm_payload, timeout=60, stream=True)
                        resp.raise_for_status()
                        for line in resp.iter_lines(decode_unicode=True):
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

                full_raw_text = ""
                sentence_buffer = ""
                clean_pos = 0
                t0 = time.time()
                leaked = False
                first_sent = False

                while True:
                    if leaked:
                        break
                    line = await sse_queue.get()
                    if line is None:
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

                    full_raw_text += content
                    # 半截标签安全增量
                    new_content, clean_pos = extract_clean_delta(full_raw_text, clean_pos)
                    if not new_content:
                        continue

                    await safe_send({"type": "text", "content": new_content})
                    sentence_buffer += new_content

                    # while 循环切句（一个 delta 可能含多个句末符），入队不等待
                    while True:
                        pos = _find_sentence_end(sentence_buffer, SENT_END)
                        if pos == -1:
                            break
                        sentence = clean_stream_sentence(sentence_buffer[:pos + 1].strip())
                        sentence_buffer = sentence_buffer[pos + 1:]
                        if not sentence:
                            continue
                        if not first_sent:
                            sentence = strip_identity_opening(sentence, persona)
                            if not sentence:
                                continue
                            first_sent = True
                        sentence, stop = hall_safe_sentence(sentence)
                        await enqueue_sentence(sentence)
                        if stop:
                            leaked = True
                            await safe_send({"type": "text", "content": refuse_text(persona)})
                            break

                if not leaked:
                    tail = clean_stream_sentence(sentence_buffer.strip())
                    if tail and not first_sent:
                        tail = strip_identity_opening(tail, persona)
                    tail, stop = hall_safe_sentence(tail)
                    await enqueue_sentence(tail)
                    if stop:
                        leaked = True
                        await safe_send({"type": "text", "content": refuse_text(persona)})

                cleaned = clean_full_answer(full_raw_text)
                if leaked or answer_leaves_exhibition(cleaned or ""):
                    leaked = True
                    full_answer = refuse_text(persona)
                else:
                    full_answer = clamp_hall_answer(cleaned or refuse_text(persona), persona)
                print(f"✅ LLM 回答完成，耗时 {time.time()-t0:.2f}s")

            except Exception as e:
                if worker_task:
                    await tts_queue.put(None)
                    await worker_task
                error_text = "抱歉,我现在有点不舒服。"
                await safe_send({"type": "error", "content": str(e)})
                await safe_send({"type": "text", "content": error_text})
                await safe_send({"type": "done", "refused": False, "intent": "knowledge",
                                 "sources": [], "suggestions": [],
                                 "sentiment": sentiment.to_dict()})
                continue

            # 钩子 / 票务只在本届展讲解后追加，越界改口不再加
            if not leaked:
                if hook_generator.should_add_hook(session):
                    hook = hook_generator.select_hook(session)
                    if hook:
                        full_answer += " " + hook
                        await safe_send({"type": "text", "content": " " + hook})
                        await enqueue_sentence(hook)

                if ticket_promoter.should_promote(question, session):
                    promo_text = ticket_promoter.get_ticket_info(question)
                    full_answer += " " + promo_text
                    await safe_send({"type": "text", "content": " " + promo_text})
                    await enqueue_sentence(promo_text)

            # 等待音频全部发完再收尾
            if worker_task:
                await tts_queue.put(None)
                await worker_task

            source_ids = [] if leaked else [h["id"] for h in hits]
            session.add_turn(question, full_answer, source_ids, sentiment, entities, "knowledge")
            persistence.save_session(session)

            if ENABLE_MEMORY:
                asyncio.create_task(
                    memory_system.save_conversation_turn(session, session.history[-1]))

            recommendations = []
            if not leaked and session.should_recommend():
                recommendations = recommendation_engine.generate_recommendations(session, top_k=3)
                session.last_recommendation_turn = len(session.history) - 1
                if recommendations:
                    session.recommendation_count += 1
                    rec_text = ("顺便说一句,"
                                + "、".join([r["title"] for r in recommendations[:2]])
                                + " 也很有意思。")
                    await safe_send({"type": "recommendation", "content": rec_text,
                                     "items": recommendations})

            await safe_send({"type": "done", "refused": leaked, "intent": "knowledge",
                             "sources": source_ids, "suggestions": recommendations,
                             "sentiment": sentiment.to_dict()})

    except WebSocketDisconnect:
        print(f"🔌 WebSocket 断开: {session_id}")
        if session:
            persistence.save_session(session)
    except Exception as e:
        print(f"❌ WebSocket 错误: {e}")
        import traceback
        traceback.print_exc()
        if session:
            persistence.save_session(session)
        try:
            await websocket.close()
        except Exception:
            pass
