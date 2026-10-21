# -*- coding: utf-8 -*-
"""TTS WebSocket：流式 PCM + 取消。"""
from __future__ import annotations
import asyncio
import json
import re
import time
import uuid
import threading
from typing import Dict, Optional

import requests
from fastapi import WebSocket, WebSocketDisconnect

from config import *
from llm.persona import system_prompt, pick_hello, chitchat_reply
from rag.intent import intent_classifier
from persona.registry import registry
from persona.runtime import resolve, bind, active
from llm.client import extract_clean_delta, clean_stream_sentence, SENT_END, clean_full_answer, _find_sentence_end, strip_identity_opening
from rag import index as rag
from rag.search import search as search_fn, works_answer_hint
from tts.cosyvoice import push_sentence_pcm, tts_output_rate
from tts import cosyvoice as tts
from tts.engine import tts_engine_name
from tts.playback import PcmPlayClock
from tts.planner import UnitPlanner
from ws.hub import tts_fanout, attach_robot_id, accept_websocket
from ws.session_wire import tts_wire_allowed
from robot.hub import robot_hub
from rag.scope import is_other_exhibition_query, clamp_hall_answer, hall_safe_sentence, locked_category, refuse_text
from rag.visit import visit_hint, guard_hall_clock, filter_spoken_sentence
from rag.dialogue import (
    is_foreign_question, FOREIGN_REPLY, note_hall, note_realtime, realtime_reply,
    rule_intent,
    suggest_hooks, visit_fact_reply, exhibit_names, titles_from_items, HALL_INTENTS,
    RealtimeMemory,
)
from services.local_feed import feed_bundle
from services.screen_hooks import (
    is_ticket_booking, booking_reply, ticket_form_event, request_ticket_form,
    artworks_event, notify_artworks, headline_event, request_headline_image,
)

_STRONG_END = "。！？.!"
_COMMANDS = (
    ("再说一遍", None),
    ("重复一遍", None),
    ("别说了", "好的,已暂停。"),
    ("声音大", "好的,我会说大声一点。"),
    ("大声点", "好的,我会说大声一点。"),
    ("声音小", "好的,我会轻声一些。"),
    ("小声点", "好的,我会轻声一些。"),
    ("暂停", "好的,已暂停。需要继续时告诉我。"),
)


def _looks_chitchat(question: str) -> bool:
    text = (question or "").strip()
    if not text or len(text) > 12:
        return False
    keys = ("你好", "您好", "嗨", "哈喽", "你是谁", "你叫什么", "谢谢", "再见",
            "你会什么", "聊聊", "介绍一下自己", "介绍下自己")
    return any(text == key or text.startswith(key) for key in keys)


def _command_reply(question: str, memory) -> str:
    text = question or ""
    for keyword, reply in _COMMANDS:
        if keyword in text:
            if reply:
                return reply
            last = getattr(memory, "last_answer", "") or ""
            return last or "还没有对话记录哦。"
    return ""


def backlog_has_strong_end(backlog) -> bool:
    """已经有一句以句号、问号或叹号收尾。"""
    for item in backlog:
        text = item if isinstance(item, str) else ""
        if text and text[-1] in _STRONG_END:
            return True
    return False


async def wait_first_sentence(queue, backlog, timeout: float, cancel_ev=None) -> bool:
    """首句等到句号类标点再开声。超时、结束或取消也停。返回文本是否已经结束。"""
    eof = False
    if timeout <= 0 or backlog_has_strong_end(backlog):
        return False
    loop = asyncio.get_running_loop()
    deadline = loop.time() + float(timeout)
    while not backlog_has_strong_end(backlog):
        if cancel_ev is not None and cancel_ev.is_set():
            break
        left = deadline - loop.time()
        if left <= 0:
            break
        try:
            item = await asyncio.wait_for(queue.get(), left)
        except asyncio.TimeoutError:
            break
        if item is None:
            eof = True
            break
        backlog.append(item)
    while not eof and not queue.empty():
        item = queue.get_nowait()
        if item is None:
            eof = True
            break
        backlog.append(item)
    return eof


async def coalesce_backlog(queue, backlog, min_units: int, timeout: float,
                           cancel_ev=None) -> bool:
    """句子还没凑满 min_units 时再等一会儿。返回文本是否已经结束。

    游客问答是逐句送进来的。第一句一到就开算，后面的句子进不了这一次 generate。
    凑满、超时、收到结束标记或取消，都会停。
    """
    eof = False
    if min_units <= 0 or timeout <= 0 or len(backlog) >= min_units:
        return False
    loop = asyncio.get_running_loop()
    deadline = loop.time() + float(timeout)
    while len(backlog) < min_units:
        if cancel_ev is not None and cancel_ev.is_set():
            break
        left = deadline - loop.time()
        if left <= 0:
            break
        try:
            item = await asyncio.wait_for(queue.get(), left)
        except asyncio.TimeoutError:
            break
        if item is None:
            eof = True
            break
        backlog.append(item)
    while not eof and not queue.empty():
        item = queue.get_nowait()
        if item is None:
            eof = True
            break
        backlog.append(item)
    return eof


async def handle_tts_request(websocket: WebSocket, safe_send, user_question: str,
                             speaker: str, enable_llm: bool, session_id: str,
                             cancel_ev: threading.Event,
                             conn_sessions: Dict[str, threading.Event],
                             latest_sid: Optional[list] = None,
                             dialogue_state=None):
    loop = asyncio.get_running_loop()
    t0 = time.time()
    seq = 0
    sent_idx = 0
    cancelled = False
    started = False
    audio_skip_reason = ""
    audio_queue = asyncio.Queue()

    if cancel_ev.is_set():
        conn_sessions.pop(session_id, None)
        return

    def _is_current() -> bool:
        return tts_wire_allowed({"type": "tts_end"}, session_id, latest_sid)

    async def emit(obj: dict):
        if not tts_wire_allowed(obj, session_id, latest_sid):
            kind = obj.get("type")
            cur = ""
            if latest_sid:
                cur = (latest_sid[0] or "")[:8]
            print(f"⏭ 丢弃旧会话 {kind} session={session_id[:8]} current={cur or '-'}")
            return
        await safe_send(obj)

    async def ensure_started(text_hint: str = ""):
        nonlocal started
        if not started:
            started = True
            await emit({"type": "tts_start", "session_id": session_id,
                        "format": "pcm", "sample_rate": tts_output_rate(),
                        "channels": 1, "sample_width": 2, "text": text_hint})

    async def gated_send(obj: dict):
        if obj.get("type") in ("tts_chunk", "tts_sentence") and not started:
            await ensure_started(obj.get("text") or "")
        await emit(obj)

    async def send_llm_answer(text: str):
        if text:
            dialogue_state.last_answer = text
        await ensure_started(text)
        await emit({"type": "llm_answer", "session_id": session_id, "text": text})

    async def _emit_tts_end():
        """游客被机器人接管时不发 tts_end，一轮 ask 只允许 robot stream 发一次 end。
        仅当本连接人设就是当前 robot 播报人设时跳过；其他人设照常收尾。
        参考 robot-ask-ws-改动说明.md。"""
        mine = ""
        try:
            mine = active().id
        except Exception:
            mine = ""
        if cancel_ev.is_set() and robot_hub.is_busy_for(mine):
            print(f"⏭ 跳过游客 tts_end（机器人本轮接管）session={session_id[:8]} persona={mine}")
            return
        if not _is_current():
            print(f"⏭ 跳过旧会话 tts_end session={session_id[:8]}")
            return
        was_cancelled = bool(cancelled or cancel_ev.is_set())
        if not started and was_cancelled:
            return
        await ensure_started()
        end = {"type": "tts_end", "session_id": session_id,
               "seq_count": seq, "sentence_count": sent_idx,
               "cancelled": was_cancelled,
               "intent": control_box.get("intent") or "knowledge",
               "lang": "zh",
               "hook_suggest": list(control_box.get("hook_suggest") or []),
               "exhibit_names": list(control_box.get("exhibit_names") or [])}
        if seq <= 0:
            end["audio"] = False
            if audio_skip_reason:
                end["skip_reason"] = audio_skip_reason
        await emit(end)

    persona = active()
    if dialogue_state is None:
        dialogue_state = RealtimeMemory()
    control_box = {"intent": "knowledge", "hook_suggest": [], "exhibit_names": []}

    def _remember(intent: str, reply: str, hall: bool = False, refused: bool = False):
        if hall:
            note_hall(dialogue_state)
        control_box["intent"] = intent
        names = [] if refused else exhibit_names(
            reply, titles_from_items(rag.active_items(persona)),
            getattr(persona, "exhibit_aliases", None) or {})
        control_box["exhibit_names"] = names
        control_box["hook_suggest"] = [] if refused else suggest_hooks(
            intent, dialogue_state, persona, names)
    pending = ""
    pending_idx = None
    first_clause = True
    pcm_bytes_sent = 0
    last_pcm_ok = True
    play_clock = PcmPlayClock(tts_output_rate())
    # tts_start 等到第一句可播文本再发，text 用口播句而不是用户问题。
    # 早发 tts_start 时，旧会话的 cancelled tts_end 会把 UE 播放器关掉。

    def on_pcm(chunk: bytes):
        nonlocal pcm_bytes_sent
        if chunk:
            pcm_bytes_sent += len(chunk)
            play_clock.add(chunk)

    async def flush_pending():
        """把攒够长度的短句送去合成，避免 CosyVoice 漏出参考音那一句。"""
        nonlocal pending, pending_idx, sent_idx
        if not pending:
            return
        await ensure_started(pending)
        idx = pending_idx if pending_idx is not None else sent_idx
        await audio_queue.put((pending, idx))
        sent_idx = idx + 1
        pending = ""
        pending_idx = None

    # CosyVoice3 / Qwen3-TTS 整句出声：小句原样入队，由合成协程按客户端缓冲决定拼多少、批多少。
    merge_chars = tts.clause_merge_chars()
    qwen_batch = tts_engine_name() == "qwen3"
    planned = bool(merge_chars) or qwen_batch

    async def send_subtitle(sent: str):
        """标点即切、立刻入队。首句和尾句都要单独合成，禁止粘包截尾。"""
        nonlocal pending, pending_idx, first_clause
        sent = (sent or "").strip()
        if not sent:
            return
        if planned:
            await ensure_started(sent)
            await audio_queue.put(sent)
            return
        if first_clause:
            if pending:
                await flush_pending()
            pending = sent
            pending_idx = sent_idx
            first_clause = False
            await flush_pending()
            return
        if pending and pending[-1] in "，。！？；、,.!?":
            await flush_pending()
        if pending:
            pending = pending + sent
        else:
            pending = sent
            pending_idx = sent_idx
        if pending[-1] in "，。！？；、,.!?":
            await flush_pending()
        elif len(pending) >= 40:
            await flush_pending()

    async def audio_worker():
        """后台协程：从队列取 (句子, 字幕序号)，串行合成推送音频"""
        nonlocal seq, cancelled, last_pcm_ok, audio_skip_reason
        last_item = None
        if not tts.cosyvoice_model:
            print(f"🔁 CosyVoice 未就绪，补加载后再合成 persona={getattr(persona, 'id', '')}")
            await loop.run_in_executor(None, tts.ensure_cosyvoice_loaded)
        model_ok = bool(tts.cosyvoice_model)
        if not model_ok:
            audio_skip_reason = "cosyvoice_not_loaded"
            print(f"❌ 不下发 tts_chunk：CosyVoice 未加载 err={tts._tts_last_error or '-'}")
        if planned:
            last_item = await planned_worker(model_ok)
        while not planned:
            item = await audio_queue.get()
            if item is None:
                break
            last_item = item
            sent, s_idx = item
            if not model_ok:
                await gated_send({
                    "type": "tts_sentence",
                    "session_id": session_id,
                    "index": s_idx,
                    "text": sent,
                })
                last_pcm_ok = False
                continue
            before = pcm_bytes_sent
            seq, cancelled = await push_sentence_pcm(
                websocket, gated_send, sent, speaker, session_id, seq, cancel_ev,
                subtitle_text=sent, subtitle_index=s_idx, on_pcm=on_pcm, persona=persona)
            last_pcm_ok = pcm_bytes_sent > before
            if not last_pcm_ok:
                print(f"⚠️ 末句/该句无 PCM idx={s_idx} persona={getattr(persona, 'id', '')} '{sent[:30]}'")
            if cancelled:
                break
        if last_item is not None and not last_pcm_ok:
            print(f"⚠️ 本轮最后一句没有声音 session={session_id[:8]} '{last_item[0][:30]}'")

    async def planned_worker(model_ok: bool):
        """整句出声引擎：推送不等播放，推完立刻合成下一段。

        客户端缓冲少时只送一个小句，缓冲够再把后面的小句拼长；Qwen3-TTS 把排着的几段一次 batch。
        """
        nonlocal seq, cancelled, last_pcm_ok, sent_idx
        planner = UnitPlanner(
            merge_chars,
            split_chars=QWEN_TTS_SPLIT_CHARS if qwen_batch else 0,
            split_marks=qwen_batch and QWEN_TTS_SPLIT_CHARS <= 0,
        )
        _diag = [0]
        backlog = []
        eof = False
        last_item = None
        bps = 2.0 * float(tts_output_rate())
        while True:
            if not backlog and not eof:
                item = await audio_queue.get()
                if item is None:
                    eof = True
                else:
                    backlog.append(item)
            while not eof and not audio_queue.empty():
                item = audio_queue.get_nowait()
                if item is None:
                    eof = True
                else:
                    backlog.append(item)
            if not backlog:
                break
            if (qwen_batch and model_ok and not eof and pcm_bytes_sent <= 0
                    and QWEN_TTS_FIRST_MS > 0
                    and not backlog_has_strong_end(backlog)):
                before_n = len(backlog)
                t_wait = time.time()
                if await wait_first_sentence(
                        audio_queue, backlog, QWEN_TTS_FIRST_MS / 1000.0, cancel_ev):
                    eof = True
                waited = time.time() - t_wait
                if len(backlog) != before_n or waited >= 0.05:
                    tts._tts_log(
                        "[TTS] 首句等待标点 %s->%s 等%.2fs eof=%s session=%s"
                        % (before_n, len(backlog), waited, int(eof), session_id[:8]))
            elif (qwen_batch and model_ok and not eof
                    and QWEN_TTS_COALESCE_MS > 0
                    and len(backlog) < QWEN_TTS_COALESCE_UNITS):
                before_n = len(backlog)
                t_wait = time.time()
                if await coalesce_backlog(
                        audio_queue, backlog, QWEN_TTS_COALESCE_UNITS,
                        QWEN_TTS_COALESCE_MS / 1000.0, cancel_ev):
                    eof = True
                waited = time.time() - t_wait
                if len(backlog) != before_n or waited >= 0.05:
                    tts._tts_log(
                        "[TTS] 开算前攒句 %s->%s 等%.2fs eof=%s session=%s"
                        % (before_n, len(backlog), waited, int(eof), session_id[:8]))
            if not backlog:
                break
            ahead = play_clock.remain(tail=0.0)
            units = planner.take(backlog, ahead,
                                 max_units=QWEN_TTS_BATCH if qwen_batch and model_ok else 1)
            if _diag[0] < 16:
                _cap = planner.char_cap(ahead)
                print(f"[DIAG] iter={_diag[0]} ahead={ahead:.2f} k={planner.k} "
                      f"cps={planner.cps} cap={_cap} units={len(units)} "
                      f"chars={sum(len(u) for u in units)}")
                _diag[0] += 1
            items = []
            for text in units:
                items.append((text, sent_idx))
                sent_idx += 1
            if not model_ok:
                for sent, s_idx in items:
                    last_item = (sent, s_idx)
                    await gated_send({"type": "tts_sentence", "session_id": session_id,
                                      "index": s_idx, "text": sent})
                last_pcm_ok = False
                continue
            batch_sec = 0.0
            batched = False
            if len(units) > 1 and not cancel_ev.is_set():
                t_pre = time.time()
                n_ok = await loop.run_in_executor(
                    None, lambda texts=list(units): tts.prefetch_batch(
                        texts, speaker, cancel_ev, persona=persona))
                batch_sec = time.time() - t_pre
                batched = n_ok == len(units)
                print(f"⚡ Qwen 批量预取 {n_ok}/{len(units)} 段 {batch_sec:.2f}s "
                      f"ahead={ahead:.2f}s session={session_id[:8]}")
            measured = []
            for sent, s_idx in items:
                last_item = (sent, s_idx)
                before = pcm_bytes_sent
                t_push = time.time()
                seq, cancelled = await push_sentence_pcm(
                    websocket, gated_send, sent, speaker, session_id, seq, cancel_ev,
                    subtitle_text=sent, subtitle_index=s_idx, on_pcm=on_pcm, persona=persona)
                measured.append((len(sent), (pcm_bytes_sent - before) / bps, time.time() - t_push))
                last_pcm_ok = pcm_bytes_sent > before
                if not last_pcm_ok:
                    print(f"⚠️ 末句/该句无 PCM idx={s_idx} persona={getattr(persona, 'id', '')} '{sent[:30]}'")
                if cancelled:
                    return last_item
            planner.observe_round(measured, batch_sec if batched else None)
        return last_item

    audio_task = asyncio.create_task(audio_worker())

    try:
        if is_foreign_question(user_question):
            print(f"🌐 非中文提问，先说明只用中文: {user_question[:40]}")
            llm_answer = FOREIGN_REPLY
            _remember("chitchat", llm_answer)
            await send_llm_answer(llm_answer)
            await send_subtitle(llm_answer)
        elif is_ticket_booking(user_question):
            print(f"🎫 门票预约，打开填写界面: {user_question[:40]}")
            await emit(ticket_form_event(user_question, session_id, getattr(persona, "id", "")))
            loop.run_in_executor(
                None,
                lambda: request_ticket_form(
                    user_question, getattr(persona, "id", ""), session_id))
            llm_answer = booking_reply()
            _remember("ticket_service", llm_answer)
            await send_llm_answer(llm_answer)
            await send_subtitle(llm_answer)
        elif rule_intent(user_question) in ("weather", "news"):
            kind = rule_intent(user_question)
            note_realtime(dialogue_state)
            bundle = {}
            if dialogue_state.realtime_streak <= 1:
                bundle = await loop.run_in_executor(None, lambda k=kind: feed_bundle(k))
            if kind == "news" and (bundle or {}).get("image_url"):
                await emit(headline_event(
                    bundle.get("title") or "", bundle.get("image_url") or "",
                    bundle.get("link") or "", session_id, getattr(persona, "id", "")))
                loop.run_in_executor(
                    None,
                    lambda b=bundle: request_headline_image(
                        b.get("title") or "", b.get("image_url") or "",
                        b.get("link") or "", getattr(persona, "id", ""), session_id))
            llm_answer = realtime_reply(dialogue_state, persona, (bundle or {}).get("text") or "")
            print(f"🌤 实时问题第{dialogue_state.realtime_streak}轮: {user_question[:40]}")
            _remember(kind, llm_answer)
            await send_llm_answer(llm_answer)
            await send_subtitle(llm_answer)
        elif visit_fact_reply(user_question, persona):
            llm_answer = visit_fact_reply(user_question, persona)
            note_hall(dialogue_state)
            print(f"🎫 参观信息用展厅句子: {user_question[:40]}")
            _remember("visit", llm_answer)
            await send_llm_answer(llm_answer)
            await send_subtitle(llm_answer)
        elif is_other_exhibition_query(user_question):
            print(f"🚫 其他展会问题，拉回本届: {user_question[:40]}")
            llm_answer = refuse_text()
            _remember("knowledge", llm_answer, refused=True)
            await send_llm_answer(llm_answer)
            await send_subtitle(llm_answer)
        elif _looks_chitchat(user_question):
            llm_answer = chitchat_reply(user_question, persona)
            _remember("chitchat", llm_answer)
            await send_llm_answer(llm_answer)
            await send_subtitle(llm_answer)
        elif _command_reply(user_question, dialogue_state):
            llm_answer = _command_reply(user_question, dialogue_state)
            _remember("command", llm_answer)
            await send_llm_answer(llm_answer)
            await send_subtitle(llm_answer)
        elif not enable_llm:
            llm_answer = user_question
            await send_llm_answer(llm_answer)
            sentences = [s.strip() for s in re.split(f"(?<=[{SENT_END}])", llm_answer) if s.strip()]
            for sent in sentences:
                await send_subtitle(sent)
                if cancelled:
                    break
        else:
            try:
                print(f"🤖 用户问题: {user_question}")
                intent, raw_score, _conf = await loop.run_in_executor(
                    None, intent_classifier.classify, user_question)
                route_intent = intent if (
                    intent in HALL_INTENTS
                    and intent_classifier.should_route_to(intent, raw_score)
                ) else "knowledge"
                if route_intent in HALL_INTENTS:
                    note_hall(dialogue_state)
                hits = await loop.run_in_executor(
                    None, lambda: search_fn(user_question, 2, locked_category(), None, active()))
                threshold = MIN_SCORE_RERANK if rag.HAS_RERANKER else MIN_SCORE_RRF

                if hits:
                    print(f"🔍 Top1 分数: {hits[0]['score']} (阈值 {threshold}) | "
                          f"{hits[0]['title']} | 检索耗时 {time.time()-t0:.2f}s")

                refused_round = False
                if cancel_ev.is_set():
                    cancelled = True
                elif not hits or hits[0]["score"] < threshold:
                    refused_round = True
                    llm_answer = refuse_text()
                    await send_llm_answer(llm_answer)
                    await send_subtitle(llm_answer)
                else:
                    contexts = [h["text"][:COMPRESS_MAX] for h in hits]
                    ref = "\n".join(f"- {c}" for c in contexts)
                    user_content = (
                        f"【参考资料】\n{ref}\n\n【问题】{user_question}\n\n"
                        f"{works_answer_hint(user_question, persona)}"
                        f"{visit_hint(user_question, persona)}"
                        "只根据参考资料讲这场展，不要提及其他展览。\n/no_think"
                    )
                    llm_payload = {"messages": [{"role": "system", "content": system_prompt()},
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
                    clock_state = {}

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
                            if not first_logged:
                                sent = strip_identity_opening(sent, persona)
                                if not sent:
                                    continue
                            sent, stop = hall_safe_sentence(sent)
                            sent = filter_spoken_sentence(
                                sent, user_question, persona, clock_state)
                            if not sent:
                                continue
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
                        if tail and not first_logged:
                            tail = strip_identity_opening(tail, persona)
                        if tail:
                            tail, stop = hall_safe_sentence(tail)
                            tail = filter_spoken_sentence(
                                tail, user_question, persona, clock_state)
                            if not first_logged:
                                first_logged = True
                            await send_subtitle(tail)
                            if stop:
                                leaked = True

                    if not cancelled:
                        if leaked:
                            refused_round = True
                        llm_answer = refuse_text() if refused_round else guard_hall_clock(
                            user_question,
                            clamp_hall_answer(clean_full_answer(full_raw) or refuse_text()),
                            persona)
                        print(f"✅ LLM 回答: {llm_answer[:50]}... | 总耗时 {time.time()-t0:.2f}s")
                        await send_llm_answer(llm_answer)
                        _remember(route_intent, llm_answer,
                                  hall=route_intent in HALL_INTENTS, refused=refused_round)
                        if control_box["exhibit_names"]:
                            await emit(artworks_event(
                                control_box["exhibit_names"], session_id,
                                getattr(persona, "id", "")))
                            found = list(control_box["exhibit_names"])
                            loop.run_in_executor(
                                None,
                                lambda names=found, reply=llm_answer: notify_artworks(
                                    names, reply, getattr(persona, "id", ""), session_id))

            except Exception as e:
                print(f"⚠️ LLM 调用失败: {e}")
                if not cancel_ev.is_set():
                    llm_answer = "抱歉，我现在有点不舒服。"
                    await send_llm_answer(llm_answer)
                    await send_subtitle(llm_answer)

        await flush_pending()
        await audio_queue.put(None)
        await audio_task

        release_fut = None
        if not tts._remote_on():
            from tts.cuda_cache import release_cuda_cache
            release_fut = loop.run_in_executor(None, release_cuda_cache, "ws-turn")
        duration = play_clock.duration or (
            pcm_bytes_sent / 2.0 / float(tts_output_rate()) if pcm_bytes_sent else 0.0)
        remain = play_clock.remain()
        if remain > 0.05 and not cancelled and not cancel_ev.is_set():
            print(f"等播完再 tts_end remain={remain:.2f}s pcm={duration:.2f}s session={session_id[:8]}")
            await asyncio.sleep(remain)
        if release_fut is not None:
            try:
                await release_fut
            except Exception as rel_err:
                print(f"⚠️ 释放 Qwen CUDA 缓存失败: {rel_err}")

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
            await flush_pending()
            await audio_queue.put(None)
            await audio_task
            cancelled = True
            release_fut = None
            if not tts._remote_on():
                from tts.cuda_cache import release_cuda_cache
                release_fut = loop.run_in_executor(None, release_cuda_cache, "ws-turn-error")
            remain = play_clock.remain()
            if remain > 0.05 and not cancel_ev.is_set():
                await asyncio.sleep(min(remain, 8.0))
            if release_fut is not None:
                try:
                    await release_fut
                except Exception as rel_err:
                    print(f"⚠️ 释放 Qwen CUDA 缓存失败: {rel_err}")
            await _emit_tts_end()
        except Exception:
            pass
    finally:
        conn_sessions.pop(session_id, None)

async def websocket_tts_stream(websocket: WebSocket, persona_id: Optional[str] = None):
    from fastapi import HTTPException
    path = websocket.scope.get("path", "")
    print(f"🔌 TTS WS 握手 path={path} persona_arg={persona_id!r} client={websocket.client}")
    runtime_log("ws.log", "[WS] tts path=%s persona=%r client=%s" % (
        path, persona_id, websocket.client))
    try:
        persona = resolve(persona_id, allow_default=True)
    except HTTPException as e:
        # 未 accept 就 close，Starlette 会回 HTTP 403，Unity 只看到 Forbidden
        try:
            await accept_websocket(websocket)
            await websocket.send_json({
                "type": "error", "code": "unknown_persona",
                "message": str(e.detail),
                "hint": registry.endpoint_hint("tts"),
            })
            await websocket.close(code=1008, reason=str(e.detail)[:120])
        except Exception:
            pass
        print(f"⚠️ TTS WS 拒绝 path={path}: {e.detail}")
        return
    with bind(persona):
        await _websocket_tts_loop(websocket, persona)


async def _websocket_tts_loop(websocket: WebSocket, persona):
    await accept_websocket(websocket)

    # 会话/取消状态按连接隔离，互不干扰
    conn_sessions: Dict[str, threading.Event] = {}
    conn_tasks: set = set()
    send_lock = asyncio.Lock()
    latest_sid = [""]
    dialogue_state = RealtimeMemory()

    async def safe_send(obj: dict):
        payload = attach_robot_id(obj, "")
        async with send_lock:
            await websocket.send_json(payload)

    def cancel_all():
        for ev in conn_sessions.values():
            ev.set()

    def begin_tts_session():
        """先切最新 session_id，再 cancel 旧任务，避免旧 tts_end 仍被视为当前轮。"""
        session_id = str(uuid.uuid4())
        latest_sid[0] = session_id
        cancel_all()
        cancel_ev = threading.Event()
        conn_sessions[session_id] = cancel_ev
        return session_id, cancel_ev

    def spawn_tts(text: str, speaker: str, enable_llm: bool):
        session_id, cancel_ev = begin_tts_session()
        task = asyncio.create_task(handle_tts_request(
            websocket, safe_send, text, speaker,
            enable_llm, session_id, cancel_ev, conn_sessions, latest_sid,
            dialogue_state))
        conn_tasks.add(task)
        task.add_done_callback(conn_tasks.discard)
        return task

    await tts_fanout.register(websocket, send_lock, conn_sessions, persona_id=persona.id)
    await safe_send({"type": "ready", "service": TTS_SERVICE_NAME,
                     "persona_id": persona.id,
                     "display_name": persona.display_name})
    print(f"🎤 TTS WebSocket 连接建立 persona={persona.id}")
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
                # 机器人正在推这一轮时，仅拦截【当前播报人设】的游客 tts/cancel。
                # 其他人设连接不受拦截；GPU 互斥由 robot acquire 409 保证。
                # 参考 robot-ask-ws-改动说明.md + 项目整体脉络.md。
                if robot_hub.is_busy_for(persona.id):
                    print(f"⏭ 忽略游客 TTS（机器人正在推流）persona={persona.id}: {user_question[:40]}")
                    continue
                if robot_hub.is_robot_ask_echo(user_question, persona_id=persona.id):
                    print(f"⏭ 忽略游客 TTS（与当前机器人 ask 相同，避免 asr 回声）"
                          f" persona={persona.id}: {user_question[:40]}")
                    continue
                if not tts.cosyvoice_model:
                    print(f"⚠️ CosyVoice 当前未就绪，本轮将补加载后再合成 persona={persona.id}")

                # 新请求打断本连接的旧会话
                spawn_tts(user_question, speaker, enable_llm)

            elif msg_type == "hello":
                hello = pick_hello(persona)
                spawn_tts(hello, data.get("speaker", "中文男"), False)
            elif msg_type == "switch_persona":
                pid = (data.get("persona_id") or "").strip()
                if pid and pid != persona.id:
                    await safe_send({
                        "type": "switch_ack", "ok": False, "code": "persona_bound",
                        "message": f"本连接绑定 {persona.id}，请改连 /ws/{pid}/tts",
                        "persona_id": persona.id,
                    })
                    continue
                from persona.stage import stage
                result = await stage.switch(persona.id, play_hello=bool(data.get("play_hello")), reason="ws")
                await safe_send({"type": "switch_ack", **result})
                if data.get("play_hello"):
                    hello = pick_hello(persona)
                    spawn_tts(hello, data.get("speaker", "中文男"), False)
            elif msg_type == "cancel":
                if robot_hub.is_busy_for(persona.id):
                    print(f"⏭ 忽略游客 cancel（机器人正在推流，本轮由 robot stream 收尾）"
                          f" persona={persona.id}")
                    continue
                print(f"收到取消指令，标记本连接所有活跃会话 persona={persona.id}")
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
