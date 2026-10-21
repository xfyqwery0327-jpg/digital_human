# -*- coding: utf-8 -*-
"""机器人 HTTP 一轮 → 前端 /ws/tts（asr → tts_start/chunk/end）。不走本机声卡。"""
from __future__ import annotations
import asyncio
import io
import re
import threading
import time
import wave
from typing import Dict, List

from config import TTS_SAMPLE_RATE, REFUSE_TEXT, ROBOT_PLAY_TAIL_SEC
from llm.client import SENT_END
from tts import cosyvoice as tts
from tts.cosyvoice import push_sentence_pcm
from ws.hub import tts_fanout, attach_robot_id
from robot.logutil import get_logger
from rag.scope import clamp_hall_answer, hall_safe_sentence

log = get_logger("monet.robot.stream", "robot-api.log")


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    if not pcm:
        return b""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


async def push_robot_asr(query: str, session_id: str, robot_id: str) -> int:
    """把 API 提问词推到前端，格式与游客 ASR 一致。"""
    text = (query or "").strip()
    if not text:
        return 0
    n = await tts_fanout.broadcast(attach_robot_id({
        "type": "asr",
        "text": text,
        "session_id": session_id,
        "robot_id": (robot_id or "").strip() or "default",
    }, (robot_id or "").strip() or "default"))
    log.info("已推 asr session=%s robot=%s clients=%s text=%s",
             session_id, robot_id, n, text[:80])
    return n


def split_speak_sentences(text: str) -> List[str]:
    sentences = [s.strip() for s in re.split(f"(?<=[{SENT_END}])", text or "") if s.strip()]
    if sentences:
        return sentences
    stripped = (text or "").strip()
    return [stripped] if stripped else []


async def push_robot_turn_to_ws(speak_text: str, speaker: str,
                                session_id: str, robot_id: str,
                                asr_text: str = "") -> Dict:
    """一轮只推一次。顺序：asr → tts_start → llm_answer → sentence/chunk → tts_end。"""
    rid = (robot_id or "").strip() or "default"
    speak_text = clamp_hall_answer(speak_text)
    raw_sentences = split_speak_sentences(speak_text)
    sentences = []
    for sent in raw_sentences:
        sent, stop = hall_safe_sentence(sent)
        sentences.append(sent)
        if stop:
            speak_text = REFUSE_TEXT
            break
    n_clients = tts_fanout.client_count()
    cancelled_n = tts_fanout.cancel_public()
    if cancelled_n:
        log.info("机器人开口前打断游客 TTS n=%s", cancelled_n)
        await asyncio.sleep(0.05)

    last_pushed = 0
    tts_started = False
    tts_ended = False
    llm_sent = False
    start_hint = sentences[0] if sentences else ""

    def _stamp(obj: dict) -> dict:
        payload = attach_robot_id(obj, rid)
        payload["robot_id"] = rid
        return payload

    async def fanout(obj: dict) -> int:
        nonlocal last_pushed, tts_started, tts_ended, llm_sent
        kind = obj.get("type")
        if kind == "tts_end":
            if tts_ended:
                log.warning("丢弃重复 tts_end session=%s robot=%s", session_id, rid)
                return last_pushed
            if not tts_started:
                log.warning("尚未 tts_start，跳过 tts_end session=%s", session_id)
                return last_pushed
            tts_ended = True
        if kind == "llm_answer":
            if llm_sent:
                return last_pushed
            llm_sent = True
        if kind in ("tts_chunk", "tts_sentence") and not tts_started:
            last_pushed = await tts_fanout.broadcast(_stamp({
                "type": "tts_start",
                "session_id": session_id,
                "format": "pcm",
                "sample_rate": TTS_SAMPLE_RATE,
                "channels": 1,
                "sample_width": 2,
                "text": start_hint,
            }))
            tts_started = True
            log.info("补发 tts_start 后再推 %s session=%s robot=%s", kind, session_id, rid)
        last_pushed = await tts_fanout.broadcast(_stamp(obj))
        if kind == "tts_start":
            tts_started = True
        return last_pushed

    if n_clients <= 0:
        log.warning("没有前端连接 /ws/tts，字幕和音频分片无人接收 robot=%s", rid)

    query = (asr_text or "").strip()
    if query:
        await fanout({
            "type": "asr",
            "text": query,
            "session_id": session_id,
        })
        log.info("已推 asr session=%s robot=%s clients=%s text=%s",
                 session_id, rid, last_pushed, query[:80])

    if not sentences:
        log.warning("本轮无播报文本 session=%s robot=%s", session_id, rid)
        return {
            "play_duration": 0.0, "wav_bytes": b"",
            "ws_pushed": last_pushed, "cancelled": False,
        }

    hint = sentences[0]
    await fanout({
        "type": "tts_start",
        "session_id": session_id,
        "format": "pcm",
        "sample_rate": TTS_SAMPLE_RATE,
        "channels": 1,
        "sample_width": 2,
        "text": hint,
    })
    log.info("已推 tts_start session=%s robot=%s clients=%s hint=%s",
             session_id, rid, last_pushed, hint[:40])
    t_play = time.time()
    await fanout({"type": "llm_answer", "text": speak_text})
    log.info("已推 llm_answer session=%s robot=%s chars=%s",
             session_id, rid, len(speak_text or ""))
    if last_pushed <= 0:
        log.warning("tts_start 未送达任何 /ws/tts 客户端，前端不会播报 session=%s robot=%s",
                    session_id, rid)
    else:
        await asyncio.sleep(0.2)

    pcm_buf = bytearray()

    def on_pcm(chunk: bytes):
        if chunk:
            pcm_buf.extend(chunk)

    seq = 0
    sent_idx = 0
    cancelled = False
    cancel_ev = threading.Event()

    log.info("开始逐句 TTS sentences=%s", len(sentences))
    t_tts = time.time()
    try:
        if tts.cosyvoice_model:
            for sent in sentences:
                seq, cancelled = await push_sentence_pcm(
                    None, fanout, sent, speaker, session_id, seq, cancel_ev,
                    subtitle_text=sent, subtitle_index=sent_idx, on_pcm=on_pcm)
                sent_idx += 1
                if cancelled:
                    break
        else:
            log.warning("TTS 未加载，只推字幕不推音频分片")
            for sent in sentences:
                await fanout({
                    "type": "tts_sentence",
                    "session_id": session_id,
                    "index": sent_idx,
                    "text": sent,
                })
                sent_idx += 1
    except Exception as e:
        log.exception("机器人 TTS 推送失败 session=%s err=%s", session_id, e)
        await fanout({"type": "error", "message": f"处理失败: {e}"})
        cancelled = True

    duration = (len(pcm_buf) / 2.0 / float(TTS_SAMPLE_RATE)) if pcm_buf else 0.0
    remain = duration + float(ROBOT_PLAY_TAIL_SEC or 0) - (time.time() - t_play)
    if remain > 0.05 and not cancelled:
        log.info("等播报结束再 tts_end remain=%.2fs pcm=%.2fs session=%s",
                 remain, duration, session_id)
        await asyncio.sleep(remain)

    await fanout({
        "type": "tts_end",
        "session_id": session_id,
        "seq_count": seq,
        "sentence_count": sent_idx,
        "cancelled": bool(cancelled),
    })
    wav_bytes = pcm_to_wav(bytes(pcm_buf), TTS_SAMPLE_RATE) if pcm_buf else b""
    log.info("已推 tts_end session=%s robot=%s seq=%s sentences=%s clients=%s tts_wall=%.2fs pcm=%.2fs",
             session_id, rid, seq, sent_idx, last_pushed, time.time() - t_tts, duration)

    return {
        "play_duration": duration,
        "wav_bytes": wav_bytes,
        "ws_pushed": last_pushed,
        "cancelled": cancelled,
    }
