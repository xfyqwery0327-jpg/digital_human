# -*- coding: utf-8 -*-
"""机器人 HTTP 一轮 → 前端 /ws/{persona}/tts。

推流顺序（参考 robot-ask-ws-改动说明.md，叠加双人设隔离）：
asr → tts_start → llm_answer → tts_sentence/tts_chunk → 等 PCM 真实时长+ROBOT_PLAY_TAIL_SEC → tts_end
仅向该 persona 的 TTS 连接推送，禁止全局广播。
"""
from __future__ import annotations
import array
import asyncio
import base64
import io
import re
import sys
import threading
import time
import wave
from typing import Dict, List

from llm.client import SENT_END, strip_identity_opening
from tts import cosyvoice as tts
from tts.cosyvoice import push_sentence_pcm, pack_tts_utterances, tts_output_rate
from tts.playback import PcmPlayClock
from ws.hub import tts_fanout, attach_robot_id
from robot.logutil import get_logger
from rag.scope import clamp_hall_answer, hall_safe_sentence, refuse_text

log = get_logger("monet.robot.stream", "robot-api.log")


def load_wav_pcm16(path) -> tuple:
    """读预合成 wav，转成单声道 s16le PCM 和采样率。"""
    with wave.open(str(path), "rb") as w:
        sr = int(w.getframerate() or 0)
        channels = int(w.getnchannels() or 1)
        width = int(w.getsampwidth() or 0)
        raw = w.readframes(w.getnframes())
    if sr <= 0 or not raw:
        raise ValueError(f"wav 无有效音频: {path}")
    if width != 2:
        raise ValueError(f"wav 需要 16-bit PCM: {path} width={width}")
    if channels == 1:
        return raw, sr
    samples = array.array("h")
    samples.frombytes(raw[:len(raw) - (len(raw) % (2 * channels))])
    if sys.byteorder != "little":
        samples.byteswap()
    mono = array.array("h")
    for i in range(0, len(samples) - channels + 1, channels):
        acc = 0
        for c in range(channels):
            acc += samples[i + c]
        mono.append(int(acc / channels))
    if sys.byteorder != "little":
        mono.byteswap()
    return mono.tobytes(), sr


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


def _resolve_persona_id(persona_id: str = "") -> str:
    """遗留不带 persona_id 的路由映射为 monet。"""
    pid = (persona_id or "").strip()
    if pid:
        return pid
    try:
        from persona.runtime import active
        return active().id or "monet"
    except Exception:
        return "monet"


async def push_robot_asr(query: str, session_id: str, robot_id: str,
                         persona_id: str = "") -> int:
    """把 API 提问词推到该人设前端，格式与游客 ASR 一致。"""
    text = (query or "").strip()
    if not text:
        return 0
    pid = _resolve_persona_id(persona_id)
    rid = (robot_id or "").strip() or "default"
    n = await tts_fanout.send(attach_robot_id({
        "type": "asr",
        "text": text,
        "session_id": session_id,
        "robot_id": rid,
    }, rid, persona_id=pid), persona_id=pid)
    log.info("已推 asr session=%s robot=%s persona=%s clients=%s text=%s",
             session_id, rid, pid, n, text[:80])
    return n


def pcm_sentence_spans(pcm: bytes, weights: List[int]) -> List[bytes]:
    """按字数把一段 int16 PCM 切成和字幕对齐的片段。最后一段吃掉余数。"""
    usable = len(pcm) - (len(pcm) % 2)
    if usable <= 0 or not weights:
        return []
    total = sum(max(int(w), 1) for w in weights) or 1
    spans: List[bytes] = []
    start = 0
    acc = 0
    last = len(weights) - 1
    for i, w in enumerate(weights):
        acc += max(int(w), 1)
        if i == last:
            end = usable
        else:
            end = (int(usable * acc / total) // 2) * 2
            if end < start:
                end = start
        spans.append(pcm[start:end])
        start = end
    return spans


def split_speak_sentences(text: str) -> List[str]:
    sentences = [s.strip() for s in re.split(f"(?<=[{SENT_END}])", text or "") if s.strip()]
    if sentences:
        return sentences
    stripped = (text or "").strip()
    return [stripped] if stripped else []


async def push_robot_turn_to_ws(speak_text: str, speaker: str,
                                session_id: str, robot_id: str,
                                asr_text: str = "", persona_id: str = "",
                                fixed_wav: str = "",
                                preserve_text: bool = False) -> Dict:
    """一轮只推一次。顺序：asr → tts_start → llm_answer → sentence/chunk → tts_end。
    只推 /ws/{persona_id}/tts；旧 /ws/tts 视为 monet。

    fixed_wav 有值时把预录 wav 转成 PCM 推流。
    preserve_text 为真时口播稿不改写；没有 fixed_wav 则交给 CosyVoice 合成。
    """
    rid = (robot_id or "").strip() or "default"
    pid = _resolve_persona_id(persona_id)
    try:
        from persona.registry import registry
        from persona.runtime import peek
        persona_obj = registry.get(pid)
        peeked = peek()
        if peeked is not None and peeked.id != pid:
            log.warning("人设上下文与路由不一致 peek=%s route=%s，按路由合成",
                        peeked.id, pid)
        if persona_obj is None:
            persona_obj = peeked
    except Exception:
        persona_obj = None
    use_canned = bool(fixed_wav)
    keep_text = use_canned or bool(preserve_text)
    canned_pcm = b""
    canned_sr = 0
    fixed_err = ""
    if not keep_text:
        speak_text = clamp_hall_answer(speak_text, persona_obj)
    sr = tts_output_rate()
    raw_sentences = split_speak_sentences(speak_text)
    sentences: List[str] = []
    leaked = False
    first = True
    if keep_text:
        sentences = list(raw_sentences)
    else:
        for sent in raw_sentences:
            if first:
                sent = strip_identity_opening(sent, persona_obj)
                first = False
                if not sent:
                    continue
            sent, stop = hall_safe_sentence(sent, persona_obj)
            if stop:
                leaked = True
                break
            if sent:
                sentences.append(sent)
        if leaked:
            speak_text = refuse_text(persona_obj)
            sentences = split_speak_sentences(speak_text)
            log.info("口播越界，改口拒答 persona=%s", pid)
        sentences = pack_tts_utterances(sentences, persona=persona_obj)
    n_clients = tts_fanout.client_count(pid)
    cancelled_n = tts_fanout.cancel_public(pid)
    if cancelled_n:
        log.info("机器人开口前打断游客 TTS n=%s persona=%s", cancelled_n, pid)
        await asyncio.sleep(0.05)

    last_pushed = 0
    tts_started = False
    tts_ended = False
    llm_sent = False
    start_hint = sentences[0] if sentences else ""

    def _stamp(obj: dict) -> dict:
        payload = attach_robot_id(obj, rid, persona_id=pid)
        payload["robot_id"] = rid
        payload["persona_id"] = pid
        return payload

    async def fanout(obj: dict) -> int:
        nonlocal last_pushed, tts_started, tts_ended, llm_sent
        kind = obj.get("type")
        if kind == "tts_end":
            if tts_ended:
                log.warning("丢弃重复 tts_end session=%s robot=%s persona=%s",
                            session_id, rid, pid)
                return last_pushed
            if not tts_started:
                log.warning("尚未 tts_start，跳过 tts_end session=%s persona=%s",
                            session_id, pid)
                return last_pushed
            tts_ended = True
        if kind == "llm_answer":
            if llm_sent:
                log.warning("丢弃重复 llm_answer session=%s persona=%s", session_id, pid)
                return last_pushed
            llm_sent = True
        if kind in ("tts_chunk", "tts_sentence") and not tts_started:
            last_pushed = await tts_fanout.send(_stamp({
                "type": "tts_start",
                "session_id": session_id,
                "format": "pcm",
                "sample_rate": sr,
                "channels": 1,
                "sample_width": 2,
                "text": start_hint,
            }), persona_id=pid)
            tts_started = True
            log.info("补发 tts_start 后再推 %s session=%s robot=%s persona=%s",
                     kind, session_id, rid, pid)
        last_pushed = await tts_fanout.send(_stamp(obj), persona_id=pid)
        if kind == "tts_start":
            tts_started = True
        return last_pushed

    if n_clients <= 0:
        log.warning("没有前端连接 /ws/%s/tts，字幕和音频分片无人接收 robot=%s persona=%s",
                    pid, rid, pid)

    query = (asr_text or "").strip()
    if query:
        await fanout({
            "type": "asr",
            "text": query,
            "session_id": session_id,
        })
        log.info("已推 asr session=%s robot=%s persona=%s clients=%s text=%s",
                 session_id, rid, pid, last_pushed, query[:80])

    if not sentences and not use_canned:
        log.warning("本轮无播报文本 session=%s robot=%s persona=%s", session_id, rid, pid)
        return {
            "play_duration": 0.0, "wav_bytes": b"",
            "ws_pushed": last_pushed, "cancelled": False,
        }

    if use_canned:
        try:
            canned_pcm, canned_sr = await asyncio.get_running_loop().run_in_executor(
                None, load_wav_pcm16, fixed_wav)
            sr = canned_sr
            log.info("固定讲稿已转 PCM wav=%s bytes=%s sr=%s",
                     fixed_wav, len(canned_pcm), sr)
        except Exception as e:
            fixed_err = str(e)
            log.exception("固定讲稿 wav 转 PCM 失败 wav=%s err=%s", fixed_wav, e)
    elif not tts.cosyvoice_model:
        log.warning("TTS 未就绪，补加载后再推 tts_start persona=%s err=%s",
                    pid, tts._tts_last_error or "-")
        await asyncio.get_running_loop().run_in_executor(
            None, tts.ensure_cosyvoice_loaded)
    if not use_canned:
        sr = tts_output_rate()
    hint = sentences[0] if sentences else (speak_text or "")[:40]
    await fanout({
        "type": "tts_start",
        "session_id": session_id,
        "format": "pcm",
        "sample_rate": sr,
        "channels": 1,
        "sample_width": 2,
        "text": hint,
    })
    log.info("已推 tts_start session=%s robot=%s persona=%s clients=%s sr=%s hint=%s",
             session_id, rid, pid, last_pushed, sr, hint[:40])
    await fanout({"type": "llm_answer", "text": speak_text})
    log.info("已推 llm_answer session=%s robot=%s persona=%s chars=%s",
             session_id, rid, pid, len(speak_text or ""))
    if last_pushed <= 0:
        log.warning("tts_start 未送达任何 /ws/%s/tts 客户端，前端不会播报 session=%s robot=%s persona=%s",
                    pid, session_id, rid, pid)
    # 不再先睡 200ms 再合成：首句立刻进 GPU，与 Unity 建播放器重叠

    pcm_buf = bytearray()
    play_clock = PcmPlayClock(sr)

    def on_pcm(chunk: bytes):
        if chunk:
            pcm_buf.extend(chunk)
            play_clock.add(chunk)

    seq = 0
    sent_idx = 0
    cancelled = False
    cancel_ev = threading.Event()

    from tts.engine import tts_engine_name
    engine = tts_engine_name()
    log.info("开始逐句 TTS sentences=%s persona=%s canned=%s preserve=%s engine=%s model=%s err=%s",
             len(sentences), pid, bool(canned_pcm), keep_text, engine,
             bool(tts.cosyvoice_model), tts._tts_last_error or "-")
    t_tts = time.time()
    skip_reason = ""
    try:
        if use_canned and fixed_err:
            skip_reason = "fixed_wav_unreadable"
            await fanout({
                "type": "error",
                "session_id": session_id,
                "message": f"固定讲稿音频不可读: {fixed_err}",
            })
        elif use_canned and canned_pcm:
            chunk_size = tts._pcm_chunk_size()
            log.info("固定讲稿延后出声 2s persona=%s", pid)
            await asyncio.sleep(2)
            if sentences:
                await fanout({
                    "type": "tts_sentence",
                    "session_id": session_id,
                    "index": 0,
                    "text": sentences[0],
                })
                sent_idx = 1
            offset = 0
            while offset < len(canned_pcm):
                if cancel_ev.is_set():
                    cancelled = True
                    break
                chunk = canned_pcm[offset:offset + chunk_size]
                offset += chunk_size
                if not chunk:
                    continue
                on_pcm(chunk)
                await fanout({
                    "type": "tts_chunk",
                    "session_id": session_id,
                    "seq": seq,
                    "data": base64.b64encode(chunk).decode(),
                })
                seq += 1
                chunk_sec = len(chunk) / 2.0 / float(sr)
                await asyncio.sleep(max(0.0, chunk_sec * 0.85))
        elif tts.cosyvoice_model and engine == "qwen3":
            # 一轮讲稿只进一次 GPU。按逗号逐句克隆会把 6 秒参考音重算十几遍，
            # 90 秒等不到 tts_end，下一轮直接 409。
            full = "".join(sentences)
            log.info("Qwen 整段合成 chars=%s sentences=%s persona=%s",
                     len(full), len(sentences), pid)
            pcm_bytes = b""
            try:
                pcm_bytes, sr = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda text=full, who=persona_obj: tts.synthesize_speech_pcm(
                        text, speaker, cancel_ev, persona=who))
            except Exception as e:
                skip_reason = "qwen_synth_fail"
                log.exception("Qwen 整段合成失败 persona=%s err=%s", pid, e)
                await fanout({
                    "type": "error",
                    "session_id": session_id,
                    "message": "Qwen3-TTS 合成失败: %s" % e,
                })
            if not pcm_bytes:
                if not skip_reason:
                    skip_reason = "qwen_empty"
                log.warning("Qwen 整段无 PCM persona=%s chars=%s", pid, len(full))
                for sent in sentences:
                    await fanout({
                        "type": "tts_sentence",
                        "session_id": session_id,
                        "index": sent_idx,
                        "text": sent,
                    })
                    sent_idx += 1
            else:
                spans = pcm_sentence_spans(pcm_bytes, [len(s) for s in sentences])
                chunk_size = tts._pcm_chunk_size()
                for sent, span in zip(sentences, spans):
                    await fanout({
                        "type": "tts_sentence",
                        "session_id": session_id,
                        "index": sent_idx,
                        "text": sent,
                    })
                    sent_idx += 1
                    offset = 0
                    while offset < len(span):
                        if cancel_ev.is_set():
                            cancelled = True
                            break
                        chunk = span[offset:offset + chunk_size]
                        offset += chunk_size
                        if not chunk:
                            continue
                        on_pcm(chunk)
                        await fanout({
                            "type": "tts_chunk",
                            "session_id": session_id,
                            "seq": seq,
                            "data": base64.b64encode(chunk).decode(),
                        })
                        seq += 1
                        chunk_sec = len(chunk) / 2.0 / float(sr or 24000)
                        await asyncio.sleep(max(0.0, chunk_sec * 0.85))
                    if cancelled:
                        break
        elif tts.cosyvoice_model:
            for sent in sentences:
                pcm_before = len(pcm_buf)
                seq, cancelled = await push_sentence_pcm(
                    None, fanout, sent, speaker, session_id, seq, cancel_ev,
                    subtitle_text=sent, subtitle_index=sent_idx, on_pcm=on_pcm,
                    persona=persona_obj)
                added = len(pcm_buf) - pcm_before
                if added <= 0:
                    log.warning("句子无 PCM idx=%s persona=%s engine=%s err=%s text=%s",
                                sent_idx, pid, engine, tts._tts_last_error or "-", sent[:40])
                sent_idx += 1
                if cancelled:
                    break
        else:
            from tts.engine import tts_engine_name
            engine = tts_engine_name()
            err = tts._tts_last_error or "模型未加载"
            skip_reason = "qwen_not_loaded" if engine == "qwen3" else "cosyvoice_not_loaded"
            log.error("不下发 tts_chunk：%s 未加载 persona=%s err=%s",
                      engine, pid, err)
            await fanout({
                "type": "error",
                "session_id": session_id,
                "message": f"{engine} 未加载，没有音频: {err}",
            })
            for sent in sentences:
                await fanout({
                    "type": "tts_sentence",
                    "session_id": session_id,
                    "index": sent_idx,
                    "text": sent,
                })
                sent_idx += 1
    except Exception as e:
        log.exception("机器人 TTS 推送失败 session=%s persona=%s err=%s",
                      session_id, pid, e)
        try:
            await fanout({"type": "error", "message": f"处理失败: {e}"})
        except Exception:
            log.exception("推送 error 失败 session=%s persona=%s", session_id, pid)
        cancelled = True

    duration = play_clock.duration or (
        (len(pcm_buf) / 2.0 / float(sr)) if pcm_buf else 0.0)
    remain = play_clock.remain()
    if remain > 0.05 and not cancelled:
        log.info("等播报结束再 tts_end remain=%.2fs pcm=%.2fs session=%s persona=%s",
                 remain, duration, session_id, pid)
        await asyncio.sleep(remain)

    end = {
        "type": "tts_end",
        "session_id": session_id,
        "seq_count": seq,
        "sentence_count": sent_idx,
        "cancelled": bool(cancelled),
        "engine": engine,
        "sample_rate": sr,
    }
    if seq <= 0:
        end["audio"] = False
        if not skip_reason:
            skip_reason = "no_pcm"
        end["skip_reason"] = skip_reason
        if tts._tts_last_error:
            end["tts_error"] = tts._tts_last_error
    await fanout(end)
    wav_bytes = pcm_to_wav(bytes(pcm_buf), sr) if pcm_buf else b""
    log.info("已推 tts_end session=%s robot=%s persona=%s seq=%s sentences=%s "
             "clients=%s tts_wall=%.2fs pcm=%.2fs",
             session_id, rid, pid, seq, sent_idx, last_pushed,
             time.time() - t_tts, duration)

    return {
        "play_duration": duration,
        "wav_bytes": wav_bytes,
        "ws_pushed": last_pushed,
        "cancelled": cancelled,
    }
