# -*- coding: utf-8 -*-
"""数字讲解员 API / WS 延迟基准测试。

测量两类交互的「首字时间 (TTFT)」与「中间回答间隔」：
  1) WS 对话 (/ws/{persona}/chat)：LLM 文本逐字流式，enable_audio=false 不触发 TTS。
  2) API (/api/{persona}/robot/ask)：立刻 202，真正内容经 TTS WS (/ws/{persona}/tts)
     以 asr -> tts_start -> llm_answer -> tts_sentence/tts_chunk -> tts_end 推流。

覆盖 local(直连 8848) 与 public(网关 face.broadmesse.net/modelAi/dh)。
"""
from __future__ import annotations
import asyncio
import json
import ssl
import statistics
import time
import uuid
from pathlib import Path

import websockets
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASES = {
    "local": "http://127.0.0.1:8848",
    "public": "https://face.broadmesse.net/modelAi/dh",
}


def ws_of(base: str) -> str:
    return base.replace("http://", "ws://").replace("https://", "wss://")


def _stats(xs):
    if not xs:
        return {"n": 0, "min": None, "max": None, "avg": None, "median": None}
    return {
        "n": len(xs),
        "min": round(min(xs) * 1000, 1),
        "max": round(max(xs) * 1000, 1),
        "avg": round(statistics.mean(xs) * 1000, 1),
        "median": round(statistics.median(xs) * 1000, 1),
    }


async def bench_chat(base: str, persona: str, question: str, timeout: float = 60.0):
    url = ws_of(base) + f"/ws/{persona}/chat"
    t_connect = time.time()
    async with websockets.connect(url, ping_interval=None,
                                 max_size=16 * 1024 * 1024) as ws:
        # chat WS 协议：服务端先等客户端发首帧(init_data)，再回 init。必须先发后收，否则死锁。
        await ws.send(json.dumps({"user_id": "bench", "session_id": None}))
        init = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
        t_init = time.time()
        assert init.get("type") == "init", f"期望 init，收到 {init.get('type')}"
        await ws.send(json.dumps({"question": question, "enable_audio": False}))
        t_sent = time.time()

        first_text_t = None
        text_ts = []
        chars = 0
        done = False
        while True:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
            except asyncio.TimeoutError:
                break
            t = time.time()
            mtype = msg.get("type")
            if mtype == "text":
                if first_text_t is None:
                    first_text_t = t
                text_ts.append(t)
                chars += len(msg.get("content", ""))
            elif mtype == "done":
                done = True
                break
            elif mtype == "error":
                return {"ok": False, "error": msg, "url": url}
        t_end = time.time()
        intervals = [text_ts[i + 1] - text_ts[i] for i in range(len(text_ts) - 1)]
        return {
            "ok": True,
            "url": url,
            "handshake_ms": round((t_init - t_connect) * 1000, 1),
            "ttft_ms": round((first_text_t - t_sent) * 1000, 1) if first_text_t else None,
            "text_frames": len(text_ts),
            "chars": chars,
            "total_ms": round((t_end - t_sent) * 1000, 1),
            "ended_with_done": done,
            "interval_ms": _stats(intervals),
        }


async def bench_api(base: str, persona: str, question: str, mode: int = 3, timeout: float = 200.0):
    url = ws_of(base) + f"/ws/{persona}/tts"
    async with websockets.connect(url, ping_interval=None,
                                 max_size=64 * 1024 * 1024) as ws:
        ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
        assert ready.get("type") == "ready", f"期望 ready，收到 {ready.get('type')}"

        t_post = time.time()
        rid = uuid.uuid4().hex
        sid = "bench_" + uuid.uuid4().hex[:8]
        resp = requests.post(
            base + f"/api/{persona}/robot/ask",
            json={"query": question, "session_id": sid, "robot_id": rid,
                  "include_audio": True, "mode": mode},
            timeout=30,
        )
        post_ms = (time.time() - t_post) * 1000
        status = resp.status_code
        body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}

        t_post_start = t_post  # 首字时间基准：POST 发起时刻
        t_start = t_llm = t_first_chunk = t_end = None
        sentence_ts = []
        chunk_ts = []
        seq_count = None
        ended = False
        err = None
        while True:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
            except asyncio.TimeoutError:
                err = "timeout_waiting_tts_end"
                break
            t = time.time()
            mtype = msg.get("type")
            if mtype == "tts_start" and t_start is None:
                t_start = t
            elif mtype == "llm_answer" and t_llm is None:
                t_llm = t
            elif mtype == "tts_sentence":
                if t_first_chunk is None and t_start is None:
                    t_start = t
                sentence_ts.append(t)
            elif mtype == "tts_chunk":
                if t_first_chunk is None:
                    t_first_chunk = t
                chunk_ts.append(t)
            elif mtype == "tts_end":
                t_end = t
                seq_count = msg.get("seq_count")
                ended = True
                break
            elif mtype == "error":
                err = msg.get("message")
        t_final = t_end or time.time()
        sentence_intervals = [sentence_ts[i + 1] - sentence_ts[i] for i in range(len(sentence_ts) - 1)]
        chunk_intervals = [chunk_ts[i + 1] - chunk_ts[i] for i in range(len(chunk_ts) - 1)]
        return {
            "ok": status == 202,
            "url": url,
            "http_status": status,
            "post_202_ms": round(post_ms, 1),
            "accepted": body.get("accepted"),
            "ttft_tts_start_ms": round((t_start - t_post_start) * 1000, 1) if t_start else None,
            "ttft_first_audio_ms": round((t_first_chunk - t_post_start) * 1000, 1) if t_first_chunk else None,
            "ttft_llm_answer_ms": round((t_llm - t_post_start) * 1000, 1) if t_llm else None,
            "sentences": len(sentence_ts),
            "audio_chunks": len(chunk_ts),
            "seq_count": seq_count,
            "total_ms": round((t_final - t_post_start) * 1000, 1),
            "ended_with_tts_end": ended,
            "sentence_interval_ms": _stats(sentence_intervals),
            "chunk_interval_ms": _stats(chunk_intervals),
            "error": err,
        }


async def wait_hub_idle(base: str, timeout: float = 120.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(base + "/health", timeout=10)
            if r.ok:
                busy = r.json().get("robot", {}).get("busy") is True
                if not busy:
                    return True
        except Exception:
            pass
        await asyncio.sleep(2)
    return False


async def main():
    results = {"chat": [], "api": []}
    print("=" * 70)
    print("数字讲解员 延迟基准测试")
    print("=" * 70)

    # ---- WS 对话（纯 LLM 文本流，不触发 TTS）----
    chat_cases = [
        ("local", "monet", "请介绍一下莫奈的艺术风格"),
        ("public", "monet", "请介绍一下莫奈的艺术风格"),
        ("public", "kumarajiva", "鸠摩罗什是谁，他翻译了哪些佛经经典"),
    ]
    for tag, persona, q in chat_cases:
        print(f"\n>>> WS chat [{tag}] persona={persona} q={q[:18]}...")
        try:
            r = await bench_chat(BASES[tag], persona, q)
        except Exception as e:
            r = {"ok": False, "error": str(e)[:200]}
        results["chat"].append({"tag": tag, "persona": persona, "question": q, **r})
        if r.get("ok"):
            print(f"    握手={r['handshake_ms']}ms  首字(TTFT)={r['ttft_ms']}ms  "
                  f"文本帧={r['text_frames']} 字数={r['chars']} 总耗时={r['total_ms']}ms")
            print(f"    文本帧间隔(ms): {r['interval_ms']}")
        else:
            print(f"    失败: {r.get('error')}")

    # ---- API + TTS 推流（口语合成管线）----
    await wait_hub_idle(BASES["public"])
    api_cases = [
        ("public", "monet", "请介绍一下莫奈的睡莲这幅作品，讲讲它的艺术特色", 3),
    ]
    for tag, persona, q, mode in api_cases:
        print(f"\n>>> API+WS tts [{tag}] persona={persona} mode={mode} q={q[:18]}...")
        try:
            r = await bench_api(BASES[tag], persona, q, mode=mode)
        except Exception as e:
            r = {"ok": False, "error": str(e)[:200]}
        results["api"].append({"tag": tag, "persona": persona, "question": q, **r})
        if r.get("ok"):
            print(f"    HTTP 202 延迟={r['post_202_ms']}ms  accepted={r['accepted']}")
            print(f"    首音(tts_start)={r['ttft_tts_start_ms']}ms  "
                  f"首帧音频={r['ttft_first_audio_ms']}ms  llm_answer={r['ttft_llm_answer_ms']}ms")
            print(f"    句子数={r['sentences']} 音频分片={r['audio_chunks']}  "
                  f"总耗时={r['total_ms']}ms  ended={r['ended_with_tts_end']}")
            print(f"    句子间隔(ms): {r['sentence_interval_ms']}")
            print(f"    音频分片间隔(ms): {r['chunk_interval_ms']}")
        else:
            print(f"    失败: {r.get('error')}  status={r.get('http_status')}")

    # 保存 JSON
    out = RESULTS_DIR / f"bench_{int(time.time())}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out}")
    return results


if __name__ == "__main__":
    asyncio.run(main())
