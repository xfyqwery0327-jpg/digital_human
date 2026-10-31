# -*- coding: utf-8 -*-
"""口语管线延迟基准：首句音出来时间 + 句与句间隔。

测量路径：
  1) 真实对话（POST /api/{p}/robot/ask mode=3）→ 经 /ws/{p}/tts 推流。
     记录 提问→首帧 tts_chunk（首句音出来时间 TTFA）、每句开始/结束、句间间隔。
  2) 受控多句（/ws/{p}/tts 发 enable_llm=False 固定四句）→ 直接测引擎的句间间隔。

首句音 = 第一个 tts_chunk 到达；句间间隔 = 上一句最后 chunk → 下一句首个 chunk。
"""
import asyncio
import base64
import json
import os
import random
import statistics
import time

import websockets
import requests

BASE_HTTP = os.environ.get("BENCH_HTTP", "http://127.0.0.1:8848")
WS_BASE = os.environ.get("BENCH_WS", "ws://127.0.0.1:8848")
WARMUP = int(os.environ.get("BENCH_WARMUP", "3"))
ROUNDS = int(os.environ.get("BENCH_ROUNDS", "9"))
SESSION_TIMEOUT = int(os.environ.get("BENCH_TIMEOUT", "260"))
SETTLE = 3.0  # 轮间等待 robot_hub.mark_idle

QUESTIONS = {
    "monet": [
        "介绍一下日出印象", "睡莲是在哪里画的", "鲁昂大教堂讲了什么",
        "讲讲莫奈的生平", "干草堆有什么特别", "这场展览的开放时间",
        "门票多少钱", "吉维尼花园是什么", "这次展览的主题是什么",
    ],
    "fuhao": [
        "妇好墓出土了什么", "甲骨文是什么", "介绍一下祭祀",
        "你怎么领兵出征", "妇好是谁", "这件青铜器是什么",
        "这场展几点开门", "门票多少钱", "玉器讲了什么",
    ],
    "kumarajiva": [
        "金刚经讲什么", "法华经是怎么译的", "舍利盒上画的是什么",
        "鸠摩罗什是谁", "你为什么去长安译经", "讲讲你的生平",
        "这场展几点开门", "造像讲了什么", "壁画有什么",
    ],
}
CONTROLLED = {
    "monet": "欢迎来到浪漫与星辰特展。我是莫奈，很高兴为你讲解。今天我们先看睡莲。随后可以前往鲁昂大教堂展区。",
    "fuhao": "欢迎来到这一场商代的展。我是妇好，既主祭祀也领兵。今日我们先看妇好墓的青铜。随后可以听听甲骨上的战事。",
    "kumarajiva": "欢迎来到龟兹佛教文化展。我是鸠摩罗什，在长安译过经。今日我们先看舍利盒。随后可以听听金刚经的故事。",
}


def _stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    xs2 = sorted(xs)
    return {
        "n": len(xs2),
        "min": round(min(xs2), 2),
        "median": round(statistics.median(xs2), 2),
        "mean": round(statistics.mean(xs2), 2),
        "max": round(max(xs2), 2),
    }


def _build_record(persona, kind, question, t0, recs, round_no=0):
    recs = sorted(recs, key=lambda r: r[0])
    sr = None
    t_tts_start = t_llm_answer = t_first_chunk = t_tts_end = None
    cur = None
    sentences = {}  # idx -> dict
    for t, m in recs:
        typ = m.get("type")
        if typ == "tts_start":
            t_tts_start = t
            sr = m.get("sample_rate") or 24000
        elif typ == "llm_answer":
            t_llm_answer = t
        elif typ == "tts_sentence":
            idx = m.get("index", cur if cur is not None else 0)
            cur = idx
            s = sentences.setdefault(idx, {"index": idx, "text": m.get("text", ""),
                                           "t_sentence": None, "t_first_chunk": None,
                                           "t_last_chunk": None, "pcm_bytes": 0})
            s["t_sentence"] = t
            if not s["text"]:
                s["text"] = m.get("text", "")
        elif typ == "tts_chunk":
            if cur is None:
                cur = 0
            s = sentences.setdefault(cur, {"index": cur, "text": "",
                                          "t_sentence": None, "t_first_chunk": None,
                                          "t_last_chunk": None, "pcm_bytes": 0})
            try:
                s["pcm_bytes"] += len(base64.b64decode(m.get("data", "")))
            except Exception:
                pass
            if s["t_first_chunk"] is None:
                s["t_first_chunk"] = t
            s["t_last_chunk"] = t
        elif typ == "tts_end":
            t_tts_end = t

    sent_list = [sentences[k] for k in sorted(sentences.keys())]
    for s in sent_list:
        s["audio_sec"] = round(s["pcm_bytes"] / 2.0 / float(sr or 24000), 2) if s["pcm_bytes"] else 0.0
        s["deliver_sec"] = round((s["t_last_chunk"] - s["t_first_chunk"]), 2) if s["t_first_chunk"] and s["t_last_chunk"] else None
        s["t_sentence"] = round(s["t_sentence"] - t0, 2) if s["t_sentence"] else None
        s["t_first_chunk"] = round(s["t_first_chunk"] - t0, 2) if s["t_first_chunk"] else None
        s["t_last_chunk"] = round(s["t_last_chunk"] - t0, 2) if s["t_last_chunk"] else None
        s["pcm_bytes"] = s["pcm_bytes"]

    gaps = []
    for i in range(len(sent_list) - 1):
        a, b = sent_list[i], sent_list[i + 1]
        ga = a.get("t_last_chunk")
        gb = b.get("t_first_chunk")
        if ga is not None and gb is not None:
            gaps.append(round(gb - ga, 2))

    return {
        "persona": persona,
        "kind": kind,
        "round": round_no,
        "question": question,
        "t_ask": round(t0, 3),
        "t_tts_start_rel": round(t_tts_start - t0, 2) if t_tts_start else None,
        "t_llm_answer_rel": round(t_llm_answer - t0, 2) if t_llm_answer else None,
        "first_audio_latency": round(t_first_chunk - t0, 2) if t_first_chunk else None,
        "wall_total": round(t_tts_end - t0, 2) if t_tts_end else None,
        "n_sentences": len(sent_list),
        "sentence_count_from_end": None,
        "sentences": [
            {k: s[k] for k in ("index", "text", "t_sentence", "t_first_chunk",
                               "t_last_chunk", "audio_sec", "deliver_sec", "pcm_bytes")}
            for s in sent_list
        ],
        "gaps": gaps,
        "gap_stats": _stats(gaps),
    }


async def _wait_for_round(frames, t0, timeout):
    sid = None
    deadline = t0 + timeout
    while time.time() < deadline:
        for t, m in frames:
            if m.get("type") == "tts_start":
                sid = m.get("session_id")
        if sid:
            break
        await asyncio.sleep(0.02)
    if not sid:
        return None, []
    while time.time() < deadline:
        for t, m in frames:
            if m.get("type") == "tts_end" and m.get("session_id") == sid:
                recs = [(tt, mm) for tt, mm in frames if mm.get("session_id") == sid]
                frames[:] = [(tt, mm) for tt, mm in frames if mm.get("session_id") != sid]
                return sid, recs
        await asyncio.sleep(0.02)
    recs = [(tt, mm) for tt, mm in frames if mm.get("session_id") == sid]
    frames[:] = [(tt, mm) for tt, mm in frames if mm.get("session_id") != sid]
    return sid, recs


async def _one_ask(ws, http_ask_uri, persona, q, frames, kind, results, round_no=0):
    frames.clear()
    t0 = time.time()
    last_status = None
    for attempt in range(15):
        try:
            r = requests.post(http_ask_uri, json={"query": q, "mode": 3,
                                                 "robot_id": "bench", "persona_id": persona},
                              timeout=30)
            last_status = r.status_code
            if r.status_code == 409:
                await asyncio.sleep(3.0)
                continue
            if r.status_code not in (200, 202):
                results.append({"persona": persona, "kind": kind, "round": round_no,
                                "question": q, "error": f"HTTP {r.status_code}"})
                return
            break
        except Exception as e:
            await asyncio.sleep(2.0)
            last_status = f"exc:{e}"
    else:
        results.append({"persona": persona, "kind": kind, "round": round_no,
                        "question": q, "error": f"retry_exhausted {last_status}"})
        return
    sid, recs = await _wait_for_round(frames, t0, SESSION_TIMEOUT)
    rec = _build_record(persona, kind, q, t0, recs, round_no)
    if not sid:
        rec["error"] = "no_tts_end_timeout"
    results.append(rec)
    await asyncio.sleep(SETTLE)


async def _one_controlled(ws, persona, text, frames, results):
    frames.clear()
    t0 = time.time()
    await ws.send(json.dumps({"type": "tts", "text": text, "enable_llm": False,
                             "speaker": "中文男"}))
    sid, recs = await _wait_for_round(frames, t0, SESSION_TIMEOUT)
    rec = _build_record(persona, "controlled", text, t0, recs, 0)
    if not sid:
        rec["error"] = "no_tts_end_timeout"
    results.append(rec)


async def run_persona(persona, results):
    ws_uri = f"{WS_BASE}/ws/{persona}/tts"
    http_ask_uri = f"{BASE_HTTP}/api/{persona}/robot/ask"
    frames = []

    async def reader(ws):
        async for msg in ws:
            try:
                m = json.loads(msg)
            except Exception:
                continue
            frames.append((time.time(), m))

    async with websockets.connect(ws_uri, max_size=None) as ws:
        task = asyncio.create_task(reader(ws))
        # 预热：冷启动 / 模型装载 / CUDA 缓存
        print(f"== {persona}: 预热 {WARMUP} 轮 ==")
        for i in range(WARMUP):
            q = QUESTIONS[persona][i % len(QUESTIONS[persona])]
            await _one_ask(ws, http_ask_uri, persona, q, frames, "warmup", results, i + 1)
        # 正式测量
        print(f"== {persona}: 测量 {ROUNDS} 轮 ==")
        for i in range(ROUNDS):
            q = QUESTIONS[persona][i % len(QUESTIONS[persona])]
            await _one_ask(ws, http_ask_uri, persona, q, frames, "mode3", results, i + 1)
            if (i + 1) % 3 == 0:
                print(f"   {persona} 已完成 {i + 1}/{ROUNDS}")
        # 受控多句
        text = CONTROLLED[persona] + f"（测{random.randint(1000, 9999)}）"
        await _one_controlled(ws, persona, text, frames, results)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def main():
    results = []
    os.makedirs("logs", exist_ok=True)
    t_all = time.time()
    personas = tuple(p for p in os.environ.get("PERSONAS", "monet,fuhao,kumarajiva").split(",") if p)
    for persona in personas:
        await run_persona(persona, results)
    elapsed = time.time() - t_all

    # 汇总
    by_p = {}
    for r in results:
        if r.get("error") or r.get("kind") in ("warmup",):
            continue
        by_p.setdefault(r["persona"], {"mode3_ttfa": [], "mode3_sent": [],
                                       "controlled_gaps": [], "controlled_ttfa": [],
                                       "controlled_n_sent": []})
        if r["kind"] == "mode3":
            if r.get("first_audio_latency") is not None:
                by_p[r["persona"]]["mode3_ttfa"].append(r["first_audio_latency"])
            by_p[r["persona"]]["mode3_sent"].append(r.get("n_sentences"))
        elif r["kind"] == "controlled":
            by_p[r["persona"]]["controlled_ttfa"].append(r.get("first_audio_latency"))
            by_p[r["persona"]]["controlled_gaps"].extend(r.get("gaps", []))
            by_p[r["persona"]]["controlled_n_sent"].append(r.get("n_sentences"))

    summary = {"total_elapsed_sec": round(elapsed, 1), "per_persona": {}}
    for p, v in by_p.items():
        summary["per_persona"][p] = {
            "mode3_first_audio_latency": _stats(v["mode3_ttfa"]),
            "mode3_sentences_per_round": _stats(v["mode3_sent"]),
            "controlled_first_audio_latency": _stats(v["controlled_ttfa"]),
            "controlled_n_sentences": _stats(v["controlled_n_sent"]),
            "controlled_inter_sentence_gap": _stats(v["controlled_gaps"]),
        }

    out_path = f"logs/bench_spoken_{int(t_all)}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": results}, f, ensure_ascii=False, indent=2)

    print("\n================ 汇总 ================")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n详细记录: {out_path}")
    return out_path


if __name__ == "__main__":
    asyncio.run(main())
