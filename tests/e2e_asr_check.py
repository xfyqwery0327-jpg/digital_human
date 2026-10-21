# -*- coding: utf-8 -*-
"""端到端收音 + ASR 校验：连 /ws/monet/tts，POST /api/monet/robot/ask，
把所有 tts_chunk 拼成 24k 单声道 wav，再用 paraformer-zh 转写，
确认前端实际收到的不是杂音/静音。

用法：
    python tests/e2e_asr_check.py --persona-id monet --mode 3 --query "莫奈介绍下自己吧"
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
import urllib.request
import wave
from pathlib import Path

OUT_DIR = Path(r"C:\digital_human\tests\out")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def http_post_json(url: str, body: dict, timeout: float = 15.0):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json; charset=utf-8"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


async def collect(host: str, port: int, persona: str, query: str, mode: int,
                  timeout: float) -> dict:
    import websockets
    ws_url = f"ws://{host}:{port}/ws/{persona}/tts"
    http_url = f"http://{host}:{port}/api/{persona}/robot/ask"
    pcm = bytearray()
    sr = 24000
    info: dict = {"chunks": 0, "sentences": [], "llm": "", "seq_max": -1,
                  "t_start": None, "t_end": None, "engine": "", "tts_error": ""}

    async with websockets.connect(ws_url, max_size=8 * 1024 * 1024) as ws:
        await asyncio.wait_for(ws.recv(), timeout=5)  # ready
        t0 = time.time()
        body = {"query": query, "robot_id": "robot-001", "session_id": None,
                "include_audio": False, "wait": False, "speaker": "中文男",
                "top_k": 2, "mode": mode}
        status, resp = await asyncio.to_thread(http_post_json, http_url, body)
        info["http"] = status
        info["session_id"] = resp.get("session_id")
        if status not in (200, 202):
            info["http_body"] = resp
            return info

        deadline = t0 + timeout
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = obj.get("type")
            if kind == "tts_start":
                info["t_start"] = round(time.time() - t0, 2)
                sr = int(obj.get("sample_rate") or sr)
            elif kind == "llm_answer":
                info["llm"] = str(obj.get("text") or "")
            elif kind == "tts_sentence":
                info["sentences"].append(str(obj.get("text") or ""))
            elif kind == "tts_chunk":
                data = obj.get("data") or ""
                if data:
                    pcm.extend(base64.b64decode(data))
                info["chunks"] += 1
                info["seq_max"] = max(info["seq_max"], int(obj.get("seq", -1)))
            elif kind == "tts_end":
                info["t_end"] = round(time.time() - t0, 2)
                info["engine"] = obj.get("engine") or ""
                info["tts_error"] = obj.get("tts_error") or ""
                break
            elif kind == "error":
                info["tts_error"] = str(obj.get("message") or obj)

    info["sr"] = sr
    info["pcm_bytes"] = len(pcm)
    info["pcm_sec"] = round(len(pcm) / 2.0 / sr, 2) if pcm else 0.0
    wav = OUT_DIR / f"e2e_{persona}_mode{mode}.wav"
    if pcm:
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(bytes(pcm))
        info["wav"] = str(wav)
    return info


def asr_check(wav_path: str) -> str:
    from funasr import AutoModel
    model = AutoModel(model="paraformer-zh")
    res = model.generate(input=wav_path, batch_size_s=300)
    if isinstance(res, list) and res:
        return str(res[0].get("text") or "")
    return ""


async def main_async(args):
    info = await collect(args.host, args.port, args.persona_id, args.query,
                         args.mode, args.timeout)
    print(json.dumps({k: v for k, v in info.items() if k != "sentences"},
                     ensure_ascii=False, indent=2))
    if not info.get("wav"):
        print("没有收到 PCM，跳过 ASR")
        return 1
    print(f"\n字幕 {len(info['sentences'])} 句：")
    for i, s in enumerate(info["sentences"]):
        print(f"  #{i} {s}")
    text = await asyncio.to_thread(asr_check, info["wav"])
    print(f"\nASR 识别：{text}")
    print(f"LLM 原文：{info['llm'][:120]}")
    return 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8848)
    p.add_argument("--persona-id", default="monet", choices=["monet", "fuhao", "kumarajiva"])
    p.add_argument("--mode", type=int, default=3, choices=[1, 2, 3])
    p.add_argument("--query", default="莫奈介绍下自己吧")
    p.add_argument("--timeout", type=float, default=240.0)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async(parse_args())))
