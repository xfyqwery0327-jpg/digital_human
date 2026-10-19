# -*- coding: utf-8 -*-
"""通过 WebSocket 向莫奈、妇好提问，等一整轮回复后再发下一题。

默认走 /ws/{persona}/tts（即使 health.tts=false 也不改道）。
CosyVoice 未加载时服务端仍会发 tts_start / llm_answer / tts_sentence / tts_end。
需要纯文本时再显式 --channel chat。

用法（项目根或 scripts 目录）:

    python scripts/ws_ask_monet_fuhao.py
    python scripts/ws_ask_monet_fuhao.py --host 10.0.7.232 --port 8848
    python scripts/ws_ask_monet_fuhao.py --persona monet
    python scripts/ws_ask_monet_fuhao.py --channel chat
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_HOST = "10.0.7.232"
DEFAULT_PORT = 8848

QUESTIONS: Dict[str, List[str]] = {
    "monet": [
        "介绍下自己",
        "介绍下作品",
        "讲讲《日出·印象》",
        "睡莲是在哪里画的",
    ],
    "fuhao": [
        "介绍下你自己",
        "墓里出土了什么",
        "甲骨文是什么",
        "介绍一下祭祀",
    ],
    "kumarajiva": [
        "介绍下你自己",
        "金刚经讲什么",
        "法华经是怎么译的",
        "舍利盒上画的是什么",
    ],
}

DISPLAY = {"monet": "莫奈", "fuhao": "妇好", "kumarajiva": "鸠摩罗什"}


def _short(text: Any, n: int = 80) -> str:
    s = "" if text is None else str(text).replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


def _loads(raw) -> dict:
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return {"type": "raw", "raw": raw[:120]}
        return obj if isinstance(obj, dict) else {"type": "raw", "raw": str(obj)[:120]}
    return {"type": "raw", "raw": str(raw)[:120]}


async def _recv(ws, timeout: float) -> dict:
    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
    return _loads(raw)


def fetch_health(host: str, port: int, timeout: float = 8.0) -> dict:
    url = f"http://{host}:{port}/health"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw else {}
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


def pick_channel(args, health: dict) -> str:
    """默认 /tts。health.tts=false 也不改走 chat（服务端仍会推协议事件）。"""
    if args.channel == "chat":
        return "chat"
    return "tts"


def _result(persona_id: str, question: str, ok: bool, elapsed: float,
            spoken: str = "", llm_text: str = "", err: str = "",
            extra: Optional[dict] = None) -> dict:
    row = {
        "persona_id": persona_id,
        "question": question,
        "ok": ok,
        "elapsed": round(elapsed, 2),
        "llm_answer": llm_text,
        "spoken": spoken or llm_text,
        "error": err,
    }
    if extra:
        row.update(extra)
    tag = "OK" if ok else "FAIL"
    print(f"  [{tag}] {elapsed:.1f}s  回复: {_short(row['spoken'] or err or '(空)', 100)}")
    return row


async def ask_one_tts(ws, persona_id: str, question: str, timeout: float) -> dict:
    t0 = time.time()
    await ws.send(json.dumps(
        {"type": "tts", "text": question, "enable_llm": True, "speaker": "中文男"},
        ensure_ascii=False))
    print(f"  >> 已发送 type=tts  {question}")

    sentences: List[str] = []
    llm_text = ""
    chunk_n = 0
    end_msg: Optional[dict] = None
    err = ""
    started = False

    while True:
        left = timeout - (time.time() - t0)
        if left <= 0:
            err = f"超时 {timeout:.0f}s，未等到 tts_end"
            break
        try:
            msg = await _recv(ws, left)
        except asyncio.TimeoutError:
            err = f"超时 {timeout:.0f}s，未等到 tts_end"
            break

        kind = msg.get("type") or ""
        if kind == "tts_chunk":
            chunk_n += 1
            continue
        if kind in ("ready", "pong"):
            continue
        if kind == "tts_start":
            started = True
            print(f"     tts_start  hint={_short(msg.get('text'), 40)}")
            continue
        if kind == "tts_sentence":
            text = str(msg.get("text") or "").strip()
            if text:
                sentences.append(text)
            print(f"     tts_sentence #{msg.get('index')} {_short(text, 48)}")
            continue
        if kind == "llm_answer":
            llm_text = str(msg.get("text") or "").strip()
            print(f"     llm_answer  {_short(llm_text, 80)}")
            continue
        if kind == "error":
            err = str(msg.get("message") or msg.get("content") or msg)
            print(f"     error  {err}")
            break
        if kind == "tts_end":
            end_msg = msg
            extra = ""
            if msg.get("skip_reason"):
                extra = f" skip_reason={msg.get('skip_reason')}"
            if msg.get("audio") is False:
                extra += " audio=False"
            print(
                f"     tts_end  seq_count={msg.get('seq_count')} "
                f"sentence_count={msg.get('sentence_count')} "
                f"cancelled={msg.get('cancelled')}{extra}"
            )
            break
        print(f"     {kind}  {_short(msg, 60)}")

    print(f"     tts_chunk  共 {chunk_n} 包（音频数据已省略）")
    spoken = "".join(sentences).strip() or llm_text
    ok = bool(end_msg) and not err and not bool((end_msg or {}).get("cancelled"))
    return _result(
        persona_id, question, ok, time.time() - t0,
        spoken=spoken, llm_text=llm_text, err=err,
        extra={
            "channel": "tts",
            "started": started,
            "chunks": chunk_n,
            "tts_end": {
                "sentence_count": (end_msg or {}).get("sentence_count"),
                "seq_count": (end_msg or {}).get("seq_count"),
                "cancelled": (end_msg or {}).get("cancelled"),
            } if end_msg else None,
        },
    )


async def ask_one_chat(ws, persona_id: str, question: str, timeout: float) -> dict:
    t0 = time.time()
    await ws.send(json.dumps({
        "question": question,
        "enable_audio": False,
        "top_k": 2,
        "speaker": "中文男",
    }, ensure_ascii=False))
    print(f"  >> 已发送 question  {question}")

    parts: List[str] = []
    buf: List[str] = []
    done: Optional[dict] = None
    err = ""

    while True:
        left = timeout - (time.time() - t0)
        if left <= 0:
            err = f"超时 {timeout:.0f}s，未等到 done"
            break
        try:
            msg = await _recv(ws, left)
        except asyncio.TimeoutError:
            err = f"超时 {timeout:.0f}s，未等到 done"
            break

        kind = msg.get("type") or ""
        if kind == "text":
            piece = str(msg.get("content") or "")
            if piece:
                parts.append(piece)
                buf.append(piece)
                joined = "".join(buf)
                while True:
                    cuts = [p for p in (joined.find(ch) for ch in "。！？") if p != -1]
                    if not cuts:
                        break
                    cut = min(cuts)
                    line = joined[:cut + 1].strip()
                    joined = joined[cut + 1:]
                    buf = [joined] if joined else []
                    if line:
                        print(f"     {line}")
            continue
        if kind == "audio":
            continue
        if kind == "error":
            err = str(msg.get("content") or msg.get("message") or msg)
            print(f"     error  {err}")
            break
        if kind == "done":
            rest = "".join(buf).strip()
            if rest:
                print(f"     {rest}")
            done = msg
            print(f"     done  refused={msg.get('refused')} intent={msg.get('intent')}")
            break
        if kind in ("init", "recommendation"):
            print(f"     {kind}  {_short(msg, 60)}")
            continue
        print(f"     {kind}  {_short(msg, 60)}")

    spoken = "".join(parts).strip()
    ok = bool(done) and not err and not bool((done or {}).get("refused"))
    if done and done.get("refused") and spoken:
        ok = True
    return _result(
        persona_id, question, ok, time.time() - t0,
        spoken=spoken, llm_text=spoken, err=err,
        extra={"channel": "chat", "done": done},
    )


async def run_persona(host: str, port: int, persona_id: str, questions: List[str],
                      timeout: float, gap: float, channel: str) -> List[dict]:
    import websockets

    url = f"ws://{host}:{port}/ws/{persona_id}/{channel}"
    name = DISPLAY.get(persona_id, persona_id)
    print(f"\n======== {name} ({persona_id}) ========")
    print(f"连接 {url}")
    results: List[dict] = []
    async with websockets.connect(
        url, open_timeout=10, close_timeout=5, max_size=16 * 1024 * 1024
    ) as ws:
        if channel == "tts":
            ready = await _recv(ws, 10)
            print(f"  ready persona={ready.get('persona_id')} name={ready.get('display_name')}")
            if ready.get("type") == "error":
                raise RuntimeError(ready.get("message") or ready)
            if ready.get("persona_id") and ready.get("persona_id") != persona_id:
                raise RuntimeError(f"连到了 {ready.get('persona_id')}，期望 {persona_id}")
            ask = ask_one_tts
        else:
            await ws.send(json.dumps({"user_id": f"ws-ask-{persona_id}"}, ensure_ascii=False))
            init = await _recv(ws, 10)
            print(f"  init persona={init.get('persona_id')} session={init.get('session_id')}")
            if init.get("type") == "error":
                raise RuntimeError(init.get("message") or init.get("content") or init)
            ask = ask_one_chat

        for i, q in enumerate(questions, 1):
            print(f"\n[{name} {i}/{len(questions)}] {q}")
            results.append(await ask(ws, persona_id, q, timeout))
            if i < len(questions) and gap > 0:
                print(f"  … 间隔 {gap:.0f}s 再问下一题")
                await asyncio.sleep(gap)
    return results


async def main_async(args) -> int:
    health = fetch_health(args.host, args.port)
    channel = pick_channel(args, health)
    print("======== 服务状态 ========")
    print(f"  {args.host}:{args.port}  status={health.get('status')}  "
          f"tts={health.get('tts')}  llm={health.get('llm')}")
    if health.get("tts_error"):
        print(f"  tts_error: {health.get('tts_error')}")
    if channel == "tts" and not health.get("tts"):
        print("  CosyVoice 未加载，仍走 /ws/{persona}/tts；"
              "会有 tts_start/llm_answer/tts_sentence/tts_end，可能没有音频包")
    elif channel == "chat":
        print("  本轮按 --channel chat 走文本通道")
    print(f"  本轮通道: {channel}")

    if health.get("status") == "unreachable":
        print(f"无法连接健康检查: {health.get('error')}")
        return 2
    if health.get("llm") is False:
        print("LLM 不可用，问答会失败。请先确认 8080 llama.cpp。")

    personas = [args.persona] if args.persona else ["monet", "fuhao", "kumarajiva"]
    all_rows: List[dict] = []
    for i, pid in enumerate(personas):
        qs = QUESTIONS[pid]
        rows = await run_persona(
            args.host, args.port, pid, qs, args.timeout, args.gap, channel)
        all_rows.extend(rows)
        if i < len(personas) - 1 and args.persona_gap > 0:
            print(f"\n…… 换下一位数字人，间隔 {args.persona_gap:.0f}s ……")
            await asyncio.sleep(args.persona_gap)

    print("\n======== 汇总 ========")
    fail = 0
    for row in all_rows:
        name = DISPLAY.get(row["persona_id"], row["persona_id"])
        tag = "OK" if row["ok"] else "FAIL"
        if not row["ok"]:
            fail += 1
        print(
            f"[{tag}] {name} | {row['question']} | {row['elapsed']}s | "
            f"{_short(row['spoken'] or row['error'] or '(空)', 90)}"
        )
    print(f"\n合计 {len(all_rows)} 题，失败 {fail}")
    return 1 if fail else 0


def parse_args():
    p = argparse.ArgumentParser(description="WS 向莫奈/妇好/鸠摩罗什各问几题")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--persona", choices=["monet", "fuhao", "kumarajiva"], default=None,
                   help="只测一个人设；默认三人各问一轮")
    p.add_argument("--channel", choices=["tts", "chat", "auto"], default="tts",
                   help="默认 /ws/{persona}/tts；chat 为可选文本通道（auto 等同 tts）")
    p.add_argument("--timeout", type=float, default=90.0, help="单题等待结束的秒数")
    p.add_argument("--gap", type=float, default=4.0, help="同一人设相邻两题之间的间隔秒")
    p.add_argument("--persona-gap", type=float, default=6.0, help="换人设前的间隔秒")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import websockets  # noqa: F401
    except ImportError:
        print("缺少 websockets：pip install websockets", file=sys.stderr)
        sys.exit(2)
    try:
        code = asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n已中断")
        code = 130
    sys.exit(code)


if __name__ == "__main__":
    main()
