# -*- coding: utf-8 -*-
"""联调 /api/robot/ask：先连 /ws/tts 再发 HTTP，只收流、不回 echo。

现场（讲解员已启动 python server8.py）:

    python scripts/test_robot_ask.py
    python scripts/test_robot_ask.py --host 127.0.0.1 --port 8848
    python scripts/test_robot_ask.py --query 莫奈介绍下自己吧
    python scripts/test_robot_ask.py --rounds 3
    python scripts/test_robot_ask.py --echo-tts   # 复现前端把 llm_answer 再发 type=tts 的双流问题

默认三轮和现场脚本一致。本脚本只 listen，不会把 asr/讲稿再丢回 type=tts。
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

DEFAULT_QUERIES = [
    "莫奈介绍下自己吧",
    "好的，你一生都有哪些作品啊",
    "那你能不能给我们介绍下睡莲这个作品呢",
]


def _short(text: Any, n: int = 48) -> str:
    s = "" if text is None else str(text).replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def http_post_json(url: str, body: dict, token: str = "", timeout: float = 15.0) -> tuple[int, dict]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if token:
        headers["X-Robot-Token"] = token
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                payload = {"raw": raw}
            return resp.status, payload
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"raw": raw}
        return e.code, payload


class TurnCollector:
    def __init__(self, query: str):
        self.query = query
        self.t0 = time.time()
        self.events: List[str] = []
        self.times: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self.robot_ids: List[str] = []
        self.session_ids: List[str] = []
        self.llm_text = ""
        self.asr_text = ""
        self.seq_max = -1
        self.chunk_bytes = 0
        self.ends: List[dict] = []
        self.errors: List[str] = []
        self.done = asyncio.Event()

    def _mark(self, kind: str):
        self.counts[kind] = self.counts.get(kind, 0) + 1
        if kind not in self.times:
            self.times[kind] = time.time() - self.t0
        if kind not in self.events:
            self.events.append(kind)

    def on_msg(self, obj: dict):
        kind = obj.get("type") or "?"
        if kind == "ready":
            return
        if kind == "pong":
            return
        self._mark(kind)
        rid = obj.get("robot_id")
        if isinstance(rid, str) and rid and rid not in self.robot_ids:
            self.robot_ids.append(rid)
        sid = obj.get("session_id")
        if isinstance(sid, str) and sid and sid not in self.session_ids:
            self.session_ids.append(sid)
        if kind == "asr":
            self.asr_text = str(obj.get("text") or "")
        elif kind == "llm_answer":
            self.llm_text = str(obj.get("text") or "")
        elif kind == "tts_chunk":
            data = obj.get("data") or ""
            self.chunk_bytes += int(len(data) * 3 / 4) if data else 0
            try:
                self.seq_max = max(self.seq_max, int(obj.get("seq", -1)))
            except (TypeError, ValueError):
                pass
        elif kind == "tts_end":
            self.ends.append(obj)
            self.done.set()
        elif kind == "error":
            self.errors.append(str(obj.get("message") or obj))

    def summary(self) -> Dict[str, Any]:
        order = " → ".join(self.events) if self.events else "(无 WS 消息)"
        end_n = self.counts.get("tts_end", 0)
        start_n = self.counts.get("tts_start", 0)
        llm_at = self.times.get("llm_answer")
        start_at = self.times.get("tts_start")
        end_at = self.times.get("tts_end")
        pcm_sec = self.chunk_bytes / 2.0 / 24000.0 if self.chunk_bytes else 0.0
        ok_end = end_n == 1
        ok_start = start_n == 1
        ok_llm = "llm_answer" in self.times and (
            start_at is None or llm_at is None or llm_at <= (end_at or 999) and
            (end_at is None or llm_at < end_at)
        )
        # llm 应紧跟 start，不能拖到 end 附近
        llm_early = bool(llm_at is not None and start_at is not None and (llm_at - start_at) < 2.0)
        return {
            "query": self.query,
            "order": order,
            "tts_start": start_n,
            "llm_answer": self.counts.get("llm_answer", 0),
            "tts_sentence": self.counts.get("tts_sentence", 0),
            "tts_chunk": self.counts.get("tts_chunk", 0),
            "tts_end": end_n,
            "asr": self.asr_text,
            "llm_preview": _short(self.llm_text, 80),
            "robot_id": ",".join(self.robot_ids) or "(空)",
            "session_id": ",".join(self.session_ids) or "(空)",
            "t_start": None if start_at is None else round(start_at, 2),
            "t_llm": None if llm_at is None else round(llm_at, 2),
            "t_end": None if end_at is None else round(end_at, 2),
            "pcm_sec": round(pcm_sec, 2),
            "seq_max": self.seq_max,
            "errors": self.errors,
            "ok_one_start": ok_start,
            "ok_one_end": ok_end,
            "ok_llm_before_end": ok_llm and llm_early,
        }


async def listen_ws(ws, collector: TurnCollector, stop: asyncio.Event):
    try:
        while not stop.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                collector.errors.append(f"非 JSON: {raw[:80]}")
                continue
            if not isinstance(obj, dict):
                continue
            kind = obj.get("type")
            extra = ""
            if kind == "asr":
                extra = f" text={_short(obj.get('text'))}"
            elif kind == "tts_start":
                extra = f" hint={_short(obj.get('text'))} robot_id={obj.get('robot_id')!r}"
            elif kind == "llm_answer":
                extra = f" chars={len(str(obj.get('text') or ''))} {_short(obj.get('text'))}"
            elif kind == "tts_sentence":
                extra = f" #{obj.get('index')} {_short(obj.get('text'), 36)}"
            elif kind == "tts_chunk":
                seq = obj.get("seq")
                extra = f" seq={seq}"
                collector.on_msg(obj)
                if seq in (0, 1) or (isinstance(seq, int) and seq % 50 == 0):
                    print(f"  WS {kind}{extra}")
                continue
            elif kind == "tts_end":
                extra = (f" seq_count={obj.get('seq_count')} "
                         f"sentence_count={obj.get('sentence_count')} "
                         f"cancelled={obj.get('cancelled')}")
            print(f"  WS {kind}{extra}")
            collector.on_msg(obj)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        collector.errors.append(f"WS 收包失败: {e}")
        collector.done.set()


async def echo_tts_if_needed(ws, collector: TurnCollector, enable: bool):
    """复现现场：收到 llm_answer 后再发 type=tts，会和 robot 流叠在一起。"""
    if not enable:
        return
    while not collector.done.is_set():
        if collector.llm_text:
            payload = json.dumps({"type": "tts", "text": collector.llm_text}, ensure_ascii=False)
            await ws.send(payload)
            print(f"  >> 已按 --echo-tts 把 llm_answer 发回 type=tts")
            return
        await asyncio.sleep(0.05)


async def run_one(ws, http_url: str, robot_id: str, session_id: Optional[str],
                  query: str, token: str, timeout: float, echo_tts: bool) -> tuple[Optional[str], Dict[str, Any]]:
    collector = TurnCollector(query)
    stop = asyncio.Event()
    listener = asyncio.create_task(listen_ws(ws, collector, stop))
    echo_task = asyncio.create_task(echo_tts_if_needed(ws, collector, echo_tts))

    body = {
        "query": query,
        "robot_id": robot_id,
        "session_id": session_id,
        "include_audio": False,
        "wait": False,
        "speaker": "中文男",
        "top_k": 2,
    }
    print(f"\nHTTP POST {http_url}")
    print(f"  query={query!r} session={session_id!r} robot={robot_id}")
    t_http = time.time()
    status, resp = await asyncio.to_thread(http_post_json, http_url, body, token)
    print(f"  <-- HTTP {status}  {time.time() - t_http:.3f}s")
    print(f"  {json.dumps(resp, ensure_ascii=False)}")

    sid = resp.get("session_id") if isinstance(resp, dict) else session_id
    if status not in (200, 202):
        stop.set()
        echo_task.cancel()
        listener.cancel()
        summary = collector.summary()
        summary["http_status"] = status
        summary["http_body"] = resp
        summary["pass"] = False
        return sid, summary

    try:
        await asyncio.wait_for(collector.done.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        collector.errors.append(f"等待 tts_end 超时 {timeout:.0f}s")

    stop.set()
    echo_task.cancel()
    await asyncio.sleep(0.05)
    listener.cancel()
    try:
        await listener
    except asyncio.CancelledError:
        pass

    summary = collector.summary()
    summary["http_status"] = status
    summary["http_session_id"] = sid
    summary["pass"] = bool(
        status in (200, 202)
        and summary["ok_one_start"]
        and summary["ok_one_end"]
        and summary["ok_llm_before_end"]
        and summary["tts_chunk"] > 0
        and not collector.errors
    )
    return sid, summary


def print_verdict(i: int, s: Dict[str, Any]):
    flag = "PASS" if s.get("pass") else "FAIL"
    print(f"\n--- 第 {i} 轮 {flag} ---")
    print(f"  顺序     {s.get('order')}")
    print(f"  start/end {s.get('tts_start')}/{s.get('tts_end')}  "
          f"(必须都是 1)")
    print(f"  llm_answer {s.get('llm_answer')}  t={s.get('t_llm')}s  "
          f"{s.get('llm_preview')}")
    print(f"  chunk={s.get('tts_chunk')} seq_max={s.get('seq_max')}  "
          f"pcm≈{s.get('pcm_sec')}s  t_end={s.get('t_end')}s")
    print(f"  robot_id={s.get('robot_id')}  session={s.get('session_id')}")
    if s.get("errors"):
        for err in s["errors"]:
            print(f"  ERR {err}")


def open_ws(ws_url: str, token: str):
    import websockets
    headers = {"X-Robot-Token": token} if token else None
    kwargs = {"max_size": 8 * 1024 * 1024}
    try:
        return websockets.connect(ws_url, additional_headers=headers, **kwargs)
    except TypeError:
        return websockets.connect(ws_url, extra_headers=headers, **kwargs)


async def main_async(args):
    try:
        import websockets  # noqa: F401
    except ImportError:
        print("需要 websockets：pip install websockets")
        sys.exit(2)

    http_url = f"http://{args.host}:{args.port}/api/robot/ask"
    ws_url = f"ws://{args.host}:{args.port}/ws/tts"
    queries = args.queries
    if args.query:
        queries = [args.query]
    if args.rounds > 0:
        queries = (queries * ((args.rounds + len(queries) - 1) // len(queries)))[: args.rounds]

    print("=" * 64)
    print("莫奈 /api/robot/ask 联调")
    print(f"  HTTP  {http_url}")
    print(f"  WS    {ws_url}")
    print(f"  robot {args.robot_id}")
    print(f"  轮次  {len(queries)}")
    print("  本脚本只收流，不回 type=tts" + ("（已开 --echo-tts 复现双流）" if args.echo_tts else ""))
    print("=" * 64)

    async with open_ws(ws_url, args.token) as ws:
        ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        print(f"WS ready: {ready}")
        session_id = args.session_id
        results = []
        for i, query in enumerate(queries, 1):
            sid, summary = await run_one(
                ws, http_url, args.robot_id, session_id, query,
                args.token, args.timeout, args.echo_tts)
            if sid:
                session_id = sid
            results.append(summary)
            print_verdict(i, summary)
            if i < len(queries) and not args.echo_tts:
                await asyncio.sleep(args.gap)

    passed = sum(1 for r in results if r.get("pass"))
    print("\n" + "=" * 64)
    print(f"结果 {passed}/{len(results)} 轮通过")
    print("=" * 64)
    return 0 if passed == len(results) else 1


def parse_args():
    p = argparse.ArgumentParser(description="测试 POST /api/robot/ask 并核对 /ws/tts 推流")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8848)
    p.add_argument("--robot-id", default="robot-001")
    p.add_argument("--session-id", default=None)
    p.add_argument("--token", default="", help="X-Robot-Token，对应 ROBOT_API_TOKEN")
    p.add_argument("--query", default="", help="只测这一句；不填则用默认三轮")
    p.add_argument("--rounds", type=int, default=0, help="轮数，0 表示默认三句各一轮")
    p.add_argument("--timeout", type=float, default=90.0, help="每轮等 tts_end 的秒数")
    p.add_argument("--gap", type=float, default=1.0, help="轮与轮之间间隔秒")
    p.add_argument("--echo-tts", action="store_true",
                   help="收到 llm_answer 后再发 type=tts，用来复现双流")
    p.add_argument("queries", nargs="*", default=DEFAULT_QUERIES)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        code = asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n已中断")
        code = 130
    sys.exit(code)
