# -*- coding: utf-8 -*-
"""WS 客户端视角的卡顿检测：模拟前端实时播放，检测 PCM 缓冲何时穿底。

服务端的 play_clock 假设「前端零延迟起播、严格实时消耗」，
但真实前端有起播缓冲/调度抖动。这里在客户端复刻播放模型：

    缓冲余额 += 每个 chunk 的音频时长   （收到时）
    缓冲余额 -= 流逝墙钟时间           （持续推进）

余额 < 0 即一次 underrun（卡顿）。同时记录每个 chunk 的到达间隔，
并把 underrun 映射到当时的字幕句 index，直接回答"第几句卡"。

用法：
    python tests/ws_stall_probe.py --persona-id monet --mode 3
    python tests/ws_stall_probe.py --persona-id monet --mode 1 --startup 1.0
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
import urllib.request
from typing import Dict, List, Optional

SR_DEFAULT = 24000


def http_post_json(url: str, body: dict, timeout: float = 15.0):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json; charset=utf-8"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


class StallProbe:
    def __init__(self, startup: float):
        self.startup = float(startup)
        self.sr = SR_DEFAULT
        self.chunks: List[dict] = []      # {t, seq, sec, bytes}
        self.sentences: List[dict] = []   # {t, index, text}
        self.t_start: Optional[float] = None
        self.t_end: Optional[float] = None
        self.t0: Optional[float] = None
        self.llm_at: Optional[float] = None
        self.engine = ""
        self.tts_error = ""
        self.total_bytes = 0

    def _rel(self, t: float) -> float:
        return round(t - self.t0, 3) if self.t0 else 0.0

    def on_msg(self, obj: dict):
        kind = obj.get("type")
        if kind == "tts_start":
            self.t_start = self._rel(time.time())
            sr = obj.get("sample_rate")
            if sr:
                self.sr = int(sr)
        elif kind == "llm_answer":
            self.llm_at = self._rel(time.time())
        elif kind == "tts_sentence":
            self.sentences.append({
                "t": self._rel(time.time()),
                "index": obj.get("index"),
                "text": str(obj.get("text") or ""),
            })
        elif kind == "tts_chunk":
            data = obj.get("data") or ""
            if not data:
                return
            raw = base64.b64decode(data)
            self.chunks.append({
                "t": self._rel(time.time()),
                "seq": obj.get("seq"),
                "sec": len(raw) / 2.0 / float(self.sr),
                "bytes": len(raw),
            })
            self.total_bytes += len(raw)
        elif kind == "tts_end":
            self.t_end = self._rel(time.time())
            self.engine = obj.get("engine") or ""
            self.tts_error = obj.get("tts_error") or ""

    # ---------- 分析 ----------

    def sentence_at(self, t: float) -> str:
        """t 时刻正在播/刚落到的字幕句。"""
        cur = "(未知)"
        for s in self.sentences:
            if s["t"] <= t:
                cur = f"#{s['index']} {s['text'][:18]}"
            else:
                break
        return cur

    def analyze(self) -> dict:
        if not self.chunks:
            return {"error": "没有收到任何 tts_chunk"}
        pcm_sec = sum(c["sec"] for c in self.chunks)
        first = self.chunks[0]["t"]
        play_from = first + self.startup          # 前端起播时刻

        # 模拟实时消耗
        bal = 0.0
        clock = play_from
        underruns: List[dict] = []
        worst = 0.0
        for c in self.chunks:
            if c["t"] > clock:                     # 期间在被消耗
                bal -= (c["t"] - clock)
                clock = c["t"]
            if bal < worst:
                worst = bal
            if bal < 0:
                # 卡顿深度 = 缺口；从穿底到下一个 chunk 到达为止
                underruns.append({
                    "t": round(c["t"], 3),
                    "deficit": round(-bal, 3),
                    "seq": c["seq"],
                    "sentence": self.sentence_at(c["t"]),
                    "gap_sec": None,   # 稍后填
                })
            bal += c["sec"]
            clock = max(clock, c["t"])
        # 收尾：把 deficit>0 的项配上"到下个 chunk 的间隔"
        for i, u in enumerate(underruns):
            nxt = underruns[i + 1]["t"] if i + 1 < len(underruns) else None
            if nxt is None:
                later = [c["t"] for c in self.chunks if c["t"] > u["t"]]
                nxt = later[0] if later else None
            u["gap_sec"] = round(nxt - u["t"], 3) if nxt is not None else None

        # 逐句交付窗口 + 句间空档（直接回答"第几句卡"）
        sent_map: Dict[int, List[float]] = {}
        for c in self.chunks:
            sidx = None
            for s in self.sentences:
                if s["t"] <= c["t"]:
                    sidx = s["index"]
                else:
                    break
            if sidx is None:
                continue
            d = sent_map.setdefault(sidx, [c["t"], c["t"]])
            d[0] = min(d[0], c["t"])
            d[1] = max(d[1], c["t"])
        sent_gaps = []
        idxs = sorted(sent_map.keys())
        for a, b in zip(idxs, idxs[1:]):
            gap = round(sent_map[b][0] - sent_map[a][1], 3)
            sent_gaps.append({"from": a, "to": b, "gap": gap})

        # chunk 到达间隔（>0.35s 视为可疑空档）
        gaps = []
        for a, b in zip(self.chunks, self.chunks[1:]):
            dt = b["t"] - a["t"]
            if dt > 0.35:
                gaps.append({
                    "from_seq": a["seq"], "to_seq": b["seq"],
                    "dt": round(dt, 3),
                    "at_t": round(b["t"], 3),
                    "sentence": self.sentence_at(b["t"]),
                })

        return {
            "chunks": len(self.chunks),
            "pcm_sec": round(pcm_sec, 2),
            "t_start": self.t_start,
            "t_llm": self.llm_at,
            "t_end": self.t_end,
            "engine": self.engine,
            "tts_error": self.tts_error,
            "sentences": len(self.sentences),
            "first_chunk_at": round(first, 3),
            "play_from": round(play_from, 3),
            "underrun_count": len(underruns),
            "worst_deficit_sec": round(-worst, 3),
            "underruns": underruns[:40],
            "big_gaps": gaps[:40],
            "sent_gaps": sent_gaps,
            "sent_map": {k: [round(v[0], 3), round(v[1], 3)] for k, v in sent_map.items()},
            "avg_chunk_sec": round(pcm_sec / len(self.chunks), 4),
        }


async def run(args):
    import websockets
    ws_url = f"ws://{args.host}:{args.port}/ws/{args.persona_id}/tts"
    http_url = f"http://{args.host}:{args.port}/api/{args.persona_id}/robot/ask"
    probe = StallProbe(args.startup)

    async with websockets.connect(ws_url, max_size=8 * 1024 * 1024) as ws:
        await asyncio.wait_for(ws.recv(), timeout=5)
        probe.t0 = time.time()
        body = {"query": args.query, "robot_id": args.robot_id,
                "session_id": args.session_id, "include_audio": False,
                "wait": False, "speaker": "中文男", "top_k": 2, "mode": args.mode}
        status, resp = await asyncio.to_thread(http_post_json, http_url, body)
        print(f"HTTP {status}  session={resp.get('session_id')}  "
              f"round={resp.get('round')}")
        if status not in (200, 202):
            print(json.dumps(resp, ensure_ascii=False))
            return 2

        deadline = time.time() + args.timeout
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                if probe.t_end is not None:
                    break
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            probe.on_msg(obj)
            if probe.t_end is not None:
                break

    r = probe.analyze()
    print(f"\n=== 卡顿探测 persona={args.persona_id} mode={args.mode} "
          f"起播延迟={args.startup}s ===")
    for k in ("chunks", "pcm_sec", "sentences", "t_start", "t_llm", "t_end",
              "engine", "tts_error", "first_chunk_at", "play_from",
              "avg_chunk_sec", "underrun_count", "worst_deficit_sec"):
        print(f"  {k:18} {r.get(k)}")
    if r.get("underruns"):
        print(f"\n  underrun（卡顿）{r['underrun_count']} 次，最严重缺口 "
              f"{r['worst_deficit_sec']}s：")
        for u in r["underruns"]:
            print(f"    t={u['t']:>7}s seq={u['seq']:<5} 缺口={u['deficit']:>6}s "
                  f"到下个chunk={u['gap_sec']}s  当时字幕: {u['sentence']}")
    else:
        print("\n  ✅ 无 underrun：客户端缓冲从未穿底")
    if r.get("big_gaps"):
        print(f"\n  chunk 到达空档 >0.35s 共 {len(r['big_gaps'])} 处：")
        for g in r["big_gaps"]:
            print(f"    t={g['at_t']:>7}s seq {g['from_seq']}->{g['to_seq']} "
                  f"间隔={g['dt']}s  当时字幕: {g['sentence']}")
    else:
        print("  ✅ 无 >0.35s 的 chunk 到达空档")

    print("\n  逐句交付窗口（首chunk~末chunk）与句间空档：")
    sm = r.get("sent_map", {})
    sg = {(g["from"], g["to"]): g["gap"] for g in r.get("sent_gaps", [])}
    for idx in sorted(sm.keys()):
        first, last = sm[idx]
        print(f"    #{idx:<2} 交付 [{first:>7}s ~ {last:>7}s]  时长≈{round(last-first,2)}s"
              + (f"  →下句空档 {sg.get((idx, idx+1),'-')}s" if (idx, idx+1) in sg else ""))
    # 句间空档汇总
    stall_gaps = [(f, t, g) for (f, t), g in sg.items() if g > 0.3]
    if stall_gaps:
        print(f"\n  ⚠️ 句间空档 >0.3s 共 {len(stall_gaps)} 处（卡顿嫌疑）：")
        for f, t, g in stall_gaps:
            print(f"    #{f} 末 → #{t} 首  间隔 {g}s")
    else:
        print("  ✅ 句间无 >0.3s 空档")

    print("\n  字幕时间线：")
    for s in probe.sentences:
        print(f"    t={s['t']:>7}s  #{s['index']:<2} {s['text'][:40]}")
    return 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8848)
    p.add_argument("--persona-id", default="monet", choices=["monet", "fuhao", "kumarajiva"])
    p.add_argument("--mode", type=int, default=3, choices=[1, 2, 3])
    p.add_argument("--query", default="莫奈介绍下自己吧")
    p.add_argument("--robot-id", default="robot-001")
    p.add_argument("--session-id", default=None)
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--startup", type=float, default=0.0,
                   help="模拟前端起播缓冲秒数（0=收到即播）")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(run(parse_args())))
