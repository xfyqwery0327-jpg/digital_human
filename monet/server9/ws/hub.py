# -*- coding: utf-8 -*-
"""/ws/tts 连接汇聚：游客会话与机器人推送共用同一条前端协议。"""
from __future__ import annotations
import asyncio
import threading
from typing import Dict, List, Optional, Tuple

from fastapi import WebSocket


def attach_robot_id(obj: dict, robot_id: str = "") -> dict:
    """每条 WS 都带 robot_id。传入非空则覆盖；否则保留已有值。"""
    out = dict(obj)
    rid = (robot_id or "").strip()
    if not rid:
        existing = out.get("robot_id")
        rid = existing.strip() if isinstance(existing, str) else (str(existing).strip() if existing else "")
    out["robot_id"] = rid
    return out


class TtsFanout:
    """登记所有 /ws/tts 客户端，供机器人 HTTP 一轮把字幕+PCM 推到 Unity/前端。"""

    def __init__(self):
        self._guard = asyncio.Lock()
        self._clients: Dict[int, Tuple[WebSocket, asyncio.Lock]] = {}
        self._cancel_bags: List[Dict[str, threading.Event]] = []

    def client_count(self) -> int:
        return len(self._clients)

    async def register(self, ws: WebSocket, send_lock: asyncio.Lock,
                       cancel_bag: Optional[Dict[str, threading.Event]] = None):
        async with self._guard:
            self._clients[id(ws)] = (ws, send_lock)
            if cancel_bag is not None and cancel_bag not in self._cancel_bags:
                self._cancel_bags.append(cancel_bag)

    async def unregister(self, ws: WebSocket,
                         cancel_bag: Optional[Dict[str, threading.Event]] = None):
        async with self._guard:
            self._clients.pop(id(ws), None)
            if cancel_bag is not None:
                try:
                    self._cancel_bags.remove(cancel_bag)
                except ValueError:
                    pass

    def cancel_public(self) -> int:
        """机器人开口前打断游客 TTS，避免两条流交错。"""
        n = 0
        for bag in list(self._cancel_bags):
            for ev in list(bag.values()):
                try:
                    ev.set()
                    n += 1
                except Exception:
                    pass
        return n

    async def broadcast(self, obj: dict) -> int:
        async with self._guard:
            items = list(self._clients.items())
        ok = 0
        dead = []
        kind = obj.get("type")
        for key, (ws, lock) in items:
            try:
                async with lock:
                    await asyncio.wait_for(ws.send_json(obj), timeout=8.0)
                ok += 1
            except Exception as e:
                print(f"[ROBOT] WS 推送失败 type={kind}: {e}")
                dead.append(key)
        if dead:
            async with self._guard:
                for k in dead:
                    self._clients.pop(k, None)
            print(f"[ROBOT] 已摘除 {len(dead)} 个失败的 /ws/tts 连接，剩余 {len(self._clients)}")
        return ok


tts_fanout = TtsFanout()
