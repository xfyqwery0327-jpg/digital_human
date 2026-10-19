# -*- coding: utf-8 -*-
"""/ws/tts 连接汇聚：按 persona_id 分发，机器人只推给对应数字人的前端。

参考 robot-ask-ws-改动说明.md：单条发送超时 8s，失败摘掉死连接。
参考 项目整体脉络.md：禁止跨人设全局广播。
"""
from __future__ import annotations
import asyncio
import threading
from typing import Dict, List, Optional, Tuple

from fastapi import WebSocket

from ws.session_wire import TTS_SESSION_WIRE, tts_wire_allowed  # noqa: F401

_LEGACY_PERSONA = "monet"


async def accept_websocket(websocket: WebSocket) -> None:
    """Accept including Unity Sec-WebSocket-Protocol, otherwise handshake can 403."""
    proto = websocket.headers.get("sec-websocket-protocol")
    if proto:
        await websocket.accept(subprotocol=proto.split(",")[0].strip())
    else:
        await websocket.accept()


def attach_robot_id(obj: dict, robot_id: str = "", persona_id: str = "") -> dict:
    """每条 WS 都带 robot_id / persona_id / display_name。旧客户端可忽略新字段。"""
    out = dict(obj)
    rid = (robot_id or "").strip()
    if not rid:
        existing = out.get("robot_id")
        rid = existing.strip() if isinstance(existing, str) else (str(existing).strip() if existing else "")
    out["robot_id"] = rid
    pid = (persona_id or out.get("persona_id") or "").strip()
    display = out.get("display_name") or ""
    try:
        from persona.runtime import active
        p = active()
        pid = pid or p.id
        display = display or p.display_name
    except Exception:
        pass
    if not pid:
        pid = _LEGACY_PERSONA
    if not display:
        try:
            from persona.registry import registry
            cfg = registry.get(pid)
            if cfg:
                display = cfg.display_name
        except Exception:
            display = pid
    out["persona_id"] = pid
    out["display_name"] = display
    return out


class TtsFanout:
    """登记 /ws/{persona_id}/tts 客户端。旧 /ws/tts 登记为 monet。"""

    def __init__(self):
        self._guard = asyncio.Lock()
        self._clients: Dict[int, Tuple[WebSocket, asyncio.Lock, str]] = {}
        self._cancel_bags: List[Tuple[str, Dict[str, threading.Event]]] = []

    def client_count(self, persona_id: str = "") -> int:
        if not persona_id:
            return len(self._clients)
        want = persona_id.strip() or _LEGACY_PERSONA
        return sum(1 for *_, pid in self._clients.values() if (pid or _LEGACY_PERSONA) == want)

    async def register(self, ws: WebSocket, send_lock: asyncio.Lock,
                       cancel_bag: Optional[Dict[str, threading.Event]] = None,
                       persona_id: str = ""):
        pid = (persona_id or "").strip() or _LEGACY_PERSONA
        async with self._guard:
            self._clients[id(ws)] = (ws, send_lock, pid)
            if cancel_bag is not None:
                self._cancel_bags.append((pid, cancel_bag))

    async def unregister(self, ws: WebSocket,
                         cancel_bag: Optional[Dict[str, threading.Event]] = None):
        async with self._guard:
            self._clients.pop(id(ws), None)
            if cancel_bag is not None:
                self._cancel_bags = [
                    item for item in self._cancel_bags if item[1] is not cancel_bag
                ]

    def cancel_public(self, persona_id: str = "") -> int:
        """打断游客 TTS。传入 persona_id 则只打断该人设连接。"""
        want = (persona_id or "").strip()
        n = 0
        for pid, bag in list(self._cancel_bags):
            if want and (pid or _LEGACY_PERSONA) != want:
                continue
            for ev in list(bag.values()):
                try:
                    ev.set()
                    n += 1
                except Exception:
                    pass
        return n

    async def send(self, obj: dict, persona_id: str) -> int:
        """仅向该 persona 的 /ws/{persona}/tts 连接推送，禁止全局广播。"""
        pid = (persona_id or "").strip() or _LEGACY_PERSONA
        return await self._fanout(obj, want=pid)

    async def broadcast(self, obj: dict, persona_id: str = "") -> int:
        """兼容旧调用。传入 persona_id 时与 send 等价；空则按 payload.persona_id 过滤。"""
        want = (persona_id or obj.get("persona_id") or "").strip()
        return await self._fanout(obj, want=want)

    async def _fanout(self, obj: dict, want: str = "") -> int:
        async with self._guard:
            items = list(self._clients.items())
        ok = 0
        dead: List[int] = []
        kind = obj.get("type")
        for key, (ws, lock, pid) in items:
            bound = pid or _LEGACY_PERSONA
            if want and bound != want:
                continue
            try:
                async with lock:
                    await asyncio.wait_for(ws.send_json(obj), timeout=8.0)
                ok += 1
            except Exception as e:
                print(f"[ROBOT] WS 推送失败 type={kind} persona={bound} want={want or '-'}: {e}")
                dead.append(key)
        if dead:
            async with self._guard:
                for k in dead:
                    self._clients.pop(k, None)
            print(f"[ROBOT] 已摘除 {len(dead)} 个失败的 /ws/tts 连接，剩余 {len(self._clients)} want={want or '-'}")
        return ok


tts_fanout = TtsFanout()
