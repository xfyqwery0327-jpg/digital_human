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


def _client_host(ws: WebSocket) -> str:
    client = getattr(ws, "client", None)
    host = getattr(client, "host", None) if client is not None else ""
    return (host or "").strip()


class TtsFanout:
    """登记 /ws/{persona_id}/tts 客户端。旧 /ws/tts 登记为 monet。

    同一台机器对同一个人设反复重连时，旧连接往往不断开，ask 会把同一段 PCM
    推到每一条上，前端就叠着播。新连接登记时关掉同主机、同人设的旧连接。
    """

    def __init__(self):
        self._guard = asyncio.Lock()
        # id(ws) -> (ws, send_lock, persona_id, client_host, cancel_bag)
        self._clients: Dict[int, Tuple] = {}
        self._cancel_bags: List[Tuple[str, Dict[str, threading.Event]]] = []

    def client_count(self, persona_id: str = "") -> int:
        if not persona_id:
            return len(self._clients)
        want = persona_id.strip() or _LEGACY_PERSONA
        return sum(
            1 for item in self._clients.values()
            if (item[2] or _LEGACY_PERSONA) == want
        )

    async def register(self, ws: WebSocket, send_lock: asyncio.Lock,
                       cancel_bag: Optional[Dict[str, threading.Event]] = None,
                       persona_id: str = ""):
        pid = (persona_id or "").strip() or _LEGACY_PERSONA
        host = _client_host(ws)
        stale = []
        async with self._guard:
            for key, item in list(self._clients.items()):
                old_ws, _lock, old_pid, old_host, old_bag = item
                if key == id(ws):
                    continue
                if (old_pid or _LEGACY_PERSONA) != pid:
                    continue
                if not host or old_host != host:
                    continue
                self._clients.pop(key, None)
                if old_bag is not None:
                    self._cancel_bags = [
                        pair for pair in self._cancel_bags if pair[1] is not old_bag
                    ]
                stale.append((old_ws, old_bag))
            self._clients[id(ws)] = (ws, send_lock, pid, host, cancel_bag)
            if cancel_bag is not None:
                self._cancel_bags.append((pid, cancel_bag))
        if not stale:
            return
        for old_ws, old_bag in stale:
            if old_bag is not None:
                for ev in list(old_bag.values()):
                    try:
                        ev.set()
                    except Exception:
                        pass
            try:
                await old_ws.close(code=1000, reason="replaced")
            except Exception as e:
                print(f"[ROBOT] 关闭旧 /ws/{pid}/tts 失败 host={host}: {e}")
        print(f"[ROBOT] 同一台机器重连 /ws/{pid}/tts，已关掉 {len(stale)} 条旧连接 host={host or '-'}")

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
        delivered = []
        for key, (ws, lock, pid, host, _bag) in items:
            bound = pid or _LEGACY_PERSONA
            if want and bound != want:
                continue
            try:
                async with lock:
                    await asyncio.wait_for(ws.send_json(obj), timeout=8.0)
                ok += 1
                if kind == "tts_start":
                    port = getattr(getattr(ws, "client", None), "port", "")
                    delivered.append(f"{host or '-'}:{port}" if port else (host or "-"))
            except Exception as e:
                print(f"[ROBOT] WS 推送失败 type={kind} persona={bound} want={want or '-'}: {e}")
                dead.append(key)
        if kind == "tts_start" and ok > 1:
            print(f"[ROBOT] tts_start 同时发给 {ok} 条 /ws/{want or '-'}/tts，会叠着播 "
                  f"clients={','.join(delivered)}")
        if dead:
            async with self._guard:
                for k in dead:
                    self._clients.pop(k, None)
            print(f"[ROBOT] 已摘除 {len(dead)} 个失败的 /ws/tts 连接，剩余 {len(self._clients)} want={want or '-'}")
        return ok


tts_fanout = TtsFanout()
