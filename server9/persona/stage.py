# -*- coding: utf-8 -*-
"""展厅舞台：同一时刻只有一个上台数字人。"""
from __future__ import annotations
import asyncio
from typing import Optional

from persona.models import PersonaConfig
from persona.registry import registry


class Stage:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._current: Optional[PersonaConfig] = None

    def current(self) -> PersonaConfig:
        if self._current is None:
            self._current = registry.require(registry.default_id)
        return self._current

    def current_id(self) -> str:
        return self.current().id

    def bootstrap(self) -> PersonaConfig:
        """进程启动：默认莫奈。"""
        if not registry._items:
            registry.load()
        self._current = registry.require(registry.default_id)
        print(f"🎭 舞台默认数字人: {self._current.display_name} ({self._current.id})")
        return self._current

    async def switch(self, persona_id: str, play_hello: bool = False, reason: str = "api") -> dict:
        pid = (persona_id or "").strip()
        target = registry.get(pid)
        if not target or not target.enabled:
            known = "、".join(p.id for p in registry.enabled()) or registry.default_id
            return {
                "ok": False,
                "code": "unknown_persona",
                "message": f"未知或未启用人设: {pid}（可用: {known}）",
            }
        async with self._lock:
            prev = self.current()
            if prev.id == target.id:
                return {
                    "ok": True, "code": "unchanged", "persona_id": target.id,
                    "display_name": target.display_name, "play_hello": False,
                }
            from ws.hub import tts_fanout
            from robot.hub import robot_hub
            from rag import index as rag
            from tts import cosyvoice as tts
            from session import active_sessions, persistence

            n = tts_fanout.cancel_public(prev.id)
            if robot_hub.session_id:
                await robot_hub.release(f"persona_switch:{reason}")
            expired = []
            for sid, sess in list(active_sessions.items()):
                if getattr(sess, "persona_id", "") != target.id:
                    persistence.save_session(sess)
                    del active_sessions[sid]
                    expired.append(sid)
            rag.bind_persona(target)
            tts.set_tts_persona(target)
            self._current = target
            payload = {
                "type": "persona_changed",
                "persona_id": target.id,
                "display_name": target.display_name,
                "from_persona_id": prev.id,
                "reason": reason,
            }
            await tts_fanout.broadcast(payload)
            print(f"🎭 切换数字人 {prev.id} → {target.id} cancel={n} drop_sessions={len(expired)}")
            return {
                "ok": True,
                "code": "switched",
                "persona_id": target.id,
                "display_name": target.display_name,
                "from_persona_id": prev.id,
                "play_hello": bool(play_hello),
                "knowledge_ready": target.knowledge_ready(),
                "voice_ready": target.voice_ready(),
            }


stage = Stage()
