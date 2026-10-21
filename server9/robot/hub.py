# -*- coding: utf-8 -*-
"""机器人会话、轮次、空闲超时。不再按 robot/session holding 互斥拒绝。"""
from __future__ import annotations
import asyncio
import time
import uuid
from typing import Optional, Dict, Tuple

from fastapi.responses import JSONResponse

from config import *
from session import SessionState, persistence, active_sessions


class RobotHub:
    """当前讲解会话。后来的 ask 在空闲时直接续接或接管，不因 robot_id/session_id 不同而 409。"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self.session_id: Optional[str] = None
        self.robot_id: Optional[str] = None
        self.round: int = 0
        self.max_rounds: int = ROBOT_MAX_ROUNDS
        self.busy: bool = False
        self.started_at: float = 0.0
        self.last_active: float = 0.0
        self.monet_session: Optional[SessionState] = None
        self.persona_id: Optional[str] = None
        self.current_request_id: Optional[str] = None
        self.current_action: Optional[str] = None
        self.last_ask_query: Optional[str] = None

    def snapshot(self) -> Dict:
        return {
            "active": bool(self.session_id),
            "session_id": self.session_id,
            "robot_id": self.robot_id,
            "persona_id": self.persona_id,
            "round": self.round,
            "max_rounds": self.max_rounds,
            "busy": self.busy,
            "current_action": self.current_action,
            "current_request_id": self.current_request_id,
            "idle_remain_sec": (
                max(0, int(ROBOT_IDLE_TIMEOUT - (time.time() - self.last_active)))
                if self.session_id and not self.busy else None
            ),
        }

    def is_busy_for(self, persona_id: str = "") -> bool:
        """busy 拦截只作用于当前正在播报的人设分组。GPU 互斥仍靠 acquire 全局 409。
        参考 robot-ask-ws-改动说明.md + 项目整体脉络.md。"""
        if not self.busy or not self.session_id:
            return False
        pid = (persona_id or "").strip()
        if not pid:
            try:
                from persona.runtime import active
                pid = active().id
            except Exception:
                pid = ""
        playing = (self.persona_id or "").strip()
        if pid and playing and pid != playing:
            return False
        return True

    def is_robot_ask_echo(self, text: str, persona_id: Optional[str] = None) -> bool:
        """前端收到 ask 推的 asr 后，常会再发一条 type=tts。
        仅在本轮 busy、且同一 persona 会话粒度拦截；讲完后允许前端自己补一次 TTS。
        不传 persona_id 时读 ContextVar（chat 链路已 bind，无需改 chat.py）。"""
        q = (text or "").strip()
        last = (self.last_ask_query or "").strip()
        if not (self.session_id and self.busy and bool(q) and q == last):
            return False
        pid = (persona_id or "").strip()
        if not pid:
            try:
                from persona.runtime import active
                pid = active().id
            except Exception:
                pid = ""
        playing = (self.persona_id or "").strip()
        if pid and playing and pid != playing:
            return False
        return True

    def _adopt_caller(self, robot_id: Optional[str], now: float):
        if robot_id:
            self.robot_id = robot_id
        self.last_active = now

    async def acquire(self, robot_id: str, session_id: Optional[str],
                      create: bool = True,
                      action: str = "",
                      persona_id: Optional[str] = None) -> Tuple[bool, Optional[Dict], Optional[JSONResponse]]:
        from persona.runtime import active
        wanted = (persona_id or "").strip() or active().id
        async with self._lock:
            now = time.time()
            if self.busy:
                return False, None, JSONResponse(status_code=409, content={
                    "ok": False, "code": "turn_in_progress",
                    "message": f"本轮介绍尚未结束，请等待{self._display_name()}讲完后再发下一轮",
                    "session_id": self.session_id,
                    "request_id": self.current_request_id,
                    "action": self.current_action,
                    "persona_id": self.persona_id,
                })

            if self.session_id and self.persona_id and self.persona_id != wanted:
                self._release_locked(f"persona_switch:{self.persona_id}->{wanted}")

            if self.session_id:
                self._adopt_caller(robot_id, now)
                return True, {
                    "session_id": self.session_id,
                    "created": False,
                    "robot_id": self.robot_id,
                    "persona_id": self.persona_id,
                }, None

            if not create:
                return False, None, JSONResponse(status_code=409, content={
                    "ok": False, "code": "no_session",
                    "message": "没有进行中的机器人会话，请先调用 /api/robot/ask",
                })

            sid = session_id or f"robot_{int(now)}_{uuid.uuid4().hex[:8]}"
            self.session_id = sid
            self.robot_id = robot_id
            self.persona_id = wanted
            self.round = 0
            self.max_rounds = ROBOT_MAX_ROUNDS
            self.busy = False
            self.started_at = now
            self.last_active = now
            self.monet_session = SessionState(robot_id, sid, self.persona_id)
            active_sessions[sid] = self.monet_session
            print(f"[ROBOT] 开启会话 session={sid} robot={robot_id} persona={wanted}")
            return True, {"session_id": sid, "created": True, "robot_id": robot_id,
                          "persona_id": wanted}, None

    async def mark_busy(self, action: str, request_id: str, query: str = "") -> None:
        async with self._lock:
            self.busy = True
            self.current_action = action
            self.current_request_id = request_id
            self.last_active = time.time()
            if query:
                self.last_ask_query = query.strip()
            print(f"[ROBOT] mark_busy action={action} request_id={request_id} "
                  f"persona={self.persona_id} robot={self.robot_id}")

    async def mark_idle(self) -> None:
        async with self._lock:
            self.busy = False
            self.current_action = None
            self.current_request_id = None
            self.last_active = time.time()

    async def bump_round(self) -> int:
        async with self._lock:
            self.round += 1
            self.last_active = time.time()
            return self.round

    async def release(self, reason: str) -> Dict:
        async with self._lock:
            return self._release_locked(reason)

    def _release_locked(self, reason: str) -> Dict:
        info = {
            "session_id": self.session_id,
            "robot_id": self.robot_id,
            "round": self.round,
            "reason": reason,
        }
        if self.session_id:
            print(f"[ROBOT] 释放会话 session={self.session_id} "
                  f"robot={self.robot_id} persona={self.persona_id} "
                  f"reason={reason} rounds={self.round}")
            persistence.save_session(self.monet_session) if self.monet_session else None
        self.session_id = None
        self.robot_id = None
        self.persona_id = None
        self.round = 0
        self.busy = False
        self.started_at = 0.0
        self.last_active = 0.0
        self.monet_session = None
        self.current_request_id = None
        self.current_action = None
        self.last_ask_query = None
        return info

    def _display_name(self) -> str:
        try:
            from persona.registry import registry
            cfg = registry.get(self.persona_id or "")
            if cfg:
                return cfg.display_name
        except Exception:
            pass
        return self.persona_id or "讲解员"

    async def idle_watch_loop(self):
        print(f"[ROBOT] 空闲看门狗已启动 timeout={ROBOT_IDLE_TIMEOUT}s max_rounds={ROBOT_MAX_ROUNDS}")
        while True:
            try:
                await asyncio.sleep(5)
                async with self._lock:
                    if (self.session_id and not self.busy
                            and time.time() - self.last_active > ROBOT_IDLE_TIMEOUT):
                        self._release_locked("idle_timeout")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[ROBOT] 空闲看门狗异常: {e}")


robot_hub = RobotHub()
