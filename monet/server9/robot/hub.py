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
        self.current_request_id: Optional[str] = None
        self.current_action: Optional[str] = None
        self.last_ask_query: Optional[str] = None

    def snapshot(self) -> Dict:
        return {
            "active": bool(self.session_id),
            "session_id": self.session_id,
            "robot_id": self.robot_id,
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

    def is_robot_ask_echo(self, text: str) -> bool:
        """前端收到 ask 推的 asr 后，常会再发一条 type=tts，避免再走一轮游客讲解。
        只在本轮还在推流时拦截；讲完后若前端漏收 robot 流，允许它自己补一次 TTS。"""
        q = (text or "").strip()
        last = (self.last_ask_query or "").strip()
        return bool(self.session_id) and self.busy and bool(q) and q == last

    def _adopt_caller(self, robot_id: Optional[str], now: float):
        if robot_id:
            self.robot_id = robot_id
        self.last_active = now

    async def acquire(self, robot_id: str, session_id: Optional[str],
                      create: bool = True,
                      action: str = "") -> Tuple[bool, Optional[Dict], Optional[JSONResponse]]:
        async with self._lock:
            now = time.time()
            if self.busy:
                return False, None, JSONResponse(status_code=409, content={
                    "ok": False, "code": "turn_in_progress",
                    "message": "本轮介绍尚未结束，请等待莫奈讲完后再发下一轮",
                    "session_id": self.session_id,
                    "request_id": self.current_request_id,
                    "action": self.current_action,
                })

            if self.session_id:
                self._adopt_caller(robot_id, now)
                return True, {
                    "session_id": self.session_id,
                    "created": False,
                    "robot_id": self.robot_id,
                }, None

            if not create:
                return False, None, JSONResponse(status_code=409, content={
                    "ok": False, "code": "no_session",
                    "message": "没有进行中的机器人会话，请先调用 /api/robot/ask",
                })

            sid = session_id or f"robot_{int(now)}_{uuid.uuid4().hex[:8]}"
            self.session_id = sid
            self.robot_id = robot_id
            self.round = 0
            self.max_rounds = ROBOT_MAX_ROUNDS
            self.busy = False
            self.started_at = now
            self.last_active = now
            self.monet_session = SessionState(robot_id, sid)
            active_sessions[sid] = self.monet_session
            print(f"[ROBOT] 开启会话 session={sid} robot={robot_id}")
            return True, {"session_id": sid, "created": True, "robot_id": robot_id}, None

    async def mark_busy(self, action: str, request_id: str, query: str = "") -> None:
        async with self._lock:
            self.busy = True
            self.current_action = action
            self.current_request_id = request_id
            self.last_active = time.time()
            if query:
                self.last_ask_query = query.strip()

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
                  f"robot={self.robot_id} reason={reason} rounds={self.round}")
            persistence.save_session(self.monet_session) if self.monet_session else None
        self.session_id = None
        self.robot_id = None
        self.round = 0
        self.busy = False
        self.started_at = 0.0
        self.last_active = 0.0
        self.monet_session = None
        self.current_request_id = None
        self.current_action = None
        self.last_ask_query = None
        return info

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
