# -*- coding: utf-8 -*-
"""人设查询与现场切换。"""
from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from config import ROBOT_API_TOKEN
from persona.registry import registry
from persona.stage import stage

router = APIRouter(tags=["persona"])


def _verify_token(x_robot_token: Optional[str] = Header(default=None, alias="X-Robot-Token")):
    if ROBOT_API_TOKEN and x_robot_token != ROBOT_API_TOKEN:
        raise HTTPException(status_code=401, detail="invalid X-Robot-Token")


@router.get("/api/persona/list")
def api_persona_list():
    return {
        "default_persona_id": registry.default_id,
        "current_persona_id": stage.current_id(),
        "personas": [p.snapshot() for p in registry.all()],
    }


@router.get("/api/persona/current")
def api_persona_current():
    p = stage.current()
    return {"ok": True, **p.snapshot()}


class SwitchReq(BaseModel):
    persona_id: str = Field(..., description="目标数字人 id，如 monet / fuhao / kumarajiva")
    play_hello: bool = False
    reason: str = "api"


@router.post("/api/persona/switch", dependencies=[Depends(_verify_token)])
async def api_persona_switch(req: SwitchReq):
    result = await stage.switch(req.persona_id, play_hello=req.play_hello, reason=req.reason)
    if not result.get("ok"):
        raise HTTPException(status_code=422, detail=result.get("message") or "switch failed")
    return result
