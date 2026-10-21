# -*- coding: utf-8 -*-
"""请求级人设上下文：API/WS 按 persona_id 绑定，禁止串到别人的规则/知识/音色。"""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator, Optional

from fastapi import HTTPException, WebSocketException, status

from persona.models import PersonaConfig
from persona.registry import registry

_active: ContextVar[Optional[PersonaConfig]] = ContextVar("active_persona", default=None)


def resolve(persona_id: Optional[str], *, allow_default: bool = True) -> PersonaConfig:
    if not registry._items:
        registry.load()
    pid = (persona_id or "").strip()
    if not pid:
        if not allow_default:
            raise HTTPException(status_code=422, detail="缺少 persona_id")
        if registry.default_id in registry._items:
            return registry.require(registry.default_id)
        from persona.stage import stage
        return stage.current()
    cfg = registry.get(pid)
    if not cfg or not cfg.enabled:
        known = "、".join(p.id for p in registry.enabled()) or registry.default_id
        raise HTTPException(
            status_code=404,
            detail=f"未知或未启用人设: {pid}（可用: {known}）",
        )
    return cfg


def resolve_ws(persona_id: Optional[str]) -> PersonaConfig:
    try:
        return resolve(persona_id, allow_default=True)
    except HTTPException as e:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason=str(e.detail))


def peek() -> Optional[PersonaConfig]:
    """当前请求是否已 bind。工作线程里没有 ContextVar，返回 None，禁止当成莫奈。"""
    return _active.get()


def active() -> PersonaConfig:
    p = _active.get()
    if p is not None:
        return p
    from persona.stage import stage
    return stage.current()


def call_bound(fn: Callable[..., Any], *args, persona: Optional[PersonaConfig] = None,
               **kwargs) -> Any:
    """在线程池里带着人设跑，避免 ContextVar 丢失后落到舞台默认莫奈。"""
    p = persona if persona is not None else active()
    with bind(p):
        return fn(*args, **kwargs)


@contextmanager
def bind(persona: PersonaConfig) -> Iterator[PersonaConfig]:
    token = _active.set(persona)
    try:
        yield persona
    finally:
        _active.reset(token)
