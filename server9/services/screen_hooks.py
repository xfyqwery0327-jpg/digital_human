# -*- coding: utf-8 -*-
"""屏幕动作：门票预约、画作名。接口地址未到，用环境变量留调用位。"""
from __future__ import annotations
import os
import re
from typing import Dict, Iterable, List

_BOOKING = re.compile(r"预约|订票|订一张|订张|买票|购票|帮我约|想订|要订|门票服务")
_BOOK_TITLE = re.compile(r"《([^》]{1,40})》")
_NORM = re.compile(r"[\s·・•]+")

BOOKING_REPLY = "好，预约请在屏幕上填写。"


def is_ticket_booking(question: str) -> bool:
    """预约、订票、买票。问票价和开放时间不走这里。"""
    return bool(_BOOKING.search(question or ""))


def booking_reply() -> str:
    return BOOKING_REPLY


def ticket_form_event(question: str, session_id: str = "", persona_id: str = "") -> dict:
    return {
        "type": "ticket_form",
        "action": "open",
        "intent": "ticket_service",
        "session_id": session_id or "",
        "persona_id": persona_id or "",
        "query": question or "",
    }


def request_ticket_form(question: str, persona_id: str = "",
                        session_id: str = "") -> dict:
    """弹出预约表单的后端接口。未配置 TICKET_BOOKING_URL 时只记录，不发请求。"""
    payload = {
        "intent": "ticket_service",
        "persona_id": persona_id or "",
        "session_id": session_id or "",
        "query": question or "",
    }
    return _post_reserved("TICKET_BOOKING_URL", payload, "门票预约")


def catalog_from_items(items: Iterable[dict]) -> List[str]:
    """从索引标题和正文里的书名号收集画作名。长名在前，避免「睡莲」截断「睡莲池」。"""
    found = []
    seen = set()
    for it in items or []:
        if not isinstance(it, dict):
            continue
        blob = (it.get("title") or "") + "\n" + (it.get("text") or "")
        for name in _BOOK_TITLE.findall(blob):
            name = name.strip()
            key = _NORM.sub("", name)
            if len(key) < 2 or key in seen:
                continue
            seen.add(key)
            found.append(name)
    found.sort(key=lambda s: len(_NORM.sub("", s)), reverse=True)
    return found


def artworks_in_text(text: str, catalog: Iterable[str]) -> List[str]:
    """从口播里截取目录中的画作名。先认书名号，再认正文里的全名。"""
    raw = text or ""
    names = [n for n in (catalog or []) if n]
    if not raw or not names:
        return []
    occupied = [False] * len(raw)
    picked: List[str] = []
    seen = set()

    def _take(start: int, end: int, canonical: str):
        key = _NORM.sub("", canonical)
        if key in seen or end <= start:
            return
        if any(occupied[start:end]):
            return
        for i in range(start, end):
            occupied[i] = True
        seen.add(key)
        picked.append(canonical)

    for m in _BOOK_TITLE.finditer(raw):
        canon = _match_catalog(m.group(1).strip(), names)
        if canon:
            _take(m.start(), m.end(), canon)

    for name in names:
        if _NORM.sub("", name) in seen:
            continue
        variants = [name]
        bare = _NORM.sub("", name)
        if bare and bare not in variants:
            variants.append(bare)
        for variant in variants:
            start = 0
            while True:
                idx = raw.find(variant, start)
                if idx < 0:
                    break
                _take(idx, idx + len(variant), name)
                if _NORM.sub("", name) in seen:
                    break
                start = idx + len(variant)
    return picked


def _match_catalog(inner: str, names: List[str]) -> str:
    key = _NORM.sub("", inner)
    if len(key) < 2:
        return ""
    for name in names:
        if _NORM.sub("", name) == key:
            return name
    return ""


_catalog_cache: Dict[str, List[str]] = {}


def catalog_for_persona(persona) -> List[str]:
    pid = getattr(persona, "id", "") or ""
    if pid in _catalog_cache:
        return _catalog_cache[pid]
    try:
        from rag import index as rag
        bundle = rag.bundle_for(persona) if persona is not None else None
        items = (bundle or {}).get("items") or []
    except Exception:
        items = []
    names = catalog_from_items(items)
    _catalog_cache[pid] = names
    return names


def notify_artworks(names: List[str], reply: str = "", persona_id: str = "",
                    session_id: str = "") -> dict:
    """画作出现后通知前端或业务接口。未配置 ARTWORK_NOTIFY_URL 时只记录。"""
    payload = {
        "persona_id": persona_id or "",
        "session_id": session_id or "",
        "names": list(names or []),
        "reply": reply or "",
    }
    if not payload["names"]:
        return {"ok": False, "code": "empty"}
    return _post_reserved("ARTWORK_NOTIFY_URL", payload, "画作")


def headline_event(title: str, image_url: str, link: str = "",
                   session_id: str = "", persona_id: str = "") -> dict:
    """前端展示头条原文图。image_url 指向 jpg，不送进语音。"""
    return {
        "type": "headline",
        "action": "show",
        "intent": "news",
        "session_id": session_id or "",
        "persona_id": persona_id or "",
        "title": title or "",
        "image_url": image_url or "",
        "link": link or "",
    }


def request_headline_image(title: str, image_url: str, link: str = "",
                           persona_id: str = "", session_id: str = "") -> dict:
    """头条配图的业务接口。未配置 HEADLINE_IMAGE_URL 时只记录，不发请求。"""
    payload = {
        "intent": "news",
        "persona_id": persona_id or "",
        "session_id": session_id or "",
        "title": title or "",
        "image_url": image_url or "",
        "link": link or "",
    }
    if not payload["image_url"]:
        return {"ok": False, "code": "empty"}
    return _post_reserved("HEADLINE_IMAGE_URL", payload, "头条图片")


def artworks_event(names: List[str], session_id: str = "", persona_id: str = "") -> dict:
    return {
        "type": "artworks",
        "session_id": session_id or "",
        "persona_id": persona_id or "",
        "names": list(names or []),
    }


def _post_reserved(env_name: str, payload: dict, label: str) -> dict:
    url = os.environ.get(env_name, "").strip()
    if not url:
        print(f"📌 {label}接口未配置，已预留 {env_name} payload_keys={list(payload)}")
        return {"ok": False, "code": "not_configured", "payload": payload}
    try:
        import requests
        resp = requests.post(url, json=payload, timeout=3)
        print(f"📌 {label}接口 status={resp.status_code} url={url}")
        return {"ok": resp.ok, "code": "sent", "status": resp.status_code}
    except Exception as e:
        print(f"⚠️ {label}接口失败: {e}")
        return {"ok": False, "code": "request_failed", "error": str(e)[:200]}
