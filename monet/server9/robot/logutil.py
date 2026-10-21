# -*- coding: utf-8 -*-
"""机器人 api 调试日志。控制台 + logs/ 滚动文件。"""
from __future__ import annotations
import json
import logging
from logging.handlers import RotatingFileHandler
from typing import Any, Dict

from config import LOG_DIR, ROBOT_LOG_LEVEL

_FMT = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _level() -> int:
    return getattr(logging, ROBOT_LOG_LEVEL, logging.DEBUG)


def get_logger(name: str, filename: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(_level())
    fh = RotatingFileHandler(
        LOG_DIR / filename,
        maxBytes=8 * 1024 * 1024,
        backupCount=8,
        encoding="utf-8",
    )
    fh.setLevel(_level())
    fh.setFormatter(_FMT)
    ch = logging.StreamHandler()
    ch.setLevel(_level())
    ch.setFormatter(_FMT)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


def payload_for_log(payload: Dict[str, Any]) -> Dict[str, Any]:
    """去掉超长 audio，方便直接看日志。"""
    out = dict(payload)
    audio = out.get("audio")
    if isinstance(audio, str) and audio:
        out["audio"] = f"<omitted base64 chars={len(audio)}>"
    text = out.get("text")
    if isinstance(text, str) and len(text) > 160:
        out["text"] = text[:160] + "..."
    return out


def dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)
