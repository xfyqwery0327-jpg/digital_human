# -*- coding: utf-8 -*-
"""机器人请求字段兼容：Pydantic v1/v2、整型 id、字符串布尔、多余字段。"""
from __future__ import annotations
from typing import Any, Dict, Optional


def coerce_opt_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="replace")
    s = str(v).strip()
    return s or None


def coerce_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off", ""):
            return False
    return bool(v)


def normalize_robot_data(data: Any) -> Any:
    """对方常见写法：session_id=数字、wait="false"、多传字段、category 数组。"""
    if not isinstance(data, dict):
        return data
    out: Dict[str, Any] = dict(data)
    if "session_id" in out:
        out["session_id"] = coerce_opt_str(out.get("session_id"))
    rid = out.get("robot_id", None)
    if rid is None or (isinstance(rid, str) and not rid.strip()):
        out["robot_id"] = "default"
    else:
        out["robot_id"] = str(rid).strip()
    if "callback_url" in out:
        out["callback_url"] = coerce_opt_str(out.get("callback_url"))
    for key in ("wait", "include_audio", "silent", "notify_peer", "play_local"):
        if key in out:
            defaults = {
                "wait": False,
                "include_audio": True,
                "silent": False,
                "notify_peer": True,
                "play_local": False,
            }
            out[key] = coerce_bool(out[key], defaults[key])
    if "top_k" in out and out["top_k"] is not None:
        try:
            out["top_k"] = int(out["top_k"])
        except (TypeError, ValueError):
            out["top_k"] = 2
    if "mode" in out and out["mode"] is not None:
        try:
            out["mode"] = int(out["mode"])
        except (TypeError, ValueError):
            out["mode"] = 1
    cat = out.get("category")
    if isinstance(cat, list):
        out["category"] = [str(x) for x in cat if x]
    elif cat is not None:
        out["category"] = coerce_opt_str(cat)
    return out


def _build_compat_model():
    from pydantic import BaseModel
    try:
        from pydantic import ConfigDict, model_validator

        class RobotCompatModel(BaseModel):
            model_config = ConfigDict(extra="ignore")

            @model_validator(mode="before")
            @classmethod
            def _normalize(cls, data):
                return normalize_robot_data(data)

        return RobotCompatModel
    except Exception:
        from pydantic import root_validator

        class RobotCompatModel(BaseModel):
            class Config:
                extra = "ignore"

            @root_validator(pre=True)
            def _normalize(cls, values):
                return normalize_robot_data(values)

        return RobotCompatModel


try:
    RobotCompatModel = _build_compat_model()
except Exception:
    RobotCompatModel = None  # 无 pydantic 时仍可导入 normalize_* 做协议自检
