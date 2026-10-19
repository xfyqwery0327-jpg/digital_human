# -*- coding: utf-8 -*-
"""模拟机器人 ↔ 莫奈 全流程调用与参数，并做兼容性自检。

用法（项目根）:
  python scripts/sim_robot_flow.py
不启动 TTS/LLM，只跑协议层（会话续接、请求字段）。
"""
from __future__ import annotations
import asyncio
import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server9"))
os.environ.setdefault("ROBOT_LOG_LEVEL", "WARNING")


def _stub_optional_deps():
    """本机默认 Python 可能没有 fastapi/requests；协议自检仍要能跑。"""
    try:
        import fastapi  # noqa: F401
    except ImportError:
        fastapi = types.ModuleType("fastapi")
        responses = types.ModuleType("fastapi.responses")

        class JSONResponse:
            def __init__(self, content=None, status_code=200, **_kw):
                self.status_code = status_code
                self.body = json.dumps(content, ensure_ascii=False).encode("utf-8")

        responses.JSONResponse = JSONResponse
        sys.modules["fastapi"] = fastapi
        sys.modules["fastapi.responses"] = responses
    try:
        import requests  # noqa: F401
    except ImportError:
        sys.modules["requests"] = types.ModuleType("requests")


_stub_optional_deps()

from robot.compat import coerce_bool, normalize_robot_data  # noqa: E402
from robot.hub import RobotHub  # noqa: E402
import config as cfg  # noqa: E402


MONET = f"http://192.168.1.10:{cfg.PORT}"
SID = "robot_1757490000_ab12cd34"
RID = "robot-01"


def dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def print_step(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_http(method: str, url: str, body=None, headers=None, status=None, resp=None):
    print(f"\n{method} {url}")
    if headers:
        for k, v in headers.items():
            print(f"  {k}: {v}")
    if body is not None:
        print("  body =")
        for line in dump(body).splitlines():
            print("   ", line)
    if status is not None:
        print(f"  <-- HTTP {status}")
    if resp is not None:
        for line in dump(resp).splitlines():
            print("   ", line)


def show_protocol():
    headers = {"Content-Type": "application/json; charset=utf-8"}
    print_step("1. 推荐时序：只调 ask（立刻 HTTP 202，声音走前端 /ws/tts）")
    print("约定：收到受理响应后不要立刻再 ask；等本轮讲完（上一轮 busy 结束）再发下一轮。")
    print("不必 hello / bye / status。query 含「再见」时口播告别并结束会话。")

    for i, query in enumerate([
        "你好莫奈，介绍下自己吧",
        "介绍下日出这个作品吧",
    ], 1):
        print_http("POST", f"{MONET}/api/robot/ask", {
            "query": query, "robot_id": RID, "session_id": SID if i > 1 else None,
            "include_audio": False, "top_k": 2,
        }, headers, 202, {
            "ok": True, "code": "accepted", "accepted": True, "done": False,
            "action": "ask", "session_id": SID, "request_id": f"ask{i:03d}",
            "robot_id": RID, "next_hint": "ask",
            "message": "已受理。讲解在后台进行。",
        })

    print_http("POST", f"{MONET}/api/robot/ask", {
        "query": "再见", "robot_id": RID, "session_id": SID,
        "include_audio": False,
    }, headers, 202, {
        "ok": True, "code": "accepted", "accepted": True, "done": False,
        "action": "ask", "session_id": SID, "request_id": "ask999",
        "robot_id": RID, "next_hint": "ask",
        "message": "已受理。讲解在后台进行。",
    })

    print_step("2. 打断：讲的过程中再 ask → 409")
    print_http("POST", f"{MONET}/api/robot/ask", {
        "query": "现在就问下一句", "robot_id": RID, "session_id": SID,
    }, headers, 409, {
        "ok": False, "code": "turn_in_progress",
        "message": "本轮介绍尚未结束，请等待莫奈讲完后再发下一轮",
        "session_id": SID, "request_id": "ask001", "action": "ask",
    })

    print_step("3. wait 字段已忽略")
    print("ask 一律立刻 HTTP 202，不会在 HTTP 里等讲完，也不再回调对方。")


def test_compat_fields():
    print_step("自检 A：对方脏字段兼容")
    samples = [
        ({"wait": "false", "include_audio": "false", "session_id": 12345, "robot_id": 1},
         {"wait": False, "include_audio": False, "session_id": "12345", "robot_id": "1"}),
        ({"wait": "true", "silent": 0, "notify_peer": "yes"},
         {"wait": True, "silent": False, "notify_peer": True}),
        ({"robot_id": "", "callback_url": "  ", "extra_field": "ignore-me"},
         {"robot_id": "default", "callback_url": None}),
        ({"category": ["exhibition_monet"], "top_k": "3"},
         {"category": ["exhibition_monet"], "top_k": 3}),
    ]
    for raw, expect in samples:
        got = normalize_robot_data(raw)
        for k, v in expect.items():
            assert got.get(k) == v, (raw, k, got.get(k), v)
        print(f"  {raw} -> { {k: got.get(k) for k in expect} }")
    assert coerce_bool("false") is False
    assert coerce_bool("true") is True
    print("  字符串布尔 / 整型 id / 多余字段：OK")


async def test_hub():
    print_step("自检 B：会话续接 / 本轮进行中 409")
    hub = RobotHub()
    ok, info, err = await hub.acquire(RID, None, create=True, action="ask")
    assert ok and info["created"] and info["session_id"]
    sid = info["session_id"]
    print(f"  ask 开会话 session={sid}")

    ok2, info2, err2 = await hub.acquire("robot-02", None, create=True, action="ask")
    assert ok2 and err2 is None and info2["session_id"] == sid
    assert hub.robot_id == "robot-02"
    print("  另一台机器人空闲时直接接管，不再 409 robot_busy：OK")

    await hub.mark_busy("ask", "req-ask")
    ok3, _, err3 = await hub.acquire(RID, sid, create=True, action="ask")
    assert not ok3
    body3 = json.loads(err3.body.decode("utf-8"))
    assert body3["code"] == "turn_in_progress"
    print("  本轮未结束再 ask → turn_in_progress：OK")

    await hub.mark_idle()
    ok4, info4, err4 = await hub.acquire(RID, sid, create=True)
    assert ok4 and not info4["created"] and err4 is None
    print("  同 session 续接：OK")

    print("  轮次不设上限；ask 里说再见即可结束会话")
    info5 = await hub.release("farewell")
    assert info5["reason"] == "farewell"
    assert hub.session_id is None
    print("  告别释放会话：OK")


def main():
    show_protocol()
    test_compat_fields()
    asyncio.run(test_hub())
    print_step("结论")
    print("协议层自检通过。现场联调请用 conda env robot 启动 python server8.py，")
    print("对方只调 POST /api/robot/ask；不再回调对方。成功受理默认 HTTP 202。")


if __name__ == "__main__":
    main()
