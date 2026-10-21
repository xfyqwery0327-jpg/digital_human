# -*- coding: utf-8 -*-
"""模拟机器人 ↔ 莫奈 全流程调用与参数，并做兼容性自检。

用法（项目根）:
  python scripts/sim_robot_flow.py
不启动 TTS/LLM，只跑协议层（独占锁、回调地址、请求字段）。
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
from robot.callback import join_robot_url, robot_callback_url  # noqa: E402
from robot.hub import RobotHub  # noqa: E402
import config as cfg  # noqa: E402


PEER = "http://10.0.0.21:9000"
CALLBACK = f"{PEER}/api/monet/callback"
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
    print_step("1. 推荐时序：hello → ask×3 → bye（wait=false，以回调为准）")
    print("约定：收到受理响应后不要立刻 ask；等 turn_end（spoken=true）再发下一轮。")
    print("callback_url 用同一个地址，靠 body.event 分支。")

    hello_req = {
        "robot_id": RID,
        "session_id": None,
        "callback_url": CALLBACK,
        "include_audio": False,
        "wait": False,
        "speaker": "中文男",
        "text": None,
    }
    print_http("POST", f"{MONET}/api/robot/hello", hello_req, headers, 202, {
        "ok": True, "accepted": True, "action": "hello",
        "session_id": SID, "request_id": "aaa111", "robot_id": RID,
        "message": "已受理。莫奈本机讲完后才会回调 turn_end，在此之前不要发下一轮",
    })
    print("\n莫奈本机播放欢迎语……播完后回调：")
    print_http("POST", CALLBACK, {
        "event": "turn_end", "ok": True, "action": "hello",
        "session_id": SID, "request_id": "aaa111", "robot_id": RID,
        "round": 0, "max_rounds": 3, "can_continue": True, "rounds_left": 3,
        "refused": False, "sources": [],
        "text": "欢迎来到我的世界。我是克洛德·莫奈，今天由我陪你看看光和颜色的故事。",
        "audio": "", "audio_format": "", "sample_rate": 0,
        "spoken": True, "played_local": True, "play_duration_sec": 5.2,
        "elapsed_sec": 6.1, "next_hint": "ask",
    }, headers, 200, {"ok": True})

    for i, query in enumerate([
        "睡莲是在哪里画的？",
        "日出·印象是哪一年的作品？",
        "吉维尼花园是什么样的？",
    ], 1):
        print_http("POST", f"{MONET}/api/robot/ask", {
            "query": query, "robot_id": RID, "session_id": SID,
            "callback_url": CALLBACK, "include_audio": False, "wait": False, "top_k": 2,
        }, headers, 202, {
            "ok": True, "accepted": True, "action": "ask",
            "session_id": SID, "request_id": f"ask{i:03d}", "robot_id": RID,
        })
        can = i < 3
        print_http("POST", CALLBACK, {
            "event": "turn_end", "ok": True, "action": "ask",
            "session_id": SID, "request_id": f"ask{i:03d}", "robot_id": RID,
            "round": i, "max_rounds": 3, "can_continue": can,
            "rounds_left": 3 - i, "refused": False,
            "sources": ["item_demo"],
            "text": f"（第{i}轮莫奈讲解文本）",
            "audio": "", "audio_format": "", "sample_rate": 0,
            "spoken": True, "played_local": True, "play_duration_sec": 8.0,
            "elapsed_sec": 10.0, "next_hint": "ask" if can else "bye",
        }, headers, 200, {"ok": True})

    print_http("POST", f"{MONET}/api/robot/bye", {
        "robot_id": RID, "session_id": SID, "callback_url": CALLBACK,
        "include_audio": False, "wait": False, "silent": False, "text": None,
    }, headers, 202, {
        "ok": True, "accepted": True, "action": "bye",
        "session_id": SID, "request_id": "bye001", "robot_id": RID,
    })
    print_http("POST", CALLBACK, {
        "event": "turn_end", "ok": True, "action": "bye",
        "session_id": SID, "request_id": "bye001", "robot_id": RID,
        "round": 3, "max_rounds": 3, "can_continue": False, "rounds_left": 0,
        "refused": False, "sources": [],
        "text": "很高兴今天和你聊天。再见，愿你把这里的光带走。",
        "audio": "", "audio_format": "", "sample_rate": 0,
        "spoken": True, "played_local": True, "play_duration_sec": 4.0,
        "elapsed_sec": 5.0, "next_hint": "bye",
    }, headers, 200, {"ok": True})
    print_http("POST", CALLBACK, {
        "event": "session_end", "reason": "bye", "session_id": SID,
        "robot_id": RID, "round": 3, "action": "bye", "ok": True,
    }, headers, 200, {"ok": True})

    print_step("2. 打断：讲的过程中再 ask → 409")
    print_http("POST", f"{MONET}/api/robot/ask", {
        "query": "现在就问下一句", "robot_id": RID, "session_id": SID,
    }, headers, 409, {
        "ok": False, "code": "turn_in_progress",
        "message": "本轮介绍尚未结束，请等待莫奈讲完（turn_end）后再发下一轮",
        "session_id": SID, "request_id": "aaa111", "action": "hello",
    })

    print_step("3. 同步模式 wait=true（对方不会实现回调时）")
    print("HTTP 会等到本机讲完才返回 turn_end 同结构 JSON（仍会再 POST 一次回调，若配置了地址）。")
    print_http("POST", f"{MONET}/api/robot/hello", {
        "robot_id": RID, "callback_url": CALLBACK, "wait": True, "include_audio": False,
    }, headers, 200, {
        "event": "turn_end", "ok": True, "action": "hello",
        "session_id": SID, "spoken": True, "played_local": True, "next_hint": "ask",
    })

    print_step("4. 无 callback_url 且未配 ROBOT_API_BASE")
    print("讲解照常进行，日志写「未配置回调地址」，notify_robot 返回 (False, no_callback_url)。")

    print_step("5. 拆路径回调（不传 callback_url，配 ROBOT_API_BASE）")
    print(f"ROBOT_API_BASE={PEER}")
    print(f"  turn_end    -> {PEER}{cfg.ROBOT_CALLBACK_TURN_END}")
    print(f"  session_end -> {PEER}{cfg.ROBOT_CALLBACK_SESSION_END}")
    print(f"  hello 预告  -> {PEER}{cfg.ROBOT_CALLBACK_HELLO}  （仅 push/hello 且 notify_peer=true）")
    print(f"  bye 预告    -> {PEER}{cfg.ROBOT_CALLBACK_BYE}  （仅 push/bye 且 notify_peer=true）")


def test_callback_urls():
    print_step("自检 A：回调地址解析")
    old_base = cfg.ROBOT_API_BASE
    cfg.ROBOT_API_BASE = PEER
    try:
        assert robot_callback_url("turn_end", CALLBACK) == CALLBACK
        assert robot_callback_url("session_end", CALLBACK) == CALLBACK
        assert robot_callback_url("turn_end") == PEER + cfg.ROBOT_CALLBACK_TURN_END
        assert robot_callback_url("session_end") == PEER + cfg.ROBOT_CALLBACK_SESSION_END
        assert join_robot_url("/x") == PEER + "/x"
        print("  override 同一 URL 收全部 event：OK")
        print("  ROBOT_API_BASE + 约定路径：OK")
    finally:
        cfg.ROBOT_API_BASE = old_base

    cfg.ROBOT_API_BASE = ""
    try:
        assert robot_callback_url("turn_end") is None
        print("  无 override 且无 BASE → None（跳过回调）：OK")
    finally:
        cfg.ROBOT_API_BASE = old_base


def test_compat_fields():
    print_step("自检 B：对方脏字段兼容")
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
    print_step("自检 C：独占锁 / 409 / 回调地址不被后来者覆盖")
    hub = RobotHub()
    ok, info, err = await hub.acquire(RID, None, CALLBACK, create=True)
    assert ok and info["created"] and info["session_id"]
    sid = info["session_id"]
    print(f"  hello 开会话 session={sid} callback={hub.callback_url}")

    ok2, _, err2 = await hub.acquire("robot-02", None, f"{PEER}/hijack", create=True)
    assert not ok2 and err2.status_code == 409
    body = json.loads(err2.body.decode("utf-8"))
    assert body["code"] == "robot_busy"
    assert hub.callback_url == CALLBACK, hub.callback_url
    print("  另一台机器人 409，且未改写占用方 callback_url：OK")

    await hub.mark_busy("hello", "req-hello")
    ok3, _, err3 = await hub.acquire(RID, sid, CALLBACK, create=True)
    assert not ok3
    body3 = json.loads(err3.body.decode("utf-8"))
    assert body3["code"] == "turn_in_progress"
    print("  本轮未结束再 ask → turn_in_progress：OK")

    await hub.mark_idle()
    ok4, info4, err4 = await hub.acquire(RID, sid, CALLBACK, create=True)
    assert ok4 and not info4["created"] and err4 is None
    print("  同 session 续接：OK")

    hub.round = 3
    print(f"  模拟 ask 满 {hub.max_rounds} 轮后 can_continue=false，应改调 bye")

    info5, url, err5 = await hub.silent_end(RID, sid, None)
    assert err5 is None and info5["reason"] == "silent_bye"
    assert url == CALLBACK
    assert hub.session_id is None
    print("  silent bye 释放并回 session_end URL：OK")

    ok6, _, err6 = await hub.acquire(RID, None, None, create=True)
    assert ok6
    await hub.mark_busy("ask", "x")
    _, _, err7 = await hub.silent_end(RID, hub.session_id, None)
    assert json.loads(err7.body.decode("utf-8"))["code"] == "turn_in_progress"
    print("  合成中 silent bye 拒绝：OK")
    await hub.release("test")


def main():
    show_protocol()
    test_callback_urls()
    test_compat_fields()
    asyncio.run(test_hub())
    print_step("结论")
    print("协议层自检通过。现场联调请用 conda env robot 启动 python server8.py，")
    print("对方用上面 JSON；include_audio 建议 false（本机已播，spoken=true 不要再播）。")
    print("若对方 HTTP 只认 200，设置环境变量 ROBOT_ACCEPTED_STATUS=200。")


if __name__ == "__main__":
    main()
