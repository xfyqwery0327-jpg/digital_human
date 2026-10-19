# -*- coding: utf-8 -*-
"""顺序三次调用 POST /api/robot/ask：问候 → 提问 → 再见。

对应原先 hello / ask / bye 三个接口，现已统一走 ask。
每一轮立刻 HTTP 202；脚本轮询 GET /health，等 robot.busy=false 再发下一轮。

用法:
  python scripts/call_robot.py
  python scripts/call_robot.py --query 介绍下日出这个作品吧
  python scripts/call_robot.py --base http://127.0.0.1:8848
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

BASE = "http://10.0.7.232:8848"
ROBOT_ID = "script-01"
TIMEOUT = 30
IDLE_TIMEOUT = 180
POLL_SEC = 2


def _url(path: str) -> str:
    return BASE.rstrip("/") + path


def _request(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    url = _url(path)
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            code = resp.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        code = e.code
    except TimeoutError:
        print(f"<<< 等待超过 {TIMEOUT}s 无 HTTP 返回。ask 应立刻 202；"
              f"请看讲解员控制台是否卡在受理之前。")
        raise SystemExit(1)
    except urllib.error.URLError as e:
        print(f"<<< 连接失败: {e}")
        raise SystemExit(1)

    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        print(f"<<< HTTP {code}\n{raw[:2000]}")
        raise SystemExit(1)
    return code, payload


def get_health() -> dict:
    code, payload = _request("GET", "/health")
    if code != 200:
        print(f"<<< GET /health HTTP {code}")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    return payload


def post_ask(query: str, robot_id: str, session_id: str | None) -> dict:
    body = {
        "robot_id": robot_id,
        "query": query,
        "include_audio": False,
        "top_k": 2,
    }
    if session_id:
        body["session_id"] = session_id
    print(f"\n>>> POST {_url('/api/robot/ask')}")
    print(json.dumps(body, ensure_ascii=False, indent=2))
    code, payload = _request("POST", "/api/robot/ask", body)
    shown = dict(payload)
    audio = shown.get("audio")
    if isinstance(audio, str) and audio:
        shown["audio"] = f"<omitted base64 chars={len(audio)}>"
    print(f"<<< HTTP {code}")
    print(json.dumps(shown, ensure_ascii=False, indent=2))
    if code == 409:
        print("※ 上一轮还在讲（409 turn_in_progress），请等 busy 结束后再发。")
        raise SystemExit(1)
    if code not in (200, 202) or payload.get("code") != "accepted":
        print("※ 未按约定受理，停止。")
        raise SystemExit(1)
    print("※ 已受理。讲解在后台进行，字幕和声音走前端 /ws/tts。")
    return payload


def wait_idle(label: str) -> dict:
    print(f"\n… 等待「{label}」讲完（轮询 GET /health，robot.busy=false）")
    deadline = time.time() + IDLE_TIMEOUT
    last = None
    while time.time() < deadline:
        health = get_health()
        robot = health.get("robot") or {}
        snap = {
            "busy": robot.get("busy"),
            "active": robot.get("active"),
            "session_id": robot.get("session_id"),
            "round": robot.get("round"),
            "current_action": robot.get("current_action"),
        }
        if snap != last:
            print(f"    health.robot = {json.dumps(snap, ensure_ascii=False)}")
            last = snap
        if not robot.get("busy"):
            print(f"※ 「{label}」本轮结束 busy=false active={robot.get('active')}")
            return robot
        time.sleep(POLL_SEC)
    print(f"<<< 超过 {IDLE_TIMEOUT}s 仍 busy，停止。")
    raise SystemExit(1)


def print_step(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    global BASE
    parser = argparse.ArgumentParser(description="三次 ask：问候 / 提问 / 再见")
    parser.add_argument("--base", default=BASE, help="讲解员地址")
    parser.add_argument("--hello", default="你好莫奈，介绍下自己吧", help="原 hello：问候语，走 ask")
    parser.add_argument("--query", default="介绍下日出这个作品吧", help="原 ask：知识问题")
    parser.add_argument("--bye", default="再见", help="原 bye：告别语，走 ask")
    parser.add_argument("--robot-id", default=ROBOT_ID)
    parser.add_argument("--skip-hello", action="store_true", help="跳过问候，直接提问")
    parser.add_argument("--skip-bye", action="store_true", help="跳过告别")
    args = parser.parse_args()
    BASE = args.base

    print_step("探活")
    health = get_health()
    print(f"GET {_url('/health')}")
    print(json.dumps({
        "status": health.get("status"),
        "tts": health.get("tts"),
        "tts_ws_clients": health.get("tts_ws_clients"),
        "robot": health.get("robot"),
    }, ensure_ascii=False, indent=2))
    robot = health.get("robot") or {}
    if robot.get("busy"):
        print("※ 当前已有一轮在讲，先等到 idle")
        wait_idle("已有会话")

    session_id = None
    steps = []
    if not args.skip_hello:
        steps.append(("原 hello → ask", args.hello))
    steps.append(("原 ask → ask", args.query))
    if not args.skip_bye:
        steps.append(("原 bye → ask", args.bye))

    for i, (label, query) in enumerate(steps, 1):
        print_step(f"{i}/{len(steps)} {label}  query={query}")
        payload = post_ask(query, args.robot_id, session_id)
        session_id = payload.get("session_id") or session_id
        robot = wait_idle(label)
        if i == len(steps) and "再见" in query:
            if robot.get("active"):
                print("※ 注意：告别后会话仍 active，预期应已释放。")
            else:
                print("※ 告别后会话已释放，符合预期。")

    print_step("完成")
    print("三次动作均走 POST /api/robot/ask。请对照前端 /ws/tts 听问候、讲解、告别。")


if __name__ == "__main__":
    main()
