# -*- coding: utf-8 -*-
"""进程级自愈。监督进程不占 GPU，只盯 TTS 子进程。

重启条件（同时满足）：
- nvidia-smi / torch reserved 显示 TTS 进程显存超过 TTS_VRAM_LIMIT_MB
- /health 或 state.json 显示不在合成
- 距离上一句结束已超过 TTS_IDLE_SEC

另外：单次合成超过 TTS_STUCK_SEC 视为卡死，空闲条件不再等待，直接结束子进程。
用户侧的间隙是「这句已经说完、下一句还没来」。冷启动要一两分钟，所以阈值不要设得太低。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

from tts_cosy3.paths import project_root
from tts_cosy3.vram import (
    clear_restart_request,
    idle_sec_required,
    query_process_mb,
    read_state,
    request_restart,
    restart_requested,
    stuck_sec,
    vram_limit_mb,
)

_CREATE_NEW_PROCESS_GROUP = 0x00000200


def _log(msg: str) -> None:
    line = f"[TTS-SUPERVISOR] {msg}"
    print(line, flush=True)
    try:
        from tts_cosy3.paths import log_dir
        with open(log_dir() / "tts-cosy3-supervisor.log", "a", encoding="utf-8") as fh:
            fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")
    except OSError:
        pass


def _child_env() -> dict:
    env = os.environ.copy()
    env["TTS_SUPERVISED"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TTS_PORT", "8810")
    env.setdefault("TTS_DTYPE", "fp16")
    env.setdefault("TTS_INT8", "0")
    return env


def _spawn() -> subprocess.Popen:
    clear_restart_request()
    cmd = [sys.executable, "-m", "tts_cosy3.tts_cosy3_server"]
    flags = _CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    _log("拉起 " + " ".join(cmd))
    return subprocess.Popen(
        cmd,
        cwd=str(project_root()),
        env=_child_env(),
        creationflags=flags,
    )


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    _log(f"结束子进程 pid={proc.pid}")
    proc.terminate()
    try:
        proc.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _health_ok(port: str) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _wait_ready(proc: subprocess.Popen, port: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            _log(f"子进程提前退出 code={proc.returncode}")
            return False
        if _health_ok(port):
            _log(f"健康检查通过 pid={proc.pid}")
            return True
        time.sleep(2)
    return False


def main() -> None:
    port = os.environ.get("TTS_PORT", "8810")
    backoff = 5
    proc = _spawn()
    ready_timeout = float(os.environ.get("TTS_READY_TIMEOUT", "240"))
    if not _wait_ready(proc, port, ready_timeout):
        _stop(proc)
        _log("首次启动未就绪，5 秒后重试")
        time.sleep(5)
        proc = _spawn()
        _wait_ready(proc, port, ready_timeout)

    busy_since = 0.0
    not_ready_since = time.time()
    while True:
        time.sleep(5)
        code = proc.poll()
        if code is not None:
            _log(f"子进程退出 code={code}，{backoff}s 后拉起")
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            proc = _spawn()
            not_ready_since = time.time()
            if _wait_ready(proc, port, ready_timeout):
                backoff = 5
            continue

        state = read_state()
        # 模型还在加载时显存会抬高，不能当成泄漏重启。
        if not state.get("ready"):
            busy_since = 0.0
            if time.time() - not_ready_since > ready_timeout:
                _log("超过就绪时间仍未健康，结束子进程后重拉")
                _stop(proc)
                not_ready_since = time.time()
            continue
        not_ready_since = time.time()
        if bool(state.get("busy")):
            if busy_since <= 0:
                busy_since = time.time()
            elif time.time() - busy_since >= stuck_sec():
                _log(f"合成卡死 {time.time() - busy_since:.0f}s，强制重启")
                _stop(proc)
                busy_since = 0.0
                continue
        else:
            busy_since = 0.0

        smi = query_process_mb(proc.pid)
        observed = smi
        reserved = int(state.get("observed_mb") or -1)
        if reserved > observed:
            observed = reserved
        last_end = float(state.get("last_request_end") or 0)
        idle = time.time() - last_end if last_end else 0
        if (
            not state.get("busy")
            and observed >= 0
            and observed >= vram_limit_mb()
            and idle >= idle_sec_required()
            and not restart_requested()
        ):
            reason = f"显存 {observed}MiB >= {vram_limit_mb()}MiB 且空闲 {idle:.0f}s"
            _log("请求空闲重启: " + reason)
            request_restart(reason)

        if restart_requested():
            grace = float(os.environ.get("TTS_RESTART_GRACE_SEC", "20"))
            _log(f"等待子进程在空闲时自己退出，最多 {grace:.0f}s")
            deadline = time.time() + grace
            while time.time() < deadline and proc.poll() is None:
                time.sleep(1)
            if proc.poll() is None:
                _log("子进程未自己退出，改为结束进程")
                _stop(proc)


if __name__ == "__main__":
    main()
