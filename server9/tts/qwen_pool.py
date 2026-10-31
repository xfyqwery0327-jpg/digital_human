# -*- coding: utf-8 -*-
"""Qwen 多路合成。主进程保留已加载的模型作为第 1 路，其余路各自一个子进程。

每路同时只算一句。调用方按句序取回 PCM，再按原顺序推给客户端。
子进程加载失败时丢掉那一路，主进程这一路继续用。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path

_SERVER9 = Path(__file__).resolve().parents[1]
_MIN_FREE_BYTES = 8 * 1024 * 1024 * 1024


def _log(msg: str) -> None:
    try:
        import tts.cosyvoice as cv
        cv._tts_log(msg)
    except Exception:
        print(msg)


def _pcm_peak(pcm: bytes) -> int:
    if not pcm:
        return 0
    import numpy as np
    samples = np.frombuffer(pcm, dtype="<i2")
    if samples.size == 0:
        return 0
    return int(np.max(np.abs(samples.astype(np.int32))))


def _free_vram() -> int:
    try:
        import torch
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    except Exception:
        return 0


class _Lane:
    def __init__(self, index: int):
        self.index = index
        self.inflight = 0
        self.alive = True

    def submit(self, text: str, persona, cancel_ev):
        raise NotImplementedError


class InProcLane(_Lane):
    """主进程里已经驻留的那份模型。"""

    def submit(self, text: str, persona, cancel_ev):
        import concurrent.futures
        fut = concurrent.futures.Future()

        def run():
            try:
                if cancel_ev is not None and cancel_ev.is_set():
                    fut.set_result((b"", 24000))
                    return
                from tts.qwen import synthesize_pcm
                fut.set_result(synthesize_pcm(text, persona=persona, cancel_ev=cancel_ev))
            except Exception as e:
                fut.set_exception(e)

        threading.Thread(target=run, name="qwen-lane-0", daemon=True).start()
        return fut


class ProcLane(_Lane):
    """独立 Python 进程，各自持有一份模型。"""

    def __init__(self, index: int):
        super().__init__(index)
        self.ready = threading.Event()
        self.failed = False
        self.pending = {}
        self._write_lock = threading.Lock()
        env = os.environ.copy()
        env["QWEN_TTS_WORKER"] = "1"
        env["QWEN_TTS_LANE"] = str(index)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        script = str(_SERVER9 / "tts" / "qwen_worker.py")
        import concurrent.futures
        self._futures = concurrent.futures
        self.proc = subprocess.Popen(
            [sys.executable, "-u", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            cwd=str(_SERVER9),
            env=env,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._read, name="qwen-lane-%s" % index, daemon=True)
        self._reader.start()

    def wait_ready(self, timeout: float) -> bool:
        self.ready.wait(timeout)
        return self.alive and not self.failed

    def close(self) -> None:
        self.alive = False
        proc = getattr(self, "proc", None)
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass

    def _fail_pending(self, exc: Exception) -> None:
        pending = list(self.pending.values())
        self.pending.clear()
        for fut in pending:
            if not fut.done():
                fut.set_exception(exc)

    def _read(self) -> None:
        try:
            stdout = self.proc.stdout
            if stdout is None:
                raise RuntimeError("worker stdout missing")
            for line in stdout:
                text = (line or "").strip()
                if not text:
                    continue
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if "ready" in msg and "id" not in msg:
                    if not msg.get("ready"):
                        self.failed = True
                        _log("[TTS] qwen worker lane=%s FAIL %s" % (
                            self.index, msg.get("error") or "not ready"))
                    self.ready.set()
                    continue
                fut = self.pending.pop(str(msg.get("id") or ""), None)
                if fut is None or fut.done():
                    continue
                if msg.get("error"):
                    fut.set_exception(RuntimeError(str(msg["error"])))
                    continue
                import base64
                pcm = base64.b64decode(msg.get("pcm") or "")
                fut.set_result((pcm, int(msg.get("sr") or 24000)))
        except Exception as e:
            self.failed = True
            if not self.ready.is_set():
                self.ready.set()
            self._fail_pending(e)
            return
        self.alive = False
        if not self.ready.is_set():
            self.failed = True
            self.ready.set()
        self._fail_pending(RuntimeError("qwen worker lane=%s exited" % self.index))

    def submit(self, text: str, persona, cancel_ev):
        del cancel_ev
        if not self.alive or self.proc.poll() is not None:
            raise RuntimeError("qwen worker lane=%s dead" % self.index)
        tts = getattr(persona, "tts", None)
        job_id = uuid.uuid4().hex
        job = {
            "id": job_id,
            "text": text,
            "persona_id": getattr(persona, "id", "") or "",
            "prompt_wav": getattr(tts, "prompt_wav", "") or "",
            "prompt_text": getattr(tts, "prompt_text", "") or "",
        }
        fut = self._futures.Future()
        self.pending[job_id] = fut
        line = json.dumps(job, ensure_ascii=False) + "\n"
        try:
            with self._write_lock:
                self.proc.stdin.write(line)
                self.proc.stdin.flush()
        except Exception:
            self.pending.pop(job_id, None)
            raise
        return fut


class QwenPool:
    kind = "procs"

    def __init__(self):
        self.lanes = []
        self._booted = False
        self._lock = threading.Lock()

    def window(self) -> int:
        alive = [lane for lane in self.lanes if lane.alive]
        return max(1, len(alive))

    def ensure_started(self) -> None:
        """主进程模型已经预热之后调用。子进程一个一个起来，避免三份同时编译把显存打满。"""
        with self._lock:
            if self._booted:
                return
            self._booted = True
        from config import QWEN_TTS_WORKERS
        lanes = [InProcLane(0)]
        extra = max(0, int(QWEN_TTS_WORKERS) - 1)
        for index in range(1, extra + 1):
            free = _free_vram()
            if free and free < _MIN_FREE_BYTES:
                _log("[TTS] qwen worker lane=%s skip free_vram=%sMB" % (index, free // (1024 * 1024)))
                break
            lane = ProcLane(index)
            if not lane.wait_ready(300):
                lane.close()
                _log("[TTS] qwen worker lane=%s not ready, stop spawning" % index)
                break
            lanes.append(lane)
            _log("[TTS] qwen worker lane=%s up" % index)
        self.lanes = lanes
        _log("[TTS] qwen workers ready n=%s" % len(lanes))

    def submit(self, text: str, speaker: str, persona, cancel_ev):
        """返回 Future[(pcm, sr)]。成功的 PCM 会先写入父进程 audio_cache。"""
        import concurrent.futures
        outer = concurrent.futures.Future()
        with self._lock:
            alive = [lane for lane in self.lanes if lane.alive]
            if not alive:
                alive = [InProcLane(0)]
                self.lanes = alive
            lane = min(alive, key=lambda item: item.inflight)
            lane.inflight += 1
        _log("[TTS] qwen pool submit lane=%s chars=%s text=%s" % (
            lane.index, len(text or ""), (text or "")[:40]))

        def _finish(src):
            with self._lock:
                lane.inflight = max(0, lane.inflight - 1)
            try:
                pcm, sr = src.result()
            except Exception as e:
                if "quiet" in str(e):
                    from tts.qwen_batch import note_pcm_quiet_retry
                    note_pcm_quiet_retry("pool", key=(text or "")[:40])
                if not outer.done():
                    outer.set_exception(e)
                return
            peak = _pcm_peak(pcm)
            if pcm and peak < 800:
                _log("[TTS] qwen pool quiet lane=%s peak=%s text=%s" % (
                    lane.index, peak, (text or "")[:40]))
                from tts.qwen_batch import note_pcm_quiet_retry
                note_pcm_quiet_retry("pool", key=(text or "")[:40])
                if not outer.done():
                    outer.set_exception(RuntimeError("quiet pcm peak=%s" % peak))
                return
            self._cache(text, speaker, persona, pcm, sr)
            if not outer.done():
                outer.set_result((pcm, sr))

        try:
            inner = lane.submit(text, persona, cancel_ev)
        except Exception as e:
            with self._lock:
                lane.inflight = max(0, lane.inflight - 1)
            outer.set_exception(e)
            return outer
        inner.add_done_callback(_finish)
        return outer

    def _cache(self, text: str, speaker: str, persona, pcm: bytes, sr: int) -> None:
        if not pcm:
            return
        try:
            import tts.cosyvoice as cv
            if cv.audio_cache is None or cv._pcm_is_silent(pcm):
                return
            pid = getattr(persona, "id", "") or ""
            cv.audio_cache.put(text, speaker, (pcm, int(sr or 24000)), fmt="pcm", persona_id=pid)
        except Exception:
            return


pool = QwenPool()
