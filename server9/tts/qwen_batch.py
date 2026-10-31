# -*- coding: utf-8 -*-
"""单进程动态 batch。默认最多 3 句进同一次 generate，播报仍按提交顺序。

=== 新增：调度层，不改 talker / Triton / CUDA Graph ===

HuggingFace ``talker.generate()`` 不把逐 token 的 main_code 交出来，
也不能在一次 generate 中途抽走某一条。静音 guard 挂在每条 BatchSession 上：
调用方若拿到 main_code，用 ``SilenceGuard.feed`` 截断后再送 Code2Wav；
当前 ``generate_voice_clone`` 只返回波形，这条路径不会假装已经拦住码本。

KV：这里的 kv_slot 只是调度槽位。模型侧的 KV 仍是这一次 generate 的整批张量，
没有按 session 池化，也没有同前缀 KV 共享。
"""
from __future__ import annotations

import threading
import time
from collections import deque

# 业务侧静音丢弃后，短时间内同一句的重算只记一次。
_QUIET_WINDOW_SEC = 5.0
_quiet_seen = {}
_quiet_lock = threading.Lock()


def _log(msg: str) -> None:
    try:
        import tts.cosyvoice as cv
        cv._tts_log(msg)
    except Exception:
        print(msg)


def _pcm_peak(pcm: bytes) -> int:
    if not pcm or len(pcm) < 2:
        return 0
    import array
    import sys
    size = len(pcm) - (len(pcm) % 2)
    samples = array.array("h")
    samples.frombytes(pcm[:size])
    if sys.byteorder != "little":
        samples.byteswap()
    peak = 0
    for value in samples:
        mag = -value if value < 0 else value
        if mag > peak:
            peak = mag
    return int(peak)


def _prefix_key(persona) -> tuple:
    tts = getattr(persona, "tts", None)
    return (
        getattr(persona, "id", "") or "",
        getattr(tts, "prompt_wav", "") or "",
        getattr(tts, "prompt_text", "") or "",
    )


def _percentile(values, pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = int(round((pct / 100.0) * (len(ordered) - 1)))
    rank = min(len(ordered) - 1, max(0, rank))
    return float(ordered[rank])


class SchedStats:
    """进程内埋点。写进现有 tts-load.log，不另起监控。"""

    def __init__(self):
        self.native_eos = 0
        self.silence_guard_force_eos = 0
        self.pcm_quiet_retry = 0
        self.token_steps = 0
        self.hidden_sessions = 0
        self.hidden_logged = False
        self.completed = 0
        self.batch_hist = {1: 0, 2: 0, 3: 0}
        self.latencies_ms = deque(maxlen=256)
        self.started = time.monotonic()
        self._lock = threading.Lock()

    def add_latency(self, elapsed_ms: float) -> None:
        with self._lock:
            self.latencies_ms.append(float(elapsed_ms))
            self.completed += 1

    def note_batch(self, size: int) -> None:
        with self._lock:
            key = int(size)
            if key in self.batch_hist:
                self.batch_hist[key] += 1
            else:
                self.batch_hist[key] = self.batch_hist.get(key, 0) + 1

    def note_end(self, reason: str) -> None:
        with self._lock:
            if reason == "native_eos":
                self.native_eos += 1
            elif reason == "silence_guard_force_eos":
                self.silence_guard_force_eos += 1

    def note_tokens(self, n: int) -> None:
        with self._lock:
            self.token_steps += int(n)

    def note_quiet(self) -> int:
        with self._lock:
            self.pcm_quiet_retry += 1
            return self.pcm_quiet_retry

    def note_hidden(self) -> bool:
        with self._lock:
            self.hidden_sessions += 1
            first = not self.hidden_logged
            self.hidden_logged = True
            return first

    def snapshot(self) -> dict:
        with self._lock:
            lats = list(self.latencies_ms)
            hist = dict(self.batch_hist)
            done = self.completed
            elapsed = max(1e-6, time.monotonic() - self.started)
            return {
                "native_eos": self.native_eos,
                "silence_guard_force_eos": self.silence_guard_force_eos,
                "pcm_quiet_retry": self.pcm_quiet_retry,
                "token_steps": self.token_steps,
                "hidden_sessions": self.hidden_sessions,
                "batch_hist": hist,
                "p50_ms": _percentile(lats, 50),
                "p95_ms": _percentile(lats, 95),
                "throughput": done / elapsed,
                "completed": done,
            }


stats = SchedStats()


def note_pcm_quiet_retry(where: str, key: str = "", book: SchedStats = None) -> None:
    """业务侧静音重算。同一句 5 秒内只记一次，避免调度丢弃和播放重算各记一笔。"""
    now = time.monotonic()
    with _quiet_lock:
        if key:
            prev = _quiet_seen.get(key)
            if prev is not None and now - prev < _QUIET_WINDOW_SEC:
                return
            _quiet_seen[key] = now
            stale = [item for item, ts in _quiet_seen.items() if now - ts > _QUIET_WINDOW_SEC]
            for item in stale:
                _quiet_seen.pop(item, None)
    target = book if book is not None else stats
    n = target.note_quiet()
    _log("[TTS] pcm_quiet_retry where=%s n=%s text=%s" % (where, n, (key or "")[:40]))


class SilenceGuard:
    """一条请求自己的连续静音计数。不要拿去给别的 session 共用。

    原生 EOS 先判断。静音只在非 EOS 的 main_code 上累加，到阈值才兜底。
    """

    def __init__(self, silence_codes=(), max_cont: int = 8, stats_obj: SchedStats = None):
        self.silence_codes = frozenset(int(c) for c in (silence_codes or ()))
        self.max_cont = max(2, int(max_cont))
        self.stats = stats_obj
        self.reset()

    def reset(self) -> None:
        """新请求必须清掉这条 session 的计数，不能沿用上一句。"""
        self.count = 0
        self.stopped = False
        self.reason = ""
        self.native_eos = False
        self.forced = False

    def observe(self, main_code: int, eos_id=None) -> str:
        """采样得到主码本之后调用。辅助码本不进这里。

        返回 speech / silence / native_eos / silence_guard_force_eos。
        已经停止后再观察，不再累加，也不再改原因。
        """
        if self.stopped:
            return self.reason or "stopped"
        code = int(main_code)
        if eos_id is not None and code == int(eos_id):
            self.native_eos = True
            self.stopped = True
            self.reason = "native_eos"
            if self.stats is not None:
                self.stats.note_end("native_eos")
            return self.reason
        if code in self.silence_codes:
            self.count += 1
            if self.count >= self.max_cont:
                self.forced = True
                self.stopped = True
                self.reason = "silence_guard_force_eos"
                if self.stats is not None:
                    self.stats.note_end("silence_guard_force_eos")
                return self.reason
            return "silence"
        self.count = 0
        return "speech"

    def feed(self, main_codes, eos_id=None) -> tuple:
        """新请求入口：先 reset，再逐个 main_code。

        返回 (keep_len, reason)。keep_len 之前的码本可以送 Code2Wav，
        触发兜底的那段静音以及它后面的码本都不送。
        原生 EOS 与模型现有切分一致：结束符本身不留在送去解码的长度里。
        """
        self.reset()
        if self.stats is not None:
            self.stats.note_tokens(len(main_codes or []))
        for index, code in enumerate(main_codes or []):
            step = self.observe(code, eos_id=eos_id)
            if step == "native_eos":
                return index, "native_eos"
            if step == "silence_guard_force_eos":
                return index + 1 - self.count, "silence_guard_force_eos"
        if self.native_eos:
            return 0, "native_eos"
        return len(main_codes or []), "max_len"


class BatchSession:
    """一条待合成句子。guard、结束状态、逻辑槽位都挂在这个对象上。"""

    def __init__(self, text: str, speaker: str, persona, cancel_ev, seq: int,
                 silence_codes=(), max_cont: int = 8, stats_obj: SchedStats = None):
        self.text = text or ""
        self.speaker = speaker or ""
        self.persona = persona
        self.cancel_ev = cancel_ev
        self.seq = int(seq)
        self.prefix_key = _prefix_key(persona)
        self.guard = SilenceGuard(silence_codes, max_cont, stats_obj=stats_obj)
        self.guard.reset()
        self.future = None
        self.kv_slot = None
        self.inflight = False
        self.done = False
        self.end_reason = ""
        self.t0 = time.perf_counter()
        self.tokens_hidden = True

    def note_hidden(self, book: SchedStats) -> None:
        self.tokens_hidden = True
        if book.note_hidden():
            _log("[TTS] silence_guard unobservable: "
                 "generate_voice_clone 不返回逐 token main_code，未按码本截断")


def playback_ready(sessions) -> list:
    """按提交顺序取已经结束的前缀。中间有一条没结束就不能先播后面的。"""
    ready = []
    for sess in sessions:
        if not getattr(sess, "done", False):
            break
        ready.append(sess)
    return ready


class QwenBatchScheduler:
    """单进程。GPU 线程只做 generate；文本整理走单独的 1 线程，不拿 qwen._lock。"""

    kind = "batch"

    def __init__(self, generate_fn=None, max_batch: int = 3, wait_ms: int = 8,
                 silence_codes=(), max_cont: int = 8, stats_obj: SchedStats = None,
                 enable_cache: bool = True):
        # 硬顶 3。再大一次 generate 的显存和 RoPE 批次都不好收。
        self.max_batch = max(1, min(3, int(max_batch)))
        self.wait_s = max(0.0, float(wait_ms) / 1000.0)
        self.silence_codes = frozenset(int(c) for c in (silence_codes or ()))
        self.max_cont = max(2, int(max_cont))
        self.stats = stats_obj if stats_obj is not None else stats
        self.generate_fn = generate_fn
        self.enable_cache = bool(enable_cache)
        self._ready = deque()
        self._free_slots = list(range(self.max_batch))
        self._cv = threading.Condition()
        self._life = threading.Lock()
        self._seq = 0
        self._stop = False
        self._thread = None
        self._prep = None

    def window(self) -> int:
        return self.max_batch

    def ensure_started(self) -> None:
        with self._life:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop = False
            self._thread = threading.Thread(
                target=self._loop, name="qwen-batch", daemon=True)
            self._thread.start()
        _log("[TTS] qwen sched=batch max=%s wait_ms=%s silence=%s max_cont=%s" % (
            self.max_batch, int(self.wait_s * 1000),
            ",".join(str(c) for c in sorted(self.silence_codes)) or "-",
            self.max_cont))

    def close(self) -> None:
        self._stop = True
        with self._cv:
            self._cv.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
        prep = self._prep
        self._prep = None
        if prep is not None:
            prep.shutdown(wait=False)

    def _prep_pool(self):
        if self._prep is None:
            from concurrent.futures import ThreadPoolExecutor
            # 只有 1 个线程。这里不调用 synthesize，也不拿 qwen._lock。
            self._prep = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-prep")
        return self._prep

    def submit(self, text: str, speaker: str, persona, cancel_ev):
        """返回 Future[(pcm, sr)]。和多进程池的 submit 用法相同。"""
        import concurrent.futures
        outer = concurrent.futures.Future()
        with self._cv:
            self._seq += 1
            seq = self._seq
        sess = BatchSession(
            text, speaker, persona, cancel_ev, seq,
            self.silence_codes, self.max_cont, stats_obj=self.stats)
        sess.future = outer

        def _on_prep(done):
            try:
                done.result()
            except Exception as exc:
                if not outer.done():
                    outer.set_exception(exc)
                return
            if not sess.text:
                if not outer.done():
                    outer.set_result((b"", 24000))
                return
            with self._cv:
                self._ready.append(sess)
                self._cv.notify()

        def _prep():
            # === 新增：CPU 预处理。禁止在这里调用 qwen._generate / 拿 _lock。 ===
            sess.text = " ".join((sess.text or "").split())
            sess.prefix_key = _prefix_key(sess.persona)
            sess.guard.reset()

        self._prep_pool().submit(_prep).add_done_callback(_on_prep)
        return outer

    def _loop(self) -> None:
        while not self._stop:
            batch, cancelled = self._take_batch(block=True)
            for sess in cancelled:
                self._complete_cancel(sess)
            if not batch:
                continue
            self._execute(batch)

    def _take_batch(self, block: bool) -> tuple:
        """凑批。已经在队列里的句子一起走；不够 3 句时最多再等 wait_ms，到点就走。

        不同参考音前缀不放进同一次 generate，也不从后面跳句，避免播放乱序。
        """
        cancelled = []
        with self._cv:
            if block:
                while not self._ready and not self._stop:
                    self._cv.wait()
            if not self._ready:
                return [], cancelled
            if block and self.wait_s > 0 and len(self._ready) < self.max_batch:
                deadline = time.monotonic() + self.wait_s
                while len(self._ready) < self.max_batch and not self._stop:
                    remain = deadline - time.monotonic()
                    if remain <= 0:
                        break
                    self._cv.wait(remain)
            batch = []
            while self._ready and len(batch) < self.max_batch:
                sess = self._ready[0]
                if batch and sess.prefix_key != batch[0].prefix_key:
                    break
                self._ready.popleft()
                if sess.future is not None and sess.future.done():
                    continue
                if sess.cancel_ev is not None and sess.cancel_ev.is_set():
                    cancelled.append(sess)
                    continue
                if not self._free_slots:
                    self._ready.appendleft(sess)
                    break
                sess.kv_slot = self._free_slots.pop()
                sess.inflight = True
                batch.append(sess)
            return batch, cancelled

    def _call_generate(self, batch, cancel_ev):
        if self.generate_fn is not None:
            return self.generate_fn(batch, cancel_ev)
        from tts.qwen import synthesize_batch
        return synthesize_batch(
            [sess.text for sess in batch],
            persona=batch[0].persona,
            cancel_ev=cancel_ev,
        )

    def _execute(self, batch) -> None:
        if not batch:
            return
        t0 = time.perf_counter()
        error = None
        rows = None
        try:
            rows = self._call_generate(batch, batch[0].cancel_ev)
        except Exception as exc:
            error = exc
        wall_ms = (time.perf_counter() - t0) * 1000.0
        try:
            if error is not None:
                for sess in batch:
                    self._fail(sess, error)
            else:
                rows = list(rows or [])
                for index, sess in enumerate(batch):
                    if index >= len(rows):
                        self._fail(sess, RuntimeError("qwen batch short"))
                        continue
                    pcm, sr = rows[index]
                    self._finish_pcm(sess, pcm, sr)
        finally:
            with self._cv:
                for sess in batch:
                    if sess.kv_slot is not None:
                        self._free_slots.append(sess.kv_slot)
                        sess.kv_slot = None
                    sess.inflight = False
                    sess.done = True
                self._cv.notify_all()
            self.stats.note_batch(len(batch))
            for sess in batch:
                self.stats.add_latency((time.perf_counter() - sess.t0) * 1000.0)
            self._log_round(batch, wall_ms)

    def _finish_pcm(self, sess: BatchSession, pcm, sr) -> None:
        # 外面没有 main_code。guard 留在 session 上，这一句不记强制 EOS。
        sess.note_hidden(self.stats)
        sess.end_reason = "pcm"
        peak = _pcm_peak(pcm or b"")
        if not pcm or peak < 800:
            note_pcm_quiet_retry("sched", key=(sess.text or "")[:40], book=self.stats)
            sess.end_reason = "quiet"
            if sess.future is not None and not sess.future.done():
                sess.future.set_exception(RuntimeError("quiet pcm peak=%s" % peak))
            return
        self._store(sess, pcm, int(sr or 24000))
        if sess.future is not None and not sess.future.done():
            sess.future.set_result((pcm, int(sr or 24000)))

    def _fail(self, sess: BatchSession, exc: Exception) -> None:
        sess.end_reason = "error"
        sess.note_hidden(self.stats)
        message = str(exc)
        if "quiet" in message:
            note_pcm_quiet_retry("sched_error", key=(sess.text or "")[:40], book=self.stats)
        if sess.future is not None and not sess.future.done():
            sess.future.set_exception(exc)

    def _complete_cancel(self, sess: BatchSession) -> None:
        sess.done = True
        sess.end_reason = "cancel"
        if sess.future is not None and not sess.future.done():
            sess.future.set_result((b"", 24000))

    def _store(self, sess: BatchSession, pcm: bytes, sr: int) -> None:
        if not self.enable_cache or not pcm:
            return
        try:
            import tts.cosyvoice as cv
            if cv.audio_cache is None or cv._pcm_is_silent(pcm):
                return
            pid = getattr(sess.persona, "id", "") or ""
            cv.audio_cache.put(
                sess.text, sess.speaker, (pcm, sr), fmt="pcm", persona_id=pid)
        except Exception:
            return

    def _log_round(self, batch, wall_ms: float) -> None:
        snap = self.stats.snapshot()
        hist = snap["batch_hist"]
        slots = ",".join(str(sess.seq) for sess in batch)
        guard = "on" if snap["token_steps"] else "off"
        _log("[TTS] qwen sched n=%s wall_ms=%.0f slots=%s kv_share=0 token_guard=%s "
             "hist=%s:%s,%s:%s,%s:%s native_eos=%s silence_guard_force_eos=%s "
             "pcm_quiet_retry=%s p50_ms=%.0f p95_ms=%.0f tput=%.2f/s" % (
                 len(batch), wall_ms, slots, guard,
                 1, hist.get(1, 0), 2, hist.get(2, 0), 3, hist.get(3, 0),
                 snap["native_eos"], snap["silence_guard_force_eos"],
                 snap["pcm_quiet_retry"], snap["p50_ms"], snap["p95_ms"],
                 snap["throughput"]))


class _LazyScheduler:
    """导入模块时不建线程。第一次 ensure_started / submit 再创建。"""

    kind = "batch"

    def __getattr__(self, name):
        return getattr(get_scheduler(), name)


_scheduler = None
_scheduler_lock = threading.Lock()


def get_scheduler() -> QwenBatchScheduler:
    global _scheduler
    with _scheduler_lock:
        if _scheduler is None:
            from config import (
                MAX_CONT_SILENCE,
                QWEN_TTS_BATCH_MAX,
                QWEN_TTS_BATCH_WAIT_MS,
                SILENCE_CODES,
            )
            _scheduler = QwenBatchScheduler(
                max_batch=QWEN_TTS_BATCH_MAX,
                wait_ms=QWEN_TTS_BATCH_WAIT_MS,
                silence_codes=SILENCE_CODES,
                max_cont=MAX_CONT_SILENCE,
                stats_obj=stats,
            )
        return _scheduler


scheduler = _LazyScheduler()
