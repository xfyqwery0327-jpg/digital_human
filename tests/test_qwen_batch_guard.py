# -*- coding: utf-8 -*-
"""静音 guard 与单进程动态 batch。不加载模型，不跑 GPU。"""
from __future__ import annotations

import array
import concurrent.futures
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER9 = ROOT / "server9"
if str(SERVER9) not in sys.path:
    sys.path.insert(0, str(SERVER9))

from tts.qwen_batch import (  # noqa: E402
    BatchSession,
    QwenBatchScheduler,
    SchedStats,
    SilenceGuard,
    note_pcm_quiet_retry,
    playback_ready,
)


class _Persona:
    def __init__(self, pid, wav):
        self.id = pid

        class _Tts:
            prompt_wav = wav
            prompt_text = "ref"

        self.tts = _Tts()


def _loud(n=200) -> bytes:
    return array.array("h", [3000] * n).tobytes()


def _session(text, pid="monet", wav="a.wav", stats=None, cancel=None, seq=1):
    persona = _Persona(pid, wav)
    sess = BatchSession(
        text, "spk", persona, cancel, seq,
        silence_codes=(1, 2), max_cont=3, stats_obj=stats)
    sess.future = concurrent.futures.Future()
    return sess


def _sched(generate, stats, max_batch=3):
    return QwenBatchScheduler(
        generate_fn=generate,
        max_batch=max_batch,
        wait_ms=0,
        silence_codes=(1, 2),
        max_cont=3,
        stats_obj=stats,
        enable_cache=False,
    )


def test_guards_do_not_share_counters():
    book = SchedStats()
    left = SilenceGuard((1,), max_cont=3, stats_obj=book)
    right = SilenceGuard((1,), max_cont=3, stats_obj=book)
    assert left.feed([1, 1, 1], eos_id=9) == (0, "silence_guard_force_eos")
    assert left.count == 3 and left.forced
    assert right.count == 0 and not right.forced
    assert right.feed([5], eos_id=9) == (1, "max_len")
    assert book.silence_guard_force_eos == 1
    assert book.native_eos == 0


def test_native_eos_beats_silence_code():
    book = SchedStats()
    guard = SilenceGuard((7, 1), max_cont=2, stats_obj=book)
    # 7 既在静音表里，又是 EOS。必须走原生结束，不能记成兜底。
    assert guard.feed([5, 7], eos_id=7) == (1, "native_eos")
    assert guard.native_eos and not guard.forced
    assert book.native_eos == 1
    assert book.silence_guard_force_eos == 0
    guard.observe(7, eos_id=7)
    assert book.native_eos == 1


def test_silence_run_reaches_threshold_and_drops_those_codes():
    guard = SilenceGuard((1,), max_cont=3)
    mains = [5, 5, 1, 1, 1, 1]
    aux = [9, 9, 8, 8, 8, 8]
    keep, reason = guard.feed(mains, eos_id=99)
    assert reason == "silence_guard_force_eos"
    assert keep == 2
    kept = [pair for pair in zip(mains, aux)][:keep]
    assert kept == [(5, 9), (5, 9)]
    assert guard.observe(1, eos_id=99) == "silence_guard_force_eos"
    assert guard.count == 3


def test_nonsilence_resets_count():
    guard = SilenceGuard((1,), max_cont=3)
    assert guard.observe(1) == "silence"
    assert guard.observe(1) == "silence"
    assert guard.observe(5) == "speech"
    assert guard.count == 0
    assert guard.feed([1, 1, 5, 1], eos_id=99) == (4, "max_len")
    assert guard.count == 1
    assert not guard.forced


def test_new_request_resets_guard():
    guard = SilenceGuard((1,), max_cont=3)
    assert guard.feed([1, 1, 1], eos_id=99)[1] == "silence_guard_force_eos"
    guard.reset()
    assert guard.count == 0 and not guard.stopped and not guard.forced
    assert guard.feed([5, 5], eos_id=99) == (2, "max_len")
    fresh = BatchSession("下一句", "spk", _Persona("monet", "a.wav"), None, 2,
                         silence_codes=(1,), max_cont=3)
    assert fresh.guard.count == 0 and not fresh.guard.stopped


def test_one_session_end_does_not_touch_the_other():
    book = SchedStats()
    seen = {}

    def generate(batch, cancel_ev):
        del cancel_ev
        seen["slots"] = [sess.kv_slot for sess in batch]
        seen["ids"] = [id(sess.guard) for sess in batch]
        batch[0].guard.feed([1, 1, 1], eos_id=99)
        assert batch[1].guard.count == 0
        assert batch[0].kv_slot != batch[1].kv_slot
        return [(_loud(), 24000), (_loud(), 24000)]

    sched = _sched(generate, book)
    try:
        first = _session("甲", stats=book, seq=1)
        second = _session("乙", stats=book, seq=2)
        third = _session("丙", stats=book, seq=3)
        sched._ready.append(first)
        sched._ready.append(second)
        sched._ready.append(third)
        batch, cancelled = sched._take_batch(block=False)
        assert cancelled == []
        assert [sess.text for sess in batch] == ["甲", "乙", "丙"]
        assert len(sched._free_slots) == 0
        sched._execute(batch)
        assert first.guard.forced and first.guard.reason == "silence_guard_force_eos"
        assert second.guard.count == 0 and not second.guard.forced
        assert first.kv_slot is None and second.kv_slot is None
        assert sorted(sched._free_slots) == [0, 1, 2]
        assert first.done and second.done and third.done
        assert not third.guard.forced
        assert len(seen["ids"]) == 3
        pcm, sr = second.future.result(timeout=1)
        assert sr == 24000 and pcm
        assert first.future.done()
    finally:
        sched.close()


def test_batch_cap_prefix_and_cancel_are_independent():
    book = SchedStats()
    calls = []

    def generate(batch, cancel_ev):
        del cancel_ev
        calls.append([sess.text for sess in batch])
        if batch[0].text == "炸":
            raise RuntimeError("boom")
        return [(_loud(), 24000) for _ in batch]

    sched = _sched(generate, book)
    try:
        rows = [
            _session("甲", stats=book, seq=1),
            _session("乙", pid="other", wav="b.wav", stats=book, seq=2),
            _session("丙", stats=book, seq=3),
            _session("丁", stats=book, seq=4),
            _session("戊", stats=book, seq=5),
        ]
        cancel = threading.Event()
        cancel.set()
        rows.append(_session("已取消", stats=book, seq=6, cancel=cancel))
        for sess in rows:
            sched._ready.append(sess)
        # 不同前缀不能跳句并进同一批，所以第一批只有「甲」。
        batch, cancelled = sched._take_batch(block=False)
        assert [sess.text for sess in batch] == ["甲"]
        assert cancelled == []
        sched._execute(batch)
        batch, cancelled = sched._take_batch(block=False)
        assert [sess.text for sess in batch] == ["乙"]
        sched._execute(batch)
        # 同前缀最多 3 条。队列里还有丙丁戊和已取消。
        batch, cancelled = sched._take_batch(block=False)
        assert [sess.text for sess in batch] == ["丙", "丁", "戊"]
        assert [sess.text for sess in cancelled] == []
        assert len(sched._ready) == 1
        sched._execute(batch)
        batch, cancelled = sched._take_batch(block=False)
        assert batch == []
        assert [sess.text for sess in cancelled] == ["已取消"]
        for sess in cancelled:
            sched._complete_cancel(sess)
        assert rows[-1].future.result(timeout=1) == (b"", 24000)
        assert rows[-1].done
        assert "丙" not in [sess.text for sess in batch]
        assert max(len(call) for call in calls) <= 3
        assert calls[0] == ["甲"]
        assert sorted(sched._free_slots) == [0, 1, 2]
    finally:
        sched.close()


def test_generate_error_frees_slots_for_the_next_session():
    book = SchedStats()

    def generate(batch, cancel_ev):
        del cancel_ev
        raise RuntimeError("boom")

    sched = _sched(generate, book, max_batch=2)
    try:
        bad = _session("炸", stats=book, seq=1)
        waiting = _session("后", stats=book, seq=2)
        sched._ready.append(bad)
        batch, _cancelled = sched._take_batch(block=False)
        sched._ready.append(waiting)
        sched._execute(batch)
        assert bad.done and bad.future.exception() is not None
        assert not waiting.done
        assert waiting.guard.count == 0
        nxt, _cancelled = sched._take_batch(block=False)
        assert [sess.text for sess in nxt] == ["后"]
        assert bad.kv_slot is None
    finally:
        sched.close()


def test_playback_stays_in_submit_order():
    book = SchedStats()
    first = _session("甲", stats=book, seq=1)
    second = _session("乙", stats=book, seq=2)
    second.done = True
    assert playback_ready([first, second]) == []
    first.done = True
    assert [sess.seq for sess in playback_ready([first, second])] == [1, 2]


def test_quiet_pcm_does_not_reset_the_other_guard():
    book = SchedStats()

    def generate(batch, cancel_ev):
        del cancel_ev
        return [(b"", 24000), (_loud(), 24000)]

    sched = _sched(generate, book)
    try:
        quiet = _session("静", stats=book, seq=1)
        loud = _session("响", stats=book, seq=2)
        loud.guard.observe(1)
        sched._ready.extend([quiet, loud])
        batch, _cancelled = sched._take_batch(block=False)
        sched._execute(batch)
        assert quiet.future.exception() is not None
        assert loud.future.result(timeout=1)[0]
        assert loud.guard.count == 1
        assert book.pcm_quiet_retry == 1
        assert book.silence_guard_force_eos == 0
    finally:
        sched.close()


def test_submit_strips_text_without_waiting_for_a_full_batch():
    book = SchedStats()
    calls = []

    def generate(batch, cancel_ev):
        del cancel_ev
        calls.append([sess.text for sess in batch])
        return [(_loud(), 24000) for _ in batch]

    sched = _sched(generate, book)
    try:
        sched.ensure_started()
        fut = sched.submit("  你好  ", "spk", _Persona("monet", "a.wav"), None)
        pcm, sr = fut.result(timeout=2)
        assert pcm and sr == 24000
        assert calls == [["你好"]]
    finally:
        sched.close()


def test_quiet_retry_dedup_same_sentence():
    book = SchedStats()
    note_pcm_quiet_retry("piece", key="同一句", book=book)
    note_pcm_quiet_retry("sched", key="同一句", book=book)
    assert book.pcm_quiet_retry == 1


def test_scheduler_defaults_are_single_process_batch():
    text = (SERVER9 / "config.py").read_text(encoding="utf-8")
    assert 'or "batch"' in text
    assert '_env_int("QWEN_TTS_BATCH_MAX", 3)' in text
    assert '_env_int("QWEN_TTS_BATCH_WAIT_MS", 8)' in text
    assert '_env_int("QWEN_TTS_MAX_CONT_SILENCE", 8)' in text
    batch_src = (SERVER9 / "tts" / "qwen_batch.py").read_text(encoding="utf-8")
    assert "torch.compile" not in batch_src
    assert ".capture(" not in batch_src
    from config import QWEN_TTS_BATCH_MAX, QWEN_TTS_SCHED
    assert QWEN_TTS_SCHED in {"batch", "procs"}
    assert 1 <= QWEN_TTS_BATCH_MAX <= 3
