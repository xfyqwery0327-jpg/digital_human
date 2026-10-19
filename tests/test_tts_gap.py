# -*- coding: utf-8 -*-
"""CosyVoice3 / Qwen3-TTS 句间空档：拼句、裁静音、批量预取、独立进程边收边推。不加载模型。"""
from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SERVER9 = ROOT / "server9"
if str(SERVER9) not in sys.path:
    sys.path.insert(0, str(SERVER9))


def _speech(sr: int, lead: float, voiced: float, tail: float) -> bytes:
    t = np.arange(int(sr * voiced)) / float(sr)
    tone = (0.5 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    pad_a = np.zeros(int(sr * lead), dtype="<i2")
    pad_b = np.zeros(int(sr * tail), dtype="<i2")
    return np.concatenate([pad_a, tone, pad_b]).tobytes()


def test_trim_keeps_speech_and_drops_long_silence():
    from tts.pcmtrim import trim_pcm16
    sr = 24000
    pcm = _speech(sr, 0.6, 1.0, 0.8)
    out = trim_pcm16(pcm, sr, head_ms=40, tail_ms=160)
    dur = len(out) / 2.0 / sr
    assert 1.15 <= dur <= 1.25, dur
    voiced = np.frombuffer(_speech(sr, 0.0, 1.0, 0.0), dtype="<i2")
    got = np.frombuffer(out, dtype="<i2")
    assert int(np.abs(got).max()) == int(np.abs(voiced).max())


def test_trim_noop_on_silence_or_zero_margin():
    from tts.pcmtrim import trim_pcm16
    sr = 24000
    silent = b"\x00\x00" * sr
    assert trim_pcm16(silent, sr) == silent
    pcm = _speech(sr, 0.5, 0.5, 0.5)
    assert trim_pcm16(pcm, sr, head_ms=0, tail_ms=0) == pcm
    head_only = trim_pcm16(pcm, sr, head_ms=0, tail_ms=100)
    assert head_only[:2000] == pcm[:2000]
    assert len(head_only) < len(pcm)


def test_group_chunks_merges_up_to_cap():
    from tts.speaktext import group_speak_chunks
    parts = ["我是莫奈，", "今天陪你看画，", "先说睡莲。", "再说日出。"]
    assert group_speak_chunks(parts, 0) == parts
    merged = group_speak_chunks(parts, 14)
    assert "".join(merged) == "".join(parts)
    assert all(len(m) <= 14 for m in merged)
    assert merged[0] == "我是莫奈，今天陪你看画，"
    long_one = "字" * 50 + "。"
    assert group_speak_chunks(["你好，", long_one], 40) == ["你好，", long_one]


def test_planner_sends_single_clause_until_measured():
    from tts.planner import UnitPlanner
    p = UnitPlanner(40)
    backlog = ["我是莫奈，", "今天陪你看画，", "先说睡莲。"]
    assert p.take(backlog, ahead_sec=10.0, max_units=8) == ["我是莫奈，"]
    assert backlog == ["今天陪你看画，", "先说睡莲。"]


def test_planner_cap_tracks_client_buffer():
    from tts.planner import UnitPlanner
    p = UnitPlanner(40, margin_sec=0.3)
    # 5 字念 1.25s，合成 0.625s：k=0.5，cps=4
    p.observe(5, 1.25, 0.625)
    assert p.char_cap(0.2) == 0
    assert p.char_cap(1.3) == 8
    assert p.char_cap(30.0) == 40
    backlog = ["一二三，", "四五六，", "七八九十。"]
    assert p.take(list(backlog), ahead_sec=1.3) == ["一二三，四五六，"]
    assert p.take(list(backlog), ahead_sec=0.0) == ["一二三，"]
    assert p.take(list(backlog), ahead_sec=1.3, max_units=3) == ["一二三，四五六，", "七八九十。"]
    rest = ["一二三，", "四五六七八九十。", "好。"]
    assert p.take(rest, ahead_sec=0.0, max_units=3) == ["一二三，"]
    assert rest == ["四五六七八九十。", "好。"]


def test_planner_batch_round_uses_longest_unit():
    from tts.planner import UnitPlanner
    p = UnitPlanner(40)
    p.observe_round([(4, 1.0, 0.01), (8, 2.0, 0.01)], batch_sec=1.99)
    assert abs(p.k - 1.0) < 1e-6
    assert abs(p.cps - 4.0) < 1e-6
    p.observe(0, 1.0, 1.0)
    p.observe(4, 0.1, 1.0)
    assert abs(p.k - 1.0) < 1e-6


class _FakeQwen:
    def __init__(self, fail_batch=False):
        self.calls = []
        self.fail_batch = fail_batch

    def create_voice_clone_prompt(self, **kwargs):
        return ["prompt"]

    def generate_voice_clone(self, text, language, voice_clone_prompt, non_streaming_mode, max_new_tokens):
        texts = text if isinstance(text, list) else [text]
        self.calls.append(list(texts))
        if self.fail_batch and len(texts) > 1:
            raise RuntimeError("OOM")
        sr = 24000
        wavs = []
        for t in texts:
            wav = np.frombuffer(_speech(sr, 0.5, 0.05 * len(t), 0.5), dtype="<i2").astype(np.float32) / 32767.0
            wavs.append(wav)
        return wavs, sr


class _Persona:
    id = "monet"

    class tts:
        prompt_wav = "voice/monet/monet-senlin_6to12s.wav"
        prompt_text = "参考句。"


def test_qwen_batch_one_generate_and_order(monkeypatch):
    import tts.qwen as q
    fake = _FakeQwen()
    monkeypatch.setattr(q, "_model", fake)
    monkeypatch.setattr(q, "_prompts", {})
    texts = ["第一句话。", "", "第二句稍微长一点。", "三。"]
    out = q.synthesize_batch(texts, _Persona(), batch_size=8)
    assert fake.calls == [["第一句话。", "第二句稍微长一点。", "三。"]]
    assert len(out) == 4
    assert out[1][0] == b""
    lens = [len(out[i][0]) / 2.0 / 24000 for i in (0, 2, 3)]
    # 头尾 0.5s 静音已裁，只剩 40ms + 160ms 余量
    assert abs(lens[0] - (0.05 * 5 + 0.2)) < 0.03
    assert abs(lens[1] - (0.05 * 9 + 0.2)) < 0.03
    assert lens[1] > lens[0] > lens[2]


def test_qwen_batch_splits_and_falls_back_per_sentence(monkeypatch):
    import tts.qwen as q
    fake = _FakeQwen(fail_batch=True)
    monkeypatch.setattr(q, "_model", fake)
    monkeypatch.setattr(q, "_prompts", {})
    texts = ["一句。", "两句。", "三句。"]
    out = q.synthesize_batch(texts, _Persona(), batch_size=2)
    assert fake.calls[0] == ["一句。", "两句。"]
    assert ["一句。"] in fake.calls and ["两句。"] in fake.calls and ["三句。"] in fake.calls
    assert all(pcm for pcm, _sr in out)


def test_remote_iter_pcm_streams_even_chunks(monkeypatch):
    import requests
    import tts.remote as remote

    class Resp:
        status_code = 200
        headers = {"X-Sample-Rate": "24000"}

        def iter_content(self, n):
            yield b"\x01\x02\x03"
            yield b"\x04\x05"
            yield b"\x06\x07\x08\x09"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())
    chunks = list(remote.iter_pcm("你好。", "monet"))
    assert [c for c, _sr in chunks] == [b"\x01\x02", b"\x03\x04", b"\x05\x06\x07\x08"]
    assert all(sr == 24000 for _c, sr in chunks)
    pcm, sr = remote.synthesize_pcm("你好。", "monet")
    assert pcm == b"\x01\x02\x03\x04\x05\x06\x07\x08" and sr == 24000


class _FakeClock:
    """ahead 秒 = 测试指定的客户端剩余缓冲；收尾等待按 0 算，测试不真睡。"""

    def __init__(self, ahead):
        self.ahead = ahead
        self.nbytes = 0

    def add(self, chunk):
        self.nbytes += len(chunk)

    def remain(self, tail=None):
        return self.ahead if tail == 0.0 else 0.0

    @property
    def duration(self):
        return self.nbytes / 2.0 / 24000


def _run_visitor_turn(monkeypatch, text, merge_chars, engine="cosyvoice",
                      synth_sec=0.01, ahead=10.0, sec_per_char=0.25):
    import ws.tts as wst
    from tts import cosyvoice as cv

    pushed = []
    prefetched = []

    async def fake_push(websocket, safe_send, sent, speaker, session_id, seq, cancel_ev,
                        subtitle_text=None, subtitle_index=None, on_pcm=None, persona=None):
        pushed.append((subtitle_index, sent))
        await asyncio.sleep(synth_sec)
        if on_pcm is not None:
            on_pcm(b"\x01\x00" * int(24000 * sec_per_char * len(sent)))
        return seq + 1, False

    def fake_prefetch(texts, speaker, cancel_ev, persona=None):
        prefetched.append(list(texts))
        return len(texts)

    monkeypatch.setattr(wst, "push_sentence_pcm", fake_push)
    monkeypatch.setattr(wst, "tts_engine_name", lambda: engine)
    monkeypatch.setattr(wst, "tts_output_rate", lambda: 24000)
    monkeypatch.setattr(wst, "PcmPlayClock", lambda sr=0: _FakeClock(ahead))
    monkeypatch.setattr(cv, "clause_merge_chars", lambda: merge_chars)
    monkeypatch.setattr(cv, "prefetch_batch", fake_prefetch)
    monkeypatch.setattr(cv, "cosyvoice_model", object())

    sent_frames = []

    async def safe_send(obj):
        sent_frames.append(obj)

    from persona.runtime import bind, resolve

    async def main():
        with bind(resolve("monet")):
            await wst.handle_tts_request(
                None, safe_send, text, "中文男", False, "sid-1",
                threading.Event(), {}, ["sid-1"])

    asyncio.run(main())
    return pushed, prefetched, sent_frames


TEXT = "我是莫奈，今天陪你看画，先说睡莲，再说日出，最后讲花园。"


def test_visitor_merges_once_buffer_is_ahead(monkeypatch):
    pushed, _pre, frames = _run_visitor_turn(monkeypatch, TEXT, merge_chars=40, ahead=10.0)
    units = [s for _i, s in pushed]
    assert units == ["我是莫奈，", "今天陪你看画，先说睡莲，再说日出，最后讲花园。"]
    assert [i for i, _s in pushed] == [0, 1]
    end = frames[-1]
    assert end["type"] == "tts_end" and end["sentence_count"] == 2


def test_visitor_stays_per_clause_when_buffer_is_dry(monkeypatch):
    pushed, _pre, _frames = _run_visitor_turn(monkeypatch, TEXT, merge_chars=40, ahead=0.0)
    assert [s for _i, s in pushed] == ["我是莫奈，", "今天陪你看画，", "先说睡莲，", "再说日出，", "最后讲花园。"]
    assert [i for i, _s in pushed] == list(range(5))


def test_visitor_keeps_per_clause_without_merge(monkeypatch):
    text = "我是莫奈，今天陪你看画，先说睡莲。"
    pushed, _pre, _frames = _run_visitor_turn(monkeypatch, text, merge_chars=0)
    assert [s for _i, s in pushed] == ["我是莫奈，", "今天陪你看画，", "先说睡莲。"]


def test_visitor_qwen_prefetches_planned_units_as_one_batch(monkeypatch):
    text = "我是莫奈，今天陪你看画，先说睡莲，再说日出，最后讲花园，然后去看桥。"
    pushed, prefetched, _frames = _run_visitor_turn(
        monkeypatch, text, merge_chars=8, engine="qwen3", ahead=10.0)
    units = [s for _i, s in pushed]
    assert "".join(units) == text
    assert units[0] == "我是莫奈，"
    assert all(len(u) <= 8 or "，" not in u[:-1] for u in units)
    assert prefetched == [units[1:]]
    assert len(prefetched[0]) >= 2
