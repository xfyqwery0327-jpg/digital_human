# -*- coding: utf-8 -*-
"""Qwen3-TTS-12Hz-0.6B-Base。Base 没有内置音色，用各人设的参考音做声音克隆。"""
from __future__ import annotations

import threading

from tts.engine import prepare_qwen_dir, qwen_model_dir

_model = None
_sample_rate = 24000
_prompts = {}
_lock = threading.Lock()
_last_error = ""


class QwenHandle:
    """占位对象。只提供 sample_rate，让现有 `if cosyvoice_model` 判断走得通。"""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = int(sample_rate or 24000)

    def __bool__(self):
        return True


def qwen_ready() -> bool:
    return _model is not None


def last_error() -> str:
    return _last_error


def _as_float_wave(audio):
    """把 Qwen 返回的波形收成一维 float32。torch 张量先落到 CPU。"""
    import numpy as np
    if audio is None:
        return np.zeros(0, dtype=np.float32)
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _float_to_pcm16(audio) -> bytes:
    import numpy as np
    arr = _as_float_wave(audio)
    if arr.size == 0:
        return b""
    pcm = np.clip(arr, -1.0, 1.0)
    return (pcm * 32767.0).astype("<i2").tobytes()


def _log_tts(msg: str, fail: bool = False):
    try:
        import tts.cosyvoice as cv
        cv._tts_log(msg, fail=fail)
    except Exception:
        print(msg)


def _load_model(model_dir):
    import torch
    from qwen_tts import Qwen3TTSModel

    use_cuda = bool(torch.cuda.is_available())
    kwargs = {
        "device_map": "cuda:0" if use_cuda else "cpu",
        "dtype": torch.bfloat16 if use_cuda else torch.float32,
    }
    try:
        return Qwen3TTSModel.from_pretrained(
            str(model_dir), attn_implementation="sdpa", **kwargs)
    except TypeError:
        return Qwen3TTSModel.from_pretrained(str(model_dir), **kwargs)


def codec_budget(text: str) -> int:
    """按字数封顶 codec token。

    12Hz，正常口播大约每个字 4 个 token。150 字大约 600 token。
    封顶 1024（约 85 秒）。配置里的 8192 约等于 11 分钟，结束符一对不上就停不下来。
    """
    n = max(1, len((text or "").strip()))
    return int(min(1024, max(96, n * 4 + 48)))


def _reset_talker_cache(model) -> None:
    """清掉上一句留在 talker 上的 RoPE 偏移和 KV。不清的话下一句位置接着涨，结束符对不上。"""
    inner = getattr(model, "model", None)
    talker = getattr(inner, "talker", None) if inner is not None else None
    if talker is None:
        return
    predictor = getattr(talker, "code_predictor", None)
    targets = [
        talker,
        predictor,
        getattr(talker, "model", None),
        getattr(predictor, "model", None) if predictor is not None else None,
    ]
    for obj in targets:
        if obj is None:
            continue
        for name in ("rope_deltas", "past_key_values"):
            if hasattr(obj, name):
                try:
                    setattr(obj, name, None)
                except Exception:
                    pass


def _prompt_for(model, persona):
    tts = getattr(persona, "tts", None) if persona is not None else None
    wav = (getattr(tts, "prompt_wav", None) or "").strip()
    ref = (getattr(tts, "prompt_text", None) or "").strip()
    pid = (getattr(persona, "id", None) or "default") if persona is not None else "default"
    if not wav:
        raise RuntimeError("人设 %s 没有参考音，Qwen3-TTS 无法克隆" % pid)
    key = (pid, wav, ref)
    with _lock:
        cached = _prompts.get(key)
    if cached is not None:
        return cached
    item = model.create_voice_clone_prompt(
        ref_audio=wav,
        ref_text=ref,
        x_vector_only_mode=not bool(ref),
    )
    with _lock:
        _prompts[key] = item
    return item


def _synthesize_piece(piece: str, persona, cancel_ev):
    """一句一算。每句前后清 RoPE，避免上一句的位置偏移把结束符顶飞。"""
    import time
    t0 = time.time()
    budget = codec_budget(piece)
    _log_tts("[TTS] qwen synth begin chars=%s budget=%s text=%s" % (len(piece), budget, piece[:40]))
    try:
        prompt = _prompt_for(_model, persona)
        with _lock:
            _reset_talker_cache(_model)
            try:
                generated = _model.generate_voice_clone(
                    text=piece,
                    language="Chinese",
                    voice_clone_prompt=prompt,
                    non_streaming_mode=True,
                    max_new_tokens=budget,
                )
            finally:
                _reset_talker_cache(_model)
    except Exception as e:
        msg = "qwen synth FAIL %s: %s text=%s" % (type(e).__name__, e, piece[:40])
        _log_tts("[TTS] " + msg, fail=True)
        raise RuntimeError(msg) from e
    if cancel_ev is not None and cancel_ev.is_set():
        return b"", _sample_rate
    wavs, sr = _unpack_wavs(generated)
    audio = wavs[0] if wavs else None
    pcm = _float_to_pcm16(audio) if audio is not None else b""
    rate = int(sr or _sample_rate)
    peak = 0
    if pcm:
        import numpy as np
        samples = np.frombuffer(pcm, dtype="<i2")
        peak = int(np.max(np.abs(samples.astype(np.int32)))) if samples.size else 0
    _log_tts("[TTS] qwen synth done chars=%s pcm=%s sr=%s peak=%s wall=%.2fs"
             % (len(piece), len(pcm), rate, peak, time.time() - t0))
    if not pcm:
        msg = "qwen synth empty chars=%s text=%s" % (len(piece), piece[:40])
        _log_tts("[TTS] " + msg, fail=True)
        raise RuntimeError(msg)
    return pcm, rate


def synthesize_pcm(text: str, persona=None, cancel_ev=None):
    """整段合成一次，返回 (pcm_bytes, sample_rate)。

    官方 generate_voice_clone 的 non_streaming_mode 默认是 False，只是把整句拆成
    模拟流式文本再喂进去，短句和带逗号的口播容易得到空波形。这里必须关掉。
    按逗号拆成十几次调用时，每次都要重算参考音，一轮 140 字会超过 90 秒。
    """
    if cancel_ev is not None and cancel_ev.is_set():
        return b"", _sample_rate
    if _model is None:
        raise RuntimeError(_last_error or "Qwen3-TTS 未加载")
    spoken = (text or "").strip()
    if not spoken:
        return b"", _sample_rate
    return _synthesize_piece(spoken, persona, cancel_ev)


def _unpack_wavs(generated):
    """兼容 (wavs, sr) 和只返回波形两种写法。"""
    if isinstance(generated, tuple) and len(generated) == 2:
        wavs, sr = generated
    else:
        wavs, sr = generated, _sample_rate
    if wavs is None:
        return [], sr
    if hasattr(wavs, "shape") and not isinstance(wavs, (list, tuple)):
        return [wavs], sr
    return list(wavs), sr


def activate() -> bool:
    """加载本地 Base 权重，并把占位对象交给 tts.cosyvoice。失败不抛到启动流程外。"""
    global _model, _sample_rate, _last_error
    import tts.cosyvoice as cv
    from config import ENABLE_AUDIO_CACHE, AUDIO_CACHE_SIZE, TTS_MAX_WORKERS
    from concurrent.futures import ThreadPoolExecutor
    from tts.cosyvoice import AudioCache

    if _model is not None and cv.cosyvoice_model is not None:
        return True
    model_dir = prepare_qwen_dir(qwen_model_dir())
    cfg = model_dir / "config.json"
    weight = model_dir / "model.safetensors"
    if not cfg.is_file() or not weight.is_file():
        _last_error = "Qwen3-TTS 权重不完整: %s" % model_dir
        cv._tts_last_error = _last_error
        cv._tts_log("[TTS] FAIL %s" % _last_error, fail=True)
        print("❌ " + _last_error)
        return False
    try:
        _model = _load_model(model_dir)
        sr = int(getattr(_model, "sample_rate", None) or 24000)
        _sample_rate = sr if sr > 0 else 24000
        cv.cosyvoice_model = QwenHandle(_sample_rate)
        cv._tts_prompt_mode = "qwen3"
        cv._tts_support_stream = False
        cv._tts_use_spk_id = True
        cv._tts_infer_mode = "qwen3"
        cv._tts_last_error = ""
        _last_error = ""
        if cv.tts_executor is None:
            cv.tts_executor = ThreadPoolExecutor(max_workers=TTS_MAX_WORKERS)
        if ENABLE_AUDIO_CACHE and cv.audio_cache is None:
            cv.audio_cache = AudioCache(max_size=AUDIO_CACHE_SIZE)
        cv._tts_log("[TTS] qwen3 model=%s sr=%s" % (model_dir, _sample_rate))
        print("✅ Qwen3-TTS 已加载 %s sr=%s" % (model_dir, _sample_rate))
        return True
    except Exception as e:
        _model = None
        cv.cosyvoice_model = None
        _last_error = "%s: %s" % (type(e).__name__, e)
        cv._tts_last_error = _last_error
        cv._tts_log("[TTS] FAIL qwen3 %s" % _last_error, fail=True)
        print("❌ Qwen3-TTS 加载失败: %s" % _last_error)
        if "qwen_tts" in _last_error or "No module named" in _last_error:
            print("   robot 环境需要: pip install qwen-tts")
        return False
