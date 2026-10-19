# -*- coding: utf-8 -*-
"""CosyVoice 加载、探测、合成。"""
from __future__ import annotations
import os
import sys
import time
import io
import base64
import hashlib
import threading
import wave
import tempfile
import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple, Callable

try:
    import numpy as np
    from scipy.io import wavfile
except ImportError:
    np = None
    wavfile = None

from config import *
from tts.load_policy import (
    apply_probe_decision,
    first_tensor as _first_tensor,
    register_succeeded,
    resolve_prompt_mode,
    seed_speaker_mode,
    tts_health_ready,
)

cosyvoice_model = None
tts_executor = None
audio_cache = None
_tts_prompt_arg = None
_tts_prompt_mode = "unknown"
_tts_support_stream = True
_tts_use_spk_id = False
_tts_infer_mode = "unknown"
_tts_last_error = ""
_infer_sem = threading.Semaphore(1)
_load_lock = threading.Lock()
_voice_spk_id = TTS_SPK_ID
_voice_prompt_text = PROMPT_TEXT
_voice_persona_id = "monet"
_load_wav_fn = None
_registered_spks = set()
_spk_emb_cache = {}
_spk_flow_prompt = {}  # persona_id → (flow_token int32, prompt_feat float)
_spk_src = {}  # persona_id → 裁切后的参考音路径
_spk_raw = {}  # persona_id → YAML 原始参考音，热路径跳过 trim
_CLAUSE_END = "，。！？；、,.!?"

class AudioCache:
    def __init__(self, max_size=100):
        self.cache = OrderedDict()
        self.max_size = max_size
        self._lock = threading.Lock()

    def _get_key(self, text: str, speaker: str, fmt: str = "pcm",
                 persona_id: str = "") -> str:
        pid = persona_id or _voice_persona_id
        # xling：embedding + 人设 flow prompt。旧 emb 缓存不得命中（那是句间换音色的版本）。
        return hashlib.md5(f"{fmt}|{pid}|xling|{text}|{speaker}".encode()).hexdigest()

    def get(self, text: str, speaker: str, fmt: str = "pcm", persona_id: str = ""):
        key = self._get_key(text, speaker, fmt, persona_id=persona_id)
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def put(self, text: str, speaker: str, audio_data, fmt: str = "pcm",
            persona_id: str = ""):
        key = self._get_key(text, speaker, fmt, persona_id=persona_id)
        with self._lock:
            self.cache[key] = audio_data
            self.cache.move_to_end(key)
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def drop(self, text: str, speaker: str, fmt: str = "pcm", persona_id: str = ""):
        key = self._get_key(text, speaker, fmt, persona_id=persona_id)
        with self._lock:
            self.cache.pop(key, None)


def resolve_tts_persona(explicit=None):
    """合成用人设：显式传入 > 当前 bind。禁止把「没 bind」当成舞台默认莫奈。"""
    if explicit is not None:
        return explicit
    try:
        from persona.runtime import peek
        return peek()
    except Exception:
        return None


def tts_min_chars(persona=None) -> int:
    """只合并「对了，」这种碎片。完整小句按标点单独合成，避免后半段被截掉。"""
    return 6


def tts_infer_mode() -> str:
    return _tts_infer_mode


def tts_output_rate() -> int:
    """tts_start 要带模型真实采样率。配置里的 24000 只是模型还没加载时的缺省。"""
    sr = getattr(cosyvoice_model, "sample_rate", None) if cosyvoice_model is not None else None
    try:
        sr = int(sr)
    except (TypeError, ValueError):
        sr = 0
    return sr if sr > 0 else int(TTS_SAMPLE_RATE)


def _remote_on() -> bool:
    """TTS_BACKEND=http 时，CosyVoice 在另一个进程，本进程只发 HTTP。"""
    return os.environ.get("TTS_BACKEND", "").strip().lower() in {"http", "remote"}


def tts_ready() -> bool:
    """/health tts：本地模型在且至少缓存了一个人设 embedding。独立进程模式只要求客户端已接上。"""
    from tts.engine import use_qwen_tts
    if use_qwen_tts():
        from tts.qwen import qwen_ready
        return qwen_ready() and cosyvoice_model is not None
    if _remote_on():
        return cosyvoice_model is not None
    return tts_health_ready(cosyvoice_model, _spk_emb_cache)


def _retry_register_speakers() -> bool:
    """模型已在内存时只补注册音色，禁止再走 CosyVoice3()。"""
    try:
        from persona.stage import stage
        from persona.registry import registry
        for p in registry.enabled():
            if p.voice_ready():
                try:
                    set_tts_persona(p)
                except Exception as e:
                    print(f"⚠️ 补注册人设音色 {p.id} 失败: {e}")
        boot = stage.current()
        if boot is not None and getattr(boot, "voice_ready", lambda: False)():
            set_tts_persona(boot)
    except Exception as e:
        print(f"⚠️ 补注册音色失败: {e}")
    return tts_ready()


def ensure_cosyvoice_loaded() -> bool:
    """请求侧可调用。已加载则补注册；未加载才 CosyVoice3()。禁止探测失败卸模型。"""
    if _remote_on():
        if cosyvoice_model is None:
            load_cosyvoice()
        return cosyvoice_model is not None
    if cosyvoice_model is not None:
        if not tts_ready():
            _retry_register_speakers()
        return True
    print("🔁 CosyVoice 未加载，请求侧补加载")
    load_cosyvoice()
    return cosyvoice_model is not None


def clause_merge_chars() -> int:
    """>0 时相邻小句拼到这个字数再合成。

    CosyVoice3（本地 / 独立进程）和 Qwen3-TTS 都是整句出声，每次调用固定要重算参考音前缀，
    逗号一句一调会让句间空档等于整句合成耗时。CosyVoice2 走 hop 流式，保持逐小句。
    """
    if not TTS_MERGE_CHARS:
        return 0
    from tts.engine import use_qwen_tts
    if use_qwen_tts() or _remote_on() or _cosyvoice3_loaded():
        return TTS_MERGE_CHARS
    if cosyvoice_model is None and "cosyvoice3" in str(COSYVOICE_MODEL).lower():
        return TTS_MERGE_CHARS
    return 0


def _trim_whole(pcm: bytes, sr: int) -> bytes:
    """整句 PCM 去头尾长静音。只给一次出整句的引擎用，流式 hop 分片禁止调用。"""
    from tts.pcmtrim import trim_pcm16
    return trim_pcm16(pcm, int(sr or TTS_SAMPLE_RATE), TTS_TRIM_HEAD_MS, TTS_TRIM_TAIL_MS)


def pack_tts_utterances(parts, persona=None, min_chars: Optional[int] = None) -> list:
    """先转口播数字，再按标点切开。尾段单独成项，禁止粘到上一句被 CosyVoice 截掉。"""
    from tts.speaktext import split_speak_chunks, to_spoken_zh
    out = []
    for raw in parts or []:
        chunks = split_speak_chunks(to_spoken_zh(raw or ""))
        if chunks:
            out.extend(chunks)
    return out


def _clone_emb(t):
    if t is None:
        return None
    import torch
    c = t.detach().clone()
    if c.dtype != torch.float32:
        c = c.float()
    return c


def _clone_cpu(t):
    """缓存/推理用副本。禁止改成 float32：speech token 是 int32。"""
    if t is None:
        return None
    c = t.detach().clone()
    if getattr(c, "device", None) is not None and c.device.type != "cpu":
        c = c.cpu()
    return c


def _flow_prompt_ok(token, feat) -> bool:
    if token is None or feat is None:
        return False
    try:
        t_shape = tuple(token.shape)
        f_shape = tuple(feat.shape)
        t_len = int(t_shape[-1]) if t_shape else 0
        f_len = int(f_shape[1]) if len(f_shape) >= 2 else 0
        return t_len > 8 and f_len > 16
    except Exception:
        return False


def _persona_flow_prompt(persona_id: str):
    pair = _spk_flow_prompt.get(persona_id)
    if pair is None:
        return None
    token, feat = pair
    if not _flow_prompt_ok(token, feat):
        _spk_flow_prompt.pop(persona_id, None)
        print(f"⚠️ 丢弃损坏 flow prompt persona={persona_id}")
        return None
    return _clone_cpu(token), _clone_cpu(feat)


def _spk_embeddings(spk_id=None, persona_id=None):
    """当前人设音色向量。优先用注册时的 clone，避免推理改写 spk2info 串台。"""
    pid = persona_id or _voice_persona_id
    if not pid:
        return None
    cached = _spk_emb_cache.get(pid)
    if cached is not None:
        if not _emb_ok(cached[0]) or not _emb_ok(cached[1]):
            _spk_emb_cache.pop(pid, None)
            print(f"⚠️ 丢弃损坏 embedding cache persona={pid}")
            return None
        return _clone_emb(cached[0]), _clone_emb(cached[1])
    if _voice_persona_id and pid != _voice_persona_id:
        print(f"⚠️ embedding 拒绝串台 want={pid} have={_voice_persona_id}")
        return None
    sid = spk_id or _voice_spk_id
    fe = getattr(cosyvoice_model, "frontend", None)
    if fe is None:
        return None
    info = (getattr(fe, "spk2info", None) or {}).get(sid)
    if not info:
        return None
    llm_emb = _first_tensor(info.get("llm_embedding"), info.get("embedding"))
    flow_emb = _first_tensor(info.get("flow_embedding"), info.get("embedding"), llm_emb)
    if llm_emb is None or flow_emb is None:
        return None
    return _clone_emb(llm_emb), _clone_emb(flow_emb)


def _remember_spk_emb(persona_id: str, spk_id: str) -> bool:
    fe = getattr(cosyvoice_model, "frontend", None)
    if fe is None:
        return False
    info = (getattr(fe, "spk2info", None) or {}).get(spk_id)
    if not info:
        return False
    llm_emb = _first_tensor(info.get("llm_embedding"), info.get("embedding"))
    flow_emb = _first_tensor(info.get("flow_embedding"), info.get("embedding"), llm_emb)
    if llm_emb is None or flow_emb is None:
        return False
    if not _emb_ok(llm_emb) or not _emb_ok(flow_emb):
        print(f"⚠️ 人设 {persona_id} embedding 全 0/NaN，拒绝缓存")
        return False
    _spk_emb_cache[persona_id] = (_clone_emb(llm_emb), _clone_emb(flow_emb))
    return True


def _remember_flow_prompt(persona_id: str, spk_id: str) -> bool:
    """在 _scrub_spk_prompt 之前，把 zero-shot 的 flow 参考音波形条件拷走。"""
    fe = getattr(cosyvoice_model, "frontend", None)
    if fe is None:
        return False
    info = (getattr(fe, "spk2info", None) or {}).get(spk_id) or {}
    token = info.get("flow_prompt_speech_token")
    if token is None:
        token = info.get("llm_prompt_speech_token")
    feat = info.get("prompt_speech_feat")
    if not _flow_prompt_ok(token, feat):
        return False
    _spk_flow_prompt[persona_id] = (_clone_cpu(token), _clone_cpu(feat))
    return True


def _cache_flow_prompt_from_wav(persona_id: str, wav_path: str) -> bool:
    """add_zero_shot 失败时，按 CosyVoice2 对齐规则从 clip 抽 flow prompt。"""
    fe = getattr(cosyvoice_model, "frontend", None)
    if fe is None or not hasattr(fe, "_extract_speech_token") or not hasattr(fe, "_extract_speech_feat"):
        return False
    try:
        speech_token, _tok_len = fe._extract_speech_token(wav_path)
        speech_feat, _feat_len = fe._extract_speech_feat(wav_path)
        token_len = min(int(speech_feat.shape[1] / 2), int(speech_token.shape[1]))
        if token_len <= 8:
            print(f"⚠️ flow prompt 过短 persona={persona_id} tok={token_len}")
            return False
        speech_feat = speech_feat[:, :2 * token_len]
        speech_token = speech_token[:, :token_len]
        if not _flow_prompt_ok(speech_token, speech_feat):
            return False
        _spk_flow_prompt[persona_id] = (_clone_cpu(speech_token), _clone_cpu(speech_feat))
        return True
    except Exception as e:
        print(f"⚠️ 抽取 flow prompt 失败 persona={persona_id}: {type(e).__name__}: {e}")
        return False


def _empty_prompt_kwargs():
    """每次新建空 prompt，禁止用 CosyVoice tts() 的可变默认张量（会串上一句参考音）。"""
    import torch
    return {
        "prompt_text": torch.zeros(1, 0, dtype=torch.int32),
        "llm_prompt_speech_token": torch.zeros(1, 0, dtype=torch.int32),
        "flow_prompt_speech_token": torch.zeros(1, 0, dtype=torch.int32),
        "prompt_speech_feat": torch.zeros(1, 0, 80, dtype=torch.float32),
        "source_speech_token": torch.zeros(1, 0, dtype=torch.int32),
    }


def _reset_cosy_hop():
    """CosyVoice2 流式会把 token_hop_len 乘到 100 且不还原，后续句首 hop 过大、尾句被跳过。"""
    m = getattr(cosyvoice_model, "model", None)
    if m is None:
        return
    if hasattr(m, "token_hop_len"):
        m.token_hop_len = 25


def _install_token2wav_guard(model):
    """speed=1.2 时 CosyVoice 用 int(mel_len/speed) 插值，短句会变成 0 帧。
    禁止全局替换 F.interpolate：HiFT 会按 5 个位置参数调用，包装函数一拦整句无 PCM。
    失败或静音时用 speed=1.0 再跑 vocoder。"""
    if model is None or not hasattr(model, "token2wav"):
        return
    orig = model.token2wav

    def guarded(*args, **kwargs):
        speed = kwargs.get("speed", 1.0)
        try:
            speech = orig(*args, **kwargs)
        except Exception as e:
            if speed != 1.0:
                print(f"⚠️ token2wav speed={speed} 失败 ({type(e).__name__}: {e})，speed=1.0 重试")
                kwargs = dict(kwargs)
                kwargs["speed"] = 1.0
                speech = orig(*args, **kwargs)
            else:
                raise
        if speech is not None and not _speech_usable({"tts_speech": speech}) and speed != 1.0:
            print("⚠️ token2wav 输出静音/NaN，speed=1.0 重试")
            kwargs = dict(kwargs)
            kwargs["speed"] = 1.0
            speech = orig(*args, **kwargs)
        return speech

    model.token2wav = guarded


def _scrub_spk_prompt(spk_id: str):
    """注册后只留 embedding。零样本路径若再取 spk2info，不会拼别人的 prompt_feat。"""
    import torch
    fe = getattr(cosyvoice_model, "frontend", None)
    if fe is None or not spk_id:
        return
    info = (getattr(fe, "spk2info", None) or {}).get(spk_id)
    if not info:
        return
    for k, v in _empty_prompt_kwargs().items():
        info[k] = v
    zero = torch.zeros(1, dtype=torch.int32)
    for k in ("prompt_text_len", "llm_prompt_speech_token_len",
              "flow_prompt_speech_token_len", "prompt_speech_feat_len"):
        if k in info:
            info[k] = zero.clone()


def _speech_len(out) -> int:
    speech = out["tts_speech"] if isinstance(out, dict) else out
    return _audio_len(speech)


# 只拦「真·没声音」。NaN/Inf 先清成 0 再看能量：有几个坏点的正常语音必须下发。
# float[-1,1]：max abs < 1e-4；已经是 int16 量纲：max abs < 1。
# 禁止用「存在 0 / 存在 NaN / RMS 偏小」整句丢弃——那会把所有 tts_chunk 都吃掉。
_SILENCE_PEAK_FLOAT = 1e-4
_SILENCE_PEAK_INT16 = 1.0
_INT16_RANGE_PEAK = 8.0
_EMB_MIN_ABS = 1e-6


def _pcm_chunk_size() -> int:
    raw = int(TTS_CHUNK_SIZE or 4096)
    cap = int(globals().get("TTS_CHUNK_MAX", 8192) or 8192)
    return max(256, min(raw, cap, 8192))


def _speech_tensor(out):
    if out is None:
        return None
    return out["tts_speech"] if isinstance(out, dict) else out


def _emb_ok(t) -> bool:
    """全 0 embedding 经 F.normalize 会变 NaN，后续整句 PCM 都是 0。"""
    if t is None:
        return False
    try:
        import torch
        if int(t.numel()) <= 0:
            return False
        x = t.detach().float()
        if not torch.isfinite(x).all():
            return False
        return float(x.abs().max().item()) > _EMB_MIN_ABS
    except Exception:
        return False


def _as_1d_float32(seg):
    """展成 1D float32。NaN/Inf 置 0，不用 squeeze（[1,1,T] 或 T=1 时会挤成空/标量）。"""
    n_bad = 0
    if seg is None:
        empty = np.zeros(0, dtype=np.float32) if np is not None else None
        return empty, 0
    if hasattr(seg, "detach"):
        import torch
        t = seg.detach().reshape(-1).float()
        n = int(t.numel())
        if n <= 0:
            empty = np.zeros(0, dtype=np.float32) if np is not None else None
            return empty, 0
        n_bad = int((~torch.isfinite(t)).sum().item())
        if n_bad:
            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        if np is None:
            return t, n_bad
        return t.cpu().numpy().astype(np.float32, copy=False), n_bad
    if np is None:
        return seg, 0
    a = np.reshape(np.asarray(seg), -1)
    if a.size == 0:
        return np.zeros(0, dtype=np.float32), 0
    a = np.array(a, dtype=np.float32, copy=True)
    bad = ~np.isfinite(a)
    n_bad = int(np.count_nonzero(bad))
    if n_bad:
        a[bad] = 0.0
    return a, n_bad


def _audio_len(a) -> int:
    if a is None:
        return 0
    if hasattr(a, "detach") and hasattr(a, "numel"):
        try:
            return int(a.numel())
        except Exception:
            return 0
    try:
        return int(a.size)
    except Exception:
        return 0


def _audio_peak_rms(a):
    n = _audio_len(a)
    if n <= 0:
        return 0.0, 0.0
    if hasattr(a, "detach"):
        x = a.detach().float().reshape(-1)
        peak = float(x.abs().max().item())
        rms = float(x.pow(2).mean().sqrt().item())
        return peak, rms
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x * x)))
    return peak, rms


def _energy_fail_reason(a, n_bad: int = 0, n: int = 0) -> str:
    """只在清完 NaN 后峰值仍极近 0 时判定静音。部分样本为 0 不算。"""
    if a is None:
        return "none"
    size = _audio_len(a)
    if size <= 0:
        return "empty"
    peak, rms = _audio_peak_rms(a)
    n = n or size
    thr = _SILENCE_PEAK_INT16 if peak >= _INT16_RANGE_PEAK else _SILENCE_PEAK_FLOAT
    if peak < thr:
        extra = f" nan={n_bad}" if n_bad else ""
        return f"silent peak={peak:.2e} rms={rms:.2e} n={n}{extra}"
    return ""


def _speech_fail_reason(out) -> str:
    sp = _speech_tensor(out)
    if sp is None:
        return "none"
    try:
        n = _audio_len(sp)
        if n <= 0:
            return "empty"
        a, n_bad = _as_1d_float32(sp)
        return _energy_fail_reason(a, n_bad, n=n)
    except Exception as e:
        return f"err:{type(e).__name__}"


def _speech_usable(out) -> bool:
    """全 0 / 清 NaN 后仍无能量才不可用。有几个 0 或 NaN 的真语音要过。"""
    return not _speech_fail_reason(out)


def _pcm_peak(pcm: bytes) -> int:
    if not pcm or np is None:
        return 0 if not pcm else (1 if any(pcm) else 0)
    n = len(pcm) - (len(pcm) % 2)
    if n <= 0:
        return 0
    a = np.frombuffer(pcm[:n], dtype=np.int16)
    if a.size == 0:
        return 0
    return int(np.max(np.abs(a.astype(np.int32))))


def _pcm_is_silent(pcm: bytes) -> bool:
    """整段 int16 峰值 < 1 才是全 0。不要因为夹了若干 0 样本就丢包。"""
    if not pcm:
        return True
    return _pcm_peak(pcm) < _SILENCE_PEAK_INT16


def _detach_tts_output(output):
    """推理线程里立刻转 CPU numpy。禁止把 CUDA tensor 丢给 asyncio 线程（会变 0/NaN）。"""
    if output is None or isinstance(output, Exception):
        return output
    sr = TTS_SAMPLE_RATE
    speech = output
    if isinstance(output, dict):
        sr = output.get("sample_rate", TTS_SAMPLE_RATE) or TTS_SAMPLE_RATE
        speech = output.get("tts_speech")
    arr, n_bad = _as_1d_float32(speech)
    return {"tts_speech": arr, "sample_rate": sr, "n_bad": n_bad}


def _float_to_pcm16(seg) -> bytes:
    """float/int16 张量 → int16 bytes。仅当清 NaN 后仍无能量才返回空，避免 AAAA。"""
    if np is None or seg is None:
        return b""
    a, n_bad = _as_1d_float32(seg)
    if _energy_fail_reason(a, n_bad):
        return b""
    peak, _rms = _audio_peak_rms(a)
    if peak >= _INT16_RANGE_PEAK:
        pcm = np.clip(a, -32767.0, 32767.0).astype(np.int16)
    else:
        pcm = (np.clip(a, -1.0, 1.0) * 32767.0).astype(np.int16)
    return pcm.tobytes()


# CosyVoice3LM.inference 把 prompt_text 和正文拼起来后，断言 token 151646
# （<|endofprompt|>）必须在里面。缺了会直接 AssertionError。
# 官方零样本是「系统前缀 + 参考句原文」再配上这段参考音的 speech token。
# 只放前缀、语音 token 留空时，模型吐的是对不上字的 codec，听起来像乱码。
# 参考句放在 prompt_text，不放进要念的正文；flow 会切掉参考音那段 mel。
_CV3_TEXT_PREFIX = "You are a helpful assistant.<|endofprompt|>"
_CV3_EOP_ID = 151646
_cv3_prefix_cpu = None


def _cosyvoice3_loaded() -> bool:
    m = getattr(cosyvoice_model, "model", None)
    return type(m).__name__ == "CosyVoice3Model"


_STREAM_HOP = 25
_STREAM_IDLE_SEC = 20.0


def _yield_tts_stream(model, model_input: dict, hop: int = _STREAM_HOP,
                      idle_sec: float = _STREAM_IDLE_SEC):
    """按 hop 吐 PCM。

    官方 tts(stream=True) 只在 LLM 正常返回后才写 llm_end。线程里一旦异常，
    外层 while 会一直 sleep，整轮等不到 tts_end，后续 ask 全是 409。
    首跳用 hop（25），之后按 stream_scale_factor 收到 token_max_hop_len（25→50→100）。
    CosyVoice3 的 flow 每次都对「参考音 + 已有 token」重跑 10 步扩散，HiFT 也对
    累计 mel 整段重算。hop 钉死在 25 时，后段 chunk 会越来越慢。
    句末把 token_hop_len 拉回首跳，避免下一句首跳过大。尾音不够一跳仍走 finalize。
    """
    import uuid
    import torch
    text = model_input["text"]
    prompt_text = model_input["prompt_text"]
    llm_prompt = model_input["llm_prompt_speech_token"]
    llm_embedding = model_input["llm_embedding"]
    flow_prompt = model_input["flow_prompt_speech_token"]
    prompt_feat = model_input["prompt_speech_feat"]
    flow_embedding = model_input["flow_embedding"]
    this_uuid = str(uuid.uuid1())
    with model.lock:
        model.tts_speech_token_dict[this_uuid] = []
        model.llm_end_dict[this_uuid] = False
        model.hift_cache_dict[this_uuid] = None
    err = {}

    def _job():
        try:
            model.llm_job(text, prompt_text, llm_prompt, llm_embedding, this_uuid)
        except Exception as e:
            err["e"] = e
            print(f"⚠️ TTS LLM 线程失败: {type(e).__name__}: {e}")
        finally:
            model.llm_end_dict[this_uuid] = True

    scale = max(1, int(getattr(model, "stream_scale_factor", 2) or 1))
    max_hop = int(getattr(model, "token_max_hop_len", 0) or 0)
    if max_hop < hop:
        max_hop = hop * 4
    hop_len = hop
    if hasattr(model, "token_hop_len"):
        model.token_hop_len = hop_len
    worker = threading.Thread(target=_job, daemon=True)
    worker.start()
    token_offset = 0
    prompt_len = 0
    if flow_prompt is not None and getattr(flow_prompt, "ndim", 0) >= 2:
        prompt_len = int(flow_prompt.shape[1])
    pad = int(np.ceil(prompt_len / float(hop)) * hop - prompt_len) if hop else 0
    lookahead = int(getattr(model.flow, "pre_lookahead_len", 0) or 0)
    idle_since = time.time()
    last_n = 0
    yielded = False
    try:
        while True:
            time.sleep(0.1)
            tokens = model.tts_speech_token_dict.get(this_uuid) or []
            n = len(tokens)
            if n != last_n:
                last_n = n
                idle_since = time.time()
            ended = bool(model.llm_end_dict.get(this_uuid))
            this_hop = hop_len + pad if token_offset == 0 else hop_len
            if n - token_offset >= this_hop + lookahead:
                piece = torch.tensor(tokens[:token_offset + this_hop + lookahead]).unsqueeze(0)
                speech = model.token2wav(
                    token=piece,
                    prompt_token=flow_prompt,
                    prompt_feat=prompt_feat,
                    embedding=flow_embedding,
                    token_offset=token_offset,
                    uuid=this_uuid,
                    stream=True,
                    finalize=False,
                    speed=1.0,
                )
                token_offset += this_hop
                hop_len = min(max_hop, hop_len * scale)
                if hasattr(model, "token_hop_len"):
                    model.token_hop_len = hop_len
                idle_since = time.time()
                yielded = True
                yield {"tts_speech": speech.detach().cpu(), "sample_rate": tts_output_rate()}
                continue
            if ended and n - token_offset < this_hop + lookahead:
                break
            if not worker.is_alive() and not ended:
                model.llm_end_dict[this_uuid] = True
                continue
            if time.time() - idle_since > idle_sec:
                print("⚠️ TTS 流式 %.0fs 无新 token，提前收尾 offset=%s n=%s" % (
                    idle_sec, token_offset, n))
                break
        worker.join(timeout=5)
        tokens = model.tts_speech_token_dict.get(this_uuid) or []
        if tokens:
            speech = model.token2wav(
                token=torch.tensor(tokens).unsqueeze(0),
                prompt_token=flow_prompt,
                prompt_feat=prompt_feat,
                embedding=flow_embedding,
                token_offset=token_offset,
                uuid=this_uuid,
                stream=False,
                finalize=True,
                speed=1.0,
            )
            yielded = True
            yield {"tts_speech": speech.detach().cpu(), "sample_rate": tts_output_rate()}
        if err.get("e") is not None and not yielded:
            raise err["e"]
    finally:
        with model.lock:
            model.tts_speech_token_dict.pop(this_uuid, None)
            model.llm_end_dict.pop(this_uuid, None)
            model.hift_cache_dict.pop(this_uuid, None)
        if hasattr(model, "token_hop_len"):
            model.token_hop_len = hop
        from tts.cuda_cache import release_cuda_cache
        release_cuda_cache("cv2-sentence", tight=True)


def _cv3_prompt_string(transcript: str) -> str:
    """系统前缀后面接参考句原文。原文里已经带前缀时不重复拼。"""
    text = (transcript or "").strip().replace("〇", "零")
    if _CV3_TEXT_PREFIX in text:
        return text
    return _CV3_TEXT_PREFIX + text


def _cv3_prefix_tokens(frontend):
    """只有系统前缀。没有参考句时的退路，发音会对不上字。"""
    return _cv3_prompt_tokens(frontend, "")


def _cv3_prompt_tokens(frontend, transcript: str):
    """prompt_text = 前缀 + 参考句。正文仍单独编码，避免把森林开场白念出来。"""
    import torch
    raw = _cv3_prompt_string(transcript)
    if not (transcript or "").strip():
        print("⚠️ CosyVoice3 参考句为空，发音会像乱码。请填写人设 prompt_text")
    tok, _ = frontend._extract_text_token(raw)
    flat = tok.detach().reshape(-1).cpu()
    if int((flat == _CV3_EOP_ID).sum().item()) < 1:
        raise RuntimeError("CosyVoice3 未编码出 <|endofprompt|>，拒绝合成以免整句无声")
    return flat.to(dtype=torch.int32).reshape(1, -1)


def _drop_tts_session(model, this_uuid: str):
    with model.lock:
        for name in ("tts_speech_token_dict", "llm_end_dict", "hift_cache_dict",
                     "mel_overlap_dict", "flow_cache_dict"):
            bucket = getattr(model, name, None)
            if isinstance(bucket, dict):
                bucket.pop(this_uuid, None)
    if hasattr(model, "token_hop_len"):
        model.token_hop_len = 25


def _yield_tts_once(model, model_input: dict, join_sec: float = 90.0):
    """整句只跑一次 flow + HiFT。

    CosyVoice3 的流式 token2wav 每次都对「参考音 + 已有 token」重跑扩散，
    再把累计 mel 整段送进 HiFT。句子一长就越来越慢，显存也会被打满，
    后面的 ask 收不到 tts_end，连接直接断。
    """
    import uuid
    import torch
    text = model_input["text"]
    prompt_text = model_input["prompt_text"]
    llm_prompt = model_input["llm_prompt_speech_token"]
    llm_embedding = model_input["llm_embedding"]
    flow_prompt = model_input["flow_prompt_speech_token"]
    prompt_feat = model_input["prompt_speech_feat"]
    flow_embedding = model_input["flow_embedding"]
    this_uuid = str(uuid.uuid1())
    with model.lock:
        model.tts_speech_token_dict[this_uuid] = []
        model.llm_end_dict[this_uuid] = False
        model.hift_cache_dict[this_uuid] = None
    err = {}

    def _job():
        try:
            model.llm_job(text, prompt_text, llm_prompt, llm_embedding, this_uuid)
        except Exception as e:
            err["e"] = e
            print(f"⚠️ TTS LLM 线程失败: {type(e).__name__}: {e}")
        finally:
            model.llm_end_dict[this_uuid] = True

    if hasattr(model, "token_hop_len"):
        model.token_hop_len = 25
    worker = threading.Thread(target=_job, daemon=True)
    worker.start()
    yielded = False
    try:
        worker.join(timeout=join_sec)
        if worker.is_alive():
            print("⚠️ TTS 整句超过 %.0fs 仍未结束，按已有 token 收尾" % join_sec)
            model.llm_end_dict[this_uuid] = True
        tokens = model.tts_speech_token_dict.get(this_uuid) or []
        if tokens:
            speech = model.token2wav(
                token=torch.tensor(tokens).unsqueeze(0),
                prompt_token=flow_prompt,
                prompt_feat=prompt_feat,
                embedding=flow_embedding,
                token_offset=0,
                uuid=this_uuid,
                stream=False,
                finalize=True,
                speed=1.0,
            )
            yielded = True
            yield {"tts_speech": speech.detach().cpu(), "sample_rate": tts_output_rate()}
        if err.get("e") is not None and not yielded:
            raise err["e"]
    finally:
        _drop_tts_session(model, this_uuid)
        from tts.cuda_cache import release_cuda_cache
        release_cuda_cache("cv3-sentence", tight=True)


def _embedding_only_gen(text: str, persona_id: str = ""):
    """text + 人设 embedding，flow 带 clip 波形。

    CosyVoice2：LLM 不带参考句，按 hop 边生成边吐 PCM。
    CosyVoice3：LLM 带「前缀 + 参考句」和这段参考音的 speech token，
    一句只声码一次。参考句不写进正文。
    流式声码器只接受 speed=1.0。
    禁止把 flow prompt 也清空：只靠 192 维向量锁不住音色，句间会像好几个人。
    """
    from tts.speaktext import group_speak_chunks, to_spoken_zh, split_speak_chunks
    pid = persona_id or _voice_persona_id
    pair = _spk_embeddings(persona_id=pid)
    if pair is None:
        raise RuntimeError(f"人设 {pid} 无人设 embedding")
    llm_emb, flow_emb = pair
    if not _emb_ok(llm_emb) or not _emb_ok(flow_emb):
        raise RuntimeError(f"人设 {pid} embedding 全 0 或 NaN，拒绝合成以免下发静音 PCM")
    flow_pair = _persona_flow_prompt(pid)
    if flow_pair is None:
        wav_path = _spk_src.get(pid)
        if wav_path and _cache_flow_prompt_from_wav(pid, wav_path):
            flow_pair = _persona_flow_prompt(pid)
    if flow_pair is None:
        print(f"⚠️ 人设 {pid} 无 flow prompt，embedding-only 容易句间换音色")
    frontend = cosyvoice_model.frontend
    model = cosyvoice_model.model
    spoken = to_spoken_zh(text)
    # 短句跳过 wetext，首包更快
    if len(spoken) <= 48:
        normalized = spoken
    else:
        try:
            normalized = frontend.text_normalize(spoken, split=False, text_frontend=True)
        except Exception:
            normalized = spoken
    if isinstance(normalized, (list, tuple)):
        normalized = "".join(str(x) for x in normalized if x)
    # wetext 会把年份里的 0 写成「〇」，模型读音不对，再换成「零」。
    if isinstance(normalized, str):
        normalized = normalized.replace("〇", "零")
    pieces = split_speak_chunks(normalized or spoken)
    if _cosyvoice3_loaded():
        # 上游已按字数拼好；这里再按逗号拆回去，每段又要重算一次参考音前缀。
        pieces = group_speak_chunks(pieces, clause_merge_chars())
    if not pieces:
        pieces = [spoken] if spoken else []

    def _call(piece: str):
        _reset_cosy_hop()
        tts_text_token, tts_text_token_len = frontend._extract_text_token(piece)
        prompt_kw = _empty_prompt_kwargs()
        cv3 = _cosyvoice3_loaded()
        if cv3:
            prompt_kw["prompt_text"] = _cv3_prompt_tokens(frontend, _voice_prompt_text)
        if flow_pair is not None:
            # flow.inference 会切掉 prompt 段，不会把参考音拼进口播。
            # CosyVoice3 的 LLM 必须看到同一段 speech token，否则 codec 对不上字。
            prompt_kw["flow_prompt_speech_token"] = _clone_cpu(flow_pair[0])
            prompt_kw["prompt_speech_feat"] = _clone_cpu(flow_pair[1])
            if cv3:
                prompt_kw["llm_prompt_speech_token"] = _clone_cpu(flow_pair[0])
        model_input = {
            "text": tts_text_token,
            "text_len": tts_text_token_len,
            "llm_embedding": _clone_emb(llm_emb),
            "flow_embedding": _clone_emb(flow_emb),
            **prompt_kw,
        }
        if cv3:
            yield from _yield_tts_once(model, model_input)
        else:
            yield from _yield_tts_stream(model, model_input)

    def _one(piece: str):
        piece = (piece or "").strip()
        if not piece:
            return
        # 流式固定 1.0。1.2 会在未收尾的 hop 上触发 token2wav 断言。
        yielded = 0
        last_reason = "empty"
        for out in _call(piece):
            reason = _speech_fail_reason(out)
            if reason:
                last_reason = reason
                continue
            yielded += max(_speech_len(out), 1)
            yield out
        if yielded <= 0:
            print(f"⚠️ TTS 无声 persona={pid} '{piece[:24]}' ({last_reason})，speed=1.0 重试")
            for out in _call(piece):
                reason = _speech_fail_reason(out)
                if reason:
                    last_reason = reason
                    continue
                yielded += max(_speech_len(out), 1)
                yield out
        if yielded <= 0:
            print(f"⚠️ TTS 重试后仍无声 persona={pid} '{piece[:24]}' ({last_reason})")

    def _iter():
        n = len(pieces)
        for i, piece in enumerate(pieces):
            yield from _one(piece)
    return _iter()


def _start_cosy_gen(text: str, persona_id: str = ""):
    """人设 embedding + clip 的 flow prompt。参考句只进 CosyVoice3 的 prompt_text。"""
    global _tts_infer_mode
    from tts.speaktext import to_spoken_zh
    text = to_spoken_zh(text)
    pid = persona_id or _voice_persona_id
    if not pid or (persona_id and _voice_persona_id and persona_id != _voice_persona_id):
        raise RuntimeError(
            f"人设未对齐 want={persona_id} current={_voice_persona_id}，拒绝用别人的声音")
    if _spk_embeddings(persona_id=pid) is None:
        raise RuntimeError(f"人设 {pid} 无 embedding，拒绝用别人的参考音合成")
    _reset_cosy_hop()
    if _persona_flow_prompt(pid) is not None or (
            _spk_src.get(pid) and _cache_flow_prompt_from_wav(pid, _spk_src[pid])):
        _tts_infer_mode = "cross_lingual"
    else:
        _tts_infer_mode = "embedding"
    return _embedding_only_gen(text, persona_id=pid)


def _tts_infer(text: str, block: bool = True,
               cancel_ev: Optional[threading.Event] = None,
               persona=None):
    """统一推理入口。
    block=True + cancel_ev → 可中断阻塞：每 50ms 检查一次 cancel_ev，
                             取消时返回 None 而不是永久挂起
    block=True             → 普通阻塞（/ws/chat、POST /tts）
    block=False            → GPU 忙时立即返回 None（已无调用方使用此路径）
    persona                → 必须从请求侧传入；工作线程没有 ContextVar
    """
    if block:
        if cancel_ev is not None:
            acquired = False
            while not cancel_ev.is_set():
                acquired = _infer_sem.acquire(blocking=True, timeout=0.05)
                if acquired:
                    break
            if not acquired:   # 被取消，未拿到锁
                return None
        else:
            _infer_sem.acquire(blocking=True)
    else:
        if not _infer_sem.acquire(blocking=False):
            return None

    if cosyvoice_model is None:
        _infer_sem.release()
        raise RuntimeError("CosyVoice 未加载")

    p = resolve_tts_persona(persona)
    try:
        if p is None:
            print(f"⚠️ TTS 无人设上下文，拒绝沿用 {_voice_persona_id} '{text[:20]}...'")
            _infer_sem.release()
            raise RuntimeError("TTS 无人设，拒绝串用上一句音色")
        ok = set_tts_persona(p)
        if not ok or _voice_persona_id != p.id:
            _infer_sem.release()
            raise RuntimeError(f"TTS 无法切到 {p.id}，拒绝用 {_voice_persona_id} 的声音")
    except RuntimeError:
        raise
    except Exception as e:
        _infer_sem.release()
        raise RuntimeError(f"TTS 切音色失败: {e}") from e

    try:
        raw_gen = _start_cosy_gen(text, persona_id=p.id)
        from tts.speaktext import to_spoken_zh as _spoken_preview
        preview = _spoken_preview(text)
        flow_pair = _persona_flow_prompt(p.id)
        ftok = int(flow_pair[0].shape[-1]) if flow_pair else 0
        ffeat = int(flow_pair[1].shape[1]) if flow_pair else 0
        wav_name = Path(_spk_src.get(p.id) or "").name or "-"
        print(f"🎤 TTS infer persona={_voice_persona_id} spk={_voice_spk_id} "
              f"mode={tts_infer_mode()} flow_tok={ftok} feat={ffeat} wav={wav_name} "
              f"'{preview[:28]}...'")
    except Exception:
        _infer_sem.release()
        raise

    def _guarded():
        try:
            yield from raw_gen
        finally:
            _infer_sem.release()

    return _guarded()


# CosyVoice speech tokenizer 拒绝 >30s；零样本克隆 3–10s 最稳。
# 默认：去头静音后跳过 2s 开场，再取最多 6s。
# 莫奈：按原文件时间轴取 6–12s。
PROMPT_WAV_MAX_SEC = 6.0
PROMPT_SKIP_INTRO_SEC = 2.0
PROMPT_MIN_SEC = 3.0
MONET_PROMPT_START_SEC = 6.0
MONET_PROMPT_END_SEC = 12.0


def _load_audio_16k_mono(src: str):
    import torchaudio
    wav, sr = torchaudio.load(str(src))
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if int(sr) != 16000:
        wav = torchaudio.functional.resample(wav, int(sr), 16000)
        sr = 16000
    return wav, 16000


def _trim_leading_silence(wav, sr: int):
    x = wav.reshape(-1)
    hop = max(int(sr * 0.02), 1)
    n = (x.numel() // hop) * hop
    if n < hop:
        return wav
    energy = x[:n].view(-1, hop).pow(2).mean(dim=1)
    peak = float(energy.max().item()) if energy.numel() else 0.0
    if peak <= 0:
        return wav
    voiced = (energy > peak * 0.02).nonzero(as_tuple=False)
    if voiced.numel() == 0:
        return wav
    start = int(voiced[0].item() * hop)
    return wav[..., start:]


def _prompt_clip_suffix(persona_id: str) -> str:
    if persona_id == "monet":
        return "_6to12s"
    return f"_male{int(PROMPT_WAV_MAX_SEC)}s"


def _slice_abs(wav, sr: int, start_sec: float, end_sec: float):
    """按原时间轴切 [start, end)，不先削静音，避免窗口被开头静音带偏。"""
    n = int(wav.shape[-1])
    if n <= 0:
        return wav
    start = int(start_sec * sr)
    end = int(end_sec * sr)
    if start >= n:
        length = max(end - start, int(PROMPT_MIN_SEC * sr))
        return wav[..., max(0, n - length):]
    end = min(max(end, start), n)
    if end <= start:
        return wav[..., :min(n, int((end_sec - start_sec) * sr))]
    return wav[..., start:end]


def _select_prompt_region(wav, sr: int, persona_id: str = ""):
    """莫奈取原文件 6–12 秒。其他人设去头静音后跳过 2 秒开场，再取最多 6 秒。"""
    if persona_id == "monet":
        return _slice_abs(wav, sr, MONET_PROMPT_START_SEC, MONET_PROMPT_END_SEC)
    wav = _trim_leading_silence(wav, sr)
    n = int(wav.shape[-1])
    max_n = int(PROMPT_WAV_MAX_SEC * sr)
    min_n = int(PROMPT_MIN_SEC * sr)
    skip = int(PROMPT_SKIP_INTRO_SEC * sr)
    if n <= 0:
        return wav
    if n - skip >= min_n:
        end = min(n, skip + max_n)
        return wav[..., skip:end]
    if n > max_n:
        return wav[..., n - max_n:]
    return wav[..., :max_n] if n > max_n else wav


def _ensure_prompt_wav(src: str, persona_id: str = "") -> str:
    """转 16k 单声道再裁参考音。莫奈用 6–12s（文件名 *_6to12s.wav）。
    已经是对应后缀的 wav 则直接用，禁止再裁一次把音色裁偏。"""
    p = Path(src)
    if not p.exists():
        raise FileNotFoundError(src)
    suffix = _prompt_clip_suffix(persona_id)
    if p.suffix.lower() == ".wav" and p.stem.endswith(suffix):
        return str(p)
    dest = p.parent / f"{p.stem}{suffix}.wav"
    if dest.exists() and dest.stat().st_mtime >= p.stat().st_mtime:
        return str(dest)

    wav = None
    sr = 16000
    try:
        wav, sr = _load_audio_16k_mono(str(p))
    except Exception as e:
        print(f"⚠️ torchaudio 读参考音失败: {e}，尝试 ffmpeg")
        tmp = p.parent / f"{p.stem}_16k.wav"
        try:
            import subprocess
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(p), "-ac", "1", "-ar", "16000", str(tmp)],
                check=True, capture_output=True)
            wav, sr = _load_audio_16k_mono(str(tmp))
        except Exception as e2:
            raise RuntimeError(f"无法将参考音转为 wav: {src} ({e2})") from e2

    orig_sec = wav.shape[-1] / float(sr)
    if persona_id == "monet":
        window = "%.1f-%.1fs" % (MONET_PROMPT_START_SEC, MONET_PROMPT_END_SEC)
    else:
        window = "skip%.1f+%.1fs" % (PROMPT_SKIP_INTRO_SEC, PROMPT_WAV_MAX_SEC)
    wav = _select_prompt_region(wav, sr, persona_id)
    new_sec = wav.shape[-1] / float(sr)
    import torchaudio
    torchaudio.save(str(dest), wav, sr)
    msg = "[TTS] prompt clip %s %.1fs -> %.1fs (%s) file=%s" % (
        p.name, orig_sec, new_sec, window, dest.name)
    try:
        _tts_log(msg)
    except Exception:
        print(msg)
    return str(dest)


def _cache_embedding_from_wav(persona_id: str, spk_id: str, wav_path: str) -> bool:
    """不跑 zero-shot 合成：只从参考音抽 speaker embedding 并缓存。"""
    fe = getattr(cosyvoice_model, "frontend", None)
    if fe is None or not hasattr(fe, "_extract_spk_embedding"):
        return False
    try:
        emb = fe._extract_spk_embedding(wav_path)
    except Exception as e:
        print(f"⚠️ 抽取 embedding 失败 persona={persona_id}: {type(e).__name__}: {e}")
        return False
    if not _emb_ok(emb):
        print(f"⚠️ 抽取 embedding 无效 persona={persona_id}")
        return False
    cloned = _clone_emb(emb)
    _spk_emb_cache[persona_id] = (cloned, _clone_emb(emb))
    info_map = getattr(fe, "spk2info", None)
    if info_map is None:
        fe.spk2info = {}
        info_map = fe.spk2info
    info = info_map.get(spk_id) or {}
    info["llm_embedding"] = _clone_emb(emb)
    info["flow_embedding"] = _clone_emb(emb)
    info["embedding"] = _clone_emb(emb)
    info_map[spk_id] = info
    _registered_spks.add(spk_id)
    _cache_flow_prompt_from_wav(persona_id, wav_path)
    _scrub_spk_prompt(spk_id)
    return True


def _seed_spk_from_prompt(persona_id: str, spk_id: str, wav_path: str,
                          prompt_text: str = "") -> str:
    """路径注册音色；失败则 embedding-only。禁止把 Tensor 传给 CosyVoice load_wav。"""
    register_result = False
    if spk_id not in _registered_spks:
        try:
            register_result = cosyvoice_model.add_zero_shot_spk(
                prompt_text or "", wav_path, spk_id)
        except Exception as e:
            print(f"⚠️ add_zero_shot_spk 失败 persona={persona_id}: "
                  f"{type(e).__name__}: {e}")
            register_result = False
        if register_succeeded(register_result):
            _registered_spks.add(spk_id)
    elif spk_id in _registered_spks:
        register_result = True

    remembered = False
    if spk_id in _registered_spks:
        remembered = _remember_spk_emb(persona_id, spk_id)
        if remembered:
            _remember_flow_prompt(persona_id, spk_id)
            _scrub_spk_prompt(spk_id)

    extract_ok = remembered or (persona_id in _spk_emb_cache)
    if not extract_ok:
        extract_ok = _cache_embedding_from_wav(persona_id, spk_id, wav_path)
    if persona_id not in _spk_flow_prompt:
        _cache_flow_prompt_from_wav(persona_id, wav_path)

    mode = seed_speaker_mode(register_result=register_result, extract_ok=extract_ok)
    if extract_ok and not mode:
        mode = "embedding"
    return mode


def set_tts_persona(persona) -> bool:
    """切换当前合成音色。同人设同文件直接复用 clone 的 embedding。"""
    global _tts_prompt_arg, _tts_use_spk_id, _voice_spk_id, _voice_prompt_text, _voice_persona_id
    if _remote_on():
        _voice_spk_id = persona.tts.spk_id or persona.id
        _voice_prompt_text = persona.tts.prompt_text or ""
        _voice_persona_id = persona.id
        _tts_use_spk_id = True
        return True
    if cosyvoice_model is None:
        return False
    wav_src = persona.tts.prompt_wav
    prompt_text = persona.tts.prompt_text or ""
    if not wav_src:
        print(f"⚠️ 人设 {persona.id} 未配置参考音")
        return False
    spk_id = persona.tts.spk_id or persona.id
    if persona.id in _spk_emb_cache and _spk_raw.get(persona.id) == wav_src:
        _reset_cosy_hop()
        _voice_spk_id = spk_id
        _voice_prompt_text = prompt_text
        _voice_persona_id = persona.id
        _tts_use_spk_id = True
        if persona.id not in _spk_flow_prompt:
            wav_path = _spk_src.get(persona.id)
            if not wav_path:
                try:
                    wav_path = _ensure_prompt_wav(wav_src, persona.id)
                    _spk_src[persona.id] = wav_path
                except Exception:
                    wav_path = ""
            if wav_path:
                _cache_flow_prompt_from_wav(persona.id, wav_path)
        _scrub_spk_prompt(spk_id)
        return True
    try:
        wav_path = _ensure_prompt_wav(wav_src, persona.id)
    except Exception as e:
        print(f"⚠️ 人设 {persona.id} 参考音不可用: {e}")
        return False
    if _spk_raw.get(persona.id) and _spk_raw.get(persona.id) != wav_src:
        _registered_spks.discard(spk_id)
        _spk_emb_cache.pop(persona.id, None)
        _spk_flow_prompt.pop(persona.id, None)
        _spk_src.pop(persona.id, None)
    _voice_spk_id = spk_id
    _voice_prompt_text = prompt_text
    _voice_persona_id = persona.id
    _tts_prompt_arg = wav_path
    _tts_use_spk_id = False
    try:
        mode = _seed_spk_from_prompt(persona.id, spk_id, wav_path, prompt_text)
        if persona.id in _spk_emb_cache:
            _tts_use_spk_id = True
            _spk_src[persona.id] = wav_path
            _spk_raw[persona.id] = wav_src
            print(f"🎤 当前音色: {persona.id} spk_id={spk_id} mode={mode or 'embedding'} "
                  f"wav={wav_path} flow_prompt={'yes' if persona.id in _spk_flow_prompt else 'NO'}")
            return True
    except Exception as e:
        print(f"⚠️ 音色注册 {spk_id} 失败: {e}")
    print(f"⚠️ 人设 {persona.id} 无可用 embedding，拒绝合成")
    return False


def _patch_torchaudio_load():
    """torchaudio 2.9+ 的 load 忽略 backend=soundfile，缺 TorchCodec 时参考音读不进来，人设就没有 embedding。"""
    import torch
    import torchaudio
    if getattr(torchaudio.load, "_dh_soundfile", False):
        return

    def _load(path, *args, **kwargs):
        import numpy as np
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(np.ascontiguousarray(data.T))
        return wav, int(sr)

    _load._dh_soundfile = True
    torchaudio.load = _load


def _tts_log(msg: str, fail: bool = False):
    global _tts_last_error
    if fail:
        _tts_last_error = msg
    runtime_log("tts-load.log", msg)


def load_cosyvoice():
    with _load_lock:
        _load_cosyvoice_unlocked()


def _load_cosyvoice_unlocked():
    global cosyvoice_model, tts_executor, audio_cache
    global _tts_prompt_arg, _tts_prompt_mode, _tts_support_stream, _tts_use_spk_id, _load_wav_fn
    global _tts_last_error
    if _remote_on():
        from tts.remote import activate_remote_tts
        activate_remote_tts()
        _tts_log("[TTS] remote %s" % os.environ.get("TTS_BASE_URL", "http://127.0.0.1:8810"))
        return
    from tts.engine import use_qwen_tts
    if use_qwen_tts():
        from tts.qwen import activate as activate_qwen_tts
        activate_qwen_tts()
        return
    if cosyvoice_model is not None:
        print("🔁 CosyVoice 已在内存，跳过 CosyVoice3() 重载，补注册音色")
        _retry_register_speakers()
        if tts_ready():
            _tts_last_error = ""
        if tts_executor is None:
            tts_executor = ThreadPoolExecutor(max_workers=TTS_MAX_WORKERS)
        if ENABLE_AUDIO_CACHE and audio_cache is None:
            audio_cache = AudioCache(max_size=AUDIO_CACHE_SIZE)
        return
    _tts_last_error = ""
    _tts_prompt_arg = None
    _tts_prompt_mode = "unknown"
    _tts_support_stream = False
    _tts_use_spk_id = False
    _tts_log("[TTS] load start ENABLE_TTS=%s numpy=%s path=%s model=%s" % (
        ENABLE_TTS, np is not None, COSYVOICE_PATH, COSYVOICE_MODEL))

    if not ENABLE_TTS or np is None:
        _tts_log("[TTS] skip: ENABLE_TTS=%s numpy=%s" % (ENABLE_TTS, np is not None), fail=True)
        return

    import torch

    try:
        print("🔍 检查 CosyVoice 依赖...")
        try:
            import hyperpyyaml  # noqa: F401
            print("   ✅ hyperpyyaml 已安装")
        except ImportError:
            _tts_log("[TTS] FAIL hyperpyyaml missing", fail=True)
            print("\n📦 解决方案:\n   pip install hyperpyyaml onnxruntime-gpu modelscope\n")
            return

        cosyvoice_path = Path(COSYVOICE_PATH)
        model_path = Path(COSYVOICE_MODEL)
        if not cosyvoice_path.exists():
            _tts_log("[TTS] FAIL CosyVoice path missing: %s" % COSYVOICE_PATH, fail=True)
            return
        if not model_path.exists():
            _tts_log("[TTS] FAIL model path missing: %s" % COSYVOICE_MODEL, fail=True)
            return
        from persona.stage import stage
        from persona.registry import registry
        boot = stage.current()
        if not boot.voice_ready():
            for p in registry.enabled():
                if p.voice_ready():
                    print(f"⚠️ 默认人设无参考音，启动探测改用 {p.id}")
                    boot = p
                    break
        probe_wav = boot.tts.prompt_wav if boot.voice_ready() else PROMPT_WAV_PATH
        prompt_text = boot.tts.prompt_text or PROMPT_TEXT
        try:
            probe_wav = _ensure_prompt_wav(probe_wav, getattr(boot, "id", "") or "")
        except Exception:
            from config import MONET_LEGACY_PROMPT_WAV
            probe_wav = MONET_LEGACY_PROMPT_WAV if Path(MONET_LEGACY_PROMPT_WAV).exists() else PROMPT_WAV_PATH
        if not Path(probe_wav).exists():
            _tts_log("[TTS] FAIL prompt wav missing: %s" % probe_wav, fail=True)
            return

        use_cv2 = (model_path / "cosyvoice2.yaml").is_file()
        if use_cv2:
            required_files = ["cosyvoice2.yaml", "flow.pt", "hift.pt", "llm.pt",
                              "campplus.onnx", "speech_tokenizer_v2.onnx"]
        else:
            required_files = ["cosyvoice3.yaml", "flow.pt", "hift.pt", "llm.pt",
                              "campplus.onnx", "speech_tokenizer_v3.onnx"]
        missing = [f for f in required_files if not (model_path / f).exists()]
        if missing:
            _tts_log("[TTS] FAIL missing model files: %s" % missing, fail=True)
            return
        _tts_log("[TTS] files ok kind=%s wav=%s" % ("cosyvoice2" if use_cv2 else "cosyvoice3", probe_wav))

        sys.path.insert(0, COSYVOICE_PATH)
        sys.path.insert(0, str(Path(COSYVOICE_PATH) / "third_party" / "Matcha-TTS"))
        _patch_torchaudio_load()

        # 只清 CosyVoice 库缓存，不要卸掉本包 tts.cosyvoice
        for module_name in [k for k in sys.modules
                            if "cosyvoice" in k.lower() and not k.startswith("tts")]:
            del sys.modules[module_name]

        if use_cv2:
            from cosyvoice.cli.cosyvoice import CosyVoice2 as CosyVoiceCls
        else:
            from cosyvoice.cli.cosyvoice import CosyVoice3 as CosyVoiceCls
        from cosyvoice.utils.file_utils import load_wav
        _load_wav_fn = load_wav
        print("✅ CosyVoice 模块导入成功")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device}")
        if device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️ CUDA 不可用，将使用 CPU（会很慢）")

        import _locale
        _locale._getdefaultlocale = lambda *args: ['en_US', 'utf8']

        kind_name = "CosyVoice2-0.5B" if use_cv2 else "Fun-CosyVoice3-0.5B"
        print("📦 加载 %s ..." % kind_name)
        import os as _os
        use_trt = bool(COSYVOICE_LOAD_TRT and torch.cuda.is_available())
        _tts_log("[TTS] %s() begin cuda=%s trt=%s" % (
            CosyVoiceCls.__name__, torch.cuda.is_available(), use_trt))
        if use_trt:
            try:
                cosyvoice_model = CosyVoiceCls(COSYVOICE_MODEL, load_trt=True, fp16=True)
            except Exception as e:
                _tts_log("[TTS] WARN TensorRT 加载失败，退回 PyTorch Flow: %s: %s" % (
                    type(e).__name__, e))
                cosyvoice_model = None
        if cosyvoice_model is None:
            cosyvoice_model = CosyVoiceCls(COSYVOICE_MODEL, fp16=True)
        _tts_log("[TTS] %s() done sr=%s" % (CosyVoiceCls.__name__, getattr(cosyvoice_model, "sample_rate", "")))
        _install_token2wav_guard(cosyvoice_model.model)
    except Exception as e:
        import traceback
        traceback.print_exc()
        _tts_log("[TTS] FAIL %s: %s" % (type(e).__name__, e), fail=True)
        cosyvoice_model = None
        return

    # CosyVoice3() 已成功：探测 / GPU 迁移 / 预热失败都不得卸模型
    try:
        try:
            if device == 'cuda':
                print("🚀 将 PyTorch 模块移至 GPU...")
                success_count = 0
                for module_name in ['llm', 'flow', 'hift']:
                    if hasattr(cosyvoice_model.model, module_name):
                        module = getattr(cosyvoice_model.model, module_name)
                        if module is not None and isinstance(module, torch.nn.Module):
                            try:
                                module.to(device)
                                param = next(module.parameters(), None)
                                if param is not None and param.device.type == 'cuda':
                                    success_count += 1
                                    print(f"{module_name}: device={param.device}, dtype={param.dtype}")
                            except Exception as e:
                                print(f"{module_name} 移动失败: {e}")
                    if hasattr(cosyvoice_model.model, 'fp16'):
                        cosyvoice_model.model.fp16 = True
                    if hasattr(cosyvoice_model.model, 'device'):
                        cosyvoice_model.model.device = torch.device(device)
                print(f"{'所有' if success_count == 3 else f'⚠️ 仅 {success_count}/3 个'} PyTorch 模块在 GPU 上")

                import onnxruntime as ort
                cuda_provider = ("CUDAExecutionProvider", {"device_id": 0})
                tok_onnx = "speech_tokenizer_v2.onnx" if use_cv2 else "speech_tokenizer_v3.onnx"
                fname_map = {'campplus_session': 'campplus.onnx',
                             'speech_tokenizer_session': tok_onnx}
                if hasattr(cosyvoice_model, 'frontend'):
                    fe = cosyvoice_model.frontend
                    for attr, fname in fname_map.items():
                        sess = getattr(fe, attr, None)
                        if sess is None:
                            continue
                        if 'CUDAExecutionProvider' in sess.get_providers():
                            continue
                        onnx_path = getattr(sess, '_model_path', None)
                        if not onnx_path:
                            candidate = Path(COSYVOICE_MODEL) / fname
                            onnx_path = str(candidate) if candidate.exists() else None
                        if onnx_path:
                            try:
                                new_sess = ort.InferenceSession(
                                    onnx_path, providers=[cuda_provider, "CPUExecutionProvider"])
                                setattr(fe, attr, new_sess)
                                print(f"   ✅ {attr} → CUDAExecutionProvider")
                            except Exception as e:
                                print(f"   ⚠️ {attr} ONNX GPU 迁移失败: {e}")
        except Exception as e:
            print(f"⚠️ GPU 迁移失败，继续注册音色: {type(e).__name__}: {e}")

        # 线上合成已是 embedding-only。探测只注册音色 / 缓存 embedding，
        # 不要跑 inference_zero_shot（0 chunk / tensor load_wav 会误判失败）。
        # CosyVoice.load_wav 只接受文件路径，禁止把 Tensor 交给 add_zero_shot_spk。
        print("📦 注册参考音色（不跑 zero-shot 合成）...")
        boot_spk = boot.tts.spk_id or TTS_SPK_ID
        _tts_prompt_arg = probe_wav
        _tts_support_stream = False
        registered_mode = _seed_spk_from_prompt(
            boot.id, boot_spk, probe_wav, prompt_text)
        if registered_mode:
            _tts_use_spk_id = boot.id in _spk_emb_cache
            print(f"   ✅ 参考音已就绪 mode={registered_mode} spk_id={boot_spk}")
        else:
            _tts_log("[TTS] WARN speaker register failed, model kept wav=%s" % probe_wav)
            print("   ⚠️ 音色注册失败，模型保留，稍后按人设再试 embedding")

        # 按人设注册音色；默认舞台人设最后再 set 一次
        for p in registry.enabled():
            if p.id != boot.id and p.voice_ready():
                try:
                    set_tts_persona(p)
                except Exception as e:
                    print(f"⚠️ 预注册人设音色 {p.id} 失败: {e}")
        try:
            set_tts_persona(boot)
        except Exception as e:
            print(f"⚠️ 预注册人设音色 {boot.id} 失败: {e}")

        decision = apply_probe_decision(
            cosyvoice_model, registered_mode, _spk_emb_cache)
        _tts_prompt_mode = decision["prompt_mode"]
        _tts_log("[TTS] prompt mode=%s stream=%s emb=%s (register-only probe, model kept)" % (
            _tts_prompt_mode, _tts_support_stream, list(_spk_emb_cache.keys())))

        # 预热失败不得把已加载的大模型卸掉，否则 /ws/tts 会整段报「TTS 服务未启用」
        print("🔥 预热 CosyVoice...")
        t0 = time.time()
        warmup_chunks = []
        try:
            _devnull2 = open(_os.devnull, 'w', encoding='utf-8', errors='ignore')
            _old_stdout2 = sys.stdout
            sys.stdout = _devnull2
            try:
                with torch.no_grad():
                    for output in _tts_infer(boot.warmup_text or '测试。', persona=boot):
                        warmup_chunks.append(
                            output['tts_speech'] if isinstance(output, dict) else output)
            finally:
                sys.stdout = _old_stdout2
                _devnull2.close()
            warmup_time = time.time() - t0
            print(f"✅ 预热完成，耗时 {warmup_time:.2f}s，共 {len(warmup_chunks)} 段")
            if device == 'cuda' and warmup_chunks:
                audio_duration = sum(c.shape[-1] for c in warmup_chunks) / TTS_SAMPLE_RATE
                actual_rtf = warmup_time / audio_duration if audio_duration > 0 else 999
                if actual_rtf < 0.5:
                    print(f"✅ GPU 加速完美（RTF={actual_rtf:.2f}，音频 {audio_duration:.2f}s）")
                elif actual_rtf < 2.5:
                    print(f"✅ GPU 已生效（RTF={actual_rtf:.2f}，音频 {audio_duration:.2f}s）")
                else:
                    print(f"❌ GPU 未生效（RTF={actual_rtf:.2f}），请检查 ONNX Runtime GPU 安装")
        except Exception as e:
            print(f"⚠️ 预热失败，语音服务仍可用: {type(e).__name__}: {e}")

        tts_executor = ThreadPoolExecutor(max_workers=TTS_MAX_WORKERS)
        if ENABLE_AUDIO_CACHE:
            audio_cache = AudioCache(max_size=AUDIO_CACHE_SIZE)
        if _tts_prompt_mode == "unknown" and cosyvoice_model is not None:
            _tts_prompt_mode = resolve_prompt_mode("unknown", True)
        if tts_ready():
            _tts_last_error = ""
        _tts_log("[TTS] ok mode=%s stream=%s spk=%s infer=embedding ready=%s" % (
            _tts_prompt_mode, _tts_support_stream, _tts_use_spk_id, tts_ready()))
        print("✅ CosyVoice 加载成功")
    except Exception as e:
        import traceback
        traceback.print_exc()
        _tts_log("[TTS] WARN post-load %s: %s (model kept)" % (type(e).__name__, e),
                 fail=True)
        if cosyvoice_model is not None and _tts_prompt_mode == "unknown":
            _tts_prompt_mode = resolve_prompt_mode("unknown", True)
        if tts_executor is None and cosyvoice_model is not None:
            tts_executor = ThreadPoolExecutor(max_workers=TTS_MAX_WORKERS)
        if ENABLE_AUDIO_CACHE and audio_cache is None:
            audio_cache = AudioCache(max_size=AUDIO_CACHE_SIZE)

def synthesize_speech_pcm(text: str, speaker: str = "中文男",
                          cancel_ev: Optional[threading.Event] = None,
                          persona=None) -> Tuple[bytes, int]:
    from tts.engine import use_qwen_tts
    if use_qwen_tts():
        if cosyvoice_model is None:
            load_cosyvoice()
        from tts.qwen import qwen_ready, synthesize_pcm as _qwen_pcm
        p = resolve_tts_persona(persona)
        pid = p.id if p is not None else ""
        if not qwen_ready():
            raise RuntimeError(_tts_last_error or "Qwen3-TTS 未加载")
        if audio_cache:
            cached = audio_cache.get(text, speaker, fmt='pcm', persona_id=pid)
            if cached:
                pcm_bytes, _sr = cached
                if pcm_bytes and not _pcm_is_silent(pcm_bytes):
                    return cached
                audio_cache.drop(text, speaker, fmt='pcm', persona_id=pid)
        pcm_bytes, sr = _qwen_pcm(text, p, cancel_ev)
        if audio_cache and pcm_bytes and not _pcm_is_silent(pcm_bytes):
            audio_cache.put(text, speaker, (pcm_bytes, sr), fmt='pcm', persona_id=pid)
        return pcm_bytes, sr
    if _remote_on():
        if cosyvoice_model is None:
            load_cosyvoice()
        from tts.remote import speaker_id_of, synthesize_pcm as _remote_pcm
        p = resolve_tts_persona(persona)
        return _remote_pcm(text, speaker_id_of(p, speaker), cancel_ev)
    if not cosyvoice_model:
        raise RuntimeError("CosyVoice 未加载")
    p = resolve_tts_persona(persona)
    pid = p.id if p is not None else ""
    if audio_cache:
        cached = audio_cache.get(text, speaker, fmt='pcm', persona_id=pid)
        if cached:
            pcm_bytes, sr = cached
            if pcm_bytes and not _pcm_is_silent(pcm_bytes):
                return cached
            print(f"⚠️ 丢弃静音 PCM 缓存 persona={pid or '-'} '{text[:24]}'")
            audio_cache.drop(text, speaker, fmt='pcm', persona_id=pid)

    import torch
    segs = []
    sr = TTS_SAMPLE_RATE
    t0 = time.time()

    gen = _tts_infer(text, block=True, persona=p)   # 阻塞等 GPU，不会返回 None
    try:
        # CosyVoice token2wav 自己开 fp16 autocast，外层再套一层容易把 NaN 打成全 0 PCM
        with torch.no_grad():
            for output in gen:
                if not _speech_usable(output):
                    reason = _speech_fail_reason(output)
                    print(f"⚠️ TTS 分段不可用 persona={pid or '-'} '{text[:24]}' ({reason})")
                    continue
                if isinstance(output, dict):
                    sr = output.get('sample_rate', TTS_SAMPLE_RATE)
                    speech = output['tts_speech']
                else:
                    speech = output
                arr, _n_bad = _as_1d_float32(speech)
                segs.append(arr)
                if cancel_ev is not None and cancel_ev.is_set():
                    break
    finally:
        gen.close()   # 无论 break/异常，确保信号量立即释放

    if not segs or (cancel_ev is not None and cancel_ev.is_set()):
        if not segs:
            print(f"⚠️ TTS 无 PCM persona={pid or '-'} '{text[:30]}'")
        return b"", sr

    if _cosyvoice3_loaded():
        pcm_bytes = b"".join(_trim_whole(_float_to_pcm16(s), sr) for s in segs)
        audio_data = np.concatenate(segs) if len(segs) > 1 else segs[0]
    else:
        audio_data = np.concatenate(segs) if len(segs) > 1 else segs[0]
        pcm_bytes = _float_to_pcm16(audio_data)
    if not pcm_bytes:
        reason = _energy_fail_reason(audio_data)
        print(f"⚠️ TTS 整句静音/NaN，不下发 persona={pid or '-'} '{text[:30]}' ({reason})")
        return b"", sr
    duration = len(pcm_bytes) / 2.0 / float(sr)
    rtf = (time.time() - t0) / duration if duration > 0 else 0
    print(f"⏱️ TTS '{text[:10]}...' | 音频 {duration:.2f}s | 耗时 {time.time()-t0:.2f}s | RTF={rtf:.2f}")

    if audio_cache:
        audio_cache.put(text, speaker, (pcm_bytes, sr), fmt='pcm', persona_id=pid)
    return pcm_bytes, sr


def synthesize_batch_pcm(texts, speaker: str = "中文男",
                         cancel_ev: Optional[threading.Event] = None,
                         persona=None) -> list:
    """Qwen3-TTS 多句一批合成，返回与 texts 等长的 [(pcm, sr)]。命中缓存的句子不进 GPU。

    其他引擎没有批量接口，逐句走 synthesize_speech_pcm。
    """
    from tts.engine import use_qwen_tts
    items = list(texts or [])
    if not use_qwen_tts():
        out = []
        for t in items:
            if cancel_ev is not None and cancel_ev.is_set():
                out.append((b"", TTS_SAMPLE_RATE))
                continue
            out.append(synthesize_speech_pcm(t, speaker, cancel_ev, persona=persona))
        return out
    if cosyvoice_model is None:
        load_cosyvoice()
    from tts.qwen import qwen_ready, synthesize_batch as _qwen_batch
    if not qwen_ready():
        raise RuntimeError(_tts_last_error or "Qwen3-TTS 未加载")
    p = resolve_tts_persona(persona)
    pid = p.id if p is not None else ""
    out = [(b"", tts_output_rate()) for _ in items]
    miss = []
    for i, t in enumerate(items):
        cached = audio_cache.get(t, speaker, fmt='pcm', persona_id=pid) if audio_cache else None
        if cached and cached[0] and not _pcm_is_silent(cached[0]):
            out[i] = cached
        elif (t or "").strip():
            miss.append(i)
    if miss:
        got = _qwen_batch([items[i] for i in miss], p, cancel_ev)
        for i, (pcm_bytes, sr) in zip(miss, got):
            out[i] = (pcm_bytes, sr)
            if audio_cache and pcm_bytes and not _pcm_is_silent(pcm_bytes):
                audio_cache.put(items[i], speaker, (pcm_bytes, sr), fmt='pcm', persona_id=pid)
    return out


def prefetch_batch(texts, speaker: str = "中文男",
                   cancel_ev: Optional[threading.Event] = None, persona=None) -> int:
    """把排队中的几句一次合成进 audio_cache，随后逐句 push_sentence_pcm 直接命中缓存。

    只对 Qwen3-TTS 生效（有批量接口）。返回成功预取的句数，失败不抛，由逐句合成兜底。
    """
    from tts.engine import use_qwen_tts
    items = [t for t in (texts or []) if (t or "").strip()]
    if len(items) < 2 or not audio_cache or not use_qwen_tts():
        return 0
    try:
        got = synthesize_batch_pcm(items, speaker, cancel_ev, persona=persona)
    except Exception as e:
        _tts_log("[TTS] qwen 预取失败，逐句合成兜底: %s: %s" % (type(e).__name__, e), fail=True)
        return 0
    return sum(1 for pcm_bytes, _sr in got if pcm_bytes)

async def synthesize_speech_streaming(websocket, safe_send, text: str, speaker: str,
                                      session_id: str, seq: int,
                                      cancel_ev: threading.Event,
                                      subtitle_text: str = None,
                                      subtitle_index: int = None,
                                      on_pcm: Optional[Callable[[bytes], None]] = None,
                                      persona=None
                                      ) -> Tuple[int, bool]:
    """流式合成并分块发送。取消判断统一走 cancel_ev（不受会话字典 pop 影响）。
    subtitle_text/subtitle_index：若提供，在首个有效 PCM chunk 发出前先发 tts_sentence。
    全 0 / 空 / 超大静音块一律不下发，避免 Unity 把 AAAA... JSON 打进日志崩掉。"""
    if _remote_on() and cosyvoice_model is None:
        load_cosyvoice()
    if not cosyvoice_model:
        raise RuntimeError("CosyVoice 未加载")

    p = resolve_tts_persona(persona)
    pid = p.id if p is not None else ""
    chunk_size = _pcm_chunk_size()
    silent_skip_logged = False
    sent_pcm = False
    subtitle_sent = False

    async def emit_subtitle():
        nonlocal subtitle_sent
        if subtitle_text is not None and not subtitle_sent:
            await safe_send({"type": "tts_sentence", "session_id": session_id,
                             "index": subtitle_index, "text": subtitle_text})
            subtitle_sent = True

    async def emit_chunk(chunk: bytes) -> bool:
        """只发有能量的 PCM。全 0 块记一次日志，seq 不涨。"""
        nonlocal seq, silent_skip_logged, sent_pcm
        if not chunk:
            return False
        if len(chunk) > chunk_size:
            chunk = chunk[:chunk_size]
        peak_i = _pcm_peak(chunk)
        if peak_i < _SILENCE_PEAK_INT16:
            if not silent_skip_logged:
                print(f"⚠️ 丢弃全 0 PCM 分片 persona={pid or '-'} '{text[:24]}' "
                      f"bytes={len(chunk)} peak={peak_i}，不下发 AAAA tts_chunk")
                silent_skip_logged = True
            return False
        if on_pcm:
            on_pcm(chunk)
        if not sent_pcm:
            print(f"🔊 下发 tts_chunk persona={pid or '-'} '{text[:24]}' "
                  f"seq={seq} bytes={len(chunk)} peak={peak_i}")
        await safe_send({"type": "tts_chunk", "session_id": session_id,
                         "seq": seq, "data": base64.b64encode(chunk).decode()})
        seq += 1
        sent_pcm = True
        return True

    async def push_pcm_bytes(pcm_bytes: bytes, sr: int, pace: bool) -> bool:
        """按 chunk_size 切片推流。静音块跳过且不按静音时长等待。"""
        offset = 0
        while offset < len(pcm_bytes):
            if cancel_ev.is_set():
                return True
            chunk = pcm_bytes[offset:offset + chunk_size]
            sent_this = False
            if chunk:
                if not _pcm_is_silent(chunk) and not sent_pcm:
                    await emit_subtitle()
                sent_this = await emit_chunk(chunk)
            offset += chunk_size
            if pace and sent_this:
                chunk_sec = len(chunk) / 2.0 / float(sr or TTS_SAMPLE_RATE)
                await asyncio.sleep(max(0.01, chunk_sec * 0.85))
            elif not pace:
                await asyncio.sleep(0)
        return False

    from tts.engine import use_qwen_tts
    if use_qwen_tts():
        loop = asyncio.get_running_loop()
        try:
            pcm_bytes, sr = await loop.run_in_executor(
                None, lambda: synthesize_speech_pcm(
                    text, speaker, cancel_ev, persona=p))
        except Exception as e:
            msg = "Qwen3-TTS 失败 persona=%s '%s': %s" % (pid or "-", text[:24], e)
            print("⚠️ " + msg)
            _tts_log("[TTS] " + msg, fail=True)
            await safe_send({
                "type": "error",
                "session_id": session_id,
                "message": msg,
            })
            await emit_subtitle()
            return seq, False
        if not pcm_bytes:
            msg = "Qwen3-TTS 空音频 persona=%s '%s'" % (pid or "-", text[:24])
            print("⚠️ " + msg)
            _tts_log("[TTS] " + msg, fail=True)
            await safe_send({
                "type": "error",
                "session_id": session_id,
                "message": msg,
            })
            await emit_subtitle()
            return seq, cancel_ev.is_set()
        await emit_subtitle()
        print("🔊 Qwen3-TTS PCM persona=%s '%s' bytes=%s sr=%s"
              % (pid or "-", text[:24], len(pcm_bytes), sr))
        # 不按播放节奏等：等这一句播完才去合成下一句，句间空档就是整句合成耗时。
        # tts_end 由调用方的 PcmPlayClock 按真实时长延后。
        cancelled = await push_pcm_bytes(pcm_bytes, sr, pace=False)
        if not sent_pcm:
            msg = "Qwen3-TTS PCM 被静音过滤 persona=%s '%s' bytes=%s" % (
                pid or "-", text[:24], len(pcm_bytes))
            print("⚠️ " + msg)
            _tts_log("[TTS] " + msg, fail=True)
            await safe_send({
                "type": "error",
                "session_id": session_id,
                "message": msg,
            })
        return seq, cancelled

    # 缓存命中（pcm）直接推完。Unity 先收到 tts_end 的问题由调用方 PcmPlayClock 兜住
    # （按已推 PCM 真实时长等到播完才发 tts_end）；这里再匀速等，下一句就要排队干等。
    if audio_cache:
        cached = audio_cache.get(text, speaker, fmt='pcm', persona_id=pid)
        if cached:
            pcm_bytes, sr = cached
            if not pcm_bytes or _pcm_is_silent(pcm_bytes):
                print(f"⚠️ 丢弃空/静音 PCM 缓存 persona={pid or '-'} '{text[:24]}'")
                audio_cache.drop(text, speaker, fmt='pcm', persona_id=pid)
                cached = None
        if cached:
            pcm_bytes, sr = cached
            sr = sr or TTS_SAMPLE_RATE
            duration = len(pcm_bytes) / 2.0 / float(sr) if pcm_bytes else 0.0
            print(f"🔁 TTS 缓存命中 persona={pid or '-'} '{text[:15]}...' | {duration:.2f}s")
            await emit_subtitle()
            cancelled = await push_pcm_bytes(pcm_bytes, sr, pace=False)
            if not sent_pcm:
                print(f"⚠️ 缓存全是静音，已跳过 tts_chunk persona={pid or '-'} '{text[:24]}'")
            return seq, cancelled

    if _remote_on():
        from tts.remote import iter_pcm as _remote_iter, speaker_id_of
        loop = asyncio.get_running_loop()
        piece_q: asyncio.Queue = asyncio.Queue()
        spk = speaker_id_of(p, speaker)

        def _pull():
            # 独立进程每合成完一段就往 HTTP 里写一段，这里收到就转发，不等整句。
            try:
                for chunk, rate in _remote_iter(text, spk, cancel_ev):
                    loop.call_soon_threadsafe(piece_q.put_nowait, (chunk, rate))
            except Exception as e:
                loop.call_soon_threadsafe(piece_q.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(piece_q.put_nowait, None)

        threading.Thread(target=_pull, daemon=True).start()
        got = bytearray()
        sr = TTS_SAMPLE_RATE
        cancelled = False
        while True:
            item = await piece_q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                print(f"⚠️ 独立 TTS 失败 persona={pid or '-'} '{text[:24]}': {item}")
                continue
            if cancelled:
                continue
            chunk, sr = item
            got.extend(chunk)
            # 不按播放节奏等，tts_end 由调用方 PcmPlayClock 按真实时长延后。
            cancelled = await push_pcm_bytes(chunk, sr, pace=False)
        if not got or not sent_pcm:
            await emit_subtitle()
            return seq, cancelled or cancel_ev.is_set()
        pcm_bytes = bytes(got)
        if not cancelled and audio_cache and not _pcm_is_silent(pcm_bytes):
            audio_cache.put(text, speaker, (pcm_bytes, sr), fmt='pcm', persona_id=pid)
        return seq, cancelled

    loop = asyncio.get_running_loop()
    seg_queue: asyncio.Queue = asyncio.Queue()
    t0 = time.time()
    first_chunk_time = None
    all_pcm = bytearray()
    # CosyVoice3 每段是 _yield_tts_once 的整句输出，可以裁头尾；CosyVoice2 是 hop 分片，不能裁。
    whole_piece = _cosyvoice3_loaded()

    def _run_inference():
        gen = None
        try:
            gen = _tts_infer(text, block=True, cancel_ev=cancel_ev, persona=p)
            if gen is None:
                return
            import torch
            with torch.no_grad():
                for output in gen:
                    if cancel_ev.is_set():
                        break
                    packed = _detach_tts_output(output)
                    loop.call_soon_threadsafe(seg_queue.put_nowait, packed)
        except Exception as e:
            loop.call_soon_threadsafe(seg_queue.put_nowait, e)
        finally:
            try:
                if gen is not None:
                    gen.close()
            except Exception:
                pass
            loop.call_soon_threadsafe(seg_queue.put_nowait, None)

    threading.Thread(target=_run_inference, daemon=True).start()

    sr = TTS_SAMPLE_RATE
    cancelled = False

    while True:
        item = await seg_queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            print(f"⚠️ 推理线程异常: {item}")
            break
        if cancel_ev.is_set():
            cancelled = True
            break

        output = item
        if not _speech_usable(output):
            reason = _speech_fail_reason(output)
            print(f"⚠️ TTS 分段不可用 persona={pid or '-'} '{text[:24]}' ({reason})")
            continue
        if isinstance(output, dict):
            sr = output.get('sample_rate', TTS_SAMPLE_RATE)
            speech = output['tts_speech']
            n_bad_pre = int(output.get('n_bad') or 0)
        else:
            speech = output
            n_bad_pre = 0
        # reshape(-1)：stream=False 常见 [1, T]，squeeze 在 T=1 时会挤成 0 维
        seg, n_bad = _as_1d_float32(speech)
        n_bad = n_bad or n_bad_pre
        pcm_bytes = _float_to_pcm16(seg)
        if not pcm_bytes:
            reason = _energy_fail_reason(seg, n_bad)
            print(f"⚠️ TTS 空/静音 PCM 段 persona={pid or '-'} '{text[:24]}' ({reason})")
            continue
        if whole_piece:
            pcm_bytes = _trim_whole(pcm_bytes, sr)
        all_pcm.extend(pcm_bytes)
        peak, rms = _audio_peak_rms(seg)
        shape = tuple(speech.shape) if hasattr(speech, "shape") else ()
        dtype = getattr(speech, "dtype", type(seg))

        if first_chunk_time is None:
            first_chunk_time = time.time() - t0
            await emit_subtitle()
            print(f"🎤 首包延迟: {first_chunk_time:.2f}s | '{text[:15]}...' "
                  f"dtype={dtype} shape={shape} peak={peak:.4g} rms={rms:.4g} "
                  f"nan={n_bad} pcm={len(pcm_bytes)}B")

        cancelled = await push_pcm_bytes(pcm_bytes, sr, pace=False)
        if cancelled:
            break

    if sent_pcm and all_pcm and not cancelled:
        full_pcm = bytes(all_pcm)
        duration = len(full_pcm) / 2.0 / float(sr or TTS_SAMPLE_RATE)
        rtf = (time.time() - t0) / duration if duration > 0 else 0
        if audio_cache and full_pcm and not _pcm_is_silent(full_pcm):
            audio_cache.put(text, speaker, (full_pcm, sr), fmt='pcm', persona_id=pid)
        print(f"✅ 流式合成: '{text[:15]}...' | {duration:.2f}s | RTF={rtf:.2f}")
    elif not cancelled:
        await emit_subtitle()
        print(f"⚠️ TTS 整句无有效 PCM，已跳过 tts_chunk persona={pid or '-'} '{text[:30]}'")

    return seq, cancelled


async def push_sentence_pcm(websocket, safe_send, sentence, speaker,
                            session_id, seq, cancel_ev,
                            subtitle_text: str = None,
                            subtitle_index: int = None,
                            on_pcm: Optional[Callable[[bytes], None]] = None,
                            persona=None) -> Tuple[int, bool]:
    return await synthesize_speech_streaming(
        websocket, safe_send, sentence, speaker, session_id, seq, cancel_ev,
        subtitle_text=subtitle_text, subtitle_index=subtitle_index, on_pcm=on_pcm,
        persona=persona)

def synthesize_speech(text: str, speaker: str = "中文男", persona=None) -> bytes:
    """WAV 合成（须在 tts_executor 内调用）"""
    if _remote_on() and cosyvoice_model is None:
        load_cosyvoice()
    if not cosyvoice_model:
        raise RuntimeError("CosyVoice 未加载")
    p = resolve_tts_persona(persona)
    pid = p.id if p is not None else ""
    if audio_cache:
        cached = audio_cache.get(text, speaker, fmt='wav', persona_id=pid)
        if cached:
            return cached
    pcm_bytes, sr = synthesize_speech_pcm(text, speaker, persona=p)
    if not pcm_bytes:
        return b""
    buffer = io.BytesIO()
    wavfile.write(buffer, sr, np.frombuffer(pcm_bytes, dtype=np.int16))
    buffer.seek(0)
    audio_bytes = buffer.read()
    if audio_cache:
        audio_cache.put(text, speaker, audio_bytes, fmt='wav', persona_id=pid)
    return audio_bytes

def text_to_base64_audio(text: str, speaker: str = "中文男", persona=None) -> str:
    return base64.b64encode(synthesize_speech(text, speaker, persona=persona)).decode('utf-8')
