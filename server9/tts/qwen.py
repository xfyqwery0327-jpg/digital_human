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


def _install_sdpa_mask_fix() -> bool:
    """修 transformers 4.57.3 + torch<2.5 的 sdpa mask 崩溃。

    masking_utils.sdpa_mask_older_torch 里：
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, -1, -1, -1)
        ...
        causal_mask |= torch.all(~causal_mask, dim=-1, keepdim=True)
    expand 出来的是 stride-0 视图，batch>1 时原地 |= 直接抛
    "more than one element of the written-to tensor refers to a single memory
    location"。批量合成走到 speech_tokenizer.decode 就崩，只能退回逐句，
    每句重算参考音 + talker batch=1，RTF 2.2~2.8 追不上播放，播报就断断续续。
    clone() 之后每行独立修补，也正是这个补丁本来的语义（按行补全无人可注意的
    query，防 NaN）。
    """
    try:
        import torch
        import transformers.masking_utils as mu
    except Exception:
        return False
    if getattr(mu, "_s9_sdpa_mask_fix", False):
        return True
    need = ("prepare_padding_mask", "_ignore_causal_mask_sdpa", "_vmap_for_bhqkv",
            "causal_mask_function", "_is_torch_greater_or_equal_than_2_5")
    if not all(hasattr(mu, n) for n in need):
        return False

    def _fixed(batch_size, cache_position, kv_length, kv_offset=0,
               mask_function=None, attention_mask=None, local_size=None,
               allow_is_causal_skip=True, allow_torch_fix=True, **kwargs):
        q_length = cache_position.shape[0]
        padding_mask = mu.prepare_padding_mask(attention_mask, kv_length, kv_offset)
        if allow_is_causal_skip and mu._ignore_causal_mask_sdpa(
                padding_mask, q_length, kv_length, kv_offset, local_size):
            return None
        kv_arange = torch.arange(kv_length, device=cache_position.device)
        kv_arange += kv_offset
        fn = mask_function or mu.causal_mask_function
        causal_mask = mu._vmap_for_bhqkv(fn, bh_indices=False)(
            None, None, cache_position, kv_arange)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, -1, -1, -1)
        if padding_mask is not None:
            causal_mask = causal_mask * padding_mask[:, None, None, :]
        if not mu._is_torch_greater_or_equal_than_2_5 and allow_torch_fix:
            causal_mask = causal_mask.clone()
            causal_mask |= torch.all(~causal_mask, dim=-1, keepdim=True)
        return causal_mask

    mu.sdpa_mask_older_torch = _fixed
    if not mu._is_torch_greater_or_equal_than_2_6:
        mu.sdpa_mask = _fixed
        try:
            from transformers.masking_utils import AttentionMaskInterface
            AttentionMaskInterface._global_mapping["sdpa"] = _fixed
        except Exception:
            pass
    mu._s9_sdpa_mask_fix = True
    print("🩹 transformers sdpa mask 批量补丁已装 (torch<2.5, batch>1 不再崩)")
    return True


def _enable_fast_kernels() -> None:
    """打开 PyTorch 自带的 Flash SDPA 和 TF32。多占一点显存，单句前向更快。"""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    flash = getattr(torch.backends.cuda, "enable_flash_sdp", None)
    if flash is not None:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)


def _load_model(model_dir):
    import torch
    from qwen_tts import Qwen3TTSModel

    _install_sdpa_mask_fix()
    _enable_fast_kernels()
    use_cuda = bool(torch.cuda.is_available())
    kwargs = {
        "device_map": "cuda:0" if use_cuda else "cpu",
        "dtype": torch.bfloat16 if use_cuda else torch.float32,
    }
    import os
    # 装了 flash-attn 时可设 QWEN_TTS_ATTN=flash_attention_2，talker 每步更快。
    attn = (os.environ.get("QWEN_TTS_ATTN") or "sdpa").strip()
    try:
        model = Qwen3TTSModel.from_pretrained(
            str(model_dir), attn_implementation=attn, **kwargs)
    except TypeError:
        model = Qwen3TTSModel.from_pretrained(str(model_dir), **kwargs)
    try:
        param = next(model.model.parameters())
        dtype_name = str(param.dtype).replace("torch.", "")
        device_name = str(param.device)
    except Exception:
        dtype_name = "bfloat16" if use_cuda else "float32"
        device_name = "cuda:0" if use_cuda else "cpu"
    # 官方包没有 capture_graph。torch.compile 也关掉了 CUDA Graph，避免 RoPE 原地写坏图。
    _log_tts("[TTS] qwen weights loaded once dtype=%s device=%s attn=%s cudagraph=off capture_graph=off" % (
        dtype_name, device_name, attn))
    return model


def _compile_predictor(model) -> None:
    """用 Triton 编译 code predictor。

    每帧还要再跑十几步。关掉 CUDA Graph：RoPE 会原地改写上一轮的输出，
    图重放会读到被覆盖的张量。动态形状避免每换一句长度就重新编译。
    """
    import torch
    try:
        import torch._inductor.config as inductor_config
        knobs = getattr(inductor_config, "triton", None)
        if knobs is not None and hasattr(knobs, "cudagraphs"):
            knobs.cudagraphs = False
    except Exception:
        pass
    inner = model.model.talker.code_predictor.model
    inner.forward = torch.compile(inner.forward, mode="default", fullgraph=False, dynamic=True)


def _warmup_voice() -> None:
    """启动时跑一句短稿，把 CUDA 初始化和编译图的耗时留在开服务，不留在第一声。"""
    from persona.registry import registry
    from persona.runtime import bind
    if not registry._items:
        registry.load()
    persona = registry.get("monet")
    if persona is None:
        enabled = list(registry.enabled())
        persona = enabled[0] if enabled else None
    if persona is None:
        return
    text = (getattr(persona, "warmup_text", None) or "测试。").strip() or "测试。"
    with bind(persona):
        synthesize_pcm(text[:12], persona=persona)


def _prepare_runtime(model_dir) -> None:
    """编译预测器并预热。编译失败就重新加载未编译的模型，再预热一次。"""
    global _model, _last_error
    import tts.cosyvoice as cv
    from config import QWEN_TTS_COMPILE
    if QWEN_TTS_COMPILE and _model is not None:
        try:
            _compile_predictor(_model)
            _warmup_voice()
            _last_error = ""
            cv._tts_last_error = ""
            _log_tts("[TTS] qwen compile+warmup ok")
            return
        except Exception as e:
            _log_tts("[TTS] qwen compile failed, reload plain: %s: %s" % (type(e).__name__, e), fail=True)
            _model = _load_model(model_dir)
    try:
        _warmup_voice()
        _last_error = ""
        cv._tts_last_error = ""
        _log_tts("[TTS] qwen warmup ok")
    except Exception as e:
        _log_tts("[TTS] qwen warmup skip %s: %s" % (type(e).__name__, e))


def codec_budget(text: str) -> int:
    """按字数封顶 codec token。

    12Hz，正常口播大约每个字 4 个 token。短句不再保底 96（约 8 秒），
    结束符没对上时第一声也不会把预算跑满。封顶 1024。
    """
    n = max(1, len((text or "").strip()))
    return int(min(1024, max(48, n * 4 + 16)))


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


def _detach_prompt(item):
    """参考音只建一次。张量断开计算图，避免预处理结果被意外留在梯度链上。"""
    import torch
    items = item if isinstance(item, (list, tuple)) else [item]
    for one in items:
        for name in ("ref_code", "ref_spk_embedding"):
            val = getattr(one, name, None)
            if isinstance(val, torch.Tensor):
                setattr(one, name, val.detach())
    return item


def _prompt_for(model, persona):
    tts = getattr(persona, "tts", None) if persona is not None else None
    wav = (getattr(tts, "prompt_wav", None) or "").strip()
    ref = (getattr(tts, "prompt_text", None) or "").strip()
    pid = (getattr(persona, "id", None) or "default") if persona is not None else "default"
    if not wav:
        raise RuntimeError("人设 %s 没有参考音，Qwen3-TTS 无法克隆" % pid)
    # xvec_only：只留说话人向量。参考文本和参考码本不进每句 prefill。
    key = (pid, wav, "xvec")
    with _lock:
        cached = _prompts.get(key)
    if cached is not None:
        return cached
    item = _detach_prompt(model.create_voice_clone_prompt(
        ref_audio=wav,
        ref_text=ref or None,
        x_vector_only_mode=True,
    ))
    items = item if isinstance(item, (list, tuple)) else [item]
    for one in items:
        one.ref_text = None
        one.ref_code = None
        one.x_vector_only_mode = True
        one.icl_mode = False
    with _lock:
        _prompts[key] = item
    _log_tts("[TTS] qwen prompt cached once xvec_only=1 persona=%s ref_prefill=0" % pid)
    return item


def release_cuda_cache(reason: str = "") -> None:
    """一轮结束时归还空闲缓存。句与句之间的快满检查在 cuda_cache。"""
    from tts.cuda_cache import release_cuda_cache as _release
    _release(reason)


def _trim(pcm: bytes, rate: int) -> bytes:
    from config import TTS_TRIM_HEAD_MS, TTS_TRIM_TAIL_MS
    from tts.pcmtrim import trim_pcm16
    return trim_pcm16(pcm, rate, TTS_TRIM_HEAD_MS, TTS_TRIM_TAIL_MS)


def _generate(pieces, persona):
    """一次只合成当前这几句。调用方按一句送来，算完立刻推，不等下一批。"""
    from config import QWEN_TTS_GREEDY
    budget = max(codec_budget(p) for p in pieces)
    prompt = _prompt_for(_model, persona)
    greedy = {}
    if QWEN_TTS_GREEDY:
        greedy = {"do_sample": False, "subtalker_dosample": False}
    with _lock:
        _reset_talker_cache(_model)
        try:
            from rag.speech_lang import qwen_language_of
            langs = [qwen_language_of(p) for p in pieces]
            generated = _model.generate_voice_clone(
                text=list(pieces) if len(pieces) > 1 else pieces[0],
                language=langs if len(pieces) > 1 else langs[0],
                voice_clone_prompt=prompt,
                non_streaming_mode=True,
                max_new_tokens=budget,
                **greedy,
            )
        finally:
            _reset_talker_cache(_model)
    return _unpack_wavs(generated)


def _to_pcm(audio, rate: int) -> bytes:
    pcm = _float_to_pcm16(audio) if audio is not None else b""
    return _trim(pcm, rate) if pcm else b""


def _pcm_peak(pcm: bytes) -> int:
    if not pcm:
        return 0
    import numpy as np
    samples = np.frombuffer(pcm, dtype="<i2")
    return int(np.max(np.abs(samples.astype(np.int32)))) if samples.size else 0


def _synthesize_piece(piece: str, persona, cancel_ev):
    """一句一算。每句前后清 RoPE，避免上一句的位置偏移把结束符顶飞。"""
    import time
    import os
    t0 = time.time()
    budget = codec_budget(piece)
    lane = os.environ.get("QWEN_TTS_LANE") or "0"
    _log_tts("[TTS] qwen synth begin lane=%s chars=%s budget=%s text=%s" % (
        lane, len(piece), budget, piece[:40]))
    try:
        wavs, sr = _generate([piece], persona)
    except Exception as e:
        msg = "qwen synth FAIL %s: %s text=%s" % (type(e).__name__, e, piece[:40])
        _log_tts("[TTS] " + msg, fail=True)
        raise RuntimeError(msg) from e
    if cancel_ev is not None and cancel_ev.is_set():
        return b"", _sample_rate
    rate = int(sr or _sample_rate)
    pcm = _to_pcm(wavs[0] if wavs else None, rate)
    peak = _pcm_peak(pcm)
    _log_tts("[TTS] qwen synth done chars=%s pcm=%s sr=%s peak=%s wall=%.2fs"
             % (len(piece), len(pcm), rate, peak, time.time() - t0))
    if not pcm or peak < 800:
        msg = "qwen synth quiet lane=%s chars=%s peak=%s text=%s" % (
            lane, len(piece), peak, piece[:40])
        _log_tts("[TTS] " + msg, fail=True)
        # === 新增：业务侧静音重算计数。原有丢弃和抛错保持不变。 ===
        from tts.qwen_batch import note_pcm_quiet_retry
        note_pcm_quiet_retry("piece", key=(piece or "")[:40])
        raise RuntimeError(msg)
    return pcm, rate


def synthesize_batch(texts, persona=None, cancel_ev=None, batch_size: int = 0) -> list:
    """多句并行合成，返回与 texts 等长的 [(pcm_bytes, sample_rate), ...]。

    某句没出声时 pcm 为 b""，由调用方决定是否单句重试。整批失败时退回逐句。
    """
    import time
    from config import QWEN_TTS_BATCH
    if _model is None:
        raise RuntimeError(_last_error or "Qwen3-TTS 未加载")
    items = [(t or "").strip() for t in (texts or [])]
    out = [(b"", _sample_rate) for _ in items]
    todo = [i for i, t in enumerate(items) if t]
    size = max(1, int(batch_size or QWEN_TTS_BATCH))
    for start in range(0, len(todo), size):
        if cancel_ev is not None and cancel_ev.is_set():
            break
        idxs = todo[start:start + size]
        pieces = [items[i] for i in idxs]
        t0 = time.time()
        _log_tts("[TTS] qwen batch begin n=%s chars=%s budget=%s first=%s"
                 % (len(pieces), [len(p) for p in pieces],
                    max(codec_budget(p) for p in pieces), pieces[0][:24]))
        try:
            wavs, sr = _generate(pieces, persona)
        except Exception as e:
            _log_tts("[TTS] qwen batch FAIL %s: %s n=%s，退回逐句"
                     % (type(e).__name__, e, len(pieces)), fail=True)
            for i in idxs:
                if cancel_ev is not None and cancel_ev.is_set():
                    break
                # === 新增：整批失败后退回逐句，记一笔静音/失败重算。 ===
                if "quiet" in str(e):
                    from tts.qwen_batch import note_pcm_quiet_retry
                    note_pcm_quiet_retry("batch_fallback", key=(items[i] or "")[:40])
                try:
                    out[i] = _synthesize_piece(items[i], persona, cancel_ev)
                except Exception:
                    out[i] = (b"", _sample_rate)
            continue
        rate = int(sr or _sample_rate)
        audio_sec = 0.0
        for k, i in enumerate(idxs):
            pcm = _to_pcm(wavs[k] if k < len(wavs) else None, rate)
            out[i] = (pcm, rate)
            audio_sec += len(pcm) / 2.0 / float(rate)
        wall = time.time() - t0
        _log_tts("[TTS] qwen batch done n=%s audio=%.2fs wall=%.2fs rtf=%.2f empty=%s"
                 % (len(pieces), audio_sec, wall, wall / audio_sec if audio_sec > 0 else 0.0,
                    sum(1 for i in idxs if not out[i][0])))
    return out


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
    import os
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
        cv._tts_log("[TTS] qwen3 model=%s sr=%s alloc=%s" % (
            model_dir, _sample_rate, os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or "-"))
        print("✅ Qwen3-TTS 已加载 %s sr=%s" % (model_dir, _sample_rate))
        _prepare_runtime(model_dir)
        if os.environ.get("QWEN_TTS_WORKER") != "1":
            # === 新增：默认单进程动态 batch。procs 才拉起旧的多进程池。 ===
            from config import QWEN_TTS_SCHED
            if QWEN_TTS_SCHED == "procs":
                from tts.qwen_pool import pool
                pool.ensure_started()
            else:
                from tts.qwen_batch import scheduler
                scheduler.ensure_started()
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
