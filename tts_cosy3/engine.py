# -*- coding: utf-8 -*-
"""推理核心。只在 TTS 进程里加载 CosyVoice3，禁止 CPU offload，禁止 vLLM。

音色沿用讲解进程里已经听顺的跨语种路径：LLM 只吃系统前缀，不把参考句再喂进去
（否则会把参考文本念出来）；Flow 用参考音的 token 和频谱把音色锁住。
每段合成使用 stream=False，段末 finish_sentence()。不要在多轮对话里复用 KV。
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import wave
from pathlib import Path

from tts_cosy3.paths import (
    project_root,
    resolve_cosyvoice_root,
    resolve_model_dir,
    speaker_cache_dir,
)
from tts_cosy3.split_text import split_tts_text
from tts_cosy3.vram import (
    finish_sentence,
    format_snapshot,
    install_reset_kv_cache,
    snapshot,
)

_CV3_PREFIX = "You are a helpful assistant.<|endofprompt|>"
_CV3_EOP_ID = 151646
_LOG = None


def log(msg: str) -> None:
    global _LOG
    text = msg if msg.endswith("\n") else msg + "\n"
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except Exception:
        pass
    if _LOG is None:
        return
    try:
        _LOG.write(text)
        _LOG.flush()
    except Exception:
        pass


def setup_log_file() -> None:
    global _LOG
    from tts_cosy3.paths import log_dir
    path = log_dir() / "tts-cosy3.log"
    _LOG = open(path, "a", encoding="utf-8", buffering=1)
    log(f"[TTS] log file {path}")


def _patch_torchaudio_load() -> None:
    """torchaudio 2.9 的 load 会忽略 backend=soundfile，Windows 上参考音读不进来。"""
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


def _spoken(text: str) -> str:
    fn = getattr(_spoken, "fn", None)
    if fn is None:
        fn = lambda s: (s or "").replace("〇", "零")  # noqa: E731
        path = project_root() / "server9" / "tts" / "speaktext.py"
        if path.is_file():
            import importlib.util
            spec = importlib.util.spec_from_file_location("dh_speaktext", path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                fn = mod.to_spoken_zh
        _spoken.fn = fn
    return fn(text or "")


def _int8_linear_cls():
    """权重量化成 int8 留在 GPU 上，前向再反量化。不把 outlier 丢到 CPU。"""
    import torch
    from torch import nn

    class Int8WeightLinear(nn.Module):
        def __init__(self, linear, compute_dtype):
            super().__init__()
            weight = linear.weight.detach()
            device = weight.device
            fp = weight.float()
            scale = fp.abs().amax(dim=1).clamp(min=1e-8) / 127.0
            quant = torch.clamp(torch.round(fp / scale.unsqueeze(1)), -127, 127).to(torch.int8)
            self.register_buffer("weight_q", quant.contiguous().to(device))
            self.register_buffer("scale", scale.to(dtype=compute_dtype, device=device).contiguous())
            if linear.bias is not None:
                bias = linear.bias.detach().to(dtype=compute_dtype, device=device).contiguous()
                self.register_buffer("bias_v", bias)
            else:
                self.bias_v = None
            self.in_features = linear.in_features
            self.out_features = linear.out_features

        def forward(self, x):
            import torch.nn.functional as F
            scale = self.scale.to(dtype=x.dtype).unsqueeze(1)
            weight = self.weight_q.to(dtype=x.dtype) * scale
            bias = None if self.bias_v is None else self.bias_v.to(dtype=x.dtype, device=x.device)
            return F.linear(x, weight, bias)

    return Int8WeightLinear


def _make_int8_linear(linear, compute_dtype):
    cls = getattr(_make_int8_linear, "cls", None)
    if cls is None:
        cls = _int8_linear_cls()
        _make_int8_linear.cls = cls
    return cls(linear, compute_dtype)


_SKIP_LINEAR = {"lm_head", "embed_tokens", "speech_embedding", "llm_decoder"}


def _replace_linear_int8(module, compute_dtype) -> int:
    import torch.nn as nn
    replaced = 0
    for name, child in list(module.named_children()):
        if name in _SKIP_LINEAR:
            continue
        if isinstance(child, nn.Linear):
            quant = _make_int8_linear(child, compute_dtype).to(child.weight.device)
            setattr(module, name, quant)
            replaced += 1
        else:
            replaced += _replace_linear_int8(child, compute_dtype)
    return replaced


def _cast_floating(module, dtype) -> None:
    for param in module.parameters():
        if param.is_floating_point() and param.dtype != dtype:
            param.data = param.data.to(dtype=dtype)


def _nbytes(module) -> int:
    total = 0
    for param in module.parameters():
        total += param.numel() * param.element_size()
    for buf in module.buffers():
        total += buf.numel() * buf.element_size()
    return total


def _assert_cuda(module) -> None:
    bad = []
    for name, param in module.named_parameters():
        if param.device.type != "cuda":
            bad.append(name)
            if len(bad) >= 8:
                break
    if bad:
        raise RuntimeError("检测到 CPU 权重，本服务禁止 offload: " + ", ".join(bad))


def _close_gen(gen) -> None:
    close = getattr(gen, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:
        pass


class TTSEngine:
    def __init__(self):
        self.cv = None
        self.sample_rate = 24000
        self.dtype_name = "fp16"
        self.int8 = False
        self.weight_mb = 0.0
        self.speakers = {}
        self._prefix = None
        self._lock = threading.Lock()
        self.busy = False
        self.last_request_end = time.time()
        self.ready = False

    def load(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用。TTS 必须常驻 GPU，禁止退回 CPU。")

        dtype_name = (os.environ.get("TTS_DTYPE") or "fp16").strip().lower()
        if dtype_name not in {"fp16", "bf16"}:
            dtype_name = "fp16"
        self.dtype_name = dtype_name
        compute = torch.float16 if dtype_name == "fp16" else torch.bfloat16

        cosy_root = resolve_cosyvoice_root()
        model_dir = resolve_model_dir()
        if not cosy_root:
            raise FileNotFoundError("找不到 CosyVoice 代码目录，请设置 COSYVOICE_PATH")
        sys.path.insert(0, cosy_root)
        matcha = str(Path(cosy_root) / "third_party" / "Matcha-TTS")
        if matcha not in sys.path:
            sys.path.insert(0, matcha)
        _patch_torchaudio_load()

        # 官方 fp16=True 只打开 autocast，不保证权重以半精度存放。
        # vLLM 会再占一块显存，和旁边的机器人 LLM 冲突，这里强制关闭。
        from cosyvoice.cli.cosyvoice import CosyVoice3
        log(f"[TTS] CosyVoice3() fp16 autocast model={model_dir}")
        self.cv = CosyVoice3(model_dir, load_trt=False, load_vllm=False, fp16=True)
        self.sample_rate = int(getattr(self.cv, "sample_rate", 24000) or 24000)

        core = self.cv.model
        if dtype_name == "bf16":
            _cast_floating(core.llm, torch.bfloat16)
            _cast_floating(core.flow, torch.float16)
            _cast_floating(core.hift, torch.float16)
            log("[TTS] LLM 权重 bf16，Flow/HiFT 保持 fp16（声码器走 bf16 会改变音色）")
        else:
            _cast_floating(core.llm, torch.float16)
            _cast_floating(core.flow, torch.float16)
            _cast_floating(core.hift, torch.float16)
            log("[TTS] llm/flow/hift 权重已转为 fp16 存储")

        use_int8 = (os.environ.get("TTS_INT8", "1").strip().lower() not in {"0", "false", "no"})
        scope = (os.environ.get("TTS_INT8_SCOPE") or "llm").strip().lower()
        if use_int8:
            body = getattr(getattr(core.llm, "llm", None), "model", None)
            if body is None:
                raise RuntimeError("找不到 Qwen2 主干，无法做 INT8。设 TTS_INT8=0 可先用半精度启动")
            n = _replace_linear_int8(body, compute if dtype_name == "bf16" else torch.float16)
            self.int8 = n > 0
            log(f"[TTS] INT8 权重量化 scope=llm linear={n}（无 CPU offload，lm_head/embedding 保持半精度）")
            if scope in {"all", "llm+flow", "flow"}:
                n_flow = _replace_linear_int8(core.flow, torch.float16)
                log(f"[TTS] INT8 额外量化 flow linear={n_flow}。这会改变音色，只在明确要压显存时打开")
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        install_reset_kv_cache(core)
        if not hasattr(self.cv, "reset_kv_cache"):
            import types
            self.cv.reset_kv_cache = types.MethodType(lambda self: self.model.reset_kv_cache(), self.cv)

        _assert_cuda(core.llm)
        _assert_cuda(core.flow)
        _assert_cuda(core.hift)
        self.weight_mb = _nbytes(core) / (1024 * 1024)
        log(
            f"[TTS] 权重大小 {self.weight_mb:.0f} MiB dtype={self.dtype_name} int8={self.int8} "
            f"sr={self.sample_rate}"
        )
        if self.weight_mb > 2048:
            log(
                "[TTS] 权重仍超过 2048 MiB。默认只量化了 Qwen2 主干。"
                "不要对 HiFT 做 INT8。若必须再压，设 TTS_INT8_SCOPE=llm+flow，并重新听音色。"
            )
        self._preload_personas()
        self._reload_cached_speakers()
        self._warmup()
        self.ready = True
        log(format_snapshot("loaded", snapshot()))

    def _warmup(self) -> None:
        if not self.speakers:
            log("[TTS] 没有预加载音色，跳过预热")
            return
        sid = next(iter(self.speakers))
        try:
            t0 = time.time()
            chunks = list(self.synthesize("测试。", sid, max_chars=36))
            log(f"[TTS] 预热 {sid} chunks={len(chunks)} {time.time() - t0:.2f}s")
        except Exception as exc:
            log(f"[TTS] 预热失败，服务仍可接受请求: {type(exc).__name__}: {exc}")
        finally:
            finish_sentence(self.cv.model)

    def _prefix_tokens(self):
        import torch
        if self._prefix is None:
            tok, _ = self.cv.frontend._extract_text_token(_CV3_PREFIX)
            flat = tok.detach().reshape(-1).cpu()
            if int((flat == _CV3_EOP_ID).sum().item()) < 1:
                raise RuntimeError("CosyVoice3 未编码出 <|endofprompt|>，拒绝合成")
            self._prefix = flat.to(dtype=torch.int32).reshape(1, -1)
        return self._prefix.clone()

    def _ensure_wav(self, src: str, speaker_id: str) -> str:
        path = Path(src)
        if not path.is_file():
            raise FileNotFoundError(src)
        dest = speaker_cache_dir() / f"{speaker_id}.wav"
        if path.suffix.lower() == ".wav":
            try:
                import soundfile as sf
                if sf.info(str(path)).duration <= 10.5:
                    # 已经是短 wav（例如莫奈 6–12 秒切片）就直接用，避免再裁一次把音色裁偏。
                    return str(path)
            except Exception:
                pass
        if dest.is_file() and dest.stat().st_mtime >= path.stat().st_mtime:
            try:
                import soundfile as sf
                if sf.info(str(dest)).duration <= 10.5:
                    return str(dest)
            except Exception:
                pass
        wav = sr = None
        try:
            import torchaudio
            wav, sr = torchaudio.load(str(path))
        except Exception as exc:
            log(f"[TTS] 直接读参考音失败，改用 ffmpeg: {exc}")
            import subprocess
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(path), "-ac", "1", "-ar", "16000", str(dest)],
                check=True, capture_output=True,
            )
            import torchaudio
            wav, sr = torchaudio.load(str(dest))
        if wav.dim() == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if int(sr) != 16000:
            import torchaudio
            wav = torchaudio.functional.resample(wav, int(sr), 16000)
            sr = 16000
        max_n = int(10 * sr)
        if int(wav.shape[-1]) > max_n:
            log(f"[TTS] 参考音超过 10 秒，只取前 10 秒 speaker={speaker_id}")
            wav = wav[..., :max_n]
        import torchaudio
        torchaudio.save(str(dest), wav, int(sr))
        return str(dest)

    def register_speaker(self, speaker_id: str, wav_path: str, prompt_text: str) -> None:
        speaker_id = (speaker_id or "").strip()
        if not speaker_id:
            raise ValueError("speaker_id 为空")
        prompt_text = (prompt_text or "").strip()
        if "<|endofprompt|>" not in prompt_text:
            prompt_text = _CV3_PREFIX + prompt_text
        wav_path = self._ensure_wav(wav_path, speaker_id)
        with self._lock:
            self.busy = True
            try:
                ok = self.cv.add_zero_shot_spk(prompt_text, wav_path, speaker_id)
                if ok is False or ok is None:
                    raise RuntimeError(f"add_zero_shot_spk 失败 speaker={speaker_id}")
                info = self.cv.frontend.spk2info.get(speaker_id) or {}
                if info.get("flow_prompt_speech_token") is None or info.get("llm_embedding") is None:
                    raise RuntimeError(f"音色登记后缺少 flow token 或 embedding speaker={speaker_id}")
                self.speakers[speaker_id] = {
                    "speaker_id": speaker_id,
                    "prompt_wav": wav_path,
                    "prompt_text": prompt_text,
                }
                meta = speaker_cache_dir() / f"{speaker_id}.json"
                meta.write_text(json.dumps(self.speakers[speaker_id], ensure_ascii=False), encoding="utf-8")
                log(f"[TTS] 音色已登记 speaker={speaker_id} wav={wav_path}")
            finally:
                self.busy = False
                self.last_request_end = time.time()
                finish_sentence(self.cv.model)

    def _preload_personas(self) -> None:
        if os.environ.get("TTS_PRELOAD_PERSONAS", "1").strip().lower() in {"0", "false", "no"}:
            return
        try:
            import yaml
        except ImportError:
            log("[TTS] 未安装 PyYAML，跳过 personas 预加载")
            return
        folder = project_root() / "personas"
        if not folder.is_dir():
            return
        for path in sorted(folder.glob("*.yaml")):
            if path.name == "registry.yaml":
                continue
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except Exception as exc:
                log(f"[TTS] 读人设失败 {path.name}: {exc}")
                continue
            if data.get("enabled") is False:
                continue
            tts = data.get("tts") or {}
            sid = str(data.get("id") or "").strip()
            wav = tts.get("prompt_wav") or ""
            if not sid or not wav:
                continue
            wav_path = wav if os.path.isabs(wav) else str(project_root() / wav)
            if not os.path.isfile(wav_path):
                log(f"[TTS] 人设 {sid} 参考音不存在: {wav_path}")
                continue
            try:
                self.register_speaker(sid, wav_path, tts.get("prompt_text") or "")
            except Exception as exc:
                log(f"[TTS] 预加载 {sid} 失败: {type(exc).__name__}: {exc}")

    def _reload_cached_speakers(self) -> None:
        for meta in speaker_cache_dir().glob("*.json"):
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            sid = data.get("speaker_id") or ""
            if sid in self.speakers:
                continue
            wav = data.get("prompt_wav") or ""
            if not wav or not os.path.isfile(wav):
                continue
            try:
                self.register_speaker(sid, wav, data.get("prompt_text") or "")
            except Exception as exc:
                log(f"[TTS] 恢复音色 {sid} 失败: {exc}")

    def _model_input(self, piece: str, speaker_id: str) -> dict:
        import torch
        info = self.cv.frontend.spk2info.get(speaker_id)
        if not info:
            raise KeyError(speaker_id)
        frontend = self.cv.frontend
        text_token, text_len = frontend._extract_text_token(piece)
        flow_tok = info.get("flow_prompt_speech_token")
        feat = info.get("prompt_speech_feat")
        llm_emb = info.get("llm_embedding") or info.get("embedding")
        flow_emb = info.get("flow_embedding") or llm_emb
        if flow_tok is None or feat is None or llm_emb is None or flow_emb is None:
            raise RuntimeError(f"音色 {speaker_id} 条件不完整")
        # 每次新建空 prompt。tts() 的默认张量是可变对象，复用会串上一句。
        return {
            "text": text_token,
            "text_len": text_len,
            "prompt_text": self._prefix_tokens(),
            "llm_prompt_speech_token": torch.zeros(1, 0, dtype=torch.int32),
            "flow_prompt_speech_token": flow_tok.detach().cpu().clone(),
            "prompt_speech_feat": feat.detach().cpu().clone(),
            "llm_embedding": llm_emb.detach().cpu().clone(),
            "flow_embedding": flow_emb.detach().cpu().clone(),
            "source_speech_token": torch.zeros(1, 0, dtype=torch.int32),
        }

    def synthesize(self, text: str, speaker_id: str, max_chars: int = 36):
        """逐段吐出 int16 PCM bytes。每段结束都 reset_kv_cache。"""
        if speaker_id not in self.speakers:
            raise KeyError(f"未知音色 {speaker_id}，已有: {', '.join(self.speakers) or '无'}")
        spoken = _spoken(text or "")
        chunks = split_tts_text(spoken, max_chars=max_chars)
        if not chunks:
            return
        if any(len(c) > max_chars for c in chunks):
            raise RuntimeError("分句失败，出现超长片段，已拒绝送入模型")
        lens = [len(c) for c in chunks]
        log(f"[TTS] 分片 n={len(chunks)} max={max_chars} lens={lens} speaker={speaker_id} preview={chunks[0][:20]!r}")
        log(format_snapshot("before", snapshot()))
        import torch
        model = self.cv.model
        with self._lock:
            self.busy = True
            try:
                for idx, piece in enumerate(chunks, start=1):
                    t0 = time.time()
                    pcm = bytearray()
                    gen = None
                    try:
                        gen = model.tts(**self._model_input(piece, speaker_id), stream=False, speed=1.0)
                        with torch.inference_mode():
                            for output in gen:
                                pcm.extend(_speech_to_pcm(output))
                    finally:
                        _close_gen(gen)
                        # 机器人单句生成：不保留这条句子的 KV、flow、hift。
                        model.reset_kv_cache()
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    elapsed = time.time() - t0
                    audio_sec = (len(pcm) / 2.0 / float(self.sample_rate)) if pcm else 0.0
                    log(
                        f"[TTS] chunk={idx}/{len(chunks)} chars={len(piece)} "
                        f"elapsed={elapsed:.2f}s audio={audio_sec:.2f}s text={piece[:24]!r}"
                    )
                    log(format_snapshot("after_chunk", snapshot()))
                    if pcm:
                        yield bytes(pcm)
            finally:
                self.busy = False
                self.last_request_end = time.time()
                log(format_snapshot("after", snapshot()))


def _speech_to_pcm(output) -> bytes:
    import numpy as np
    speech = output["tts_speech"] if isinstance(output, dict) else output
    if speech is None:
        return b""
    if hasattr(speech, "detach"):
        arr = speech.detach().float().reshape(-1).cpu().numpy()
    else:
        arr = np.reshape(np.asarray(speech), -1).astype("float32", copy=False)
    if arr.size == 0:
        return b""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak < 1e-4:
        return b""
    if peak > 8.0:
        pcm = np.clip(arr, -32767.0, 32767.0).astype(np.int16)
    else:
        pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
    return pcm.tobytes()


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm)
    return buf.getvalue()


def wav_stream_header(sample_rate: int) -> bytes:
    """data 长度写成 0xFFFFFFFF，边合成边发。多数播放器能接着读 PCM。"""
    import struct
    data_len = 0xFFFFFFFF
    byte_rate = int(sample_rate) * 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        data_len,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        int(sample_rate),
        byte_rate,
        2,
        16,
        b"data",
        data_len,
    )
    return header
