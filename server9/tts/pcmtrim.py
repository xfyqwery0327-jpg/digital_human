# -*- coding: utf-8 -*-
"""整句 PCM 头尾静音裁剪。只依赖 numpy，独立 TTS 进程也按文件路径加载这里。

CosyVoice3 / Qwen3-TTS 每次整句出声，句首句尾各带 0.3～0.6 秒留白。
逐句拼起来，每个逗号处会叠出将近一秒的空档。
只裁头尾，不动句中停顿；流式分片不要调用（会把半句话的尾巴当成静音裁掉）。
"""
from __future__ import annotations

_FRAME_MS = 10
# 帧峰值低于「整句峰值 × 该比例」且低于绝对门限才算静音。气声、轻辅音要保住。
_REL_THRESH = 0.03
_ABS_THRESH = 200


def trim_pcm16(pcm: bytes, sample_rate: int, head_ms: int = 40, tail_ms: int = 160) -> bytes:
    """int16 单声道 PCM 去掉头尾长静音，保留 head_ms / tail_ms 的余量。

    找不到有声帧、或参数为 0 时原样返回，宁可不裁也不吃掉整句。
    """
    if not pcm or sample_rate <= 0 or (head_ms <= 0 and tail_ms <= 0):
        return pcm
    try:
        import numpy as np
    except ImportError:
        return pcm
    usable = len(pcm) - (len(pcm) % 2)
    if usable <= 0:
        return pcm
    audio = np.frombuffer(pcm[:usable], dtype="<i2")
    hop = max(1, int(sample_rate * _FRAME_MS / 1000))
    n_frames = audio.size // hop
    if n_frames < 3:
        return pcm
    frames = np.abs(audio[:n_frames * hop].astype(np.int32)).reshape(n_frames, hop).max(axis=1)
    peak = int(frames.max()) if frames.size else 0
    if peak <= 0:
        return pcm
    thresh = min(_ABS_THRESH, max(1, int(peak * _REL_THRESH)))
    voiced = np.nonzero(frames > thresh)[0]
    if voiced.size == 0:
        return pcm
    start = int(voiced[0]) * hop
    end = min(audio.size, (int(voiced[-1]) + 1) * hop)
    if head_ms > 0:
        start = max(0, start - int(sample_rate * head_ms / 1000))
    else:
        start = 0
    if tail_ms > 0:
        end = min(audio.size, end + int(sample_rate * tail_ms / 1000))
    else:
        end = audio.size
    if start <= 0 and end >= audio.size:
        return pcm[:usable]
    return audio[start:end].tobytes()
