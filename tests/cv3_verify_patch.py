# -*- coding: utf-8 -*-
"""端到端验证 cv3 采样补丁：走 server9 真实路径（set_tts_persona + _tts_infer）。

用法（robot 环境）：
  set COSYVOICE_PATH=C:\\ro\\CosyVoice
  set COSYVOICE_MODEL=C:\\ro\\CosyVoice\\pretrained_models\\Fun-CosyVoice3-0.5B
  set PYTHONPATH=C:\\digital_human\\server9
  python tests/cv3_verify_patch.py [轮数]
"""
import os
import sys
import time
import wave

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CV_PATH = os.environ.get("COSYVOICE_PATH", r"C:\ro\CosyVoice")
MODEL_DIR = os.environ.get(
    "COSYVOICE_MODEL", r"C:\ro\CosyVoice\pretrained_models\Fun-CosyVoice3-0.5B")
if CV_PATH not in sys.path:
    sys.path.insert(0, CV_PATH)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT_DIR, exist_ok=True)

import tts.cosyvoice as ttscv
from tts.cosyvoice import (_install_cv3_sampling_guard, _tts_infer,
                           set_tts_persona, load_cosyvoice)

# 0) persona registry/stage 先 bootstrap（load_cosyvoice 内部读 stage.current()）
from persona.registry import registry
from persona.stage import stage
if not registry._items:
    registry.load()
stage.bootstrap()

# 1) 真实加载路径（含 token2wav guard + 采样补丁）
t0 = time.time()
load_cosyvoice()
print("load_cosyvoice %.1fs model=%s" % (time.time() - t0, ttscv.cosyvoice_model is not None))
assert ttscv.cosyvoice_model is not None, "模型没加载上"
_install_cv3_sampling_guard(ttscv.cosyvoice_model.model)  # 幂等

# 2) 真人设对象
monet = registry.require("monet")
assert set_tts_persona(monet), "set_tts_persona 失败"
print("persona ok:", ttscv._voice_persona_id, "prompt_text=", ttscv._voice_prompt_text[:30])

TEXTS = [
    "我是克洛德·莫奈，印象派的创始人之一，人们称我为印象派之父。",
    "我主张走出画室，在户外直接作画，捕捉阳光在自然中瞬息变化的光影。",
    "《睡莲》是我晚年最重要的系列，描绘了吉维尼花园里的水景。",
]


def save_wav(path, pcm, sr):
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    peak = float(np.abs(pcm).max()) if pcm.size else 0.0
    if peak > 1.0:
        pcm = pcm / peak
    data = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(data.tobytes())
    return len(data) / float(sr), peak


# 3) 逐段跑生产入口 _tts_infer，统计每段 PCM 时长
for i in range(N):
    text = TEXTS[i % len(TEXTS)]
    t0 = time.time()
    chunks = []
    for out in _tts_infer(text, block=True, persona=monet):
        sp = out["tts_speech"] if isinstance(out, dict) else out
        chunks.append(np.asarray(sp.reshape(-1).cpu().numpy(), dtype=np.float32))
    wall = time.time() - t0
    pcm = np.concatenate(chunks) if chunks else np.zeros(0)
    sec, peak = save_wav(os.path.join(OUT_DIR, "cv3_patch_%d.wav" % i), pcm, 24000)
    expect = len(text) / 4.6  # 约 4.6 字/秒
    print("run=%d chars=%d pcm=%.2fs wall=%.1fs peak=%.2f 期望≈%.1fs %s" % (
        i, len(text), sec, wall, peak, expect,
        "OK" if sec >= expect * 0.55 else "⚠️过短"), flush=True)
print("DONE")
