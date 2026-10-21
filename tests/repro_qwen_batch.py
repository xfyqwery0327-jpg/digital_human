# -*- coding: utf-8 -*-
"""复现 qwen batch 合成报错：unsupported operation: more than one element ..."""
import os
import sys
import time
import traceback

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from qwen_tts import Qwen3TTSModel

MODEL_DIR = r"C:\digital_human\models\Qwen3-TTS-12Hz-0.6B-Base"
PROMPT_WAV = r"C:\digital_human\voice\monet\monet-senlin_6to12s.wav"
PROMPT_TEXT = "隆重登场的光影魔法森林。我们是这片森林的守护者。"

print("loading model on CPU ...", flush=True)
t0 = time.time()
model = Qwen3TTSModel.from_pretrained(
    MODEL_DIR, attn_implementation="sdpa",
    device_map="cpu", dtype=torch.float32)
print("loaded in %.1fs sr=%s" % (time.time() - t0, getattr(model, "sample_rate", "?")), flush=True)

item = model.create_voice_clone_prompt(
    ref_audio=PROMPT_WAV, ref_text=PROMPT_TEXT, x_vector_only_mode=False)
print("prompt items:", type(item), len(item), flush=True)

pieces = [
    "人们称我为“印象派之父”。",
    "我主张走出画室，",
    "在户外直接作画，",
]

print("\n=== A) batch text + single prompt item (server9 当前用法) ===", flush=True)
try:
    import sys
    sys.path.insert(0, r"C:\digital_human\server9")
    from tts.qwen import _install_sdpa_mask_fix
    print("sdpa fix installed:", _install_sdpa_mask_fix(), flush=True)
except Exception as e:
    print("sdpa fix install FAIL:", type(e).__name__, e, flush=True)
try:
    wavs, sr = model.generate_voice_clone(
        text=list(pieces),
        language=["Chinese"] * len(pieces),
        voice_clone_prompt=item,
        non_streaming_mode=True,
        max_new_tokens=100,
    )
    print("OK batch ->", len(wavs), "wavs", flush=True)
    import numpy as np
    for k, w in enumerate(wavs):
        a = np.asarray(getattr(w, "audio", w), dtype=np.float32).reshape(-1)
        print("   wav[%d] samples=%d peak=%.3f" % (k, a.size, float(np.abs(a).max()) if a.size else 0.0), flush=True)
except Exception as e:
    print("FAIL:", type(e).__name__, str(e)[:300], flush=True)
    traceback.print_exc()

print("\n=== B) 每个 batch item 用独立 clone 的 prompt ===", flush=True)
try:
    import copy
    items = [copy.deepcopy(item[0]) for _ in pieces]
    wavs, sr = model.generate_voice_clone(
        text=list(pieces),
        language=["Chinese"] * len(pieces),
        voice_clone_prompt=items,
        non_streaming_mode=True,
        max_new_tokens=100,
    )
    print("OK cloned-prompt batch ->", len(wavs), "wavs", flush=True)
except Exception as e:
    print("FAIL:", type(e).__name__, str(e)[:300], flush=True)
    traceback.print_exc()

print("\n=== C) 单句（对照） ===", flush=True)
try:
    wavs, sr = model.generate_voice_clone(
        text=pieces[0],
        language="Chinese",
        voice_clone_prompt=item,
        non_streaming_mode=True,
        max_new_tokens=100,
    )
    print("OK single ->", len(wavs), "wavs", flush=True)
except Exception as e:
    print("FAIL:", type(e).__name__, str(e)[:300], flush=True)
