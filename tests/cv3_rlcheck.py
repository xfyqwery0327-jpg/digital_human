# -*- coding: utf-8 -*-
"""试 llm.rl.pt：2512 同时发了 base / RL 两个 checkpoint，llm.pt 可能是没调好的 base。"""
import os, sys
import numpy as np
CV_PATH = r"C:/ro/CosyVoice"
MODEL_DIR = r"C:/ro/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
sys.path.insert(0, CV_PATH)
OUT_DIR = r"C:/digital_human/tests/out"
ASSET_WAV = os.path.join(CV_PATH, "asset", "zero_shot_prompt.wav")
TEXT = "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。"
PROMPT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"

def save_wav(path, pcm, sr):
    import wave
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    peak = float(np.abs(pcm).max()) if pcm.size else 0.0
    if peak > 1.0: pcm = pcm / peak
    data = (np.clip(pcm, -1, 1) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(sr)); w.writeframes(data.tobytes())
    return len(data)/float(sr), peak

from cosyvoice.cli.cosyvoice import CosyVoice3
cos = CosyVoice3(MODEL_DIR, load_trt=False, load_vllm=False, fp16=False)
import torch
rl = os.path.join(MODEL_DIR, "llm.rl.pt")
sd = torch.load(rl, map_location="cpu", weights_only=True)
missing, unexpected = cos.model.llm.load_state_dict(sd, strict=False)
print("rl load missing=%d unexpected=%d" % (len(missing), len(unexpected)))
cos.model.llm.to(cos.model.device).eval()
from tts.cosyvoice import _install_cv3_sampling_guard
_install_cv3_sampling_guard(cos.model)
chunks = []
for out in cos.inference_zero_shot(TEXT, PROMPT, ASSET_WAV, stream=False, speed=1.0, text_frontend=True):
    chunks.append(out["tts_speech"].reshape(-1).cpu().numpy())
p = os.path.join(OUT_DIR, "cv3_rl.wav")
sec, peak = save_wav(p, np.concatenate(chunks) if chunks else np.zeros(0), cos.sample_rate)
print("RL dur=%.2fs peak=%.2f -> %s" % (sec, peak, p))
import torchaudio
from funasr import AutoModel
asr = AutoModel(model="paraformer-zh", device="cuda:0", disable_update=True)
w, sr = torchaudio.load(p)
p16 = os.path.join(OUT_DIR, "cv3_rl_16k.wav")
torchaudio.save(p16, torchaudio.functional.resample(w, sr, 16000), 16000)
r = asr.generate(input=p16)
print("ASR RL ->", r[0]["text"] if r else "EMPTY")
print("DONE")
