# -*- coding: utf-8 -*-
"""对比 CosyVoice3 官方 inference_zero_shot 与 server9 手拼路径的音质/token 差异。

用法（robot 环境）：
  set COSYVOICE_PATH=C:\\ro\\CosyVoice
  set COSYVOICE_MODEL=C:\\ro\\CosyVoice\\pretrained_models\\Fun-CosyVoice3-0.5B
  set PYTHONPATH=C:\\digital_human\\server9
  python tests/cv3_compare.py
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

MONET_WAV = os.path.join(ROOT, "voice", "monet", "monet-senlin_6to12s.wav")
MONET_TEXT = "隆重登场的光影魔法森林。我们是这片森林的守护者。"
CV3_PREFIX = "You are a helpful assistant.<|endofprompt|>"
TEST_TEXT = "我是克洛德·莫奈，印象派的创始人之一，人们称我为印象派之父。"

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT_DIR, exist_ok=True)


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
    return dict(path=path, sec=len(data) / float(sr), peak=peak,
                rms=float(np.sqrt(np.mean(pcm ** 2))) if pcm.size else 0.0)


def main():
    from cosyvoice.cli.cosyvoice import CosyVoice3

    print("loading CosyVoice3 from", MODEL_DIR)
    t0 = time.time()
    cos = CosyVoice3(MODEL_DIR, load_trt=False, load_vllm=False, fp16=False)
    print("loaded %.1fs sr=%d" % (time.time() - t0, cos.sample_rate))

    fe = cos.frontend
    model = cos.model

    # ---------- 路径 A：官方 inference_zero_shot ----------
    # 官方约定：prompt_text 自带 "You are a helpful assistant.<|endofprompt|>" 前缀
    t0 = time.time()
    chunks = []
    for out in cos.inference_zero_shot(TEST_TEXT, CV3_PREFIX + MONET_TEXT,
                                       MONET_WAV, stream=False, speed=1.0,
                                       text_frontend=True):
        chunks.append(out["tts_speech"].reshape(-1).cpu().numpy())
    wall_a = time.time() - t0
    pcm_a = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    info_a = save_wav(os.path.join(OUT_DIR, "cv3_official.wav"), pcm_a,
                      cos.sample_rate)
    print("OFFICIAL wall=%.1fs %s" % (wall_a, info_a))

    # ---------- 路径 B：server9 手拼（cv3 分支） ----------
    # 复刻 _embedding_only_gen 的 cv3 分支 + _yield_tts_once
    import tts.cosyvoice as ttscv
    ttscv.cosyvoice_model = cos  # 生产里由 load_cosyvoice() 设置，脚本需手动对齐
    from tts.cosyvoice import (
        _cv3_prompt_tokens, _yield_tts_once, _clone_cpu, _empty_prompt_kwargs,
        _cache_flow_prompt_from_wav, _spk_flow_prompt, _spk_emb_cache,
        _reset_cosy_hop,
    )

    emb = fe._extract_spk_embedding(MONET_WAV)
    ok = _cache_flow_prompt_from_wav("monet", MONET_WAV)
    print("flow prompt cached:", ok, "keys:", list(_spk_flow_prompt.keys()))
    flow_pair = _spk_flow_prompt.get("monet")
    if flow_pair is None:
        raise SystemExit("no flow prompt extracted")

    _reset_cosy_hop()
    tts_text_token, tts_text_token_len = fe._extract_text_token(TEST_TEXT)
    prompt_kw = _empty_prompt_kwargs()
    prompt_kw["prompt_text"] = _cv3_prompt_tokens(fe, MONET_TEXT)
    prompt_kw["flow_prompt_speech_token"] = _clone_cpu(flow_pair[0])
    prompt_kw["prompt_speech_feat"] = _clone_cpu(flow_pair[1])
    prompt_kw["llm_prompt_speech_token"] = _clone_cpu(flow_pair[0])
    model_input = {
        "text": tts_text_token,
        "text_len": tts_text_token_len,
        "llm_embedding": _clone_cpu(emb),
        "flow_embedding": _clone_cpu(emb),
        **prompt_kw,
    }
    print("prompt_text tokens:", int(prompt_kw["prompt_text"].shape[1]),
          "llm_prompt_token:", int(prompt_kw["llm_prompt_speech_token"].shape[1]),
          "flow_prompt_token:", int(prompt_kw["flow_prompt_speech_token"].shape[1]),
          "prompt_feat:", int(prompt_kw["prompt_speech_feat"].shape[1]))

    t0 = time.time()
    chunks_b = []
    for out in _yield_tts_once(model, model_input, join_sec=120.0):
        chunks_b.append(out["tts_speech"].reshape(-1).cpu().numpy())
    wall_b = time.time() - t0
    pcm_b = np.concatenate(chunks_b) if chunks_b else np.zeros(0, dtype=np.float32)
    info_b = save_wav(os.path.join(OUT_DIR, "cv3_server9.wav"), pcm_b,
                      cos.sample_rate)
    print("SERVER9  wall=%.1fs %s" % (wall_b, info_b))

    # ---------- 对比 ----------
    def spec(pcm, sr):
        if pcm.size < 512:
            return {}
        n = 1 << int(np.log2(pcm.size))
        x = pcm[:n] * np.hanning(n)
        sp = np.abs(np.fft.rfft(x))
        f = np.fft.rfftfreq(n, 1.0 / sr)
        band = (f > 3000) & (f < 8000)
        total = float(sp.sum()) + 1e-9
        hi_ratio = float(sp[band].sum()) / total
        centroid = float((f * sp).sum() / total)
        return dict(hi_ratio=hi_ratio, centroid=centroid)

    sa, sb = spec(pcm_a, cos.sample_rate), spec(pcm_b, cos.sample_rate)
    print("spec official:", sa)
    print("spec server9:", sb)
    n = min(pcm_a.size, pcm_b.size)
    if n:
        corr = float(np.corrcoef(pcm_a[:n], pcm_b[:n])[0, 1])
        print("waveform corr=%.4f (len %d vs %d)" % (corr, pcm_a.size, pcm_b.size))

    # ---------- ASR 校验（人耳替代） ----------
    try:
        import torchaudio
        from funasr import AutoModel
        asr = AutoModel(model="paraformer-zh", device="cuda:0", disable_update=True)

        def _asr(path):
            w, sr = torchaudio.load(path)
            w16 = torchaudio.functional.resample(w, sr, 16000)
            p16 = path.replace(".wav", "_16k.wav")
            torchaudio.save(p16, w16, 16000)
            r = asr.generate(input=p16)
            return r[0]["text"] if r else "EMPTY"

        print("ASR official ->", _asr(os.path.join(OUT_DIR, "cv3_official.wav")))
        print("ASR server9  ->", _asr(os.path.join(OUT_DIR, "cv3_server9.wav")))
    except Exception as e:
        print("ASR skip:", type(e).__name__, e)
    print("DONE ->", OUT_DIR)


if __name__ == "__main__":
    main()
