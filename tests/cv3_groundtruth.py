# -*- coding: utf-8 -*-
"""金标准测试：用官方 README 里的 asset/zero_shot_prompt.wav + 示例文本合成。
若这也出杂音 → 整个运行时（模型权重/代码版本）有问题，与人设数据无关。
  set COSYVOICE_PATH=C:\\ro\\CosyVoice
  set COSYVOICE_MODEL=C:\\ro\\CosyVoice\\pretrained_models\\Fun-CosyVoice3-0.5B
  set PYTHONPATH=C:\\digital_human\\server9
  python tests/cv3_groundtruth.py
"""
import os
import sys

import numpy as np

CV_PATH = os.environ.get("COSYVOICE_PATH", r"C:\ro\CosyVoice")
MODEL_DIR = os.environ.get(
    "COSYVOICE_MODEL", r"C:\ro\CosyVoice\pretrained_models\Fun-CosyVoice3-0.5B")
if CV_PATH not in sys.path:
    sys.path.insert(0, CV_PATH)

OUT_DIR = r"C:\digital_human\tests\out"
ASSET_WAV = os.path.join(CV_PATH, "asset", "zero_shot_prompt.wav")
CASES = [
    ("八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。",
     "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"),
    ("今天天气不错，我们一起去公园散步吧。",
     "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"),
]


def save_wav(path, pcm, sr):
    import wave
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


def main():
    from cosyvoice.cli.cosyvoice import CosyVoice3

    cos = CosyVoice3(MODEL_DIR, load_trt=False, load_vllm=False, fp16=False)
    print("asset wav:", ASSET_WAV, os.path.exists(ASSET_WAV))
    from tts.cosyvoice import _install_cv3_sampling_guard
    _install_cv3_sampling_guard(cos.model)

    for k, (text, prompt) in enumerate(CASES):
        chunks = []
        try:
            for out in cos.inference_zero_shot(text, prompt, ASSET_WAV,
                                               stream=False, speed=1.0,
                                               text_frontend=True):
                chunks.append(out["tts_speech"].reshape(-1).cpu().numpy())
        except Exception as e:
            print("case%d synth FAIL %s: %s" % (k, type(e).__name__, e))
        pcm = np.concatenate(chunks) if chunks else np.zeros(0)
        p = os.path.join(OUT_DIR, "cv3_gt_%d.wav" % k)
        sec, peak = save_wav(p, pcm, cos.sample_rate)
        print("case%d dur=%.2fs peak=%.2f -> %s" % (k, sec, peak, p))

    try:
        import torchaudio
        from funasr import AutoModel
        asr = AutoModel(model="paraformer-zh", device="cuda:0", disable_update=True)
        for k in range(len(CASES)):
            p = os.path.join(OUT_DIR, "cv3_gt_%d.wav" % k)
            if not os.path.exists(p):
                continue
            w, sr = torchaudio.load(p)
            p16 = os.path.join(OUT_DIR, "cv3_gt_%d_16k.wav" % k)
            torchaudio.save(p16, torchaudio.functional.resample(w, sr, 16000), 16000)
            r = asr.generate(input=p16)
            print("ASR case%d -> %s" % (k, r[0]["text"] if r else "EMPTY"))
    except Exception as e:
        print("ASR skip:", type(e).__name__, e)
    print("DONE")


if __name__ == "__main__":
    main()
