# -*- coding: utf-8 -*-
"""消融实验：判断 CosyVoice3 杂音坏在 LLM 还是坏在 flow/HiFT 声码器。

拿参考音自己抽出来的真实 speech token，直接喂 token2wav：
  - 出来的音能听/ASR 能识别 → 声码器是好的，杂音来自 LLM 生成的 token；
  - 出来的音还是杂音 → flow/HiFT（或 token 与声码器的码本对不上）坏。

另外对比 speech_tokenizer_v3.onnx 与 speech_tokenizer_v3.batch.onnx 的输出。

用法（robot 环境）：
  set COSYVOICE_PATH=C:\\ro\\CosyVoice
  set COSYVOICE_MODEL=C:\\ro\\CosyVoice\\pretrained_models\\Fun-CosyVoice3-0.5B
  set PYTHONPATH=C:\\digital_human\\server9
  python tests/cv3_vocoder_check.py
"""
import os
import sys

import numpy as np
import torch

CV_PATH = os.environ.get("COSYVOICE_PATH", r"C:\ro\CosyVoice")
MODEL_DIR = os.environ.get(
    "COSYVOICE_MODEL", r"C:\ro\CosyVoice\pretrained_models\Fun-CosyVoice3-0.5B")
if CV_PATH not in sys.path:
    sys.path.insert(0, CV_PATH)

MONET_WAV = r"C:\digital_human\voice\monet\monet-senlin_6to12s.wav"
OUT_DIR = r"C:\digital_human\tests\out"


def save_wav(path, pcm, sr):
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    peak = float(np.abs(pcm).max()) if pcm.size else 0.0
    if peak > 1.0:
        pcm = pcm / peak
    data = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(data.tobytes())
    return len(data) / float(sr), peak


def main():
    from cosyvoice.cli.cosyvoice import CosyVoice3

    cos = CosyVoice3(MODEL_DIR, load_trt=False, load_vllm=False, fp16=False)
    fe = cos.frontend
    model = cos.model
    print("llm dtype:", next(cos.model.llm.parameters()).dtype)

    # ---------- 1) 两个 speech tokenizer onnx 是否一致 ----------
    import onnxruntime as ort
    a_path = os.path.join(MODEL_DIR, "speech_tokenizer_v3.onnx")
    b_path = os.path.join(MODEL_DIR, "speech_tokenizer_v3.batch.onnx")
    tok_a, _ = fe._extract_speech_token(MONET_WAV)
    print("frontend tokenizer (%s): %s ..." % (os.path.basename(a_path), tok_a.reshape(-1)[:12].tolist()))
    if os.path.exists(b_path):
        try:
            sess = ort.InferenceSession(b_path, providers=["CPUExecutionProvider"])
            print("batch onnx inputs:", [(i.name, i.shape) for i in sess.get_inputs()],
                  "outputs:", [(o.name, o.shape) for o in sess.get_outputs()])
        except Exception as e:
            print("batch onnx load FAIL:", type(e).__name__, e)

    # ---------- 2) 真实 token 直接过 token2wav ----------
    speech_token, _ = fe._extract_speech_token(MONET_WAV)
    speech_feat, _ = fe._extract_speech_feat(MONET_WAV)
    emb = fe._extract_spk_embedding(MONET_WAV)
    token_len = min(int(speech_feat.shape[1] / 2), int(speech_token.shape[1]))
    speech_feat = speech_feat[:, :2 * token_len]
    speech_token = speech_token[:, :token_len]
    print("真实 token n=%d feat=%d emb=%s" % (
        speech_token.shape[1], speech_feat.shape[1], tuple(emb.shape)))

    this_uuid = "vocoder-check"
    with model.lock:
        model.tts_speech_token_dict[this_uuid] = []
        model.llm_end_dict[this_uuid] = False
        model.hift_cache_dict[this_uuid] = None
    t0 = __import__("time").time()
    speech = model.token2wav(
        token=speech_token.to(model.device),
        prompt_token=speech_token.to(model.device),
        prompt_feat=speech_feat.to(model.device),
        embedding=emb.to(model.device),
        token_offset=0,
        uuid=this_uuid,
        stream=False,
        finalize=True,
        speed=1.0,
    )
    pcm = speech.reshape(-1).cpu().numpy()
    sec, peak = save_wav(os.path.join(OUT_DIR, "cv3_vocoder_real.wav"), pcm, cos.sample_rate)
    print("真实 token -> token2wav: %.2fs peak=%.2f wall=%.1fs" % (sec, peak, __import__("time").time() - t0))

    # ---------- 3) ASR ----------
    try:
        import torchaudio
        from funasr import AutoModel
        asr = AutoModel(model="paraformer-zh", device="cuda:0", disable_update=True)
        w, sr = torchaudio.load(os.path.join(OUT_DIR, "cv3_vocoder_real.wav"))
        p16 = os.path.join(OUT_DIR, "cv3_vocoder_real_16k.wav")
        torchaudio.save(p16, torchaudio.functional.resample(w, sr, 16000), 16000)
        r = asr.generate(input=p16)
        print("ASR 真实token声码 ->", r[0]["text"] if r else "EMPTY")
    except Exception as e:
        print("ASR skip:", type(e).__name__, e)
    print("DONE")


if __name__ == "__main__":
    main()
