# -*- coding: utf-8 -*-
"""深挖：为什么 server9 手拼 cv3 路径只生成 1.3s 语音（token 数异常少）。"""
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CV_PATH = os.environ.get("COSYVOICE_PATH", r"C:\ro\CosyVoice")
MODEL_DIR = os.environ.get(
    "COSYVOICE_MODEL", r"C:\ro\CosyVoice\pretrained_models\Fun-CosyVoice3-0.5B")
sys.path.insert(0, CV_PATH)

MONET_WAV = os.path.join(ROOT, "voice", "monet", "monet-senlin_6to12s.wav")
MONET_TEXT = "隆重登场的光影魔法森林。我们是这片森林的守护者。"
CV3_PREFIX = "You are a helpful assistant.<|endofprompt|>"
TEST_TEXT = "我是克洛德·莫奈，印象派的创始人之一，人们称我为印象派之父。"

from cosyvoice.cli.cosyvoice import CosyVoice3

cos = CosyVoice3(MODEL_DIR, load_trt=False, load_vllm=False, fp16=False)
fe = cos.frontend
model = cos.model
llm = model.llm

# ---- 拦截 llm.inference，打印输入与生成统计 ----
orig_inference = llm.inference
stats = []


def traced_inference(text, text_len, prompt_text, prompt_text_len,
                     prompt_speech_token, prompt_speech_token_len,
                     embedding, sampling=25, max_token_text_ratio=20,
                     min_token_text_ratio=2, uuid=""):
    total_text = int(text_len.item()) + int(prompt_text_len.item())
    min_len = int((total_text - int(prompt_text_len.item())) * min_token_text_ratio)
    max_len = int((total_text - int(prompt_text_len.item())) * max_token_text_ratio)
    rec = dict(tag=uuid[:6] or "?",
               text_tok=int(text.shape[1]), prompt_text_tok=int(prompt_text.shape[1]),
               prompt_speech_tok=int(prompt_speech_token.shape[1]),
               min_len=min_len, max_len=max_len)
    stats.append(rec)
    n = 0
    last = []
    for tok in orig_inference(text, text_len, prompt_text, prompt_text_len,
                              prompt_speech_token, prompt_speech_token_len,
                              embedding, sampling, max_token_text_ratio,
                              min_token_text_ratio, uuid):
        n += 1
        last.append(tok)
        yield tok
    rec["gen"] = n
    rec["last_tokens"] = [int(t) for t in last[-6:]]
    print("[trace]", rec, flush=True)


llm.inference = traced_inference


def run_official():
    print("=== OFFICIAL ===", flush=True)
    t0 = time.time()
    pcm = []
    for out in cos.inference_zero_shot(TEST_TEXT, CV3_PREFIX + MONET_TEXT,
                                       MONET_WAV, stream=False, speed=1.0):
        pcm.append(out["tts_speech"].reshape(-1).cpu().numpy())
    pcm = np.concatenate(pcm)
    print("official sec=%.2f wall=%.1f" % (pcm.size / 24000, time.time() - t0), flush=True)
    return pcm


def run_server9():
    print("=== SERVER9 ===", flush=True)
    import tts.cosyvoice as ttscv
    ttscv.cosyvoice_model = cos
    from tts.cosyvoice import (_cv3_prompt_tokens, _yield_tts_once,
                               _clone_cpu, _empty_prompt_kwargs,
                               _cache_flow_prompt_from_wav, _spk_flow_prompt,
                               _reset_cosy_hop)
    emb = fe._extract_spk_embedding(MONET_WAV)
    _cache_flow_prompt_from_wav("monet", MONET_WAV)
    flow_pair = _spk_flow_prompt["monet"]
    _reset_cosy_hop()
    text_tok, text_len = fe._extract_text_token(TEST_TEXT)
    prompt_kw = _empty_prompt_kwargs()
    prompt_kw["prompt_text"] = _cv3_prompt_tokens(fe, MONET_TEXT)
    prompt_kw["flow_prompt_speech_token"] = _clone_cpu(flow_pair[0])
    prompt_kw["prompt_speech_feat"] = _clone_cpu(flow_pair[1])
    prompt_kw["llm_prompt_speech_token"] = _clone_cpu(flow_pair[0])
    mi = dict(text=text_tok, text_len=text_len,
              llm_embedding=_clone_cpu(emb), flow_embedding=_clone_cpu(emb),
              **prompt_kw)
    t0 = time.time()
    pcm = []
    for out in _yield_tts_once(model, mi, join_sec=120.0):
        pcm.append(out["tts_speech"].reshape(-1).cpu().numpy())
    pcm = np.concatenate(pcm) if pcm else np.zeros(0)
    print("server9 sec=%.2f wall=%.1f" % (pcm.size / 24000, time.time() - t0), flush=True)
    return pcm


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "s9"):
        run_server9()
    if which in ("both", "off"):
        run_official()
    print("STATS:", stats, flush=True)
