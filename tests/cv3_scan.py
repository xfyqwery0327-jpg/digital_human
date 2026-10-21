# -*- coding: utf-8 -*-
"""扫参：min/max_token_text_ratio + 特殊 token 屏蔽，找稳定区间。"""
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
from cosyvoice.llm.llm import Qwen2LM

MIN_RATIO = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
MAX_RATIO = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
N = int(sys.argv[3]) if len(sys.argv) > 3 else 10

cos = CosyVoice3(MODEL_DIR, load_trt=False, load_vllm=False, fp16=True)
fe = cos.frontend
llm = cos.model.llm

orig_sampling_ids = Qwen2LM.sampling_ids
orig_inference = Qwen2LM.inference


def patched_sampling_ids(self, weighted_scores, decoded_tokens, sampling, ignore_eos=True):
    scores = weighted_scores
    if ignore_eos is True:
        for t in self.stop_token_ids:
            scores[t] = -float("inf")
    else:
        for t in self.stop_token_ids:
            if t != self.eos_token:
                scores[t] = -float("inf")
    return orig_sampling_ids(self, scores, decoded_tokens, sampling, ignore_eos)


def patched_inference(self, text, text_len, prompt_text, prompt_text_len,
                      prompt_speech_token, prompt_speech_token_len, embedding,
                      sampling=25, max_token_text_ratio=20,
                      min_token_text_ratio=2, uuid=""):
    return orig_inference(self, text, text_len, prompt_text, prompt_text_len,
                          prompt_speech_token, prompt_speech_token_len, embedding,
                          sampling, MAX_RATIO, MIN_RATIO, uuid)


Qwen2LM.sampling_ids = patched_sampling_ids
Qwen2LM.inference = patched_inference

emb = fe._extract_spk_embedding(MONET_WAV)
prompt_text = fe._extract_text_token(CV3_PREFIX + MONET_TEXT)[0]
prompt_speech = fe._extract_speech_token(MONET_WAV)[0]
prompt_feat = fe._extract_speech_feat(MONET_WAV)[0]
tl = min(int(prompt_feat.shape[1] / 2), int(prompt_speech.shape[1]))
prompt_speech = prompt_speech[:, :tl]

gens = []
for run in range(N):
    text = fe._extract_text_token(TEST_TEXT)[0]
    tokens = list(llm.inference(
        text=text, text_len=torch.tensor([text.shape[1]], dtype=torch.int32),
        prompt_text=prompt_text,
        prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32),
        prompt_speech_token=prompt_speech,
        prompt_speech_token_len=torch.tensor([prompt_speech.shape[1]], dtype=torch.int32),
        embedding=emb, uuid="scan%d" % run))
    gens.append(len(tokens))
    print("min=%.1f max=%.1f run=%d gen=%d (%.1fs)" % (
        MIN_RATIO, MAX_RATIO, run, len(tokens), len(tokens) / 25.0), flush=True)
g = np.array(gens)
print("SUMMARY min=%.1f max=%.1f n=%d min=%d p50=%d max=%d 短于180=%d 达到上限=%d" % (
    MIN_RATIO, MAX_RATIO, N, g.min(), int(np.median(g)), g.max(),
    int((g < 180).sum()), int((g >= MAX_RATIO * 30).sum())), flush=True)
