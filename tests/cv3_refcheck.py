# -*- coding: utf-8 -*-
"""用官方参考音/参考句测试 CosyVoice3 采样稳定性（排除人设数据因素）。"""
import os
import sys
import time

import numpy as np
import torch

CV_PATH = os.environ.get("COSYVOICE_PATH", r"C:\ro\CosyVoice")
MODEL_DIR = os.environ.get(
    "COSYVOICE_MODEL", r"C:\ro\CosyVoice\pretrained_models\Fun-CosyVoice3-0.5B")
sys.path.insert(0, CV_PATH)

PREFIX = "You are a helpful assistant.<|endofprompt|>"
REF_WAV = os.path.normpath(os.path.join(CV_PATH, "asset", "zero_shot_prompt.wav"))
REF_TEXT = "希望你以后能够做的比我还好呦。"
TEST_TEXT = "我是克洛德·莫奈，印象派的创始人之一，人们称我为印象派之父。"

from cosyvoice.cli.cosyvoice import CosyVoice3
from cosyvoice.llm.llm import Qwen2LM

cos = CosyVoice3(MODEL_DIR, load_trt=False, load_vllm=False, fp16=True)
fe = cos.frontend
llm = cos.model.llm

orig_sampling_ids = Qwen2LM.sampling_ids
log = []


def traced(self, weighted_scores, decoded_tokens, sampling, ignore_eos=True):
    top_ids = orig_sampling_ids(self, weighted_scores, decoded_tokens, sampling, ignore_eos)
    log.append((len(decoded_tokens), int(top_ids), int(top_ids) in self.stop_token_ids))
    return top_ids


Qwen2LM.sampling_ids = traced

emb = fe._extract_spk_embedding(REF_WAV)
prompt_text = fe._extract_text_token(PREFIX + REF_TEXT)[0]
prompt_speech = fe._extract_speech_token(REF_WAV)[0]
prompt_feat = fe._extract_speech_feat(REF_WAV)[0]
tl = min(int(prompt_feat.shape[1] / 2), int(prompt_speech.shape[1]))
prompt_speech = prompt_speech[:, :tl]
print("ref prompt: text_tok=%d speech_tok=%d feat=%d" % (
    prompt_text.shape[1], prompt_speech.shape[1], prompt_feat.shape[1]), flush=True)

N = 6
for run in range(N):
    text = fe._extract_text_token(TEST_TEXT)[0]
    log.clear()
    tokens = list(llm.inference(
        text=text, text_len=torch.tensor([text.shape[1]], dtype=torch.int32),
        prompt_text=prompt_text,
        prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32),
        prompt_speech_token=prompt_speech,
        prompt_speech_token_len=torch.tensor([prompt_speech.shape[1]], dtype=torch.int32),
        embedding=emb, uuid="ref%d" % run))
    stops = [(s, t) for s, t, is_stop in log if is_stop]
    print("run=%d gen=%d stops=%s" % (run, len(tokens), stops[:3]), flush=True)
