# -*- coding: utf-8 -*-
"""验证修复方案：A=屏蔽 min_len 前的全部特殊 token；B=贪心解码。各跑 N 次。"""
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

cos = CosyVoice3(MODEL_DIR, load_trt=False, load_vllm=False, fp16=False)
fe = cos.frontend
llm = cos.model.llm
size = llm.speech_token_size

emb = fe._extract_spk_embedding(MONET_WAV)
prompt_text = fe._extract_text_token(CV3_PREFIX + MONET_TEXT)[0]
prompt_speech = fe._extract_speech_token(MONET_WAV)[0]
prompt_feat = fe._extract_speech_feat(MONET_WAV)[0]
tl = min(int(prompt_feat.shape[1] / 2), int(prompt_speech.shape[1]))
prompt_speech = prompt_speech[:, :tl]

MODE = sys.argv[1] if len(sys.argv) > 1 else "fix"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6

orig_sampling_ids = Qwen2LM.sampling_ids
stop_set = set(llm.stop_token_ids)


def patched(self, weighted_scores, decoded_tokens, sampling, ignore_eos=True):
    scores = weighted_scores.clone()
    if ignore_eos is True:
        # min_len 之前：屏蔽全部特殊 token（sos/eos/task/fill/其余196）
        for t in self.stop_token_ids:
            scores[t] = -float("inf")
    else:
        # min_len 之后：只允许 eos，屏蔽 sos/task_id/fill 等
        for t in self.stop_token_ids:
            if t != self.eos_token:
                scores[t] = -float("inf")
    if MODE == "greedy":
        return int(scores.argmax().item())
    return orig_sampling_ids(self, scores, decoded_tokens, sampling, ignore_eos)


Qwen2LM.sampling_ids = patched

for run in range(N):
    text = fe._extract_text_token(TEST_TEXT)[0]
    t0 = time.time()
    tokens = list(llm.inference(
        text=text, text_len=torch.tensor([text.shape[1]], dtype=torch.int32),
        prompt_text=prompt_text,
        prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32),
        prompt_speech_token=prompt_speech,
        prompt_speech_token_len=torch.tensor([prompt_speech.shape[1]], dtype=torch.int32),
        embedding=emb, uuid="probe%d" % run))
    # 统计：非法 token（特殊 token 泄露）、重复率
    special = sum(1 for t in tokens if int(t) in stop_set)
    rep = 0
    for i in range(1, len(tokens)):
        if int(tokens[i]) == int(tokens[i - 1]):
            rep += 1
    print("mode=%s run=%d gen=%d wall=%.1fs special=%d rep=%d(%.0f%%)" % (
        MODE, run, len(tokens), time.time() - t0, special, rep,
        100.0 * rep / max(len(tokens), 1)), flush=True)
