# -*- coding: utf-8 -*-
"""量化 CosyVoice3LM 早停概率：连续跑 N 次同一输入，记录每次生成 token 数。"""
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

cos = CosyVoice3(MODEL_DIR, load_trt=False, load_vllm=False, fp16=True)
fe = cos.frontend
model = cos.model
llm = model.llm
print("llm class:", type(llm).__name__,
      "speech_token_size:", llm.speech_token_size,
      "stop_token_ids[:6]:", llm.stop_token_ids[:6],
      "sos/eos/task/fill:", llm.sos, llm.eos_token, llm.task_id, llm.fill_token)

# 拦截 sampling_ids：记录每步采样结果与 stop token 在分布中的排名
orig_sampling_ids = Qwen2LM.sampling_ids
log = []


def traced_sampling_ids(self, weighted_scores, decoded_tokens, sampling, ignore_eos=True):
    top_ids = orig_sampling_ids(self, weighted_scores, decoded_tokens, sampling, ignore_eos)
    is_stop = int(top_ids) in self.stop_token_ids
    rank = -1
    if is_stop:
        order = weighted_scores.argsort(descending=True)
        rank = int((order == int(top_ids)).nonzero()[0].item())
    log.append(dict(step=len(decoded_tokens), tok=int(top_ids), stop=is_stop,
                    ignore_eos=ignore_eos, rank=rank))
    return top_ids


Qwen2LM.sampling_ids = traced_sampling_ids

emb = fe._extract_spk_embedding(MONET_WAV)
prompt_text = fe._extract_text_token(CV3_PREFIX + MONET_TEXT)[0]
prompt_speech = fe._extract_speech_token(MONET_WAV)[0]
prompt_feat = fe._extract_speech_feat(MONET_WAV)[0]
# 官方对齐：resample_rate==24000 → truncate
size = llm.speech_token_size
tl = min(int(prompt_feat.shape[1] / 2), int(prompt_speech.shape[1]))
prompt_speech = prompt_speech[:, :tl]
prompt_feat = prompt_feat[:, :2 * tl]

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
for run in range(N):
    text = fe._extract_text_token(TEST_TEXT)[0]
    log.clear()
    t0 = time.time()
    tokens = list(llm.inference(
        text=text, text_len=torch.tensor([text.shape[1]], dtype=torch.int32),
        prompt_text=prompt_text,
        prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32),
        prompt_speech_token=prompt_speech,
        prompt_speech_token_len=torch.tensor([prompt_speech.shape[1]], dtype=torch.int32),
        embedding=emb, uuid="probe%d" % run))
    stops = [r for r in log if r["stop"]]
    early = [r for r in stops if r["step"] < 60]
    print("run=%d gen=%d wall=%.1fs stop_at=%s early60=%s ranks=%s" % (
        run, len(tokens), time.time() - t0,
        [r["step"] for r in stops][:5],
        [r["step"] for r in early][:5],
        [r["rank"] for r in stops][:5]), flush=True)
