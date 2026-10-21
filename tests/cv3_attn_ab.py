# -*- coding: utf-8 -*-
"""消融实验：Qwen2 encoder 在 sdpa 与 eager 下的输出差异。

BlankEN 的 config 写着 transformers_version 4.40.1，环境里是 4.57.3。
若同一份权重、同一份输入，两种 attention 实现的 hidden states 差异巨大，
说明 CosyVoice3 的 LLM 被新版 transformers 的 mask/RoPE 改动带崩，
生成的 speech token 就是杂音源头。

  set COSYVOICE_PATH=C:\\ro\\CosyVoice
  set COSYVOICE_MODEL=C:\\ro\\CosyVoice\\pretrained_models\\Fun-CosyVoice3-0.5B
  set PYTHONPATH=C:\\digital_human\\server9
  python tests/cv3_attn_ab.py
"""
import os
import sys

import numpy as np
import torch

CV_PATH = os.environ.get("COSYVOICE_PATH", r"C:\ro\\CosyVoice")
MODEL_DIR = os.environ.get(
    "COSYVOICE_MODEL", r"C:\\ro\\CosyVoice\pretrained_models\Fun-CosyVoice3-0.5B")
if CV_PATH not in sys.path:
    sys.path.insert(0, CV_PATH)

MONET_WAV = r"C:\digital_human\voice\monet\monet-senlin_6to12s.wav"
MONET_TEXT = "隆重登场的光影魔法森林。我们是这片森林的守护者。"
CV3_PREFIX = "You are a helpful assistant.<|endofprompt|>"
TEST_TEXT = "我是克洛德·莫奈，印象派的创始人之一，人们称我为印象派之父。"
OUT_DIR = r"C:\digital_human\tests\out"


def build_lm_input(llm, fe):
    text, text_len = fe._extract_text_token(TEST_TEXT)
    prompt_text, prompt_text_len = fe._extract_text_token(CV3_PREFIX + MONET_TEXT)
    speech_token, speech_token_len = fe._extract_speech_token(MONET_WAV)
    emb = fe._extract_spk_embedding(MONET_WAV)
    dev = llm.llm.model.device
    text = torch.concat([prompt_text, text], dim=1).to(dev)
    text_emb = llm.llm.model.model.embed_tokens(text)
    sos_emb = llm.speech_embedding.weight[llm.sos].reshape(1, 1, -1)
    task_emb = llm.speech_embedding.weight[llm.task_id].reshape(1, 1, -1)
    ps = speech_token.to(dev)
    lm_input = torch.concat([sos_emb, text_emb, task_emb, llm.speech_embedding(ps)], dim=1)
    return lm_input, emb.to(dev)


def main():
    from cosyvoice.cli.cosyvoice import CosyVoice3

    cos = CosyVoice3(MODEL_DIR, load_trt=False, load_vllm=False, fp16=False)
    llm = cos.model.llm
    fe = cos.frontend
    encoder = llm.llm.model
    print("attn impl:", encoder.config._attn_implementation)

    lm_input, emb = build_lm_input(llm, fe)
    T = lm_input.shape[1]
    print("lm_input:", tuple(lm_input.shape))

    with torch.inference_mode():
        # ---- sdpa（当前） ----
        masks = torch.tril(torch.ones((1, T, T), device=lm_input.device)).to(torch.bool)
        y_sdpa, _ = llm.llm.forward_one_step(lm_input, masks=masks, cache=None)

        # ---- eager ----
        from transformers import Qwen2ForCausalLM
        blanken = os.path.join(MODEL_DIR, "CosyVoice-BlankEN")
        enc_eager = Qwen2ForCausalLM.from_pretrained(
            blanken, attn_implementation="eager", local_files_only=True)
        missing, unexpected = enc_eager.load_state_dict(encoder.state_dict(), strict=False)
        print("eager load missing=%d unexpected=%d" % (len(missing), len(unexpected)))
        enc_eager = enc_eager.to(encoder.device).eval()
        saved = llm.llm.model
        llm.llm.model = enc_eager
        try:
            y_eager, _ = llm.llm.forward_one_step(lm_input, masks=masks, cache=None)
        finally:
            llm.llm.model = saved

    a = y_sdpa[:, -1].float().reshape(-1)
    b = y_eager[:, -1].float().reshape(-1)
    diff = (a - b).abs()
    cos_sim = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    print("last-position hidden: sdpa norm=%.3f eager norm=%.3f" % (a.norm(), b.norm()))
    print("  max|diff|=%.4f  mean|diff|=%.4f  cosine=%.4f" % (diff.max(), diff.mean(), cos_sim))

    # 整段 hidden 都看一眼（前向一次性）
    with torch.inference_mode():
        m2 = torch.ones((1, T), device=lm_input.device, dtype=torch.bool)
        o_sdpa = encoder(inputs_embeds=lm_input, attention_mask=m2, output_hidden_states=True, return_dict=True)
        llm.llm.model = enc_eager
        try:
            o_eager = enc_eager(inputs_embeds=lm_input, attention_mask=m2, output_hidden_states=True, return_dict=True)
        finally:
            llm.llm.model = saved
    hs = o_sdpa.hidden_states[-1].float()
    he = o_eager.hidden_states[-1].float()
    d = (hs - he).abs()
    print("full forward hidden: max|diff|=%.4f mean|diff|=%.4f cosine=%.4f"
          % (d.max(), d.mean(), float(torch.nn.functional.cosine_similarity(hs.reshape(-1), he.reshape(-1), dim=0))))

    # ---- decoder 输出对比 ----
    with torch.inference_mode():
        p_sdpa = llm.llm_decoder(y_sdpa[:, -1]).log_softmax(dim=-1)
        p_eager = llm.llm_decoder(y_eager[:, -1]).log_softmax(dim=-1)
    top_s = int(p_sdpa.argmax(dim=-1))
    top_e = int(p_eager.argmax(dim=-1))
    print("decoder top1: sdpa=%d eager=%d %s" % (top_s, top_e, "同" if top_s == top_e else "不同"))
    print("sdpa top5:", torch.topk(p_sdpa, 5).indices.reshape(-1).tolist())
    print("eager top5:", torch.topk(p_eager, 5).indices.reshape(-1).tolist())
    print("DONE")


if __name__ == "__main__":
    main()
