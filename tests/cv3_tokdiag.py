# -*- coding: utf-8 -*-
"""诊断 CosyVoice3 杂音：对比「真实语音 token」与「LLM 生成 token」的分布。

若生成 token 分布与真实语音 token 分布差异巨大 → LLM（Qwen2 encoder/decoder）坏；
若分布一致 → 坏在 flow/HiFT 声码器。

用法（robot 环境）：
  set COSYVOICE_PATH=C:\\ro\\CosyVoice
  set COSYVOICE_MODEL=C:\\ro\\CosyVoice\\pretrained_models\\Fun-CosyVoice3-0.5B
  set PYTHONPATH=C:\\digital_human\\server9
  python tests/cv3_tokdiag.py
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
MONET_TEXT = "隆重登场的光影魔法森林。我们是这片森林的守护者。"
CV3_PREFIX = "You are a helpful assistant.<|endofprompt|>"
TEST_TEXT = "我是克洛德·莫奈，印象派的创始人之一，人们称我为印象派之父。"


def stats(name, tok):
    tok = np.asarray(tok, dtype=np.int64).reshape(-1)
    tok = tok[(tok >= 0) & (tok < 6561)]
    hist = np.bincount(tok, minlength=6561).astype(np.float64)
    p = hist / max(hist.sum(), 1)
    nz = p[p > 0]
    ent = float(-(nz * np.log(nz)).sum())
    print("%-12s n=%4d uniq=%4d min=%4d max=%4d mean=%7.1f std=%6.1f entropy=%.2f bits (max %.2f)"
          % (name, tok.size, int((hist > 0).sum()), tok.min() if tok.size else -1,
             tok.max() if tok.size else -1, tok.mean() if tok.size else 0,
             tok.std() if tok.size else 0, ent, np.log2(6561)))
    return hist


def main():
    from cosyvoice.cli.cosyvoice import CosyVoice3

    cos = CosyVoice3(MODEL_DIR, load_trt=False, load_vllm=False, fp16=False)
    fe = cos.frontend
    llm = cos.model.llm
    print("llm type:", type(llm).__name__, "speech_token_size:", llm.speech_token_size,
          "sos/eos/task/fill:", llm.sos, llm.eos_token, llm.task_id, llm.fill_token,
          "stops:", len(llm.stop_token_ids))

    # 1) 文本 token 化是否正常
    tok, tlen = fe._extract_text_token(CV3_PREFIX + MONET_TEXT)
    flat = tok.reshape(-1).cpu().numpy()
    print("prompt_text tokens: n=%d min=%d max=%d 含151646=%s"
          % (flat.size, flat.min(), flat.max(), int((flat == 151646).sum())))
    ttok, _ = fe._extract_text_token(TEST_TEXT)
    tf = ttok.reshape(-1).cpu().numpy()
    print("target_text tokens: n=%d min=%d max=%d" % (tf.size, tf.min(), tf.max()))

    # 2) 真实语音 token（参考音）
    ref_tok, _ = fe._extract_speech_token(MONET_WAV)
    h_ref = stats("REF_REAL", ref_tok.reshape(-1).cpu().numpy())

    # 3) LLM 生成的 token（先原生，再装 server9 采样补丁各跑一次）
    captured = {}

    def _run(tag):
        captured[tag] = None
        orig_infer = llm.inference

        def _spy(*a, **k):
            gen = orig_infer(*a, **k)
            toks = []

            def _wrap():
                try:
                    for i in gen:
                        toks.append(i)
                        yield i
                finally:
                    captured[tag] = np.asarray(toks, dtype=np.int64).reshape(-1)
            return _wrap()

        llm.inference = _spy
        try:
            for out in cos.inference_zero_shot(TEST_TEXT, CV3_PREFIX + MONET_TEXT,
                                               MONET_WAV, stream=False, speed=1.0,
                                               text_frontend=True):
                pass
        except Exception as e:
            print("!! [%s] 合成阶段异常（不影响 token 统计）: %s: %s"
                  % (tag, type(e).__name__, e))
        finally:
            llm.inference = orig_infer

    _run("raw")
    from tts.cosyvoice import _install_cv3_sampling_guard
    _install_cv3_sampling_guard(cos.model)
    _run("patched")
    for tag in ("raw", "patched"):
        gen = captured.get(tag)
        if gen is None:
            print("!! [%s] 没抓到生成 token" % tag)
            continue
        print("[%s] generated n=%d 前40: %s" % (tag, gen.size, gen[:40].tolist()))
        print("[%s] generated 后20: %s" % (tag, gen[-20:].tolist()))
        print("[%s] special 次数: sos=%d eos=%d task=%d fill=%d"
              % (tag, int((gen == llm.sos).sum()), int((gen == llm.eos_token).sum()),
                 int((gen == llm.task_id).sum()), int((gen == llm.fill_token).sum())))
        h_gen = stats("GEN_" + tag, gen)

        # 4) 两个分布的 JS 散度 + top token 重合度
        def norm(h):
            s = h.sum()
            return h / s if s > 0 else h
        pr, pg = norm(h_ref), norm(h_gen)
        m = 0.5 * (pr + pg)

        def kl(a, b):
            mask = a > 0
            return float((a[mask] * np.log(a[mask] / b[mask])).sum())
        js = 0.5 * kl(pr, m) + 0.5 * kl(pg, m)
        print("[%s] JS(REF,GEN) = %.4f bits  (0=同分布, 1=完全不重叠)" % (tag, js))
        top_ref = set(np.argsort(-pr)[:50].tolist())
        top_gen = set(np.argsort(-pg)[:50].tolist())
        print("[%s] top50 重合: %d/50" % (tag, len(top_ref & top_gen)))
        print("[%s] REF top10: %s" % (tag, np.argsort(-pr)[:10].tolist()))
        print("[%s] GEN top10: %s" % (tag, np.argsort(-pg)[:10].tolist()))
    rng = np.random.default_rng(0)
    stats("RANDOM", rng.integers(0, 6561, size=300))
    print("DONE")


if __name__ == "__main__":
    main()
