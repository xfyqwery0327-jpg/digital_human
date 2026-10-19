# -*- coding: utf-8 -*-
"""TTS 加载决策与口播切分回归（不加载真实 CosyVoice GPU）。"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER9 = ROOT / "server9"
if str(SERVER9) not in sys.path:
    sys.path.insert(0, str(SERVER9))

from tts.load_policy import (  # noqa: E402
    apply_probe_decision,
    first_tensor,
    register_succeeded,
    resolve_prompt_mode,
    seed_speaker_mode,
    tts_health_ready,
)


class AmbiguousTensor:
    """模拟多值 Tensor：`if tensor` / `tensor or x` 会炸。"""

    def __init__(self, name="emb"):
        self.name = name

    def __bool__(self):
        raise RuntimeError(
            "Boolean value of Tensor with more than one value is ambiguous")

    def __len__(self):
        return 80


def test_first_tensor_or_does_not_crash():
    t = AmbiguousTensor("llm")
    other = AmbiguousTensor("flow")
    try:
        _ = t or other
        raise AssertionError("AmbiguousTensor.or 应当抛错")
    except RuntimeError as e:
        assert "Boolean value of Tensor" in str(e)
    assert first_tensor(t, other) is t
    assert first_tensor(None, t) is t
    assert first_tensor(None, None) is None

    info = {"llm_embedding": t, "embedding": other}
    llm_emb = first_tensor(info.get("llm_embedding"), info.get("embedding"))
    flow_emb = first_tensor(info.get("flow_embedding"), info.get("embedding"), llm_emb)
    assert llm_emb is t
    assert flow_emb is other


def test_probe_failure_does_not_unload_dummy_model():
    dummy = object()
    mode = seed_speaker_mode(register_result=False, extract_ok=False)
    decision = apply_probe_decision(dummy, mode, emb_cache={}, warmup_error="0 chunks")
    assert decision["unload"] is False
    assert decision["keep_model"] is True
    assert decision["prompt_mode"] == "embedding"
    assert dummy is not None

    cached = {"monet": (object(), object())}
    mode2 = seed_speaker_mode(register_result=False, extract_ok=True)
    decision2 = apply_probe_decision(dummy, mode2, cached, warmup_error="warmup boom")
    assert decision2["unload"] is False
    assert decision2["keep_model"] is True
    assert decision2["prompt_mode"] == "embedding"
    assert decision2["health_tts"] is True


def test_register_tensor_result_is_not_bool_tested():
    t = AmbiguousTensor("ok")
    try:
        if t:
            raise AssertionError("不该走到 if tensor")
    except RuntimeError:
        pass
    assert register_succeeded(t) is True
    assert seed_speaker_mode(register_result=t, extract_ok=False) == "path"
    assert register_succeeded(False) is False
    assert register_succeeded(None) is False
    assert register_succeeded(True) is True


def test_prompt_mode_not_unknown_when_model_up():
    assert resolve_prompt_mode("unknown", True) == "embedding"
    assert resolve_prompt_mode("path", True) == "path"
    assert resolve_prompt_mode("tensor", True) == "tensor"
    assert resolve_prompt_mode("", False) == "unknown"


def test_health_tts_requires_model_and_embedding():
    assert tts_health_ready(None, {"monet": 1}) is False
    assert tts_health_ready(object(), {}) is False
    assert tts_health_ready(object(), {"monet": (1, 2)}) is True


def test_load_cosyvoice_source_no_longer_unloads_on_probe():
    src = (SERVER9 / "tts" / "cosyvoice.py").read_text(encoding="utf-8")
    assert "FAIL prompt infer path/tensor both failed" not in src
    load_fn = src.split("def load_cosyvoice")[1].split("def synthesize_speech_pcm")[0]
    assert "cosyvoice_model.inference_zero_shot" not in load_fn
    assert "model kept" in load_fn
    # 构造失败才允许卸模型；探测失败路径不得再写 None
    assert "WARN speaker register failed, model kept" in load_fn


def test_pcm_play_clock_waits_last_sentence():
    import time
    from tts.playback import PcmPlayClock

    clock = PcmPlayClock(24000)
    # 24kHz int16 单声道：48000 bytes = 1.0s
    clock.add(b"\x00\x00" * 24000)
    remain = clock.remain(tail=0.0)
    assert 0.85 <= remain <= 1.05, remain
    assert abs(clock.duration - 1.0) < 1e-6
    time.sleep(0.2)
    remain2 = clock.remain(tail=0.0)
    assert remain2 < remain
    assert remain2 >= 0.6


def test_pack_tts_leftover_not_glued():
    from tts.cosyvoice import pack_tts_utterances
    from tts.speaktext import split_speak_chunks

    parts = ["你好。", "这是下一段。"]
    out = pack_tts_utterances(parts)
    assert out == ["你好。", "这是下一段。"]

    glued = pack_tts_utterances(["对了，", "睡莲很好看。"])
    assert glued[-1] == "睡莲很好看。"
    assert "对了，睡莲很好看。" not in glued

    chunks = split_speak_chunks("你好。我是莫奈，今天讲睡莲。")
    assert chunks[0] == "你好。"
    assert chunks[-1] == "今天讲睡莲。"
    assert any("我是莫奈" in c for c in chunks)


def test_spoken_zh_decimal_height_and_ticket():
    from tts.cosyvoice import pack_tts_utterances
    from tts.speaktext import split_speak_chunks, to_spoken_zh

    assert to_spoken_zh("1.4米") == "一点四米"
    assert to_spoken_zh("1-1.4米儿童") == "一到一点四米儿童"
    assert to_spoken_zh("19.9元") == "十九点九元"
    assert to_spoken_zh("1.04米") == "一点零四米"
    assert to_spoken_zh("1897年") == "一八九七年"
    assert to_spoken_zh("2026年") == "二零二六年"
    assert to_spoken_zh("二〇二六年") == "二零二六年"
    assert to_spoken_zh("29元") == "二十九元"
    assert to_spoken_zh("65周岁") == "六十五周岁"

    kb = (
        "有优惠票。1-1.4米儿童、65周岁以上人群可购买优惠票："
        "早鸟价29元，正价39元，原价89元。"
    )
    spoken = to_spoken_zh(kb)
    assert "1.4" not in spoken
    assert "一到一点四米" in spoken
    assert "六十五周岁" in spoken
    assert "二十九元" in spoken
    assert "三十九元" in spoken
    assert "八十九元" in spoken

    raw_chunks = split_speak_chunks("身高约1.4米。门票29元。")
    assert any("1.4米" in c for c in raw_chunks)
    assert not any(c.endswith("1.") for c in raw_chunks)
    assert not any(c.startswith("4米") for c in raw_chunks)

    packed = pack_tts_utterances([kb])
    blob = "".join(packed)
    assert "一到一点四米" in blob
    assert "1." not in blob
    assert packed[-1] == "原价八十九元。"


def test_silent_nan_skip_vs_real_speech():
    import numpy as np
    from tts.cosyvoice import (
        _energy_fail_reason,
        _float_to_pcm16,
        _pcm_is_silent,
        _speech_usable,
    )

    silent = np.zeros(256, dtype=np.float32)
    assert _energy_fail_reason(silent)
    assert _float_to_pcm16(silent) == b""
    assert _speech_usable({"tts_speech": silent}) is False

    rng = np.random.default_rng(0)
    speech = rng.uniform(-0.35, 0.35, 1024).astype(np.float32)
    pcm = _float_to_pcm16(speech)
    assert pcm
    assert not _pcm_is_silent(pcm)
    assert _speech_usable({"tts_speech": speech}) is True

    mixed = speech.copy()
    mixed[3] = np.nan
    mixed[8] = np.inf
    pcm2 = _float_to_pcm16(mixed)
    assert pcm2, "少量 NaN 清零后应仍下发真语音"


def test_detach_tts_output_stays_on_cpu_numpy():
    import numpy as np
    from tts.cosyvoice import _detach_tts_output

    rng = np.random.default_rng(1)
    speech = rng.uniform(-0.4, 0.4, 512).astype(np.float32)
    packed = _detach_tts_output({"tts_speech": speech, "sample_rate": 24000})
    assert packed["sample_rate"] == 24000
    assert packed["tts_speech"].dtype == np.float32
    assert packed["tts_speech"].shape == (512,)


def test_token2wav_guard_does_not_patch_interpolate():
    src = (SERVER9 / "tts" / "cosyvoice.py").read_text(encoding="utf-8")
    guard = src.split("def _install_token2wav_guard")[1].split("def _scrub_spk_prompt")[0]
    assert "F.interpolate =" not in guard
    assert "_interp_min1" not in guard
    assert "speed=1.0 重试" in guard


def test_chunk_path_retries_load_instead_of_skipping():
    tts_src = (SERVER9 / "tts" / "cosyvoice.py").read_text(encoding="utf-8")
    assert "def ensure_cosyvoice_loaded" in tts_src
    assert "跳过 CosyVoice3() 重载" in tts_src
    assert "_detach_tts_output" in tts_src
    ws_src = (SERVER9 / "ws" / "tts.py").read_text(encoding="utf-8")
    assert "ensure_cosyvoice_loaded" in ws_src
    assert "不下发 tts_chunk" in ws_src
    assert "tts_wire_allowed" in ws_src
    assert "await ensure_started(user_question)" not in ws_src
    robot_src = (SERVER9 / "robot" / "stream.py").read_text(encoding="utf-8")
    assert "ensure_cosyvoice_loaded" in robot_src
    assert "skip_reason" in robot_src


def test_ws_ask_defaults_to_tts_channel():
    spec = importlib.util.spec_from_file_location(
        "ws_ask_monet_fuhao", ROOT / "scripts" / "ws_ask_monet_fuhao.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = argparse.Namespace(channel="tts")
    assert mod.pick_channel(args, {"tts": False}) == "tts"
    args_auto = argparse.Namespace(channel="auto")
    assert mod.pick_channel(args_auto, {"tts": False}) == "tts"
    args_chat = argparse.Namespace(channel="chat")
    assert mod.pick_channel(args_chat, {"tts": True}) == "chat"

    src = (ROOT / "scripts" / "ws_ask_monet_fuhao.py").read_text(encoding="utf-8")
    assert 'default="tts"' in src


def test_monet_prompt_wav_exists():
    wav = ROOT / "voice" / "monet" / "monet-senlin_6to12s.wav"
    yaml_text = (ROOT / "personas" / "monet.yaml").read_text(encoding="utf-8")
    assert "voice/monet/monet-senlin_6to12s.wav" in yaml_text
    assert wav.is_file(), f"缺少参考音: {wav}"


def test_stream_ends_when_llm_thread_dies():
    import pytest
    import threading
    torch = pytest.importorskip("torch")
    from tts.cosyvoice import _yield_tts_stream

    class Flow:
        pre_lookahead_len = 0

    class Model:
        def __init__(self):
            self.lock = threading.Lock()
            self.flow = Flow()
            self.token_hop_len = 100
            self.tts_speech_token_dict = {}
            self.llm_end_dict = {}
            self.hift_cache_dict = {}
            self.calls = []

        def llm_job(self, text, prompt_text, llm_prompt, llm_embedding, uuid):
            self.tts_speech_token_dict[uuid].extend([7, 8, 9, 10, 11])
            raise RuntimeError("llm boom")

        def token2wav(self, token, prompt_token, prompt_feat, embedding,
                      token_offset, uuid, stream=False, finalize=False, speed=1.0):
            self.calls.append((int(token_offset), bool(finalize), int(token.shape[-1])))
            return torch.ones(1, 4)

    model = Model()
    zeros = torch.zeros(1, 0)
    feat = torch.zeros(1, 0, 80)
    out = list(_yield_tts_stream(model, {
        "text": zeros,
        "prompt_text": zeros,
        "llm_prompt_speech_token": zeros,
        "llm_embedding": torch.zeros(1, 192),
        "flow_prompt_speech_token": torch.zeros(1, 4),
        "prompt_speech_feat": feat,
        "flow_embedding": torch.zeros(1, 192),
    }, hop=2, idle_sec=2))
    assert out, "LLM 线程异常后仍应吐出已有 token 的 PCM"
    assert model.calls[-1][1] is True
    assert model.token_hop_len == 2
    assert model.tts_speech_token_dict == {}


def test_stream_hop_grows_then_resets():
    """CosyVoice3 不能把 hop 钉死在 25，否则每跳都重跑整段 flow+HiFT。"""
    import threading
    import pytest
    torch = pytest.importorskip("torch")
    from tts.cosyvoice import _yield_tts_stream

    class Flow:
        pre_lookahead_len = 0

    class Model:
        def __init__(self):
            self.lock = threading.Lock()
            self.flow = Flow()
            self.token_hop_len = 100
            self.token_max_hop_len = 8
            self.stream_scale_factor = 2
            self.tts_speech_token_dict = {}
            self.llm_end_dict = {}
            self.hift_cache_dict = {}
            self.calls = []

        def llm_job(self, text, prompt_text, llm_prompt, llm_embedding, uuid):
            self.tts_speech_token_dict[uuid].extend(list(range(20)))

        def token2wav(self, token, prompt_token, prompt_feat, embedding,
                      token_offset, uuid, stream=False, finalize=False, speed=1.0):
            self.calls.append((int(token_offset), bool(finalize)))
            return torch.ones(1, 4)

    model = Model()
    zeros = torch.zeros(1, 0)
    out = list(_yield_tts_stream(model, {
        "text": zeros,
        "prompt_text": zeros,
        "llm_prompt_speech_token": zeros,
        "llm_embedding": torch.zeros(1, 192),
        "flow_prompt_speech_token": zeros,
        "prompt_speech_feat": torch.zeros(1, 0, 80),
        "flow_embedding": torch.zeros(1, 192),
    }, hop=2, idle_sec=2))
    stream_offsets = [off for off, final in model.calls if not final]
    assert stream_offsets == [0, 2, 6]
    assert model.calls[-1] == (14, True)
    assert len(out) == len(model.calls)
    assert model.token_hop_len == 2


def test_qwen_turn_is_one_pcm_sliced_by_sentence():
    from robot.stream import pcm_sentence_spans
    pcm = b"\x01\x00" * 100
    spans = pcm_sentence_spans(pcm, [1, 1, 2])
    assert [len(s) for s in spans] == [50, 50, 100]
    assert b"".join(spans) == pcm
    src = (SERVER9 / "robot" / "stream.py").read_text(encoding="utf-8")
    branch = src.split('engine == "qwen3"')[1].split("elif tts.cosyvoice_model:")[0]
    assert "synthesize_speech_pcm" in branch
    assert "push_sentence_pcm" not in branch


def test_cv3_prompt_keeps_transcript_out_of_spoken_text():
    from tts.cosyvoice import _cv3_prompt_string, _CV3_TEXT_PREFIX
    assert _cv3_prompt_string("") == _CV3_TEXT_PREFIX
    line = "隆重登场的光影魔法森林。我们是这片森林的守护者。"
    packed = _cv3_prompt_string(line)
    assert packed.startswith(_CV3_TEXT_PREFIX)
    assert packed.endswith(line)
    assert _cv3_prompt_string(_CV3_TEXT_PREFIX + line) == _CV3_TEXT_PREFIX + line


def test_tts_once_vocodes_a_single_time():
    """CosyVoice3 不能按 hop 重跑整段 HiFT，否则后句越来越慢。"""
    import threading
    import pytest
    torch = pytest.importorskip("torch")
    from tts.cosyvoice import _yield_tts_once

    class Model:
        def __init__(self):
            self.lock = threading.Lock()
            self.token_hop_len = 100
            self.tts_speech_token_dict = {}
            self.llm_end_dict = {}
            self.hift_cache_dict = {}
            self.calls = []

        def llm_job(self, text, prompt_text, llm_prompt, llm_embedding, uuid):
            self.tts_speech_token_dict[uuid].extend([1, 2, 3, 4])

        def token2wav(self, token, prompt_token, prompt_feat, embedding,
                      token_offset, uuid, stream=False, finalize=False, speed=1.0):
            self.calls.append((int(token_offset), bool(stream), bool(finalize), int(token.shape[-1])))
            return torch.ones(1, 8)

    model = Model()
    zeros = torch.zeros(1, 0)
    out = list(_yield_tts_once(model, {
        "text": zeros,
        "prompt_text": zeros,
        "llm_prompt_speech_token": zeros,
        "llm_embedding": torch.zeros(1, 192),
        "flow_prompt_speech_token": zeros,
        "prompt_speech_feat": torch.zeros(1, 0, 80),
        "flow_embedding": torch.zeros(1, 192),
    }))
    assert model.calls == [(0, False, True, 4)]
    assert len(out) == 1
    assert model.token_hop_len == 25
    assert model.tts_speech_token_dict == {}
    assert model.hift_cache_dict == {}


def test_flow_prompt_locks_speaker_not_empty_prompt():
    src = (SERVER9 / "tts" / "cosyvoice.py").read_text(encoding="utf-8")
    gen = src.split("def _embedding_only_gen")[1].split("def _start_cosy_gen")[0]
    assert "_spk_flow_prompt" in src
    assert "flow_prompt_speech_token" in gen
    assert "prompt_speech_feat" in gen
    assert "cross_lingual" in src
    assert "llm_prompt_speech_token" in gen or "_empty_prompt_kwargs" in gen
    assert "You are a helpful assistant.<|endofprompt|>" in src
    assert "_cv3_prompt_tokens(frontend, _voice_prompt_text)" in gen
    assert 'prompt_kw["llm_prompt_speech_token"]' in gen
    assert "_yield_tts_once(model, model_input)" in gen
    assert "_yield_tts_stream(model, model_input)" in gen
    assert "_reset_cosy_hop()" in gen
    start = src.split("def _start_cosy_gen")[1].split("def _tts_infer")[0]
    assert "会拼参考音，留白并串音色" not in start
    ensure = src.split("def _ensure_prompt_wav")[1].split("def _cache_embedding_from_wav")[0]
    assert "endswith(suffix)" in ensure
    assert "|xling|" in src
    robot_src = (SERVER9 / "robot" / "stream.py").read_text(encoding="utf-8")
    assert "peek() or registry.get(pid)" not in robot_src
    assert "registry.get(pid)" in robot_src
    reg_src = (SERVER9 / "persona" / "registry.py").read_text(encoding="utf-8")
    assert "目录里若只有一个音，也认" not in reg_src


def test_monet_prompt_region_is_6_to_12():
    import numpy as np
    from tts.cosyvoice import (
        _prompt_clip_suffix, _select_prompt_region, MONET_PROMPT_START_SEC,
        MONET_PROMPT_END_SEC,
    )
    assert _prompt_clip_suffix("monet") == "_6to12s"
    assert _prompt_clip_suffix("fuhao") == "_male6s"
    assert MONET_PROMPT_START_SEC == 6.0
    assert MONET_PROMPT_END_SEC == 12.0
    sr = 16000
    wav = np.zeros((1, sr * 20), dtype=np.float32)
    wav[0, : sr * 6] = 9.0
    wav[0, sr * 6: sr * 12] = 1.0
    wav[0, sr * 12:] = 5.0
    out = _select_prompt_region(wav, sr, "monet")
    assert out.shape[-1] == int((MONET_PROMPT_END_SEC - MONET_PROMPT_START_SEC) * sr)
    flat = out.reshape(-1)
    assert float(flat[0]) == 1.0
    assert float(flat[-1]) == 1.0
    assert float(flat.max()) == 1.0


def test_ask_reply_keeps_two_sentences():
    from llm.client import shorten_ask_reply
    text = "睡莲画在吉维尼。我画了三十多年。光线一直在变。你想听哪一段？"
    assert shorten_ask_reply(text) == "睡莲画在吉维尼。我画了三十多年。"
    assert shorten_ask_reply("一句就够了。") == "一句就够了。"


def test_fit_spoken_chars_stays_between_150_and_200():
    from llm.client import fit_spoken_chars, spoken_len
    sentence = "我在吉维尼画睡莲，水面的光每天都不一样。"
    assert spoken_len(sentence) == 20
    fitted = fit_spoken_chars(sentence * 11, 150, 200)
    assert 150 <= spoken_len(fitted) <= 200
    assert fitted.endswith("。")
    assert fit_spoken_chars(sentence * 8, 150, 200) == sentence * 8
    hard = fit_spoken_chars("光" * 260, 150, 200)
    assert spoken_len(hard) == 200
    assert hard.endswith("。")
    short = "睡莲画在吉维尼。"
    assert fit_spoken_chars(short, 150, 200) == short


def test_audio_cache_key_differs_by_persona():
    from tts.cosyvoice import AudioCache, _flow_prompt_ok
    cache = AudioCache()
    k_monet = cache._get_key("你好。", "中文男", "pcm", persona_id="monet")
    k_fuhao = cache._get_key("你好。", "中文男", "pcm", persona_id="fuhao")
    assert k_monet != k_fuhao
    import numpy as np
    tok = np.arange(20, dtype=np.int32).reshape(1, 20)
    feat = np.zeros((1, 40, 80), dtype=np.float32)
    assert _flow_prompt_ok(tok, feat) is True
    empty_tok = np.zeros((1, 0), dtype=np.int32)
    empty_feat = np.zeros((1, 0, 80), dtype=np.float32)
    assert _flow_prompt_ok(empty_tok, empty_feat) is False
    try:
        import torch
        from tts.cosyvoice import _clone_cpu
        t = torch.arange(12, dtype=torch.int32).reshape(1, 12)
        cloned = _clone_cpu(t)
        assert cloned.dtype == torch.int32
        assert cloned.device.type == "cpu"
        assert cloned.shape == t.shape
    except ImportError:
        pass
