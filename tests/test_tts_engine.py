# -*- coding: utf-8 -*-
"""语音管线名字和 Qwen 权重文件名，不加载模型。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER9 = ROOT / "server9"
if str(SERVER9) not in sys.path:
    sys.path.insert(0, str(SERVER9))

from tts.engine import normalize_tts_engine, pending_weight_renames, prepare_qwen_dir  # noqa: E402
from tts.qwen import _as_float_wave, _reset_talker_cache, _unpack_wavs, codec_budget  # noqa: E402


def test_normalize_tts_engine():
    assert normalize_tts_engine("qwen") == "qwen3"
    assert normalize_tts_engine("Qwen3-TTS") == "qwen3"
    assert normalize_tts_engine("cosyvoice") == "cosyvoice"
    assert normalize_tts_engine("3") == "cosyvoice"
    assert normalize_tts_engine("") == "cosyvoice"
    assert normalize_tts_engine(None) == "cosyvoice"


def test_pending_weight_renames_only_when_canonical_missing(tmp_path):
    model = tmp_path / "Qwen3-TTS-12Hz-0.6B-Base"
    tok = model / "speech_tokenizer"
    tok.mkdir(parents=True)
    (model / "model.safetensors_").write_bytes(b"w")
    (model / "vocab.json_").write_text("{}", encoding="utf-8")
    (tok / "model.safetensors_").write_bytes(b"t")
    (model / "model.safetensors").write_bytes(b"keep")
    pending = {(src.relative_to(model).as_posix(), dst.name) for src, dst in pending_weight_renames(model)}
    assert ("model.safetensors_", "model.safetensors") not in pending
    assert ("vocab.json_", "vocab.json") in pending
    assert ("speech_tokenizer/model.safetensors_", "model.safetensors") in pending
    prepare_qwen_dir(model)
    assert (model / "vocab.json").is_file()
    assert (tok / "model.safetensors").is_file()
    assert (model / "model.safetensors").read_bytes() == b"keep"
    assert pending_weight_renames(model) == []


def test_unpack_qwen_wavs_accepts_pair_and_bare_wave():
    import numpy as np
    wave = np.linspace(-0.2, 0.2, 8, dtype=np.float32)
    wavs, sr = _unpack_wavs(( [wave], 24000 ))
    assert sr == 24000
    assert len(wavs) == 1
    pcm_src = _as_float_wave(wavs[0])
    assert pcm_src.shape == (8,)
    bare, sr2 = _unpack_wavs(wave)
    assert sr2 == 24000
    assert _as_float_wave(bare[0]).shape == (8,)


def test_qwen_codec_budget_stays_short_and_cache_resets():
    assert codec_budget("我是克洛德·莫奈，") == 96
    assert codec_budget("字" * 140) == 140 * 4 + 48
    assert codec_budget("字" * 400) == 1024

    class Predictor:
        rope_deltas = "stale"

    class Talker:
        def __init__(self):
            self.rope_deltas = "stale"
            self.code_predictor = Predictor()

    class Inner:
        talker = None

    class Model:
        def __init__(self):
            self.model = Inner()
            self.model.talker = Talker()

    model = Model()
    _reset_talker_cache(model)
    assert model.model.talker.rope_deltas is None
    assert model.model.talker.code_predictor.rope_deltas is None


def test_local_bin_load_replaces_transformers_check(tmp_path):
    import sys
    import types

    from rag.binload import allow_local_torch_load, embed_dir_needs_bin

    only_bin = tmp_path / "gte"
    only_bin.mkdir()
    (only_bin / "pytorch_model.bin").write_bytes(b"x")
    assert embed_dir_needs_bin(only_bin) is True
    (only_bin / "model.safetensors").write_bytes(b"s")
    assert embed_dir_needs_bin(only_bin) is False

    name = "transformers.modeling_utils"
    previous = sys.modules.get(name)
    module = types.ModuleType(name)

    def _blocked():
        raise ValueError("torch load blocked")

    module.check_torch_load_is_safe = _blocked
    sys.modules[name] = module
    saved = []
    try:
        for mod_name, mod in list(sys.modules.items()):
            fn = getattr(mod, "check_torch_load_is_safe", None) if mod_name.startswith("transformers") else None
            if fn is not None:
                saved.append((mod, fn))
        assert allow_local_torch_load() >= 1
        module.check_torch_load_is_safe()
    finally:
        for mod, fn in saved:
            try:
                mod.check_torch_load_is_safe = fn
            except Exception:
                pass
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
