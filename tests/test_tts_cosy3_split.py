# -*- coding: utf-8 -*-
"""分句和 KV 重置不依赖 GPU。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tts_cosy3.split_text import clamp_max_chars, split_tts_text
from tts_cosy3.vram import reset_kv_cache


def test_short_sentence_stays():
    assert split_tts_text("你好。", max_chars=36) == ["你好。"]


def test_long_text_never_exceeds_cap():
    text = "我" * 100
    parts = split_tts_text(text, max_chars=36)
    assert parts
    assert all(len(p) <= 36 for p in parts)
    assert "".join(parts) == text


def test_strong_punct_not_glued():
    parts = split_tts_text("你好。这是下一段。", max_chars=36)
    assert parts[0] == "你好。"
    assert parts[-1] == "这是下一段。"


def test_decimal_point_kept():
    parts = split_tts_text("身高1.4米就到这里。", max_chars=36)
    assert len(parts) == 1
    assert "1.4" in parts[0]


def test_api_clamp_is_30_to_40():
    assert clamp_max_chars(None) == 36
    assert clamp_max_chars(10) == 30
    assert clamp_max_chars(80) == 40
    assert clamp_max_chars(32) == 32


def test_reset_kv_cache_clears_session_and_hop():
    class _Node:
        def __init__(self):
            self.past_key_values = ("k", "v")
            self.cache = type("DynamicCache", (), {})()

        def children(self):
            return []

    class _Model:
        def __init__(self):
            self.tts_speech_token_dict = {"u": [1, 2, 3]}
            self.llm_end_dict = {"u": True}
            self.hift_cache_dict = {"u": {"mel": object()}}
            self.flow_cache_dict = {"u": object()}
            self.mel_overlap_dict = {}
            self.token_hop_len = 100
            self.llm = _Node()

    model = _Model()
    reset_kv_cache(model)
    assert model.tts_speech_token_dict == {}
    assert model.hift_cache_dict == {}
    assert model.flow_cache_dict == {}
    assert model.token_hop_len == 25
    assert model.llm.past_key_values is None


def test_modules_import():
    import tts_cosy3.engine  # noqa: F401
    import tts_cosy3.supervisor  # noqa: F401
    import tts_cosy3.tts_cosy3_server  # noqa: F401
    server9 = ROOT / "server9"
    if str(server9) not in sys.path:
        sys.path.insert(0, str(server9))
    import tts.remote  # noqa: F401
    from pathlib import Path
    src = Path(__file__).resolve().parents[1].joinpath("tts_cosy3", "engine.py").read_text(encoding="utf-8")
    assert "model.reset_kv_cache()" in src
    assert "gc.collect()" in src
    assert "torch.cuda.empty_cache()" in src
    assert "stream=False" in src
