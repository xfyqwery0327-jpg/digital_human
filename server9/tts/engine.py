# -*- coding: utf-8 -*-
"""语音管线。讲解员固定走独立服务 qwen-tts-service，默认 http://127.0.0.1:8812。

进程内 Qwen3-TTS、SGLang WebSocket、CosyVoice 不再作为启动选项。
last-start.txt 里旧的 tts=cosyvoice 也不会再被选中。
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QWEN_DIRNAME = "Qwen3-TTS-12Hz-0.6B-Base"

_QWEN = {
    "qwen", "qwen3", "qwen3-tts", "qwen3tts",
    "qwen-service", "qwen-tts-service", "service",
}


def normalize_tts_engine(raw) -> str:
    token = ("" if raw is None else str(raw)).strip().lower()
    if token in _QWEN:
        return "qwen3"
    return "cosyvoice"


def tts_engine_name() -> str:
    """固定 qwen3。旧的 cosyvoice / sglang / 进程内标记不再选用。"""
    return "qwen3"


def use_qwen_tts() -> bool:
    return tts_engine_name() == "qwen3"


def qwen_model_dir() -> Path:
    env = os.environ.get("QWEN_TTS_MODEL", "").strip()
    if env:
        return Path(env)
    return ROOT / "models" / QWEN_DIRNAME


# 下载工具有时把完成文件落成「标准名_」。加载前改回 Hugging Face 认的名字。
_WEIGHT_ALIASES = (
    ("model.safetensors_", "model.safetensors"),
    ("vocab.json_", "vocab.json"),
    ("speech_tokenizer/model.safetensors_", "speech_tokenizer/model.safetensors"),
)


def pending_weight_renames(model_dir: Path) -> list:
    """返回 (现有带下划线的文件, 应改成的标准名)。标准名已在则不动。"""
    root = Path(model_dir)
    found = []
    for src_rel, dst_rel in _WEIGHT_ALIASES:
        src = root / src_rel
        dst = root / dst_rel
        if src.is_file() and not dst.exists():
            found.append((src, dst))
    return found


def prepare_qwen_dir(model_dir: Path) -> Path:
    root = Path(model_dir)
    for src, dst in pending_weight_renames(root):
        src.rename(dst)
    return root
