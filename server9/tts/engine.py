# -*- coding: utf-8 -*-
"""语音管线选择。改 deploy/last-start.txt 的 tts=，重启讲解员后生效。

cosyvoice  用上次记下的 CosyVoice2 或 CosyVoice3
qwen3      用 models/Qwen3-TTS-12Hz-0.6B-Base，参考音沿用人设 prompt wav
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAST_START = ROOT / "deploy" / "last-start.txt"
QWEN_DIRNAME = "Qwen3-TTS-12Hz-0.6B-Base"

_QWEN = {"qwen", "qwen3", "qwen3-tts", "qwen3tts"}


def normalize_tts_engine(raw) -> str:
    token = ("" if raw is None else str(raw)).strip().lower()
    if token in _QWEN:
        return "qwen3"
    return "cosyvoice"


def _read_last(key: str) -> str:
    try:
        lines = LAST_START.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    prefix = key + "="
    for line in lines:
        row = line.strip()
        if row.lower().startswith(prefix):
            return row.split("=", 1)[1].strip()
    return ""


def tts_engine_name() -> str:
    """环境变量优先。没设时读 last-start.txt，缺省 cosyvoice。"""
    env = os.environ.get("TTS_ENGINE", "").strip()
    if env:
        return normalize_tts_engine(env)
    saved = _read_last("tts")
    if saved:
        return normalize_tts_engine(saved)
    return "cosyvoice"


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
