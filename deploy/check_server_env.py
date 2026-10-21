# -*- coding: utf-8 -*-
"""核对讲解员 conda 环境。版本以能跑通 CosyVoice3 / Qwen3-TTS 的那套为准。"""
from __future__ import annotations

import importlib.metadata as metadata
import sys

PINS = {
    "transformers": "4.57.3",
    "qwen-tts": "0.1.1",
    "numpy": "1.26.4",
    "llama-cpp-python": "0.3.36",
    "onnxruntime": "1.18.0",
    "faiss-cpu": "1.15.0",
    "sentence-transformers": "6.0.1",
    "openai-whisper": "20231117",
}
IMPORTS = ("fastapi", "uvicorn", "faiss", "sentence_transformers", "whisper")


def main() -> int:
    bad = []
    missing = []
    ver = sys.version.split()[0]
    if sys.version_info[:2] != (3, 10):
        bad.append("python %s want 3.10" % ver)
    else:
        print("python %s" % ver)

    try:
        import torch
    except Exception as e:
        missing.append("torch not installed: %s" % e)
        torch = None
    if torch is not None:
        got = torch.__version__
        print("torch %s cuda=%s" % (got, torch.cuda.is_available()))
        if got != "2.11.0+cu126":
            bad.append("torch %s want 2.11.0+cu126" % got)
        elif not torch.cuda.is_available():
            bad.append("torch 2.11.0+cu126 installed but CUDA unavailable")

    try:
        import torchaudio
        print("torchaudio %s" % torchaudio.__version__)
        if torchaudio.__version__ != "2.11.0+cu126":
            bad.append("torchaudio %s want 2.11.0+cu126" % torchaudio.__version__)
    except Exception as e:
        missing.append("torchaudio not installed: %s" % e)

    for name, want in PINS.items():
        try:
            got = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append("%s not installed, want %s" % (name, want))
            continue
        print("%s %s" % (name, got))
        if got != want:
            bad.append("%s %s want %s" % (name, got, want))

    for mod in IMPORTS:
        try:
            __import__(mod)
        except Exception as e:
            missing.append("%s import failed: %s" % (mod, e))

    if missing or bad:
        print("FAIL")
        for line in missing + bad:
            print(line)
        return 2 if missing and not bad else 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
