# -*- coding: utf-8 -*-
"""路径。代码放在仓库任意盘符都能跑，不把业务逻辑写死在 E: 或 D:。"""
from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def package_dir() -> Path:
    return Path(__file__).resolve().parent


def run_dir() -> Path:
    path = package_dir() / "run"
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_dir() -> Path:
    path = project_root() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def speaker_cache_dir() -> Path:
    path = package_dir() / "cache" / "speakers"
    path.mkdir(parents=True, exist_ok=True)
    return path


def restart_request_path() -> Path:
    return run_dir() / "restart.request"


def state_path() -> Path:
    return run_dir() / "state.json"


def _first_dir(candidates) -> str:
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw)
        if path.is_dir():
            return str(path)
    return ""


def resolve_cosyvoice_root() -> str:
    root = project_root()
    return _first_dir([
        os.environ.get("COSYVOICE_PATH"),
        root / "CosyVoice",
        Path(r"D:\projects\CosyVoice"),
        Path(r"E:\projects\Robot_Intelligent_Voice_System\CosyVoice"),
        Path(r"E:\projects\digital_human\CosyVoice"),
        Path(r"C:\ro\CosyVoice"),
    ])


def resolve_model_dir() -> str:
    """只认 Fun-CosyVoice3-0.5B。目录里必须有 cosyvoice3.yaml。"""
    env = os.environ.get("COSYVOICE_MODEL") or ""
    name = "Fun-CosyVoice3-0.5B"
    root = resolve_cosyvoice_root()
    candidates = [
        env,
        str(Path(root) / "pretrained_models" / name) if root else "",
        rf"D:\projects\CosyVoice\pretrained_models\{name}",
        rf"E:\projects\Robot_Intelligent_Voice_System\CosyVoice\pretrained_models\{name}",
        rf"C:\ro\CosyVoice\pretrained_models\{name}",
    ]
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw)
        if path.is_dir() and (path / "cosyvoice3.yaml").is_file():
            return str(path)
    searched = "\n".join(f"  - {c}" for c in candidates if c)
    raise FileNotFoundError(
        "找不到 Fun-CosyVoice3-0.5B（需要 cosyvoice3.yaml）。已查找:\n" + searched
    )
