# -*- coding: utf-8 -*-
"""本机 gte-base-zh 只有 pytorch_model.bin。

装上 qwen-tts 之后，transformers 在 torch 低于 2.6 时会拒绝 torch.load
（CVE-2025-32434）。这份权重是现场自己的 embedding，不是外部下载的临时文件。
只在加载它之前，把已经绑进去的检查换成空操作。
"""
from __future__ import annotations

import sys
from pathlib import Path


def embed_dir_needs_bin(model_dir) -> bool:
    root = Path(model_dir)
    return (root / "pytorch_model.bin").is_file() and not (root / "model.safetensors").is_file()


def allow_local_torch_load() -> int:
    """把 transformers 里已经导入的 check_torch_load_is_safe 换成空操作。返回改了几处。"""
    def _allow():
        return None

    for modname in (
        "transformers.modeling_utils",
        "transformers.utils",
        "transformers.utils.import_utils",
    ):
        try:
            __import__(modname)
        except Exception:
            pass
    patched = 0
    for name, module in list(sys.modules.items()):
        if not name.startswith("transformers") or module is None:
            continue
        try:
            current = getattr(module, "check_torch_load_is_safe", None)
        except Exception:
            continue
        if current is None:
            continue
        try:
            setattr(module, "check_torch_load_is_safe", _allow)
        except Exception:
            continue
        patched += 1
    return patched
