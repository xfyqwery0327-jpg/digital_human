# -*- coding: utf-8 -*-
"""从魔搭社区下载 gte-base-zh 到 models/gte-base-zh。

现场不能访问 HuggingFace 时用这个。下载后讲解员离线加载，不必再连外网。
模型与现有 FAISS 索引同为 768 维，不要换成其它 embedding。

    conda activate robot
    pip install modelscope -i https://mirrors.aliyun.com/pypi/simple/
    python scripts/download_gte_modelscope.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "models" / "gte-base-zh"
MODELSCOPE_ID = "AI-ModelScope/gte-base-zh"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / name).exists() for name in (
        "modules.json", "config.json", "pytorch_model.bin", "model.safetensors"))


def main() -> int:
    if _ready(DEST):
        print(f"已存在，跳过下载: {DEST}")
        return 0

    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.setdefault("MODELSCOPE_DOMAIN", "www.modelscope.cn")

    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        print("未安装 modelscope。请先（可用阿里云镜像）:")
        print("  pip install modelscope -i https://mirrors.aliyun.com/pypi/simple/")
        return 1

    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"从魔搭下载 {MODELSCOPE_ID}")
    print(f"保存到 {DEST}")
    try:
        snapshot_download(MODELSCOPE_ID, local_dir=str(DEST))
    except TypeError:
        snapshot_download(MODELSCOPE_ID, cache_dir=str(ROOT / "models"))
        # 旧版 API 落到 cache，再尽量定位
        guessed = list((ROOT / "models").rglob("config.json"))
        if guessed and not _ready(DEST):
            import shutil
            src = guessed[0].parent
            if src.resolve() != DEST.resolve():
                if DEST.exists():
                    shutil.rmtree(DEST)
                shutil.copytree(src, DEST)

    if not _ready(DEST):
        print("下载完成但未找到 config.json / 权重，请检查目录:")
        print(f"  {DEST}")
        if DEST.is_dir():
            for p in sorted(DEST.rglob("*"))[:30]:
                print("   ", p.relative_to(DEST))
        return 1

    print("下载完成。启动服务时会自动使用该目录。")
    print("  python C:\\digital_human\\deploy\\start-server.ps1 所调用的 uvicorn")
    print(f"  或设环境变量 EMBED_MODEL={DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
