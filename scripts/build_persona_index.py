# -*- coding: utf-8 -*-
"""从已解析的切块生成某人设的 FAISS 四件套，并写入 index/{persona_id}/。

现场 robot 环境（已有 gte-base-zh）:

    python scripts/build_persona_index.py --persona fuhao --publish

本机若无 embedding 模型，加 --model thenlper/gte-base-zh 允许下载。
必须先在工作台完成「解析预览」，保证 kb/{persona}/chunks.json 存在。
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_HUB_OFFLINE", None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", required=True, help="如 fuhao / monet")
    parser.add_argument("--publish", action="store_true", help="同时写入 index/{persona}/ 线上目录")
    parser.add_argument("--model", default="", help="覆盖 embedding 模型路径或 HuggingFace 名")
    args = parser.parse_args()

    chunks_file = ROOT / "kb" / args.persona / "chunks.json"
    if not chunks_file.exists():
        raise SystemExit(f"没有切块文件: {chunks_file}\n请先在 /admin/knowledge 解析预览。")
    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    if not chunks:
        raise SystemExit("chunks.json 为空")

    import jieba
    import numpy as np
    import faiss
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    model_id = args.model
    if not model_id:
        homes = [
            ROOT / "models" / "gte-base-zh",
            ROOT / "models" / "thenlper-gte-base-zh",
            Path.home() / ".cache" / "huggingface" / "hub" / "models--thenlper--gte-base-zh",
            Path(r"C:\Users\BroadMesse\.cache\huggingface\hub\models--thenlper--gte-base-zh"),
        ]
        model_id = ""
        for local in homes:
            if not local.exists():
                continue
            if (local / "config.json").exists() or (local / "modules.json").exists():
                model_id = str(local)
                break
            snaps = sorted(local.glob("snapshots/*"))
            if snaps:
                model_id = str(snaps[0])
                break
        if not model_id:
            model_id = "thenlper/gte-base-zh"
    print(f"加载 embedding: {model_id}")
    embedder = SentenceTransformer(model_id, device="cpu")
    texts = [c["text"] for c in chunks]
    vecs = embedder.encode(
        texts, batch_size=8, normalize_embeddings=True, show_progress_bar=True
    ).astype("float32")
    index = faiss.IndexFlatIP(int(vecs.shape[1]))
    index.add(vecs)
    faiss_map = list(range(len(chunks)))
    docs = [list(jieba.cut(t)) for t in texts]

    dests = [ROOT / "kb" / args.persona / "draft"]
    if args.publish:
        dests.append(ROOT / "index" / args.persona)
    for dest in dests:
        dest.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(dest / "faiss.index"))
        (dest / "items.json").write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        (dest / "faiss_map.json").write_text(
            json.dumps(faiss_map, ensure_ascii=False), encoding="utf-8")
        with open(dest / "bm25.pkl", "wb") as f:
            pickle.dump({"docs": docs}, f)
        print(f"已写入 {dest}  条目={len(chunks)} 向量维={vecs.shape[1]}")


if __name__ == "__main__":
    main()
