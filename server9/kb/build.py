# -*- coding: utf-8 -*-
"""从切块生成 FAISS + BM25 四件套。"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import List

from kb.store import INDEX_FILES, draft_dir, save_chunks


def can_build() -> tuple:
    from rag import index as rag
    if not getattr(rag, "HAS_RAG_DEPS", False):
        return False, "本机未安装 faiss / sentence-transformers / jieba，无法从原文生成索引。可改用「上传索引包」。"
    if rag.embedder is None:
        return False, "embedding 模型未加载（检查 EMBED_MODEL 路径），无法生成向量。"
    return True, ""


def build_draft(persona_id: str, chunks: List[dict]) -> dict:
    ok, reason = can_build()
    if not ok:
        save_chunks(persona_id, chunks)
        raise RuntimeError(reason)
    from rag import index as rag
    import jieba
    import numpy as np
    import faiss
    from rank_bm25 import BM25Okapi

    if not chunks:
        raise RuntimeError("没有可建库的文案块，请先解析预览。")
    texts = [c["text"] for c in chunks]
    vecs = rag.embedder.encode(texts, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss_map = list(range(len(chunks)))
    docs = [list(jieba.cut(t)) for t in texts]
    dest = draft_dir(persona_id)
    dest.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(dest / "faiss.index"))
    (dest / "items.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "faiss_map.json").write_text(json.dumps(faiss_map, ensure_ascii=False), encoding="utf-8")
    with open(dest / "bm25.pkl", "wb") as f:
        pickle.dump({"docs": docs}, f)
    save_chunks(persona_id, chunks)
    missing = [n for n in INDEX_FILES if not (dest / n).exists()]
    if missing:
        raise RuntimeError(f"草稿不完整，缺: {missing}")
    return {
        "ok": True,
        "count": len(chunks),
        "vectors": int(index.ntotal),
        "draft_dir": str(dest),
    }


def validate_pack(folder: Path) -> None:
    from kb.store import INDEX_FILES, index_ready
    if not index_ready(folder):
        raise RuntimeError("索引包必须包含 faiss.index、items.json、faiss_map.json、bm25.pkl")
    items = json.loads((folder / "items.json").read_text(encoding="utf-8"))
    fmap = json.loads((folder / "faiss_map.json").read_text(encoding="utf-8"))
    import pickle
    with open(folder / "bm25.pkl", "rb") as f:
        docs = pickle.load(f)["docs"]
    if len(docs) != len(items):
        raise RuntimeError(f"BM25 篇数({len(docs)}) 与 items({len(items)}) 不一致")
    if len(fmap) < 1:
        raise RuntimeError("faiss_map 为空")
