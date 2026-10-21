# -*- coding: utf-8 -*-
"""混合检索：向量 + BM25 + RRF。"""
from __future__ import annotations
from typing import Optional, List

import jieba
import numpy as np

from config import *
from session import SessionState
from rag.intent import ner_system
from rag import index as rag
from rag.scope import locked_category

def search(query: str, top_k: int, category, session: Optional[SessionState] = None):
    if session:
        query, _ = ner_system.resolve_coreference(query, session)
    category = locked_category(category)
    target_cats = category if isinstance(category, list) else [category]
    q_vec = rag.embedder.encode([query], normalize_embeddings=True).astype("float32")
    D, I = rag.index.search(q_vec, VEC_TOPK)
    vec_hits = {}
    for rank, row in enumerate(I[0]):
        if row == -1:
            continue
        item_idx = rag.faiss_map[row]
        if rag.items[item_idx]["category"] not in target_cats:
            continue
        if item_idx not in vec_hits:
            vec_hits[item_idx] = rank
    tokens = list(jieba.cut(query))
    scores = rag.bm25.get_scores(tokens)
    ranked_idx = sorted((i for i, it in enumerate(rag.items)
                         if it["category"] in target_cats and scores[i] > 0),
                        key=lambda i: -scores[i])[:BM25_TOPK]
    bm25_hits = {item_idx: rank for rank, item_idx in enumerate(ranked_idx)}
    rrf = {}
    recall_src = {}
    for item_idx, rank in vec_hits.items():
        rrf[item_idx] = rrf.get(item_idx, 0.0) + 1.0 / (RRF_K + rank)
        recall_src.setdefault(item_idx, set()).add("vector")
    for item_idx, rank in bm25_hits.items():
        rrf[item_idx] = rrf.get(item_idx, 0.0) + 1.0 / (RRF_K + rank)
        recall_src.setdefault(item_idx, set()).add("bm25")
    seen = set()
    fused = []
    for item_idx, score in sorted(rrf.items(), key=lambda x: -x[1]):
        key = rag.items[item_idx].get("dedup_key") or rag.items[item_idx]["id"]
        if key in seen:
            continue
        seen.add(key)
        fused.append((item_idx, score))
    cand = fused[:RERANK_CAND]
    if rag.HAS_RERANKER and rag.reranker:
        pairs = [(query, rag.items[i]["text"][:COMPRESS_MAX]) for i, _ in cand]
        r_scores = rag.reranker.predict(pairs)
        probs = 1.0 / (1.0 + np.exp(-np.array(r_scores)))
        ranked = sorted(zip(cand, probs), key=lambda x: -x[1])
        # rerank 分支同样截断 top_k
        return [_fmt(rag.items[i], i, recall_src, prob, "rerank")
                for (i, _rrf), prob in ranked[:top_k]]
    return [_fmt(rag.items[i], i, recall_src, score, None)
            for i, score in fused[:top_k]]

def _fmt(it, item_idx, recall_src, score, mode):
    src = recall_src.get(item_idx, set())
    if mode == "rerank":
        recall = "rerank"
    elif len(src) == 2:
        recall = "both"
    elif "bm25" in src:
        recall = "bm25"
    else:
        recall = "vector"
    return {"id": it["id"], "text": it["text"], "source": it.get("source", ""),
            "category": it["category"], "title": it.get("title", ""),
            "score": round(float(score), 4), "recall": recall}


def search_or_refuse(query, top_k, category=None, session=None):
    """混合检索；低于阈值则 refused=True，hits 仍返回便于打日志。"""
    hits = search(query, top_k, locked_category(category), session)
    threshold = MIN_SCORE_RERANK if rag.HAS_RERANKER else MIN_SCORE_RRF
    refused = (not hits) or hits[0]["score"] < threshold
    return hits, refused, threshold
