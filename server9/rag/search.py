# -*- coding: utf-8 -*-
"""混合检索：向量 + BM25 + RRF。始终使用请求人设自己的索引，不读别人的库。"""
from __future__ import annotations
import re
from typing import Optional, List

try:
    import jieba
    import numpy as np
except ImportError:
    jieba = None
    np = None

from config import *
from session import SessionState
from rag.intent import ner_system
from rag import index as rag
from rag.scope import locked_category


_GENERIC_WORKS = re.compile(
    r"(介绍|讲讲|看看|讲一下|介绍下|介绍一下).{0,12}(作品|展品|这些画|画作|名画)|"
    r"(本[厅展]|这场展).{0,12}(作品|展品|画)|"
    r"(有哪些|有什么)(作品|展品|画)|"
    r"^(作品|展品|这些画)"
)
_OTHER_ARTIST = (
    "阿尔弗雷德", "德勒", "本迪克森", "雷诺阿", "德加", "毕沙罗", "塞尚",
    "西斯莱", "热里科", "马奈", "德拉克洛瓦", "米勒", "柯罗", "科罗",
    "杜米埃", "莫里索", "卡萨特", "宝琳", "普尼", "阿佩", "自由引导人民",
    "拾穗者", "费加罗的婚礼",
)
_SELF_WORK = (
    "莫奈", "日出", "睡莲", "吉维尼", "卡美尔", "卡米尔", "干草堆", "鲁昂",
    "穿绿裙", "穿绿衣", "滑铁卢", "圣拉查尔", "野罂粟", "翁费勒",
    "威斯敏斯特", "暴风雨", "泰晤士", "圣日尔曼",
)
_BIO_SKIP = ("个人生活", "家庭情况", "父亲", "童年", "妻子", "结婚", "儿子",
             "是谁", "全名", "出生和逝世", "基本资料", "私人生活")


def is_generic_works_query(query: str, persona=None) -> bool:
    q = (query or "").strip()
    if not q or "《" in q:
        return False
    if persona is not None and getattr(persona, "id", "") != "monet":
        return False
    return bool(_GENERIC_WORKS.search(q))


def works_answer_hint(query: str, persona=None) -> str:
    if not is_generic_works_query(query, persona):
        return ""
    return "若观众是在问作品，只介绍本厅正在展出的我（莫奈）的画，不要介绍其他画家。\n"


def _rewrite_query(query: str, persona, generic: bool) -> str:
    if generic:
        return (query + " 本厅展出的莫奈作品 日出·印象 睡莲 吉维尼花园 "
                "卡美尔 滑铁卢桥 圣拉查尔火车站 野罂粟")
    return query


def _item_kind(item) -> int:
    title = (item.get("title") or "") + (item.get("question") or "")
    text = (item.get("text") or "")[:240]
    blob = title + text
    if any(x in title for x in _OTHER_ARTIST):
        return 2
    if any(x in title for x in _BIO_SKIP) and "《" not in title:
        return 2
    if any(x in blob for x in _SELF_WORK):
        return 0
    if any(x in text for x in _OTHER_ARTIST):
        return 2
    return 1


def _prefer_self_works(generic: bool, items, fused, persona):
    """泛问「介绍作品」时，把莫奈本厅画排到前面，压掉其他画家条目。"""
    if not generic or persona is None or getattr(persona, "id", "") != "monet":
        return fused
    return sorted(fused, key=lambda x: (_item_kind(items[x[0]]), -x[1]))


def _prefer_self_work_hits(generic: bool, hits, persona):
    if not generic or persona is None or getattr(persona, "id", "") != "monet" or not hits:
        return hits
    self_hits = [h for h in hits if _item_kind(h) == 0]
    if not self_hits:
        self_hits = [h for h in hits if _item_kind(h) == 1]
    if not self_hits:
        return hits
    if self_hits[0]["score"] < hits[0]["score"]:
        lead = dict(self_hits[0])
        lead["score"] = hits[0]["score"]
        self_hits = [lead] + self_hits[1:]
    return self_hits


def search(query: str, top_k: int, category, session: Optional[SessionState] = None,
           persona=None):
    if session:
        query, _ = ner_system.resolve_coreference(query, session)
    generic_works = is_generic_works_query(query, persona)
    query = _rewrite_query(query, persona, generic_works)
    bundle = rag.bundle_for(persona)
    if not bundle or jieba is None or np is None or rag.embedder is None:
        return []
    idx = bundle["index"]
    items = bundle["items"]
    faiss_map = bundle["faiss_map"]
    bm25 = bundle["bm25"]
    if idx is None or not items or bm25 is None:
        return []
    category = locked_category(category, persona)
    target_cats = category if isinstance(category, list) else [category]
    q_vec = rag.embedder.encode([query], normalize_embeddings=True).astype("float32")
    D, I = idx.search(q_vec, VEC_TOPK)
    vec_hits = {}
    for rank, row in enumerate(I[0]):
        if row == -1:
            continue
        item_idx = faiss_map[row]
        if items[item_idx]["category"] not in target_cats:
            continue
        if item_idx not in vec_hits:
            vec_hits[item_idx] = rank
    tokens = list(jieba.cut(query))
    scores = bm25.get_scores(tokens)
    ranked_idx = sorted((i for i, it in enumerate(items)
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
        key = items[item_idx].get("dedup_key") or items[item_idx]["id"]
        if key in seen:
            continue
        seen.add(key)
        fused.append((item_idx, score))
    fused = _prefer_self_works(generic_works, items, fused, persona)
    cand = fused[:RERANK_CAND]
    if rag.HAS_RERANKER and rag.reranker:
        pairs = [(query, items[i]["text"][:COMPRESS_MAX]) for i, _ in cand]
        r_scores = rag.reranker.predict(pairs)
        probs = 1.0 / (1.0 + np.exp(-np.array(r_scores)))
        ranked = sorted(zip(cand, probs), key=lambda x: -x[1])
        hits = [_fmt(items[i], i, recall_src, prob, "rerank")
                for (i, _rrf), prob in ranked]
        return _prefer_self_work_hits(generic_works, hits, persona)[:top_k]
    hits = [_fmt(items[i], i, recall_src, score, None)
            for i, score in fused[:top_k]]
    if generic_works and fused:
        orig_best = max((s for _, s in fused), default=0.0)
        if hits and hits[0]["score"] < orig_best:
            hits[0] = dict(hits[0], score=round(float(orig_best), 4))
    return hits


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


def search_or_refuse(query, top_k, category=None, session=None, persona=None):
    """混合检索；低于阈值则 refused=True，hits 仍返回便于打日志。"""
    hits = search(query, top_k, locked_category(category, persona), session, persona=persona)
    threshold = MIN_SCORE_RERANK if rag.HAS_RERANKER else MIN_SCORE_RRF
    refused = (not hits) or hits[0]["score"] < threshold
    return hits, refused, threshold
