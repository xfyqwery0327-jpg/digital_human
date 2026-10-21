# -*- coding: utf-8 -*-
"""加载 FAISS / BM25 / embedding / 意图向量。"""
from __future__ import annotations
import json
import pickle

import faiss
import jieba
import jieba.posseg as pseg
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import *
from tts.cosyvoice import load_cosyvoice

embedder = None
index = None
items = []
faiss_map = []
bm25 = None
reranker = None
intent_vecs = None
HAS_RERANKER = False
HAS_INTENT = False

def load_jieba_dict():
    try:
        if DICT_PATH.exists():
            jieba.load_userdict(str(DICT_PATH))
            print(f"📖 已加载专名词典: {DICT_PATH}")
    except Exception as e:
        print(f"⚠️ 词典加载失败: {e}")

def load_index():
    global embedder, index, items, faiss_map, bm25, reranker, HAS_RERANKER, intent_vecs, HAS_INTENT
    load_jieba_dict()
    print("🧮 加载 embedding 模型:", EMBED_MODEL)
    embedder = SentenceTransformer(EMBED_MODEL,device='cpu')
    print(f"📦 加载索引: {INDEX_DIR}")
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    items = json.loads((INDEX_DIR / "items.json").read_text(encoding="utf-8"))
    faiss_map = json.loads((INDEX_DIR / "faiss_map.json").read_text(encoding="utf-8"))
    with open(INDEX_DIR / "bm25.pkl", "rb") as f:
        docs = pickle.load(f)["docs"]
        bm25 = BM25Okapi(docs)
    # BM25 分数按 items 下标索引，必须等长同序
    if len(docs) != len(items):
        raise RuntimeError(
            f"BM25 docs({len(docs)}) 与 items({len(items)}) 数量不一致，"
            f"请重新运行 build 脚本保证两者等长同序")
    print(f"✅ FAISS {index.ntotal} 向量 | items {len(items)} | BM25 {len(docs)} 篇")
    if RERANK_MODEL:
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder(RERANK_MODEL)
            HAS_RERANKER = True
            print(f"✅ reranker: {RERANK_MODEL}")
        except Exception as e:
            print(f"⚠️ reranker 加载失败: {e}")
    try:
        INTENT_EXAMPLES = {
            "command": ["声音大一点", "音量调大", "大声点", "小声点", "声音小一点",
                        "暂停", "别说了", "停", "继续", "再说一遍", "重复一遍"],
            "chitchat": ["你好", "嗨", "您好", "你是谁", "你叫什么", "谢谢", "再见",
                         "今天天气怎么样", "你会什么", "聊聊", "讲个笑话"],
            "knowledge": ["莫奈", "浪漫与星辰展有哪些内容", "莫奈名画是啥",
                          "这个展品是什么", "介绍一下这些名画"],
        }
        intent_vecs = {}
        for label, exs in INTENT_EXAMPLES.items():
            v = embedder.encode(exs, normalize_embeddings=True).mean(axis=0)
            intent_vecs[label] = v / np.linalg.norm(v)
        HAS_INTENT = True
        print(f"✅ 意图识别: {list(intent_vecs.keys())}")
    except Exception as e:
        print(f"⚠️ 意图识别初始化失败: {e}")
    print("🔥 预热 embedding / 分词模型...")
    embedder.encode(["warmup"], normalize_embeddings=True)
    list(pseg.cut("预热词性标注模型"))  # pseg 首次调用要加载词性模型，提前预热
    print("✅ 预热完成")
    load_cosyvoice()
