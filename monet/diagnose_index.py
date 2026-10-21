#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库诊断脚本 - 检查索引内容和检索效果
"""
import json
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import jieba
from rank_bm25 import BM25Okapi

# ============================================================
# 配置（和 server.py 保持一致）
# ============================================================

INDEX_DIR = Path("index")
EMBED_MODEL = r"C:\Users\BroadMesse\.cache\huggingface\hub\models--thenlper--gte-base-zh\snapshots\71ab7947d6fac5b64aa299e6e40e6c2b2e85976c"
EXHIBITION_CATEGORY = ["exhibition_fuhao", "exhibition_monet"]

VEC_TOPK = 50
BM25_TOPK = 50
RRF_K = 60

# ============================================================
# 加载索引
# ============================================================

print("🔍 加载知识库...")
embedder = SentenceTransformer(EMBED_MODEL)
index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
items = json.loads((INDEX_DIR / "items.json").read_text(encoding="utf-8"))
faiss_map = json.loads((INDEX_DIR / "faiss_map.json").read_text(encoding="utf-8"))

with open(INDEX_DIR / "bm25.pkl", "rb") as f:
    docs = pickle.load(f)["docs"]
    bm25 = BM25Okapi(docs)

print(f"✅ 索引加载完成")
print(f"   FAISS 向量数: {index.ntotal}")
print(f"   知识条目数: {len(items)}")
print(f"   BM25 文档数: {len(docs)}")

# ============================================================
# 1. 检查知识库内容
# ============================================================

print("\n" + "=" * 60)
print("📊 知识库内容统计")
print("=" * 60)

# 统计 category
categories = {}
for item in items:
    cat = item.get("category", "未分类")
    categories[cat] = categories.get(cat, 0) + 1

print(f"\n📁 Category 分布:")
for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
    print(f"   {cat}: {count} 条")

# 检查目标 category（支持列表）
target_cats = EXHIBITION_CATEGORY if isinstance(EXHIBITION_CATEGORY, list) else [EXHIBITION_CATEGORY]
found_cats = [c for c in target_cats if c in categories]

if found_cats:
    total = sum(categories[c] for c in found_cats)
    print(f"\n✅ 找到目标 category: {found_cats} (共 {total} 条)")
else:
    print(f"\n❌ 未找到目标 category: {target_cats}")
    print(f"   可用的 category: {list(categories.keys())}")

# 查看前几条数据样例
print(f"\n📄 前 5 条数据样例:")
for i, item in enumerate(items[:5]):
    print(f"\n--- 条目 {i+1} ---")
    print(f"ID: {item.get('id')}")
    print(f"Category: {item.get('category')}")
    print(f"Title: {item.get('title', 'N/A')}")
    print(f"Source: {item.get('source', 'N/A')}")
    print(f"Text (前100字): {item.get('text', '')[:100]}...")

# ============================================================
# 2. 测试检索效果
# ============================================================

print("\n" + "=" * 60)
print("🔍 测试检索效果")
print("=" * 60)

test_queries = [
    "莫奈的代表作是什么？",
    "印象派是什么风格？",
    "睡莲这幅画有什么特点？",
    "莫奈",
    "印象派",
    "睡莲"
]

for query in test_queries:
    print(f"\n--- 查询: {query} ---")
    
    # FAISS 向量检索
    q_vec = embedder.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(q_vec, VEC_TOPK)
    
    vec_hits_all = []
    vec_hits_filtered = []
    
    for rank, (row, dist) in enumerate(zip(I[0], D[0])):
        if row == -1:
            continue
        item_idx = faiss_map[row]
        it = items[item_idx]
        vec_hits_all.append((item_idx, dist, it))
        
        # 支持列表形式的 category
        if it["category"] in target_cats:
            vec_hits_filtered.append((item_idx, dist, it))
    
    print(f"   FAISS 检索:")
    print(f"      全部结果: {len(vec_hits_all)} 条")
    print(f"      过滤后({target_cats}): {len(vec_hits_filtered)} 条")
    
    if vec_hits_all:
        print(f"      Top 3 全部结果:")
        for i, (idx, dist, it) in enumerate(vec_hits_all[:3]):
            print(f"         {i+1}. [{it['category']}] dist={dist:.4f} | {it.get('title', it['text'][:30])}")
    
    if vec_hits_filtered:
        print(f"      Top 3 过滤结果:")
        for i, (idx, dist, it) in enumerate(vec_hits_filtered[:3]):
            print(f"         {i+1}. dist={dist:.4f} | {it.get('title', it['text'][:30])}")
    else:
        print(f"      ⚠️ 过滤后无结果！")
    
    # BM25 检索
    tokens = list(jieba.cut(query))
    scores = bm25.get_scores(tokens)
    
    ranked_all = sorted(enumerate(scores), key=lambda x: -x[1])[:BM25_TOPK]
    ranked_filtered = [(i, s) for i, s in ranked_all if items[i]["category"] in target_cats]
    
    print(f"   BM25 检索:")
    print(f"      分词: {tokens}")
    print(f"      全部结果: {len([s for s in scores if s > 0])} 条")
    print(f"      过滤后({target_cats}): {len(ranked_filtered)} 条")
    
    if ranked_filtered:
        print(f"      Top 3 过滤结果:")
        for i, (idx, score) in enumerate(ranked_filtered[:3]):
            it = items[idx]
            print(f"         {i+1}. score={score:.4f} | {it.get('title', it['text'][:30])}")
    else:
        print(f"      ⚠️ 过滤后无结果！")
    
    # RRF 融合
    rrf = {}
    for item_idx, _, _ in vec_hits_filtered:
        rank = next((i for i, (idx, _, _) in enumerate(vec_hits_filtered) if idx == item_idx), 999)
        rrf[item_idx] = rrf.get(item_idx, 0.0) + 1.0 / (RRF_K + rank)
    
    for item_idx, _ in ranked_filtered:
        rank = next((i for i, (idx, _) in enumerate(ranked_filtered) if idx == item_idx), 999)
        rrf[item_idx] = rrf.get(item_idx, 0.0) + 1.0 / (RRF_K + rank)
    
    fused = sorted(rrf.items(), key=lambda x: -x[1])[:5]
    
    print(f"   RRF 融合:")
    print(f"      融合结果: {len(fused)} 条")
    
    if fused:
        print(f"      Top 3:")
        for i, (idx, score) in enumerate(fused[:3]):
            it = items[idx]
            print(f"         {i+1}. RRF={score:.4f} | {it.get('title', it['text'][:50])}")
    else:
        print(f"      ⚠️ 无融合结果！")

# ============================================================
# 3. 诊断建议
# ============================================================

print("\n" + "=" * 60)
print("💡 诊断建议")
print("=" * 60)

if not found_cats:
    print(f"\n❌ 问题：category 不匹配")
    print(f"   当前配置: {target_cats}")
    print(f"   实际 category: {list(categories.keys())}")
    print(f"\n🔧 解决方案：")
    print(f"   在 server.py 和 diagnose_index.py 中修改 EXHIBITION_CATEGORY")
else:
    print(f"\n✅ Category 配置正确")
    print(f"   目标: {found_cats}")
    print(f"   数据量: {sum(categories[c] for c in found_cats)} 条")