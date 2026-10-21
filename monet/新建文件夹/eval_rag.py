# -*- coding: utf-8 -*-
"""eval_rag.py - RAG 召回自动评测"""
import os
import re
import sys
import pickle

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
INDEX_PATH = os.path.join(BASE_DIR, "rag_index.pkl")
TOP_K = 5

# ==================== 手工测试集 ====================
# 判定方式（满足其一即命中）：
#   venue: 命中指定展区 | category: 命中指定类别 | keywords: 文本含任一关键词
# 建议每个展厅/展区都覆盖 2-3 条，越多越准
TEST_CASES = [
    # --- 妇好展厅 ---
    {"query": "妇好是谁",           "keywords": ["武丁", "王后"]},
    {"query": "妇好墓出土了哪些文物", "venue": ["artifact"], "keywords": ["出土", "文物"]},
    {"query": "妇好打过哪些仗",      "venue": ["military"], "keywords": ["征", "伐", "军"]},
    {"query": "商朝的祭祀是怎样的",   "venue": ["ritual"], "keywords": ["祭祀"]},
    # --- 总部展厅 ---
    {"query": "公司的发展历程",      "category": ["headquarters"], "keywords": ["发展", "成立", "历程"]},
    {"query": "公司获得过哪些荣誉",   "category": ["headquarters"], "keywords": ["荣誉", "奖", "铜牌"]},
    # --- B1 万里江河 ---
    {"query": "兵马俑",             "venue": ["qin_han"], "keywords": ["兵马俑", "秦"]},
    {"query": "清明上河图",          "venue": ["song_yuan_ming_qing"], "keywords": ["清明上河图", "宋"]},
    {"query": "远古时期的展区",      "venue": ["pre_qin"], "keywords": ["先秦", "远古"]},
    {"query": "有什么AI互动体验",    "venue": ["ai_interactive", "ai_tech_hall"], "keywords": ["AI", "智能", "互动"]},
]

def resolve_embed_path():
    p = os.path.join(MODEL_DIR, "iic", "nlp_gte_sentence-embedding_chinese-base")
    if os.path.exists(p):
        return p
    for root, dirs, files in os.walk(MODEL_DIR):
        if "config.json" in files and "embedding" in root.lower():
            return root
    return None

def build_qa_cases(chunks):
    """从索引里的 QA 块自动生成测试用例：问题 → 命中它自己"""
    cases = []
    for i, c in enumerate(chunks):
        if c.get("type") == "qa":
            m = re.match(r"问：(.+?)\n答：", c["text"], re.S)
            if m:
                q = m.group(1).strip()
                if len(q) >= 2:
                    cases.append({"query": q, "chunk_id": i, "auto": True})
    return cases

def is_hit(case, chunk, idx):
    if "chunk_id" in case:
        return idx == case["chunk_id"]
    if "venue" in case and chunk["venue"] in case["venue"]:
        return True
    if "category" in case and chunk["category"] in case["category"]:
        return True
    if "keywords" in case and any(kw in chunk["text"] for kw in case["keywords"]):
        return True
    return False

def main():
    print("📦 加载索引...")
    with open(INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    chunks = data["chunks"]
    index = faiss.deserialize_index(data["index"])

    # 预检：测试集期望的 venue 必须在索引里存在，否则该用例永远失败
    venue_counts = Counter(c["venue"] for c in chunks)
    print(f"   共 {len(chunks)} 块，venue 分布: {dict(venue_counts)}\n")
    for case in TEST_CASES:
        for v in case.get("venue", []):
            if venue_counts.get(v, 0) == 0:
                print(f"⚠️  用例「{case['query']}」期望 venue={v}，但索引中没有该展区的块（将依赖 keywords 判定）")

    print("\n🧠 加载 embedding 模型...")
    embedder = SentenceTransformer(resolve_embed_path())

    # 合并手工用例 + QA 自动用例
    qa_cases = build_qa_cases(chunks)
    all_cases = TEST_CASES + qa_cases
    print(f"\n🧪 共 {len(all_cases)} 个用例（手工 {len(TEST_CASES)} + QA自动 {len(qa_cases)}）\n")

    # 批量编码所有问题
    queries = [c["query"] for c in all_cases]
    embs = embedder.encode(queries, batch_size=32).astype("float32")
    faiss.normalize_L2(embs)
    scores, ids = index.search(embs, TOP_K)

    # 逐用例算首个命中排名
    results = []
    for ci, case in enumerate(all_cases):
        rank = None
        for r, idx in enumerate(ids[ci], 1):
            if is_hit(case, chunks[idx], idx):
                rank = r
                break
        results.append({"case": case, "rank": rank,
                        "top": [(scores[ci][r], ids[ci][r]) for r in range(TOP_K)]})

    # ==================== 汇总 ====================
    def summarize(subset, name):
        if not subset:
            return
        n = len(subset)
        r1 = sum(1 for r in subset if r["rank"] == 1) / n
        r3 = sum(1 for r in subset if r["rank"] and r["rank"] <= 3) / n
        r5 = sum(1 for r in subset if r["rank"] and r["rank"] <= 5) / n
        mrr = sum(1 / r["rank"] if r["rank"] else 0 for r in subset) / n
        print(f"  {name:<10} 用例 {n:>2} | R@1 {r1:6.1%} | R@3 {r3:6.1%} | R@5 {r5:6.1%} | MRR {mrr:.3f}")

    print("=" * 70)
    print("📊 评测结果")
    print("=" * 70)
    summarize(results, "总体")
    summarize([r for r in results if not r["case"].get("auto")], "手工用例")
    summarize([r for r in results if r["case"].get("auto")], "QA自动")

    # 失败/低排名用例明细
    bad = [r for r in results if r["rank"] is None or r["rank"] > 3]
    if bad:
        print("\n" + "=" * 70)
        print("❌ 未命中或排名靠后的用例（需要重点优化）")
        print("=" * 70)
        for r in bad:
            tag = "未命中" if r["rank"] is None else f"排名第{r['rank']}"
            print(f"\n「{r['case']['query']}」 → {tag}")
            for score, idx in r["top"][:3]:
                c = chunks[idx]
                print(f"   ({score:.3f}) {c['category']}/{c['venue']} | {c['section'][:25]}")
    else:
        print("\n✅ 全部用例都在 Top3 内命中！")

if __name__ == "__main__":
    main()