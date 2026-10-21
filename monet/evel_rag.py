# -*- coding: utf-8 -*-
"""
eval_rag.py - RAG 检索质量评测

评测集优先级:
  1. test/test.json 或 test/test.xlsx（独立评测集，推荐）
  2. 否则从 index/items.json 抽出所有 QA 对做留一评测（偏乐观，仅作参考）

评测方式:
  --mode http   （默认）调 localhost:8848/search，评的是线上真实链路（含 category/reranker）
  --mode local  本地加载索引直接评（无需起 server）

指标: hit@1 / hit@3 / hit@5 / MRR，按 category 分组 + 输出 bad case

用法:
  conda activate robot
  cd C:\ro
  python eval_rag.py --mode http
  python eval_rag.py --mode http --topk 3 --out badcases.json
"""
import sys
import json
import argparse
import numpy as np
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# ============================================================
# 评测集加载
# ============================================================
def norm(s: str) -> str:
    """去空白标点，用于答案匹配"""
    import re
    return re.sub(r"[\s\u3000，。！？、；：“”‘’（）《》【】\-—…·]", "", s)

def load_testset():
    """返回 [{question, answer, category}]。
    优先 test/，否则从 items.json 抽 QA。"""
    test_dir = Path("test")

    # 1. test.json
    jf = test_dir / "test.json"
    if jf.exists():
        data = json.loads(jf.read_text(encoding="utf-8"))
        out = []
        for d in data:
            q = (d.get("question") or "").strip()
            a = (d.get("answer") or "").strip()
            if q and a:
                out.append({"question": q, "answer": a,
                            "category": d.get("category")})
        print(f"✅ 评测集: {jf}（{len(out)} 条）")
        return out

    # 2. test.xlsx（列: 问题/答案 或 question/answer）
    xf = test_dir / "test.xlsx"
    if xf.exists():
        import pandas as pd
        df = pd.read_excel(xf)
        qcol = next((c for c in df.columns if str(c).strip() in ("问题", "question", "Q")), None)
        acol = next((c for c in df.columns if str(c).strip() in ("答案", "answer", "A")), None)
        out = []
        for _, row in df.iterrows():
            q = str(row[qcol]).strip()
            a = str(row[acol]).strip()
            if q and a and q != "nan" and a != "nan":
                out.append({"question": q, "answer": a, "category": None})
        print(f"✅ 评测集: {xf}（{len(out)} 条）")
        return out

    # 3. 从 items.json 抽 QA（留一评测，偏乐观）
    items_f = Path("index/items.json")
    if items_f.exists():
        items = json.loads(items_f.read_text(encoding="utf-8"))
        out = []
        for it in items:
            if it.get("question"):
                q, a = it["question"], it["text"]
                # answer = 去掉问题行后的正文
                if "\n" in a:
                    a = a.split("\n", 1)[1]
                out.append({"question": q, "answer": a, "category": it["category"]})
        print(f"⚠️ 未找到 test/，改用 items.json 留一评测（{len(out)} 条，结果偏乐观）")
        return out

    print("❌ 无评测集：请提供 test/test.json 或 test/test.xlsx")
    return []

# ============================================================
# 检索调用
# ============================================================
def make_search(mode):
    if mode == "http":
        import requests
        def f(query, topk, category):
            try:
                r = requests.post("http://localhost:8848/search",
                                  json={"query": query, "top_k": topk,
                                        "category": category, "compress": False},
                                  timeout=10).json()
                return r.get("results", [])
            except Exception as e:
                print(f"   ⚠️ 检索失败({e})，请先启动 server.py 或改用 --mode local")
                return []
        return f
    else:
        # 本地模式：直接 import server 的检索函数
        sys.path.insert(0, str(Path(__file__).parent))
        import server
        server.load_index()
        def f(query, topk, category):
            return server.search(query, topk, category)
        return f

# ============================================================
# 命中判定
# ============================================================
def is_hit(answer: str, result_text: str) -> bool:
    """检索结果文本是否包含正确答案的关键内容。
    取答案前 20 字（或全答，取短者）作为匹配串，去空白标点后判包含。"""
    a = norm(answer)
    if not a:
        return False
    key = a[:20] if len(a) > 20 else a
    return key in norm(result_text)

# ============================================================
# 主评测
# ============================================================
def evaluate(cases, search_fn, topk):
    per_cat = {}      # category -> {hits_at_k, n, mrr_sum}
    bad = []
    n_total = len(cases)

    for i, c in enumerate(cases, 1):
        q, a, cat = c["question"], c["answer"], c.get("category")
        results = search_fn(q, topk, cat)
        texts = [r["text"] for r in results]

        hit_rank = None
        for rank, t in enumerate(texts, 1):
            if is_hit(a, t):
                hit_rank = rank
                break

        cat_key = cat or "(未分组)"
        stat = per_cat.setdefault(cat_key, {"n": 0, "hits": {k: 0 for k in (1, 3, 5)},
                                            "mrr": 0.0})
        stat["n"] += 1
        if hit_rank:
            for k in (1, 3, 5):
                if hit_rank <= k:
                    stat["hits"][k] += 1
            stat["mrr"] += 1.0 / hit_rank
        else:
            bad.append({"question": q, "answer": a, "category": cat,
                        "returned_sources": [r.get("source") for r in results],
                        "returned_scores": [r.get("score") for r in results]})

        if i % 50 == 0:
            print(f"   进度 {i}/{n_total}")

    return per_cat, bad

# ============================================================
# 输出
# ============================================================
def report(per_cat, bad, n_total, topk, out_path):
    print("\n" + "=" * 60)
    print("📊 评测结果")
    print("=" * 60)

    def agg(stat):
        n = stat["n"]
        return {k: stat["hits"][k] / n for k in (1, 3, 5)}, stat["mrr"] / n

    print(f"\n{'类别':<22}{'样本':>6}{'hit@1':>9}{'hit@3':>9}{'hit@5':>9}{'MRR':>9}")
    print("-" * 64)
    for cat, stat in sorted(per_cat.items()):
        h, mrr = agg(stat)
        print(f"{cat:<22}{stat['n']:>6}{h[1]*100:>8.1f}%{h[3]*100:>8.1f}%"
              f"{h[5]*100:>8.1f}%{mrr:>9.3f}")

    # 整体
    total = {"n": n_total, "hits": {k: 0 for k in (1, 3, 5)}, "mrr": 0.0}
    for stat in per_cat.values():
        total["n"] = stat["n"]
        for k in (1, 3, 5):
            total["hits"][k] += stat["hits"][k]
        total["mrr"] += stat["mrr"]
    n = total["n"] or 1
    h, mrr = agg(total)
    print("-" * 64)
    print(f"{'整体':<22}{n:>6}{h[1]*100:>8.1f}%{h[3]*100:>8.1f}%"
          f"{h[5]*100:>8.1f}%{mrr:>9.3f}")

    print(f"\n❌ bad case（top{topk} 未命中）: {len(bad)} / {n_total}")
    for b in bad[:20]:
        print(f"   Q: {b['question'][:40]}")
        print(f"      返回: {b['returned_sources']} | 分数: {b['returned_scores']}")

    if out_path:
        Path(out_path).write_text(json.dumps(bad, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\n💾 bad case 已保存: {out_path}")

    print("\n💡 解读：")
    print("   hit@1 < 70%  → 检索本身有坑（先别堆推理，修检索）")
    print("   hit@1 70~85% → 可接受，重点看 bad case 是'没召回'还是'召回错'")
    print("   hit@1 > 85%   → 基础好，再谈意图识别/条件 CoT 才有意义")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="http", choices=["http", "local"])
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", default="badcases.json")
    args = ap.parse_args()

    cases = load_testset()
    if not cases:
        return

    print(f"\n🔍 评测模式: {args.mode} | top_k={args.topk} | 样本 {len(cases)}")
    search_fn = make_search(args.mode)
    per_cat, bad = evaluate(cases, search_fn, args.topk)
    report(per_cat, bad, len(cases), args.topk, args.out)

if __name__ == "__main__":
    main()