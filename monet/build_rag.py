#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_rag.py - 离线索引构建（跑一次，生成 index/ 供 server.py 加载）

链路：
  docx/xlsx → 章节切块(500字/重叠100, 超长降级滑窗)
  → QA 双向量多表示索引(纯问题 + 问答全文, 共享 dedup_key)
  → gte 编码 + L2 归一化 + FAISS(IndexFlatIP)
  → jieba(专名词典) + BM25Okapi
  → 保存 index/

知识库现在只有莫奈展（含浪漫与星辰票务），所有条目 category=exhibition_monet，
与 server.py 的 EXHIBITION_CATEGORY = ["exhibition_monet"] 对应。

用法:
  conda activate robot
  cd C:\ro                ← 必须和 server4.py 启动目录一致，
  python build_rag.py        两边读写的才是同一个 index/
"""
import sys
import json
import pickle
import hashlib
from pathlib import Path

# 强制 UTF-8（Windows）
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# ============================================================
# 配置
# ============================================================

# ★ 原始文档目录：莫奈讲解词 docx 放这里（不存在会自动创建）
DATA_DIR = Path(r"C:\knowledge_base")
OUTPUT_DIR = Path("index")       # 索引输出目录（相对当前运行目录）
# ★ 必须与 server4.py 的 EMBED_MODEL 一字不差（建索引和线上检索共用同一模型）
#   如果下面这行报「模型路径不存在」，直接打开 server4.py 把那边的这一行原样复制过来
EMBED_MODEL = r"C:\Users\BroadMesse\.cache\huggingface\hub\models--thenlper--gte-base-zh\snapshots\71ab7947d6fac5b64aa299e6e40e6c2b2e85976c"
CHUNK_SIZE = 500                 # 章节切块字数
CHUNK_OVERLAP = 100              # 超长章节滑窗重叠字数
DICT_PATH = Path("dict/jieba_dict.txt")   # 专名词典（可选，不存在则跳过）

# 文件名/目录 → 知识库类别（只保留莫奈展）
CATEGORY_MAP = {
    "monet": "exhibition_monet",   # 莫奈展（英文文件名）
    "莫奈":  "exhibition_monet",   # 莫奈展（中文文件名）
    "浪漫":  "exhibition_monet",   # 浪漫与星辰展 → 莫奈
    "星辰":  "exhibition_monet",   # 浪漫与星辰展 → 莫奈
}

# ============================================================
# 票务数据自动生成
# ============================================================

def ensure_ticket_qa():
    """确保票务QA文件存在，不存在则自动创建"""
    ticket_file = DATA_DIR / "浪漫与星辰票务QA.xlsx"

    if ticket_file.exists():
        print(f"✅ 票务QA文件已存在: {ticket_file.name}")
        return

    print(f"📝 生成票务QA文件: {ticket_file.name}")

    ticket_qa = [
        {
            "问题": "浪漫与星辰展览的开放时间是什么？",
            "答案": "展览周期：2026年7月10日至2026年10月10日。每日开放时间 9:00-18:00，周六 9:00-21:00。17:00停止入馆（周六20:00），周一闭馆（法定节假日除外）。"
        },
        {
            "问题": "浪漫与星辰展览在哪里举办？",
            "答案": "展览地址：贵阳市观山湖区林城东路107号贵州省博物馆负一楼6号厅。"
        },
        {
            "问题": "浪漫与星辰门票多少钱？",
            "答案": "单人票：早鸟价49元（即日起至7月10日0:00），正价78元（7月10日0:00至10月10日17:00），原价89元。"
        },
        {
            "问题": "浪漫与星辰有学生票吗？",
            "答案": "有学生票。全日制本科及以下在校学生（含大专、中专、高中、初中、小学）可购买学生票：早鸟价39元，正价49元，原价89元。入馆时须出示有效学生证。"
        },
        {
            "问题": "浪漫与星辰有优惠票吗？",
            "答案": "有优惠票。1-1.4米儿童、65周岁以上人群可购买优惠票：早鸟价29元，正价39元，原价89元。凭票+身份证证件入场。"
        },
        {
            "问题": "浪漫与星辰有免票政策吗？",
            "答案": "有免票政策。身高1米以下儿童、残障人士凭证件免费入场。注意：6周岁（含）以下人群须购票成人陪同观展。"
        },
        {
            "问题": "浪漫与星辰有什么特殊票种？",
            "答案": "盖章手册：19.9元/本（非门票，集章手册，不可邮寄，仅限现场兑换领取）。国庆特惠单人套票：88元（2026.9.25-9.30期间售卖），套餐包含1张单人票+1本盖章册子，原价118元。"
        },
        {
            "问题": "浪漫与星辰早鸟票什么时候能买？",
            "答案": "早鸟票销售时间：即日起至2026年7月10日0:00。早鸟价：单人票49元、优惠票29元、学生票39元。"
        },
        {
            "问题": "浪漫与星辰展览什么时候停止入馆？",
            "答案": "平日17:00停止入馆，周六20:00停止入馆。展览使用时间：2026年7月10日至2026年10月10日。"
        },
    ]

    import pandas as pd
    df = pd.DataFrame(ticket_qa)
    df.to_excel(ticket_file, index=False, engine='openpyxl')
    print(f"   ✅ 已生成 {len(ticket_qa)} 条票务问答")

# ============================================================
# 莫奈高频 FAQ 自动生成
# ============================================================

def ensure_monet_faq():
    """确保莫奈高频FAQ文件存在，不存在则自动创建。
    覆盖检索擦线/串台的高频问题，QA 纯问题向量让这类问题直接精确命中。
    数字与细节请对照展厅讲解词核对，有出入以讲解词为准。"""
    faq_file = DATA_DIR / "monet高频QA.xlsx"

    if faq_file.exists():
        print(f"✅ 莫奈FAQ文件已存在: {faq_file.name}")
        return

    print(f"📝 生成莫奈FAQ文件: {faq_file.name}")

    monet_faq = [
        {
            "问题": "莫奈的代表作是什么？",
            "答案": "莫奈的代表作包括：《日出·印象》（1872年，画的是勒阿弗尔港的日出，印象派的名称正来源于这幅画）、《睡莲》系列（晚年在吉维尼花园持续创作约30年，共250余幅）、《鲁昂大教堂》系列（同一座教堂在不同时间、不同光线下画了30多幅）、《干草堆》系列。这些作品的共同主题是捕捉光与色彩的瞬间变化。"
        },
        {
            "问题": "莫奈是谁？介绍一下莫奈。",
            "答案": "克洛德·莫奈（Claude Monet，1840-1926），法国印象派创始人之一，被称为'印象派之父'。他主张走出画室到户外直接作画，捕捉自然光线的瞬息变化。晚年定居吉维尼，在自家花园修建睡莲池塘并以此为题材创作了大量作品。"
        },
        {
            "问题": "《日出·印象》这幅画的名字是怎么来的？",
            "答案": "1872年莫奈在勒阿弗尔港画下这幅日出景象，参展时随手取名《日出·印象》。1874年首展时，评论家勒鲁瓦在讽刺文章中借这个标题嘲笑整个画派'只凭印象作画'，'印象派'这个原本带贬义的称呼反而被画家们接受，成为艺术史上最重要的流派名称之一。"
        },
        {
            "问题": "莫奈为什么画了那么多睡莲？",
            "答案": "莫奈晚年定居吉维尼，亲手设计了带日本桥的睡莲池塘。从1897年直到1926年去世，他画了约250幅睡莲，因为水面的反光、浮萍与天空倒影的交织让光色变化无穷无尽，同一个池塘在清晨、正午、黄昏呈现完全不同的面貌。晚年他视力衰退，笔下的睡莲愈发抽象自由，被视为抽象艺术的先声。"
        },
        {
            "问题": "什么是印象派？",
            "答案": "印象派是19世纪60-70年代兴起于法国的绘画流派，代表画家有莫奈、雷诺阿、德加、毕沙罗等。核心主张：走出画室户外写生，描绘光线下色彩的瞬间印象，用短促笔触并置纯色而非在调色板上调和。名称来自莫奈《日出·印象》，最初是评论界的嘲讽用语。"
        },
        {
            "问题": "莫奈为什么反复画同一个场景？",
            "答案": "莫奈相信同一景物在不同时刻、不同天气、不同季节的光线下是完全不同的画面。他常常同时支起十几块画布，随着光线推移逐一切换作画。《干草堆》《鲁昂大教堂》《睡莲》都是这样的系列创作，目的是记录光的时间性——这是他艺术方法最独特的地方。"
        },
        {
            "问题": "莫奈的吉维尼花园是什么样的？",
            "答案": "1883年莫奈搬到巴黎郊外的吉维尼，亲手打造了两座花园：花园'诺曼底园'种满四季花卉；水园引河水修建池塘，种植睡莲，架设日本式拱桥，周围植柳树与竹子。这座花园既是他的家，也是他后半生最重要的创作对象，《睡莲》系列全部诞生于此。"
        },
    ]

    import pandas as pd
    df = pd.DataFrame(monet_faq)
    df.to_excel(faq_file, index=False, engine='openpyxl')
    print(f"   ✅ 已生成 {len(monet_faq)} 条莫奈问答")

# ============================================================
# 专名词典
# ============================================================

def load_jieba_dict():
    try:
        import jieba
        if DICT_PATH.exists():
            jieba.load_userdict(str(DICT_PATH))
            print(f"📖 已加载专名词典: {DICT_PATH}")
    except Exception as e:
        print(f"⚠️ 词典加载失败（忽略）: {e}")

# ============================================================
# 文档读取
# ============================================================

def load_docx(path):
    """docx → [(标题, 正文)]，按标题样式识别章节"""
    from docx import Document
    doc = Document(path)
    chapters, cur_title, buf = [], "", []

    def flush():
        nonlocal cur_title, buf
        text = "\n".join(buf).strip()
        if text:
            chapters.append({"title": cur_title or path.stem, "text": text})
        buf = []

    for p in doc.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        if p.style and p.style.name and p.style.name.lower().startswith("heading"):
            flush()
            cur_title = t
        else:
            buf.append(t)
    flush()
    return chapters

def load_qa_xlsx(path):
    """xlsx → QA 对列表。列名支持 问题/答案 或 question/answer"""
    import pandas as pd
    df = pd.read_excel(path)
    # 统一列名
    qcol = next((c for c in df.columns if str(c).strip() in ("问题", "question", "Q")), None)
    acol = next((c for c in df.columns if str(c).strip() in ("答案", "answer", "A")), None)
    if not qcol or not acol:
        print(f"⚠️ {path.name} 未识别到 问题/答案 列，跳过")
        return []
    pairs = []
    for _, row in df.iterrows():
        q = str(row[qcol]).strip()
        a = str(row[acol]).strip()
        if q and a and q != "nan" and a != "nan":
            pairs.append({"question": q, "answer": a})
    return pairs

def category_of(path):
    name = path.stem.lower() + path.parent.name.lower()
    for key, cat in CATEGORY_MAP.items():
        if key in name:
            return cat
    return "exhibition_monet"   # 兜底默认（知识库现在只有莫奈）

# ============================================================
# 切块
# ============================================================

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """超长章节滑窗切块；长度 <= size 直接返回"""
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks

# ============================================================
# 主流程
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("📚 构建 RAG 索引（莫奈展）")
    print("=" * 60)

    # 0a. 配置自检：embedding 模型路径必须真实存在，否则直接给出可操作的报错
    if not Path(EMBED_MODEL).exists():
        print(f"❌ embedding 模型路径不存在:\n   {EMBED_MODEL}")
        print("   请打开 server4.py，把那边 EMBED_MODEL 那一行原样复制到本文件（两边必须一字不差）")
        return

    # 0b. 文档目录不存在则创建（pandas 写 xlsx 不会自动建目录）
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"❌ 无法创建文档目录 {DATA_DIR}: {e}")
        print("   请检查 DATA_DIR 的盘符/路径是否正确")
        return
    print(f"📂 文档目录: {DATA_DIR}")

    # 1. 确保票务QA与莫奈FAQ文件存在
    ensure_ticket_qa()
    ensure_monet_faq()

    load_jieba_dict()
    import jieba
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    import faiss

    # 2. 读取所有文档 → 条目(items)
    items = []   # 每个 item: {id, dedup_key, question, text, source, category, title}
    n_qa, n_chunk = 0, 0

    for path in sorted(DATA_DIR.rglob("*")):
        if path.suffix.lower() not in (".docx", ".xlsx"):
            continue
        if path.name.startswith("~$"):
            continue   # 跳过 Office 打开文件时的临时锁文件
        cat = category_of(path)
        print(f"📄 读取 {path.name} → {cat}")

        if path.suffix.lower() == ".docx":
            for ch in load_docx(path):
                for i, c in enumerate(chunk_text(ch["text"])):
                    did = hashlib.md5(f"{path}#{ch['title']}#{i}".encode()).hexdigest()[:12]
                    items.append({
                        "id": did, "dedup_key": did,
                        "question": None,
                        "text": c,
                        "source": f"{path.name}#{ch['title']}",
                        "category": cat, "title": ch["title"],
                    })
                    n_chunk += 1
        else:
            for pair in load_qa_xlsx(path):
                did = hashlib.md5(f"{path}#{pair['question']}".encode()).hexdigest()[:12]
                items.append({
                    "id": did, "dedup_key": did,
                    "question": pair["question"],
                    "text": f"{pair['question']}\n{pair['answer']}",
                    "source": path.name,
                    "category": cat, "title": pair["question"][:30],
                })
                n_qa += 1

    if not items:
        print(f"❌ {DATA_DIR} 目录下没有可用的 docx/xlsx，退出")
        return

    print(f"\n✅ 共 {len(items)} 条：QA {n_qa} 条，章节块 {n_chunk} 条")
    if n_chunk == 0:
        print("⚠️ 警告：没有读到任何 docx 章节，知识库里只有自动生成的 QA！")
        print(f"⚠️ 请确认莫奈讲解词 docx 已放入: {DATA_DIR}")

    # 3. 多表示：QA 建 2 向量(纯问题 + 全文)，章节块建 1 向量
    reps, rep_to_item = [], []   # rep_to_item[i] = items 下标
    for idx, it in enumerate(items):
        if it["question"]:
            reps.append(it["question"])            # 表示1: 纯问题
            rep_to_item.append(idx)
            reps.append(it["text"])                # 表示2: 问答全文
            rep_to_item.append(idx)
        else:
            reps.append(it["text"])                # 单表示: 正文
            rep_to_item.append(idx)
    print(f"   多表示向量数: {len(reps)}（其中 {len(items)} 个条目）")

    # 4. gte 编码 + L2 归一化
    print(f"\n🧮 加载 embedding 模型: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    print("   编码中...")
    vecs = embedder.encode(reps, normalize_embeddings=True,
                           show_progress_bar=True, batch_size=32).astype("float32")

    # 5. FAISS (IndexFlatIP：归一化后内积 = 余弦相似度)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    print(f"✅ FAISS 索引: {index.ntotal} 条向量, dim={dim}")

    # 6. BM25（对唯一条目建文档，全文分词）
    docs = [list(jieba.cut(it["text"])) for it in items]
    bm25 = BM25Okapi(docs)
    print(f"✅ BM25 语料: {len(docs)} 篇")

    # 7. 保存
    OUTPUT_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(OUTPUT_DIR / "faiss.index"))
    with open(OUTPUT_DIR / "items.json", "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / "faiss_map.json", "w", encoding="utf-8") as f:
        json.dump(rep_to_item, f)                       # faiss 第 i 行 → items 下标
    with open(OUTPUT_DIR / "bm25.pkl", "wb") as f:
        pickle.dump({"docs": docs}, f)                  # 分词后语料（server 端重建 BM25）

    print("\n" + "=" * 60)
    print("✅ 索引构建完成，已保存到 index/:")
    print("   faiss.index    稠密向量索引")
    print("   items.json     条目元数据（text/source/category/dedup_key）")
    print("   faiss_map.json 向量行 → 条目 映射")
    print("   bm25.pkl       BM25 分词语料")
    print("=" * 60)

if __name__ == "__main__":
    main()