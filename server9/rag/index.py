# -*- coding: utf-8 -*-
"""加载 FAISS / BM25 / embedding / 意图向量。"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

try:
    import faiss
    import jieba
    import jieba.posseg as pseg
    import numpy as np
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    HAS_RAG_DEPS = True
except ImportError as e:
    faiss = jieba = pseg = np = BM25Okapi = SentenceTransformer = None
    HAS_RAG_DEPS = False
    print(f"⚠️ RAG 依赖未安装，知识检索将不可用: {e}")

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
_persona_indexes = {}
_persona_intent_vecs = {}
_bound_persona_id = None


def bundle_for(persona=None):
    """取指定人设的索引包；未传则用请求上下文里的人设。"""
    pid = None
    if persona is not None:
        pid = getattr(persona, "id", None) or str(persona)
    if not pid:
        try:
            from persona.runtime import active
            pid = active().id
        except Exception:
            pid = _bound_persona_id
    return _persona_indexes.get(pid)


def active_items(persona=None):
    b = bundle_for(persona)
    if b:
        return b["items"]
    return items

def load_jieba_dict():
    if jieba is None:
        return
    try:
        if DICT_PATH.exists():
            jieba.load_userdict(str(DICT_PATH))
            print(f"📖 已加载专名词典: {DICT_PATH}")
    except Exception as e:
        print(f"⚠️ 词典加载失败: {e}")

def _load_bundle(index_dir):
    from pathlib import Path
    d = Path(index_dir)
    idx = faiss.read_index(str(d / "faiss.index"))
    its = json.loads((d / "items.json").read_text(encoding="utf-8"))
    fmap = json.loads((d / "faiss_map.json").read_text(encoding="utf-8"))
    with open(d / "bm25.pkl", "rb") as f:
        docs = pickle.load(f)["docs"]
        bm = BM25Okapi(docs)
    if len(docs) != len(its):
        raise RuntimeError(
            f"BM25 docs({len(docs)}) 与 items({len(its)}) 数量不一致，"
            f"请重新运行 build 脚本保证两者等长同序: {d}")
    print(f"✅ [{d.name}] FAISS {idx.ntotal} 向量 | items {len(its)} | BM25 {len(docs)} 篇")
    return {"index": idx, "items": its, "faiss_map": fmap, "bm25": bm}


def bind_persona(persona) -> None:
    """把全局检索指针切到指定人设的索引。"""
    global index, items, faiss_map, bm25, _bound_persona_id
    pid = getattr(persona, "id", None) or str(persona)
    bundle = _persona_indexes.get(pid)
    if not bundle:
        index = None
        items = []
        faiss_map = []
        bm25 = None
        _bound_persona_id = pid
        print(f"⚠️ 人设 {pid} 无知识索引，检索将拒答")
        return
    index = bundle["index"]
    items = bundle["items"]
    faiss_map = bundle["faiss_map"]
    bm25 = bundle["bm25"]
    _bound_persona_id = pid


def reload_persona_index(persona) -> bool:
    """发布后热加载某人设索引。"""
    pid = getattr(persona, "id", None) or str(persona)
    path = getattr(persona, "index_path", None)
    if path is None:
        return False
    if not HAS_RAG_DEPS:
        print(f"⚠️ 无法热加载 {pid}：缺少 FAISS 依赖")
        return False
    try:
        _persona_indexes[pid] = _load_bundle(path)
    except Exception as e:
        print(f"⚠️ 热加载 {pid} 失败: {e}")
        return False
    from persona.stage import stage
    if stage.current_id() == pid:
        bind_persona(persona)
    return True


def load_index():
    global embedder, index, items, faiss_map, bm25, reranker, HAS_RERANKER, intent_vecs, HAS_INTENT
    from persona.stage import stage
    from persona.registry import registry
    if not registry._items:
        registry.load()
    stage.bootstrap()
    if not HAS_RAG_DEPS:
        print("⚠️ 以降级模式启动：无 FAISS/embedding，仅人设与 HTTP 可用")
        load_cosyvoice()
        return
    load_jieba_dict()
    if not EMBED_MODEL or not Path(EMBED_MODEL).exists():
        print("⚠️ embedding 模型未找到，知识检索不可用。")
        print("   现场请用魔搭下载到 models/gte-base-zh：")
        print("   pip install modelscope -i https://mirrors.aliyun.com/pypi/simple/")
        print("   python scripts/download_gte_modelscope.py")
        load_cosyvoice()
        return
    print("🧮 加载 embedding 模型:", EMBED_MODEL)
    from rag.binload import allow_local_torch_load, embed_dir_needs_bin
    if embed_dir_needs_bin(EMBED_MODEL):
        patched = allow_local_torch_load()
        print("embedding 是 pytorch_model.bin，已允许在 torch<2.6 下加载本机权重 patches=%s" % patched)
        runtime_log("startup.log", "[BOOT] allow pytorch_model.bin patches=%s" % patched)
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    from persona.registry import registry
    for p in registry.enabled():
        if p.knowledge_ready():
            try:
                print(f"📦 加载索引: {p.index_path}")
                _persona_indexes[p.id] = _load_bundle(p.index_path)
            except Exception as e:
                print(f"⚠️ 人设 {p.id} 索引加载失败: {e}")
        else:
            print(f"⚠️ 人设 {p.id} 知识库未就绪: {p.index_path}")
    bind_persona(stage.current())
    if index is None and not _persona_indexes:
        print("⚠️ 没有任何可用知识索引，请检查 index/ 或 index/monet/（服务仍启动）")
    if RERANK_MODEL:
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder(RERANK_MODEL)
            HAS_RERANKER = True
            print(f"✅ reranker: {RERANK_MODEL}")
        except Exception as e:
            print(f"⚠️ reranker 加载失败: {e}")
    try:
        shared_cmd = ["声音大一点", "音量调大", "大声点", "小声点", "声音小一点",
                      "暂停", "别说了", "停", "继续", "再说一遍", "重复一遍"]
        shared_chat = ["你好", "嗨", "您好", "你是谁", "你叫什么", "谢谢", "再见",
                       "今天天气怎么样", "你会什么", "聊聊"]
        for p in registry.enabled():
            examples = dict(p.intent_examples or {})
            examples.setdefault("command", shared_cmd)
            examples.setdefault("chitchat", shared_chat)
            examples.setdefault("knowledge", [p.display_name, "这个展品是什么", "介绍一下"])
            vecs = {}
            for label, exs in examples.items():
                if not exs:
                    continue
                v = embedder.encode(exs, normalize_embeddings=True).mean(axis=0)
                vecs[label] = v / np.linalg.norm(v)
            _persona_intent_vecs[p.id] = vecs
            if p.id == stage.current_id():
                intent_vecs = vecs
        HAS_INTENT = bool(_persona_intent_vecs)
        print(f"✅ 意图识别: {list(_persona_intent_vecs.keys())}")
    except Exception as e:
        print(f"⚠️ 意图识别初始化失败: {e}")
    print("🔥 预热 embedding / 分词模型...")
    embedder.encode(["warmup"], normalize_embeddings=True)
    list(pseg.cut("预热词性标注模型"))  # pseg 首次调用要加载词性模型，提前预热
    print("✅ 预热完成")
    load_cosyvoice()
