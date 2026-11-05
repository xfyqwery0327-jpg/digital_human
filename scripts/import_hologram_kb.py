# -*- coding: utf-8 -*-
"""把全息柜三个人物的自述和观众问答导入知识库，并重建向量索引。

观众实录另存 kb/{persona}/transcript.json。问句对上时按原话回答。
"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER9 = ROOT / "server9"
if str(SERVER9) not in sys.path:
    sys.path.insert(0, str(SERVER9))

from rag.transcript import (  # noqa: E402
    normalize_question, paragraphs_of_docx, parse_labeled_qa, parse_narrative,
    parse_transcript, parse_transcript_aside,
)

DEFAULT_SRC = Path(r"C:\Users\BroadMesse\Desktop\全息柜人物数据导入")
PERSONAS = (
    ("fuhao", "妇好", "exhibition_fuhao"),
    ("monet", "莫奈", "exhibition_monet"),
    ("kumarajiva", "鸠摩罗什", "exhibition_kumarajiva"),
)


def _find_docx(folder: Path, kind: str) -> Path:
    hits = [p for p in folder.glob("*.docx") if kind in p.name and not p.name.startswith("~$")]
    if len(hits) != 1:
        raise FileNotFoundError("%s 里没有唯一的%s文稿: %s" % (folder, kind, hits))
    return hits[0]


def _persona_dir(src: Path, name: str) -> Path:
    hits = [p for p in src.iterdir() if p.is_dir() and name in p.name]
    if len(hits) != 1:
        raise FileNotFoundError("找不到 %s 的目录: %s" % (name, hits))
    return hits[0]


def _items_for(pid: str, category: str, self_name: str, qa_name: str,
               narrative: list, qa: list) -> tuple:
    rows = []
    transcript = []
    for i, sec in enumerate(narrative, 1):
        title = sec["title"]
        body = sec["text"]
        text = body if title == "开场" else "%s\n%s" % (title, body)
        rows.append({
            "id": "%s_hologram_self_%02d" % (pid, i),
            "title": title,
            "text": text,
            "source": self_name,
            "category": category,
            "dedup_key": "hologram:self:%s:%02d" % (pid, i),
            "kind": "self",
        })
    for i, pair in enumerate(qa, 1):
        question = pair["question"]
        answer = pair["answer"]
        rows.append({
            "id": "%s_hologram_qa_%02d" % (pid, i),
            "title": question,
            "question": question,
            "answer": answer,
            "text": "问：%s\n答：%s" % (question, answer),
            "source": qa_name,
            "category": category,
            "dedup_key": "hologram:qa:%s:%02d" % (pid, i),
            "kind": "transcript",
        })
        transcript.append({"question": question, "answer": answer})
    return rows, transcript


def _load_json(path: Path) -> list:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _merge(existing: list, fresh: list) -> list:
    kept = [row for row in existing if not str(row.get("dedup_key") or "").startswith("hologram:")]
    kept.extend(fresh)
    return kept


def _embedder():
    from sentence_transformers import SentenceTransformer
    from config import EMBED_MODEL
    from rag.binload import allow_local_torch_load, embed_dir_needs_bin

    if embed_dir_needs_bin(EMBED_MODEL):
        allow_local_torch_load()
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def _rebuild(pid: str, items: list, model) -> None:
    import jieba
    import faiss
    import numpy as np
    from config import DICT_PATH

    if DICT_PATH.exists():
        jieba.load_userdict(str(DICT_PATH))
    texts = [row["text"] for row in items]
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype("float32")
    index = faiss.IndexFlatIP(int(vecs.shape[1]))
    index.add(np.ascontiguousarray(vecs))
    docs = [list(jieba.cut(text)) for text in texts]
    dest = ROOT / "index" / pid
    dest.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(dest / "faiss.index"))
    (dest / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "faiss_map.json").write_text(
        json.dumps(list(range(len(items))), ensure_ascii=False), encoding="utf-8")
    with open(dest / "bm25.pkl", "wb") as handle:
        pickle.dump({"docs": docs}, handle)
    # 草稿与线上一致，发布页不会把刚导入的材料盖掉。
    draft = ROOT / "kb" / pid / "draft"
    draft.mkdir(parents=True, exist_ok=True)
    for name in ("faiss.index", "items.json", "faiss_map.json", "bm25.pkl"):
        shutil.copy2(dest / name, draft / name)
    print("index", pid, "items", len(items), "vectors", int(index.ntotal))


def import_all(src: Path) -> None:
    model = _embedder()
    for pid, name, category in PERSONAS:
        folder = _persona_dir(src, name)
        self_path = _find_docx(folder, "自述")
        qa_path = _find_docx(folder, "问答")
        narrative = parse_narrative(paragraphs_of_docx(self_path))
        qa = parse_transcript(paragraphs_of_docx(qa_path))
        if len(qa) < 10 or len(narrative) < 8:
            raise RuntimeError("%s 解析过少 self=%s qa=%s" % (pid, len(narrative), len(qa)))
        fresh, transcript = _items_for(pid, category, self_path.name, qa_path.name, narrative, qa)
        sources = ROOT / "kb" / pid / "sources"
        sources.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self_path, sources / self_path.name)
        shutil.copy2(qa_path, sources / qa_path.name)
        extra_rows, extra_pairs = _booth_qa(pid, category)
        transcript = _merge_transcript(transcript, extra_pairs)
        transcript_path = ROOT / "kb" / pid / "transcript.json"
        transcript_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
        chunks_path = ROOT / "kb" / pid / "chunks.json"
        chunks = _add_missing(_merge(_load_json(chunks_path), fresh), extra_rows)
        chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        live = _add_missing(_merge(_load_json(ROOT / "index" / pid / "items.json"), fresh), extra_rows)
        print(pid, "self", len(narrative), "qa", len(qa), "extra", len(extra_pairs), "live", len(live))
        _rebuild(pid, live, model)


def append_asides(src: Path) -> None:
    """问答实录的开场和结束语补进已建好的索引，不再重算全部向量。"""
    import faiss
    import jieba
    import numpy as np
    from config import DICT_PATH

    if DICT_PATH.exists():
        jieba.load_userdict(str(DICT_PATH))
    model = _embedder()
    for pid, name, category in PERSONAS:
        folder = _persona_dir(src, name)
        qa_path = _find_docx(folder, "问答")
        frames = parse_transcript_aside(paragraphs_of_docx(qa_path))
        live_path = ROOT / "index" / pid / "items.json"
        items = _load_json(live_path)
        have = {row.get("dedup_key") for row in items}
        fresh = []
        for i, frame in enumerate(frames, 1):
            key = "hologram:aside:%s:%02d" % (pid, i)
            if key in have:
                continue
            fresh.append({
                "id": "%s_hologram_aside_%02d" % (pid, i),
                "title": frame["title"],
                "text": frame["text"],
                "source": qa_path.name,
                "category": category,
                "dedup_key": key,
                "kind": "transcript",
            })
        if fresh:
            vecs = model.encode(
                [row["text"] for row in fresh], normalize_embeddings=True,
                show_progress_bar=False).astype("float32")
            index_path = ROOT / "index" / pid / "faiss.index"
            index = faiss.read_index(str(index_path))
            index.add(np.ascontiguousarray(vecs))
            items.extend(fresh)
            staged = index_path.with_suffix(".index.staged")
            faiss.write_index(index, str(staged))
            staged.replace(index_path)
            _replace_text(ROOT / "index" / pid / "items.json",
                          json.dumps(items, ensure_ascii=False, indent=2))
            fmap = json.loads((ROOT / "index" / pid / "faiss_map.json").read_text(encoding="utf-8"))
            fmap.extend(range(len(fmap), len(fmap) + len(fresh)))
            _replace_text(ROOT / "index" / pid / "faiss_map.json",
                          json.dumps(fmap, ensure_ascii=False))
            with open(ROOT / "index" / pid / "bm25.pkl", "rb") as handle:
                packed = pickle.load(handle)
            docs = packed["docs"]
            docs.extend(list(jieba.cut(row["text"])) for row in fresh)
            _replace_pickle(ROOT / "index" / pid / "bm25.pkl", {"docs": docs})
            print(pid, "aside added", len(fresh), "vectors", int(index.ntotal), "docs", len(docs))
        else:
            print(pid, "aside already in")
        _sync_hologram_chunks(pid, items)
        _copy_draft(pid)


def _replace_text(path: Path, text: str) -> None:
    staged = path.with_suffix(path.suffix + ".staged")
    staged.write_text(text, encoding="utf-8")
    staged.replace(path)


def _replace_pickle(path: Path, payload) -> None:
    staged = path.with_suffix(path.suffix + ".staged")
    with open(staged, "wb") as handle:
        pickle.dump(payload, handle)
    staged.replace(path)


def _copy_draft(pid: str) -> None:
    draft = ROOT / "kb" / pid / "draft"
    if not draft.is_dir():
        return
    for filename in ("faiss.index", "items.json", "faiss_map.json", "bm25.pkl"):
        src = ROOT / "index" / pid / filename
        dst = draft / filename
        staged = dst.with_suffix(dst.suffix + ".staged")
        shutil.copy2(src, staged)
        staged.replace(dst)


def _sync_hologram_chunks(pid: str, items: list) -> None:
    """把索引里的全息柜条目补回 chunks，已有的不重复。"""
    chunks_path = ROOT / "kb" / pid / "chunks.json"
    chunks = _load_json(chunks_path)
    have = {row.get("dedup_key") for row in chunks}
    extra = [
        row for row in items
        if str(row.get("dedup_key") or "").startswith("hologram:") and row.get("dedup_key") not in have
    ]
    if not extra:
        return
    chunks.extend(extra)
    _replace_text(chunks_path, json.dumps(chunks, ensure_ascii=False, indent=2))


def _booth_qa(pid: str, category: str) -> tuple:
    """展台后补的问答对。不占用全息柜稿的 hologram 编号，整库重导时仍保留。"""
    import hashlib
    path = ROOT / "kb" / pid / "sources" / "问答对.txt"
    if not path.is_file():
        return [], []
    pairs = parse_labeled_qa(path.read_text(encoding="utf-8"))
    rows = []
    kept = []
    for pair in pairs:
        question = (pair.get("question") or "").strip()
        answer = (pair.get("answer") or "").strip()
        if not question or not answer:
            continue
        digest = hashlib.sha1(normalize_question(question).encode("utf-8")).hexdigest()[:8]
        rows.append({
            "id": "%s_booth_qa_%s" % (pid, digest),
            "title": question,
            "question": question,
            "answer": answer,
            "text": "问：%s\n答：%s" % (question, answer),
            "source": path.name,
            "category": category,
            "dedup_key": "booth:qa:%s:%s" % (pid, digest),
            "kind": "transcript",
        })
        kept.append({"question": question, "answer": answer})
    return rows, kept


def _merge_transcript(existing: list, extra: list) -> list:
    seen = {normalize_question(row.get("question") or "") for row in existing}
    merged = list(existing)
    for row in extra:
        key = normalize_question(row.get("question") or "")
        if not key or key in seen:
            continue
        merged.append({"question": row["question"], "answer": row["answer"]})
        seen.add(key)
    return merged


def _add_missing(existing: list, extra: list) -> list:
    have = {row.get("dedup_key") for row in existing}
    merged = list(existing)
    for row in extra:
        if row.get("dedup_key") in have:
            continue
        merged.append(row)
        have.add(row.get("dedup_key"))
    return merged


def append_booth_qa(pid: str) -> None:
    """把 sources/问答对.txt 补进实录和向量索引，只给新问句算向量。"""
    import faiss
    import jieba
    import numpy as np
    from config import DICT_PATH

    category = next(item[2] for item in PERSONAS if item[0] == pid)
    rows, pairs = _booth_qa(pid, category)
    if not rows:
        raise FileNotFoundError("%s 没有可导入的问答对" % pid)
    transcript_path = ROOT / "kb" / pid / "transcript.json"
    transcript = _merge_transcript(_load_json(transcript_path), pairs)
    _replace_text(transcript_path, json.dumps(transcript, ensure_ascii=False, indent=2))
    chunks_path = ROOT / "kb" / pid / "chunks.json"
    chunks = _add_missing(_load_json(chunks_path), rows)
    _replace_text(chunks_path, json.dumps(chunks, ensure_ascii=False, indent=2))

    live_path = ROOT / "index" / pid / "items.json"
    items = _load_json(live_path)
    fresh = [row for row in rows if row["dedup_key"] not in {item.get("dedup_key") for item in items}]
    if not fresh:
        print(pid, "booth qa already in", len(rows))
        _copy_draft(pid)
        return
    if DICT_PATH.exists():
        jieba.load_userdict(str(DICT_PATH))
    model = _embedder()
    vecs = model.encode(
        [row["text"] for row in fresh], normalize_embeddings=True,
        show_progress_bar=False).astype("float32")
    index_path = ROOT / "index" / pid / "faiss.index"
    index = faiss.read_index(str(index_path))
    if index.ntotal != len(items):
        raise RuntimeError("%s 向量数 %s 和条目 %s 不一致" % (pid, index.ntotal, len(items)))
    index.add(np.ascontiguousarray(vecs))
    items.extend(fresh)
    staged = index_path.with_suffix(".index.staged")
    faiss.write_index(index, str(staged))
    staged.replace(index_path)
    _replace_text(live_path, json.dumps(items, ensure_ascii=False, indent=2))
    fmap = json.loads((ROOT / "index" / pid / "faiss_map.json").read_text(encoding="utf-8"))
    fmap.extend(range(len(fmap), len(fmap) + len(fresh)))
    _replace_text(ROOT / "index" / pid / "faiss_map.json", json.dumps(fmap, ensure_ascii=False))
    with open(ROOT / "index" / pid / "bm25.pkl", "rb") as handle:
        packed = pickle.load(handle)
    docs = packed["docs"]
    if len(docs) != index.ntotal - len(fresh):
        raise RuntimeError("%s bm25 %s 和原条目数不一致" % (pid, len(docs)))
    docs.extend(list(jieba.cut(row["text"])) for row in fresh)
    _replace_pickle(ROOT / "index" / pid / "bm25.pkl", {"docs": docs})
    _copy_draft(pid)
    print(pid, "booth qa added", len(fresh), "vectors", int(index.ntotal))


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    if str(folder) == "aside":
        append_asides(DEFAULT_SRC)
    elif str(folder) == "booth-qa":
        who = sys.argv[2] if len(sys.argv) > 2 else "fuhao"
        append_booth_qa(who)
    else:
        import_all(folder)
