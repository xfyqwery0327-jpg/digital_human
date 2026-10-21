# -*- coding: utf-8 -*-
"""莫奈 / 妇好 冒烟：离线 FAISS + 在线 HTTP API + WebSocket。

不依赖 TTS / LLM。默认先测本地索引，再连已启动的讲解员。

用法（项目根，digital_human 环境）:

    python scripts/test_monet_fuhao_smoke.py
    python scripts/test_monet_fuhao_smoke.py --offline
    python scripts/test_monet_fuhao_smoke.py --wait 180
    python scripts/test_monet_fuhao_smoke.py --persona monet
    python scripts/test_monet_fuhao_smoke.py --base http://10.0.7.232:8848
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

INDEX_FILES = ("faiss.index", "items.json", "faiss_map.json", "bm25.pkl")

CASES: Dict[str, dict] = {
    "monet": {
        "display_name": "莫奈",
        "category": "exhibition_monet",
        "query": "介绍一下莫奈",
        "in_keywords": ("莫奈", "睡莲", "印象", "吉维尼", "日出"),
        "cross": "妇好是谁",
        "must_not": ("妇好", "武丁", "殷墟", "甲骨文", "鸠摩罗什"),
    },
    "fuhao": {
        "display_name": "妇好",
        "category": "exhibition_fuhao",
        "query": "妇好介绍下自己",
        "in_keywords": ("妇好", "甲骨", "商", "武丁", "殷墟", "青铜"),
        "cross": "莫奈是谁",
        "must_not": ("莫奈", "睡莲", "印象派", "吉维尼", "鸠摩罗什"),
    },
    "kumarajiva": {
        "display_name": "鸠摩罗什",
        "category": "exhibition_kumarajiva",
        "query": "鸠摩罗什介绍下自己",
        "in_keywords": ("鸠摩罗什", "龟兹", "金刚", "法华", "译", "长安", "舍利"),
        "cross": "莫奈是谁",
        "must_not": ("莫奈", "睡莲", "印象派", "妇好", "殷墟"),
    },
}


@dataclass
class Result:
    name: str
    ok: bool
    detail: str = ""
    skipped: bool = False


@dataclass
class Suite:
    rows: List[Result] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "", skipped: bool = False):
        self.rows.append(Result(name, ok, detail, skipped))
        tag = "SKIP" if skipped else ("PASS" if ok else "FAIL")
        extra = f"  {detail}" if detail else ""
        print(f"  [{tag}] {name}{extra}")

    def check(self, name: str, cond: bool, detail: str = ""):
        self.add(name, bool(cond), detail)

    def skip(self, name: str, reason: str):
        self.add(name, True, reason, skipped=True)


def _short(text: Any, n: int = 80) -> str:
    s = "" if text is None else str(text).replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def _http_json(method: str, url: str, body: Optional[dict] = None,
               token: str = "", timeout: float = 20.0) -> Tuple[int, dict]:
    data = None
    headers: Dict[str, str] = {}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    if token:
        headers["X-Robot-Token"] = token
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                payload = {"raw": raw}
            return resp.status, payload
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"raw": raw}
        return e.code, payload
    except urllib.error.URLError as e:
        raise ConnectionError(str(e.reason or e)) from e


def test_faiss(persona_id: str, suite: Suite) -> None:
    spec = CASES[persona_id]
    index_dir = ROOT / "index" / persona_id
    missing = [n for n in INDEX_FILES if not (index_dir / n).is_file()]
    suite.check(
        f"{persona_id} 索引四件套",
        not missing,
        "缺 " + ",".join(missing) if missing else str(index_dir),
    )
    if missing:
        return

    try:
        import faiss
        import numpy as np
    except ImportError as e:
        suite.skip(f"{persona_id} 加载 FAISS", f"{e}（请用 digital_human 环境）")
        return

    try:
        idx = faiss.read_index(str(index_dir / "faiss.index"))
    except Exception as e:
        suite.add(f"{persona_id} 读取 faiss.index", False, f"{type(e).__name__}: {e}")
        return
    suite.check(f"{persona_id} FAISS 可打开", True, f"ntotal={idx.ntotal} dim={idx.d}")
    suite.check(f"{persona_id} 向量数 > 0", idx.ntotal > 0, f"ntotal={idx.ntotal}")

    items = json.loads((index_dir / "items.json").read_text(encoding="utf-8"))
    fmap = json.loads((index_dir / "faiss_map.json").read_text(encoding="utf-8"))
    suite.check(f"{persona_id} items 非空", bool(items), f"n={len(items)}")
    suite.check(
        f"{persona_id} faiss_map 与 ntotal 对齐",
        isinstance(fmap, list) and len(fmap) == idx.ntotal,
        f"map={len(fmap)} ntotal={idx.ntotal}",
    )
    suite.check(
        f"{persona_id} faiss_map 下标不越界",
        isinstance(fmap, list) and fmap and max(fmap) < len(items) and min(fmap) >= 0,
        f"items={len(items)} min={min(fmap) if fmap else None} max={max(fmap) if fmap else None}",
    )

    bad_cat = [i for i in items if i.get("category") != spec["category"]]
    suite.check(
        f"{persona_id} 条目类目={spec['category']}",
        not bad_cat,
        f"越界 {len(bad_cat)}/{len(items)}" if bad_cat else f"n={len(items)}",
    )

    try:
        with open(index_dir / "bm25.pkl", "rb") as f:
            packed = pickle.load(f)
        docs = packed.get("docs") if isinstance(packed, dict) else packed
        suite.check(
            f"{persona_id} BM25 docs 与 items 等长",
            isinstance(docs, list) and len(docs) == len(items),
            f"docs={len(docs) if isinstance(docs, list) else type(docs).__name__} items={len(items)}",
        )
    except Exception as e:
        suite.add(f"{persona_id} 读取 bm25.pkl", False, f"{type(e).__name__}: {e}")

    if idx.ntotal <= 0:
        return
    try:
        try:
            vec = np.asarray(idx.reconstruct(0), dtype="float32").reshape(1, -1)
        except Exception:
            vec = np.empty((idx.d,), dtype="float32")
            idx.reconstruct(0, vec)
            vec = vec.reshape(1, -1)
        _d, I = idx.search(vec, min(3, idx.ntotal))
        top = int(I[0][0])
        suite.check(
            f"{persona_id} 自检索 top1=0",
            top == 0,
            f"I={I[0].tolist()}",
        )
    except Exception as e:
        suite.add(f"{persona_id} FAISS reconstruct/search", False, f"{type(e).__name__}: {e}")


def wait_health(base: str, seconds: float) -> Tuple[int, dict]:
    deadline = time.time() + max(0.0, seconds)
    last_err = "未尝试"
    while True:
        try:
            return _http_json("GET", f"{base.rstrip('/')}/health", timeout=5.0)
        except Exception as e:
            last_err = str(e)
            if time.time() >= deadline:
                raise ConnectionError(last_err) from e
            time.sleep(2.0)


def test_http(persona_ids: Sequence[str], base: str, token: str, suite: Suite) -> None:
    base = base.rstrip("/")
    try:
        code, health = _http_json("GET", f"{base}/health")
    except Exception as e:
        suite.add("GET /health 可连", False, str(e))
        return
    suite.check("GET /health", code == 200, f"HTTP {code}")
    if code != 200:
        return
    suite.check("health.status=healthy", health.get("status") == "healthy", _short(health.get("status")))
    suite.check("health.chunks>0", int(health.get("chunks") or 0) > 0, f"chunks={health.get('chunks')}")
    suite.check("health.vectors>0", int(health.get("vectors") or 0) > 0, f"vectors={health.get('vectors')}")
    listed = [p.get("id") for p in (health.get("personas") or [])]
    for pid in persona_ids:
        suite.check(f"health.personas 含 {pid}", pid in listed, f"personas={listed}")

    code, listing = _http_json("GET", f"{base}/api/persona/list")
    suite.check("GET /api/persona/list", code == 200, f"HTTP {code}")
    ids = [x.get("id") for x in (listing.get("personas") or [])] if code == 200 else []
    snaps = {x.get("id"): x for x in (listing.get("personas") or [])} if code == 200 else {}

    for pid in persona_ids:
        spec = CASES[pid]
        suite.check(f"list 含 {pid}", pid in ids, f"personas={ids}")
        snap = snaps.get(pid) or {}
        if snap:
            suite.check(
                f"{pid} knowledge_ready",
                bool(snap.get("knowledge_ready")),
                _short(snap),
            )

        code, body = _http_json("GET", f"{base}/admin/kb/{pid}/status", token=token)
        suite.check(f"GET /admin/kb/{pid}/status", code == 200, f"HTTP {code} {_short(body)}")
        if code == 200:
            suite.check(
                f"{pid} kb live 就绪",
                bool(body.get("live_ready") or body.get("state") == "live"),
                f"state={body.get('state')} live_ready={body.get('live_ready')} items={body.get('live_items')}",
            )

        code, body = _http_json(
            "POST", f"{base}/api/{pid}/search",
            {"query": spec["query"], "top_k": 3},
            timeout=30.0,
        )
        suite.check(f"{pid} 本展 search HTTP", code == 200, f"HTTP {code} {_short(body)}")
        if code == 200:
            refused = bool(body.get("refused"))
            results = body.get("results") or []
            suite.check(
                f"{pid} 本展 search 有结果",
                (not refused) and bool(results),
                f"refused={refused} n={len(results)} persona={body.get('persona_id')}",
            )
            suite.check(
                f"{pid} search 回写 persona_id",
                body.get("persona_id") == pid,
                repr(body.get("persona_id")),
            )
            if results:
                blob = json.dumps(results, ensure_ascii=False)
                leaked = [w for w in spec["must_not"] if w in blob]
                suite.check(f"{pid} search 未串对方", not leaked, ",".join(leaked) if leaked else "OK")
                hit_kw = [k for k in spec["in_keywords"] if k in blob]
                suite.check(
                    f"{pid} search 命中本展关键词",
                    bool(hit_kw),
                    "命中 " + "/".join(hit_kw) if hit_kw else _short(blob, 60),
                )

        code, body = _http_json(
            "POST", f"{base}/api/{pid}/search",
            {"query": spec["cross"], "top_k": 2},
        )
        suite.check(f"{pid} 串台 search HTTP", code == 200, f"HTTP {code}")
        if code == 200:
            suite.check(
                f"{pid} 串台 search 拒答",
                bool(body.get("refused")) and not (body.get("results") or []),
                f"refused={body.get('refused')} reason={body.get('reason')}",
            )

        code, body = _http_json(
            "POST",
            f"{base}/api/{pid}/robot/ask",
            {
                "query": spec["query"],
                "robot_id": f"smoke-{pid}",
                "include_audio": False,
                "wait": False,
                "top_k": 2,
            },
            token=token,
            timeout=8.0,
        )
        suite.check(
            f"{pid} robot/ask 立刻受理",
            code in (200, 202, 409),
            f"HTTP {code} {_short(body)}",
        )
        if code in (200, 202) and "persona_id" in body:
            suite.check(f"{pid} ask.persona_id", body.get("persona_id") == pid, repr(body.get("persona_id")))

    code, body = _http_json("POST", f"{base}/api/not-a-persona/search", {"query": "你好"})
    suite.check("未知人设 search 404/422", code in (404, 422), f"HTTP {code}")

    code, body = _http_json("POST", f"{base}/search", {"query": CASES["monet"]["query"], "top_k": 1})
    suite.check("旧路由 POST /search", code == 200, f"HTTP {code} persona={body.get('persona_id')}")


async def _ws_tts_probe(host: str, port: int, persona_id: str, timeout: float) -> dict:
    import websockets

    url = f"ws://{host}:{port}/ws/{persona_id}/tts"
    ready = None
    pong = None
    async with websockets.connect(url, open_timeout=timeout, close_timeout=3) as ws:
        raw = await _ws_recv(ws, timeout)
        ready = json.loads(raw) if isinstance(raw, str) else raw
        await ws.send(json.dumps({"type": "ping"}))
        raw = await _ws_recv(ws, timeout)
        pong = json.loads(raw) if isinstance(raw, str) else raw
    return {"ready": ready, "pong": pong}


async def _ws_chat_probe(host: str, port: int, persona_id: str, timeout: float) -> dict:
    import websockets

    url = f"ws://{host}:{port}/ws/{persona_id}/chat"
    async with websockets.connect(url, open_timeout=timeout, close_timeout=3) as ws:
        await ws.send(json.dumps({"user_id": f"smoke-{persona_id}"}))
        raw = await _ws_recv(ws, timeout)
        init = json.loads(raw) if isinstance(raw, str) else raw
    return {"init": init}


async def _ws_unknown_probe(host: str, port: int, timeout: float) -> dict:
    import websockets

    url = f"ws://{host}:{port}/ws/not-a-persona/tts"
    code = None
    http_status = None
    try:
        async with websockets.connect(url, open_timeout=timeout, close_timeout=3) as ws:
            try:
                await _ws_recv(ws, min(timeout, 5.0))
            except Exception as e:
                rcvd = getattr(e, "rcvd", None)
                code = getattr(e, "code", None) or getattr(rcvd, "code", None)
    except Exception as e:
        rcvd = getattr(e, "rcvd", None)
        response = getattr(e, "response", None)
        code = getattr(e, "code", None) or getattr(rcvd, "code", None)
        http_status = getattr(e, "status_code", None) or getattr(response, "status_code", None)
        if code is None and http_status is None:
            return {"code": None, "http_status": None, "error": f"{type(e).__name__}: {e}"}
        return {"code": code, "http_status": http_status, "error": f"{type(e).__name__}: {e}"}
    return {"code": code, "http_status": http_status}


async def _ws_recv(ws, timeout: float):
    import asyncio
    return await asyncio.wait_for(ws.recv(), timeout=timeout)


def test_ws(persona_ids: Sequence[str], base: str, timeout: float, suite: Suite) -> None:
    try:
        import websockets  # noqa: F401
    except ImportError as e:
        suite.skip("websockets 库", f"{e}（pip install websockets）")
        return

    from urllib.parse import urlparse
    import asyncio

    u = urlparse(base if "://" in base else f"http://{base}")
    host = u.hostname or "127.0.0.1"
    port = int(u.port or 80)

    async def _run():
        for pid in persona_ids:
            spec = CASES[pid]
            try:
                tts = await _ws_tts_probe(host, port, pid, timeout)
            except Exception as e:
                suite.add(f"{pid} WS /tts 可连", False, f"{type(e).__name__}: {e}")
                continue
            ready = tts.get("ready") or {}
            pong = tts.get("pong") or {}
            suite.check(
                f"{pid} TTS ready",
                ready.get("type") == "ready" and ready.get("persona_id") == pid,
                _short(ready),
            )
            suite.check(
                f"{pid} TTS display_name",
                ready.get("display_name") == spec["display_name"],
                repr(ready.get("display_name")),
            )
            suite.check(f"{pid} TTS ping/pong", pong.get("type") == "pong", _short(pong))

            try:
                chat = await _ws_chat_probe(host, port, pid, timeout)
            except Exception as e:
                suite.add(f"{pid} WS /chat 可连", False, f"{type(e).__name__}: {e}")
                continue
            init = chat.get("init") or {}
            suite.check(
                f"{pid} chat init",
                init.get("type") == "init" and init.get("persona_id") == pid,
                _short(init),
            )
            suite.check(
                f"{pid} chat session_id 非空",
                bool(init.get("session_id")),
                repr(init.get("session_id")),
            )

        try:
            unknown = await _ws_unknown_probe(host, port, timeout)
            code = unknown.get("code")
            http_status = unknown.get("http_status")
            # 未 accept 时 Starlette 可能直接 HTTP 403；accept 后再关则是 1008
            suite.check(
                "未知人设 WS 被拒绝",
                code == 1008 or http_status in (403, 404),
                f"code={code} http={http_status} {_short(unknown.get('error'))}",
            )
        except Exception as e:
            suite.add("未知人设 WS 被拒绝", False, f"{type(e).__name__}: {e}")

    asyncio.run(_run())


def main() -> int:
    parser = argparse.ArgumentParser(description="莫奈/妇好 API + WS + FAISS 冒烟")
    parser.add_argument("--persona", choices=("monet", "fuhao", "kumarajiva", "all"), default="all")
    parser.add_argument("--offline", action="store_true", help="只测本地 FAISS，不连服务")
    parser.add_argument("--base", default="http://10.0.7.232:8848")
    parser.add_argument("--token", default="")
    parser.add_argument("--wait", type=float, default=0.0, help="等待 /health 就绪的秒数")
    parser.add_argument("--ws-timeout", type=float, default=8.0)
    args = parser.parse_args()

    persona_ids = ["monet", "fuhao", "kumarajiva"] if args.persona == "all" else [args.persona]
    suite = Suite()

    print("=" * 72)
    print("离线 FAISS")
    print("=" * 72)
    for pid in persona_ids:
        print(f"\n--- {pid} ---")
        test_faiss(pid, suite)

    if not args.offline:
        print("\n" + "=" * 72)
        print(f"在线 HTTP / WS  {args.base}")
        print("=" * 72)
        if args.wait > 0:
            print(f"  等待服务就绪，最多 {args.wait:.0f}s …")
            try:
                code, health = wait_health(args.base, args.wait)
                print(f"  /health HTTP {code} status={health.get('status')} "
                      f"chunks={health.get('chunks')} vectors={health.get('vectors')}")
            except Exception as e:
                suite.add("等待 /health", False, str(e))
                return _summary(suite)

        test_http(persona_ids, args.base, args.token, suite)
        print("\n--- WebSocket ---")
        test_ws(persona_ids, args.base, args.ws_timeout, suite)

    return _summary(suite)


def _summary(suite: Suite) -> int:
    passed = sum(1 for r in suite.rows if r.ok and not r.skipped)
    failed = [r for r in suite.rows if not r.ok]
    skipped = sum(1 for r in suite.rows if r.skipped)
    print("\n" + "=" * 72)
    print(f"合计 {len(suite.rows)} 项  通过 {passed}  失败 {len(failed)}  跳过 {skipped}")
    if failed:
        print("失败项:")
        for r in failed:
            print(f"  - {r.name}: {r.detail}")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
