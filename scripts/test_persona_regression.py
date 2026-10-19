# -*- coding: utf-8 -*-
"""莫奈 / 妇好 双人设回归。

默认只跑离线：人设 YAML、索引四件套、参考音、类目锁定、串台拒答、口播稿不越界。
讲解员已启动时再加 --live（HTTP 检索/健康）或 --ask（robot-ask + WS 时序）。

用法（项目根）:

    python scripts/test_persona_regression.py
    python scripts/test_persona_regression.py --persona monet
    python scripts/test_persona_regression.py --persona fuhao
    python scripts/test_persona_regression.py --live
    python scripts/test_persona_regression.py --live --ask
    python scripts/test_persona_regression.py --live --base http://127.0.0.1:8848
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
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
        "other_category": "exhibition_fuhao",
        "in_domain": [
            "介绍一下莫奈",
            "睡莲是在哪里画的",
            "日出印象这幅画",
        ],
        "in_keywords": ("莫奈", "睡莲", "印象", "吉维尼", "日出"),
        "cross": [
            "妇好是谁",
            "甲骨文是什么",
            "殷墟在哪",
            "武丁的王后",
            "鸠摩罗什是谁",
        ],
        "other_exhibition": [
            "别的展览还有什么",
            "上一届展讲什么",
            "卢浮宫还有什么展",
        ],
        "index_must_not": ("妇好", "武丁", "殷墟", "甲骨文", "鸠摩罗什"),
        "script_must_not": ("妇好", "武丁", "商代", "甲骨", "殷墟"),
        "prompt_must_not": ("妇好", "甲骨", "殷墟", "武丁"),
        "voice_prefer": ("yinpin_16k.wav", "cross_lingual_prompt.wav", "yinpin.mp3"),
    },
    "fuhao": {
        "display_name": "妇好",
        "category": "exhibition_fuhao",
        "other_category": "exhibition_monet",
        "in_domain": [
            "妇好介绍下自己",
            "墓里出土了什么",
            "甲骨文是什么",
        ],
        "in_keywords": ("妇好", "甲骨", "商", "武丁", "殷墟", "青铜"),
        "cross": [
            "莫奈是谁",
            "睡莲是什么画",
            "日出印象",
            "吉维尼花园",
            "鸠摩罗什是谁",
        ],
        "other_exhibition": [
            "别的展览还有什么",
            "上一届展讲什么",
            "卢浮宫还有什么展",
        ],
        "index_must_not": ("莫奈", "睡莲", "印象派", "吉维尼", "日出·印象", "鸠摩罗什"),
        "script_must_not": ("莫奈", "睡莲", "印象派", "吉维尼", "克洛德"),
        "prompt_must_not": ("莫奈", "睡莲", "印象派", "吉维尼"),
        "voice_prefer": ("jimeng-2026-09-11-5129.mp3", "yinpin_16k.wav", "yinpin.mp3"),
    },
    "kumarajiva": {
        "display_name": "鸠摩罗什",
        "category": "exhibition_kumarajiva",
        "other_category": "exhibition_monet",
        "in_domain": [
            "介绍一下鸠摩罗什",
            "金刚经讲什么",
            "舍利盒上画了什么",
        ],
        "in_keywords": ("鸠摩罗什", "龟兹", "金刚", "法华", "译", "长安", "舍利"),
        "cross": [
            "莫奈是谁",
            "妇好是谁",
            "睡莲是什么画",
            "甲骨文是什么",
        ],
        "other_exhibition": [
            "别的展览还有什么",
            "上一届展讲什么",
            "卢浮宫还有什么展",
        ],
        "index_must_not": ("莫奈", "睡莲", "印象派", "妇好", "殷墟", "武丁"),
        "script_must_not": ("莫奈", "睡莲", "印象派", "妇好", "武丁", "克洛德"),
        "prompt_must_not": ("莫奈", "睡莲", "印象派", "妇好", "甲骨", "殷墟"),
        "voice_prefer": ("prompt.wav",),
        "voice_required": False,
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


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit(
            f"需要 PyYAML 才能读人设: {e}\n"
            "可先: pip install pyyaml"
        ) from e
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _as_list(raw) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(x) for x in raw if str(x).strip()]


def _compile(patterns: Sequence[str]) -> List[re.Pattern]:
    out = []
    for pat in patterns:
        try:
            out.append(re.compile(pat, re.IGNORECASE))
        except re.error:
            out.append(re.compile(re.escape(pat), re.IGNORECASE))
    return out


def _is_cross(query: str, data: dict) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    for term in _as_list(data.get("forbidden_terms")):
        if term and term in q:
            return True
    for rx in _compile(_as_list(data.get("cross_persona_patterns"))):
        if rx.search(q):
            return True
    return False


def _is_other_exhibition(query: str, data: dict) -> bool:
    if _is_cross(query, data):
        return True
    patterns = _as_list(data.get("other_exhibition_patterns"))
    if not patterns:
        return False
    for rx in _compile(patterns):
        if rx.search(query or ""):
            return True
    return False


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


def _short(text: Any, n: int = 60) -> str:
    s = "" if text is None else str(text).replace("\n", " ")
    return s if len(s) <= n else s[: n] + "…"


def _join_scripts(data: dict) -> str:
    parts: List[str] = []
    parts.extend(_as_list(data.get("hello_texts")))
    parts.extend(_as_list(data.get("bye_texts")))
    chat = data.get("chitchat") or {}
    if isinstance(chat, dict):
        parts.extend(str(v) for v in chat.values())
    hooks = data.get("hooks") or {}
    if isinstance(hooks, dict):
        for v in hooks.values():
            parts.extend(_as_list(v))
    parts.append(str(data.get("warmup_text") or ""))
    parts.append(str(data.get("refuse_text") or ""))
    return "\n".join(parts)


def run_offline(persona_id: str, suite: Suite):
    spec = CASES[persona_id]
    yaml_path = ROOT / "personas" / f"{persona_id}.yaml"
    suite.check(f"{persona_id} YAML 存在", yaml_path.is_file(), str(yaml_path))
    if not yaml_path.is_file():
        return
    data = _load_yaml(yaml_path)
    suite.check(
        f"{persona_id} id/显示名",
        data.get("id") == persona_id and data.get("display_name") == spec["display_name"],
        f"id={data.get('id')!r} name={data.get('display_name')!r}",
    )
    cats = _as_list(data.get("categories"))
    suite.check(
        f"{persona_id} 知识类目锁定",
        cats == [spec["category"]],
        f"categories={cats}",
    )
    suite.check(
        f"{persona_id} 未混入对方类目",
        spec["other_category"] not in cats,
        f"other={spec['other_category']}",
    )

    index_dir = ROOT / "index" / persona_id
    missing = [n for n in INDEX_FILES if not (index_dir / n).is_file()]
    suite.check(f"{persona_id} 索引四件套", not missing, "缺 " + ",".join(missing) if missing else str(index_dir))

    items: List[dict] = []
    items_path = index_dir / "items.json"
    if items_path.is_file():
        items = json.loads(items_path.read_text(encoding="utf-8"))
        suite.check(f"{persona_id} 索引非空", bool(items), f"n={len(items)}")
        bad_cat = [i for i in items if i.get("category") != spec["category"]]
        suite.check(
            f"{persona_id} 条目类目全是 {spec['category']}",
            not bad_cat,
            f"越界 {len(bad_cat)}/{len(items)}" if bad_cat else f"n={len(items)}",
        )
        blob = "\n".join(str(i.get("text") or "") for i in items)
        leaked = [w for w in spec["index_must_not"] if w in blob]
        suite.check(
            f"{persona_id} 知识库未串入对方关键词",
            not leaked,
            "命中 " + ",".join(leaked) if leaked else "干净",
        )
        for q in spec["in_domain"]:
            keys = [k for k in spec["in_keywords"] if k in q] or list(spec["in_keywords"][:2])
            found = any(any(k in (i.get("text") or "") for k in keys) for i in items)
            suite.check(f"{persona_id} 本展可检索「{q}」", found, "关键词 " + "/".join(keys))

        map_path = index_dir / "faiss_map.json"
        if map_path.is_file():
            fmap = json.loads(map_path.read_text(encoding="utf-8"))
            suite.check(
                f"{persona_id} faiss_map 长度合法",
                isinstance(fmap, list) and len(fmap) >= 1 and max(fmap) < len(items),
                f"map={len(fmap)} items={len(items)} max={max(fmap) if fmap else None}",
            )

    vdir = ROOT / "voice" / persona_id
    voices = []
    if vdir.is_dir():
        voices = [p.name for p in vdir.iterdir() if p.suffix.lower() in {".wav", ".mp3"}]
    prefer_hit = next((n for n in spec["voice_prefer"] if (vdir / n).is_file()), None)
    yaml_wav = str((data.get("tts") or {}).get("prompt_wav") or "")
    wav_ok = bool(prefer_hit) or any((ROOT / yaml_wav).is_file() for _ in [0] if yaml_wav)
    if yaml_wav:
        p = Path(yaml_wav)
        if not p.is_absolute():
            p = ROOT / p
        wav_ok = p.is_file() or bool(prefer_hit)
    voice_detail = prefer_hit or yaml_wav or ("目录空" if not voices else ",".join(voices[:4]))
    if spec.get("voice_required", True) or wav_ok:
        suite.check(f"{persona_id} 参考音就绪", wav_ok, voice_detail)
    else:
        suite.skip(f"{persona_id} 参考音就绪", "资料未附参考音，放入 voice/%s/ 后即可克隆" % persona_id)
    # 参考音必须落在本人 voice 目录，禁止误用对方目录
    if yaml_wav:
        p = Path(yaml_wav)
        if not p.is_absolute():
            p = ROOT / p
        other = ROOT / "voice" / ("fuhao" if persona_id == "monet" else "monet")
        try:
            mixed = other.resolve() in p.resolve().parents or p.resolve().parent == other.resolve()
        except OSError:
            mixed = False
        suite.check(f"{persona_id} 参考音未串用对方目录", not mixed, str(p))

    prompt = str(data.get("system_prompt") or "")
    # 系统提示里允许「禁止讲对方」；只在非禁令行里算串台
    accidental = []
    for w in spec["prompt_must_not"]:
        for line in prompt.splitlines():
            if w in line and not re.search(r"禁止|不要|别|不是|勿|并非", line):
                accidental.append(f"{w}@{line.strip()[:40]!r}")
    suite.check(
        f"{persona_id} 系统提示未把对方当自己",
        not accidental,
        "；".join(accidental[:3]) if accidental else "OK",
    )

    scripts = _join_scripts(data)
    leaked_s = [w for w in spec["script_must_not"] if w in scripts]
    suite.check(
        f"{persona_id} 欢迎/再见/闲聊未越界",
        not leaked_s,
        "命中 " + ",".join(leaked_s) if leaked_s else "OK",
    )
    refuse = str(data.get("refuse_text") or "")
    suite.check(f"{persona_id} 拒答文案非空", bool(refuse.strip()), _short(refuse, 40))

    for q in spec["in_domain"]:
        suite.check(
            f"{persona_id} 本展问题不误拒「{q}」",
            not _is_other_exhibition(q, data),
        )
    for q in spec["cross"]:
        suite.check(
            f"{persona_id} 串台必拒「{q}」",
            _is_cross(q, data) or _is_other_exhibition(q, data),
        )
    for q in spec["other_exhibition"]:
        suite.check(
            f"{persona_id} 他展必拒「{q}」",
            _is_other_exhibition(q, data),
        )


def run_scope_module(persona_ids: Sequence[str], suite: Suite):
    """用真实 rag.scope 再验一遍，避免 YAML 规则与代码分叉。"""
    sys.path.insert(0, str(ROOT / "server9"))
    try:
        import fastapi  # noqa: F401
    except ImportError:
        import types
        fastapi = types.ModuleType("fastapi")

        class HTTPException(Exception):
            def __init__(self, status_code=400, detail=""):
                self.status_code = status_code
                self.detail = detail

        class WebSocketException(Exception):
            def __init__(self, code=1008, reason=""):
                self.code = code
                self.reason = reason

        fastapi.HTTPException = HTTPException
        fastapi.WebSocketException = WebSocketException
        fastapi.status = types.SimpleNamespace(WS_1008_POLICY_VIOLATION=1008)
        fastapi.APIRouter = object
        fastapi.Depends = lambda *a, **k: None
        fastapi.Header = lambda *a, **k: None
        fastapi.Query = lambda *a, **k: None
        sys.modules["fastapi"] = fastapi
        sys.modules["fastapi.responses"] = types.ModuleType("fastapi.responses")
        sys.modules["fastapi.responses"].JSONResponse = object
        sys.modules["fastapi.staticfiles"] = types.ModuleType("fastapi.staticfiles")
        sys.modules["fastapi.staticfiles"].StaticFiles = object
    try:
        from persona.registry import registry
        from rag.scope import is_cross_persona_query, is_other_exhibition_query
        registry.load()
    except Exception as e:
        suite.skip("导入 rag.scope / registry", f"{type(e).__name__}: {e}")
        return

    loaded = {p.id for p in registry.all()}
    for pid in persona_ids:
        spec = CASES[pid]
        p = registry.get(pid)
        suite.check(f"{pid} registry 已加载", p is not None and pid in loaded, f"loaded={sorted(loaded)}")
        if p is None:
            continue
        suite.check(f"{pid} knowledge_ready()", p.knowledge_ready(), str(p.index_path))
        need_voice = spec.get("voice_required", True)
        if need_voice or p.voice_ready():
            suite.check(f"{pid} voice_ready()", p.voice_ready(), p.tts.prompt_wav)
        else:
            suite.skip(f"{pid} voice_ready()", "资料未附参考音")
        snap = p.snapshot()
        ready_ok = bool(snap.get("knowledge_ready")) and (
            bool(snap.get("voice_ready")) or not need_voice)
        suite.check(f"{pid} snapshot 含就绪标记", ready_ok, str(snap))
        for q in spec["in_domain"]:
            suite.check(
                f"{pid} scope 本展放行「{q}」",
                not is_other_exhibition_query(q, p),
            )
        for q in spec["cross"]:
            suite.check(
                f"{pid} scope 串台拦截「{q}」",
                is_cross_persona_query(q, p),
            )
        other = registry.get("fuhao" if pid == "monet" else "monet")
        if other:
            wav_a = Path(p.tts.prompt_wav).resolve() if p.tts.prompt_wav else None
            wav_b = Path(other.tts.prompt_wav).resolve() if other.tts.prompt_wav else None
            same = bool(wav_a and wav_b and wav_a == wav_b and wav_a.exists())
            suite.check(f"{pid} 参考音与对方不是同一文件", not same, f"{wav_a} vs {wav_b}")


def run_live(persona_ids: Sequence[str], base: str, token: str, suite: Suite):
    base = base.rstrip("/")
    try:
        code, health = _http_json("GET", f"{base}/health")
    except Exception as e:
        suite.add("GET /health 可连", False, str(e))
        return
    suite.check("GET /health", code == 200, f"HTTP {code}")
    if code != 200:
        return
    for field in ("status", "chunks", "vectors", "tts", "robot"):
        suite.check(f"health 保留字段 {field}", field in health, f"keys={list(health)[:12]}")
    suite.check("health.status=healthy", health.get("status") == "healthy")

    code, listing = _http_json("GET", f"{base}/api/persona/list")
    suite.check("GET /api/persona/list", code == 200, f"HTTP {code}")
    ids = [x.get("id") for x in (listing.get("personas") or [])] if code == 200 else []
    for pid in persona_ids:
        spec = CASES[pid]
        suite.check(f"list 含 {pid}", pid in ids, f"personas={ids}")
        code, body = _http_json("POST", f"{base}/api/{pid}/search", {"query": spec["in_domain"][0], "top_k": 2})
        suite.check(
            f"{pid} 本展 search HTTP",
            code == 200,
            f"HTTP {code} {_short(body)}",
        )
        if code == 200:
            refused = bool(body.get("refused"))
            results = body.get("results") or []
            suite.check(
                f"{pid} 本展 search 有结果且不拒",
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
                leaked = [w for w in spec["index_must_not"] if w in blob]
                suite.check(f"{pid} search 命中未串对方", not leaked, ",".join(leaked) if leaked else "OK")

        code, body = _http_json("POST", f"{base}/api/{pid}/search", {"query": spec["cross"][0], "top_k": 2})
        suite.check(f"{pid} 串台 search HTTP", code == 200, f"HTTP {code}")
        if code == 200:
            suite.check(
                f"{pid} 串台 search 拒答",
                bool(body.get("refused")) and not (body.get("results") or []),
                f"refused={body.get('refused')} reason={body.get('reason')} n={len(body.get('results') or [])}",
            )

        code, body = _http_json(
            "POST",
            f"{base}/api/{pid}/robot/ask",
            {
                "query": spec["in_domain"][0],
                "robot_id": f"regress-{pid}",
                "include_audio": False,
                "wait": False,
                "top_k": 2,
            },
            token=token,
            timeout=8.0,
        )
        # 未接 WS 时仍应立刻 202；忙线可能 409
        suite.check(
            f"{pid} robot/ask 立刻受理",
            code in (200, 202, 409),
            f"HTTP {code} {_short(body)}",
        )
        if code in (200, 202) and "persona_id" in body:
            suite.check(
                f"{pid} ask.persona_id=={pid}",
                body.get("persona_id") == pid,
                repr(body.get("persona_id")),
            )

    code, body = _http_json("POST", f"{base}/api/not-a-persona/search", {"query": "你好"})
    suite.check("未知人设 search 应 404/422", code in (404, 422), f"HTTP {code}")

    code, body = _http_json("POST", f"{base}/search", {"query": CASES["monet"]["in_domain"][0], "top_k": 1})
    suite.check("旧路由 POST /search 仍可用", code == 200, f"HTTP {code} persona={body.get('persona_id')}")


def _split_base(base: str) -> Tuple[str, str]:
    from urllib.parse import urlparse
    u = urlparse(base if "://" in base else f"http://{base}")
    host = u.hostname or "127.0.0.1"
    port = str(u.port or (443 if u.scheme == "https" else 80))
    return host, port


def run_ask(persona_ids: Sequence[str], base: str, token: str, timeout: float, suite: Suite):
    script = ROOT / "scripts" / "test_robot_ask.py"
    if not script.is_file():
        suite.add("scripts/test_robot_ask.py", False, "文件不存在")
        return
    host, port = _split_base(base)
    py = sys.executable
    for pid in persona_ids:
        spec = CASES[pid]
        cmd = [
            py, str(script),
            "--persona-id", pid,
            "--host", host,
            "--port", port,
            "--query", spec["in_domain"][0],
            "--timeout", str(int(timeout)),
        ]
        if token:
            cmd.extend(["--token", token])
        print(f"\n  >> {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, cwd=str(ROOT), timeout=timeout + 30)
        except subprocess.TimeoutExpired:
            suite.add(f"{pid} robot-ask 联调", False, f"子进程超时 {timeout + 30:.0f}s")
            continue
        suite.check(f"{pid} robot-ask 联调退出码", proc.returncode == 0, f"code={proc.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="莫奈/妇好双人设回归")
    parser.add_argument(
        "--persona", choices=("monet", "fuhao", "kumarajiva", "all"), default="all")
    parser.add_argument("--live", action="store_true", help="连讲解员做 HTTP 检索/health/ask 受理")
    parser.add_argument("--ask", action="store_true", help="再跑 robot-ask + WS（需讲解员与 GPU）")
    parser.add_argument("--base", default="http://127.0.0.1:8848")
    parser.add_argument("--token", default="")
    parser.add_argument("--timeout", type=float, default=90.0, help="--ask 单轮等待秒")
    parser.add_argument("--skip-scope", action="store_true", help="不导入 server9 模块")
    args = parser.parse_args()

    persona_ids = ["monet", "fuhao", "kumarajiva"] if args.persona == "all" else [args.persona]
    suite = Suite()

    print("=" * 72)
    print("离线：人设资料 / 索引 / 音色 / 串台隔离")
    print("=" * 72)
    for pid in persona_ids:
        print(f"\n--- {pid} ---")
        run_offline(pid, suite)

    if not args.skip_scope:
        print("\n--- rag.scope / registry ---")
        run_scope_module(persona_ids, suite)

    if args.live:
        print("\n" + "=" * 72)
        print(f"在线 HTTP  {args.base}")
        print("=" * 72)
        run_live(persona_ids, args.base, args.token, suite)

    if args.ask:
        print("\n" + "=" * 72)
        print("在线 robot-ask + WS")
        print("=" * 72)
        run_ask(persona_ids, args.base, args.token, args.timeout, suite)

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
