#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 bench_spoken_*.json 重新聚合：首句音(TTFA) + 句间间隔(gap)，并标注 409/空轮。"""
import glob, json, statistics as st

def med(xs):
    return round(st.median(xs), 2) if xs else None

def load():
    fs = sorted(glob.glob("logs/bench_spoken_*.json"))
    if not fs:
        print("NO FILE"); return None
    print("LOAD", fs[-1])
    return json.load(open(fs[-1], encoding="utf-8"))

def main():
    d = load()
    if not d: return
    recs = d.get("records", [])
    out = []
    for persona in ["monet", "fuhao", "kumarajiva"]:
        mode3 = [r for r in recs if r.get("persona") == persona and r.get("kind") == "mode3"]
        warm = [r for r in recs if r.get("persona") == persona and r.get("kind") == "warmup"]
        ok = [r for r in mode3 if "error" not in r and r.get("sentences")]
        err = [r for r in mode3 if "error" in r]
        empty = [r for r in mode3 if "error" not in r and not r.get("sentences")]
        # TTFA = 第一句 t_first_chunk（相对 t_ask 的秒数）
        ttfa = [round(r["sentences"][0]["t_first_chunk"], 2) for r in ok]
        nsent = [r.get("n_sentences") for r in ok]
        # 句间间隔：合并所有成功多句轮
        all_gaps = []
        large = []
        for r in ok:
            for g in r.get("gaps", []):
                all_gaps.append(g)
                if g >= 3.0:
                    large.append(g)
        # 多句轮里的大间隔位置（每轮 gaps 列表）
        p = {
            "persona": persona,
            "warmup_n": len(warm),
            "measured_n": len(mode3),
            "ok_n": len(ok),
            "err_409_n": sum(1 for r in err if "409" in str(r.get("error",""))),
            "err_other_n": sum(1 for r in err if "409" not in str(r.get("error",""))),
            "empty_n": len(empty),
            "ttfa": {
                "values": ttfa,
                "min": round(min(ttfa),2) if ttfa else None,
                "median": med(ttfa),
                "mean": round(st.mean(ttfa),2) if ttfa else None,
                "max": round(max(ttfa),2) if ttfa else None,
            },
            "n_sentences_per_ok_round": nsent,
            "gap_all": {
                "n": len(all_gaps),
                "zeros": sum(1 for g in all_gaps if g < 0.05),
                "median": med(all_gaps),
                "max": round(max(all_gaps),2) if all_gaps else None,
                "large_count": len(large),
                "large_values": [round(x,2) for x in large],
            },
        }
        out.append(p)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
