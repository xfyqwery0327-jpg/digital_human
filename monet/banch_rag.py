# -*- coding: utf-8 -*-
"""RAG 服务延迟压测 + 端到端 /ask 首字延迟"""
import time
import statistics
import requests

RAG_URL = 'http://127.0.0.1:8848/search'
ASK_URL = 'http://127.0.0.1:5000/ask'

QUESTIONS = [
    '妇好是谁？',
    '你打过哪些仗？',
    '你墓里那个大鼎是做什么用的？',
    '商朝的玉从哪里来？',
    '武丁对你怎么样？',
]

def bench_rag():
    print('=' * 60)
    print('1. RAG 检索延迟（top_k=2, 每问 3 次取中位）')
    all_meds = []
    for q in QUESTIONS:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            r = requests.post(RAG_URL, json={'query': q, 'top_k': 1}, timeout=10)
            r.raise_for_status()
            times.append(time.perf_counter() - t0)
        m = statistics.median(times)
        all_meds.append(m)
        hit = r.json().get('results', [])
        preview = hit[0]['text'][:30] if hit else '(空)'
        print('   %.3fs | %s → %s...' % (m, q, preview))
    print('   RAG 中位: %.3fs | 最差: %.3fs' % (
        statistics.median(all_meds), max(all_meds)))
    return statistics.median(all_meds)

def bench_e2e():
    print('')
    print('2. 端到端 /ask 首字延迟（含 RAG + LLM prefill）')
    firsts = []
    for q in QUESTIONS:
        t0 = time.perf_counter()
        r = requests.post(ASK_URL, json={'question': q}, stream=True, timeout=60)
        r.raise_for_status()
        first = None
        text = ''
        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                if first is None:
                    first = time.perf_counter() - t0
                text += chunk
                if len(text) > 30:
                    break
        r.close()
        firsts.append(first or 0)
        print('   首字 %.3fs | %s → %s' % (first or 0, q, text[:25]))
    print('   端到端首字中位: %.3fs   ← 游客真实体感' % statistics.median(firsts))
    return statistics.median(firsts)

if __name__ == '__main__':
    rag_med = bench_rag()
    e2e_med = bench_e2e()
    print('')
    print('3. 拆帐')
    print('   RAG 检索        %.3fs' % rag_med)
    print('   LLM TTFT(上线)  ~1.27s (bench_llm 实测)')
    print('   端到端首字      %.3fs' % e2e_med)
    print('   差值(其他开销)  %.3fs' % (e2e_med - rag_med - 1.27))
    print('=' * 60)