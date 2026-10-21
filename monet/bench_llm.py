# -*- coding: utf-8 -*-
"""
LLM 服务基准 v3：TTFT / decode / 首句可送 TTS / prefix cache(nonce隔离) / 上线模拟 / 并发有效性
用法: python bench_llm.py
复测前建议重启服务:
  python -m llama_cpp.server --model ... --interrupt_requests False ...
v3 变更:
  1) ThinkFilter 出口 lstrip 前导空白, 空 think 标签残留换行不再进 TTS/字数统计
  2) 首段切分优先强/弱断句, cap 18→24 仅兜底, 减少半句硬切
  3) 新增上线模拟段: top-2 检索片段每轮变化, 只有人设吃缓存, 测真实 TTFT
  4) 末尾输出三点标定(全缓存/上线/冷启)
注意: 保存为 UTF-8; 不要贴进 REPL, 存文件跑
"""
import json
import time
import uuid
import statistics
from concurrent.futures import ThreadPoolExecutor

import requests

# ---------------- 配置 ----------------
HOST     = 'http://127.0.0.1:8080'
CHAT_URL = HOST + '/v1/chat/completions'
MODEL    = 'qwen3'
RUNS     = 5
MAX_TOK  = 256
CONC     = [1, 4]
FIRST_SEG_CAP = 24        # 首段强制切分上限(兜底), v2 的 18 会切出半句
SOFT_MIN      = 8         # 弱断句(逗号)最小字数

PERSONA = (
    '你是妇好，商王武丁的王后，也是商朝第一位领兵出征的女将军。\n'
    '你正在殷墟博物馆为游客讲解。说话风格要求：\n'
    '1. 以第一人称"我"自述，语气亲切、有画面感，像长辈讲往事；\n'
    '2. 每次回答控制在 100 字以内，适合口语播报，不要用书面语的长句；\n'
    '3. 不使用括号、书名号、星号等无法朗读的符号；\n'
    '4. 回答末尾自然地引出一个可以继续追问的方向；\n'
    '5. 遇到史料无记载的问题，坦诚说明，不编造。'
)

RAG_CHUNKS = [
    '妇好墓于1976年在河南安阳小屯村北发现，编号五号墓，是殷墟考古中唯一保存完整、'
    '未被盗掘且能与甲骨文记载相互印证墓主身份的商代王室墓葬。墓室为长方形竖穴，'
    '南北长五点六米，东西宽四米，深七点五米，随葬品共一千九百二十八件。',
    '妇好墓出土青铜器四百六十八件，总重约一千六百公斤，其中司母辛大方鼎两件，'
    '每件高八十厘米、重一百二十八公斤，鼎腹内壁铸有"司母辛"三字铭文，'
    '是妇好去世后子辈为其铸造的祭器。另有偶方彝、三联甑、妇好鸮尊等重器。',
    '甲骨文中关于妇好的记载超过两百条，内容涉及征伐、祭祀、生育与疾病。'
    '卜辞记载武丁命妇好率军征讨土方、巴方与羌方，其中征羌方一役出兵一万三千人，'
    '是商代甲骨文所见规模最大的一次军事行动。',
    '妇好墓出土玉器七百五十五件，玉料多来自新疆和田与辽宁岫岩，'
    '说明商代已存在远距离玉料贸易网络。另出土海贝六千八百八十枚，'
    '产自南海与东海，作为货币使用，反映商王朝的贸易范围。',
    '妇好还担任占卜官，甲骨文中有其主持祭祀的记录。她去世后武丁多次为其举行'
    '冥婚仪式，将她的灵魂分别嫁给先王大乙、大甲与祖乙，祈求祖先庇护，'
    '这种规格在商代女性中极为罕见。',
]

QUESTIONS = [
    '妇好是谁？',
    '你打过哪些仗？',
    '你墓里那个大鼎是做什么用的？',
    '商朝的玉从哪里来？',
    '武丁对你怎么样？',
]

def build_messages(question, nonce='', chunk_ids=None):
    """nonce: 塞 system 最前制造 cache miss; chunk_ids: 指定检索片段模拟上线变动"""
    chunks = RAG_CHUNKS if chunk_ids is None else [RAG_CHUNKS[i] for i in chunk_ids]
    
    # 🔧 Level 1 优化：只取 1 条片段 + 压缩到前 2 句
    chunks = chunks[:1]
    knowledge = '\n'.join(['\n'.join(c.split('\n')[:2]) for c in chunks])
    
    sys_txt = (('[ctx-' + nonce + ']\n') if nonce else '') + PERSONA
    return [
        {'role': 'system', 'content': sys_txt},
        {'role': 'user', 'content': f'参考：{knowledge}\n\n游客问：{question} /no_think'}
    ]

# ---------------- think 过滤 ----------------
# 标签用拼接构造, 源码中不出现完整标签序列, 防复制链路剥离
TAG_O = '<' + 'think' + '>'
TAG_C = '<' + '/think' + '>'
LEN_O, LEN_C = len(TAG_O), len(TAG_C)

class ThinkFilter:
    """流式剔除 think 标签对 + 出口清洗前导空白"""

    def __init__(self):
        self.buf = ''
        self.in_think = False
        self.done = False
        self.started = False   # 是否已吐出首个非空白字符

    def _clean(self, s):
        """think 滤除后残留的前导换行/空格会污染 TTS 并让 cap 提前触发"""
        if not self.started and s:
            s = s.lstrip('\n\r\t \u3000')
            if s:
                self.started = True
        return s

    def feed(self, delta):
        if self.done:
            return self._clean(delta)
        self.buf += delta
        out = ''
        while True:
            if not self.in_think:
                i = self.buf.find(TAG_O)
                if i == -1:
                    if len(self.buf) > LEN_O:
                        out += self.buf[:-LEN_O]
                        self.buf = self.buf[-LEN_O:]
                    return self._clean(out)
                out += self.buf[:i]
                self.buf = self.buf[i + LEN_O:]
                self.in_think = True
            else:
                j = self.buf.find(TAG_C)
                if j == -1:
                    self.buf = self.buf[-LEN_C:]
                    return self._clean(out)
                rest = self.buf[j + LEN_C:]
                self.buf = ''
                self.in_think = False
                self.done = True   # 只处理一对, 之后直通
                out += rest
                return self._clean(out)

    def flush(self):
        """流结束取回暂存; 未闭合 think 内容丢弃"""
        if self.in_think:
            self.buf = ''
            return ''
        out, self.buf = self.buf, ''
        return self._clean(out)

# ---------------- 分句 ----------------
SENT_STRONG = '。！？；!?;'
SENT_SOFT   = '，,、'

def first_seg_hit(text):
    """返回 (是否命中, 原因); 优先自然断句, cap 仅兜底防长难句拖延首响"""
    if len(text) >= 4 and text[-1] in SENT_STRONG:
        return True, 'strong'
    if len(text) >= SOFT_MIN and text[-1] in SENT_SOFT:
        return True, 'soft'
    if len(text) >= FIRST_SEG_CAP:
        return True, 'cap'
    return False, ''

# ---------------- token 计数 ----------------
def count_tokens_endpoint(text):
    """llama_cpp.server: /extras/tokenize/count; llama-server.exe: /tokenize"""
    for url, body, pick in (
        (HOST + '/extras/tokenize/count', {'input': text},
         lambda d: d.get('count')),
        (HOST + '/tokenize', {'content': text},
         lambda d: len(d.get('tokens') or []) or None),
    ):
        try:
            r = requests.post(url, json=body, timeout=20)
            if r.ok:
                n = pick(r.json())
                if n:
                    return n, url.rsplit('/', 1)[-1]
        except Exception:
            pass
    return None, None

def probe_prompt_tokens():
    """非流式 max_tokens=1 探针读 usage.prompt_tokens; 独立 nonce 不污染 A/B 缓存"""
    payload = {
        'model': MODEL,
        'messages': build_messages(QUESTIONS[0], nonce='probe-' + uuid.uuid4().hex[:8]),
        'max_tokens': 1,
        'temperature': 0,
        'stream': False,
    }
    try:
        r = requests.post(CHAT_URL, json=payload, timeout=180)
        r.raise_for_status()
        u = r.json().get('usage') or {}
        return u.get('prompt_tokens')
    except Exception as e:
        print('  探针失败: %s' % e)
        return None

# ---------------- 单次请求 ----------------
def one_run(question, max_tok=MAX_TOK, nonce='', chunk_ids=None):
    payload = {
        'model': MODEL,
        'messages': build_messages(question, nonce, chunk_ids),
        'max_tokens': max_tok,
        'temperature': 0.7,
        'stream': True,
        'stream_options': {'include_usage': True},
    }
    t0 = time.perf_counter()
    r = requests.post(CHAT_URL, json=payload, stream=True, timeout=180)
    if r.status_code in (400, 422):
        payload.pop('stream_options', None)
        t0 = time.perf_counter()
        r = requests.post(CHAT_URL, json=payload, stream=True, timeout=180)
    r.raise_for_status()

    ttft = first_sent = None
    first_txt, first_why = '', ''
    n_chunk = 0
    usage = None
    finish = None
    text = ''
    filt = ThinkFilter()

    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith('data: '):
            continue
        data = line[6:]
        if data.strip() == '[DONE]':
            break
        obj = json.loads(data)
        if obj.get('usage'):
            usage = obj['usage']
        for ch in obj.get('choices', []):
            if ch.get('finish_reason'):
                finish = ch['finish_reason']
            delta = (ch.get('delta') or {}).get('content') or ''
            if not delta:
                continue
            n_chunk += 1
            vis = filt.feed(delta)
            if not vis:
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
            text += vis
            if first_sent is None:
                hit, why = first_seg_hit(text)
                if hit:
                    first_sent = time.perf_counter() - t0
                    first_txt, first_why = text, why
    text += filt.flush()
    total = time.perf_counter() - t0

    n_out = (usage or {}).get('completion_tokens') or n_chunk
    n_in = (usage or {}).get('prompt_tokens')
    decode_tps = (n_out - 1) / (total - ttft) if ttft and total > ttft else 0
    return {'ttft': ttft, 'first_sent': first_sent, 'first_txt': first_txt,
            'first_why': first_why, 'total': total, 'n_in': n_in, 'n_out': n_out,
            'decode_tps': decode_tps, 'finish': finish, 'text': text}

def med(rows, key):
    vals = [r[key] for r in rows if r.get(key)]
    return statistics.median(vals) if vals else 0

# ---------------- 主流程 ----------------
def main():
    print('=' * 66)
    print('1. prompt 规模标定（非流式探针, 不用 tiktoken）')
    n_in_probe = probe_prompt_tokens()
    n_ep, ep = count_tokens_endpoint(PERSONA + '\n\n'.join(RAG_CHUNKS[:1]))
    print('   探针 prompt_tokens        %s' % (n_in_probe or '取不到'))
    if n_ep:
        print('   tokenize 端点(/%s)   %s (不含 chat template, 仅参考)' % (ep, n_ep))

    print('')
    print('2. 预热（冷启动不计入）')
    w = one_run(QUESTIONS[0], max_tok=32)
    print('   预热 TTFT=%.3fs' % (w['ttft'] or 0))

    print('')
    print('3. 单流基准 · %d 轮取中位数（共享前缀全缓存 ← 理想值）' % RUNS)
    rows = []
    for i in range(RUNS):
        res = one_run(QUESTIONS[i % len(QUESTIONS)])
        rows.append(res)
        print('   [%d] TTFT %.3fs | 首句 %.3fs(%s) | decode %5.1f tok/s | out %s | %s' % (
            i, res['ttft'] or 0, res['first_sent'] or 0, res['first_why'] or '-',
            res['decode_tps'], res['n_out'], res['finish'] or '?'))
        print('       首句: %s' % (res['first_txt'] or '(未命中)'))
    ideal_ttft = med(rows, 'ttft')

    print('')
    print('   中位数汇总')
    print('     prompt tokens          %s' % (n_in_probe or med(rows, 'n_in') or '未知'))
    print('     TTFT                   %.3f s' % ideal_ttft)
    print('     首句可送 TTS           %.3f s   ← 决定体感' % med(rows, 'first_sent'))
    print('     decode 速度            %.1f tok/s  ← 别叫吞吐量' % med(rows, 'decode_tps'))
    fs = [r['first_sent'] for r in rows if r.get('first_sent')]
    if fs:
        print('     首句 min/max           %.3f / %.3f s' % (min(fs), max(fs)))

    print('')
    print('4. prefix cache A/B（nonce 隔离: 冷=独占前缀, 热=复用同前缀换问题）')
    nonce = uuid.uuid4().hex[:8]
    cold = one_run(QUESTIONS[1], max_tok=32, nonce=nonce)   # miss
    warm = one_run(QUESTIONS[2], max_tok=32, nonce=nonce)   # hit
    cold_ttft = cold['ttft'] or 0
    if cold['ttft'] and warm['ttft']:
        drop = (1 - warm['ttft'] / cold['ttft']) * 100
        print('   冷 TTFT %.3fs → 热 TTFT %.3fs（降幅 %.0f%%）' % (
            cold['ttft'], warm['ttft'], drop))
        if n_in_probe:
            print('   prefill 速度(冷)  %.0f tok/s' % (n_in_probe / cold['ttft']))
        if drop < 30:
            print('   ⚠ 降幅 <30%: 前缀未复用, 或 prefill 本就极短, 对照 prefill 速度判断')
        else:
            print('   ✓ 前缀复用生效')
    else:
        print('   A/B 测量失败, 跳过')

    print('')
    print('5. 上线模拟（top-1 检索片段每轮变化, 只有人设吃缓存 ← 真实体感）')
    combos = [(0,), (1,), (2,), (3,), (4,)]
    one_run(QUESTIONS[0], max_tok=24, chunk_ids=(3,))
    prows = []
    for i, ids in enumerate(combos):
        res = one_run(QUESTIONS[i % len(QUESTIONS)], max_tok=96, chunk_ids=ids)
        prows.append(res)
        print('   [%d] 片段%s | TTFT %.3fs | 首句 %.3fs(%s)' % (
            i, list(ids), res['ttft'] or 0,
            res['first_sent'] or 0, res['first_why'] or '-'))
        print('       首句: %s' % (res['first_txt'] or '(未命中)'))
    live_ttft = med(prows, 'ttft')
    print('   上线模拟中位: TTFT %.3f s | 首句可送 TTS %.3f s' % (
        live_ttft, med(prows, 'first_sent')))

    print('')
    print('6. 串行对照（4 请求依次跑, 作为并发段基线）')
    t0 = time.perf_counter()
    ser = [one_run(QUESTIONS[i % len(QUESTIONS)]) for i in range(4)]
    ser_wall = time.perf_counter() - t0
    ser_out = sum(r['n_out'] or 0 for r in ser)
    ser_med_out = statistics.median([r['n_out'] or 0 for r in ser])
    print('   串行: 墙钟 %.1fs | 总 out %d | 聚合 %.1f tok/s | 单请求 out 中位 %d' % (
        ser_wall, ser_out, ser_out / ser_wall, ser_med_out))

    print('')
    print('7. 并发段（含空返/截断检测; 服务端应已加 --interrupt_requests False）')
    for c in CONC:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as ex:
            res = list(ex.map(
                lambda i: one_run(QUESTIONS[i % len(QUESTIONS)]), range(c)))
        wall = time.perf_counter() - t0
        outs = [r['n_out'] or 0 for r in res]
        agg = sum(outs) / wall
        worst = max((r['ttft'] or 0) for r in res)
        fins = [r['finish'] or 'none' for r in res]
        print('   %d 并发: 聚合 %6.1f tok/s | 最差 TTFT %.3fs | 墙钟 %.1fs' % (
            c, agg, worst, wall))
        print('         每路 out %s | finish_reason %s' % (outs, fins))
        short = [o for o in outs if ser_med_out and o < ser_med_out * 0.6]
        if c > 1 and (short or any(f == 'none' for f in fins)):
            print('         ⚠ 存在空返/截断（对照串行基线 out 中位 %d）, 该档聚合吞吐无效;' % ser_med_out)
            print('           确认服务端已加 --interrupt_requests False;')
            print('           真并发需换 llama.cpp 官方 llama-server.exe -np %d' % c)
        elif c > 1 and wall < ser_wall * 0.8:
            print('         注: 墙钟明显短于串行 %.1fs, 存在真实批处理收益' % ser_wall)
        elif c > 1:
            print('         注: 输出完整且墙钟≈串行, 属排队串行, 单游客场景可接受')

    print('')
    print('8. 样例输出 + 三点标定')
    print('   样例: ' + rows[0]['text'][:150].replace('\n', ' '))
    print('   TTFT 标定: 全缓存理想 %.3fs | 上线模拟(top-1变动) %.3fs | 无缓存冷启 %.3fs' % (
        ideal_ttft, live_ttft, cold_ttft))
    print('=' * 66)

if __name__ == '__main__':
    main()