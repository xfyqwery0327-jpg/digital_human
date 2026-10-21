# -*- coding: utf-8 -*-
"""
chat_qwen.py - RAG + Qwen3 多轮对话 v4
v3 已有:
  1. category 关键词路由
  2. 人称改写（规则）
  3. 启动探测 server 能力，自动设定拒答阈值
  4. BM25 感知拒答
  5. 低分拒答 + 无资料如实告知
  6. 关闭 Qwen3 思考模式 + 首句 lstrip（保护 TTS）
v4 新增:
  7. 意图识别（调 server /intent，server 没加则自动兜底 knowledge）
  8. 指令直连（音量/暂停/重复等，不进 LLM）
  9. 多轮指代消解（条件触发 LLM 改写："它→司母辛鼎"）
 10. 反思补检索（检索不到 → LLM 换问法再查一次，而非直接编造）
 11. 题型路由（决定 top_k 与是否开思考）
用法:
  conda activate robot
  cd C:\ro
  python chat.py     （需先启动 server.py）
"""
import sys
import requests
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# 强制 UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# ============================================================
# 配置
# ============================================================
RAG_URL = "http://localhost:8848/search"
RAG_HEALTH_URL = "http://localhost:8848/health"
RAG_INTENT_URL = "http://localhost:8848/intent"
MODELS_DIR = Path("models/models")

PERSONA = "妇好"           # 机器人扮演的角色（用于人称改写）；通用助手设为 None
DEFAULT_CATEGORY = None    # 关键词路由失败时的兜底：None = 全库检索

# 拒答阈值（启动时由 detect_server_mode 自动选择）
MIN_SCORE_VEC = 0.45
MIN_SCORE_RERANK = 0.50
MIN_SCORE_BM25 = 0.25
MIN_SCORE = MIN_SCORE_VEC
SERVER_HAS_RERANKER = False

# 题型 → 检索条数（列表型要多返回几条，否则"打过哪些仗"只讲一场）
TOP_K_BY_QTYPE = {"factoid": 2, "list": 5, "explanation": 3,
                  "comparison": 4, "opinion": 2}

# 哪些题型开启思考模式（CoT）。
# ⚠️ 思考会产生额外 token 且可能残留思考标记进 TTS，所以默认全关。
#    确认你的 Qwen 模板能干净剥离思考标记、且能接受延迟后，再往集合里加题型。
THINKING_QTYPES = set()   # 例如 {"explanation", "comparison"}

# 关键词 → 知识库类别 路由表（与 build_rag.py 的 CATEGORY_MAP 对应）
CATEGORY_KEYWORDS = {
    "exhibition_fuhao":        ["妇好", "商朝", "武丁", "殷墟", "甲骨", "青铜", "玉",
                                "司母辛", "司母戊", "鸮尊", "祭祀", "冥婚", "羌方"],
    "exhibition_monet":        ["莫奈", "睡莲", "印象派", "浪漫与星辰", "票价", "门票", "票"],
    "headquarters":            ["魔墙", "宽创", "荣誉墙", "发展历程", "设计院", "总部"],
    "exhibition_wanlijianghe": ["万里江河", "兵马俑", "清明上河图", "秦汉", "隋唐",
                                "宋元明清", "魏晋", "先秦"],
}

# ============================================================
# 查找 Qwen 模型
# ============================================================
def find_qwen_model():
    for d in MODELS_DIR.rglob("*"):
        if d.is_dir() and "qwen" in d.name.lower():
            snapshots = d / "snapshots"
            if snapshots.exists():
                subdirs = [x for x in snapshots.iterdir() if x.is_dir()]
                if subdirs:
                    return str(subdirs[0])
            if (d / "config.json").exists():
                return str(d)
    return None

# ============================================================
# RAG 服务能力探测 + 意图分类（客户端）
# ============================================================
def detect_server_mode():
    global MIN_SCORE, SERVER_HAS_RERANKER
    try:
        h = requests.get(RAG_HEALTH_URL, timeout=3).json()
        SERVER_HAS_RERANKER = bool(h.get("reranker"))
        MIN_SCORE = MIN_SCORE_RERANK if SERVER_HAS_RERANKER else MIN_SCORE_VEC
        print(f"✅ RAG 服务: {h.get('chunks')} 块 | bm25={h.get('bm25')} | "
              f"reranker={SERVER_HAS_RERANKER} | 拒答阈值={MIN_SCORE}")
        if not h.get("intent"):
            print("⚠️ server 未启用意图识别（/intent），chat 端将按 knowledge 兜底")
    except Exception as e:
        print(f"⚠️ RAG 服务未连接（{e}），检索功能不可用，请先启动 server.py")

def classify_intent(query):
    """调 server /intent；服务不可用时兜底为 knowledge（不影响主流程）"""
    try:
        r = requests.post(RAG_INTENT_URL, json={"query": query}, timeout=3).json()
        return r.get("intent", "knowledge"), r.get("score", 0)
    except Exception:
        return "knowledge", 0

# ============================================================
# 查询理解（规则改写 + 路由）
# ============================================================
def rewrite_query(query):
    """检索前规则改写：无检索语义的代词换成人物名（只影响检索，不影响对话原文）"""
    if not PERSONA:
        return query
    q = query
    if "你" in q and "你好" not in q:
        q = q.replace("你们", "商朝人").replace("你", PERSONA)
    if "她" in q:
        q = q.replace("她", PERSONA)
    return q

def route_category(query):
    """按关键词路由到对应知识库类别"""
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(k in query for k in kws):
            return cat
    return DEFAULT_CATEGORY

# ============================================================
# 检索（混合检索 + BM25 感知拒答）
# ============================================================
def search_knowledge(query, top_k=2):
    rq = rewrite_query(query)
    cat = route_category(rq)
    print(f"   🔍 检索: {rq} [{cat or '全库'}] top_k={top_k}")
    try:
        response = requests.post(RAG_URL, json={
            "query": rq, "top_k": top_k, "category": cat, "compress": True,
        }, timeout=5)
        results = response.json().get("results", [])
    except Exception as e:
        print(f"⚠️  检索失败: {e}")
        return []

    kept = []
    for r in results:
        score = r.get("score", 0)
        recall = r.get("recall", "vector")
        if score >= MIN_SCORE:
            kept.append(r)
        elif (not SERVER_HAS_RERANKER
              and recall in ("bm25", "both")
              and score >= MIN_SCORE_BM25):
            kept.append(r)

    if not kept and results:
        top = results[0]
        print(f"   ⚠️ 最高分 {top.get('score', 0):.3f}"
              f"（{top.get('recall', '?')}）未过阈值，按无资料处理")
    return kept

# ============================================================
# 指令直连（不进 LLM）
# ============================================================
COMMAND_ACTIONS = [
    (("大一点", "大声"), "volume_up", "好的，我把音量调大一点。"),
    (("小一点", "小声"), "volume_down", "好的，我把音量调小一点。"),
    (("暂停", "停止", "别说了", "闭嘴"), "pause", "好的，我先不说了，需要我时再叫我。"),
    (("继续",), "resume", "好的，我们继续。"),
    (("重复", "再说一遍"), "repeat", None),  # None → 重播上一条回答
]

def handle_command(user_input, bot):
    """指令直接执行，不进 LLM。返回 True 表示已处理"""
    for keywords, action, reply in COMMAND_ACTIONS:
        if any(k in user_input for k in keywords):
            if action == "repeat":
                last = next((h["content"] for h in reversed(bot.history)
                             if h["role"] == "assistant"), None)
                reply = last or "刚才我还没有讲过内容呢。"
            print(f"   🎮 指令: {action}")
            # TODO: 在这里对接机器人本体控制（音量/播放器/移动）
            print(f"\n🤖 助手: {reply}\n")
            return True
    return False

# ============================================================
# 多轮对话
# ============================================================
class ChatBot:
    NEED_REWRITE_HINT = ("它", "他", "这个", "那个", "这些", "那些", "这里", "那里")

    def __init__(self, model_path):
        print("\n🧠 加载 Qwen 模型...")
        print(f"   路径: {model_path}")

        if not torch.cuda.is_available():
            print("⚠️  警告: 未检测到 CUDA，将使用 CPU（速度较慢）")
            device_map = "cpu"
            torch_dtype = torch.float32
        else:
            print(f"   ✅ 检测到 GPU: {torch.cuda.get_device_name(0)}")
            print(f"   ✅ 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            device_map = "cuda:0"
            torch_dtype = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("   🔧 加载中...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device_map, torch_dtype=torch_dtype,
            trust_remote_code=True, low_cpu_mem_usage=True).eval()

        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        print(f"   ✅ 模型设备: {device} | 数据类型: {dtype}")
        if torch.cuda.is_available():
            print(f"   ✅ 已用显存: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

        self.history = []
        self.system_prompt = """你是殷墟博物馆的智能导览助手，为游客提供专业讲解。
基于参考资料回答问题，不编造信息。
如果参考资料与问题无关或没有资料，如实告知游客并引导到你熟悉的话题。
回答要求：准确、简洁（100字内）、生动、口语化，适合语音播报。
不使用括号、书名号等无法朗读的符号。"""

    # ---------- 题型判断（规则版，零延迟） ----------
    def infer_qtype(self, query):
        if any(k in query for k in ("哪些", "有哪些", "多少种", "几种", "列出", "都打过", "都出土")):
            return "list"
        if any(k in query for k in ("为什么", "为何", "原因", "怎么回事", "怎么来的", "如何")):
            return "explanation"
        if any(k in query for k in ("区别", "比较", "哪个更", "谁更", "还是")):
            return "comparison"
        if any(k in query for k in ("你觉得", "评价", "厉害吗", "重要吗", "怎么样")):
            return "opinion"
        return "factoid"

    # ---------- 多轮指代消解（条件触发） ----------
    def needs_llm_rewrite(self, query):
        if not self.history:
            return False
        if any(w in query for w in self.NEED_REWRITE_HINT):
            return True
        return len(query) <= 5   # "多重？""还有呢"

    def rewrite_with_llm(self, query):
        lines = []
        for h in self.history[-4:]:
            role = "游客" if h["role"] == "user" else "讲解员"
            lines.append(f"{role}：{h['content'][:80]}")
        prompt = ("对话记录：\n" + "\n".join(lines) + f"\n游客：{query}\n\n"
                  "把游客最后这句话改写成一个不依赖上下文、带明确对象名称的完整问题。"
                  "只输出改写后的问题，不要解释。")
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=40, do_sample=False)
        rewritten = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().strip('"')
        if not rewritten or len(rewritten) > 50 or "\n" in rewritten:
            return query
        print(f"   ✏️ 指代消解: {query} → {rewritten}")
        return rewritten

    # ---------- 反思补检索（检索不到时换问法，而非编造） ----------
    def reflect_and_research(self, query):
        prompt = (f"游客问：{query}\n"
                  "知识库第一次检索没有找到资料。请分析这句话想了解的核心信息点，"
                  "给出2到3个更可能命中的检索词或改写问法，用顿号分隔。"
                  "只输出检索词，不要解释，不要回答原问题。")
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=30, do_sample=False)
        alt = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if not alt or len(alt) > 40:
            return None
        print(f"   🔄 反思换问法: {query} → {alt}")
        return alt

    # ---------- 单轮对话 ----------
    def chat(self, user_input, use_rag=True, allow_freeform=False):
        meta = {}

        # 0. 题型判断
        qtype = self.infer_qtype(user_input)
        meta["qtype"] = qtype

        # 1. 多轮指代消解
        search_query = user_input
        if use_rag and self.needs_llm_rewrite(user_input):
            search_query = self.rewrite_with_llm(user_input)

        # 2. 第一次检索
        top_k = TOP_K_BY_QTYPE.get(qtype, 2)
        contexts = search_knowledge(search_query, top_k=top_k) if use_rag else []

        # 3. 反思补检索：无命中换问法再查一次
        if use_rag and not contexts:
            alt = self.reflect_and_research(user_input)
            if alt:
                contexts = search_knowledge(alt, top_k=top_k)
                meta["reflected"] = True

        # 4. 构建提示词
        if contexts:
            context_text = '\n\n'.join([c['text'] for c in contexts])
            prompt = f"参考：{context_text}\n\n游客问：{user_input}"
        elif use_rag and not allow_freeform:
            prompt = f"知识库中没有找到与这个问题相关的资料。\n\n游客问：{user_input}"
        else:
            prompt = user_input

        messages = [{"role": "system", "content": self.system_prompt}]
        for h in self.history[-4:]:
            messages.append(h)
        messages.append({"role": "user", "content": prompt})

        # 5. 生成（是否开思考由 THINKING_QTYPES 决定，默认全关）
        enable_thinking = qtype in THINKING_QTYPES
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking)
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)
        generation_kwargs = dict(**model_inputs, streamer=streamer,
                                 max_new_tokens=200, temperature=0.7, top_p=0.9, do_sample=True)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        print("\n🤖 助手: ", end="", flush=True)
        response = ""
        started = False
        for new_text in streamer:
            if not started:
                new_text = new_text.lstrip("\n")
                if new_text:
                    started = True
            print(new_text, end="", flush=True)
            response += new_text

        thread.join()
        print()

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response, contexts, meta

    def reset_history(self):
        self.history = []
        print("🔄 对话历史已清空")

# ============================================================
# 主程序
# ============================================================
def main():
    print("\n" + "="*60)
    print("🤖 殷墟博物馆 RAG + Qwen3 多轮对话系统 v4")
    print("="*60)

    detect_server_mode()

    model_path = find_qwen_model()
    if not model_path:
        print("❌ 找不到 Qwen 模型，请确认模型在 models/models/ 目录下")
        sys.exit(1)

    bot = ChatBot(model_path)

    print("\n" + "="*60)
    print("✅ 系统已启动")
    print(f"   角色: {PERSONA or '通用助手'} | 拒答阈值: {MIN_SCORE} | "
          f"reranker: {SERVER_HAS_RERANKER}")
    print("\n命令：")
    print("  - 输入问题开始对话")
    print("  - 输入 /clear 清空历史")
    print("  - 输入 /quit 退出")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("🙋 游客: ").strip()
            if not user_input:
                continue
            if user_input == "/quit":
                print("\n👋 再见！")
                break
            if user_input == "/clear":
                bot.reset_history()
                continue

            # 意图分流
            intent, iscore = classify_intent(user_input)
            print(f"   🎯 意图: {intent} ({iscore:.3f})")

            if intent == "command":
                if handle_command(user_input, bot):
                    continue
                intent = "chitchat"   # 动作表没匹配到 → 按闲聊兜底

            response, contexts, meta = bot.chat(
                user_input, use_rag=True, allow_freeform=(intent == "chitchat"))

            if contexts:
                print(f"\n📚 参考来源（题型={meta.get('qtype')}"
                      f"{'，反思补检索' if meta.get('reflected') else ''}）:")
                for i, c in enumerate(contexts, 1):
                    print(f"   [{i}] {c.get('source', '未知')} [{c.get('category', '')}] "
                          f"({c.get('recall', '?')}, {c.get('score', 0):.3f})")
            print()

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()