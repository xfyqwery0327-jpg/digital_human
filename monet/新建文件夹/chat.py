# -*- coding: utf-8 -*-
"""
chat_qwen.py - RAG + Qwen3 多轮对话 (GPU优化 + 流式输出)
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
MODELS_DIR = Path("models/models")

# ============================================================
# 查找 Qwen 模型
# ============================================================
def find_qwen_model():
    """查找 Qwen 模型目录"""
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
# RAG 检索
# ============================================================
def search_knowledge(query, top_k=2):
    """从知识库检索"""
    try:
        response = requests.post(RAG_URL, json={
            "query": query,
            "top_k": top_k
        }, timeout=5)
        return response.json()["results"]
    except Exception as e:
        print(f"⚠️  检索失败: {e}")
        return []

# ============================================================
# 多轮对话
# ============================================================
class ChatBot:
    def __init__(self, model_path):
        print("\n🧠 加载 Qwen 模型...")
        print(f"   路径: {model_path}")
        
        # 检查 CUDA
        if not torch.cuda.is_available():
            print("⚠️  警告: 未检测到 CUDA，将使用 CPU（速度较慢）")
            device_map = "cpu"
            torch_dtype = torch.float32
        else:
            print(f"   ✅ 检测到 GPU: {torch.cuda.get_device_name(0)}")
            print(f"   ✅ 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            device_map = "cuda:0"
            torch_dtype = torch.float16
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 加载模型（GPU + FP16）
        print("   🔧 加载中...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).eval()
        
        # 验证设备
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        print(f"   ✅ 模型设备: {device}")
        print(f"   ✅ 数据类型: {dtype}")
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"   ✅ 已用显存: {memory_allocated:.2f} GB")
        
        # 对话历史
        self.history = []
        
        # 系统提示词
        self.system_prompt = """你是殷墟博物馆的智能导览助手"妇好"，专门为游客提供专业、准确、生动的讲解服务。

你的职责：
1. 基于知识库内容回答问题，不编造信息
2. 如果知识库中没有相关信息，明确告知用户
3. 保持专业、友好、易懂的语气
4. 可以结合上下文进行连贯对话
5. 适当补充背景知识，让讲解更生动

回答要求：
- 准确：基于知识库事实
- 简洁：重点突出，避免冗长（150字以内）
- 生动：适当使用比喻、故事化表达
- 互动：可以引导游客继续探索"""
    
    def chat(self, user_input, use_rag=True):
        """单轮对话（流式输出）"""
        # 1. RAG 检索
        contexts = []
        if use_rag:
            contexts = search_knowledge(user_input, top_k=2)
        
        # 2. 构建提示词
        if contexts:
            context_text = "\n\n".join([
                f"【参考资料{i+1}】\n{c['text'][:300]}"
                for i, c in enumerate(contexts)
            ])
            
            prompt = f"""参考以下知识库资料回答问题：

{context_text}

问题：{user_input}

要求：用150字以内简洁回答。"""
        else:
            prompt = user_input
        
        # 3. 构建消息列表
        messages = [{"role": "system", "content": self.system_prompt}]
        for h in self.history[-4:]:
            messages.append(h)
        messages.append({"role": "user", "content": prompt})
        
        # 4. 准备输入
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt"
        ).to(self.model.device)
        
        # 5. 流式生成
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True
        )
        
        generation_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # 6. 实时输出
        print("\n🤖 妇好: ", end="", flush=True)
        response = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            response += new_text
        
        thread.join()
        print()
        
        # 7. 更新历史
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        
        return response, contexts
    
    def reset_history(self):
        """清空对话历史"""
        self.history = []
        print("🔄 对话历史已清空")

# ============================================================
# 主程序
# ============================================================
def main():
    print("\n" + "="*60)
    print("🤖 殷墟博物馆 RAG + Qwen3 多轮对话系统 (流式输出)")
    print("="*60)
    
    # 1. 查找模型
    model_path = find_qwen_model()
    if not model_path:
        print("❌ 找不到 Qwen 模型")
        print("请确认模型在 models/models/ 目录下")
        sys.exit(1)
    
    # 2. 初始化
    bot = ChatBot(model_path)
    
    # 3. 欢迎语
    print("\n" + "="*60)
    print("✅ 系统已启动（支持流式输出）")
    print("\n命令：")
    print("  - 输入问题开始对话")
    print("  - 输入 /clear 清空历史")
    print("  - 输入 /quit 退出")
    print("="*60 + "\n")
    
    # 4. 对话循环
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
            
            # 流式生成
            response, contexts = bot.chat(user_input)
            
            # 显示来源
            if contexts:
                print(f"\n📚 参考来源:")
                for i, c in enumerate(contexts, 1):
                    print(f"   [{i}] {c['source']} (相似度: {c['score']:.3f})")
            
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