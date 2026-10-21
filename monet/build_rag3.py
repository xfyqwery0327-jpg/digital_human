#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_rag3.py - 离线索引构建（跑一次，生成 index/ 供 server.py 加载）

链路：
  docx/xlsx → 章节切块(500字/重叠100, 超长降级滑窗)
  → QA 双向量多表示索引(纯问题 + 问答全文, 共享 dedup_key)
  → gte 编码 + L2 归一化 + FAISS(IndexFlatIP)
  → jieba(专名词典) + BM25Okapi
  → 保存 index/

知识库现在为「宽创国际」（企业荣誉一览表 + 展厅讲解词 + 自动生成的高频QA），
所有条目 category=exhibition_broadmesse，
与 server.py 的 EXHIBITION_CATEGORY = ["exhibition_broadmesse"] 对应。

用法:
  conda activate robot
  cd C:\ro                ← 必须和 server4.py 启动目录一致，
  python build_rag.py        两边读写的才是同一个 index/
"""
import re
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

# ★ 原始文档目录：把「宽创国际企业荣誉一览表2026.1.16.docx」「宽创展厅 讲解词.docx」放这里
DATA_DIR = Path(r"C:\knowledge_base3")
OUTPUT_DIR = Path("index3")       # 索引输出目录（相对当前运行目录）
# ★ 必须与 server4.py 的 EMBED_MODEL 一字不差（建索引和线上检索共用同一模型）
EMBED_MODEL = r"C:\Users\BroadMesse\.cache\huggingface\hub\models--thenlper--gte-base-zh\snapshots\71ab7947d6fac5b64aa299e6e40e6c2b2e85976c"
CHUNK_SIZE = 500                 # 章节切块字数
CHUNK_OVERLAP = 100              # 超长章节滑窗重叠字数
DICT_PATH = Path("dict/jieba_dict.txt")   # 专名词典（不存在则自动生成）

DEFAULT_CATEGORY = "exhibition_broadmesse"

# 文件名/目录 → 知识库类别（知识库现在只有宽创）
CATEGORY_MAP = {
    "宽创":        DEFAULT_CATEGORY,
    "broadmesse":  DEFAULT_CATEGORY,
    "荣誉":        DEFAULT_CATEGORY,
    "讲解词":      DEFAULT_CATEGORY,
    "展厅":        DEFAULT_CATEGORY,
}

# ============================================================
# 专名词典自动生成
# ============================================================

PROPER_NOUNS = [
    "宽创国际", "宽创", "Broadmesse", "宽小志", "张东", "张东博士", "李丹",
    "张江", "张江总部", "张江科学城", "宽创国际产业园", "宽创文化科技研究院",
    "国家文化出口重点企业", "国家高新技术企业", "文博文旅EPCO", "EPCO",
    "全球会展活动服务商", "一带一路", "上海世博会", "德国馆", "迪拜世博会",
    "邦德激光", "美国工博会", "柏林IFA", "IFA", "用友", "郑州航空港",
    "荣誉墙", "魔墙区", "人工智能展示馆", "人工智能科技馆", "万里江河",
    "智联万象", "智联·万象", "城市客厅", "远古回响", "兵马俑AI数字交互区",
    "龟兹壁画", "克孜尔石窟", "菱格故事画", "科举四宴", "大运河漕运",
    "名窑瓷器", "清明上河图", "盛世修典", "文明长河", "互动桌", "互动翻书",
    "四库全书", "永乐大典", "张仲景", "望闻问切", "妇好", "殷墟",
    "数字孪生", "智能体平台", "表情机器人", "算力平台", "传动控制", "数据采集",
    "激光雷达", "裸眼3D", "沉浸式", "生成式人工智能", "区块链甲级机构",
    "上海市科技小巨人企业", "上海市企业技术中心", "上海市专精特新企业",
    "上海市设计引领示范企业", "上海市五一劳动奖状", "弘博奖", "金展奖",
    "全国博物馆十大陈列展览精品奖", "全国十佳文博技术产品及服务奖",
    "上海交大", "上海交通大学", "复旦", "财大", "中国贸促会", "五部委",
    "元宇宙", "国际设计科学学会", "土耳其伊斯坦布尔国际书展", "巴厘岛水利遗址博物馆",
]

def ensure_jieba_dict():
    """专名词典不存在则自动生成（每行: 词 词频 词性）"""
    if DICT_PATH.exists():
        print(f"📖 专名词典已存在: {DICT_PATH}")
        return
    DICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DICT_PATH, "w", encoding="utf-8") as f:
        for w in PROPER_NOUNS:
            f.write(f"{w} 1000 n\n")
    print(f"📝 已生成专名词典: {DICT_PATH}（{len(PROPER_NOUNS)} 词）")

def load_jieba_dict():
    try:
        import jieba
        if DICT_PATH.exists():
            jieba.load_userdict(str(DICT_PATH))
            print(f"📖 已加载专名词典: {DICT_PATH}")
    except Exception as e:
        print(f"⚠️ 词典加载失败（忽略）: {e}")

# ============================================================
# 宽创高频 FAQ 自动生成
# ============================================================

def ensure_broadmesse_faq():
    """确保宽创高频FAQ文件存在，不存在则自动创建。
    覆盖参观者最常问的公司概况/荣誉/历程/展区问题，
    QA 纯问题向量让这类问题直接精确命中。内容以荣誉一览表和讲解词为准。"""
    faq_file = DATA_DIR / "宽创高频QA.xlsx"

    if faq_file.exists():
        print(f"✅ 宽创FAQ文件已存在: {faq_file.name}")
        return

    print(f"📝 生成宽创FAQ文件: {faq_file.name}")

    faq = [
        {
            "问题": "宽创国际是一家什么公司？介绍一下宽创。",
            "答案": "宽创国际成立于2005年，在德国慕尼黑、中国香港、上海张江三地同步创立，是中国科技+文化跨界的领军企业。宽创有两张身份证：一张是全球会展活动服务商，一张是文博文旅EPCO。目前服务覆盖全球180多个国家、300多个会展城市，是中国企业出海的品牌全案赋能伙伴；同时已落地1600多个博物馆及文旅项目，数十次承接一带一路重大国家文化工程。"
        },
        {
            "问题": "宽创是什么时候成立的？",
            "答案": "宽创国际于2005年成立，在德国慕尼黑、中国香港、上海张江三地同时启动。创立初衷是让中国企业在海外展会上拿得到好位置、讲得出好故事。"
        },
        {
            "问题": "宽创的创始人是谁？张东博士是谁？",
            "答案": "宽创国际的创始人是张东博士。他同时是国际设计科学学会元宇宙研究会会长、上海市文化十大创新创业人物、浦东文创企业十大年度领军人物、上海交通大学文创学院产业导师。公司总经理为李丹。"
        },
        {
            "问题": "宽创的英文名Broadmesse是什么意思？",
            "答案": "宽创国际的英文名 Broadmesse 中，messe 是德语「展览会」的意思，体现了公司起源于德国会展业务的背景。"
        },
        {
            "问题": "宽创的两张身份证是什么？宽创的主营业务是什么？",
            "答案": "宽创有两张身份证：一张是全球会展活动服务商，帮助中国企业出海参展、从边缘走到中央；一张是文博文旅EPCO，为博物馆、展馆、文旅项目提供从策划、设计、施工到运营的全链条服务。核心定位是「沉浸式AI+文化」。"
        },
        {
            "问题": "宽创获得过哪些国家级荣誉？",
            "答案": "国家级荣誉包括：国家高新技术企业、国家文化出口重点企业（业内唯一由国家五部委联合认定）、国家文化和旅游科技创新工程项目、国家文化和旅游科技创新研发项目、国家优质工程奖、国家网信办生成式人工智能服务备案。"
        },
        {
            "问题": "宽创获得过哪些上海市级荣誉？",
            "答案": "市级荣誉包括：上海市科技小巨人企业、上海市企业技术中心、上海市专精特新企业、上海市设计引领示范企业、上海市五一劳动奖状、上海市标准化试点、上海市和谐劳动关系达标企业、上海市文化和旅游局科学技术进步一等奖、打响「上海文化」品牌工作创新案例·上海优秀文化品牌、上海文化企业十佳提名企业。"
        },
        {
            "问题": "宽创获得过哪些区级荣誉？",
            "答案": "区级荣誉包括：企业研发机构、重点优势企业、高成长性总部、总部经济十大经典样本、十大文创引领企业、优秀守合同重信用企业、守合同重信用标杆企业、科技创新示范基地、科普基地、特色研学基地、张江科学城重大产业项目、科技创新英才、科技创新成果。"
        },
        {
            "问题": "宽创获得过哪些行业奖项？",
            "答案": "重要行业成果奖项包括：全国十佳文博技术产品及服务奖、全国博物馆十大陈列展览精品奖、全国AI元宇宙1000优秀产品目录与示范案例（五星级）、中国AI元宇宙星级供应商（五星级）、年度十大最具品牌价值设计机构、商业展示空间国际创新设计奖、中国建筑装饰行业十大最具影响力设计机构（展陈空间类）、中国国际建筑装饰及设计艺术博览会最具空间艺术美学TOP10、中国展览艺术与展示技术创意大赛金展奖/创新设计卓越奖/视觉呈现典范奖/互动体验领航奖、中国博物馆及相关产品技术「弘博奖」、多媒体风云盛典最佳元宇宙探索奖、多媒体十大领军企业奖、上海名牌、上海市文化企业十佳提名奖。"
        },
        {
            "问题": "国家文化出口重点企业是什么？宽创为什么能获得？",
            "答案": "国家文化出口重点企业由国家五部委联合认定，宽创是业内唯一一家获此认定的企业。这个荣誉的含金量不在于做了多少场展览，而在于把多少中国品牌、中国文物、中国文化送上了世界级舞台。"
        },
        {
            "问题": "宽创的发展历程是怎样的？",
            "答案": "2005年三地同步创立；2006年被中国贸促会评为中国出口企业的晴雨表；2008年布局「科技+文化展示中国」，与全球研发机构合作AI、大数据；2010年承接上海世博会德国馆等国家级文化科技工程；2012-2013年成立宽创文化科技研究院，走国内国外双线战略，覆盖全球300多个会展城市；2014年成为一带一路重点合作企业，开始引进国际顶级IP；2015年全资打造张江总部大楼，成立宽创国际产业园；2017年至今战略词为「沉浸式AI+文化」。"
        },
        {
            "问题": "宽创和上海世博会有什么关系？",
            "答案": "2010年上海世博会期间，宽创凭借出色的出海及展陈经验，承接了德国馆等多个国家级文化科技工程，得到各方一致好评。"
        },
        {
            "问题": "宽创的业务逻辑是什么？",
            "答案": "宽创的业务逻辑有三点：一是以数智科技助力实体经济智能跃迁，如郑州航空港数字城市展厅、用友等项目；二是以沉浸式体验唤醒文化的内生力，宽创是文旅行业首家完成生成式人工智能大模型服务备案的企业，也是区块链甲级机构；三是跨越山海的文化力量，助力中国文化和中国形象出海，如土耳其伊斯坦布尔国际书展中国馆、迪拜世博会中国主宾国、巴厘岛水利遗址博物馆中国馆等。"
        },
        {
            "问题": "宽创做了哪些海外项目？",
            "答案": "代表性海外项目包括：土耳其伊斯坦布尔国际书展中国馆、迪拜世博会中国主宾国、巴厘岛水利遗址博物馆中国馆等，为中国的国际形象赋能。此外宽创在全球20多个国家组建了在地供应商团队。"
        },
        {
            "问题": "邦德激光的故事是什么？",
            "答案": "邦德激光是宽创服务过的典型客户。最初展位只有9平米标准标摊，几年后扩大到36平米，再到96、300、600平米，去年在美国工博会站上1000平米中央展区，现场放起礼炮，单场展会签约近一个亿。从9平米到1000平米，是中国制造从参与者走向领跑者的缩影。"
        },
        {
            "问题": "中国企业出海会遇到哪些坑？宽创怎么帮企业解决？",
            "答案": "主要有三类：一是专利与合规，曾有企业赴欧参展当天收到侵权投诉，宽创当地团队两小时到场协助解决，现在所有出海客户展品上飞机前必过合规预检；二是文化与传播，如在东南亚误用宗教忌色、在欧洲海报写「全国销量第一」无人理会，宽创在全球20多个国家组建了在地人供应商团队；三是合同与法律，如轻率签署独家代理协议被索赔千万。"
        },
        {
            "问题": "为什么中国企业在国际展会上位置总在角落？",
            "答案": "国际顶级展会有一套清晰的行业地位分配规则，好位置只对行业领军企业开放。主办方看三样东西：技术先进性、品牌声誉、往届参展表现，是一场积分游戏。宽创陪企业用三年、五年、十年时间从边缘走到中央，从36平米走到2000平米，从参与者变成被主办方主动邀请的人。"
        },
        {
            "问题": "全球会展哪个国家最强？出海重点国家和行业有哪些？",
            "答案": "德国是全球会展规模、影响力、场馆设施最好的国家，每年全球三分之二的顶级专业展会在德国举办。德、美、俄是出展重点国家，综合、纺织、机械是2025、2026年出国参展企业的前十大行业。以柏林IFA为例，全球3200多家参展商中中国占1200多家，超过三分之一。"
        },
        {
            "问题": "宽创展厅有哪些区域？参观路线是什么？",
            "答案": "参观路线依次为：一楼荣誉墙（铜牌墙区域）→ 魔墙区（公司介绍与发展历程）→ 负一楼人工智能展示馆（灯箱、移动屏中国经济与中企出海蓝图）→ 万里江河AI+文博展区 → 智联·万象AI科技展厅，结束后乘电梯前往二楼。"
        },
        {
            "问题": "万里江河展区有什么内容？",
            "答案": "万里江河以文明演进与多元一体为主线，包括：博物馆1.0-4.0进化论L幕联动、百件文物看中国与世界数字查询系统、远古回响沉浸影像、兵马俑AI数字交互区、张仲景望闻问切中医体验、龟兹壁画数字活化区、科举四宴动态影像、大运河漕运数字投影、名窑瓷器3D Mapping投影、清明上河图数字VR互动、盛世修典互动翻书、文明长河综合互动桌。"
        },
        {
            "问题": "智联万象AI科技展厅有什么内容？",
            "答案": "智联·万象AI科技展厅集中展示宽创前沿AI技术与软硬件研发成果，包括：城市客厅弧形大屏、殷墟妇好AI微短剧、数字孪生手影变动物、智能体平台裸眼3D、具身智能表情机器人、AI算力平台可视化大屏与问答终端、AI传动控制互动墙、AI数据采集八种传感器与激光雷达演示。"
        },
        {
            "问题": "博物馆4.0是什么意思？",
            "答案": "博物馆4.0时代不再只是摆文物，而是围绕人的体验来设计，让各年龄层都能通过AI更好地理解文化，例如用AI让商代乐器奏出曲子、和古人聊天、让古画动起来。"
        },
        {
            "问题": "2025年中国经济和出口数据是多少？",
            "答案": "2025年中国经济总量预计达140万亿人民币，占全球17%；出口额近27万亿，货物贸易总额连续多年全球第一。中国是全球唯一拥有联合国全部门类工业体系的国家：41个工业大类、207个中类、666个小类。自2020年起连续6年国际专利申请量全球第一，2025年专利申请量接近70万件，占全球近一半。"
        },
        {
            "问题": "宽创和哪些高校有合作？",
            "答案": "宽创注重产学研合作，与上海交大、财大、复旦等高校及行业协会形成合作，是干训班的研学基地；张东博士担任上海交通大学文创学院产业导师。"
        },
    ]

    import pandas as pd
    df = pd.DataFrame(faq)
    df.to_excel(faq_file, index=False, engine='openpyxl')
    print(f"   ✅ 已生成 {len(faq)} 条宽创问答")

# ============================================================
# 文档读取
# ============================================================

_TITLE_RE = re.compile(r"^(第[一二三四五六七八九十\d]+页|.{1,30}[：:])$")

def is_title(p, text):
    """标题判定：Heading 样式，或以「：」结尾的短行，或「第X页...」"""
    if p.style and p.style.name and p.style.name.lower().startswith("heading"):
        return True
    return len(text) <= 30 and bool(_TITLE_RE.match(text))

def load_docx(path):
    """docx → [(标题, 正文)]，按标题样式/标题特征识别章节"""
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
        if is_title(p, t):
            flush()
            cur_title = t.rstrip("：:")
        else:
            buf.append(t)
    flush()
    return chapters

def load_qa_xlsx(path):
    """xlsx → QA 对列表。列名支持 问题/答案 或 question/answer"""
    import pandas as pd
    df = pd.read_excel(path)
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
    return DEFAULT_CATEGORY   # 兜底默认（知识库现在只有宽创）

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
    print("📚 构建 RAG 索引（宽创国际）")
    print("=" * 60)

    # 0a. 配置自检
    if not Path(EMBED_MODEL).exists():
        print(f"❌ embedding 模型路径不存在:\n   {EMBED_MODEL}")
        print("   请打开 server4.py，把那边 EMBED_MODEL 那一行原样复制到本文件（两边必须一字不差）")
        return

    # 0b. 文档目录不存在则创建
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"❌ 无法创建文档目录 {DATA_DIR}: {e}")
        return
    print(f"📂 文档目录: {DATA_DIR}")

    # 1. 确保 FAQ 与词典存在
    ensure_broadmesse_faq()
    ensure_jieba_dict()

    load_jieba_dict()
    import jieba
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    import faiss

    # 2. 读取所有文档 → 条目(items)
    items = []
    n_qa, n_chunk = 0, 0

    for path in sorted(DATA_DIR.rglob("*")):
        if path.suffix.lower() not in (".docx", ".xlsx"):
            continue
        if path.name.startswith("~$"):
            continue
        cat = category_of(path)
        print(f"📄 读取 {path.name} → {cat}")

        if path.suffix.lower() == ".docx":
            chapters = load_docx(path)
            print(f"   识别到 {len(chapters)} 个章节")
            for ch in chapters:
                for i, c in enumerate(chunk_text(ch["text"])):
                    did = hashlib.md5(f"{path}#{ch['title']}#{i}".encode()).hexdigest()[:12]
                    # 正文前拼上章节标题，让「荣誉墙」「万里江河」等区域名参与检索
                    text = f"{ch['title']}\n{c}" if ch["title"] and ch["title"] != path.stem else c
                    items.append({
                        "id": did, "dedup_key": did,
                        "question": None,
                        "text": text,
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
        print(f"⚠️ 请确认「宽创展厅 讲解词.docx」「宽创国际企业荣誉一览表.docx」已放入: {DATA_DIR}")

    # 3. 多表示：QA 建 2 向量(纯问题 + 全文)，章节块建 1 向量
    reps, rep_to_item = [], []
    for idx, it in enumerate(items):
        if it["question"]:
            reps.append(it["question"])
            rep_to_item.append(idx)
            reps.append(it["text"])
            rep_to_item.append(idx)
        else:
            reps.append(it["text"])
            rep_to_item.append(idx)
    print(f"   多表示向量数: {len(reps)}（其中 {len(items)} 个条目）")

    # 4. gte 编码 + L2 归一化
    print(f"\n🧮 加载 embedding 模型: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    print("   编码中...")
    vecs = embedder.encode(reps, normalize_embeddings=True,
                           show_progress_bar=True, batch_size=32).astype("float32")

    # 5. FAISS
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    print(f"✅ FAISS 索引: {index.ntotal} 条向量, dim={dim}")

    # 6. BM25
    docs = [list(jieba.cut(it["text"])) for it in items]
    bm25 = BM25Okapi(docs)
    print(f"✅ BM25 语料: {len(docs)} 篇")

    # 7. 保存
    OUTPUT_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(OUTPUT_DIR / "faiss.index"))
    with open(OUTPUT_DIR / "items.json", "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / "faiss_map.json", "w", encoding="utf-8") as f:
        json.dump(rep_to_item, f)
    with open(OUTPUT_DIR / "bm25.pkl", "wb") as f:
        pickle.dump({"docs": docs}, f)

    print("\n" + "=" * 60)
    print("✅ 索引构建完成，已保存到 index/:")
    print("   faiss.index    稠密向量索引")
    print("   items.json     条目元数据（text/source/category/dedup_key）")
    print("   faiss_map.json 向量行 → 条目 映射")
    print("   bm25.pkl       BM25 分词语料")
    print("=" * 60)

if __name__ == "__main__":
    main()