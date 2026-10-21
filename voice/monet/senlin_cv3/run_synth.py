# -*- coding: utf-8 -*-
"""用 Fun-CosyVoice3-0.5B 零样本，把 6 段参考音分别合成 3 篇莫奈讲稿。"""
import sys
import json
import time
from pathlib import Path

import _locale
_locale._getdefaultlocale = lambda *args: ["en_US", "utf8"]

ROOT = Path(r"D:\projects\CosyVoice")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "Matcha-TTS"))

import numpy as np
import soundfile as sf
import torch
import torchaudio


def _load_audio(path, *args, **kwargs):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(np.ascontiguousarray(data.T)), int(sr)


def _save_audio(path, src, sample_rate, *args, **kwargs):
    if hasattr(src, "detach"):
        src = src.detach().cpu()
    arr = src.numpy()
    if arr.ndim == 2:
        arr = arr.T
    sf.write(str(path), arr, int(sample_rate))


torchaudio.load = _load_audio
torchaudio.save = _save_audio

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402

HERE = Path(r"E:\projects\digital_human\voice\monet\senlin_cv3")
CLIP_DIR = HERE / "clips"
OUT_DIR = HERE / "synth"
LOG = HERE / "progress.txt"


def log(msg: str):
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
MODEL = r"D:\projects\CosyVoice\pretrained_models\Fun-CosyVoice3-0.5B"
PREFIX = "You are a helpful assistant.<|endofprompt|>"

# 按整段录音的时间戳切到各窗口，并把「隆众→隆重」。
PROMPTS = {
    "00_06": "你好，欢迎大家来到我们大美中山隆",
    "03_09": "大家来到我们大美中山隆重登场的光影魔法森林。",
    "06_12": "隆重登场的光影魔法森林。我们是这片森林的守护者。",
    "09_15": "我们是这片森林的守护者。奇幻的魔法森林一年一度",
    "12_18": "奇幻的魔法森林一年一度的狂欢派对即将开始。大家要",
    "18_21": "做好准备。接下来故事将在魔法",
}

SCRIPTS = {
    "1": "我是克劳德·莫奈，大家爱叫我“印象派之父”。我这一辈子，大概画了两千多幅油画、五百多件素描。其中《睡莲》我前后画了180多幅；另外还有《干草堆》25幅、《鲁昂大教堂》30多幅、《白杨树》24幅。我喜欢对同一个场景反复画，比如干草堆、教堂、伦敦的雾，就是为了捕捉不同时辰光线的变化。",
    "2": "我一生追逐光与瞬间。《印象·日出》捕捉勒阿弗尔港清晨的橙红与灰蓝，让“印象派”由此得名；而《睡莲》是我持续创作的一个庞大系列，其中最常被提及的一幅是《睡莲池》水面倒影与天空交融，几乎成了我晚年的象征。《干草堆》《白杨树》《鲁昂大教堂》用不同时辰的光让寻常景物变幻；《日本桥》《撑阳伞的女人》《圣拉扎尔火车站》则留住花园、风与蒸汽中的色彩。我画的不是物体，而是光在空气里的呼吸。",
    "3": "我1900年画的《睡莲池》，现藏芝加哥艺术学院。吉维尼池塘里，我种下睡莲，本想画植物，却被水面变幻的光迷住，于是绕到侧面，不画日本桥，把水面和植被都收进画里。技法上，先用稀薄棕绿薄涂铺底，干后以厚笔触叠加。睡莲与叶用含铅白厚涂堆出，定住再加彩色；常一笔蘸多色，不调匀，让颜色自己说话。画布纬纱粗，我垂直涂色，颜料吃进纹理，水面便有了晃动波纹。",
}

LABELS = {
    "00_06": "0–6 秒",
    "03_09": "3–9 秒",
    "06_12": "6–12 秒",
    "09_15": "9–15 秒",
    "12_18": "12–18 秒",
    "18_21": "18–21 秒",
}
TITLES = {
    "1": "讲稿 1 · 作品数量与反复画",
    "2": "讲稿 2 · 追逐光与瞬间",
    "3": "讲稿 3 · 1900 年《睡莲池》",
}


def synth_one(model, text, prompt, wav_path):
    chunks = []
    for out in model.inference_zero_shot(
            text, PREFIX + prompt, str(wav_path), stream=False, speed=1.0):
        chunks.append(out["tts_speech"].detach().cpu())
    if not chunks:
        raise RuntimeError("no audio")
    gap = torch.zeros(1, int(model.sample_rate * 0.12))
    parts = []
    for i, c in enumerate(chunks):
        if c.dim() == 1:
            c = c.unsqueeze(0)
        parts.append(c)
        if i != len(chunks) - 1:
            parts.append(gap)
    return torch.cat(parts, dim=1)


def write_index():
    cards = []
    for sid, title in TITLES.items():
        rows = []
        for key, label in LABELS.items():
            rel = f"synth/{key}/s{sid}.wav"
            rows.append(
                f'<div class="row"><div class="lab">{label}</div>'
                f'<audio controls preload="none" src="{rel}"></audio></div>'
            )
        cards.append(f'<section><h2>{title}</h2>{"".join(rows)}</section>')
    refs = []
    for key, label in LABELS.items():
        refs.append(
            f'<div class="row"><div class="lab">{label}</div>'
            f'<div class="sub">{PROMPTS[key]}</div>'
            f'<audio controls preload="none" src="clips/{key}.wav"></audio></div>'
        )
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>莫奈森林参考音对比</title>
<style>
body {{ font-family: "Segoe UI", "PingFang SC", sans-serif; margin: 32px; background: #f6f3ee; color: #1c1915; }}
h1 {{ font-weight: 600; }}
h2 {{ margin: 28px 0 12px; font-size: 18px; }}
section {{ background: #fff; border-radius: 12px; padding: 8px 20px 16px; margin-bottom: 18px; }}
.row {{ display: grid; grid-template-columns: 110px 1fr; gap: 8px 16px; align-items: center; padding: 8px 0; border-top: 1px solid #eee; }}
.lab {{ font-variant-numeric: tabular-nums; }}
.sub {{ grid-column: 2; color: #666; font-size: 13px; margin-top: -6px; }}
audio {{ width: 100%; }}
</style>
</head>
<body>
<h1>莫奈 · 森林录音参考音对比</h1>
<p>同一篇讲稿横着听，比较哪一段参考音更像这个人。上面是合成结果，最下面是原始截取。</p>
{''.join(cards)}
<section><h2>原始参考音</h2>{''.join(refs)}</section>
</body>
</html>
"""
    (HERE / "index.html").write_text(html, encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (HERE / "prompt_text.json").write_text(
        json.dumps(PROMPTS, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.write_text("", encoding="utf-8")
    log("loading " + MODEL)
    t0 = time.time()
    model = AutoModel(model_dir=MODEL, fp16=True)
    log(f"loaded {time.time()-t0:.1f}s sr={model.sample_rate}")
    for key, prompt in PROMPTS.items():
        wav = CLIP_DIR / f"{key}.wav"
        dest_dir = OUT_DIR / key
        dest_dir.mkdir(parents=True, exist_ok=True)
        for sid, text in SCRIPTS.items():
            dest = dest_dir / f"s{sid}.wav"
            if dest.exists() and dest.stat().st_size > 10000:
                log(f"skip {key} s{sid}")
                continue
            log(f"synth clip={key} script={sid}")
            t1 = time.time()
            audio = synth_one(model, text, prompt, wav)
            torchaudio.save(str(dest), audio, model.sample_rate)
            dur = audio.shape[-1] / model.sample_rate
            log(f"wrote {key}/s{sid}.wav {dur:.1f}s in {time.time()-t1:.1f}s")
    write_index()
    log("done")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log(traceback.format_exc())
        raise
