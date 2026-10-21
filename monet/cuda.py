import sys
from pathlib import Path
import time

COSYVOICE_PATH = r"C:\ro\CosyVoice"
COSYVOICE_MODEL = r"C:\ro\CosyVoice\pretrained_models\CosyVoice-300M-SFT"

sys.path.insert(0, COSYVOICE_PATH)
sys.path.insert(0, str(Path(COSYVOICE_PATH) / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice

print("🔍 加载模型...")
cosyvoice_model = CosyVoice(COSYVOICE_MODEL)

print("\n📍 检查各模块设备:")
for attr in ['flow', 'llm', 'hift', 'model']:
    if hasattr(cosyvoice_model, attr):
        obj = getattr(cosyvoice_model, attr)
        if hasattr(obj, 'parameters'):
            try:
                device = next(obj.parameters()).device
                print(f"  {attr:10s} -> {device}")
            except StopIteration:
                print(f"  {attr:10s} -> 没有参数")

print("\n⏱️  测试合成速度...")
start = time.time()
for output in cosyvoice_model.inference_sft("今天天气真不错，莫奈的画展非常精彩。", spk_id="中文男"):
    pass
elapsed = time.time() - start
print(f"✅ 合成耗时: {elapsed:.2f} 秒")