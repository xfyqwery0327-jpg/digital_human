import ctypes, os, sys

paths = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
    r"C:\Users\BroadMesse\miniconda3\envs\robot\Lib\site-packages\nvidia\cudnn\bin"
]
for p in paths:
    if os.path.exists(p):
        os.add_dll_directory(p)
        os.environ["PATH"] = p + ";" + os.environ.get("PATH", "")

dll_path = r"C:\Users\BroadMesse\miniconda3\envs\robot\Lib\site-packages\nvidia\cudnn\bin\cudnn64_8.dll"
print(f"文件物理存在: {os.path.exists(dll_path)}")

# 尝试加载，并捕获具体错误码
try:
    ctypes.CDLL(dll_path)
    print("✅ cudnn64_8.dll 加载成功")
except OSError as e:
    print(f"❌ 加载失败: {e}")
    print("常见原因: 缺少 zlibwapi.dll 或 cublas64_12.dll 等依赖")

# 检查 zlibwapi
for name in ["zlibwapi", "cublas64_12", "cublasLt64_12", "cudart64_12"]:
    try:
        ctypes.CDLL(name)
        print(f"  ✅ {name}.dll")
    except OSError:
        print(f"  ❌ {name}.dll 缺失")