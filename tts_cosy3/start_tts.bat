@echo off
rem 独立 TTS 进程。不要和讲解 / LLM 用同一个 python 进程。
rem 先开本窗口，等日志出现「健康检查通过」，再开机器人服务。
cd /d "%~dp0\.."
if not defined PYTHON set "PYTHON=python"
set "TTS_HOST=127.0.0.1"
set "TTS_PORT=8810"
set "TTS_DTYPE=fp16"
set "TTS_INT8=1"
set "TTS_INT8_SCOPE=llm"
set "TTS_VRAM_LIMIT_MB=4096"
set "TTS_IDLE_SEC=10"
set "TTS_STUCK_SEC=180"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo TTS 目录 %CD%
echo 端口 %TTS_PORT%  显存阈值 %TTS_VRAM_LIMIT_MB% MiB
"%PYTHON%" -m tts_cosy3.supervisor
pause
