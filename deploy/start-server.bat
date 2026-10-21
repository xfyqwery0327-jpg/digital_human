@echo off
rem 双击或计划任务可调用。默认后台启动。
rem 前台看日志：  start-server.bat -Foreground
setlocal
cd /d "%~dp0\.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-server.ps1" %*
endlocal
