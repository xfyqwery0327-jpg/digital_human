@chcp 65001 >nul
@echo off
if /i "%~1"=="__cp__" goto args_ready
cmd /c ""%~f0" __cp__ %*"
exit /b %ERRORLEVEL%
:args_ready
shift
setlocal EnableExtensions
cd /d "%~dp0"
set "RC=0"
set "FROM_MENU="

if "%~1"=="" goto menu
if /I "%~1"=="on" goto on
if /I "%~1"=="off" goto off
if /I "%~1"=="status" goto status
if /I "%~1"=="restart" goto restart
if /I "%~1"=="help" goto help
if "%~1"=="1" goto on
if "%~1"=="2" goto off
if "%~1"=="3" goto status
if "%~1"=="4" goto llm4
if "%~1"=="5" goto llm8
if "%~1"=="6" goto restart
if "%~1"=="9" goto restart
if /I "%~1"=="4b" goto llm4
if /I "%~1"=="8b" goto llm8
if /I "%~1"=="llm" goto llmarg
if /I "%~1"=="7" goto retired
if /I "%~1"=="8" goto retired
if /I "%~1"=="cv" goto retired
if /I "%~1"=="cv2" goto retired
if /I "%~1"=="cv3" goto retired
if /I "%~1"=="cosyvoice2" goto retired
if /I "%~1"=="cosyvoice3" goto retired
if /I "%~1"=="tts" goto retired
if /I "%~1"=="qwen" goto retired
if /I "%~1"=="qwen3" goto retired
echo 未知参数：%~1
echo.
set "RC=1"
goto showhelp

:menu
set "FROM_MENU=1"
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy\switch.ps1" -Action saved
echo.
echo 1 开启  按上次的讲解模型启动 8080 和讲解员 8848
echo 2 关闭
echo 3 状态
echo 4 改用 Qwen3-4B，并启动讲解模型
echo 5 改用 Qwen3-8B，并启动讲解模型
echo 6 重启讲解员
echo 语音固定走 qwen-tts-service，地址 http://127.0.0.1:8812。
echo 进程内 Qwen3-TTS、SGLang WebSocket、CosyVoice 不再切换。
choice /C 123456 /N /M "请选择:"
if errorlevel 6 goto restart
if errorlevel 5 goto llm8
if errorlevel 4 goto llm4
if errorlevel 3 goto status
if errorlevel 2 goto off
goto on

:help
set "RC=0"
goto showhelp

:showhelp
echo 用法：
echo   switch.bat              菜单
echo   switch.bat on ^| off ^| status ^| restart
echo   switch.bat llm 4 ^| llm 8
echo   switch.bat 4b ^| 8b
echo 同一时间只开一个讲解模型。
echo 4 或 llm 4 是 Qwen3-4B，文件 models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf。
echo 5 或 llm 8 是 Qwen3-8B，文件 models\Qwen3-8B-Q4_K_M.gguf。
echo 语音固定走独立服务 qwen-tts-service，地址 http://127.0.0.1:8812。
echo 进程内 Qwen3-TTS、SGLang WebSocket、CosyVoice 已停用，菜单里不能再选。
echo 6、restart、9 重启讲解员，只动 8848，不动 8080。
echo 数字 2 是关闭。数字 4 和 5 才是讲解模型。
echo 开机自启读 deploy\last-start.txt 里的讲解模型。语音不再跟着旧记录切换。
goto finish

:retired
echo 进程内 Qwen3-TTS、SGLang WebSocket、CosyVoice 已经停用。
echo 讲解员固定走 qwen-tts-service，地址 http://127.0.0.1:8812。
echo 这里只切换讲解模型 Qwen3-4B 或 Qwen3-8B。
set "RC=1"
goto finish

:on
set "ACT=on"
set "WHICH="
goto run

:off
set "ACT=off"
set "WHICH="
goto run

:status
set "ACT=status"
set "WHICH="
goto run

:restart
set "ACT=restart"
set "WHICH="
goto run

:llm4
set "ACT=llm"
set "WHICH=4"
goto run

:llm8
set "ACT=llm"
set "WHICH=8"
goto run

:llmarg
set "ACT=llm"
set "WHICH=%~2"
goto run

:run
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy\switch.ps1" -Action "%ACT%" -Which "%WHICH%"
set "RC=%ERRORLEVEL%"
goto finish

:finish
if defined FROM_MENU (
  echo.
  pause
)
exit /b %RC%
