@echo off
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
if "%~1"=="6" goto cv3
if "%~1"=="7" goto cv2
if "%~1"=="8" goto ttsqwen
if "%~1"=="9" goto restart
if /I "%~1"=="4b" goto llm4
if /I "%~1"=="8b" goto llm8
if /I "%~1"=="llm" goto llmarg
if /I "%~1"=="cv" goto cvarg
if /I "%~1"=="cv2" goto cv2
if /I "%~1"=="cv3" goto cv3
if /I "%~1"=="cosyvoice2" goto cv2
if /I "%~1"=="cosyvoice3" goto cv3
if /I "%~1"=="tts" goto ttsarg
if /I "%~1"=="qwen" goto ttsqwen
if /I "%~1"=="qwen3" goto ttsqwen
echo 未知参数：%~1
echo.
set "RC=1"
goto showhelp

:menu
set "FROM_MENU=1"
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy\switch.ps1" -Action saved
echo.
echo 1 开启  按上次配置启动讲解模型 8080 和讲解员 8848
echo 2 关闭
echo 3 状态
echo 4 改用 Qwen3-4B，并启动讲解模型
echo 5 改用 Qwen3-8B，并启动讲解模型
echo 6 改用 CosyVoice3
echo 7 改用 CosyVoice2
echo 8 改用 Qwen3-TTS
echo 9 重启讲解员
echo 改用语音只写配置，不会重启。要换上正在跑的讲解员，再选 9。
choice /C 123456789 /N /M "请选择:"
if errorlevel 9 goto restart
if errorlevel 8 goto ttsqwen
if errorlevel 7 goto cv2
if errorlevel 6 goto cv3
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
echo   switch.bat cv 3 ^| cv 2
echo   switch.bat tts qwen ^| tts cv
echo 同一时间只开一个讲解模型。4 是 Qwen3-4B，8 是 Qwen3-8B。
echo 6 或 cv 3 改用 CosyVoice3，7 或 cv 2 改用 CosyVoice2，8 或 tts qwen 改用 Qwen3-TTS。
echo 改用只记下配置。9 或 restart 才重启讲解员，只动 8848，不动 8080。
echo 数字 2 是关闭，不是 CosyVoice2。
echo 开机自启读 deploy\last-start.txt，以最后一次这里启动的配置为准。
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

:cv3
set "ACT=cv"
set "WHICH=3"
goto run

:cv2
set "ACT=cv"
set "WHICH=2"
goto run

:ttsqwen
set "ACT=tts"
set "WHICH=qwen"
goto run

:llmarg
set "ACT=llm"
set "WHICH=%~2"
goto run

:cvarg
set "ACT=cv"
set "WHICH=%~2"
goto run

:ttsarg
set "ACT=tts"
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
