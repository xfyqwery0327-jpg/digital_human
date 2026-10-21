@echo off
setlocal EnableExtensions
cd /d "%~dp0"
for %%I in ("%~dp0.") do set "NAME=%%~nxI"
set "SRC=%~dp0"
set "DEST=D:\projects\copy\%NAME%"
set "ARCHIVE=D:\projects\copy\%NAME%.7z"
set "XD=.git .venv .venv-kb .pytest_cache __pycache__ run .cursor .codegraph logs sessions out CosyVoice"

if "%~1"=="1" goto mode1
if "%~1"=="2" goto mode2
if "%~1"=="3" goto mode3
if /I "%~1"=="help" goto help
if "%~1"=="" goto menu
echo 未知模式：%~1
goto help

:menu
echo 复制目标：%DEST%
echo 1 第一次复制需要带上模型，文件
echo 2 只需要复制项目代码文件，并覆盖
echo 3 将copy好的文件打包7z
choice /C 123 /N /M "请输入 1、2 或 3："
if errorlevel 3 goto mode3
if errorlevel 2 goto mode2
if errorlevel 1 goto mode1
goto :eof

:help
echo 用法：copy.bat 1 ^| 2 ^| 3
echo 目标：D:\projects\copy\当前项目英文名
echo 1. 第一次复制需要带上模型，文件
echo 2. 只需要复制项目代码文件，并覆盖
echo 3. 将copy好的文件打包7z
echo CosyVoice 是目录联接，不会复制语音权重。
exit /b 0

:mode1
echo 模式 1：复制项目、文档、models 和运行所需数据
echo   %DEST%
if not exist "D:\projects\copy" mkdir "D:\projects\copy"
robocopy "%SRC%." "%DEST%" /E /COPY:DAT /R:2 /W:2 /MT:8 /XJ /XD %XD% /XF *.pyc *.rar
set "RC=%ERRORLEVEL%"
if %RC% GEQ 8 (
  echo 复制失败，robocopy 退出码 %RC%
  exit /b %RC%
)
echo 模式 1 完成。
exit /b 0

:mode2
echo 模式 2：只覆盖项目代码，不复制模型，也不删除目标里已有的模型
echo   %DEST%
if not exist "D:\projects\copy" mkdir "D:\projects\copy"
robocopy "%SRC%." "%DEST%" /E /IS /IT /COPY:DAT /R:2 /W:2 /MT:8 /XJ /XD %XD% models /XF *.pyc *.rar
set "RC=%ERRORLEVEL%"
if %RC% GEQ 8 (
  echo 复制失败，robocopy 退出码 %RC%
  exit /b %RC%
)
echo 模式 2 完成。目标里已有的模型保持不动。
exit /b 0

:mode3
if not exist "%DEST%\" (
  echo 还没有复制目录：%DEST%
  echo 请先运行 copy.bat 1 或 copy.bat 2
  exit /b 1
)
call :find_7z
if not defined SEVEN (
  echo 未找到 7z.exe。请安装 7-Zip。
  exit /b 1
)
echo 模式 3：打包 %ARCHIVE%
if exist "%ARCHIVE%" del /f "%ARCHIVE%"
"%SEVEN%" a -t7z -mx=3 -y "%ARCHIVE%" "%DEST%"
if errorlevel 1 (
  echo 打包失败。
  exit /b 1
)
echo 模式 3 完成。
exit /b 0

:find_7z
set "SEVEN="
if exist "D:\Program Files\7-Zip\7z.exe" set "SEVEN=D:\Program Files\7-Zip\7z.exe"
if not defined SEVEN if exist "C:\Program Files\7-Zip\7z.exe" set "SEVEN=C:\Program Files\7-Zip\7z.exe"
if not defined SEVEN if exist "C:\Program Files (x86)\7-Zip\7z.exe" set "SEVEN=C:\Program Files (x86)\7-Zip\7z.exe"
if defined SEVEN exit /b 0
for /f "delims=" %%P in ('where 7z 2^>nul') do (
  if not defined SEVEN set "SEVEN=%%P"
)
exit /b 0
