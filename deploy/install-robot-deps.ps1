# 给展项服务器的 conda 环境 robot 补语音依赖。
# 先装 qwen-tts，再强制换回 onnxruntime-gpu==1.23.2，不升级 torch。
#
#   powershell -ExecutionPolicy Bypass -File C:\digital_human\deploy\install-robot-deps.ps1
#   deploy\install-robot-deps.bat

param(
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "server9\app.py"))) {
    $Root = "C:\digital_human"
}
if (-not (Test-Path (Join-Path $Root "server9\app.py"))) {
    throw ("server9\app.py not found, Root=" + $Root)
}

function Find-RobotPython {
    $paths = New-Object System.Collections.Generic.List[string]
    if ($env:CONDA_PREFIX) {
        $paths.Add((Join-Path $env:CONDA_PREFIX "python.exe"))
    }
    if ($env:USERPROFILE) {
        $paths.Add((Join-Path $env:USERPROFILE "miniconda3\envs\robot\python.exe"))
        $paths.Add((Join-Path $env:USERPROFILE "anaconda3\envs\robot\python.exe"))
        $paths.Add((Join-Path $env:USERPROFILE "Miniconda3\envs\robot\python.exe"))
        $paths.Add((Join-Path $env:USERPROFILE "Anaconda3\envs\robot\python.exe"))
    }
    $paths.Add("C:\Users\BroadMesse\miniconda3\envs\robot\python.exe")
    $paths.Add("C:\Users\BroadMesse\Miniconda3\envs\robot\python.exe")
    $paths.Add("C:\ProgramData\miniconda3\envs\robot\python.exe")
    $paths.Add("C:\ProgramData\anaconda3\envs\robot\python.exe")
    foreach ($p in $paths) {
        if ($p -and (Test-Path $p)) { return $p }
    }
    return $null
}

if (-not $Python) { $Python = Find-RobotPython }
if (-not $Python -or -not (Test-Path $Python)) {
    throw "找不到 robot 环境的 python.exe。先建 conda 环境 robot，或用 -Python 指定。"
}

Write-Host ("python=" + $Python)
Write-Host "1/2 pip install qwen-tts"
& $Python -m pip install qwen-tts
if ($LASTEXITCODE -ne 0) { throw "qwen-tts 安装失败" }

Write-Host "2/2 pip install --force-reinstall --no-deps onnxruntime-gpu==1.23.2"
& $Python -m pip install --force-reinstall --no-deps onnxruntime-gpu==1.23.2
if ($LASTEXITCODE -ne 0) { throw "onnxruntime-gpu 1.23.2 安装失败" }

Write-Host "核对 qwen_tts 与 onnxruntime"
& $Python -c "import qwen_tts, onnxruntime as ort; ver=ort.__version__; assert ver=='1.23.2', ver; print('qwen_tts ok'); print('onnxruntime', ver, ort.get_device())"
if ($LASTEXITCODE -ne 0) { throw "导入核对失败。onnxruntime 必须是 1.23.2，且能 import qwen_tts。" }

Write-Host "robot 环境追加包已装好。改过包之后执行 switch.bat restart，只重启 8848。"
