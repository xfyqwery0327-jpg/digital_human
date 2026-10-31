# Conda env digital_human_server: Python 3.10 + CUDA torch for the narrator.
# The old env digital_human is Python 3.14. Check it, do not modify it.
#   powershell -File deploy\server-env.ps1 -Action check
#   powershell -File deploy\server-env.ps1 -Action ensure
param(
    [ValidateSet("ensure", "check")]
    [string]$Action = "ensure"
)

$ErrorActionPreference = "Stop"
$EnvName = "digital_human_server"
$OldName = "digital_human"
$Checker = Join-Path $PSScriptRoot "check_server_env.py"

function Find-EnvPython([string]$name) {
    $candidates = @()
    if ($env:USERPROFILE) {
        $candidates += (Join-Path $env:USERPROFILE (".conda\envs\" + $name + "\python.exe"))
        $candidates += (Join-Path $env:USERPROFILE ("miniconda3\envs\" + $name + "\python.exe"))
        $candidates += (Join-Path $env:USERPROFILE ("anaconda3\envs\" + $name + "\python.exe"))
        $candidates += (Join-Path $env:USERPROFILE ("Miniconda3\envs\" + $name + "\python.exe"))
        $candidates += (Join-Path $env:USERPROFILE ("Anaconda3\envs\" + $name + "\python.exe"))
    }
    $candidates += ("D:\conda-envs\" + $name + "\python.exe")
    $candidates += ("C:\Users\BroadMesse\miniconda3\envs\" + $name + "\python.exe")
    $candidates += ("C:\ProgramData\miniconda3\envs\" + $name + "\python.exe")
    $candidates += ("C:\ProgramData\anaconda3\envs\" + $name + "\python.exe")
    # 本机 conda base 可能在 D:\anaconda（conda create 默认把环境建在 base\envs 下）
    $bases = @()
    try {
        $cb = (& conda info --base 2>$null | Select-Object -First 1)
        if ($cb) { $bases += ($cb + "").Trim() }
    } catch { }
    $bases += @("D:\anaconda", "D:\anaconda3", "C:\anaconda", "C:\ProgramData\anaconda3")
    foreach ($b in $bases) {
        if ($b) { $candidates += (Join-Path $b ("envs\" + $name + "\python.exe")) }
    }
    foreach ($p in $candidates) {
        if ($p -and (Test-Path $p)) { return $p }
    }
    return $null
}

function Invoke-EnvCheck([string]$python) {
    $log = Join-Path $env:TEMP "digital-human-env-check.txt"
    & $python $Checker > $log 2>&1
    $code = $LASTEXITCODE
    if ($null -eq $code) { $code = 1 }
    if (Test-Path $log) { Get-Content -Path $log | Out-Host }
    return [int]$code
}

function Install-ServerPins([string]$python) {
    Write-Host ("install pins " + $python)
    & $python -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) { throw "torch install failed" }
    & $python -m pip install "numpy==1.26.4" "transformers==4.57.3" "qwen-tts==0.1.1"
    if ($LASTEXITCODE -ne 0) { throw "transformers install failed" }
    & $python -m pip install "fastapi==0.141.1" "uvicorn==0.53.0" "uvicorn[standard]" "websockets==14.1" "pydantic==2.13.5" "requests==2.34.2" "PyYAML==6.0.3" "soundfile==0.12.1" "hyperpyyaml==1.2.3" "modelscope==1.40.1" "safetensors==0.8.0" "jieba==0.42.1" "rank-bm25==0.2.2" "faiss-cpu==1.15.0" "sentence-transformers==6.0.1" "onnxruntime==1.18.0"
    if ($LASTEXITCODE -ne 0) { throw "runtime deps install failed" }
    & $python -m pip install "openai-whisper==20231117" --no-build-isolation --no-deps
    if ($LASTEXITCODE -ne 0) { throw "whisper install failed" }
    & $python -m pip install "tiktoken==0.14.0" "inflect==7.3.1" "wetext==0.1.8" "conformer==0.3.2" "diffusers==0.29.0" "hydra-core==1.3.2" "omegaconf==2.3.0" "antlr4-python3-runtime==4.9.3" "lightning==2.6.6" "pytorch-lightning==2.6.6" "librosa==0.10.2" "numba==0.67.0" "llvmlite==0.49.0" "x-transformers==2.0.0" "pyworld==0.3.5" "matplotlib==3.10.9" "onnx==1.16.0" "kaldifst==1.8.0" "einops==0.8.0" "pyarrow==25.0.1" "sse-starlette==3.5.0" "pydantic-settings==2.15.0" "starlette-context==0.3.6"
    if ($LASTEXITCODE -ne 0) { throw "cosyvoice deps install failed" }
    # llama-cpp-python 仅当讲解员本机自托管 LLM 时才必需；当前架构 LLM 走独立
    # 8080 进程，server9 通过 HTTP 调用 OpenAI 接口，自身并不 import 它，因此
    # 该包缺失不会阻塞讲解员启动。PyPI 全平台只有 sdist（无 Windows wheel），
    # conda-forge 在本机网络下求解极易卡死（曾卡 20+ 分钟），故做如下处理：
    # 已装则跳过；否则限时 600s 尝试；超时/失败都不阻塞讲解员启动。
    $haveLLM = $false
    try { & $python -c "import llama_cpp" 2>$null; $haveLLM = $true } catch { }
    if ($haveLLM) {
        Write-Host "llama-cpp-python already present, skip"
    } else {
        Write-Host "attempting conda-forge llama-cpp-python (best-effort, 600s timeout)"
        $p = Start-Process -FilePath "conda" -ArgumentList @(
            "install", "-n", $EnvName, "-c", "conda-forge",
            "llama-cpp-python=0.3.35", "--freeze-installed", "-y"
        ) -NoNewWindow -PassThru
        if (-not $p.Wait(600000)) {
            Write-Host "conda solve timed out; killing and continuing (narrator starts via external 8080)"
            try { $p.Kill() } catch { }
        } elseif ($p.ExitCode -ne 0) {
            Write-Host "llama-cpp-python install failed; narrator can still start, LLM may need external 8080"
        }
    }
    & $python -m pip install "transformers==4.57.3" "numpy==1.26.4" "onnxruntime==1.18.0"
    if ($LASTEXITCODE -ne 0) { throw "pin restore failed" }
    & $python -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) { throw "torch pin restore failed" }
}

function Report-OldEnv {
    $old = Find-EnvPython $OldName
    if (-not $old) {
        Write-Host ("old env " + $OldName + " not found")
        return
    }
    Write-Host ("check old env " + $OldName + " " + $old)
    $code = Invoke-EnvCheck $old
    if ($code -eq 0) {
        Write-Host ("old env " + $OldName + " matches")
    } else {
        Write-Host ("old env " + $OldName + " does not match, launcher will not use it")
    }
}

Report-OldEnv

$python = Find-EnvPython $EnvName
if (-not $python) {
    if ($Action -eq "check") {
        Write-Host ("env " + $EnvName + " missing")
        exit 1
    }
    Write-Host ("conda create " + $EnvName + " python=3.10")
    conda create -n $EnvName python=3.10 pip -y
    if ($LASTEXITCODE -ne 0) { throw "conda create failed" }
    $python = Find-EnvPython $EnvName
    if (-not $python) { throw "python.exe missing after conda create" }
    Install-ServerPins $python
}

Write-Host ("check " + $EnvName + " " + $python)
$code = Invoke-EnvCheck $python
if ($code -ne 0 -and $Action -eq "ensure") {
    Write-Host ("env " + $EnvName + " mismatch, installing pins")
    Install-ServerPins $python
    $code = Invoke-EnvCheck $python
}
if ($code -ne 0) {
    Write-Host ($EnvName + " check failed, refuse to start")
    exit $code
}
Write-Host ("env ok " + $python)
exit 0
