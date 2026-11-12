param(
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "server9\app.py"))) {
    $Root = "C:\digital_human"
}
if (-not (Test-Path (Join-Path $Root "server9\app.py"))) {
    throw ("server9\app.py not found, Root=" + $Root)
}

$Logs = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$bootLog = Join-Path $Logs "autostart.log"
$outLog = Join-Path $Logs "digital-human.out.log"

function Write-Boot([string]$msg) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $bootLog -Value ("[" + $stamp + "] " + $msg)
    Write-Host $msg
}

function Test-Port8848 {
    try {
        $c = New-Object System.Net.Sockets.TcpClient
        $iar = $c.BeginConnect("127.0.0.1", 8848, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne(800)
        if (-not $ok) { $c.Close(); return $false }
        $c.EndConnect($iar)
        $c.Close()
        return $true
    } catch {
        return $false
    }
}

function Wait-QwenTts([string]$BaseUrl, [int]$Seconds) {
    # 8812 要等模型进显存才监听。8848 只在自己启动时探一次，超时 2 秒。
    # 这里先等到 /health 成功，再拉起讲解员，避免 tts_error 被写成「口播入口未就绪」。
    $url = $BaseUrl.TrimEnd("/") + "/health"
    Write-Boot ("waiting for " + $url + " up to " + $Seconds + "s")
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
            if ($resp.StatusCode -eq 200) {
                $payload = $resp.Content | ConvertFrom-Json
                if ($payload.ok -eq $true -or $payload.tts -eq $true) {
                    $names = ""
                    if ($payload.speakers) { $names = (@($payload.speakers) -join ",") }
                    Write-Boot ("qwen-tts-service ready speakers=" + $names)
                    return $true
                }
            }
        } catch {
        }
        Start-Sleep -Seconds 2
    }
    Write-Boot ("qwen-tts-service not ready after " + $Seconds + "s")
    return $false
}

function Find-RobotPython {
    # 本机讲解员在 miniconda 的 robot。digital_human_server 只有 torch，缺 fastapi。
    $names = @("robot", "digital_human_server")
    $roots = @()
    if ($env:USERPROFILE) {
        $roots += (Join-Path $env:USERPROFILE ".conda\envs")
        $roots += (Join-Path $env:USERPROFILE "miniconda3\envs")
    }
    $roots += @(
        "C:\Users\BroadMesse\miniconda3\envs",
        "D:\anaconda\envs",
        "D:\conda-envs"
    )
    $seen = @{}
    foreach ($name in $names) {
        foreach ($root in $roots) {
            $py = Join-Path $root ($name + "\python.exe")
            if ($seen.ContainsKey($py)) { continue }
            $seen[$py] = $true
            if (-not (Test-Path $py)) { continue }
            & cmd.exe /c "`"$py`" -c `"import fastapi,uvicorn,faiss,requests`" >nul 2>nul"
            if ($LASTEXITCODE -eq 0) { return $py }
        }
    }
    return $null
}

try {
    Write-Boot ("script start Root=" + $Root + " user=" + $env:USERNAME)

    if (Test-Port8848) {
        Write-Boot "port 8848 already listening, skip"
        exit 0
    }

    $env:PYTHONUNBUFFERED = "1"
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONPATH = Join-Path $Root "server9"
    # 不同 batch / 句长的显存块尽量接在同一段上，避免每种尺寸各占一块、只升不降。
    if (-not $env:PYTORCH_CUDA_ALLOC_CONF) {
        $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
    }
    $env:MONET_HOST = "0.0.0.0"
    $env:MONET_PORT = "8848"
    if (-not $env:COSYVOICE_PATH) {
        foreach ($c in @(
            (Join-Path $Root "CosyVoice"),
            "D:\projects\CosyVoice",
            "E:\projects\Robot_Intelligent_Voice_System\CosyVoice",
            "C:\ro\CosyVoice"
        )) {
            if (Test-Path $c) { $env:COSYVOICE_PATH = $c; break }
        }
        if (-not $env:COSYVOICE_PATH) { $env:COSYVOICE_PATH = "C:\ro\CosyVoice" }
    }
    # 不按 last-start 选择 CosyVoice 权重。讲解口播只走 qwen-tts-service。
    if (Test-Path Env:COSYVOICE_MODEL) {
        Remove-Item Env:COSYVOICE_MODEL
    }

    # 语音固定走独立 qwen-tts-service。不读 last-start 的 tts= / cosyvoice=。
    # 进程内 Qwen3-TTS、SGLang WebSocket、CosyVoice 不再作为启动选项。
    $env:TTS_ENGINE = "qwen3"
    $env:TTS_QWEN_REMOTE = "1"
    if (-not $env:TTS_QWEN_URL) {
        $env:TTS_QWEN_URL = "http://127.0.0.1:8812"
    }
    if (Test-Path Env:TTS_BACKEND) {
        Remove-Item Env:TTS_BACKEND
    }

    $Python = Find-RobotPython
    if (-not $Python) {
        Write-Boot "python not found or digital_human_server version check failed"
        throw "python not found, conda env digital_human_server is missing or versions do not match"
    }
    $CondaEnv = Split-Path $Python
    $env:CONDA_PREFIX = $CondaEnv
    $extras = @(
        (Join-Path $CondaEnv "Scripts"),
        (Join-Path $CondaEnv "Library\bin"),
        (Join-Path $CondaEnv "Library\usr\bin"),
        (Join-Path $CondaEnv "Library\mingw-w64\bin"),
        (Join-Path $CondaEnv "Lib\site-packages\nvidia\cudnn\bin"),
        (Join-Path $CondaEnv "Lib\site-packages\nvidia\cublas\bin"),
        (Join-Path $CondaEnv "Lib\site-packages\torch\lib"),
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
    ) | Where-Object { $_ -and (Test-Path $_) }
    $env:PATH = ($CondaEnv + ";" + ($extras -join ";") + ";" + $env:PATH)
    Write-Boot ("python=" + $Python + " tts=qwen-tts-service url=" + $env:TTS_QWEN_URL + " remote=" + $env:TTS_QWEN_REMOTE + " alloc=" + $env:PYTORCH_CUDA_ALLOC_CONF)
    $ttsReady = Wait-QwenTts $env:TTS_QWEN_URL 180
    if (-not $ttsReady) {
        Write-Boot "start 8848 without a ready 8812; health tts_error will keep the probe failure"
    }

    Set-Location $Root
    Add-Content -Path $outLog -Value ("`r`n==== " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " start ====`r`n")
    Write-Boot ("starting uvicorn log=" + $outLog)

    # 用 Start-Process 独立后台拉起 uvicorn，避免 5.1 把 python/torch 的 stderr
    # 当作 NativeCommandError 而抛错（cmd /c 阻塞方式在开机场景下会因 torch 的
    # redirects NOTE 触发 $ErrorActionPreference=Stop 直接退出，导致 8848 起不来）。
    $p = Start-Process -FilePath $Python `
        -ArgumentList @("-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8848", "--log-level", "info") `
        -WorkingDirectory $Root -WindowStyle Hidden `
        -RedirectStandardOutput $outLog -RedirectStandardError ($outLog + ".err") -PassThru
    Write-Boot ("uvicorn started pid=" + $p.Id)
    exit 0
} catch {
    Write-Boot ("ERROR: " + $_.Exception.Message)
    throw
}
