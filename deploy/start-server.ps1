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

function Find-RobotPython {
    # 环境 digital_human_server 已安装且经手动验证可用，无需每次开机重跑
    # server-env.ps1（它 import torch 会打印 NOTE，5.1 在 Stop 下当错误抛出，
    # 导致开机任务 8848 起不来）。直接检测固定路径。
    $candidates = @(
        (Join-Path $env:USERPROFILE ".conda\envs\digital_human_server\python.exe"),
        "D:\anaconda\envs\digital_human_server\python.exe"
    )
    foreach ($path in $candidates) {
        if ($path -and (Test-Path $path)) { return $path }
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
    $choice = ""
    if (-not $env:COSYVOICE_MODEL) {
        $choice = "3"
        $token = $null
        $last = Join-Path $Root "deploy\last-start.txt"
        if (Test-Path $last) {
            foreach ($line in @(Get-Content -Path $last -ErrorAction SilentlyContinue)) {
                if ($line -match '^\s*cosyvoice\s*=\s*(\S+)') { $token = $Matches[1]; break }
            }
        }
        if (-not $token) {
            $marker = Join-Path $Root "deploy\cosyvoice.model"
            if (Test-Path $marker) {
                $raw = ((Get-Content -Path $marker -Raw -ErrorAction SilentlyContinue) + "").Trim()
                if ($raw) { $token = ($raw -split "\s+")[0] }
            }
        }
        $t = (($token + "").Trim().ToLower())
        if ($t -in @("2", "cv2", "cosyvoice2", "2-0.5b", "cosyvoice2-0.5b")) { $choice = "2" }
        $dirname = $(if ($choice -eq "2") { "CosyVoice2-0.5B" } else { "Fun-CosyVoice3-0.5B" })
        $env:COSYVOICE_MODEL = Join-Path $env:COSYVOICE_PATH ("pretrained_models\" + $dirname)
    } elseif ($env:COSYVOICE_MODEL -match "CosyVoice2") {
        $choice = "2"
    } elseif ($env:COSYVOICE_MODEL -match "CosyVoice3") {
        $choice = "3"
    }

    $ttsEngine = "cosyvoice"
    if (-not $env:TTS_ENGINE) {
        $ttsToken = $null
        $lastTts = Join-Path $Root "deploy\last-start.txt"
        if (Test-Path $lastTts) {
            foreach ($line in @(Get-Content -Path $lastTts -ErrorAction SilentlyContinue)) {
                if ($line -match '^\s*tts\s*=\s*(\S+)') { $ttsToken = $Matches[1]; break }
            }
        }
        $ttsName = (($ttsToken + "").Trim().ToLower())
        if ($ttsName -in @("qwen", "qwen3", "qwen3-tts", "qwen3tts")) { $ttsEngine = "qwen3" }
        $env:TTS_ENGINE = $ttsEngine
    } else {
        $ttsName = $env:TTS_ENGINE.Trim().ToLower()
        if ($ttsName -in @("qwen", "qwen3", "qwen3-tts", "qwen3tts")) { $ttsEngine = "qwen3" }
        else { $ttsEngine = "cosyvoice" }
        $env:TTS_ENGINE = $ttsEngine
    }
    if ($ttsEngine -eq "qwen3" -and -not $env:QWEN_TTS_MODEL) {
        $env:QWEN_TTS_MODEL = Join-Path $Root "models\Qwen3-TTS-12Hz-0.6B-Base"
    }
    if ($ttsEngine -eq "qwen3" -and $env:TTS_QWEN_URL) {
        Write-Boot ("TTS_QWEN_URL=" + $env:TTS_QWEN_URL + " 已设置，讲解员将使用这个地址而不是默认的 8812")
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
    Write-Boot ("python=" + $Python + " tts=" + $env:TTS_ENGINE + " alloc=" + $env:PYTORCH_CUDA_ALLOC_CONF)
    Write-Boot ("cosyvoice_path=" + $env:COSYVOICE_PATH)
    Write-Boot ("cosyvoice=" + $choice + " model=" + $env:COSYVOICE_MODEL)
    if ($env:TTS_ENGINE -eq "qwen3") {
        $directTts = if ($env:TTS_QWEN_URL) { $env:TTS_QWEN_URL } else { "http://127.0.0.1:8812" }
        Write-Boot ("qwen3=" + $env:QWEN_TTS_MODEL + " tts_url=" + $directTts)
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
