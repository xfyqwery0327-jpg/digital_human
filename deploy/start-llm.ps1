param(
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "server9\app.py"))) {
    $Root = "C:\digital_human"
}

$Logs = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$bootLog = Join-Path $Logs "llm-autostart.log"
$outLog = Join-Path $Logs "llama.out.log"

function Write-Boot([string]$msg) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $bootLog -Value ("[" + $stamp + "] " + $msg)
    Write-Host $msg
}

function Get-LlmTag {
    $token = $null
    $last = Join-Path $Root "deploy\last-start.txt"
    if (Test-Path $last) {
        foreach ($line in @(Get-Content -Path $last -ErrorAction SilentlyContinue)) {
            if ($line -match '^\s*llm\s*=\s*(\S+)') { $token = $Matches[1]; break }
        }
    }
    if (-not $token) {
        $marker = Join-Path $Root "deploy\llm.model"
        if (Test-Path $marker) {
            $raw = ((Get-Content -Path $marker -Raw -ErrorAction SilentlyContinue) + "").Trim()
            if ($raw) { $token = ($raw -split "\s+")[0] }
        }
    }
    $t = (($token + "").Trim().ToLower())
    if ($t -in @("4", "4b", "qwen3-4b", "qwen3-4b-instruct-2507")) { return "4" }
    return "8"
}

function Get-LlmFileName([string]$tag) {
    if ($tag -eq "4") { return "Qwen3-4B-Instruct-2507-Q4_K_M.gguf" }
    return "Qwen3-8B-Q4_K_M.gguf"
}

function Update-LlamaCfg([string]$modelPath) {
    $cfgPath = Join-Path $Root "deploy\llama_cfg.json"
    if (-not (Test-Path $cfgPath)) { return }
    $leaf = Split-Path $modelPath -Leaf
    $rel = "models/" + $leaf
    $raw = [System.IO.File]::ReadAllText($cfgPath)
    $updated = [regex]::Replace($raw, '"model"\s*:\s*"[^"]*"', ('"model": "' + $rel + '"'), 1)
    if ($updated -eq $raw) { return }
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($cfgPath, $updated, $utf8)
}

function Test-Port8080 {
    try {
        $c = New-Object System.Net.Sockets.TcpClient
        $iar = $c.BeginConnect("127.0.0.1", 8080, $null, $null)
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
    $script = Join-Path $PSScriptRoot "server-env.ps1"
    $log = Join-Path $env:TEMP "digital-human-server-env.log"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script -Action ensure > $log 2>&1
    $code = $LASTEXITCODE
    if (Test-Path $log) { Get-Content -Path $log | Out-Host }
    if ($code -ne 0) { return $null }
    $py = Join-Path $env:USERPROFILE ".conda\envs\digital_human_server\python.exe"
    if (Test-Path $py) { return $py }
    return $null
}

try {
    Write-Boot ("script start user=" + $env:USERNAME)

    if (Test-Port8080) {
        Write-Boot "port 8080 already listening, skip"
        exit 0
    }

    $tag = Get-LlmTag
    $RelModel = "models\" + (Get-LlmFileName $tag)
    $Model = Join-Path $Root $RelModel
    $WorkDir = $Root
    if (-not (Test-Path $Model)) {
        Write-Boot ("model not found: " + $RelModel + " root=" + $Root)
        throw ("model not found: " + $RelModel + " root=" + $Root)
    }
    Update-LlamaCfg $RelModel

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
        (Join-Path $CondaEnv "Lib\site-packages\llama_cpp\lib"),
        (Join-Path $CondaEnv "Lib\site-packages\torch\lib"),
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
    ) | Where-Object { $_ -and (Test-Path $_) }
    $env:PATH = ($CondaEnv + ";" + ($extras -join ";") + ";" + $env:PATH)
    $env:PYTHONUNBUFFERED = "1"
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"

    Write-Boot ("python=" + $Python + " llm=" + $tag + " model=" + $RelModel + " n_ctx=2048 chatml")
    Set-Location $WorkDir
    Add-Content -Path $outLog -Value ("`r`n==== " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " start ====`r`n")

    # Same flags as the old C:\ro bat:
    # python -m llama_cpp.server --model ... --n_gpu_layers -1 --n_ctx 2048 --host 127.0.0.1 --port 8080 --chat_format chatml
    $line = " -m llama_cpp.server --model `"$RelModel`" --n_gpu_layers -1 --n_ctx 2048 --host 127.0.0.1 --port 8080 --chat_format chatml"
    Write-Boot "starting llama_cpp.server (blocking)"
    cmd.exe /c "`"$Python`"$line >> `"$outLog`" 2>&1"
    $code = $LASTEXITCODE
    Write-Boot ("llama_cpp.server exit=" + $code)
    exit $code
} catch {
    Write-Boot ("ERROR: " + $_.Exception.Message)
    throw
}
