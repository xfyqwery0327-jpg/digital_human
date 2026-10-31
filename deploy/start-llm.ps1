# 开机拉起大模型 8080：用 llama-server.exe 加载 Qwen3-8B，关闭思考模式。
# 历史 start-llm.ps1 用 python llama_cpp.server，但 digital_human_server 环境未装
# llama_cpp 模块，且模型带思考会耗光 max_tokens 导致 content 为空，故改为原生 llama-server。
# 参考实际运行日志(llama.out.log)：Qwen3-8B-Q4_K_M.gguf, n_slots=4, n_ctx_slot=2048。
param(
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "models\Qwen3-8B-Q4_K_M.gguf"))) {
    $Root = "C:\digital_human"
}

$Logs = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$bootLog = Join-Path $Logs "llm-autostart.log"
$outLog = Join-Path $Logs "llama.out.log"
$errLog = Join-Path $Logs "llama.err.log"

function Write-Boot([string]$msg) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $bootLog -Value ("[" + $stamp + "] " + $msg)
    Write-Host $msg
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

try {
    Write-Boot ("script start user=" + $env:USERNAME)

    if (Test-Port8080) {
        Write-Boot "port 8080 already listening, skip"
        exit 0
    }

    # llama-server.exe 复用 Robot_Intelligent_Voice_System 的二进制（讲解员目录无此 exe）。
    $exe = "D:\projects\Robot_Intelligent_Voice_System\models\llama\llama-server.exe"
    if (-not (Test-Path $exe)) {
        Write-Boot ("llama-server.exe not found: " + $exe)
        throw "llama-server.exe not found"
    }
    $model = Join-Path $Root "models\Qwen3-8B-Q4_K_M.gguf"
    if (-not (Test-Path $model)) {
        Write-Boot ("model not found: " + $model)
        throw "model not found"
    }
    $dir = Split-Path $exe

    if (Test-Path $outLog) { Remove-Item $outLog -Force -ErrorAction SilentlyContinue }
    if (Test-Path $errLog) { Remove-Item $errLog -Force -ErrorAction SilentlyContinue }

    # 与历史运行一致的槽位：-np 4 -c 8192(每槽2048)。--reasoning off 关闭思考，直接输出 content。
    $args = @(
        "-m", $model,
        "-ngl", "99",
        "-fa", "on",
        "-c", "8192",
        "-np", "4",
        "-t", "8",
        "--host", "127.0.0.1",
        "--port", "8080",
        "--jinja",
        "--reasoning", "off"
    )
    Write-Boot ("starting llama-server model=" + (Split-Path $model -Leaf) + " reasoning=off")
    if ($Foreground) {
        & $exe @args
        $code = $LASTEXITCODE
    } else {
        $p = Start-Process -FilePath $exe -ArgumentList $args `
            -WorkingDirectory $dir -WindowStyle Hidden `
            -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
        Write-Boot ("llama-server started pid=" + $p.Id)
        exit 0
    }
    Write-Boot ("llama-server exit=" + $code)
    exit $code
} catch {
    Write-Boot ("ERROR: " + $_.Exception.Message)
    throw
}
