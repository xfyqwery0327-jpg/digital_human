# 开机拉起大模型 8080：llama-server.exe，关闭思考模式。
# 不带 -Model 时读 deploy\last-start.txt 的 llm=，再读 deploy\llm.model。
# 两处都没有就用 Qwen3-8B-Q4_K_M.gguf。
# 4B 与 switch.ps1 同一文件名：Qwen3-4B-Instruct-2507-Q4_K_M.gguf。
param(
    [switch]$Foreground,
    [string]$Model = ""
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

function Resolve-LlmTag([string]$raw) {
    $t = (($raw + "").Trim().ToLower())
    if ($t -in @("4", "4b", "qwen3-4b", "qwen3-4b-instruct-2507")) { return "4" }
    if ($t -in @("8", "8b", "qwen3-8b")) { return "8" }
    return ""
}

function Resolve-LlmPath {
    $names = @{
        "4" = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
        "8" = "Qwen3-8B-Q4_K_M.gguf"
    }
    $tag = Resolve-LlmTag $Model
    if (-not $tag) {
        $last = Join-Path $Root "deploy\last-start.txt"
        if (Test-Path $last) {
            foreach ($line in @(Get-Content -Path $last -ErrorAction SilentlyContinue)) {
                if ($line -match '^\s*llm\s*=\s*(\S+)') {
                    $tag = Resolve-LlmTag $Matches[1]
                    if ($tag) { break }
                }
            }
        }
    }
    if (-not $tag) {
        $marker = Join-Path $Root "deploy\llm.model"
        if (Test-Path $marker) {
            $raw = ((Get-Content -Path $marker -Raw -ErrorAction SilentlyContinue) + "").Trim()
            if ($raw) { $tag = Resolve-LlmTag (($raw -split "\s+")[0]) }
        }
    }
    if (-not $tag) { $tag = "8" }
    return Join-Path $Root ("models\" + $names[$tag])
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
    $model = Resolve-LlmPath
    if (-not (Test-Path $model)) {
        Write-Boot ("model not found: " + $model)
        throw "model not found"
    }
    $dir = Split-Path $exe

    if (Test-Path $outLog) { Remove-Item $outLog -Force -ErrorAction SilentlyContinue }
    if (Test-Path $errLog) { Remove-Item $errLog -Force -ErrorAction SilentlyContinue }

    # 与历史运行一致的槽位：-np 4 -c 8192(每槽2048)。--reasoning off 关闭思考，直接输出 content。
    # 不写 --slot-save-path，KV / prompt cache 只在本进程内存里，进程退出即清空。
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
