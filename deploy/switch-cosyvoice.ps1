# 切换开机使用的 CosyVoice 权重。写入 deploy\cosyvoice.model，下次启动生效。
# 用法：
#   powershell -ExecutionPolicy Bypass -File deploy\switch-cosyvoice.ps1
#   powershell -ExecutionPolicy Bypass -File deploy\switch-cosyvoice.ps1 3
#   powershell -ExecutionPolicy Bypass -File deploy\switch-cosyvoice.ps1 2 -Restart
param(
    [Parameter(Position = 0)]
    [string]$Version = "",
    [switch]$Restart
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "server9\app.py"))) {
    $Root = "C:\digital_human"
}
if (-not (Test-Path (Join-Path $Root "server9\app.py"))) {
    throw ("server9\app.py not found, Root=" + $Root)
}

$Marker = Join-Path $Root "deploy\cosyvoice.model"

function Get-CosyVoiceRoot {
    foreach ($c in @(
        (Join-Path $Root "CosyVoice"),
        "E:\projects\Robot_Intelligent_Voice_System\CosyVoice",
        "C:\ro\CosyVoice"
    )) {
        if (Test-Path $c) { return $c }
    }
    return "C:\ro\CosyVoice"
}

function ConvertTo-Choice([string]$raw) {
    $token = (($raw + "").Trim().ToLower().Split() | Select-Object -First 1)
    if ($token -in @("2", "cv2", "cosyvoice2", "2-0.5b", "cosyvoice2-0.5b")) { return "2" }
    if ($token -in @("3", "cv3", "cosyvoice3", "3-0.5b", "fun", "fun-cosyvoice3-0.5b")) { return "3" }
    return $null
}

function Get-ModelDir([string]$choice) {
    $name = $(if ($choice -eq "2") { "CosyVoice2-0.5B" } else { "Fun-CosyVoice3-0.5B" })
    return Join-Path (Get-CosyVoiceRoot) ("pretrained_models\" + $name)
}

function Get-SavedChoice {
    if (-not (Test-Path $Marker)) { return "3" }
    $saved = ConvertTo-Choice (Get-Content -Path $Marker -Raw -ErrorAction SilentlyContinue)
    if ($saved) { return $saved }
    return "3"
}

function Get-ListenPid([int]$port) {
    $pat = ":" + $port + "\s+.*LISTENING"
    $lines = netstat -ano -p tcp | Select-String -Pattern $pat
    foreach ($line in $lines) {
        $parts = ($line.ToString().Trim() -split "\s+") | Where-Object { $_ }
        if (-not $parts) { continue }
        $procId = $parts[-1]
        if ($procId -match "^\d+$" -and [int]$procId -gt 0) { return [int]$procId }
    }
    return 0
}

$current = Get-SavedChoice
if (-not $Version) {
    $dir = Get-ModelDir $current
    Write-Host ("当前选择: " + $current + "  " + $dir)
    Write-Host ("权重目录存在: " + (Test-Path $dir))
    Write-Host "切换: switch-cosyvoice.ps1 2  或  3"
    Write-Host "马上重启讲解员: 加上 -Restart（只动 8848，不动 8080）"
    exit 0
}

$choice = ConvertTo-Choice $Version
if (-not $choice) {
    throw ("无法识别版本: " + $Version + "。请用 2 或 3")
}

$dir = Get-ModelDir $choice
if (-not (Test-Path $dir)) {
    throw ("找不到权重目录: " + $dir)
}
$yaml = $(if ($choice -eq "2") { "cosyvoice2.yaml" } else { "cosyvoice3.yaml" })
if (-not (Test-Path (Join-Path $dir $yaml))) {
    throw ("目录里没有 " + $yaml + ": " + $dir)
}

Set-Content -Path $Marker -Value ($choice + "`r`n") -Encoding ascii
$lastStart = Join-Path $Root "deploy\last-start.txt"
$llmSaved = "8"
if (Test-Path $lastStart) {
    foreach ($line in @(Get-Content -Path $lastStart -ErrorAction SilentlyContinue)) {
        if ($line -match '^\s*llm\s*=\s*(\S+)') { $llmSaved = $Matches[1].Trim(); break }
    }
}
$stamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
$lastText = "llm=" + $llmSaved + "`r`ncosyvoice=" + $choice + "`r`nstarted=" + $stamp + "`r`n"
Set-Content -Path $lastStart -Value $lastText -Encoding ascii -NoNewline
Write-Host ("已切换到 " + $choice + ": " + $dir)
Write-Host ("记录: " + $Marker)

if (-not $Restart) {
    $listen = Get-ListenPid 8848
    if ($listen -gt 0) {
        Write-Host ("8848 仍由进程 " + $listen + " 占用，当前这次讲解还是旧模型。")
        Write-Host ("下次开机，或执行 switch-cosyvoice.ps1 " + $choice + " -Restart 后才会换上。")
    } else {
        Write-Host "8848 没在听。下次 start-server.ps1 / 开机任务会用这个模型。"
    }
    exit 0
}

$BootLog = Join-Path $Root "logs\autostart.log"
$dirname = Split-Path $dir -Leaf

function Stop-Listen8848 {
    $listen = Get-ListenPid 8848
    if ($listen -le 0) { return }
    Write-Host ("停止 8848 进程 " + $listen)
    Stop-Process -Id $listen -Force -ErrorAction SilentlyContinue
}

function Wait-Until([scriptblock]$cond, [int]$seconds) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        if (& $cond) { return $true }
        Start-Sleep -Milliseconds 400
    }
    return [bool](& $cond)
}

function Start-NarratorDirect {
    Write-Host "直接启动 start-server.ps1"
    Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -WorkingDirectory $Root -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $Root "deploy\start-server.ps1")
    ) | Out-Null
}

# 计划任务是 IgnoreNew：旧实例还在 Running 时，Start-ScheduledTask 会被丢掉，
# 8848 已被杀掉，autostart.log 也不会出现新的 model=。
$task = Get-ScheduledTask -TaskName "DigitalHuman" -ErrorAction SilentlyContinue
if ($task) {
    if ($task.State -eq "Running") {
        Write-Host "先停止计划任务 DigitalHuman"
        Stop-ScheduledTask -TaskName "DigitalHuman" -ErrorAction SilentlyContinue
    }
}
Stop-Listen8848
$taskStopped = Wait-Until {
    $now = Get-ScheduledTask -TaskName "DigitalHuman" -ErrorAction SilentlyContinue
    return ((-not $now) -or ($now.State -ne "Running"))
} 15
if (-not $taskStopped) {
    Write-Host "计划任务仍显示 Running，再停一次 8848"
    Stop-Listen8848
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -like "*start-server.ps1*" -and $_.ProcessId -ne $PID } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Wait-Until {
        $now = Get-ScheduledTask -TaskName "DigitalHuman" -ErrorAction SilentlyContinue
        return ((-not $now) -or ($now.State -ne "Running"))
    } 10 | Out-Null
}
Wait-Until { (Get-ListenPid 8848) -le 0 } 8 | Out-Null

$logBefore = 0L
if (Test-Path $BootLog) { $logBefore = [int64](Get-Item $BootLog).Length }
if ($task) {
    Write-Host "启动计划任务 DigitalHuman"
    Start-ScheduledTask -TaskName "DigitalHuman"
} else {
    Start-NarratorDirect
}

$logged = Wait-Until {
    if (-not (Test-Path $BootLog)) { return $false }
    if ([int64](Get-Item $BootLog).Length -le $logBefore) { return $false }
    $hit = Get-Content $BootLog -Tail 12 | Where-Object { $_ -like ("*model=*" + $dirname + "*") } | Select-Object -Last 1
    if ($hit) {
        $script:ModelLine = $hit
        return $true
    }
    return $false
} 20

if (-not $logged) {
    Write-Host "计划任务没有写出新的 model=，改为直接启动"
    Start-NarratorDirect
    $logged = Wait-Until {
        if (-not (Test-Path $BootLog)) { return $false }
        if ([int64](Get-Item $BootLog).Length -le $logBefore) { return $false }
        $hit = Get-Content $BootLog -Tail 12 | Where-Object { $_ -like ("*model=*" + $dirname + "*") } | Select-Object -Last 1
        if ($hit) {
            $script:ModelLine = $hit
            return $true
        }
        return $false
    } 20
}

if ($logged) {
    Write-Host ("已启动: " + $script:ModelLine)
    exit 0
}
Write-Host "启动后 logs\autostart.log 仍没有新的 model=。8848 可能没起来。"
Write-Host "在服务器上执行: Stop-ScheduledTask -TaskName DigitalHuman; Start-ScheduledTask -TaskName DigitalHuman"
exit 1
