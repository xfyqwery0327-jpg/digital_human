# 把讲解员注册为 Windows 服务（NSSM）。管理员 PowerShell：
#   Set-ExecutionPolicy -Scope Process Bypass
#   C:\digital_human\deploy\install-windows-service.ps1
#
# 要求：conda 环境 robot 已装好；C:\ro\CosyVoice 已存在。

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "server9\app.py"))) {
    $Root = "C:\digital_human"
}
if (-not (Test-Path (Join-Path $Root "server9\app.py"))) {
    throw "找不到 server9\app.py，当前 Root=$Root"
}

$pyCandidates = @(
    (Join-Path $env:CONDA_PREFIX "python.exe"),
    (Join-Path $env:USERPROFILE "miniconda3\envs\robot\python.exe"),
    "C:\Users\BroadMesse\miniconda3\envs\robot\python.exe"
) | Where-Object { $_ -and (Test-Path $_) }
if (-not $pyCandidates) {
    $cmd = Get-Command python -ErrorAction Stop
    $pyCandidates = @($cmd.Source)
}
$Python = $pyCandidates[0]
$Logs = Join-Path $Root "logs"
$ServiceName = "DigitalHuman"
$Cosy = "C:\ro\CosyVoice"

New-Item -ItemType Directory -Force -Path $Logs | Out-Null

$nssm = Get-Command nssm -ErrorAction SilentlyContinue
if (-not $nssm) {
    $local = Join-Path $PSScriptRoot "nssm.exe"
    if (Test-Path $local) {
        $nssmPath = $local
    } else {
        Write-Host "未找到 nssm。请先安装: winget install NSSM.NSSM"
        Write-Host "或把 nssm.exe 放到 $PSScriptRoot"
        exit 1
    }
} else {
    $nssmPath = $nssm.Source
}

$envExtra = @(
    "PYTHONUNBUFFERED=1",
    "PYTHONPATH=$(Join-Path $Root 'server9')",
    "COSYVOICE_PATH=$Cosy",
    "MONET_HOST=0.0.0.0",
    "MONET_PORT=8848"
) -join "`n"

& $nssmPath stop $ServiceName 2>$null
& $nssmPath remove $ServiceName confirm 2>$null
& $nssmPath install $ServiceName $Python
& $nssmPath set $ServiceName AppParameters "-m uvicorn app:app --host 0.0.0.0 --port 8848 --log-level info"
& $nssmPath set $ServiceName AppDirectory $Root
& $nssmPath set $ServiceName AppEnvironmentExtra $envExtra
& $nssmPath set $ServiceName AppStdout (Join-Path $Logs "digital-human.out.log")
& $nssmPath set $ServiceName AppStderr (Join-Path $Logs "digital-human.err.log")
& $nssmPath set $ServiceName AppRotateFiles 1
& $nssmPath set $ServiceName Start SERVICE_AUTO_START
& $nssmPath set $ServiceName AppRestartDelay 5000
& $nssmPath start $ServiceName

Write-Host "已安装并启动 Windows 服务: $ServiceName"
Write-Host "  项目根: $Root"
Write-Host "  Python: $Python"
Write-Host "  CosyVoice: $Cosy"
Write-Host "  状态: Get-Service $ServiceName"
Write-Host "  停止: nssm stop $ServiceName"
Write-Host "  卸载: nssm stop $ServiceName; nssm remove $ServiceName confirm"
Write-Host "  日志: $Logs"
