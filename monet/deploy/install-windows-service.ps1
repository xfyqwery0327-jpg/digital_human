# 把莫奈讲解员注册为 Windows 服务（本机没有 systemctl，用 NSSM 实现同等效果）
# 管理员 PowerShell 执行:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\deploy\install-windows-service.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = (Get-Command python -ErrorAction Stop).Source
$Entry = Join-Path $Root "server8.py"
$Logs = Join-Path $Root "logs"
$ServiceName = "MonetGuide"

New-Item -ItemType Directory -Force -Path $Logs | Out-Null

$nssm = Get-Command nssm -ErrorAction SilentlyContinue
if (-not $nssm) {
    $local = Join-Path $PSScriptRoot "nssm.exe"
    if (Test-Path $local) {
        $nssmPath = $local
    } else {
        Write-Host "未找到 nssm。请先安装: winget install NSSM.NSSM"
        Write-Host "或把 nssm.exe 放到 $($PSScriptRoot)"
        exit 1
    }
} else {
    $nssmPath = $nssm.Source
}

& $nssmPath stop $ServiceName 2>$null
& $nssmPath remove $ServiceName confirm 2>$null
& $nssmPath install $ServiceName $Python $Entry
& $nssmPath set $ServiceName AppDirectory $Root
& $nssmPath set $ServiceName AppStdout (Join-Path $Logs "monet-guide.out.log")
& $nssmPath set $ServiceName AppStderr (Join-Path $Logs "monet-guide.err.log")
& $nssmPath set $ServiceName AppRotateFiles 1
& $nssmPath set $ServiceName Start SERVICE_AUTO_START
& $nssmPath set $ServiceName AppRestartDelay 5000
& $nssmPath start $ServiceName

Write-Host "已安装并启动 Windows 服务: $ServiceName"
Write-Host "  状态: Get-Service $ServiceName"
Write-Host "  停止: nssm stop $ServiceName"
Write-Host "  卸载: nssm stop $ServiceName; nssm remove $ServiceName confirm"
Write-Host "  日志: $Logs"
