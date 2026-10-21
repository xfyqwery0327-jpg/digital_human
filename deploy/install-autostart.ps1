# Register Windows auto-start. Run as Administrator:
#   powershell -ExecutionPolicy Bypass -File C:\digital_human\deploy\install-autostart.ps1
#
# Default: start after user logon.
# Boot before logon:
#   powershell -ExecutionPolicy Bypass -File C:\digital_human\deploy\install-autostart.ps1 -AtStartup
#
# Uninstall:
#   powershell -ExecutionPolicy Bypass -File C:\digital_human\deploy\uninstall-autostart.ps1

param(
    [switch]$AtStartup,
    [int]$DelaySeconds = 40
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "server9\app.py"))) {
    $Root = "C:\digital_human"
}
if (-not (Test-Path (Join-Path $Root "server9\app.py"))) {
    throw ("server9\app.py not found, Root=" + $Root)
}

$TaskName = "DigitalHuman"
$StartPs1 = Join-Path $Root "deploy\start-server.ps1"
$UninstallPs1 = Join-Path $Root "deploy\uninstall-autostart.ps1"
$Log1 = Join-Path $Root "logs\autostart.log"
$Log2 = Join-Path $Root "logs\digital-human.out.log"
$Pwsh = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$arg = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$StartPs1`""

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

$action = New-ScheduledTaskAction -Execute $Pwsh -Argument $arg -WorkingDirectory $Root
if ($AtStartup) {
    $trigger = New-ScheduledTaskTrigger -AtStartup
    $principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
    $mode = "AtStartup SYSTEM"
} else {
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
    $mode = ("AtLogOn " + $env:USERNAME)
}
if ($DelaySeconds -gt 0) {
    $trigger.Delay = ("PT" + [string]$DelaySeconds + "S")
}

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 5 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force | Out-Null

Write-Host ("Registered task: " + $TaskName)
Write-Host ("  mode: " + $mode)
Write-Host ("  delay: " + [string]$DelaySeconds + "s")
Write-Host ("  root: " + $Root)
Write-Host ("  try now: Start-ScheduledTask -TaskName " + $TaskName)
Write-Host ("  status:  Get-ScheduledTask -TaskName " + $TaskName)
Write-Host ("  remove:  powershell -ExecutionPolicy Bypass -File " + $UninstallPs1)
Write-Host ("  log:     " + $Log1)
Write-Host ("           " + $Log2)
