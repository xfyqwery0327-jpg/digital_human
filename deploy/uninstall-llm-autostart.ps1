$ErrorActionPreference = "Stop"
$TaskName = "DigitalHumanLLM"
$t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $t) {
    Write-Host ("task not found: " + $TaskName)
    exit 0
}
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
Write-Host ("removed task: " + $TaskName)
