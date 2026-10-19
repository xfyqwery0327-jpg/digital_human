# 从开发机打一份可上传包（不含 CosyVoice / conda / 旧代码）。
#   powershell -ExecutionPolicy Bypass -File .\deploy\pack-for-server.ps1
#   powershell -ExecutionPolicy Bypass -File .\deploy\pack-for-server.ps1 -Dest C:\digital_human
#
# 上传目标：服务器 C:\digital_human
# CosyVoice 留在服务器 C:\ro\CosyVoice，不要打进本包。

param(
    [string]$Dest = "",
    [switch]$WithEmbedModel
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Dest) {
    $Dest = Join-Path $Root "_pack\digital_human"
}

Write-Host "源: $Root"
Write-Host "目标: $Dest"

New-Item -ItemType Directory -Force -Path $Dest | Out-Null

$dirs = @(
    "server9",
    "personas",
    "index",
    "voice",
    "dict",
    "scripts",
    "deploy",
    "kb"
)
foreach ($d in $dirs) {
    $src = Join-Path $Root $d
    $dst = Join-Path $Dest $d
    if (-not (Test-Path $src)) {
        Write-Host "跳过缺失目录 $d"
        continue
    }
    Write-Host "复制 $d ..."
    robocopy $src $dst /E /NFL /NDL /NJH /NJS /nc /ns /np `
        /XD __pycache__ .git /XF *.pyc *.pyo | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy 失败: $d code=$LASTEXITCODE" }
}

$files = @(
    "ROBOT_API.md",
    "项目整体脉络.md",
    "robot-ask-ws-双人设迁移说明.md"
)
foreach ($f in $files) {
    $src = Join-Path $Root $f
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $Dest $f) -Force
    }
}

New-Item -ItemType Directory -Force -Path (Join-Path $Dest "logs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Dest "sessions") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Dest "models") | Out-Null

if ($WithEmbedModel) {
    $snap = "snapshots\71ab7947d6fac5b64aa299e6e40e6c2b2e85976c"
    $hubs = @(
        (Join-Path $Root "models\gte-base-zh"),
        (Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--thenlper--gte-base-zh\$snap"),
        "C:\Users\BroadMesse\.cache\huggingface\hub\models--thenlper--gte-base-zh\$snap"
    )
    $hit = $hubs | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($hit) {
        $modelDest = Join-Path $Dest "models\gte-base-zh"
        Write-Host "复制 embedding 模型 -> models\gte-base-zh"
        robocopy $hit $modelDest /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "复制 embedding 失败 code=$LASTEXITCODE" }
    } else {
        Write-Host "未找到本机 gte-base-zh 缓存，服务器需自备 HuggingFace 模型或设 EMBED_MODEL"
    }
}

Write-Host ""
Write-Host "已打包到: $Dest"
Write-Host "不要上传: CosyVoice、.venv-kb、monet 旧目录、分集剧本知识库、sessions、logs"
Write-Host "服务器解压/复制到 C:\digital_human 后执行:"
Write-Host "  conda activate robot"
Write-Host "  powershell -ExecutionPolicy Bypass -File C:\digital_human\deploy\start-server.ps1"
