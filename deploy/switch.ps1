# 数字讲解员开关。由项目根目录 switch.bat 调用。
# 开机任务读 deploy\last-start.txt：llm=4|8，tts=cosyvoice|qwen3，cosyvoice=2|3。
# 改用语音只写配置。要换上正在跑的讲解员，再执行 restart，只动 8848。
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("on", "off", "status", "saved", "llm", "cv", "tts", "restart")]
    [string]$Action,
    [string]$Which = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "server9\app.py"))) {
    $Root = "C:\digital_human"
}
if (-not (Test-Path (Join-Path $Root "server9\app.py"))) {
    throw ("server9\app.py not found, Root=" + $Root)
}

$LastStart = Join-Path $Root "deploy\last-start.txt"
$LlmMarker = Join-Path $Root "deploy\llm.model"
$CvMarker = Join-Path $Root "deploy\cosyvoice.model"
$LlmNames = @{
    "4" = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    "8" = "Qwen3-8B-Q4_K_M.gguf"
}

function Normalize-Llm([string]$raw) {
    $t = (($raw + "").Trim().ToLower())
    if ($t -in @("4", "4b", "qwen3-4b", "qwen3-4b-instruct-2507")) { return "4" }
    if ($t -in @("8", "8b", "qwen3-8b")) { return "8" }
    return $null
}

function Normalize-Cv([string]$raw) {
    $t = (($raw + "").Trim().ToLower())
    if ($t -in @("2", "cv2", "cosyvoice2", "2-0.5b", "cosyvoice2-0.5b")) { return "2" }
    if ($t -in @("3", "cv3", "cosyvoice3", "3-0.5b", "fun", "fun-cosyvoice3-0.5b")) { return "3" }
    return $null
}

function Llm-Title([string]$tag) {
    if ($tag -eq "4") { return "Qwen3-4B" }
    return "Qwen3-8B"
}

function Cv-Title([string]$tag) {
    if ($tag -eq "2") { return "CosyVoice2" }
    return "CosyVoice3"
}

function Read-LastKey([string]$key) {
    if (-not (Test-Path $LastStart)) { return $null }
    $prefix = $key + "="
    foreach ($line in @(Get-Content -Path $LastStart -ErrorAction SilentlyContinue)) {
        if ($line.StartsWith($prefix)) { return $line.Substring($prefix.Length).Trim() }
    }
    return $null
}

function Get-SavedLlm {
    $tag = Normalize-Llm (Read-LastKey "llm")
    if ($tag) { return $tag }
    if (Test-Path $LlmMarker) {
        $raw = ((Get-Content -Path $LlmMarker -Raw -ErrorAction SilentlyContinue) + "").Trim()
        if ($raw) {
            $tag = Normalize-Llm (($raw -split "\s+")[0])
            if ($tag) { return $tag }
        }
    }
    return "8"
}

function Get-SavedCv {
    $tag = Normalize-Cv (Read-LastKey "cosyvoice")
    if ($tag) { return $tag }
    if (Test-Path $CvMarker) {
        $raw = ((Get-Content -Path $CvMarker -Raw -ErrorAction SilentlyContinue) + "").Trim()
        if ($raw) {
            $tag = Normalize-Cv (($raw -split "\s+")[0])
            if ($tag) { return $tag }
        }
    }
    return "3"
}

function Normalize-Tts([string]$raw) {
    $t = (($raw + "").Trim().ToLower())
    if ($t -in @("qwen", "qwen3", "qwen3-tts", "qwen3tts")) { return "qwen3" }
    if ($t -in @("cv", "cosyvoice", "cosy", "cosyvoice2", "cosyvoice3")) { return "cosyvoice" }
    return $null
}

function Get-SavedTts {
    $tag = Normalize-Tts (Read-LastKey "tts")
    if ($tag) { return $tag }
    return "cosyvoice"
}

function Voice-Title([string]$tts, [string]$cv) {
    if ($tts -eq "qwen3") { return "Qwen3-TTS" }
    return (Cv-Title $cv)
}

function Get-QwenDir {
    return Join-Path $Root "models\Qwen3-TTS-12Hz-0.6B-Base"
}

function Get-LlmPath([string]$tag) {
    return Join-Path $Root ("models\" + $LlmNames[$tag])
}

function Get-CosyRoot {
    foreach ($c in @(
        (Join-Path $Root "CosyVoice"),
        "E:\projects\Robot_Intelligent_Voice_System\CosyVoice",
        "C:\ro\CosyVoice"
    )) {
        if (Test-Path $c) { return $c }
    }
    return "C:\ro\CosyVoice"
}

function Get-CvDir([string]$tag) {
    $name = $(if ($tag -eq "2") { "CosyVoice2-0.5B" } else { "Fun-CosyVoice3-0.5B" })
    return Join-Path (Get-CosyRoot) ("pretrained_models\" + $name)
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

function Save-Choice([string]$llm, [string]$cv, [string]$tts = "") {
    if (-not $tts) { $tts = Get-SavedTts }
    Set-Content -Path $LlmMarker -Value ($llm + "`r`n") -Encoding ascii
    Set-Content -Path $CvMarker -Value ($cv + "`r`n") -Encoding ascii
    $stamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
    $text = "llm=" + $llm + "`r`ntts=" + $tts + "`r`ncosyvoice=" + $cv + "`r`nstarted=" + $stamp + "`r`n"
    Set-Content -Path $LastStart -Value $text -Encoding ascii -NoNewline
    Update-LlamaCfg (Get-LlmPath $llm)
    Write-Host ("已记下开机配置：讲解 " + (Llm-Title $llm) + "，语音 " + (Voice-Title $tts $cv))
    Write-Host ("记录: " + $LastStart)
}

function Get-ListenPids([int]$port) {
    $ids = @()
    $pat = ":" + $port + "\s+.*LISTENING"
    $lines = netstat -ano -p tcp | Select-String -Pattern $pat
    foreach ($line in $lines) {
        $parts = ($line.ToString().Trim() -split "\s+") | Where-Object { $_ }
        if (-not $parts) { continue }
        $procId = $parts[-1]
        if ($procId -match "^\d+$") {
            $n = [int]$procId
            if ($n -gt 0 -and ($ids -notcontains $n)) { $ids += $n }
        }
    }
    return $ids
}

function Get-CmdLine([int]$procId) {
    $p = Get-CimInstance Win32_Process -Filter ("ProcessId=" + $procId) -ErrorAction SilentlyContinue
    if ($p) { return $p.CommandLine }
    return $null
}

function Tag-FromCmd([string]$cmd) {
    if (-not $cmd) { return $null }
    if ($cmd -like "*Qwen3-4B*") { return "4" }
    if ($cmd -like "*Qwen3-8B*") { return "8" }
    return $null
}

function Get-LlmApiText {
    if (@($(Get-ListenPids 8080)).Count -eq 0) { return $null }
    try {
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8080/v1/models" -TimeoutSec 2
        $bits = New-Object System.Collections.Generic.List[string]
        foreach ($item in @($resp.data)) {
            if ($item.id) { $bits.Add([string]$item.id) }
            if ($item.root) { $bits.Add([string]$item.root) }
        }
        return ($bits -join " ")
    } catch {
        return $null
    }
}

function Get-LlmRunningTag {
    $pids = @($(Get-ListenPids 8080))
    foreach ($procId in $pids) {
        $tag = Tag-FromCmd (Get-CmdLine $procId)
        if ($tag) { return $tag }
    }
    $apiTag = Tag-FromCmd (Get-LlmApiText)
    if ($apiTag) { return $apiTag }
    if ($pids.Count -eq 0) { return $null }
    $procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -like "*llama_cpp.server*" }
    foreach ($p in $procs) {
        $tag = Tag-FromCmd $p.CommandLine
        if ($tag) { return $tag }
    }
    return "unknown"
}

function Wait-Until([scriptblock]$cond, [int]$seconds) {
    $deadline = (Get-Date).AddSeconds($seconds)
    do {
        if (& $cond) { return $true }
        Start-Sleep -Milliseconds 400
    } while ((Get-Date) -lt $deadline)
    return [bool](& $cond)
}

function Stop-Task([string]$name) {
    $task = $null
    try { $task = Get-ScheduledTask -TaskName $name -ErrorAction Stop } catch { return }
    if ($task.State -ne "Running") { return }
    Write-Host ("停止计划任务 " + $name)
    Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    Wait-Until {
        try { $now = Get-ScheduledTask -TaskName $name -ErrorAction Stop } catch { return $true }
        return ($now.State -ne "Running")
    } 15 | Out-Null
}

function Stop-ByCommand([string]$like) {
    $procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -like $like -and $_.ProcessId -ne $PID }
    foreach ($p in @($procs)) {
        Write-Host ("结束进程 " + $p.ProcessId)
        & taskkill.exe /T /F /PID $p.ProcessId | Out-Null
    }
}

function Stop-Port([int]$port) {
    foreach ($procId in @($(Get-ListenPids $port))) {
        Write-Host ("停止 " + $port + " 进程 " + $procId)
        & taskkill.exe /T /F /PID $procId | Out-Null
    }
}

function Stop-Llm {
    Stop-Task "DigitalHumanLLM"
    Stop-ByCommand "*start-llm.ps1*"
    Stop-ByCommand "*llama_cpp.server*"
    Stop-Port 8080
    Wait-Until { @($(Get-ListenPids 8080)).Count -eq 0 } 10 | Out-Null
    Start-Sleep -Seconds 2
}

function Stop-Narrator {
    Stop-Task "DigitalHuman"
    Stop-ByCommand "*start-server.ps1*"
    Stop-ByCommand "*uvicorn app:app*"
    Stop-Port 8848
    Wait-Until { @($(Get-ListenPids 8848)).Count -eq 0 } 10 | Out-Null
}

function Start-ScriptTask([string]$taskName, [string]$scriptPath) {
    $task = $null
    try { $task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop } catch { $task = $null }
    if ($task -and $task.State -eq "Running") {
        Write-Host ($taskName + " 已在运行")
        return
    }
    if ($task) {
        Write-Host ("启动计划任务 " + $taskName)
        Start-ScheduledTask -TaskName $taskName
        return
    }
    Write-Host ("没有计划任务 " + $taskName + "，直接启动")
    Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -WorkingDirectory $Root -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $scriptPath
    ) | Out-Null
}

function Show-LogTail([string]$path) {
    if (-not (Test-Path $path)) {
        Write-Host ("没有日志 " + $path)
        return
    }
    Write-Host ("--- " + $path + " ---")
    Get-Content -Path $path -Tail 15 | ForEach-Object {
        if ($_.Length -gt 240) { Write-Host ($_.Substring(0, 240) + " ...") }
        else { Write-Host $_ }
    }
}

function Wait-Ready([bool]$needLlm, [string]$llmTag, [bool]$needCv, [int]$seconds) {
    $deadline = (Get-Date).AddSeconds($seconds)
    $next = (Get-Date).AddSeconds(8)
    $llmOk = -not $needLlm
    $cvOk = -not $needCv
    while ((Get-Date) -lt $deadline) {
        if (-not $llmOk) {
            $running = Get-LlmRunningTag
            if ($running -eq $llmTag) { $llmOk = $true }
            elseif ($running -and $running -ne "unknown") {
                Write-Host ("8080 上不是 " + (Llm-Title $llmTag))
                return $false
            }
        }
        if (-not $cvOk -and @($(Get-ListenPids 8848)).Count -gt 0) { $cvOk = $true }
        if ($llmOk -and $cvOk) { return $true }
        if ((Get-Date) -ge $next) {
            $left = [int]($deadline - (Get-Date)).TotalSeconds
            if ($left -lt 0) { $left = 0 }
            Write-Host ("正在加载，大约还剩 " + $left + " 秒")
            $next = (Get-Date).AddSeconds(10)
        }
        Start-Sleep -Seconds 1
    }
    return ($llmOk -and $cvOk)
}

function Assert-LlmFile([string]$tag) {
    $path = Get-LlmPath $tag
    if (-not (Test-Path $path)) {
        throw ("找不到讲解模型: " + $path)
    }
    return $path
}

function Assert-CvDir([string]$tag) {
    $dir = Get-CvDir $tag
    if (-not (Test-Path $dir)) {
        throw ("找不到语音权重: " + $dir)
    }
    $yaml = $(if ($tag -eq "2") { "cosyvoice2.yaml" } else { "cosyvoice3.yaml" })
    if (-not (Test-Path (Join-Path $dir $yaml))) {
        throw ("权重目录里没有 " + $yaml + ": " + $dir)
    }
    return $dir
}

function Assert-QwenDir {
    $dir = Get-QwenDir
    if (-not (Test-Path (Join-Path $dir "config.json"))) {
        throw ("找不到 Qwen3-TTS 权重: " + $dir)
    }
    $weight = Join-Path $dir "model.safetensors"
    $alias = Join-Path $dir "model.safetensors_"
    if (-not (Test-Path $weight) -and -not (Test-Path $alias)) {
        throw ("Qwen3-TTS 目录里没有 model.safetensors: " + $dir)
    }
    return $dir
}

function Assert-Voice([string]$tts, [string]$cv) {
    if ($tts -eq "qwen3") { return Assert-QwenDir }
    return Assert-CvDir $cv
}

function Show-Saved {
    $llm = Get-SavedLlm
    $cv = Get-SavedCv
    $tts = Get-SavedTts
    Write-Host ("下次开机：讲解 " + (Llm-Title $llm) + "，语音 " + (Voice-Title $tts $cv))
}

function Show-Status {
    $llm = Get-SavedLlm
    $cv = Get-SavedCv
    $tts = Get-SavedTts
    Write-Host ("开机配置：讲解 " + (Llm-Title $llm) + "，语音 " + (Voice-Title $tts $cv))
    if (Test-Path $LastStart) {
        $started = Read-LastKey "started"
        if (-not $started) {
            foreach ($line in @(Get-Content -Path $LastStart -ErrorAction SilentlyContinue)) {
                if ($line -match '^\s*started\s*=\s*(.+)$') { $started = $Matches[1].Trim(); break }
            }
        }
        if ($started) {
            Write-Host ("上次 switch.bat 启动: " + $started)
        } else {
            Write-Host "还没用 switch.bat 启动过。开机先按上面这份默认配置。"
        }
    } else {
        Write-Host "还没有 switch.bat 启动记录，开机按 8B + CosyVoice3"
    }
    Write-Host "讲解模型文件:"
    foreach ($tag in @("4", "8")) {
        $path = Get-LlmPath $tag
        if (Test-Path $path) {
            $mb = [math]::Round((Get-Item $path).Length / 1MB, 0)
            Write-Host ("  " + (Llm-Title $tag) + "  " + $mb + " MB")
        } else {
            Write-Host ("  " + (Llm-Title $tag) + "  文件不在  " + $path)
        }
    }
    $cvDir = Get-CvDir $cv
    Write-Host ("CosyVoice 目录: " + $cvDir)
    Write-Host ("  存在: " + (Test-Path $cvDir))
    $qwenDir = Get-QwenDir
    $qwenWeight = Join-Path $qwenDir "model.safetensors"
    $qwenAlias = Join-Path $qwenDir "model.safetensors_"
    Write-Host ("Qwen3-TTS 目录: " + $qwenDir)
    if (Test-Path $qwenWeight) {
        Write-Host "  权重: model.safetensors"
    } elseif (Test-Path $qwenAlias) {
        Write-Host "  权重文件名是 model.safetensors_，首次加载会改成 model.safetensors"
    } else {
        Write-Host "  没有模型权重"
    }
    $running = Get-LlmRunningTag
    $pids8080 = @($(Get-ListenPids 8080))
    if ($pids8080.Count -eq 0) {
        Write-Host "8080 讲解模型：没在听"
    } elseif ($running -eq "unknown" -or -not $running) {
        Write-Host ("8080 讲解模型：在听，PID " + ($pids8080 -join ",") + "，认不出是哪一个")
    } else {
        Write-Host ("8080 讲解模型：在听 " + (Llm-Title $running) + "  PID " + ($pids8080 -join ","))
    }
    $pids8848 = @($(Get-ListenPids 8848))
    if ($pids8848.Count -eq 0) {
        Write-Host "8848 讲解员：没在听"
    } else {
        Write-Host ("8848 讲解员：在听  PID " + ($pids8848 -join ","))
    }
    foreach ($name in @("DigitalHumanLLM", "DigitalHuman")) {
        try {
            $task = Get-ScheduledTask -TaskName $name -ErrorAction Stop
            Write-Host ("计划任务 " + $name + "：" + $task.State)
        } catch {
            Write-Host ("计划任务 " + $name + "：未注册")
        }
    }
}

function Start-LlmNow {
    Start-ScriptTask "DigitalHumanLLM" (Join-Path $Root "deploy\start-llm.ps1")
}

function Start-NarratorNow {
    Start-ScriptTask "DigitalHuman" (Join-Path $Root "deploy\start-server.ps1")
}

try {
    if ($Action -eq "saved") {
        Show-Saved
        exit 0
    }
    if ($Action -eq "status") {
        Show-Status
        exit 0
    }
    if ($Action -eq "off") {
        Stop-Narrator
        Stop-Llm
        if (@($(Get-ListenPids 8080)).Count -gt 0 -or @($(Get-ListenPids 8848)).Count -gt 0) {
            Write-Host "还有端口没停干净。"
            Show-Status
            exit 1
        }
        Write-Host "已关闭。下次登录仍按上次启动的配置自动拉起。"
        exit 0
    }
    if ($Action -eq "llm") {
        $tag = Normalize-Llm $Which
        if (-not $tag) { throw "请指定 4 或 8。例如：switch.bat llm 4" }
        Assert-LlmFile $tag | Out-Null
        $cv = Get-SavedCv
        Save-Choice $tag $cv
        $running = Get-LlmRunningTag
        if ($running -eq $tag) {
            Write-Host ("8080 已经是 " + (Llm-Title $tag) + "，没有重启。")
            exit 0
        }
        if ($running) { Stop-Llm }
        Start-LlmNow
        if (Wait-Ready $true $tag $false 120) {
            Write-Host ("讲解模型已启动：" + (Llm-Title $tag))
            exit 0
        }
        if (@($(Get-ListenPids 8080)).Count -gt 0) {
            Write-Host ("8080 已在听，但没有确认是 " + (Llm-Title $tag) + "。可打开 http://127.0.0.1:8080/v1/models")
        } else {
            Write-Host "讲解模型没有在 120 秒内听上 8080。"
        }
        Show-LogTail (Join-Path $Root "logs\llm-autostart.log")
        Show-LogTail (Join-Path $Root "logs\llama.out.log")
        exit 1
    }
    if ($Action -eq "cv") {
        $tag = Normalize-Cv $Which
        if (-not $tag) { throw "请指定 2 或 3。例如：switch.bat cv 3" }
        Assert-CvDir $tag | Out-Null
        $llm = Get-SavedLlm
        Assert-LlmFile $llm | Out-Null
        Save-Choice $llm $tag "cosyvoice"
        Write-Host ("已改用 " + (Cv-Title $tag) + "。正在跑的 8848 不会自己换。")
        exit 0
    }
    if ($Action -eq "tts") {
        $tag = Normalize-Tts $Which
        if (-not $tag) { throw "请指定 qwen 或 cv。例如：switch.bat tts qwen" }
        $llm = Get-SavedLlm
        $cv = Get-SavedCv
        Assert-LlmFile $llm | Out-Null
        Assert-Voice $tag $cv | Out-Null
        Save-Choice $llm $cv $tag
        Write-Host ("已改用 " + (Voice-Title $tag $cv) + "。正在跑的 8848 不会自己换。")
        exit 0
    }
    if ($Action -eq "restart") {
        $llm = Get-SavedLlm
        $cv = Get-SavedCv
        $tts = Get-SavedTts
        Assert-LlmFile $llm | Out-Null
        Assert-Voice $tts $cv | Out-Null
        Write-Host ("重启讲解员，语音 " + (Voice-Title $tts $cv) + "。8080 不动。")
        Stop-Narrator
        Start-NarratorNow
        if (Wait-Ready $false "" $true 240) {
            Write-Host "讲解员已按当前语音配置重启。"
            exit 0
        }
        Write-Host "讲解员没有在 240 秒内听上 8848。"
        Show-LogTail (Join-Path $Root "logs\autostart.log")
        Show-LogTail (Join-Path $Root "logs\digital-human.out.log")
        exit 1
    }
    if ($Action -eq "on") {
        $llm = Get-SavedLlm
        $cv = Get-SavedCv
        $tts = Get-SavedTts
        Assert-LlmFile $llm | Out-Null
        Assert-Voice $tts $cv | Out-Null
        Save-Choice $llm $cv
        $needLlm = $true
        $running = Get-LlmRunningTag
        if ($running -eq $llm) {
            Write-Host ("8080 已经是 " + (Llm-Title $llm))
            $needLlm = $false
        } elseif ($running) {
            Stop-Llm
            Start-LlmNow
        } else {
            Start-LlmNow
        }
        $needCv = $true
        if (@($(Get-ListenPids 8848)).Count -gt 0) {
            Write-Host "8848 已在听，没有重启讲解员。改用语音后要生效，请再选重启。"
            $needCv = $false
        } else {
            Start-NarratorNow
        }
        if (-not $needLlm -and -not $needCv) {
            Write-Host "两路都已按当前配置在跑。"
            exit 0
        }
        $wait = 150
        if ($needLlm -and -not $needCv) { $wait = 120 }
        if (Wait-Ready $needLlm $llm $needCv $wait) {
            Write-Host "已按上次配置启动。"
            exit 0
        }
        Write-Host "启动超时。看下面的日志。"
        Show-LogTail (Join-Path $Root "logs\llm-autostart.log")
        Show-LogTail (Join-Path $Root "logs\llama.out.log")
        Show-LogTail (Join-Path $Root "logs\autostart.log")
        exit 1
    }
    throw ("未知动作: " + $Action)
} catch {
    Write-Host ("失败: " + $_.Exception.Message)
    exit 1
}
