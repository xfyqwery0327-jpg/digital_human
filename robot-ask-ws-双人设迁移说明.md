# robot-ask / ws-tts 双人设迁移说明

日期：2026-09-18  
参考：`项目整体脉络.md`、`D:\projects\monet\项目整体脉络.md`、`D:\projects\monet\robot-ask-ws-改动说明.md`

范围：把单人莫奈已验证的 robot-ask / ws-tts 推流修复迁到 monet / fuhao 双人设，最小改动，兼容旧路由。

未改：yaml 人设加载、RAG 知识库、`/api/persona/switch`、`/ws/{persona}/chat`、`/health` 原有字段。

---

## 1）完整回归测试用例清单

通过条件（每轮）：HTTP **202**；`tts_start` / `tts_end` 各 **1** 次；`llm_answer` 紧跟 `tts_start`（距 start < 2s，且在 `tts_end` 之前）；有 `tts_chunk`；下行带正确 `persona_id` / `display_name`；robot 轮带 `robot_id`。

顺序必须是：

```text
asr → tts_start → llm_answer → tts_sentence/tts_chunk… →（PCM 真实时长 + ROBOT_PLAY_TAIL_SEC）→ tts_end
```

### A. 单人等价（莫奈）

| # | 场景 | 操作 | 期望 |
|---|------|------|------|
| A1 | 正常三轮 | `python scripts/test_robot_ask.py --persona-id monet` | 3 轮 PASS |
| A2 | 缓存命中 | 同一句再 ask 一次（欢迎稿 / 固定句） | 日志 `按播放节奏推流`；`tts_end` 不早于约 0.85×PCM |
| A3 | 无前端连接 | 不连 WS，只 POST ask | HTTP 202；日志 `没有前端连接 /ws/monet/tts` 且 `ws_pushed<=0` 告警 |
| A4 | 回声双流 | `--echo-tts` | 日志 `忽略游客 TTS（机器人正在推流）persona=monet`；仍只有 1 次 start/end |
| A5 | 讲完后补 TTS | 等 busy=false 后再发 type=tts | 不再被 `is_robot_ask_echo` 拦住 |
| A6 | 并发 ask | 上一轮未完再 ask | HTTP 409 `turn_in_progress` |
| A7 | 旧路由 | `--legacy` | `/api/robot/ask` + `/ws/tts` 行为与 monet 相同 |

### B. 妇好等价

| # | 场景 | 操作 | 期望 |
|---|------|------|------|
| B1 | 正常三轮 | `python scripts/test_robot_ask.py --persona-id fuhao` | 3 轮 PASS；欢迎/知识走妇好稿 |
| B2 | 无连接告警 | 不连 `/ws/fuhao/tts` 就 ask | 告警路径是 `/ws/fuhao/tts`，不是 monet |
| B3 | 回声 | fuhao 连接上 `--echo-tts` | 只拦截 fuhao 分组 |
| B4 | 连错 WS | 连 `/ws/monet/tts` 却 POST fuhao ask | monet 连接收不到；fuhao 无连接告警 |

### C. 双人设隔离

| # | 场景 | 操作 | 期望 |
|---|------|------|------|
| C1 | 分组广播 | 同时连 monet/fuhao TTS，只 ask fuhao | 仅 fuhao 连接收到 asr…tts_end；monet 连接静默 |
| C2 | busy 分组拦截 | fuhao robot 播报中，monet 游客发 type=tts | monet 不被忽略；fuhao 游客 tts/cancel 被忽略 |
| C3 | 回声不串台 | fuhao ask query=「你好」时，monet 游客也说「你好」 | monet 不被 `is_robot_ask_echo` 误拦 |
| C4 | GPU 全局互斥 | fuhao busy 时 POST monet ask | 409 `turn_in_progress`（GPU 单路） |
| C5 | 会话隔离 | 拿 monet session_id 去 fuhao ask | 释放旧会话、开新人设会话 |
| C6 | 下行字段 | 任意 robot 轮 | 每条 WS 含 `persona_id`、`display_name`、`robot_id` |
| C7 | 游客字段 | 游客自己 type=tts | `persona_id` 有值，`robot_id` 为空串 |

### D. 异常与韧性

| # | 场景 | 期望 |
|---|------|------|
| D1 | 单连接 send 超时 | 8s 超时后摘掉该连接，其它连接继续；日志含 type + persona |
| D2 | 重复 tts_end / llm_answer | 内部丢弃，前端仍各 1 次 |
| D3 | 未知 persona | HTTP 404；WS close 1008 |
| D4 | 再见 | query 含「再见」→ 该人设告别稿 → 释放会话；HTTP 仍 202 |

### 推荐命令

```powershell
python scripts/test_robot_ask.py --persona-id monet
python scripts/test_robot_ask.py --persona-id fuhao
python scripts/test_robot_ask.py --legacy
python scripts/test_robot_ask.py --persona-id fuhao --echo-tts --rounds 1
python scripts/test_robot_ask.py --persona-id monet --query 你好
```

---

## 2）变更摘要

### 解决的故障（与单人版等价）

| 故障 | 修复 |
|------|------|
| 缓存 PCM 瞬间推完，`tts_end` 过早 | `cosyvoice.py` 缓存分片按时长 **0.85 倍** 匀速推 |
| `llm_answer` 后置 | `stream.py` 在 `tts_start` 后立刻推 `llm_answer` |
| `tts_start` 后 Unity 未就绪 | `tts_start` 送达后 **sleep 200ms** |
| `tts_end` 早于真实播放 | 等 PCM 时长 + `ROBOT_PLAY_TAIL_SEC` 再发 end |
| 前端回声双流抢占 | 当前播报人设 busy 时忽略该分组 `type=tts` / `cancel`；游客被接管不再补 `tts_end` |
| 重复 start/end/llm | fanout 内部去重 |
| WS 超时 2s 误摘连接 | fanout 单条超时 **8s**，失败打 type/persona 并摘死连接 |
| 无连接无告警 | `ws_pushed<=0` 明确 warning，路径含 `/ws/{persona}/tts` |

HTTP 仍立刻 **202**，无 `turn_end` 回调。声音只走前端 WS。

### 双人设适配点

1. **`TtsFanout.send(obj, persona_id)`**：只推向该人设 `/ws/{id}/tts`；旧 `/ws/tts` 登记为 `monet`。
2. **busy 拦截按分组**：`robot_hub.is_busy_for(persona_id)` 只拦正在播报的那个人；其他人设 WS 上行不拦。
3. **GPU 全局互斥不变**：`acquire` 时任意人设 busy 都 409。
4. **回声 per-persona**：`is_robot_ask_echo(text, persona_id=...)`；chat 未改文件，走 ContextVar。
5. **所有 robot/fanout/stream 日志带 `persona_id`**。
6. **下行填 `persona_id` / `display_name`；robot 轮填 `robot_id`**。
7. **旧路由** `/api/robot/ask`、`/ws/tts`、`/ws/chat` = monet，修复同样生效。

### 修改文件

- `server9/robot/stream.py`
- `server9/tts/cosyvoice.py`
- `server9/ws/tts.py`
- `server9/robot/hub.py`
- `server9/ws/hub.py`
- `server9/robot/api.py`
- `scripts/test_robot_ask.py`（新增，`--persona-id`）

### 上线风险

| 风险 | 说明 | 缓解 |
|------|------|------|
| `tts_end` 变晚 | 等真实 PCM，一轮 HTTP 后台更长，busy 窗口变长 | 符合播放；下一轮 ask 需等讲完 |
| 缓存句变慢 | 0.85 倍推流，听感对齐播放，不再「闪一下」 | 预期行为 |
| 另一人设游客 TTS | robot 播 fuhao 时 monet 游客 type=tts 不拦，可能和 GPU 抢算力 | robot 侧仍 409；现场避免同时开口 |
| 旧 Unity 忽略新字段 | `persona_id` 可忽略 | 兼容；须改连带人设的 WS 才能接妇好 |
| 必须先连对 WS | 连错人设等于没连 | 看 `ws_pushed<=0` 告警里的路径 |

重启讲解员进程后生效。Ask **之前** 展项须连上对应 `ws://主机:8848/ws/{persona}/tts` 并保持不断。

---

## 3）正常 / 异常场景日志样例（含 persona_id）

### 正常：妇好一轮

```text
[ROBOT] 开启会话 session=robot_1758_ab12 robot=robot-01 persona=fuhao
[ROBOT] mark_busy action=ask request_id=e3b0c442 persona=fuhao robot=robot-01
已推 asr session=robot_1758_ab12 robot=robot-01 persona=fuhao clients=1 text=妇好介绍下自己吧
已推 tts_start session=robot_1758_ab12 robot=robot-01 persona=fuhao clients=1 hint=我是妇好
已推 llm_answer session=robot_1758_ab12 robot=robot-01 persona=fuhao chars=42
开始逐句 TTS sentences=2 persona=fuhao
🔁 TTS 缓存命中 persona=fuhao '我是妇好。今日由我…' | 4.80s，按播放节奏推流
等播报结束再 tts_end remain=0.31s pcm=4.80s session=robot_1758_ab12 persona=fuhao
已推 tts_end session=robot_1758_ab12 robot=robot-01 persona=fuhao seq=18 sentences=2 clients=1 tts_wall=5.12s pcm=4.80s
前端 WS 推送结束 ws_clients=1 duration=4.80s request_id=e3b0c442 persona=fuhao
本轮完成 session=robot_1758_ab12 persona=fuhao round=1 6.01s farewell=False
```

### 正常：旧路由莫奈

```text
🎤 TTS WebSocket 连接建立 persona=monet
已受理 202 request_id=… session=… persona=monet（响应发出后再开讲）
已推 tts_start … persona=monet clients=1
已推 llm_answer … persona=monet
已推 tts_end … persona=monet … tts_wall≈pcm
```

### 异常：无连接

```text
没有前端连接 /ws/fuhao/tts，字幕和音频分片无人接收 robot=robot-01 persona=fuhao
tts_start 未送达任何 /ws/fuhao/tts 客户端，前端不会播报 session=… robot=robot-01 persona=fuhao
本轮未送达前端，Unity 不会播报。请确认展项已连接 /ws/fuhao/tts 且保持不断 request_id=… persona=fuhao
```

HTTP 仍是 202，没有 turn_end 回调。

### 异常：回声双流被拦（仅当前人设）

```text
⏭ 忽略游客 TTS（机器人正在推流）persona=fuhao: 我是妇好。今日由我陪你看这一场…
⏭ 忽略游客 cancel（机器人正在推流，本轮由 robot stream 收尾） persona=fuhao
⏭ 跳过游客 tts_end（机器人本轮接管）session=a1b2c3d4 persona=fuhao
```

此时 `/ws/monet/tts` 上的游客 `type=tts` **不应**出现上述忽略日志。

### 异常：跨人设 GPU 互斥

```text
会话未就绪 request_id=… status=409 body={"ok":false,"code":"turn_in_progress","persona_id":"fuhao",...}
```

### 异常：单连接失败不拖死其它连接

```text
[ROBOT] WS 推送失败 type=tts_chunk persona=fuhao want=fuhao: timeout
[ROBOT] 已摘除 1 个失败的 /ws/tts 连接，剩余 1 want=fuhao
```

### 不应再出现

- robot 推流过程中夹一行 `🤖 用户问题: 我是妇好…` / `🤖 用户问题: 我是克洛德…`（讲稿被当成游客新问题）
- 同一轮两个 `tts_end` 或两个 `llm_answer`
- `tts_end` 比 PCM 时长早出十几秒（缓存倒流）
- fuhao 的 PCM 推到 `/ws/monet/tts`
