# 莫奈讲解员 ↔ 机器人 互动接口

服务地址默认：`http://<讲解员Windows主机>:8848`  
Swagger：`http://<主机>:8848/docs`  
健康检查：`GET /health`（含 `robot` 字段）

双方用 **HTTP JSON** 互调。讲解员负责检索、以莫奈身份介绍、合成语音；机器人负责到场/离场/提问，并在收到「本轮结束」回调后再进行下一轮。

---

## 1. 推荐时序（2～3 轮不被打断）

```
机器人                         莫奈讲解员 :8848
  |                                |
  | POST /api/robot/hello          |
  |------------------------------->|  独占通道，合成欢迎语
  | 202 {session_id, request_id}   |
  |<-------------------------------|
  |                                |  本机音箱播放欢迎语
  |                                |  ……讲完……
  | POST /api/monet/on_turn_end    |
  |<-------------------------------|  event=turn_end, spoken=true（此时已经讲完）
  | 播放 audio，播完再往下          |
  |                                |
  | POST /api/robot/ask            |  第 1 轮提问
  | {query, session_id}            |
  |------------------------------->|  检索 + 介绍 + TTS
  | 202                            |
  |<-------------------------------|
  | POST /api/monet/on_turn_end    |
  |<-------------------------------|  action=ask, round=1, can_continue=true
  |                                |
  | POST /api/robot/ask  (第2轮)   |
  |------------------------------->|
  | ... turn_end round=2           |
  |                                |
  | POST /api/robot/ask  (第3轮)   |
  |------------------------------->|
  | ... turn_end round=3           |
  |     can_continue=false         |
  |                                |
  | POST /api/robot/bye            |
  |------------------------------->|  合成再见
  | POST /api/monet/on_turn_end    |
  |<-------------------------------|  action=bye
  | POST /api/monet/on_session_end |
  |<-------------------------------|  释放独占，其它游客端可恢复
```

**硬规则**

1. 收到 `202` 只表示已受理，**莫奈本机讲完后才会回调 `event=turn_end`**。
2. 收到 `turn_end` 之前不要发下一轮 `ask` / `bye`，否则返回 `409 turn_in_progress`。
3. `turn_end` 里 `spoken=true` 表示展厅这边已经播完，**机器人不要再播一遍 audio**，直接进入下一动作。
4. 独占期间，游客端 `/ws/tts`、`/ws/chat`、`POST /tts` 会被拒绝（`robot_busy`）。
5. 默认最多 **3** 轮 `ask`（不含欢迎/再见）。第 3 轮 `can_continue=false`，请改调 `/bye`。
6. 两轮之间超过 **120 秒** 无请求，讲解员自动释放独占并回调 `session_end`（`reason=idle_timeout`）。

---

## 2. 公共约定

| 项 | 说明 |
|----|------|
| Content-Type | `application/json; charset=utf-8` |
| 鉴权 | 若讲解员配置了 `ROBOT_API_TOKEN`，双方请求都带请求头 `X-Robot-Token: <token>` |
| 音频 | 默认由讲解员 Windows 本机音箱播放。`turn_end` 在**播完后**才发送。`include_audio=true` 时回调里仍带一份 WAV 备份，但 `spoken=true` 时请勿再播。 |
| `wait` | 默认 `false`：立刻 202，**讲完后**回调。联调可 `true`，HTTP 会等到讲完再返回 |

错误码：

| HTTP | code | 含义 |
|------|------|------|
| 401 | — | `X-Robot-Token` 不正确 |
| 409 | `robot_busy` | 另一台机器人或另一段会话占用中 |
| 409 | `turn_in_progress` | 本轮还没回调 turn_end |
| 409 | `max_rounds` | 已满 3 轮，请 `/bye` |
| 409 | `no_session` | 没有进行中的会话（静默结束时视为成功，不报此错） |
| 422 | — | `query` 为空等参数错误 |

---

## 3. 机器人 → 莫奈（讲解员提供）

Base：`http://<讲解员IP>:8848`

### 3.1 到场欢迎 `POST /api/robot/hello`

开启独占并说欢迎语。说完回调 `turn_end`（`action=hello`）。

请求：

```json
{
  "robot_id": "robot-01",
  "session_id": null,
  "callback_url": "http://10.0.0.21:9000/api/monet/on_turn_end",
  "include_audio": true,
  "wait": false,
  "speaker": "中文男",
  "text": null
}
```

| 字段 | 必填 | 说明 |
|------|------|------|
| robot_id | 否 | 机器人编号，默认 `default`。换一台机器人时必须不同，否则会 409 |
| session_id | 否 | 不填则新建。同一会话续接时带回上一轮返回的 id |
| callback_url | 否 | 本会话所有事件都 POST 到这个完整 URL（靠 body 里的 `event` 区分）。不填则走环境变量 `ROBOT_API_BASE` + 约定路径 |
| include_audio | 否 | 默认 true |
| wait | 否 | 默认 false |
| text | 否 | 自定义欢迎语；不填则随机内置三句之一 |

立即返回（`wait=false`）：

```json
{
  "ok": true,
  "accepted": true,
  "action": "hello",
  "session_id": "robot_1757490000_ab12cd34",
  "request_id": "e3b0c44298fc1c14",
  "robot_id": "robot-01",
  "message": "已受理，介绍完成后会回调 turn_end；请勿在回调前发送下一轮"
}
```

请保存 `session_id`，后续 `ask` / `bye` 带上。

### 3.2 提问讲解 `POST /api/robot/ask`

检索知识库，以莫奈第一人称介绍。介绍完成回调 `turn_end`（`action=ask`）。

```json
{
  "query": "睡莲是在哪里画的？",
  "robot_id": "robot-01",
  "session_id": "robot_1757490000_ab12cd34",
  "callback_url": "http://10.0.0.21:9000/api/monet/on_turn_end",
  "include_audio": true,
  "wait": false,
  "top_k": 2
}
```

未先 `hello` 也可以直接 `ask`（会自动开独占会话），但推荐先欢迎再提问。

检索分数不够时仍会说话：「这个我还不太清楚…」，`refused=true`，本轮仍算一轮。

### 3.3 再见并结束 `POST /api/robot/bye`

说再见，然后依次回调 `turn_end`（`action=bye`）和 `session_end`，释放独占。

```json
{
  "robot_id": "robot-01",
  "session_id": "robot_1757490000_ab12cd34",
  "callback_url": "http://10.0.0.21:9000/api/monet/on_turn_end",
  "include_audio": true,
  "wait": false,
  "text": null,
  "silent": false
}
```

`silent=true`：不合成语音，只回调 `session_end` 并立刻释放。正在合成时仍会 `409 turn_in_progress`。没有会话时返回成功（可重复调用）。

### 3.4 查询状态 `GET /api/robot/status`

```json
{
  "ok": true,
  "active": true,
  "session_id": "robot_1757490000_ab12cd34",
  "robot_id": "robot-01",
  "round": 1,
  "max_rounds": 3,
  "busy": false,
  "current_action": null,
  "idle_remain_sec": 96,
  "callback_url": "http://10.0.0.21:9000/api/monet/on_turn_end"
}
```

`busy=true` 表示本轮还在检索/合成，不要发下一轮。

### 3.5 莫奈主动开口（中控或传感器调用讲解员）

讲解员也可以**主动**找机器人互动，不需要等机器人先 hello。

| 接口 | 说明 |
|------|------|
| `POST /api/robot/push/hello` | 先通知机器人 `event=hello`，再播欢迎语，说完 `turn_end` |
| `POST /api/robot/push/bye` | 先通知机器人 `event=bye`，再播再见，说完 `turn_end` + `session_end` |

```json
{
  "robot_id": "robot-01",
  "callback_url": "http://10.0.0.21:9000/api/monet/on_turn_end",
  "include_audio": true,
  "wait": false,
  "text": null,
  "notify_peer": true
}
```

`notify_peer=false` 时不预告，只在说完后回调 `turn_end`。

---

## 4. 莫奈 → 机器人（机器人必须实现）

讲解员在「一轮话说完」或「会话结束」时 **主动 POST** 机器人。  
机器人至少实现 **一个** 接收地址；用 JSON 里的 `event` 分支即可。

### 4.1 推荐：单一回调（最简单）

请求里带 `callback_url`，例如 `http://<机器人IP>:9000/api/monet/callback`。  
所有事件都打到这里。

### 4.2 或者按路径拆分（配置 `ROBOT_API_BASE`）

假设 `ROBOT_API_BASE=http://10.0.0.21:9000`：

| 事件 | 默认路径 | 何时调用 |
|------|----------|----------|
| `hello` | `/api/monet/on_hello` | 仅 `push/hello` 且 `notify_peer=true`，**开口前** |
| `bye` | `/api/monet/on_bye` | 仅 `push/bye` 且 `notify_peer=true`，**开口前** |
| `turn_end` | `/api/monet/on_turn_end` | **本机把这一轮话讲完之后**（`spoken=true`，不要再播 audio） |
| `session_end` | `/api/monet/on_session_end` | 再见结束、静默结束、空闲超时、进程关闭 |

机器人接口请在 **8 秒内** 返回 2xx。失败会重试 2 次。不需要返回音频。

成功响应示例：`{"ok": true}`

### 4.3 `turn_end` 请求体（机器人务必处理这个）

```json
{
  "event": "turn_end",
  "ok": true,
  "action": "ask",
  "session_id": "robot_1757490000_ab12cd34",
  "request_id": "e3b0c44298fc1c14",
  "robot_id": "robot-01",
  "round": 1,
  "max_rounds": 3,
  "can_continue": true,
  "rounds_left": 2,
  "refused": false,
  "sources": ["item_sleeping_lotus_01"],
  "text": "我在吉维尼的花园里种了一整片睡莲……",
  "audio": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAAAA...",
  "audio_format": "wav",
  "sample_rate": 24000,
  "spoken": true,
  "played_local": true,
  "play_duration_sec": 6.42,
  "elapsed_sec": 8.10,
  "next_hint": "ask",
  "callback_ok": true,
  "callback_detail": "http_200"
}
```

| 字段 | 说明 |
|------|------|
| action | `hello` / `ask` / `bye` |
| round | 仅 `ask` 累加，欢迎/再见不计入 3 轮 |
| can_continue | false 时不要再 `ask`，请 `bye` |
| next_hint | `ask` 或 `bye` |
| refused | 知识库没匹配上 |
| text | 字幕/播报文本 |
| audio | 可选备份 WAV；`spoken=true` 时不要再播 |
| spoken | true 表示莫奈已经在本机讲完，这是发 turn_end 的前提 |
| played_local | 声卡是否真正播放成功 |
| play_duration_sec | 音频时长（秒） |
| request_id | 与 202 返回值对应，用于去重 |

收到后建议：`spoken=true` 则莫奈已讲完，直接进入下一动作。若 `can_continue` 则等待观众下一句再 `ask`，否则 `bye`。

### 4.4 `session_end` 请求体

```json
{
  "event": "session_end",
  "ok": true,
  "reason": "bye",
  "session_id": "robot_1757490000_ab12cd34",
  "robot_id": "robot-01",
  "round": 2,
  "action": "bye"
}
```

`reason`：`bye` / `silent_bye` / `idle_timeout` / `server_shutdown`。

### 4.5 主动互动预告 `hello` / `bye`（可选）

仅当中控调了讲解员的 `/api/robot/push/hello` 或 `/push/bye` 时出现，让机器人停步、看向展项、不要抢话。

```json
{
  "event": "hello",
  "action": "hello",
  "session_id": "robot_1757490000_ab12cd34",
  "request_id": "e3b0c44298fc1c14",
  "robot_id": "robot-01",
  "ok": true,
  "message": "莫奈开始招呼，请等待 turn_end"
}
```

---

## 5. 环境变量（讲解员 Windows 机）

可在启动前设置，不必改代码。PowerShell 示例：

```powershell
$env:ROBOT_API_BASE = "http://10.0.0.21:9000"
$env:ROBOT_API_TOKEN = "replace-me"
$env:ROBOT_MAX_ROUNDS = "3"
$env:ROBOT_IDLE_TIMEOUT = "120"
python server8.py
```

| 变量 | 默认 | 说明 |
|------|------|------|
| ROBOT_API_BASE | 空 | 机器人根地址。请求未带 `callback_url` 时使用 |
| ROBOT_API_TOKEN | 空 | 非空则双方都要带 `X-Robot-Token` |
| ROBOT_MAX_ROUNDS | 3 | ask 轮次上限 |
| ROBOT_IDLE_TIMEOUT | 120 | 轮次间隔超时（秒） |
| ROBOT_TURN_END_AFTER_PLAY | 1 | `1`：本机播完才发 turn_end；`0`：合成完立刻发（旧行为） |
| ROBOT_PLAY_TAIL_SEC | 0.25 | 播完后再多等的尾音（秒），避免切得太急 |
| ROBOT_CALLBACK_TURN_END | `/api/monet/on_turn_end` | |
| ROBOT_CALLBACK_SESSION_END | `/api/monet/on_session_end` | |
| ROBOT_CALLBACK_HELLO | `/api/monet/on_hello` | |
| ROBOT_CALLBACK_BYE | `/api/monet/on_bye` | |
| ROBOT_CALLBACK_TIMEOUT | 8 | 回调超时秒 |
| ROBOT_CALLBACK_RETRY | 2 | 失败重试次数 |

若机器人四个事件都收在同一个 URL，把四个 `ROBOT_CALLBACK_*` 都设成同一个完整地址，或每次请求都传 `callback_url`。

---

## 6. 联调 curl（PowerShell）

把 `MONET`、`ROBOT` 换成真实 IP。机器人需先起一个能收 POST 的服务。

```powershell
$MONET = "http://127.0.0.1:8848"
$CALLBACK = "http://10.0.0.21:9000/api/monet/on_turn_end"

# 欢迎
$hello = Invoke-RestMethod -Method Post "$MONET/api/robot/hello" -ContentType "application/json" -Body (@{
  robot_id = "robot-01"
  callback_url = $CALLBACK
} | ConvertTo-Json)
$sid = $hello.session_id

# 等讲解员本机把欢迎语讲完、机器人收到 turn_end 后，再提问
Invoke-RestMethod -Method Post "$MONET/api/robot/ask" -ContentType "application/json" -Body (@{
  robot_id = "robot-01"
  session_id = $sid
  callback_url = $CALLBACK
  query = "睡莲是在哪里画的？"
} | ConvertTo-Json)

# 第二轮、第三轮同样 ask……

# 结束
Invoke-RestMethod -Method Post "$MONET/api/robot/bye" -ContentType "application/json" -Body (@{
  robot_id = "robot-01"
  session_id = $sid
  callback_url = $CALLBACK
} | ConvertTo-Json)

# 状态
Invoke-RestMethod "$MONET/api/robot/status"
```

中控让莫奈先开口：

```powershell
Invoke-RestMethod -Method Post "$MONET/api/robot/push/hello" -ContentType "application/json" -Body (@{
  robot_id = "robot-01"
  callback_url = $CALLBACK
  notify_peer = $true
} | ConvertTo-Json)
```

---

## 7. 机器人侧最小实现要点

1. 提供 `POST` 接口接收 JSON，按 `event` 分支。
2. `turn_end`：去重 `request_id`。`spoken=true` 表示莫奈已经讲完，不要再播 `audio`，可以直接 `ask` 或 `bye`。
3. `can_continue=false` 或 `next_hint=bye` 时调讲解员 `/api/robot/bye`。
4. 离开展项、急停：等 `busy=false` 后 `/bye`；若来不及说话用 `/bye` + `"silent": true`。
5. 不要并发调用 hello/ask/bye。同一时刻全馆只有一路机器人讲解。
