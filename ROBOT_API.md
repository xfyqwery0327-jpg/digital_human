# 数字讲解员 ↔ 机器人 互动接口

服务地址默认：`http://<讲解员Windows主机>:8848`  
Swagger：`http://<主机>:8848/docs`  
健康检查：`GET /health`（含 `robot`、`persona`、`personas` 字段）

**人设必须进 URL。** 莫奈与妇好的知识库、声音、规则、介绍稿互不串台：

| 人设 | 机器人提问 | TTS 前端 | 对话 |
|------|------------|----------|------|
| 莫奈 | `POST /api/monet/robot/ask` | `ws://主机:8848/ws/monet/tts` | `/ws/monet/chat` |
| 妇好 | `POST /api/fuhao/robot/ask` | `ws://主机:8848/ws/fuhao/tts` | `/ws/fuhao/chat` |

兼容旧地址：`POST /api/robot/ask`、`/ws/tts`、`/ws/chat` 走默认人设（莫奈）。不要用它们讲妇好。

已删除、请勿再调：`/api/robot/hello`、`/api/robot/bye`、`/api/robot/status`、`/api/robot/push/hello`、`/api/robot/push/bye`。

---

## 1. 推荐时序

```
机器人                         讲解员 :8848
  |                                |
  | POST /api/{persona_id}/robot/ask|  提问（自动开会话）
  | {query, robot_id}              |
  |------------------------------->|  立刻 HTTP 202
  | 202 {session_id, request_id}   |
  |<-------------------------------|
  |                                |  后台检索 + 介绍
  |                                |  推前端 /ws/tts 播报
  |                                |
  | POST /api/robot/ask  (下一轮)   |  须等上一轮讲完
  |------------------------------->|
  | ...                            |
  | POST /api/robot/ask            |  query 含「再见」时口播告别并结束会话
  | {query: "再见"}                |
```

**硬规则**

1. 收到 HTTP `202` 只表示已受理。`wait` 已忽略，不会在 HTTP 里等讲完。声音走前端 `/ws/tts`。
2. 上一轮还在讲时再 `ask`，返回 `409 turn_in_progress`。
3. **不必 hello / bye / status**。直接 `ask` 即可自动开会话。
4. `query` 里带「再见 / 拜拜 / 我走了」等，说完该人设自己的告别后结束会话。
5. 两轮之间超过 **120 秒** 无请求，讲解员自动结束会话。
6. 机器人讲解时游客仍可走同一人设的 `/ws/{persona_id}/tts` 咨询；另一人设的连接收不到这条口播。
7. `query` 为「你好」时播该人设自己的欢迎稿，不会拿莫奈的稿给妇好。

---

## 2. 公共约定

| 项 | 说明 |
|----|------|
| Content-Type | `application/json; charset=utf-8` |
| 鉴权 | 若讲解员配置了 `ROBOT_API_TOKEN`，请求带 `X-Robot-Token: <token>` |
| 音频 | 由莫奈前端 `/ws/tts` 播放。`include_audio=true` 时本轮结果里仍可带 WAV 备份，现场建议 `false` |
| `wait` | **已忽略**。ask 一律立刻 HTTP 202 |

错误码：

| HTTP | code | 含义 |
|------|------|------|
| 401 | — | `X-Robot-Token` 不正确 |
| 409 | `turn_in_progress` | 上一轮还没讲完 |
| 422 | — | `query` 为空等参数错误 |

---

## 3. 提问讲解 `POST /api/{persona_id}/robot/ask`

检索**该人设**知识库，用该人设第一人称介绍。未开会话时自动创建。

```json
{
  "query": "这件器物是什么？",
  "robot_id": "robot-01",
  "session_id": null,
  "include_audio": false,
  "speaker": "中文男",
  "top_k": 2
}
```

| 字段 | 必填 | 说明 |
|------|------|------|
| query | 是 | 观众/机器人问题。含「再见」等则走告别 |
| robot_id | 否 | 机器人编号，默认 `default`。空闲时可直接换一台 |
| session_id | 否 | 不填则新建或续接当前空闲会话 |
| include_audio | 否 | 默认 true；现场建议 false（声音走前端） |
| wait | 否 | 已忽略 |
| speaker | 否 | 默认 `中文男` |
| top_k | 否 | 检索条数，默认 2 |
| category | 否 | 已忽略，始终只检索该人设知识库 |
| persona_id | 否 | 旧接口 `/api/robot/ask` 可用；新接口以 URL 为准 |

立即返回 HTTP **202**：

```json
{
  "ok": true,
  "code": "accepted",
  "accepted": true,
  "done": false,
  "action": "ask",
  "session_id": "robot_1757490000_ab12cd34",
  "request_id": "e3b0c44298fc1c14",
  "robot_id": "robot-01",
  "round": 0,
  "can_continue": true,
  "next_hint": "ask",
  "text": "",
  "refused": false,
  "sources": [],
  "message": "已受理。讲解在后台进行。"
}
```

检索分数不够时仍会说话：「这个我还不太清楚…」，本轮仍算一轮。

后台推到前端 `/ws/tts` 的 `asr`、`tts_start`、`tts_sentence`、`tts_chunk`、`llm_answer`、`tts_end` **每条都带 `robot_id`**（与 ask 请求里的一致；未传则为 `default`）。游客咨询的同类消息 `robot_id` 为空字符串。

需要看会话是否 busy 时用 `GET /health` 里的 `robot` 字段，不要调已删除的 `/api/robot/status`。

---

## 4. 环境变量（讲解员 Windows 机）

可在启动前设置，不必改代码。PowerShell 示例：

```powershell
$env:ROBOT_API_TOKEN = "replace-me"
$env:ROBOT_IDLE_TIMEOUT = "120"
python server8.py
```

| 变量 | 默认 | 说明 |
|------|------|------|
| ROBOT_API_TOKEN | 空 | 非空则请求要带 `X-Robot-Token` |
| ROBOT_IDLE_TIMEOUT | 120 | 轮次间隔超时（秒），超时后本地释放会话 |
| ROBOT_ACCEPTED_STATUS | 202 | ask 受理 HTTP 状态码 |

现场须用 conda env **`robot`** 启动：`python server8.py`。

---

## 5. 联调 curl（PowerShell）

```powershell
$MONET = "http://127.0.0.1:8848"

Invoke-RestMethod -Method Post "$MONET/api/robot/ask" -ContentType "application/json" -Body (@{
  robot_id = "robot-01"
  query = "睡莲是在哪里画的？"
} | ConvertTo-Json)

# 等莫奈讲完后再问下一句
Invoke-RestMethod -Method Post "$MONET/api/robot/ask" -ContentType "application/json" -Body (@{
  robot_id = "robot-01"
  query = "再见"
} | ConvertTo-Json)

# 看会话快照（不是 /api/robot/status）
Invoke-RestMethod "$MONET/health"
```

脚本：`python scripts/call_robot.py --query 睡莲是在哪里画的？`

---

## 6. 机器人侧最小实现要点

1. 只调 `POST /api/robot/ask`，立刻处理 202。不必实现接收回调的接口。
2. 观众说「再见」时，把这句话当作 `query` 即可。
3. 离开展项、急停：可不再调接口，空闲超时会自己结束。
4. 不要并发调用；上一轮还在讲时不要发下一轮。
5. 前端需连着 `/ws/tts`，否则字幕和声音不会出。
