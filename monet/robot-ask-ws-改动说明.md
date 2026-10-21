# /api/robot/ask 推流改动说明

日期：2026-09-17  
范围：讲解员收到机器人 `POST /api/robot/ask` 后，往前端 `/ws/tts` 推字幕和 PCM。

---

## 1. 问题

现场现象：

- `/api/robot/ask` 返回 **202**，机器人侧觉得「对方没收到」
- 前端收不到可用的 `tts_start` 播报，或播报被切成两次
- `llm_answer` 来得太晚，`tts_end` 来得太早
- 同一轮会出现两个 `end` / 两路 TTS 叠在一条 WebSocket 上

说明：202 只表示「已受理」。HTTP `turn_end` 回调此前已按需求去掉，讲解内容只走 `/ws/tts`。

---

## 2. 原因

1. **TTS 缓存瞬间倒流**：同一句命中缓存时，约 30 秒 PCM 在几十毫秒内全部推完，立刻跟 `tts_end`。Unity 还没按 `tts_start` 建好播放器。
2. **`llm_answer` 放在整段合成之后**：前端要等音频推完才拿到全文。
3. **游客回声双流**：前端收到 `asr` 会再发 `type=tts`（原问题，原先已拦截）；收到 `llm_answer` 后又会 `cancel`，再用讲稿发一条 `type=tts`。第二条对不上 ask 原文，后端当成游客新问题，和 robot 流抢同一条 `/ws/tts`、同一块 GPU。

---

## 3. 现在的推流顺序

一轮只走 robot 这一条流：

```text
asr → tts_start → llm_answer → tts_sentence / tts_chunk… →（等到 PCM 时长）→ tts_end
```

约束：

- `tts_start` / `tts_end` 各一次
- `llm_answer` 紧跟 `tts_start`，不再等合成结束
- `tts_end` 按音频真实时长发送，合成比播放快时先等播完
- 机器人 `busy` 期间忽略前端的 `type=tts` 和 `type=cancel`

---

## 4. 修改文件

### 4.1 `server9/robot/stream.py`

- 推流顺序改为：`asr` → `tts_start` → `llm_answer` → 分句音频 → `tts_end`
- `tts_start` 送达后等待 200ms，给 Unity 建播放器
- 内部丢弃重复的 `tts_end` / `llm_answer`
- 分片推完后，按 PCM 时长 + `ROBOT_PLAY_TAIL_SEC` 再发 `tts_end`
- `tts_start` 未送达任何客户端时打 warning

### 4.2 `server9/tts/cosyvoice.py`

- 缓存命中不再 `sleep(0)` 瞬间倒完
- 按分片音频时长的 0.85 倍匀速推流（日志：`TTS 缓存命中 … 按播放节奏推流`）

### 4.3 `server9/ws/tts.py`

- 机器人 `busy` 时忽略所有游客 `type=tts`（含把讲稿当新问题）
- 机器人 `busy` 时忽略游客 `type=cancel`，本轮由 robot stream 收尾
- 游客会话被机器人接管时不再补发 `tts_end`，避免第二个 end

### 4.4 `server9/robot/hub.py`

- `is_robot_ask_echo` 仅在本轮 `busy` 时拦截「问题原文」回声
- 讲完后若前端漏收 robot 流，允许它自己再要一次 TTS

### 4.5 `server9/ws/hub.py`

- `/ws/tts` 单条发送超时 2s → 8s
- 推送失败时打印消息类型，并摘掉死连接

### 4.6 `server9/robot/api.py`

- `ws_pushed <= 0` 时打 warning：展项未连接 `/ws/tts` 则不会播报

### 4.7 `scripts/test_robot_ask.py`（新增）

联调脚本：先连 `/ws/tts`，再 `POST /api/robot/ask`，只收流、不回 echo。

```powershell
python scripts/test_robot_ask.py
python scripts/test_robot_ask.py --host 127.0.0.1 --port 8848 --robot-id robot-001
python scripts/test_robot_ask.py --query 莫奈介绍下自己吧
python scripts/test_robot_ask.py --echo-tts
```

默认三轮：介绍自己 → 有哪些作品 → 介绍睡莲。  
通过条件：HTTP 202、`tts_start`/`tts_end` 各 1 次、`llm_answer` 在 `end` 之前、有音频分片。

`--echo-tts` 会在收到 `llm_answer` 后再发 `type=tts`，用来复现双流。

---

## 5. 未改动的约定

- HTTP 仍立刻 202，不回调机器人 `turn_end`
- 声音只走前端 `/ws/tts`，不走本机声卡
- 本轮未结束再 ask 仍返回 409 `turn_in_progress`

---

## 6. 上线

重启讲解员进程后生效。展项须在 ask **之前**连上 `ws://主机:8848/ws/tts` 并保持不断。

正常日志应类似：

```text
已推 asr / tts_start / llm_answer
⏭ 忽略游客 TTS（机器人正在推流）   # 若前端仍回声
开始逐句 TTS / 缓存命中按播放节奏推流
等播报结束再 tts_end
已推 tts_end  tts_wall≈pcm 时长
```

不应再出现：robot 推流过程中夹一行 `🤖 用户问题: 我是克罗德…`（那是讲稿被当成游客新问题）。
