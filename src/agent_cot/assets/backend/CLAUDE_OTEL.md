# Claude Code 原生 OpenTelemetry 集成

> v0.16.0 新增。把 Claude Code 自带的 OTel 上报 直接喂给 dashboard 后端。
> 与 `transcript` / `hooks` 是**三条独立通道**，本通道是 Claude 进程**亲口
> 上报的官方真值**，权威性最高，不需要从转录推断。

---

## 一、架构

```
Claude Code 进程（用户配 OTEL_*）
     │
     │ OTLP/HTTP/JSON
     │ POST http://localhost:8765/v1/{logs,metrics,traces}
     ▼
agent-dashboard FastAPI 后端
  services/claude_otel_receiver.py
     │
     │ 按 session.id 分桶落盘
     ▼
~/.claude/state/otel/<session_id>/
  events.jsonl     # logs/events 通道
  metrics.jsonl    # metrics 通道
  traces.jsonl     # traces 通道（需要开 beta）
     │
     │ GET /api/sessions/<sid>/otel
     ▼
前端 ClaudeOtelPanel（DetailPanel.tsx 内）
```

零 collector 部署 —— FastAPI 直接当 OTel collector 用。所有数据可二次拿
出来发给 Phoenix / Langfuse / SigNoz / Honeycomb。

---

## 二、一键开启

**项目级别**（推荐，仅本项目）：编辑 `.claude/settings.json` 加 `env` 字段：

```json
{
  "env": {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
    "OTEL_LOGS_EXPORTER": "otlp",
    "OTEL_METRICS_EXPORTER": "otlp",
    "OTEL_EXPORTER_OTLP_PROTOCOL": "http/json",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:8765",
    "OTEL_LOG_USER_PROMPTS": "1",
    "OTEL_LOG_TOOL_DETAILS": "1",
    "OTEL_METRIC_EXPORT_INTERVAL": "10000",
    "OTEL_LOGS_EXPORT_INTERVAL": "5000"
  }
}
```

> ⚠️ Claude Code 启动时**一次性读取** settings.json 进内存，改完必须**重启
> Claude Code 进程**才会生效。

可选（开 traces beta，会出 `claude_code.interaction` 树）：

```json
"CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1",
"OTEL_TRACES_EXPORTER": "otlp"
```

可选（连 API 请求/响应原始 body 都要——含工具参数、对话历史、token 详细
切片，**注意：thinking 内容会被官方主动 redact，OTel 也拿不到**）：

```json
"OTEL_LOG_RAW_API_BODIES": "1"
```

---

## 三、验证流程

### 1. 后端必须在跑

```powershell
# 健康检查
curl.exe http://127.0.0.1:8765/api/health
# {"status":"ok"}
```

### 2. 重启 Claude Code，触发一次对话

随便发条 prompt，让 Claude 调几个工具。

### 3. 看落盘

```powershell
ls $env:USERPROFILE\.claude\state\otel
# 期待看到：<session_id>/events.jsonl, metrics.jsonl, traces.jsonl
```

### 4. 看后端 API

```powershell
curl.exe http://127.0.0.1:8765/api/otel/sessions
# {"sessions":[{"session_id":"...","events_bytes":...,...}]}
```

### 5. 看前端

打开 dashboard，进入 Claude session 详情。`📡 Claude Code 原生 OTel`
section 会出现，包含：

- KPI 行：input/output tokens · cache R/W · cost · events/metrics/spans 计数
- 模型分布芯片
- Prompt 时间线（按 `prompt.id` 关联，每个 prompt 卡内含其触发的 api_request
  和 tool_result）
- Metrics 总览（折叠）
- Traces 树（折叠，需要开 beta）

---

## 四、可观测信号清单

| 信号 | 端点 | 主要数据 |
| --- | --- | --- |
| **Metrics** | POST /v1/metrics | `claude_code.session.count` · `claude_code.token.usage` · `claude_code.cost.usage` · `claude_code.lines_of_code.count` · `claude_code.code_edit_tool.decision` · `claude_code.active_time.total` |
| **Events** | POST /v1/logs | `claude_code.user_prompt` · `claude_code.api_request` · `claude_code.api_error` · `claude_code.tool_result` · `claude_code.tool_decision` · `claude_code.permission_mode_changed` · `claude_code.mcp_server_connection` · `claude_code.api_request_body` (开 RAW) · `claude_code.api_response_body` (开 RAW) |
| **Traces** | POST /v1/traces | `claude_code.interaction` 根 → `claude_code.llm_request` / `claude_code.tool` / `claude_code.hook` |

**关联属性**：所有事件都带 `prompt.id` UUID —— 能把 user_prompt 触发的所有
API/tool 串到一起。前端 panel 就是按它分组的。

---

## 五、与现有通道的关系

| 通道 | 来源 | 用途 |
| --- | --- | --- |
| **Transcript** (`~/.claude-internal/projects/.../*.jsonl`) | Claude 自己写 | 含完整 message content + thinking 块（如果模型支持） |
| **Hooks** (`~/.claude/state/events/*/events.jsonl`) | `claude_stream_hook.py` 抓取 | 捕获 27 个 hook 事件，含 PreToolUse / PostToolUse / SubagentStart / Compact 等 transcript 不暴露的内部事件 |
| **OTel** (`~/.claude/state/otel/*/`) | Claude 进程主动 OTLP 推送 | **官方真值**：精确的 token/cost/duration/TTFT/retry/prompt.id 关联 |

三者**互不替代**：
- Transcript 是唯一带消息正文 + thinking 块的通道
- Hooks 是唯一带"内部生命周期事件"的通道
- OTel 是唯一带"精确量化指标"的通道（cost/tokens/timings 都是 API 真值）

cot-extractor 当前主链路仍以 transcript 为基础。OTel 数据在前端独立 section
展示，**不污染** SessionCoT 主结构。后续如需把 OTel 当作 token/cost 的权威
来源（取代从 transcript 估算），可在 `cot_otel_enricher.py` 里加 OTel 读取。

---

## 六、常见排查

### Q: 重启 Claude 了，settings.json 也加了 env，但 `~/.claude/state/otel/` 是空的

可能原因：
1. **后端没跑** —— 先 `curl http://127.0.0.1:8765/api/health`
2. **端口被防火墙拦** —— Windows Defender 可能拦本地高位端口；改成 `OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:8765` 试试（用 127.0.0.1 而不是 localhost）
3. **Claude Code 没读到 settings.json** —— 检查项目根目录有没有 `.claude/settings.json`，并且当前工作目录是这个项目
4. **Metrics 默认 60s 才推一次** —— 如果只是测试发了一句话，等 1 分钟再看；上面 settings 里我已经把间隔改成 10s/5s，但生效要重启
5. **空 prompt** —— 没触发任何 api_request 时，自然不会有 events

### Q: 后端日志报 `JSONDecodeError`

Claude 也支持 `OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf`，但本 receiver
只解 JSON。**必须设置 `OTEL_EXPORTER_OTLP_PROTOCOL=http/json`**，不然会
发 protobuf 二进制过来，receiver 读不懂。

### Q: 我想把数据再转发到 Phoenix / Langfuse

写一个 Python 后处理脚本，读 `~/.claude/state/otel/<sid>/*.jsonl` 用
opentelemetry SDK 重放即可。或者改 receiver，落盘的同时也走真实 SDK
forward —— 把 receiver 当 OTel Collector 的简化版用。

---

## 七、隐私 / 合规提示

- `OTEL_LOG_USER_PROMPTS=1` 会把用户输入的 prompt 完整记录到 `events.jsonl`
- `OTEL_LOG_TOOL_DETAILS=1` 会记录 Bash 命令、文件路径等
- `OTEL_LOG_RAW_API_BODIES=1` 会记录完整对话历史（最敏感）
- 所有数据都落在**用户本机** `~/.claude/state/otel/`，**不会上传任何远端**
- 想清理：`rm -rf ~/.claude/state/otel/<session_id>` 即可
