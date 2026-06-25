# Codex CLI Critic Subagent 实战教程

> 用一个独立的"评审 subagent"，对主 Codex agent 的每轮工作做实时 turn-level 评估，最终产出结构化报告供 `agent-quality-eval` 直接消费。
>
> 业界对应做法：Agent-as-a-Judge（[arXiv 2410.10934](https://arxiv.org/abs/2410.10934)）、AutoGen CriticAgent、OpenAI CriticGPT、Reflexion、CRITIC、LangGraph Supervisor、Patronus / Galileo Runtime Guardrails。

---

## 一、前置认知

### 1.1 "实时"的真实含义

真·同步实时（每个 token 都评）会阻塞主 agent，业界没人这么做。**主流"实时" = turn-level 异步 critic**：

- 每一轮（user → assistant → tool → assistant ...）结束后触发一次 critic
- Critic 独立进程跑，不阻塞下一轮
- 或者按"重要节点"触发（文件被修改、命令被执行、计划被更新）

本教程实现的就是 turn-level 异步 critic 模式。

### 1.2 Codex CLI 的局限

Codex CLI **没有 Claude Code 那种 `.claude/agents/*.md` 的原生 subagent 系统**，因此采用以下变通：

- 用 `notify` hook 在主 agent 完成一轮时触发外部脚本
- 脚本 fork 一个独立的 `codex exec` 子进程作为 critic
- Critic 跑专用 profile（低成本模型、只读沙盒、关响应存储）
- 评审报告以 JSON 落盘到统一目录，供下游产品消费

---

## 二、项目结构

```
~/.codex/
├── config.toml                    # Codex 配置（含 notify hook + critic profile）
├── agents/
│   └── critic-prompt.md           # critic 评审提示词
└── critic/
    ├── run-critic.py              # hook 脚本：解析事件 → 调 critic
    ├── reports/                   # 评审报告输出目录（按 session_id-时间戳命名）
    └── transcripts/               # 主会话 transcript 缓存（可选）
```

---

## 三、Step 1：写 critic 评审提示词

`~/.codex/agents/critic-prompt.md`

```markdown
你是一名 Agent 行为评审员（critic）。下面给你的是主 agent 刚刚完成一轮的完整原始数据，
请按以下 5 维 + 2 诊断输出一份 JSON 评审报告。

【评审维度】
1. task_success: resolved | partial | unresolved
2. tool_use: correct | suboptimal | wrong
3. action_advancement: on_track | drift | redundant | lost
4. instruction_following: yes | partial | no
5. faithfulness: grounded | partial | hallucinated

【诊断侧栏】
6. efficiency: { tokens, elapsed_seconds, tool_calls_total, tool_calls_failed, verdict }
7. reliability: { has_critical_error: bool, note: "..." }

【输出 JSON】
{
  "task_success": "...",
  "tool_use": "...",
  "action_advancement": "...",
  "instruction_following": "...",
  "faithfulness": "...",
  "efficiency": { "tokens": 0, "elapsed_seconds": 0, "tool_calls_total": 0, "tool_calls_failed": 0, "verdict": "normal" },
  "reliability": { "has_critical_error": false, "note": "" },
  "summary": "120-220字 一段完整自然语言评审"
}

【影响导向原则】
- 单步失败被恢复 → 不算扣分
- 只有影响最终交付时才给负面结论

只返回 JSON，不要 Markdown 包裹。
```

---

## 四、Step 2：写 hook 脚本

`~/.codex/critic/run-critic.py`

```python
#!/usr/bin/env python3
"""Codex notify hook: 主 agent 每完成一轮就把数据投给 critic。"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home() / ".codex" / "critic"
HOME.mkdir(parents=True, exist_ok=True)
REPORTS = HOME / "reports"
REPORTS.mkdir(exist_ok=True)
PROMPT_PATH = Path.home() / ".codex" / "agents" / "critic-prompt.md"
PROMPT = PROMPT_PATH.read_text(encoding="utf-8")


def main():
    # 防止 critic 进程再触发 notify 导致递归
    if os.environ.get("CODEX_DISABLE_NOTIFY") == "1":
        return

    # Codex 把事件 JSON 作为 stdin / argv[1] 传进来
    raw = sys.stdin.read() if not sys.stdin.isatty() else (sys.argv[1] if len(sys.argv) > 1 else "{}")
    try:
        event = json.loads(raw)
    except Exception:
        event = {"raw": raw}

    # 只在感兴趣的事件触发
    if event.get("type") not in {"agent-turn-complete", "tool-result", "session-end"}:
        return

    session_id = event.get("session_id", "unknown")
    turn_data = {
        "session_id": session_id,
        "user_message": event.get("user_message"),
        "assistant_turns": event.get("assistant_turns", []),
        "tool_results": event.get("tool_results", []),
        "final_response": event.get("final_response"),
        "runtime_metrics": event.get("metrics", {}),
    }

    critic_input = (
        PROMPT
        + "\n\n【主 agent 本轮原始数据】\n"
        + json.dumps(turn_data, ensure_ascii=False, indent=2)
    )

    # fork 一个独立 Codex 进程做评审；--profile critic 用低成本模型
    child_env = {**os.environ, "CODEX_DISABLE_NOTIFY": "1"}
    proc = subprocess.run(
        ["codex", "exec", "--profile", "critic", "-q", "--json", critic_input],
        capture_output=True,
        text=True,
        timeout=120,
        env=child_env,
    )

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_file = REPORTS / f"{session_id}-{ts}.json"
    out_file.write_text(
        json.dumps(
            {
                "event": event,
                "critic_stdout": proc.stdout,
                "critic_stderr": proc.stderr[-2000:] if proc.stderr else "",
                "returncode": proc.returncode,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
```

- macOS / Linux：`chmod +x ~/.codex/critic/run-critic.py`
- Windows：脚本本身不需要可执行位，下一步用 `python` 显式调用即可

---

## 五、Step 3：配置 Codex

`~/.codex/config.toml`

```toml
# 默认 profile —— 主 agent，跑你的实际编程任务
model = "gpt-5"
approval_policy = "on-request"

# 通知 hook —— 每个事件都会调用这个命令
# Codex 会把事件 JSON 通过 stdin 传给脚本
# Windows 写法（注意路径用正斜杠或双反斜杠）：
notify = ["python", "-u", "C:/Users/<你>/.codex/critic/run-critic.py"]
# macOS / Linux 写法：
# notify = ["python3", "-u", "/Users/<你>/.codex/critic/run-critic.py"]

# critic 用的独立 profile —— 单独配低成本模型，避免高额 token
[profiles.critic]
model = "gpt-5-mini"
approval_policy = "never"        # critic 不需要交互
disable_response_storage = true  # critic 看到的是你的代码，关闭训练留存
sandbox_mode = "read-only"       # critic 不应该改你的代码
notify = []                      # 双保险：critic 进程内不再触发 hook
```

---

## 六、Step 4：跑起来

```bash
# 终端 1 —— 正常用主 agent
codex
> 帮我把 src/app.py 里的 foo 函数重构成异步

# 终端 2 —— 实时看 critic 报告（macOS / Linux）
tail -f ~/.codex/critic/reports/*.json

# Windows PowerShell
Get-Content -Path "$HOME\.codex\critic\reports\*.json" -Wait -Tail 50
```

每完成一轮，`~/.codex/critic/reports/` 目录会多一份评审报告，结构化 JSON 直接可被你现有 `agent-quality-eval` 后端加载展示。

---

## 七、Step 5：把报告接进 `agent-quality-eval`

新流程：

1. 主 agent（Codex）正常工作
2. critic hook 异步产出 JSON 评审到 `~/.codex/critic/reports/`
3. `agent-quality-eval` 后端加一个 watcher（如 Python `watchdog` 库），监听该目录
4. 新文件出现 → 解析 JSON → 写入 trace 数据库 → 前端实时刷新

最小 watcher 示例：

```python
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

class CriticReportHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith(".json"):
            return
        # 解析 JSON，写入 agent-quality-eval 后端数据库
        ingest_critic_report(event.src_path)

observer = Observer()
observer.schedule(CriticReportHandler(), str(Path.home() / ".codex" / "critic" / "reports"), recursive=False)
observer.start()
```

至此就把现有"离线点击 Eval 按钮"模式升级为"主 agent 一边写代码、评审一边产出"的 sidecar 模式。

---

## 八、必须注意的坑

1. **Codex `notify` hook 触发时机有限** —— 只在 turn 结束、工具失败、需要审批等事件触发，**不会在每个 LLM token 上触发**。这正好对应业界 "turn-level critic" 的最佳实践。
2. **递归调用陷阱** —— critic 进程不要再触发 hook，否则会无限递归。本教程通过 `CODEX_DISABLE_NOTIFY=1` 环境变量 + critic profile 内 `notify = []` 双保险防止。
3. **token 成本** —— critic 每轮都跑一次，成本会翻倍。**一定用 mini 模型**（gpt-5-mini / haiku 类）做 critic，主模型用大的。
4. **隐私** —— critic 进程会读取主会话完整 transcript，务必 `disable_response_storage = true` 防止你的代码被留存训练。
5. **超时与失败** —— critic 子进程 `timeout=120s`，超时不会影响主 agent。建议 hook 脚本里加 try/except 兜底，所有异常都写到 stderr 落盘，不抛出。
6. **跨模型独立性更强** —— 想要更强的"第二意见"，可让 critic 走另一个厂商（如 Anthropic Claude），实现 cross-vendor judging，避免同模型偏见。

---

## 九、与 Claude Code 方案的对比

| | Codex CLI（本教程） | Claude Code |
|---|---|---|
| Subagent 原生支持 | ❌ 用 `notify` hook + `codex exec` 子进程 | ✅ `.claude/agents/critic.md` 一文件搞定 |
| 触发方式 | 事件驱动（hook） | 主 agent 通过 Task 工具显式调用 |
| 隔离性 | 独立进程，沙盒可控 | 同进程内子会话 |
| 上手成本 | 中（需写脚本 + 配 hook） | 低（10 行配置） |
| 适合场景 | 想要进程级隔离、跨厂商 critic | 想要最小化基础设施 |

如需 Claude Code 版本，可另写一份简化教程。

---

## 十、总结

- **是否可行**：完全可行，且已是业界主流方向之一（Agent-as-a-Judge）。
- **Codex 路径**：notify hook + `codex exec --profile critic` 独立进程。
- **核心交付物**：4 个文件 —— `critic-prompt.md` / `run-critic.py` / `config.toml` / watcher 接入代码。
- **下一步**：把 critic 报告 schema 与 `agent-quality-eval` 现有 trace 数据库对齐，实现 sidecar 实时展示。
