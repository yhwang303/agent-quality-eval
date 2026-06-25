# Agent Quality Eval — 产品化技术审查报告

**版本**: 0.1.19  
**审查日期**: 2026-06-23  
**审查范围**: 架构演进、Hook 可靠性、产品化缺口与改进建议

---

## 1. 架构演进分析：v0.1.17 vs 当前版本

### 1.1 v0.1.17 的本质问题

v0.1.17 的评审路径是：

```
主 Agent 完成任务 → 落盘 cot.json → 自定义子 Agent 读取并分析 → 输出 Eval
```

用户的质疑是正确的：**这与传统 LLM Judge 没有本质区别**，仅仅是把"裁判"换成了"子 Agent"。

核心症结在于：
- **数据是静态的**：子 Agent 拿到的 `cot.json` 是已完成任务的快照，无法感知过程中的动态风险
- **时序失真**：等到任务结束后再评审，已经无法对高风险行为发出实时预警
- **与 LLM Judge 的差距仅在数据结构上**：cot.json 比原始对话记录更结构化，但本质上仍是"事后验尸"
- **无法干预**：主 Agent 已完成任务，评审结论只能作为历史记录，无法影响本轮行为

### 1.2 当前版本的实质性改进

当前版本引入了双层评审架构：

```
主 Agent 运行中 ─── IDE Hook Pulse ──→ Live Supervisor (实时写入 state.json)
                                              ↓ (每12次事件或失败事件触发 LLM)
                                         model_snapshots[]
主 Agent 结束 ────── Stop Event ───────→ Turn Critic (读取 cot.json + live state)
                                              ↓
                                         turn_N.json (最终评审报告)
```

**真实带来的改变**：

| 维度 | v0.1.17 | 当前版本 |
|------|---------|---------|
| 评审触发时机 | 任务结束后 | 任务过程中持续 + 结束后汇总 |
| 风险发现时机 | 事后 | 实时（每次 hook pulse）|
| 数据维度 | cot.json 静态快照 | 动态事件流 + 累积状态 |
| 风险等级追踪 | 无 | low/medium/high 实时更新 |
| 跨轮次状态 | 无 | turn_index_approx 近似计数 |
| LLM 调用策略 | 单次全量分析 | 节流调用（12事件间隔+45s冷却）|

**当前版本仍有的局限**：
- Live Supervisor 是**只读监督**，无法实时打断主 Agent 或修改其行为
- `turn_index_approx` 是事件计数近似值，非精确轮次号（依赖 `UserPromptSubmit` 计数推断）
- 最终评审仍需等待 `cot.json` 落盘（最多等待 45 秒），存在竞争窗口

---

## 2. Hook 可靠性分析

### 2.1 各 IDE Hook 触发机制对比

| IDE | Hook 配置文件 | 触发协议 | stdin 格式 | 注入机制 |
|-----|-------------|---------|-----------|---------|
| Claude Code | `~/.claude/settings.json` | `hooks.<Event>[]` | JSON payload | `claude_hooks_merger.py` 注入 |
| Codex | `~/.codex/hooks.json` | `hooks.<Event>[].hooks[]` | JSON payload | `codex_hooks_merger.py` 注入 |
| Cursor | `~/.cursor/settings.json` | 同 Claude 结构 | JSON payload | `cursor_hooks_merger.py` 注入 |
| CodeBuddy | 独立配置路径 | 类似结构 | JSON payload | `codebuddy_hooks_merger.py` 注入 |
| VSCode | 扩展配置 | 不同机制 | 不定 | 部分支持 |

### 2.2 Codex Hook 不触发的根因分析

结合代码结构和历史问题，Codex hook 不触发的高概率原因：

**问题 1：双层嵌套结构差异**

Claude Code 的 hooks 结构：
```json
{ "hooks": { "Stop": [{ "command": "python ..." }] } }
```

Codex 的 hooks 结构（多一层 `hooks[]`）：
```json
{ "hooks": { "Stop": [{ "hooks": [{ "type": "command", "command": "python ..." }] }] } }
```

`codex_hooks_merger.py` 正确处理了这个差异（`_entry_from_hookentry` 生成内层 `{"type","command"}` + 外层 `{"hooks": [...]}` 包装），但如果 Codex 版本更新导致结构变化，注入内容会被静默忽略。

**问题 2：`_strip_owned_from_group` 逻辑有缺陷**

```python
def _strip_owned_from_group(group: dict[str, Any]) -> dict[str, Any] | None:
    hooks = group.get("hooks")
    kept = [h for h in hooks if not is_codex_owned_command(h.get("command", ""))]
    if not kept:
        return None  # 整个 group 被删除
    group["hooks"] = kept  # 直接修改了传入的 group（copy.deepcopy 在上层）
    return group
```

这里直接修改了 `group`，虽然上层有 `copy.deepcopy`，但如果 `copy.deepcopy` 没有覆盖到所有路径，可能导致状态污染。更重要的是，整个 group 如果只包含自己的 hooks 就会被删除，重新注入时会追加到末尾——如果 Codex 有顺序依赖，可能导致执行异常。

**问题 3：`continue: true` 输出路径问题**

`agent_critic_hook.py` 在 `main()` 结束时写 `{"continue":true}`，但这在 Python hook 中可能不是 Codex 所需的格式。Codex 的 hook 执行环境是否会读取 stdout 取决于具体版本实现，若 Codex 期望 exit code 而非 stdout，则这个机制无效。

**问题 4：Python 可执行路径解析链不可靠**

```python
py = os.environ.get("COT_PYTHON") or _read_runtime_python() or sys.executable or "python"
```

- 在 Codex（Electron 沙箱进程）中，`sys.executable` 可能指向 Codex 自身的 Node 环境，而非用户的 Python
- 最终降级到裸 `"python"` 字符串，在 Windows 环境下若 PATH 未配置会直接失败
- 失败是静默的：`Popen` 异常被 `except Exception` 捕获后仅写日志，不影响 hook 返回值

**问题 5：`session_id` 提取失败时的降级行为**

```python
session_id = args.session_id or _pick(payload, ("session_id", "sessionId", ...))
```

若 Codex 发来的 payload 结构变化导致无法提取 `session_id`，`cmd` 中没有 `--session-id`，导致 critic runner 无法关联会话，报告会落到 "unknown" 目录而非正确会话目录，前端查询失败。

### 2.3 各 IDE Hook 可靠性评级

| IDE | 整体可靠性 | 主要风险点 |
|-----|----------|----------|
| Claude Code | ★★★★☆ | Python 路径解析；cot.json 竞争窗口 |
| Codex | ★★★☆☆ | 双层结构脆弱；session_id 提取；Python 路径 |
| Cursor | ★★★★☆ | 与 Claude 结构相同，风险类似 |
| CodeBuddy | ★★★☆☆ | session_id 需加 "codebuddy-" 前缀（已处理）；依赖特有字段 |
| VSCode | ★★☆☆☆ | 非原生 hook 机制，集成程度最低 |

---

## 3. 产品化缺口审查

### 3.1 高危缺口

**[HIGH-1] cot.json 等待超时后评审静默失败**

`find_cot_file()` 轮询 45 秒后若 cot.json 未落盘，critic runner 仅记录日志，不写任何报告文件。前端因此会陷入"正在生成评估报告"状态直到轮询超时，然后显示"未找到报告"，用户无法区分是"Hook 没触发"还是"COT 提取失败"还是"评审模型出错"。

**[HIGH-2] 单点故障：runtime.json 未初始化时全链路崩溃**

`data_root()` 和 `_read_runtime_python()` 都依赖 `~/.agent-cot/runtime.json`。首次安装如果没有引导用户创建这个文件，所有后续操作都会降级到默认路径，且 Python 可执行路径解析可能完全失败。缺少明确的"未初始化"检测与用户提示。

**[HIGH-3] Live Supervisor 与 Turn Critic 的状态合并无验证**

`critic.py` 在 `_build_report()` 中将 `live_critic_state` 直接合并为 `base["live_supervisor"]`，但没有校验 `schema_version` 是否匹配，也没有处理 live state 的 `session_id` 与当前会话不一致的情况（例如 `_safe_name()` 对特殊字符的不同处理可能导致路径不匹配）。

### 3.2 中危缺口

**[MED-1] `_should_reuse_existing_report()` 的 600 秒时间窗口会阻塞手动重跑**

当评审处于 `status=running` 状态时，10 分钟内重复触发的 hook 会被跳过（除非 `source_event="api-rerun"`）。但如果第一次 critic 进程意外挂死（无 Stop 事件），该会话的自动评审将被永久阻塞，用户也无法通过前端触发重跑（除非前端明确传递 `source_event=api-rerun`）。

**[MED-2] `turn_index_approx` 不准确导致轮次快照文件命名混乱**

Live Supervisor 用 `UserPromptSubmit` 事件计数近似推算轮次号，但：
- 子 Agent 触发的 `UserPromptSubmit` 也会计数，导致主轮次计数偏高
- 若 `SessionStart` 先于 `UserPromptSubmit` 触发，初始值为 0，`turn_0.json` 不会被写入（`_state_turn_index` 中 `value > 0` 才写）

**[MED-3] model_snapshots 上限仅 6 条，长会话信息丢失**

`_bounded_append(snapshots, ..., 6)` 保留最近 6 次模型快照，对于超过 72 个事件（6 × 12）的长会话，早期关键的风险判断会被滚动丢弃，导致最终评审报告中的 `live_supervisor.model_snapshots` 只能反映会话末尾状态。

**[MED-4] 前端轮询时间窗口（5 分钟）与 critic 等待时间（45s）+ 模型调用时间不匹配**

实际评审链路：
```
hook触发(0s) → spawn子进程(~1s) → find_cot_file等待(最多45s) → LLM调用(10~30s) → 写报告(~1s)
```

总耗时最坏情况约 77 秒。前端 5 分钟轮询理论上足够，但若 cot.json 提取本身有问题（如 IDE 未写入），则 45s 超时后 critic 静默退出，前端会继续轮询直到 5 分钟结束，整个过程用户看到的是"正在生成"但实际已经失败。

**[MED-5] 多进程写入竞争**

`write_live_critic_state()` 使用 `tmp → os.replace` 原子写，但 `turn_tmp` 和主 `state.json` 是两次独立写操作。若 Windows 文件系统锁导致第一次写成功、第二次写失败，会出现 `state.json` 与 `turn_N.json` 内容不一致的状态。

### 3.3 低危缺口/体验问题

**[LOW-1] `_observation_for_event` 未覆盖的事件类型**

当前仅覆盖了约 15 种事件类型，其他事件（如 Cursor 特有的事件名、CodeBuddy 特有事件）会返回 `None`，既不记录观察也不计入风险。未来新 IDE 接入时，风险事件会被静默丢弃。

**[LOW-2] Codex hook 的 `is_codex_owned_command` 仅匹配文件名**

```python
_CODEX_OWNED_NAMES = {"codex_stream_hook.py", "codex_sidecar_collector.py", "agent_critic_hook.py"}
```

`agent_critic_hook.py` 与 Claude hook 重名，若用户手动在 Codex 中添加了其他名为 `agent_critic_hook.py` 的脚本，会被误删。应使用完整路径匹配而非仅文件名。

**[LOW-3] 日志文件无轮换机制**

`critic-hook.log` 和 `critic-runner.log` 是追加写入，长期运行会无限增长，在 Windows 上大文件追加可能造成性能问题。

**[LOW-4] `_deterministic_structured()` 的中文内容无法国际化**

降级评审报告中的分析文本是硬编码中文，如果产品未来需要支持非中文用户，这部分无法适配。

---

## 4. 改进建议（优先级排序）

### P0 — 必须修复

**[FIX-1] 统一 Python 可执行路径验证**

在 `_spawn()` 中增加 Python 可用性检查：
```python
# 在 spawn 前验证 python 可用
import shutil
resolved_py = shutil.which(py)
if not resolved_py:
    _log("python_not_found", python=py)
    return  # 早退，避免 Popen 失败被静默忽略
py = resolved_py
```

**[FIX-2] 评审失败时写入明确的错误状态报告**

`find_cot_file()` 超时后，应写入 `status=error` 的占位报告文件，前端得以区分"未开始"、"进行中"、"失败"三种状态：
```python
# find_cot_file 超时后
_write_error_report(session_id, agent_type, "cot_file_not_found", source_event)
```

**[FIX-3] 为 Codex hook 增加触发验证机制**

在 `agent_critic_hook.py` 成功 spawn 后，写入一个 `hook_fired_<session_id>.flag` 标记文件。前端可查询此文件来区分"Hook 从未触发"与"Hook 触发但评审失败"，这直接解决了用户反馈的 Codex hook 不触发无法感知的问题。

### P1 — 近期修复

**[FIX-4] 修复 `_strip_owned_from_group` 的 in-place 修改**

```python
def _strip_owned_from_group(group: dict[str, Any]) -> dict[str, Any] | None:
    new_group = dict(group)  # shallow copy
    hooks = new_group.get("hooks")
    if not isinstance(hooks, list):
        return new_group
    kept = [h for h in hooks if not (isinstance(h, dict) and is_codex_owned_command(h.get("command", "")))]
    if not kept:
        return None
    new_group["hooks"] = kept
    return new_group
```

**[FIX-5] 强化 session_id 提取失败时的处理**

当无法从 payload 提取 session_id 时，应生成一个基于时间戳的临时 ID 并记录 `source=generated`，而非完全省略 `--session-id` 参数：
```python
session_id = session_id or f"auto-{int(time.time())}"
```

**[FIX-6] `_should_reuse_existing_report` 增加进程存活检查**

对于 `status=running` 的报告，应检查写入时间戳是否超过阈值（当前 600s），若超过直接视为挂死状态并允许重跑，不需要前端显式传 `api-rerun`。

### P2 — 中期优化

**[OPT-1] 事件覆盖白名单改为黑名单 + 通用规则**

`_observation_for_event` 当前是显式枚举，建议补充通用规则：
```python
# 任何包含 "Error" 或 "Fail" 的事件名自动视为高风险
if "Error" in event or "Fail" in event or "Denied" in event:
    return "high", f"Risk event: {event}"
```

**[OPT-2] 增加 model_snapshots 数量上限，或改为存储摘要索引**

将 6 条上限提升到 20 条，或改为记录时间戳+摘要的轻量索引，完整数据按需从 turn 文件读取。

**[OPT-3] `critic-hook.log` 和 `critic-runner.log` 增加文件大小轮换**

```python
from logging.handlers import RotatingFileHandler
# 或简单实现：open 时检查文件大小，超过 10MB 则 rotate
```

**[OPT-4] Live Supervisor 中文 LLM Prompt 改为可配置**

将 prompt 模板提取到 `settings.json` 或 `critic_settings.json`，支持用户自定义语言与评审重点：
```json
{
  "live_critic_prompt_template": "You are a ...",
  "live_critic_language": "zh-CN"
}
```

### P3 — 长期架构演进

**[ARC-1] 真正的实时干预能力**

当前架构是"只读监督"。若需要实现真正的实时干预（如高风险操作前暂停），需要在 hook 中根据 live state 的 risk_level 返回不同的 `continue` 响应：
```json
// 当 risk_level=high 时
{"continue": false, "message": "高风险操作已被 Live Critic 暂停，请确认"}
```
这需要 IDE 支持 hook 返回 `continue: false`（Claude Code 目前支持，Codex 需验证）。

**[ARC-2] Turn 级别精确追踪替代近似计数**

将 `turn_index_approx` 改为从 IDE 传来的精确 turn ID（Claude Code hook payload 中包含 `turn_id` 字段），提升轮次快照文件的准确性。

**[ARC-3] 统一 hook 健康检查 endpoint**

在 CLI 中增加 `agent-cot doctor --ide=codex` 命令，自动验证：
1. Python 可执行路径是否可用
2. hooks.json 格式是否正确
3. 发送测试 hook event 并验证 critic 能否正常响应
4. 检查 runtime.json 是否存在且格式正确

---

## 5. 总结

当前版本（0.1.19）相对 v0.1.17 **有实质性架构升级**：从纯事后评审升级为"运行中实时观察 + 结束后最终评审"的双层监督，这不仅是 LLM Judge 范式的改进，而是引入了过程可观测性和风险动态追踪能力。

但产品化层面仍存在若干关键问题：
- **Hook 可靠性不透明**：失败是静默的，用户无法感知"Hook 从未触发"与"评审在后台失败"的区别
- **错误传播链路不完整**：任何环节失败后的状态报告缺失，前端只能呈现模糊的"生成中"
- **Codex 集成最脆弱**：双层 hook 结构、Python 路径解析、session_id 提取三重风险叠加

优先级最高的改进是：Hook 触发验证标记（FIX-3）、Python 路径验证（FIX-1）、失败状态报告写入（FIX-2）。这三项可以将当前最大的用户体验盲区消除，同时不需要对现有架构做破坏性修改。
