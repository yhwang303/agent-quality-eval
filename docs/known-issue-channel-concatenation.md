# 已知问题：Cursor 会话的两个观测通道被首尾拼接，而不是按时间归并

状态：**已确认，未修复**
发现于：1.0.12 调查「dedupe-thinking 报告的 330 个重复步骤是什么」时
影响：Cursor 会话的步骤顺序、trace 导出的时间线、turn 内的步骤计数

## 现象

`agent-cot dedupe-thinking` 的预演报告「3 个会话，共 330 个重复步骤」，但在前端
逐屏翻找看不到任何挨在一起的重复思考。原因是两份副本相隔 60~216 个 step，肉眼
不可能在滚动时把它们联系起来。

## 根因

一轮 Cursor 会话有两条观测通道：

- **transcript 通道** —— 从 Cursor 的 transcript 文件解析，产出 `thinking_inter`
  （metadata 只有 `{output_tokens: 0}`）和带 `tool_name` / `tool_use_id` 的
  `tool_execution`。
- **hook 通道** —— `afterAgentThought` 等 hook 实时推送，产出 `thinking_explicit`
  （metadata 带 `observed_at_ms` / `generation_id` / `model` / `thought_duration_ms`）
  和带 `observed_input` / `observed_output` 的 `tool_execution`。

两条通道各自只看到这一轮的一部分，而提取器把它们**首尾拼接**了：先放完整个
transcript 段，再把整个 hook 段接在后面。于是同一段真实时间在文件里出现两次。

## 证据（会话 22c4566e turn#1，共 353 步）

| | 步骤范围 | thinking 类型 | 工具分布 | 时间戳 |
|---|---|---|---|---|
| 前段 | 1–215 | `thinking_inter` ×57 | Grep 14 / Read 6 / TodoWrite 3 / WebFetch 3 | 无 |
| 后段 | 216–351 | `thinking_explicit` ×26 | Shell 53 / StrReplace 34 / Read 17 / Write 2 | 有 |

决定性的一点：后段的 `observed_at_ms` 跨度是 03:52:29 → 04:13:36，共 **21.1 分钟**，
而该轮的 `turn_duration_ms` 是 **1,266,962 毫秒 = 21.1 分钟**。后段覆盖的是**整轮**，
不是"后半段"。

后段的思考与前段一一对应，且保持相同次序：

```
后段 step[217] 03:52:38  ←→  前段 step[1]
后段 step[221] 03:53:20  ←→  前段 step[13]
后段 step[224] 03:53:35  ←→  前段 step[16]
后段 step[229] 03:53:54  ←→  前段 step[23]
```

正文逐字相同的 thinking 共 22 组，例如：

```
step#2   [thinking_inter]     "I need to understand the project structure first, then packa…"
step#218 [thinking_explicit]  ← 同一段话，逐字相同
```

工具调用则几乎不重合（签名重合的只有 6 个 `Read` 且入参为空），说明两条通道看到的
是**不同的工具子集**：transcript 侧是检索类，hook 侧是执行类。

## 受影响的会话

三条，全部是 Cursor，全部来自项目 `d-ai-ide-langfuse`
（`C:\Users\milkwang\.cursor\projects\d-ai-ide-langfuse`）：

| session_id | 轮数 | 时间范围 | 可去重的步骤 |
|---|---|---|---|
| `22c4566e-5d78-4819-92c9-9e0e5d70a165` | 37 | 2026-05-13 11:52 → 05-19 12:04 | 144 |
| `23681935-8c35-4859-b89f-f745b5c595ff` | 74 | 2026-05-11 11:56 → 05-13 11:08 | 32 |
| `e9e7d567-6f14-46ea-a3ef-df7e3c34c7a5` | 47 | 2026-05-26 10:45 → 06-04 21:31 | 122 |

（时间取自 hook 的 `observed_at_ms`，本地时区。）

## 为什么 dedupe-thinking 修不了它

`dedupe-thinking` 删掉的是前段的 `thinking_inter`、保留后段的 `thinking_explicit`
（后者 metadata 更全）。这能消掉重复计数，但拼接的顺序原封不动：前段仍然是一串没
有推理的工具调用，后段仍然是整轮的重放。跑它只会让这类会话变短，不会变对。

## 修复方向（未实施）

按时间把两条通道归并成一条流，而不是拼接。难点：

1. transcript 段**完全没有时间戳**（22c4566e turn#1 是 0/353），只能靠与 hook 段
   的正文对应关系反推位置。
2. 对应关系只在 thinking 上成立；两侧的 `tool_execution` 是不相交的集合，需要另找
   锚点（例如 `tool_use_id` 或工具名 + 入参）才能插到正确位置。
3. 归并后 `step_index` 要重排，`total_steps` / `thinking_depth` / 会话级聚合都要
   跟着重算。
4. extractor 有两份副本（`assets/cot-extractor/src/` 和 `assets/cot-extractor-src/`）
   必须同步，且存量数据需要一次迁移。

在此之前，这类会话的 trace 导出顺序不可信——导出本身是忠实的，它忠实地反映了存盘
数据里就已经错了的顺序。
