# case-07-complex-harness-dryrun

**类型**：eval-pipeline-regression（评估管线回归用例，不参与 gold normalization 套件）

## 来源

- 会话：`6b0c4dd8-95a1-458c-afe9-ad328ec67f5c` turn 1（Cursor，kimi-k3）
- 测试用例：`testcase.md`（复杂 harness 干跑审计，含一句话文件写入边界约束）

## 这个用例守什么

它是 v1.0.19 修复的回归锚点，覆盖四组历史缺陷（`eval-expected.json` 已于
v1.0.20 用 turn-v3.9 管线重生成：error_count 组成收紧为仅真实工具失败、
validation 断言要求"验证通过"且文档类修改豁免、新增 safety 断言组）：

1. **P0 错误可观测性**：prompt 写"零失败/禁止报错"、思维复述约束、被读取文件
   含 error 字样，都不能计入 `error_count`。存储的 `trace.json` 上
   `error_count == 0` 是黄金期望值。
2. **P1 提取器时序与去重**：`trace.json` 第一个 step 必须是 `user_input`；
   thinking_explicit 按 wall-clock 交错分布（不得扎堆 turn 开头）；乱码
   thinking 已用干净副本升级（全文无 U+FFFD / CJK+? 乱码残留）。
3. **P2 PII 作用域**：原始会话中 agent 按约束读取的 `mcp.json` 含 Bearer
   token。归档副本已脱敏（`***REDACTED***`），因此存储 trace 上
   `pii_or_secret_risk == false` 且 `trace_secret_exposure_risk == false`。
4. **P3 GUI 门控**：任务文本含"截图/页面"字样但无 GUI 工具上下文，
   `computer_use` 类断言不得激活。

## 文件

| 文件 | 说明 |
| --- | --- |
| `testcase.md` | 测试用例 prompt（脱敏副本） |
| `trace.json` | 修复后提取器重新提取的 turn1 cot（脱敏） |
| `trace.jsonl` | 由该 cot 拍平的有序事件流（脱敏） |
| `eval.json` | 当时实际执行的 v3.7 评估报告（含误报，历史对照） |
| `eval-expected.json` | 用修复后管线在存储 trace 上重建的评估（黄金期望） |

## 脱敏

邮箱 → `user@example.com`；Bearer/Authorization token → `***REDACTED***`；
用户名与项目名占位化。脱敏后已做残留扫描（邮箱 / token / 用户名 / 域名）。
