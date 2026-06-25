# Agent Quality Eval V3 技术报告

## 1. 背景与目标

本轮工作围绕 `agent-quality-eval` 的 turn-level eval 能力进行重构。初始问题是旧版 Eval 面板以泛化 `Quality Score` 为核心，指标高度主观，任意两个不相关 trace 也能被同一套宽泛评分比较，缺少对真实 Agent 行为、工具链路、最终交付和回归风险的专业判断。

重构目标包括：

- 参考 `D:\letsgoagenteval\Scripts\eval_pipeline_v3` 的 v3 思路，将评估从主观质量分改为声明式断言和可解释评估报告。
- 将 turn eval 改为面向真实 Agent trace 的综合断言体系，覆盖最终交付、执行完整性、工具使用、代码/研究/计划等常见任务形态。
- 保留 token 等关键 KPI，同时新增工具使用数量、工具类别、MCP/RAG/检索/搜索/浏览器等细分统计。
- 支持 A/B testing 与 regression detection，使两个 trace 可以作为 baseline/candidate 进行横向比较。
- 引入可配置 LLM Judge，但默认不强制启用；配置 TIMI AI 或 DeepSeek 后才进行语义评审。
- 从本地 CBM 长期记忆数据库抽取真实开发问题，构建可复用测试数据集。
- 每次功能迭代打包为独立版本 exe，方便安装验收。

## 2. 总体架构变更

### 2.1 后端 Eval 架构

原有 eval 输出以 `overall_score / quality_score` 为核心。重构后改为 v3 断言模型：

- `eval_version: "v3"`
- `overall_score` 保留兼容旧存储，但语义改为 `assertion_pass_rate`
- 新增 `assertion_set`
- 新增 `assertion_results`
- 新增 `assertion_groups`
- 新增 `pipeline.ab_testing`
- 新增 `pipeline.regression_gate`
- 新增 `judge.structured.review_markdown`

旧字段保留是为了避免已有存储和前端读取链路断裂，但 UI 不再以 `Quality Score` 命名或展示。

### 2.2 断言体系

v3 断言改为综合模板，不再显式在 UI 顶部展示粗糙的 `task_profile` 判定。断言覆盖以下方向：

| 分组 | 作用 | 典型断言 |
|---|---|---|
| 任务结果 | 判断是否有最终回答和可用交付 | `final-response-present`、`answer-min-length` |
| 执行完整性 | 判断是否有明显错误、是否采集耗时和 token | `no-error`、`token-usage-present`、`duration-present` |
| 工具使用 | 判断工具调用是否被统计、是否有工具错误、是否引用工具结果 | `tool-counts-present`、`tool-error-free`、`tool-result-referenced` |
| 代码交付 | 判断代码修改、验证、文件说明是否完整 | `file-edit-evidence`、`validation-after-edit`、`changed-files-mentioned` |
| 研究与证据 | 判断检索、来源、综合表达是否充分 | `retrieval-evidence`、`source-grounding`、`research-synthesis` |
| 计划执行 | 判断复杂任务是否有计划、计划是否与最终回答一致 | `plan-evidence`、`plan-adherence` |
| GUI/浏览器操作 | 判断浏览器/GUI 操作是否有最终状态证据 | `computer-use-observed`、`final-state-evidence-present` |

其中 GUI/浏览器相关断言已从固定断言改为动态适用，避免非 computer-use 任务出现无意义的 `final-state-evidence-present`。

## 3. LLM Judge 演进

### 3.1 初版 LLM Judge 问题

早期 LLM Judge 输出过于启发式，容易出现：

- 一段散文式评价，结构不稳定。
- 维度过多但信息密度低。
- 直接基于后加工 trace 判断，忽略原始数据流。
- 前端重复展示两份相同内容。
- 输出“部分解决”等用户明确不想看到的措辞。
- 在模型输出不合规时进入 error 兜底，兜底内容把 token、耗时、工具调用显示成 0。

### 3.2 v3.1 严格模板

按 `llm-judge-template-v3.md` v3.1，LLM Judge 改为严格 JSON schema：

- `summary`
- `metrics`
- `relevance`
- `instruction_following`
- `tool_use`
- `reasoning`
- `faithfulness`
- `task_completion`
- `overall_verdict`

后端负责：

- 校验顶层字段。
- 校验枚举。
- 校验字段长度。
- 校验禁用短语。
- 校验 `overall_verdict` 的确定性推导。
- 根据 JSON 确定性拼装 `review_markdown`。

该阶段提升了结构稳定性，但也暴露了新问题：校验过严导致真实模型输出很容易被拒收，进入兜底状态后体验变差。

### 3.3 token 和工具统计为 0 的问题

问题现象：

- LLM 评审区域显示 `0 tokens / 0.0s / 0 次工具调用`
- 状态为 `error`
- 内容显示“缺少足够证据判定完成”等兜底文案

根因：

- 模型输出未通过 v3.1 强校验。
- 后端进入 `_fallback_structured_turn_judge`。
- 旧 fallback 没有接入 `runtime_metrics`，默认构造了全 0 统计。

修复：

- `_run_optional_turn_judge` 在调用 judge 前构造 `expected_metrics`。
- `_fallback_structured_turn_judge(reason, runtime_metrics)` 接收真实统计。
- parse 失败、重试失败、兜底展示都保留真实 `total_tokens / elapsed_seconds / tool_calls_total / tool_calls_failed`。
- 前端旧缓存需要重新 Eval 才能看到新统计。

### 3.4 v3.2 宽松模板

最新模板改为 `v3.2`，设计取向从“强校验模型”转为“相信模型 + 原始数据”。

后端实现调整：

- LLM 只输出一个 JSON 对象。
- 每个维度为 `verdict + review`。
- 不再让 LLM 输出 `review_markdown`。
- 后端根据 JSON 字段确定性拼装 `review_markdown`。
- 不再做黑名单强拦截。
- 不再做引用格式强校验。
- 不再做影响导向反向匹配强校验。
- JSON 解析失败仅重试 1 次。
- 枚举非法时回退到接近的合法默认值。
- 超长 review 截断到 400 字。
- 不再在前端展示 `状态：error`。

当前 v3.2 JSON schema：

| 字段 | 说明 |
|---|---|
| `summary` | 120-220 字自然语言总结 |
| `efficiency` | `verdict + review`，评估 token、耗时、工具次数 |
| `relevance` | `verdict + review`，评估用户目标与最终回复相关性 |
| `instruction_following` | `verdict + review`，评估显式/隐式约束 |
| `tool_use` | `verdict + review`，评估工具选择、失败恢复与影响 |
| `reasoning` | `verdict + review`，评估推理轨迹 |
| `faithfulness` | `verdict + review`，评估最终声称与工具结果支撑 |
| `task_completion` | `verdict + review`，评估最终交付 |
| `overall_verdict` | 后端根据规则覆盖 |

`overall_verdict` 推导规则：

| 条件 | 结果 |
|---|---|
| `task_completion.verdict == resolved` 且 `instruction_following.verdict != no` | `resolved` |
| `task_completion.verdict == unresolved` 或 `instruction_following.verdict == no` | `unresolved` |
| 其它 | `partial` |

## 4. 前端 Eval 面板修复

### 4.1 移除泛化描述

删除了顶部 “V3 Assertion Method” 一类描述性文案，让 Eval 面板直接进入评估结果本身。

### 4.2 中文化与视觉优化

修复内容：

- Eval 分组、断言说明、失败原因改为中文展示。
- 二值断言不再用 100% 展示，改为通过/失败状态。
- 对勾和失败状态改为更明显的绿色/红色立体视觉。
- LLM 评审区域改为高亮框展示后端拼装的 `review_markdown`。
- 轻量解析 Markdown 标题和段落，避免 `**结论**` 等标记裸露。

### 4.3 工具统计

新增和优化了工具统计：

- 总工具调用数。
- 工具种类数。
- MCP 工具调用数。
- RAG 调用数。
- Retrieval 调用数。
- Search 调用数。
- Browser/GUI 调用数。
- Shell / Read / Write / Edit/Patch 等常见工具类别。
- 工具名调用明细。
- 工具错误按工具名展开。

用户反馈“不要写唯一工具”，因此前端改为直接显示工具使用数量和各类工具调用次数。

## 5. A/B Testing 与 Regression Gate

### 5.1 A/B 入口

前端增加 baseline/candidate 选择入口：

- 当前 trace 可设为 Base。
- 当前 trace 可设为候选。
- 选择后按钮有 active 反馈。
- 支持单个清空。
- 支持一键清空。
- 支持打开 A/B 子窗口进行横向对比。

### 5.2 A/B 子窗口

A/B 子窗口展示：

- baseline 与 candidate 的基本信息。
- 回归门禁状态。
- 改进/退化/阻断统计。
- 断言分组通过率对比。
- token 与工具使用对比。
- 断言差异明细。

### 5.3 柱状图修复

问题：

- 初版柱状图为横向条形图，空间利用差。
- Base 与候选颜色相近，对比不明显。

修复：

- 分组通过率改为竖向双柱图。
- 每个分组一张小竖柱图。
- Base 使用橙色系。
- 候选使用蓝色系。
- 顶部增加 legend。

### 5.4 明细 label 修复

问题：

- `Token 与工具使用` 只罗列数值，没标明哪列是 Base、哪列是候选。
- `断言差异明细` 只显示两个状态，缺少明确列名。

修复：

- `Token 与工具使用` 增加表头：指标 / Base / 候选。
- 每一行数值也增加行内 `Base` / `候选` 标签。
- `断言差异明细` 增加表头：断言 / Base / 候选 / 变化。
- 每一行通过/失败状态也增加行内标签。

## 6. API 设置与模型提供商

新增设置入口：

- 顶部按钮由“设置”改为“API 设置”。
- 支持 TIMI AI。
- 支持 DeepSeek。
- 支持选择对应模型。
- 支持输入 API Key。
- 未配置时 LLM Judge 跳过，不强制调用。
- 配置后 eval 默认可调用 LLM 评审。

该部分参考了 `D:\agent-memory` 中 OpenAI-compatible provider 的连接方式。

## 7. CBM 真实数据集

用户要求测试集不能只用合成样例，应从真实开发过程中的 bug 和问题构建。

数据来源：

- SQLite: `C:\Users\milkwang\.codebuddy-mem\codebuddy-mem.db`
- 主表：`session_summaries`
- 证据补充：`observations`

抽样项目：

- `agent-memory`
- `ai-ide-langfuse`
- `agent-quality-eval`
- `letsgoagenteval`
- `shadow-folk`

样例类别：

- `bugfix`
- `regression`
- `eval_ab`
- `tool_trace`
- `build_release`
- `research_or_diagnosis`

产物：

- `src/agent_quality_eval/templates/cbm_real_cases.yaml`
- `src/agent_quality_eval/templates/cbm_real_cases_evidence.jsonl`

数据集设计：

- YAML 可作为 eval config 直接加载。
- JSONL 保留候选来源、脱敏 hash、证据片段和选择理由。
- 强脱敏路径、邮箱、token、密钥、内部链接、人员信息。
- 首版目标 30-50 个真实样例。

## 8. 打包与运行问题修复

### 8.1 exe 启动后窗口消失

问题：

- 双击 exe 后弹出终端，一段时间后消失。
- 用户无法确定前端网站是否生成。

处理：

- 改为打包后进行 smoke test。
- 使用 `http://127.0.0.1:8801/` 检查前端是否返回 200。
- 每次打包后确认加载新 JS/CSS 资源。

### 8.2 旧进程堆积

问题：

- 多次安装/启动后后台出现多个 `agent-quality-eval-*` 进程。
- 可能导致前端展示旧内容或端口冲突。

修复：

- 每次打包/验证前停止旧 `agent-quality-eval*` 进程。
- 版本号自动递增。
- exe 命名带版本号，例如 `agent-quality-eval-0.1.9.exe`。

### 8.3 前端旧缓存

问题：

- 安装新 exe 后右侧 Eval 面板仍显示旧内容。

根因：

- 旧 eval report 已持久化在本地数据库。
- 前端优先读取历史缓存。

处理：

- 对旧版 eval cache 增加过滤。
- 明确提示用户旧报告需要重新点击 Eval 生成。
- 每次前端 build 后同步 `frontend/dist` 到 `src/agent_cot/assets/frontend-dist`，避免 exe 带旧静态资源。

## 9. 测试与验证

后端测试：

```bash
py -3.12 -m py_compile src\agent_quality_eval\evaluation\session_eval.py
$env:PYTHONPATH='src'; pytest tests\evaluation\test_eval_core.py -q
```

前端测试：

```bash
cd frontend
npm run build
```

打包：

```bash
py -3.12 -m PyInstaller --clean --noconfirm agent-quality-eval.spec
```

冒烟验证：

```bash
Start-Process dist\agent-quality-eval-<version>.exe
Invoke-WebRequest http://127.0.0.1:8801/
```

最近版本验证：

| 版本 | 后端测试 | 前端构建 | exe 冒烟 |
|---|---|---|---|
| 0.1.7 | 10 passed | 通过 | 200 |
| 0.1.8 | 10 passed | 通过 | 200 |
| 0.1.9 | 10 passed | 通过 | 200 |

## 10. 版本迭代矩阵

| 版本 | 主要目标 | 核心变更 | 修复的问题 | 验证结果 |
|---|---|---|---|---|
| 0.1.2 | 初版 v3 eval 改造 | 引入 `eval_version=v3`、`assertion_pass_rate`、`assertion_set`、`assertion_results` | 旧 `Quality Score` 主观、不可解释 | 后端测试通过 |
| 0.1.3 | CBM 真实数据集 | 从 `codebuddy-mem.db` 抽取真实开发案例，生成 YAML + JSONL | 测试集不真实、缺少历史缺陷样例 | YAML 可加载，JSONL 可审计 |
| 0.1.4 | 综合断言与工具统计 | 移除粗糙 task_profile 展示；新增工具类别统计、错误明细 | 非 computer-use 任务出现无关断言；工具统计不完整 | 前端构建通过 |
| 0.1.5 | LLM Judge 与 API 设置 | 增加 TIMI/DeepSeek 设置；LLM 评审基于原始数据流 | LLM 评审过于启发式；没有 API 配置入口 | API 设置可保存 |
| 0.1.6 | 交互与打包体验 | baseline/candidate active 状态、清空操作、版本化 exe | 选择 Base/候选无反馈；exe 名不带版本；后台进程堆积 | 打包成功 |
| 0.1.7 | v3 严格 LLM 模板 | 严格 JSON schema、重试、后端拼装 `review_markdown` | LLM 输出结构漂移；重复展示两份评审 | 10 passed，前端 build 通过，exe 200 |
| 0.1.8 | v3.1 统计修复 | fallback 接入真实 runtime metrics；token/工具不再为 0 | LLM error 兜底显示 `0 tokens / 0.0s / 0 工具` | 10 passed，前端 build 通过，exe 200 |
| 0.1.9 | v3.2 与 A/B 可视化 | 宽松 LLM schema；自然语言段落；竖向双柱图；Base/候选 label | A/B 图表横向且颜色不明显；明细无 Base/候选标签；LLM 强校验过度 | 10 passed，前端 build 通过，exe 200 |

## 11. 当前状态

当前最新 exe：

```text
D:\agent-quality-eval\dist\agent-quality-eval-0.1.9.exe
```

当前最新行为：

- Eval 面板使用 v3 断言结果。
- A/B 对比支持竖向双柱图。
- Token 和工具使用对比明确标记 Base/候选。
- LLM Judge 使用 v3.2 宽松自然语言模板。
- `review_markdown` 由后端拼装。
- 前端不再展示 `状态：error` 系统话术。

## 12. 后续建议

- 将 iWiki 同步接入正式 MCP，避免依赖浏览器页面编辑。
- 为 A/B 对比增加导出 JSON/Markdown 功能，便于审计。
- 为 CBM 数据集增加自动刷新命令，定期补充真实案例。
- 为 LLM Judge 增加 provider 调用日志摘要，便于定位 API 失败、模型格式漂移和耗时异常。
- 增加端到端 UI 截图测试，覆盖 Eval 面板、A/B 弹窗、API 设置、LLM 评审展示。
