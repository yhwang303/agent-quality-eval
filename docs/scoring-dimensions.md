# Agent 评估维度面板（无总分版）

> 取消跨维度加权综合分，改为分维度并列展示。每一维都能在 2 家以上业界框架找到对应概念，避免"独创无背书"。

---

## 设计原则

1. **不计算综合质量分**，不再展示 `quality_score`。每个维度独立报告。
2. **维度分两类**：核心面板（5 项，定性 verdict + 简要说明）+ 诊断侧栏（2 项，数字直显，不打分）。
3. **每项只用枚举值**（resolved / partial / unresolved 之类），不用 0–1 浮点分——避免分数聚集问题。
4. **PII / 阻断失败作为独立 Safety Gate**，红绿灯展示，不进任何维度。

---

## 一、核心面板（5 项）

### 1. Task Success（任务完成）

| | |
|---|---|
| 中文标签 | 任务完成 |
| 英文标签 | Task Success |
| 枚举 | `resolved` / `partial` / `unresolved` |
| 含义 | 用户的核心请求是否被实际满足 |
| 数据来源 | Agent Critic `task_completion.verdict` + `task_outcome` 断言组 |
| 业界对应 | SWE-bench Resolved Rate · τ-bench pass^1 · WebArena Success · GAIA Exact Match · Galileo Action Completion · AgentBoard Progress Rate |
| 参考链接 | [SWE-bench](https://www.swebench.com/) · [τ²-bench](https://arxiv.org/abs/2506.07982) · [Galileo Agent Leaderboard v2](https://huggingface.co/blog/pratikbhavsar/agent-leaderboard-v2) |

### 2. Tool Use（工具使用）

| | |
|---|---|
| 中文标签 | 工具使用 |
| 英文标签 | Tool Use |
| 枚举 | `correct` / `suboptimal` / `wrong` |
| 含义 | 工具选择是否正确 + 参数是否有效；二者合并为一项判定，与 DeepEval 做法一致 |
| 子项展开（折叠区） | Tool Selection（选对工具）· Tool Argument Correctness（参数正确） |
| 数据来源 | Agent Critic `tool_use.verdict` + 断言：`tool-error-free` / `tool-args-valid` / `tool-results-used-in-final` |
| 业界对应 | Galileo Tool Selection Quality · Arize Phoenix Function Call · DeepEval ToolCorrectness + ArgumentCorrectness |
| 参考链接 | [Galileo AI Agent Metrics](https://galileo.ai/blog/ai-agent-metrics) · [DeepEval Agent Eval Guide](https://www.confident-ai.com/blog/llm-agent-evaluation-complete-guide) |

### 3. Action Advancement（轨迹推进）

| | |
|---|---|
| 中文标签 | 轨迹推进 |
| 英文标签 | Action Advancement |
| 枚举 | `on_track` / `drift` / `redundant` / `lost` |
| 含义 | 每一步推理与动作是否朝目标推进；是否存在冗余循环或方向回退 |
| 数据来源 | Agent Critic `reasoning.verdict` |
| 业界对应 | Galileo Action Advancement · LangSmith Trajectory Eval · Patronus Trajectory · Arize Convergence Score |
| 参考链接 | [LangSmith Agent Eval](https://docs.smith.langchain.com/evaluation/tutorials/agents) · [Patronus AI Agents](https://www.patronus.ai/agents) |

### 4. Instruction Following（指令遵循）

| | |
|---|---|
| 中文标签 | 指令遵循 |
| 英文标签 | Instruction Following |
| 枚举 | `yes` / `partial` / `no` |
| 含义 | 用户明确写出的硬约束（指定 MCP / 文件 / 格式 / 步骤）是否被全部遵守 |
| 数据来源 | Agent Critic `instruction_following.verdict` |
| 业界对应 | Anthropic Instruction Following · OpenAI Evals rubric · LangSmith Final Response Eval |
| 参考链接 | [Anthropic Demystifying Evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) · [OpenAI Graders](https://platform.openai.com/docs/guides/graders) |

### 5. Faithfulness（忠实度）

| | |
|---|---|
| 中文标签 | 忠实度 |
| 英文标签 | Faithfulness |
| 枚举 | `grounded` / `partial` / `hallucinated` |
| 含义 | 最终回复的关键声称是否都有 tool_result / 原始 trace 支撑（替代旧的 "evidence 证据支撑"，对齐业界叫法） |
| 数据来源 | Agent Critic `faithfulness.verdict` |
| 业界对应 | RAGAS Faithfulness · DeepEval Faithfulness · Patronus TRAIL · Arize Phoenix Hallucination |
| 参考链接 | [RAGAS](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) · [Patronus TRAIL](https://www.patronus.ai/agents) |

---

## 二、诊断侧栏（2 项，数字直显，不打分）

### 6. Efficiency（效率）

| | |
|---|---|
| 中文标签 | 效率 |
| 英文标签 | Efficiency |
| 展示形式 | 直接展示数字：`tokens / elapsed_seconds / tool_calls_total / tool_calls_failed / step_count`；附 Agent Critic 给的 `normal` / `high` / `excessive` 标签作为参考 |
| 含义 | 资源消耗是否与任务复杂度匹配；**不打分、不进面板顶部** |
| 业界对应 | OSWorld WES+/WES- · Arize Convergence Score · DeepEval Step Efficiency |
| 参考链接 | [OSWorld-Human / WES](https://mlsys.wuklab.io/posts/oshuman) · [Arize Agent Evaluation](https://arize.com/docs/ax/learn/evaluation-concepts/agent-evaluation) |

### 7. Reliability（可靠性）

| | |
|---|---|
| 中文标签 | 可靠性 |
| 英文标签 | Reliability |
| 展示形式 | 短期：展示"本轮是否存在 critical / high 阻断失败"红绿灯。长期：实现 pass^k 后展示重跑成功率 |
| 含义 | 同一任务多次重跑的稳定性；与质量量纲不同，**绝不与其它维度合并** |
| 业界对应 | τ-bench pass^k · OSWorld-Human · Sierra Reliability paper |
| 参考链接 | [τ²-bench paper](https://arxiv.org/abs/2506.07982) |

---

## 三、独立 Safety Gate（红绿灯）

不进任何维度，命中即整条 trace 标红：

- **PII / Secret Leak**：检测到隐私或密钥泄露
- **Critical Tool Error**：critical 级别工具错误（如越权调用）
- **No Final Response**：未生成最终回复

业界对应：Anthropic Safety Evals · Patronus Percival

---

## 四、迁移映射（旧 6 维 → 新面板）

| 旧维度 (v0.1.11) | 旧权重 | 新位置 | 备注 |
|---|---|---|---|
| 任务结果 outcome | 26% | 核心 #1 Task Success | 保留，去权重 |
| 语义相关 semantic_alignment | 18% | 核心 #4 Instruction Following | 拆出 |
| 语义相关 semantic_alignment | (同上) | 核心 #3 Action Advancement 的输入信号之一 | "相关性"语义并入轨迹/指令遵循 |
| 执行轨迹 trajectory | 20% | 核心 #2 Tool Use + 核心 #3 Action Advancement | 拆为两项 |
| 证据支撑 evidence | 16% | 核心 #5 Faithfulness | 重命名，对齐业界 |
| 资源使用 resource_use | 8% | 诊断 #6 Efficiency | 降级为诊断 |
| 可靠性 reliability | 12% | 诊断 #7 Reliability + Safety Gate | PII / 阻断下沉为 gate |

---

## 五、前端展示建议

**主视图（横排或网格）**：
```
[Task Success] [Tool Use] [Action Advancement] [Instruction Following] [Faithfulness]
   resolved      correct       on_track                 yes                grounded
```

**右侧 / 底部诊断条**：
```
Efficiency: 14M tokens · 1141s · 99 calls (53 failed) · LLM 判 high
Reliability: 本轮无 critical 失败  |  pass^k: 未启用
```

**Safety Gate**：顶部红色横条仅在命中时出现。

**不再显示**：`quality_score` 数字 · `综合评分构成` 卡片 · 任何 0–1 浮点主分。

---

## 六、A/B 对比策略

- 每个核心维度逐项对比 verdict 变化（resolved → partial 算回退；partial → resolved 算改进）
- 诊断侧栏直接对比数字差值（tokens / latency / step_count）
- **不再有"综合质量分下降 X%"这类判定**——回归只看核心维度的 verdict 退化和 Safety Gate 新增

---

## 七、可引用一句话

> 评估方法参考 Anthropic / OpenAI / Galileo / DeepEval / Patronus / LangSmith / Arize Phoenix 的主流做法，以及 SWE-bench / τ-bench / AgentBoard / OSWorld 等学术 benchmark；不采用单一加权综合分，转为分维度并列报告 + 独立诊断侧栏 + Safety Gate 的"仪表盘"模式。

