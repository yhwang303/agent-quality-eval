# Agent LLM 评审模板 v2

> 基于业界 agent eval 主流做法重写，替换 v1 的 Lv 0–4 锚定打分方案。

## 一、为什么 v1 不专业

v1 用 6 维 × 5 档 Likert（Lv 0–4），实际跑出来的问题正如截图所示：

- 模型输出大量"假大空套话"（"声称达成但 trace 中无明确成功信号"）
- `trace_findings` / `claim_audit` 区块过长，重复信息堆叠
- Lv 1 / Lv 2 这类档位本身没有专业含义，读者无法据此行动
- 评审输入是"已经生成的 trace"，是后处理派生数据，不是原始 agent 行为

研究层面也支持 **不要用细粒度 Likert**：
G-Eval 论文（Liu et al. 2023, *NLG Evaluation using GPT-4 with Better Human Alignment*）和后续多个复现都显示，**LLM-as-judge 在 5 档以上 Likert 上一致性差**；二元/三元判断 + 多个独立子项聚合显著更稳。

---

## 二、业界 agent eval 的主流做法

| 项目 / 框架 | 评审做法 |
|---|---|
| **Galileo Agentic Evaluations** | 三个核心二元指标：**Tool Selection Quality**（工具选得对不对）、**Action Advancement**（每步是否朝目标推进）、**Action Completion**（最终是否完成用户请求） |
| **LangSmith Trajectory Evaluators** | 把实际 trajectory 与期望 trajectory 逐步对齐，每步判 match / mismatch；不打浮点分 |
| **DeepEval `ToolCorrectnessMetric` / `TaskCompletionMetric`** | 工具调用正确性是独立二元指标；任务完成度是 G-Eval 推理 + 简单 0/1 |
| **τ-bench / AgentBench / SWE-bench** | 唯一硬指标是**任务是否被解决**（pass@k），过程指标只做诊断不参与总分 |
| **OpenAI Evals** | yes/no rubric + 引用原始消息片段，禁止自由散文 |
| **Anthropic eval cookbook** | 给 judge 的 criteria 写成"如果出现 X 就标记 Y"的判定规则，而非分档 |
| **Patronus / Arize Phoenix** | Tool Selection、Tool Argument Quality、Function Call Hallucination 都是独立二元判定 |

**共识**：

1. **工具/动作选择是 agent eval 的首要指标**——超过"回答清晰度"等表面属性。
2. **任务是否被解决** = 唯一允许做总判定的字段，其它都是诊断。
3. **能用二元就别用 Likert**，能用枚举就别用浮点。
4. 评审输入用**原始 agent 消息**（user → assistant 含 thinking → tool_call → tool_result → final），不是后处理 trace。

---

## 三、v2 设计

### 3.1 输入（必须是原始数据，不是 trace）

```
- user_message              用户原始消息（含明确约束，如指定 MCP / 文件 / 格式）
- assistant_turns[]         每个 turn 含：thinking（如有）、tool_calls[]、content
- tool_results[]            每个 tool_call 对应的原始返回
- final_response            最终回复
```

**禁止**输入：v1 那种已聚合的 metrics 表、已经摘要的 trace 卡片。

### 3.2 评审 4 个核心问题（全部二元 / 三元，无浮点）

| # | 问题 | 判定值 | 说明 |
|---|---|---|---|
| 1 | **intent_understood** | yes / partial / no | agent 是否识别了用户的具体诉求与约束 |
| 2 | **tool_action_aligned** | yes / partial / no | 实际工具调用是否匹配用户意图（用户说用 MCP X，agent 是否真的调了 X） |
| 3 | **reasoning_on_track** | yes / drift / lost | 推理路径是否朝目标推进，有无明显跑偏 |
| 4 | **task_resolved** | yes / partial / no | 用户的具体请求是否在最终回复里被实际解决 |

**不再有**：6 维 Likert、加权浮点总分、Lv 0–4、anchor_quote、trace_findings、claim_audit。

### 3.3 顶层 verdict

只用 `task_resolved` 决定，不再加权。

| `task_resolved` | `verdict` |
|---|---|
| yes | resolved |
| partial | partial |
| no | unresolved |

如果 `tool_action_aligned == no`（用户要求用 MCP X，agent 完全没用），verdict 强制降到 `unresolved`，无论最终输出看起来多好——这是业界对 instruction following 的硬要求。

---

## 四、JSON 输出 Schema（紧凑）

```json
{
  "review": "一段完整中文评语（120-220 字），覆盖 token、耗时、工具调用、结果相关性、问题解决度、推理偏差/冗余和验证证据；禁止写“部分解决/已解决/未解决”这类标签",

  "intent": {
    "user_goal": "用一句话复述用户实际想完成的事（≤80 字）",
    "explicit_constraints": [
      "用户写明的硬约束，如：必须使用 MCP X / 必须读取 a.csv / 必须返回 JSON。最多 4 条，每条 ≤40 字"
    ],
    "intent_understood": "yes | partial | no",
    "intent_note": "≤60 字。仅在 partial / no 时填，写明哪条约束被忽略"
  },

  "tool_actions": {
    "expected_actions": [
      "根据用户意图应该发生的工具调用或行为（≤4 条，每条 ≤40 字）"
    ],
    "actual_key_actions": [
      {
        "step": 1,
        "action": "工具名 + 关键参数摘要（≤60 字）",
        "verdict": "on_target | off_target | unnecessary | error"
      }
    ],
    "tool_action_aligned": "yes | partial | no",
    "alignment_note": "≤80 字。指出最关键的一处偏离或匹配证据"
  },

  "reasoning": {
    "reasoning_on_track": "yes | drift | lost",
    "drift_evidence": "≤80 字。仅在 drift / lost 时填，引用具体 turn 序号，例如 turn#3"
  },

  "outcome": {
    "delivered": "用一句话复述 agent 实际交付了什么（≤80 字）",
    "task_resolved": "yes | partial | no",
    "missing_or_wrong": "≤80 字。仅在 partial / no 时填，写明缺什么 / 错在哪"
  },

  "verdict": "resolved | partial | unresolved",
  "headline": "一句话结论（≤50 字），必须能让读者一眼判断这次执行的价值"
}
```

**字段约束**：

- 所有数组上限 4 条；所有字符串字段上限 80 字；`headline` 上限 50 字。
- `review` 是唯一给前端直接展示的自然语言评审，长度 120-220 字。
- 不允许新增字段，不允许写浮点分数。
- 引用必须用 `turn#N` 或 `tool_call#N`，不允许写"看起来"、"似乎"、"高概率"这类模糊措辞。

---

## 五、Prompt 文本（直接替换 `_build_structured_turn_judge_prompt`）

```
你是 Agent 行为评审器。只输出一个 JSON 对象，不要 Markdown、不要解释、不要代码块。
所有字段必须填写；所有自然语言字段使用中文。

【评审输入】（原始数据，不要做派生）
- 用户消息：{user_message}
- assistant 各轮（含 thinking、tool_calls、content）：{assistant_turns_json}
- 工具返回：{tool_results_json}
- 最终回复：{final_response}

【评审任务】
按以下顺序回答四个核心问题，不要发散：

0. review：写一段完整中文评语，必须覆盖 token 消耗、耗时、工具调用/检索效率、最终输出与用户问题的相关性、是否解决用户问题、推理/思考是否跑偏或冗余、是否有验证证据。
   - review 不允许出现“部分解决”“已解决”“未解决”这三个标签化短语；要用自然语言描述价值、缺口和证据。
   - 如果最终回复或原始数据中明确提到创建分支、算法替换、文件修改、提交或验证，必须把它作为交付线索；只有原始数据和最终回复都无此类证据时，才可说未见代码改动。
   - 区分“原始数据观察到”“最终回复声称”“尚未验证”，禁止把未验证的声称直接当成事实。

1. intent：用户实际想完成什么？是否写明了硬约束（指定 MCP / 文件 / 格式 等）？
   - intent_understood ∈ {yes, partial, no}

2. tool_actions：根据 intent，本应发生哪些关键工具调用？
   实际发生了哪些？逐条标注 on_target / off_target / unnecessary / error。
   - 如果用户明确要求使用某 MCP / 工具，但实际调用未包含 → tool_action_aligned = no
   - 如果调用正确但参数明显不符合 query → partial
   - tool_action_aligned ∈ {yes, partial, no}

3. reasoning：assistant 的 thinking 与执行路径是否朝目标推进？
   是否存在跑偏（drift）或彻底走错（lost）？
   - reasoning_on_track ∈ {yes, drift, lost}

4. outcome：最终回复是否实际解决了用户请求？
   - task_resolved ∈ {yes, partial, no}

【硬规则】
- 不允许输出浮点分数、Lv 等级、百分比。
- 引用必须用 turn#N 或 tool_call#N，不允许写"看起来"、"似乎"、"高概率"。
- 任何字符串字段 ≤ 80 字，headline ≤ 50 字，数组 ≤ 4 项。
- verdict 由 task_resolved 决定：yes→resolved / partial→partial / no→unresolved。
- 例外：若 tool_action_aligned == no，verdict 强制为 unresolved。

【输出 schema】
（粘贴本文档第四节的 JSON schema）

只返回 JSON 对象。
```

---

## 六、前端渲染建议（极简）

新模板天然只有 4 块 + 1 个总判定，前端不需要复杂卡片：

1. **顶部一条横幅**：`verdict` 徽章（绿 resolved / 黄 partial / 红 unresolved）+ `headline` 一行字。
2. **意图块**：`user_goal` 一行 + `explicit_constraints` chip 列表 + `intent_understood` 徽章。
3. **动作块**（最重要）：`expected_actions` vs `actual_key_actions` 左右对照表，每行一个动作，右侧 `verdict` 徽章（on_target / off_target / unnecessary / error）。
4. **推理块**：单行 `reasoning_on_track` 徽章 + 一行 `drift_evidence`。
5. **结果块**：`delivered` 一行 + `missing_or_wrong` 一行。

整页应该在一屏内读完。**不再有 Trace findings / Claim audit 折叠区**。

---

## 七、与 v1 的差异速查

| | v1 | v2 |
|---|---|---|
| 输入 | 已聚合 trace + metrics | 原始 user/assistant/tool 消息 |
| 维度 | 6 个 Likert | 4 个二元/三元 |
| 评分 | Lv 0–4 + 加权浮点 | 无浮点；仅枚举 verdict |
| 工具调用评估 | 隐藏在 `tool_use_appropriateness` 一档里 | **独立第一等公民**，含 expected vs actual 对照 |
| 用户约束（如指定 MCP） | 没有专门字段 | `explicit_constraints` + `tool_action_aligned` 联动硬规则 |
| 输出长度 | 不限 | 字段级硬上限 |
| 冗余区块 | trace_findings + claim_audit | 全部删除 |
| 总判定来源 | 加权分数阈值 | 仅 `task_resolved`，附 instruction-following 硬门控 |

---

## 八、落地路线

- **A**：直接用 v2 prompt 替换 `_build_structured_turn_judge_prompt`，归一化函数同步收紧到 v2 字段；前端按第六节重做卡片。
- **B**：先在 3–5 个真实 turn 上干跑 v2（不入库），把输出贴出来对照截图判断观感是否解决；再决定是否落地。

> 推荐 B，因为 v2 把字段砍到极致，干跑结果会非常直观地暴露"模型还是在写套话"还是"真的能引用 turn#N"。
