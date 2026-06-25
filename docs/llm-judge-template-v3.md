# Agent LLM 评审输出模板 v3.2

## 一、设计取向

- **相信模型 + 原始数据**：评审输入已包含完整 transcript、tool_calls、tool_results、最终回复与 runtime_metrics。原始数据足够生成详尽评审，**不再做严苛结构校验与兜底显示**。
- **每节 = 一个判定 + 一段自然语言**：放弃 bullet 列点式结构化输出，每个维度由模型用一段流畅中文做出实质性评审。
- **影响导向仍是原则**：单步失败 / 局部跑偏不必上升为整体问题，仅当影响最终交付时才作为负面结论。
- **不要 error 兜底**：即使模型输出有瑕疵，也尽量展示其内容；不再出现"error"状态、不再展示"评审输出未通过校验"这类系统话术。

---

## 二、评审维度（7 项）

| # | 维度 | 判定枚举 |
|---|---|---|
| 1 | 效率（Cost / Latency） | `normal` / `high` / `excessive` |
| 2 | 意图与相关性 | `aligned` / `partial` / `off` |
| 3 | 指令遵循 | `yes` / `partial` / `no` |
| 4 | 工具使用 | `correct` / `suboptimal` / `wrong` |
| 5 | 推理路径 | `on_track` / `drift` / `redundant` / `lost` |
| 6 | 忠实度 | `grounded` / `partial` / `hallucinated` |
| 7 | 任务完成 | `resolved` / `partial` / `unresolved` |

每个维度的产出 = 一个枚举值 + 一段 80–180 字自然语言评审。

---

## 三、JSON Schema（精简）

```json
{
  "summary": "120-220字 一段完整自然语言总结。要求：先给整体判断，再概述用户诉求 / agent 关键动作 / 最终交付质量 / 主要影响因素。读完即可知道这次 agent 跑得怎么样。",

  "efficiency": {
    "verdict": "normal | high | excessive",
    "review": "80-180字。围绕 runtime_metrics 给出的 token、耗时、工具调用次数与失败数，对照任务复杂度判断是否合理。可指出有无明显冗余或异常波动；若过程中失败已恢复且不影响交付，应明确说明。"
  },

  "relevance": {
    "verdict": "aligned | partial | off",
    "review": "80-180字。说清用户实际目标、最终回复如何回应该目标，以及二者对齐程度。引用最终回复的关键片段或 turn#N 作为依据。"
  },

  "instruction_following": {
    "verdict": "yes | partial | no",
    "review": "80-180字。先识别用户提出的硬约束（指定 MCP / 文件 / 格式 / 步骤等），再逐项判断是否被满足。若用户未提显式约束，明确说明，并基于隐含意图做合理评判。"
  },

  "tool_use": {
    "verdict": "correct | suboptimal | wrong",
    "review": "100-220字。说明应当发生的关键工具调用 vs 实际发生的关键调用是否匹配；重点解释失败或偏离的调用是被恢复了还是实际影响了结果；引用 tool_call#N。仅当存在实际影响最终交付的失败或错误时，才给出 wrong。"
  },

  "reasoning": {
    "verdict": "on_track | drift | redundant | lost",
    "review": "80-180字。客观描述 agent 的推理轨迹与关键节点；若有冗余检索或方向回退，需说明是否被自我修正；若最终任务完成，verdict 不应为 lost。"
  },

  "faithfulness": {
    "verdict": "grounded | partial | hallucinated",
    "review": "80-180字。评估最终回复中的关键声称是否都有原始 tool_result 支撑。若发现无据声称，列举并指出影响范围；若全部有据，明确说明。"
  },

  "task_completion": {
    "verdict": "resolved | partial | unresolved",
    "review": "120-220字。逐项说明用户最初请求与最终交付的对应关系：哪些诉求被实际满足、产出是什么形式（代码 / 文档 / 结论 / 数据）、是否存在影响用户使用的实质缺口。"
  },

  "overall_verdict": "resolved | partial | unresolved"
}
```

---

## 四、`review_markdown`（前端高亮框展示内容，由后端确定性拼装）

由后端按下列模板从 JSON 字段直接拼装；LLM **不**生成 markdown：

```markdown
**结论**：{summary}

**效率** · {efficiency.verdict}
{efficiency.review}

**相关性** · {relevance.verdict}
{relevance.review}

**指令遵循** · {instruction_following.verdict}
{instruction_following.review}

**工具使用** · {tool_use.verdict}
{tool_use.review}

**推理路径** · {reasoning.verdict}
{reasoning.review}

**忠实度** · {faithfulness.verdict}
{faithfulness.review}

**任务完成** · {task_completion.verdict}
{task_completion.review}
```

每节正文是一段流畅段落，**不使用 bullet 列点**。

---

## 五、宽松校验规则（仅做基础健全性检查，不做严苛拦截）

1. JSON 必须能成功解析；解析失败重试 1 次
2. 枚举字段必须使用规定值之一；不合法时**回退为最接近的合法值**（如 `unclear` → `partial`），不重试、不报错
3. 字段长度只做**上限保护**（防超长污染前端），不设下限；任何 `review` 字段超过 400 字自动截断到 400 字并加省略号
4. **不**做禁用短语校验；**不**做引用格式校验；**不**做"影响导向"反向匹配校验
5. `overall_verdict` 仍由后端按规则推导覆盖：
   - `task_completion.verdict == resolved` 且 `instruction_following.verdict != no` → `resolved`
   - `task_completion.verdict == unresolved` 或 `instruction_following.verdict == no` → `unresolved`
   - 其它 → `partial`
6. **没有"error"状态**。即使模型返回有瑕疵的 JSON，也尽力解析并展示；只有 JSON 完全无法解析时才记录到日志，不在前端显示系统话术。

---

## 六、影响导向（写进 prompt，不做后端强校验）

模型在 prompt 中被告知：
- 单个 tool_call 失败、单个 turn 回退，**本身不是问题**；只有当它影响最终交付时才作为负面结论
- 失败被重试 / 替代方案恢复的，应在 `tool_use.review` 中"路径波动但已恢复"地描述，verdict 不应因此给 wrong
- 若 `task_completion.verdict == resolved`，`reasoning.verdict` 不应给 lost

后端**不**对此做反向校验；信任模型在 prompt 引导下做正确判断。

---

## 七、Prompt 主体（用于 `_build_structured_turn_judge_prompt`）

```
你是一名资深 Agent 行为评审员。只输出一个 JSON 对象，不要 Markdown、不要解释、不要代码块包裹。所有自然语言字段使用中文。

【评审输入】（原始数据，请直接据此评审，无需任何兜底）
- 用户消息：{user_message}
- assistant 各轮（含 thinking、tool_calls、content）：{assistant_turns_json}
- 工具返回：{tool_results_json}
- 最终回复：{final_response}
- 运行指标：{runtime_metrics_json}

【评审思路】
你拥有完整的 transcript、工具调用与返回、最终回复以及运行统计，完全足以给出一份详尽的评审。不要因为输入"看起来不完整"就回避结论；当原始数据无法支撑某个维度的判断时，直接以中性 verdict（partial / suboptimal 等）+ 一段说明带过即可。

请按 7 个维度产出评审，每个维度给一个 verdict + 一段 80–180 字的自然语言 review：
1. efficiency：基于 runtime_metrics 与任务复杂度判断 token / 耗时 / 工具次数是否合理。
2. relevance：最终回复是否对齐用户原始目标。
3. instruction_following：用户的硬约束是否被满足；若无硬约束，明确说明并基于隐含意图做合理评判。
4. tool_use：工具调用是否匹配预期；区分"已恢复的失败"与"影响结果的失败"。
5. reasoning：推理轨迹是否健康；若任务最终完成，不应判 lost。
6. faithfulness：最终回复的关键声称是否有 tool_result 支撑。
7. task_completion：用户的最初请求是否被实际满足。

最后产出 summary：120–220 字的一段完整自然语言总结，覆盖整体判断 + 用户诉求 + agent 关键动作 + 最终交付质量 + 主要影响因素。

【影响导向】
- 单步失败、单点跑偏不必上升为整体问题；只有真正影响最终交付时才作为负面结论。
- 失败被重试或替代方案恢复的，应被自然描述为"路径有波动但已恢复"。
- 不要因为局部异常就给出 wrong / lost / unresolved。

【输出 schema】
（粘贴第三节 JSON Schema）

【输出风格】
- 每个 review 字段是一段连贯自然语言，不要用 bullet 列点。
- 不要写"未提供"、"无法判断"、"需重新生成"之类的系统话术；如证据不足，自然语言中以"基于现有数据，整体判断为…"的口吻继续给出评审。
- summary 是用户唯一会通读的段落，必须自然、专业、信息量充足。

只返回 JSON 对象。
```

---

## 八、前端渲染规则

- LLM 评审区域展示后端拼装好的 `review_markdown`。每节为「**维度** · verdict」标题 + 一段段落正文；不出现 bullet。
- 顶部展示 `overall_verdict` 徽章（绿 / 黄 / 红）。
- **删除"状态：error"行**：评审模块不展示系统状态，只展示模型 / overall_verdict / review_markdown。
- 模型信息行可保留（仅展示模型名）。
- 折叠区可展示 raw JSON 供调试。

---

## 九、与 v3.1 差异

| | v3.1 | v3.2 |
|---|---|---|
| 每节产出 | 多字段拆分（expected_actions / actual_key_actions / key_moments / grounded_examples / unsupported_claims 等）+ bullet | 单字段 `review`，一段自然语言 |
| 校验强度 | 强校验 + 黑名单短语 + 影响导向反向校验 + 重试 2 次 + error 兜底 | 仅 JSON 可解析 + 枚举合法性回退 + 上限截断；无重试、无黑名单、无 error 兜底 |
| 引用要求 | 强制 `turn#N` / `tool_call#N` | 鼓励但不强制 |
| 错误展示 | "error"状态 + "评审输出不符合 v3.1 模板" 等系统话术 | 不展示系统状态，尽力展示模型产出 |
| 占位空话 | 黑名单拒收 | 不做拦截，依赖 prompt 风格引导 |
| 设计哲学 | 不信任模型，靠校验收口 | 信任模型 + 原始数据，提供清晰契约后放手 |
