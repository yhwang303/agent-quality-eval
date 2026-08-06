#!/usr/bin/env python3
"""
CoT Extractor — Claude / Cursor 共用的思维链提取核心。

从 IDE 的 transcript (.jsonl) 文件中提取完整的思维链（Chain-of-Thought），
包括：
  - 用户输入
  - 工具调用决策（tool_decision）
  - 工具执行结果（tool_execution）
  - 中间思考轮（thinking_intermediate）
  - 显式思考内容（thinking_explicit，需 Extended Thinking 模式）
  - 策略转换（strategy_shift，自动推断）
  - 错误恢复（error_recovery，自动推断）
  - 最终回复（final_response）

──────────────────────────────────────────────────────────────────────
  v0.15.0 · claude-only 分支登记表（v0.16.0 抽 IdeAdapter 时的检索锚）
──────────────────────────────────────────────────────────────────────
本文件里**目前**所有"仅 Claude 触发，Cursor 永远不走"的逻辑：

  1. ``_detect_agent_type(msgs)``
        从 transcript 顶层信号判定 IDE。返回 'claude' / 'cursor' /
        'unknown'，决定下面这些分支是否启用。

  2. ``_merge_claude_continuation_turns(turns)``
        合并 Claude transcript 里被 tool_result 切碎的"伪 turn"。
        只在 ``agent_type == 'claude'`` 时调用。

  3. ``_merge_tool_result_into_current(prev_user, cur_user)``
        Claude 把同一个 tool_use_id 的 tool_result 同时记在两条 user 消
        息里时，按 id 去重保留最新；Cursor 不会触发这条路径。

  4. ``_attach_claude_hook_events(session_cot)``
        读 ``~/.claude/state/events/<sid>/events.jsonl``（由
        ``claude-code/hooks/claude_stream_hook.py`` 落地）+ transcript 顶
        层 ``permissionMode`` 行，把事件分流到四条新时间线
        （``subagent_timeline`` / ``permission_events`` /
        ``compact_events`` / ``notification_events``）。

  5. ``_claude_events_path(session_id)``
        返回上一条用到的事件文件路径。

  6. ``ERROR_RECOVERY`` step 的 timestamp 继承
        见 ``extract_turn_cot``：Claude 路径下从触发它的失败 tool_execution
        继承时间戳，否则前端 §12.1#12 体检会少 2 个时间戳。

未来抽 ``IdeAdapter`` 时：1/2/3/6 应聚拢到 ``ClaudeAdapter`` 的
``transcript_iter`` / ``turn_boundary`` / ``post_process`` 钩子；4/5 是
独立的 hook 桥接层，可单独抽到 ``adapter.attach_hook_events()``。
"""

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# v0.8.0 新增：LLM / RAG / Web Search 调用分类器（软导入；任何加载/调用失败都
# 不应该影响主提取流程，分类只是装饰性元数据）
try:
    from cot_invocation_classifier import (
        classify as _classify_invocation,
        extract_prompt as _extract_invocation_prompt,
        extract_recall as _extract_invocation_recall,
        diagnose_recall_unavailable as _diagnose_recall_unavailable,
    )
except Exception:  # pragma: no cover — 软失败
    _classify_invocation = None  # type: ignore
    _extract_invocation_prompt = None  # type: ignore
    _extract_invocation_recall = None  # type: ignore
    _diagnose_recall_unavailable = None  # type: ignore

# v0.9.0 新增：L5 Execution Trace —— 临时脚本/文件产物追踪器
try:
    from cot_script_tracker import build_script_artifacts as _build_script_artifacts
except Exception:  # pragma: no cover — 软失败
    _build_script_artifacts = None  # type: ignore

# v0.11.0 新增：OpenTelemetry GenAI 视图增强器（client-side 合成 trace/span/token/cost/eval/...）
try:
    from cot_otel_enricher import enrich_session_with_otel as _enrich_otel
except Exception:  # pragma: no cover — 软失败
    _enrich_otel = None  # type: ignore


# ─── 步骤类型常量 ──────────────────────────────────────────

class StepType:
    USER_INPUT          = "user_input"
    TOOL_RESULT_INPUT   = "tool_result_input"
    THINKING_INTER      = "thinking_inter"       # 注意：与前端 STEP_CFG 保持一致
    THINKING_EXPLICIT   = "thinking_explicit"
    PRE_TOOL_REASONING  = "pre_tool_reasoning"   # 工具调用前的文字说明（行为推断CoT）
    TOOL_DECISION       = "tool_decision"
    TOOL_EXECUTION      = "tool_execution"
    STRATEGY_SHIFT      = "strategy_shift"
    ERROR_RECOVERY      = "error_recovery"
    FINAL_RESPONSE      = "final_response"


# ─── 数据结构 ──────────────────────────────────────────────

@dataclass
class ReasoningDigest:
    """摘要式推理 — 每一步自动生成的可解释推理信息（不依赖 LLM）"""
    why: str = ""              # 简短理由：为什么执行这一步
    evidence: str = ""         # 证据引用：基于什么信息做出判断
    basis: str = ""            # 决策依据：选择这个行动的逻辑
    next_plan: str = ""        # 下一步计划：预期接下来做什么

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DecisionTrace:
    """决策轨迹 — 记录工具选择和参数推断的上下文"""
    trigger_context: str = ""     # 触发调用的上下文条件
    tool_selection_reason: str = ""  # 为什么选择这个工具
    param_inference: str = ""     # 参数是如何推断的
    continuation_reason: str = ""  # 为什么结果后继续下一步

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class StateEvolution:
    """状态演化 — 记录每步的上下文变化"""
    context_hash: str = ""        # 输入上下文的 hash（用于追踪变化）
    evidence_summary: str = ""    # 本步新增的证据摘要
    action_schema: str = ""       # 本步的 action 类型描述
    termination_check: str = ""   # 终止条件检查结果

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ErrorTrace:
    """错误形成路径 — 追踪错误的产生和传播"""
    is_error_origin: bool = False    # 是否是错误的起源步骤
    error_step_index: int = -1       # 关联的错误起源步骤索引
    referenced_by: List[int] = field(default_factory=list)  # 被哪些后续步骤引用
    correction_opportunity: bool = False  # 是否有纠正机会但未纠正
    contradicts_final: bool = False  # 是否与最终答案矛盾

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ThoughtStep:
    """思维链中的单个步骤"""
    step_index: int           # 全局步骤序号（从 1 开始）
    turn_index: int           # 所属 turn 编号
    step_type: str            # StepType 中的类型
    content: str              # 步骤内容（文本）
    metadata: Dict = field(default_factory=dict)
    tool_name: str = ""       # 工具名（tool_decision/tool_execution 类型）
    tool_use_id: str = ""     # 工具调用 ID（用于配对 decision 和 execution）
    tokens: int = 0           # 消耗的 token 数（如有）
    timestamp: Optional[str] = None   # ISO 时间戳（来自 transcript）
    duration_ms: Optional[float] = None  # 步骤耗时（毫秒），与上一步的时间差
    # ── 新增：行为可观测性增强字段 ──
    reasoning_digest: Optional[ReasoningDigest] = None   # 摘要式推理
    decision_trace: Optional[DecisionTrace] = None       # 决策轨迹
    state_evolution: Optional[StateEvolution] = None      # 状态演化
    error_trace: Optional[ErrorTrace] = None              # 错误形成路径
    # ── v0.11.0：OTel GenAI 视图（client-side 合成，由 cot_otel_enricher 注入） ──
    # 形如 {trace_id, span_id, parent_span_id, attributes, input_messages, output_messages, retrieval_documents, ...}
    otel: Optional[Dict] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        # 将 None 的增强字段转为 None（而非 asdict 的默认空 dict）
        for key in ('reasoning_digest', 'decision_trace', 'state_evolution', 'error_trace'):
            val = getattr(self, key)
            d[key] = val.to_dict() if val else None
        return d


@dataclass
class TurnCoT:
    """单个 user turn 的完整思维链"""
    turn_index: int
    user_query: str                        # 用户问题（第一个 user_input 的内容）
    steps: List[ThoughtStep] = field(default_factory=list)
    tool_calls: List[str] = field(default_factory=list)   # 工具调用序列（名称列表）
    strategy_shifts: int = 0              # 策略转换次数
    thinking_depth: int = 0              # 思考深度（中间思考轮数）
    total_steps: int = 0
    has_error_recovery: bool = False
    final_response: str = ""
    usage: Dict = field(default_factory=dict)
    complexity_score: float = 0.0
    turn_start_time: Optional[str] = None   # Turn 开始时间
    turn_duration_ms: Optional[float] = None  # Turn 活跃时长（毫秒，排除空闲间隔）
    # v0.14.5：wall-clock 跨度 vs 活跃时长（active）—— 用户 IDE 留着不关、
    # 第二天回来继续干同一 turn 时，max(observed_at_ms) - min(observed_at_ms)
    # 会爆出 1000+ 分钟，但实际 agent 只活动了几分钟。区分两个数字：
    #   - turn_duration_ms        ：active 时长（剔除 >5min 的相邻间隔）
    #   - turn_wallclock_span_ms  ：max - min 总跨度（含 idle）
    #   - turn_idle_ms            ：被剔除的总 idle 时长
    # 前端展示 active；wall-clock 走 tooltip / 高级视图。
    turn_wallclock_span_ms: Optional[float] = None
    turn_idle_ms: Optional[float] = None
    cot_summary: Optional[str] = None       # LLM 生成的 CoT 摘要（标准思维链格式）
    # ── v0.14.2：基于 hook 的真值时间戳（vs transcript 估算）──
    # 由 beforeSubmitPrompt / stop hook 直接给出的毫秒时间戳，比从 transcript
    # ts 推断更准（transcript 上很多 turn 没有时间戳）。前端优先用这两个，缺
    # 了再回退到 turn_start_time / turn_duration_ms。
    turn_start_ms_observed: Optional[int] = None
    turn_end_ms_observed: Optional[int] = None
    turn_duration_ms_observed: Optional[float] = None
    # ── 子会话级别字段（每个 turn 等同于一次"子会话" = 一次完整交互） ──
    interaction_summary: str = ""           # 一句话摘要（取 user_query 首行，用于前端子会话标题）
    turn_quality_score: Optional[float] = None  # 0~1 简单质量分：final_response 缺失/错误恢复/策略转换会扣分
    quality_signals: Dict = field(default_factory=dict)  # 质量分拆解，便于调试
    # ── v0.11.0：OTel GenAI 视图（client-side 合成）──
    # turn 级 OTel：{trace_id, span_id, parent_span_id, model, provider, finish_reasons, token_usage, request_params, response, ...}
    otel: Optional[Dict] = None
    # turn 级 eval（response-verifier 注入或 turn_quality_signal 兜底）
    eval: Optional[Dict] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["steps"] = [s.to_dict() for s in self.steps]
        return d


@dataclass
class PlanSnapshot:
    """Agent plan 在某一刻的快照 —— 从 TodoWrite 工具调用的参数里抽取。

    每次 agent 主动更新 TODO 清单都会留下一条；串起来就是"agent 是怎么一步
    步 plan / re-plan 这个任务的"的显式时间线（补 L4 计划层）。

    v0.10.0 起：
    - ``todos`` 直接保留**完整顺序的 todo 列表（带 id/content/status）**，前端可以
      直接按顺序渲染 checklist。
    - ``diff`` 是与上一条同 turn 快照的 status 迁移：``newly_completed`` /
      ``newly_started`` / ``newly_added`` / ``removed`` —— 让前端能直观看到
      "这次 TodoWrite 完成了哪条 / 启动了哪条 / 新增了哪条"。
    """
    at_step: int                         # 全局 step_index
    turn_index: int                      # 所属 turn
    timestamp: Optional[str] = None      # wall clock（如有）
    in_progress: List[str] = field(default_factory=list)
    completed: List[str] = field(default_factory=list)
    pending: List[str] = field(default_factory=list)
    cancelled: List[str] = field(default_factory=list)
    total: int = 0                        # 清单长度
    # v0.10.0：完整顺序 todos & diff
    todos: List[Dict] = field(default_factory=list)        # [{id, content, status, idx}]
    diff: Optional[Dict] = None                            # vs 上一条快照
    snapshot_index: int = 0                                # 全局第几次 TodoWrite
    # ── v0.14.3：plan 状态滞后推断 ──
    # 用户痛点：agent 经常完成多条 todo 后只攒一次 TodoWrite 才打勾，导致
    # 同一个 turn 末态展示的 plan 总是"落后于实际进展"——比如最后一帧显示
    # 4/9 完成，实际 9/9 都干完了。后端没法严格判定"已经做了"，但可以
    # 用四个观察事实做推断：
    #   1. 这个 snapshot 是不是它所在 turn 的最后一帧 plan？
    #   2. 它后面这个 turn 还有多少 step 没有再调 TodoWrite？
    #   3. 这个 turn 是不是已经出 final_response 了？
    #   4. final_response 文本里有没有"完成 / done / finished"等强信号？
    #
    # is_likely_stale = 满足 (1)+(2 大于阈值) 或 (1)+(3) → 这一帧很可能滞后
    # inferred_completed_ids = is_likely_stale 时，所有 in_progress 项 + 视为推断完成
    # lag_steps_to_turn_end = 该 snapshot 后面到 turn 末还有多少非 TodoWrite 步
    inferred_completed_ids: List[str] = field(default_factory=list)
    lag_steps_to_turn_end: int = 0
    is_likely_stale: bool = False
    stale_reason: Optional[str] = None       # 'lag_too_many_steps' / 'turn_finalized' / 'final_response_signal' / None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ModeTransition:
    """Agent 模式转换：plan → agent / agent → plan / debug 等。

    Cursor 提供 SwitchMode 工具，agent 调一次就是一次 mode transition。
    我们把这条调用单独拎出来做时间线，前端按"📋 Plan / 🛠️ Agent / 🐞 Debug /
    ❓ Ask"区分着色，比埋在工具流里更醒目。

    特殊：``trigger`` 字段区分这条 transition 是哪种来源
    - ``"switch_mode"``：agent 主动调 SwitchMode 工具（显式）
    - ``"create_plan"``：agent 在 plan 模式里调 CreatePlan 工具（plan 模式典型动作）
    - ``"implicit_back_to_agent"``：用户确认 plan 后系统隐式切回 agent；从下一个
      user 消息推断
    """
    at_step: int                          # 全局 step_index（tool_decision）
    turn_index: int                       # 所属 turn
    target_mode_id: str                   # plan / agent / debug / ask ...
    explanation: Optional[str] = None     # SwitchMode.input.explanation
    timestamp: Optional[str] = None
    prev_mode_id: Optional[str] = None    # 上一次 transition 的 target；首条为 None
    trigger: str = "switch_mode"          # 来源类型（见 docstring）

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PlanProposal:
    """Agent 在 plan 模式下通过 CreatePlan 工具正式提交的 plan 文档。

    与 PlanSnapshot 不同，PlanSnapshot 来自 TodoWrite（任务清单），是滚动更新
    的；PlanProposal 来自 CreatePlan（结构化文档），通常一次大任务只产出一两次，
    是给用户审阅"我准备这么干"的最终蓝图。
    """
    at_step: int
    turn_index: int
    name: str = ""
    overview: str = ""
    plan: str = ""                        # markdown 全文
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class InvocationStats:
    """LLM / RAG / Web Search 调用聚合统计（v0.8.0）。

    所有数字来自把 ``ToolDecision.metadata.invocation_category`` 在整个 session
    范围内做 group-by。分类规则见 ``cot_invocation_classifier.py``（白名单优先，
    启发式回退；任何分类失败直接 ``None``）。

    Fields
    ------
    llm_calls
        显式调用 LLM 的次数（CLI / HTTP / MCP server 任一命中即算）。
    rag_queries
        查 RAG / 向量库 / 知识库的次数。
    web_searches
        ``WebSearch`` 类工具的次数。
    llm_call_distribution / rag_query_distribution
        按 ``tool_name`` 分桶计数，便于前端展示"哪个工具被用得最多"。
        ``web_searches`` 通常只有 ``WebSearch`` 单工具，所以不单独记 dist。
    """
    llm_calls: int = 0
    rag_queries: int = 0
    web_searches: int = 0
    llm_call_distribution: Dict[str, int] = field(default_factory=dict)
    rag_query_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SessionCoT:
    """整个 session 的思维链汇总"""
    session_id: str
    transcript_path: str
    extracted_at: str
    turns: List[TurnCoT] = field(default_factory=list)
    total_tool_calls: int = 0
    total_strategy_shifts: int = 0
    total_thinking_steps: int = 0
    tool_call_distribution: Dict = field(default_factory=dict)
    avg_steps_per_turn: float = 0.0
    avg_complexity: float = 0.0
    # ── v0.7.0 新增：补 L4 / L2 实时结果 ──
    plan_timeline: List[PlanSnapshot] = field(default_factory=list)
    # cursor-stream.js 产出的 events.jsonl 合并统计（注入/命中/总条数），None 表示未启用实时流
    observed_events: Optional[Dict] = None
    # ── v0.10.0 新增：模式转换时间线（plan ↔ agent ↔ debug ↔ ask）──
    mode_transitions: List[ModeTransition] = field(default_factory=list)
    # CreatePlan 工具调用产出的正式 plan 文档（每个 turn 0~N 个）
    plan_proposals: List[PlanProposal] = field(default_factory=list)
    # ── v0.8.0 新增：LLM / RAG / Web Search 调用分类聚合 ──
    invocation_stats: Optional[InvocationStats] = None
    # ── v0.9.0 新增：L5 Execution Trace —— 临时脚本/文件产物追踪 ──
    # 每个被 Write/StrReplace/Delete touched 过的文件 path 都会出一条 ScriptArtifact，
    # 含完整生命周期（创建→执行→删除）+ Shell 命令的反向关联。
    script_artifacts: List[Dict] = field(default_factory=list)
    script_stats: Optional[Dict] = None
    # ── v0.11.0：OpenTelemetry GenAI 视图（client-side 合成，由 cot_otel_enricher 注入）──
    # 顶层视图 dict：{schema, trace_id, root_span_id, model, provider, totals, token_usage, eval, request_params, missing_signals, ...}
    otel_view: Optional[Dict] = None
    # OTel 资源属性：service.name / service.version / deployment.environment / host.name 等
    resource_attributes: Optional[Dict] = None
    # ── v0.14.2：Cursor session 生命周期 hook（sessionStart / sessionEnd / stop / beforeSubmitPrompt /
    #             beforeTabFileRead / afterTabFileEdit）落地的元数据 ──
    # session_meta 里全部字段都是「Cursor IDE 在用户那一端真实发生过」的事实，比 transcript 推断更可信：
    #   {cursor_version, user_email, workspace_roots, session_start_ms_observed, session_end_ms_observed,
    #    session_duration_ms_observed, hook_events_observed: {sessionStart: n, sessionEnd: n, stop: n, ...},
    #    transcript_path}
    session_meta: Optional[Dict] = None
    # 用户在 IDE 里手动操作（不是 agent 的工具调用！）的时间线：
    #   每条形如 {kind: 'tab_read'|'tab_edit'|'submit_prompt', t: ms, file_path?, edits_count?,
    #             added_lines?, removed_lines?, generation_id?, prompt_chars?}
    # 用来回答"这一轮里用户除了发指令还做了什么 / 之后是否手动改了 agent 的产出"。
    user_activity: List[Dict] = field(default_factory=list)

    # ── v0.15.0: Claude / Cursor 区分标识 + Claude 独有时间线（transcript-first） ──
    #
    # 这些字段是为前端"按 IDE 类型差异化展示"准备的——它们只装 transcript 已经
    # 原生告诉我们的事实，不做任何额外推理：
    #   * agent_type         : "claude" | "cursor" | "unknown"，由 _detect_agent_type
    #                          根据 transcript 顶层结构特征判定。
    #   * subagent_timeline  : Claude `Task` 工具触发的 sidechain 子代理时间线，
    #                          每条 {t_ms, sub_agent_id, prompt, summary, tool_use_id, ...}。
    #   * permission_events  : Claude `permission-mode` 记录（plan/acceptEdits/...）的
    #                          切换时间线，每条 {t_ms, mode, prev_mode, source}。
    #   * compact_events     : Claude `Compact`（上下文压缩）触发记录，每条
    #                          {t_ms, before_tokens, after_tokens, trigger}。
    #   * notification_events: Claude `Notification` hook 上来的提示信息，每条
    #                          {t_ms, kind, message}。
    # 这些字段为 Cursor session 时保持空 list，是无侵入扩展。
    agent_type: Optional[str] = None
    subagent_timeline: List[Dict] = field(default_factory=list)
    permission_events: List[Dict] = field(default_factory=list)
    compact_events: List[Dict] = field(default_factory=list)
    notification_events: List[Dict] = field(default_factory=list)
    # v0.15.1：第 5 条时间线 environment_events——装 Claude 27 hook 里
    # 剩下那批"IDE / 环境层"事件，跟 agent 行为不直接相关但对回放上下文
    # 极有价值：
    #   * ConfigChange         配置变更（settings 改了哪个 key）
    #   * InstructionsLoaded   全局/项目指令文件被加载
    #   * CwdChanged           工作目录切换
    #   * FileChanged          文件被外部改动（非 agent 工具）
    #   * WorktreeCreate/Remove git worktree 操作
    # 同时把 27 hook 实际触发计数汇总到 session_meta.hook_events_observed，
    # 让前端能一眼看到"装的 27 个 hook 这次真触发了多少 / 哪些"。
    environment_events: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["turns"] = [t.to_dict() for t in self.turns]
        d["plan_timeline"] = [p.to_dict() for p in self.plan_timeline]
        d["mode_transitions"] = [m.to_dict() for m in self.mode_transitions]
        d["plan_proposals"] = [p.to_dict() for p in self.plan_proposals]
        return d


# ─── Transcript 解析工具函数 ───────────────────────────────

def _get_timestamp(msg: Dict) -> Optional[str]:
    """从 transcript 消息中提取时间戳"""
    # Claude Code transcript 中时间戳字段
    for key in ("timestamp", "created_at", "ts", "time"):
        v = msg.get(key)
        if v:
            return str(v)
    return None


def _ts_to_ms(ts: Optional[str]) -> Optional[float]:
    """将 ISO 时间戳转换为毫秒时间戳"""
    if not ts:
        return None
    try:
        from datetime import datetime, timezone
        # 支持多种格式
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%dT%H:%M:%S.%f+00:00", "%Y-%m-%dT%H:%M:%S+00:00"):
            try:
                dt = datetime.strptime(ts, fmt)
                return dt.timestamp() * 1000
            except ValueError:
                continue
        # 尝试 fromisoformat
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return dt.timestamp() * 1000
    except Exception:
        return None


def _get_role(msg: Dict) -> Optional[str]:
    # Claude Code transcript: role 放在顶层的 "type"
    t = msg.get("type")
    if t in ("user", "assistant"):
        return t
    # Cursor transcript: role 放在顶层的 "role"
    r_top = msg.get("role")
    if r_top in ("user", "assistant"):
        return r_top
    # 兜底：某些导出格式把 role 放在 message 内
    m = msg.get("message", {})
    if isinstance(m, dict):
        r = m.get("role")
        if r in ("user", "assistant"):
            return r
    return None


def _get_content(msg: Dict) -> Any:
    if "message" in msg and isinstance(msg.get("message"), dict):
        return msg["message"].get("content")
    return msg.get("content")


def _get_usage(msg: Dict) -> Optional[Dict]:
    m = msg.get("message", {})
    if isinstance(m, dict):
        return m.get("usage")
    return None


def _get_stop_reason(msg: Dict) -> Optional[str]:
    m = msg.get("message", {})
    if isinstance(m, dict):
        return m.get("stop_reason")
    return None


def _get_model(msg: Dict) -> str:
    m = msg.get("message", {})
    if isinstance(m, dict):
        return m.get("model") or "claude"
    return "claude"


def _extract_text_only(content: Any) -> str:
    """只提取 text 类型的 block"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            x.get("text", "") for x in content
            if isinstance(x, dict) and x.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    return ""


def _is_tool_result_msg(msg: Dict) -> bool:
    """判断是否是工具结果消息（user 角色，包含 tool_result block）"""
    if _get_role(msg) != "user":
        return False
    content = _get_content(msg)
    if isinstance(content, list):
        return any(
            isinstance(x, dict) and x.get("type") == "tool_result"
            for x in content
        )
    return False


def _is_thinking_msg(msg: Dict) -> bool:
    """
    判断是否是中间思考轮（隐式 CoT）。
    特征：output_tokens=0 且无 stop_reason，但有文本内容。
    """
    usage = _get_usage(msg) or {}
    if usage.get("output_tokens", 0) > 0:
        return False
    if _get_stop_reason(msg):
        return False
    content = _get_content(msg)
    text = _extract_text_only(content)
    return len(text.strip()) > 0


def _is_error_result(result_text: str) -> bool:
    """判断工具执行结果是否包含错误信息"""
    if not result_text:
        return False
    error_patterns = [
        r"error", r"failed", r"exception", r"traceback",
        r"not found", r"permission denied", r"no such file",
        r"command not found", r"syntax error", r"cannot",
        r"unable to", r"invalid", r"undefined",
    ]
    lower = result_text.lower()
    return any(re.search(p, lower) for p in error_patterns)


def _extract_tool_result_text(tr_block: Dict) -> str:
    """从 tool_result block 中提取文本内容"""
    content = tr_block.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            x.get("text", "") for x in content
            if isinstance(x, dict) and x.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    return str(content) if content else ""


# ─── 策略转换检测 ──────────────────────────────────────────

def _detect_strategy_shifts(tool_sequence: List[Tuple[str, str, bool]]) -> List[int]:
    """
    检测工具调用序列中的策略转换点。

    Args:
        tool_sequence: [(tool_name, tool_use_id, is_error), ...]

    Returns:
        策略转换发生在哪些步骤索引（在 tool_sequence 中的位置）
    """
    shifts = []
    if len(tool_sequence) < 2:
        return shifts

    for i in range(1, len(tool_sequence)):
        prev_tool, _, prev_error = tool_sequence[i - 1]
        curr_tool, _, _ = tool_sequence[i]

        # 信号 1：前一个工具执行出错，后一个工具不同 → 策略转换
        if prev_error and curr_tool != prev_tool:
            shifts.append(i)
            continue

        # 信号 2：连续 3 次相同工具调用且 tool_use_id 相同（真正的重试模式）
        # 注意：同一工具处理不同文件（不同 tool_use_id）是正常的批量操作，不算策略转换
        if i >= 2:
            t0_name, t0_id, _ = tool_sequence[i - 2]
            t1_name, t1_id, _ = tool_sequence[i - 1]
            t2_name, t2_id, _ = tool_sequence[i]
            # 只有工具名相同且 tool_use_id 也相同（真正的重试）才算策略转换
            if t0_name == t1_name == t2_name and (t0_id == t1_id or t1_id == t2_id):
                shifts.append(i)
                continue

        # 信号 3：从"探索"工具（Read/Bash/Glob/Grep）切换到"执行"工具（Write/Edit/MultiEdit）
        explore_tools = {"Read", "Bash", "Glob", "Grep", "LS", "WebSearch", "WebFetch"}
        execute_tools = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
        if prev_tool in explore_tools and curr_tool in execute_tools:
            # 只有在有过错误的情况下才算策略转换（正常流程不算）
            if any(e for _, _, e in tool_sequence[:i]):
                shifts.append(i)

    return list(set(shifts))


# ─── 主提取函数 ────────────────────────────────────────────

def extract_turn_cot(
    user_msg: Dict,
    assistant_msgs: List[Dict],
    turn_index: int,
    global_step_offset: int = 0,
    user_msg_ts: Optional[str] = None,
    assistant_msg_timestamps: Optional[List[Optional[str]]] = None,
) -> TurnCoT:
    """
    从单个 user turn 的消息中提取思维链。

    Args:
        user_msg: 用户消息（transcript 中的 user 角色消息）
        assistant_msgs: 该 turn 中所有 assistant 消息（按顺序）
        turn_index: turn 编号
        global_step_offset: 全局步骤偏移量（用于跨 turn 的步骤编号）

    Returns:
        TurnCoT 对象
    """
    steps: List[ThoughtStep] = []
    step_idx = global_step_offset + 1
    tool_sequence: List[Tuple[str, str, bool]] = []  # (tool_name, tool_use_id, is_error)

    # 时间戳辅助变量
    _ts_list = assistant_msg_timestamps or []
    _prev_ts_ms: Optional[float] = _ts_to_ms(user_msg_ts)  # 上一个时间点
    _turn_start_ms: Optional[float] = _prev_ts_ms
    _am_idx = 0  # 当前 assistant 消息索引

    # ── Step 1: 用户输入 ──────────────────────────────────
    user_content = _get_content(user_msg)
    if isinstance(user_content, list):
        for block in user_content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")

            if btype == "text":
                text = block.get("text", "").strip()
                if text:
                    steps.append(ThoughtStep(
                        step_index=step_idx,
                        turn_index=turn_index,
                        step_type=StepType.USER_INPUT,
                        content=text,
                        metadata={"block_type": "text"},
                        timestamp=user_msg_ts,
                    ))
                    step_idx += 1

            elif btype == "tool_result":
                result_text = _extract_tool_result_text(block)
                tool_use_id = block.get("tool_use_id", "")
                is_error = _is_error_result(result_text)
                # 工具结果的时间戳就是 user_msg 的时间戳
                cur_ts_ms = _ts_to_ms(user_msg_ts)
                dur = round(cur_ts_ms - _prev_ts_ms, 2) if (cur_ts_ms and _prev_ts_ms) else None
                if cur_ts_ms:
                    _prev_ts_ms = cur_ts_ms
                # v0.16.2: 反向查找对应 tool_decision 的 tool_name，把它一起写到
                # tool_execution 的顶层和 metadata 里。这样：
                #   1) 前端 SpanTree 能直接显示 "Tool Execution → Read"，不再
                #      只有干瘪的 "Tool Execution"
                #   2) 前端可按 tool_use_id 把 D-E 配对重排，避免 Claude 并发
                #      tool_use 时 transcript 里出现的 D-D-E-E 堆叠观感
                tool_name_for_result = ""
                for s in reversed(steps):
                    if s.step_type == StepType.TOOL_DECISION and s.tool_use_id == tool_use_id:
                        tool_name_for_result = s.tool_name
                        break
                steps.append(ThoughtStep(
                    step_index=step_idx,
                    turn_index=turn_index,
                    step_type=StepType.TOOL_EXECUTION,
                    content=result_text[:2000],  # 截断过长的工具结果
                    tool_name=tool_name_for_result,
                    tool_use_id=tool_use_id,
                    metadata={
                        "is_error": is_error,
                        "result_len": len(result_text),
                        "truncated": len(result_text) > 2000,
                        # v0.16.2: 把 tool_name + tool_use_id 也镜像到 metadata，
                        # 跟 tool_decision 的写法保持一致（后者也在 metadata 里）。
                        "tool_name": tool_name_for_result,
                        "tool_use_id": tool_use_id,
                    },
                    timestamp=user_msg_ts,
                    duration_ms=dur,
                ))
                step_idx += 1
                # 记录到工具序列（用于策略转换检测）
                tool_sequence.append((tool_name_for_result, tool_use_id, is_error))

    elif isinstance(user_content, str):
        text = user_content.strip()
        if text:
            steps.append(ThoughtStep(
                step_index=step_idx,
                turn_index=turn_index,
                step_type=StepType.USER_INPUT,
                content=text,
                metadata={"block_type": "text"},
                timestamp=user_msg_ts,
            ))
            step_idx += 1

    # ── Step 2: 遍历 assistant 消息 ───────────────────────
    usage_agg = {"input_tokens": 0, "output_tokens": 0,
                 "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    final_response_text = ""

    for am in assistant_msgs:
        content = _get_content(am)
        usage = _get_usage(am) or {}
        stop_reason = _get_stop_reason(am)
        output_tokens = usage.get("output_tokens", 0)

        # 当前 assistant 消息的时间戳
        am_ts = _ts_list[_am_idx] if _am_idx < len(_ts_list) else None
        am_ts_ms = _ts_to_ms(am_ts)
        dur_for_am = round(am_ts_ms - _prev_ts_ms, 2) if (am_ts_ms and _prev_ts_ms) else None
        if am_ts_ms:
            _prev_ts_ms = am_ts_ms
        _am_idx += 1

        # 聚合 usage
        usage_agg["input_tokens"] = usage.get("input_tokens", 0)
        usage_agg["output_tokens"] += output_tokens
        usage_agg["cache_creation_input_tokens"] += usage.get("cache_creation_input_tokens", 0)
        usage_agg["cache_read_input_tokens"] += usage.get("cache_read_input_tokens", 0)

        if not isinstance(content, list):
            # 纯文本 content
            text = _extract_text_only(content)
            if text.strip():
                if _is_thinking_msg(am):
                    steps.append(ThoughtStep(
                        step_index=step_idx,
                        turn_index=turn_index,
                        step_type=StepType.THINKING_INTER,
                        content=text,
                        tokens=output_tokens,
                        metadata={"output_tokens": output_tokens},
                        timestamp=am_ts,
                        duration_ms=dur_for_am,
                    ))
                    step_idx += 1
                elif stop_reason == "end_turn":
                    final_response_text = text
                    steps.append(ThoughtStep(
                        step_index=step_idx,
                        turn_index=turn_index,
                        step_type=StepType.FINAL_RESPONSE,
                        content=text,
                        tokens=output_tokens,
                        metadata={"stop_reason": stop_reason, "output_tokens": output_tokens},
                        timestamp=am_ts,
                        duration_ms=dur_for_am,
                    ))
                    step_idx += 1
            continue

        # 遍历 content blocks
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")

            if btype == "thinking":
                # Extended Thinking 显式思考
                thinking_text = block.get("thinking", "").strip()
                if thinking_text:
                    steps.append(ThoughtStep(
                        step_index=step_idx,
                        turn_index=turn_index,
                        step_type=StepType.THINKING_EXPLICIT,
                        content=thinking_text,
                        tokens=output_tokens,
                        metadata={"source": "extended_thinking"},
                        timestamp=am_ts,
                        duration_ms=dur_for_am,
                    ))
                    step_idx += 1

            elif btype == "text":
                text = block.get("text", "").strip()
                if not text:
                    continue
                if _is_thinking_msg(am):
                    # 中间思考轮（隐式 CoT）
                    steps.append(ThoughtStep(
                        step_index=step_idx,
                        turn_index=turn_index,
                        step_type=StepType.THINKING_INTER,
                        content=text,
                        tokens=output_tokens,
                        metadata={"output_tokens": output_tokens},
                        timestamp=am_ts,
                        duration_ms=dur_for_am,
                    ))
                    step_idx += 1
                elif stop_reason == "tool_use":
                    # 工具调用前的文字说明 → 行为推断 CoT
                    # Claude 在决定调用工具前有时会先输出一段文字解释意图
                    steps.append(ThoughtStep(
                        step_index=step_idx,
                        turn_index=turn_index,
                        step_type=StepType.PRE_TOOL_REASONING,
                        content=text,
                        tokens=output_tokens,
                        metadata={"output_tokens": output_tokens, "before_tool_use": True},
                        timestamp=am_ts,
                        duration_ms=dur_for_am,
                    ))
                    step_idx += 1
                elif stop_reason == "end_turn":
                    # 最终回复
                    final_response_text = text
                    steps.append(ThoughtStep(
                        step_index=step_idx,
                        turn_index=turn_index,
                        step_type=StepType.FINAL_RESPONSE,
                        content=text,
                        tokens=output_tokens,
                        metadata={"stop_reason": stop_reason, "output_tokens": output_tokens},
                        timestamp=am_ts,
                        duration_ms=dur_for_am,
                    ))
                    step_idx += 1
                else:
                    # 其他文本（如 stop_reason=max_tokens 等）
                    steps.append(ThoughtStep(
                        step_index=step_idx,
                        turn_index=turn_index,
                        step_type=StepType.THINKING_INTER,
                        content=text,
                        tokens=output_tokens,
                        metadata={"output_tokens": output_tokens, "stop_reason": stop_reason},
                        timestamp=am_ts,
                        duration_ms=dur_for_am,
                    ))
                    step_idx += 1

            elif btype == "tool_use":
                # 工具调用决策
                tool_name = block.get("name", "unknown")
                tool_use_id = block.get("id", "")
                # Cursor transcript 不带 tool_use_id —— 直接按原逻辑去重会把同一 turn 里多次工具调用
                # 全部折叠成一条。这里在 id 缺失时合成一个稳定的伪 id，确保每次调用都独立保留。
                if not tool_use_id:
                    tool_use_id = f"cursor:t{turn_index}:s{step_idx}:{tool_name}"
                tool_input = block.get("input", {})
                input_summary = json.dumps(tool_input, ensure_ascii=False)[:500]

                # 去重：同一个 tool_use_id 只保留一次（transcript 中可能因 streaming 重复记录）
                if any(s.tool_use_id == tool_use_id and s.step_type == StepType.TOOL_DECISION for s in steps):
                    continue

                _decision_meta: Dict[str, Any] = {
                    "tool_name": tool_name,
                    "tool_use_id": tool_use_id,
                    "tool_input": tool_input,
                    "input_summary": input_summary,
                }
                # v0.15.1: 抽 Claude 自己写在 tool_input 里的"意图说明"作为独立字段。
                # Claude 在调用 Bash / Edit / Task 等工具时通常会带一个简短的 description /
                # prompt 字段，这天然就是"为什么调用这个工具"的一句话推理——把它抽出来挂在
                # metadata.tool_intent 上，前端在缺乏 extended thinking 块时把它当作
                # 思考替代品突出展示，让 Claude 4.x（非 thinking 变体）的会话也能看到
                # 每个工具调用上方的"行级思考"。源字段按优先级试探，全部容错。
                if isinstance(tool_input, dict):
                    intent_text: Optional[str] = None
                    intent_source: Optional[str] = None
                    for src_key in ("description", "task_description",
                                    "instructions", "intent"):
                        v = tool_input.get(src_key)
                        if isinstance(v, str) and v.strip():
                            intent_text = v.strip()
                            intent_source = src_key
                            break
                    if not intent_text:
                        # Task / Agent 工具：用 prompt 第一行作为 intent
                        prompt_v = tool_input.get("prompt")
                        if isinstance(prompt_v, str) and prompt_v.strip():
                            first_line = prompt_v.strip().splitlines()[0].strip()
                            if first_line:
                                # 取第一行 + 限长，过长说明 prompt 没格式化好，砍 200 字
                                intent_text = first_line[:200]
                                intent_source = "prompt_first_line"
                    if intent_text:
                        _decision_meta["tool_intent"] = intent_text
                        _decision_meta["tool_intent_source"] = intent_source
                # v0.8.0: 把每一个 tool_decision 按调用语义打 LLM/RAG/Web 标签 +
                # 抽 prompt 预览。整段 try/except 兜底——分类是装饰性元数据，
                # 任何异常都不允许影响主提取流程。
                if _classify_invocation is not None:
                    try:
                        _shell_cmd = (
                            tool_input.get("command")
                            if isinstance(tool_input, dict) else None
                        )
                        _cat = _classify_invocation(
                            tool_name=tool_name,
                            mcp_server=block.get("server_name") or block.get("mcp_server"),
                            tool_input=tool_input,
                            command=_shell_cmd,
                        )
                        if _cat:
                            _decision_meta["invocation_category"] = _cat
                            if _extract_invocation_prompt is not None:
                                _prev, _full = _extract_invocation_prompt(tool_input)
                                if _prev:
                                    _decision_meta["prompt_preview"] = _prev
                                if _full:
                                    _decision_meta["prompt_full_chars"] = _full
                    except Exception:
                        pass

                steps.append(ThoughtStep(
                    step_index=step_idx,
                    turn_index=turn_index,
                    step_type=StepType.TOOL_DECISION,
                    content=f"调用工具 {tool_name}：{input_summary}",
                    tool_name=tool_name,
                    tool_use_id=tool_use_id,
                    metadata=_decision_meta,
                    timestamp=am_ts,
                    duration_ms=dur_for_am,
                ))
                step_idx += 1

    # ── Step 2.5: 为孤立的 tool_decision 合成占位 tool_execution ──
    # 场景：Cursor transcript 只记录 assistant 的 tool_use，不写入工具结果，
    # 导致 tool_decision 找不到配对的 tool_execution。前端无法展示「调用→结果」链路。
    # 策略：若 turn 内完全没有真实的 tool_execution，则为每个 tool_decision 紧跟插入一个
    #      synthetic 节点，标注"结果未记录"，保留 Claude Code 的正常行为不受影响。
    _has_real_exec = any(s.step_type == StepType.TOOL_EXECUTION for s in steps)
    if not _has_real_exec:
        _exec_ids = set()
        # 从后往前插入 synthetic execution，避免索引偏移
        to_synth: List[int] = [
            i for i, s in enumerate(steps)
            if s.step_type == StepType.TOOL_DECISION and s.tool_use_id not in _exec_ids
        ]
        for i in reversed(to_synth):
            dec = steps[i]
            steps.insert(i + 1, ThoughtStep(
                step_index=0,  # 末尾统一重新编号
                turn_index=turn_index,
                step_type=StepType.TOOL_EXECUTION,
                content="(工具执行结果未记录，Cursor transcript 不包含 tool_result block)",
                tool_name=dec.tool_name,
                tool_use_id=dec.tool_use_id,
                metadata={
                    "is_error": False,
                    "result_len": 0,
                    "truncated": False,
                    "synthetic": True,
                    "synthetic_reason": "cursor_transcript_no_tool_result",
                    "tool_name": dec.tool_name,
                    "tool_use_id": dec.tool_use_id,
                },
                timestamp=dec.timestamp,
                duration_ms=None,
            ))

    # ── Step 3: 检测策略转换 ──────────────────────────────
    # 在已有步骤中找出 tool_decision 和 tool_execution 的序列
    tool_seq_for_shift: List[Tuple[str, str, bool]] = []
    tool_decision_step_indices: List[int] = []  # steps 列表中的索引

    for i, s in enumerate(steps):
        if s.step_type == StepType.TOOL_DECISION:
            tool_decision_step_indices.append(i)
            # 找对应的 tool_execution
            is_error = False
            for s2 in steps:
                if s2.step_type == StepType.TOOL_EXECUTION and s2.tool_use_id == s.tool_use_id:
                    is_error = s2.metadata.get("is_error", False)
                    break
            tool_seq_for_shift.append((s.tool_name, s.tool_use_id, is_error))

    shift_indices = _detect_strategy_shifts(tool_seq_for_shift)

    # 在对应的 tool_decision 步骤后插入 strategy_shift 步骤
    # 注意：插入时需要从后往前插，避免索引偏移
    inserted_shifts = 0
    for shift_i in sorted(shift_indices):
        if shift_i < len(tool_decision_step_indices):
            insert_pos = tool_decision_step_indices[shift_i] + inserted_shifts
            prev_tool = tool_seq_for_shift[shift_i - 1][0] if shift_i > 0 else ""
            curr_tool = tool_seq_for_shift[shift_i][0]
            shift_step = ThoughtStep(
                step_index=0,  # 后面重新编号
                turn_index=turn_index,
                step_type=StepType.STRATEGY_SHIFT,
                content=f"策略转换：从 {prev_tool} 切换到 {curr_tool}",
                metadata={
                    "from_tool": prev_tool,
                    "to_tool": curr_tool,
                    "shift_index": shift_i,
                },
            )
            steps.insert(insert_pos, shift_step)
            inserted_shifts += 1

    # ── Step 4: 检测错误恢复 ──────────────────────────────
    error_recovery_count = 0
    for i, s in enumerate(steps):
        if s.step_type == StepType.TOOL_EXECUTION and s.metadata.get("is_error"):
            # 在错误执行步骤后插入 error_recovery 步骤
            recovery_step = ThoughtStep(
                step_index=0,
                turn_index=turn_index,
                step_type=StepType.ERROR_RECOVERY,
                content=f"检测到错误，进行恢复处理：{s.content[:200]}",
                metadata={
                    "error_tool_use_id": s.tool_use_id,
                    "error_content": s.content[:500],
                },
                # v0.15.0：继承失败工具执行的时间戳，保 §12.1#12 100% 覆盖
                timestamp=s.timestamp,
                duration_ms=None,
            )
            steps.insert(i + 1 + error_recovery_count, recovery_step)
            error_recovery_count += 1

    # ── Step 5: 重新编号所有步骤 ──────────────────────────
    for i, s in enumerate(steps):
        s.step_index = global_step_offset + i + 1

    # ── Step 6: 统计指标 ──────────────────────────────────
    tool_calls = [s.tool_name for s in steps if s.step_type == StepType.TOOL_DECISION]
    strategy_shifts = len([s for s in steps if s.step_type == StepType.STRATEGY_SHIFT])
    thinking_depth = len([s for s in steps if s.step_type in (
        StepType.THINKING_INTER, StepType.THINKING_EXPLICIT, StepType.PRE_TOOL_REASONING
    )])
    has_error_recovery = error_recovery_count > 0

    # 任务复杂度评分
    tc = len(tool_calls)
    ss = strategy_shifts
    er = error_recovery_count
    td = thinking_depth
    tool_diversity = len(set(tool_calls))
    complexity_score = round(0.3 * tc + 0.3 * ss + 0.2 * er + 0.2 * td, 2)

    # 用户问题（取第一个 user_input 的内容）
    user_query = ""
    for s in steps:
        if s.step_type == StepType.USER_INPUT:
            user_query = s.content
            break

    # 计算 Turn 总耗时
    # v0.14.4：注意——extract_turn_cot 是在 _attach_cursor_events 之前跑的，
    # 此时 step.metadata.observed_at_ms 还没被注入。所以这里只能用
    # transcript 原生 timestamp（很稀疏），observed_at_ms 兜底由
    # _recompute_turn_durations 在事件注入完成后再补一遍。
    turn_end_ms: Optional[float] = None
    for s in reversed(steps):
        if s.timestamp:
            turn_end_ms = _ts_to_ms(s.timestamp)
            break
    turn_duration = None
    if _turn_start_ms and turn_end_ms and turn_end_ms > _turn_start_ms:
        turn_duration = round(turn_end_ms - _turn_start_ms, 2)
    turn_start_ms = _turn_start_ms

    # ── Step 6.5: Cursor 模式推断 final_response ─────────────
    # Cursor transcript 的 assistant 消息没有 stop_reason 字段，
    # 导致最后一段「回答总结」被误判为 thinking_inter。
    # 规则：若当前 final_response_text 为空，则找「最后一个 TOOL_DECISION 之后」
    # 唯一/最末尾的 thinking_inter 作为隐式 final response。
    if not final_response_text:
        last_tool_i = -1
        for i, s in enumerate(steps):
            if s.step_type in (StepType.TOOL_DECISION, StepType.TOOL_EXECUTION):
                last_tool_i = i
        for i in range(len(steps) - 1, last_tool_i, -1):
            s = steps[i]
            if s.step_type == StepType.THINKING_INTER and s.content.strip():
                s.step_type = StepType.FINAL_RESPONSE
                md = dict(s.metadata) if isinstance(s.metadata, dict) else {}
                md["inferred_final"] = True
                md["inferred_reason"] = "cursor_no_stop_reason"
                s.metadata = md
                final_response_text = s.content
                break

    # ── Step 7: 生成行为可观测性增强数据 ─────────────
    _enrich_observability(steps, user_query, final_response_text)

    # ── Step 8: 子会话摘要 + 简单质量分 ─────────────
    # 子会话摘要：用户提问首行（跳过 XML 包装标签行/图片占位/附件样板），再 fallback 到最终回复首行
    import re as _re
    def _first_line_clean(s: str, limit: int = 60) -> str:
        if not s:
            return ""
        # Cursor / 脚本封装 user message 常见模式：
        # [Image] + <image_files>...</image_files> + <user_query>真正的问题</user_query>
        # 优先抽取 <user_query> 中内容（若存在）
        m = _re.search(r"<user_query>\s*(.+?)\s*</user_query>", s, _re.DOTALL)
        if m:
            s = m.group(1)
        # 常见样板段（附件/图片说明）跳过直到非样板行
        _boilerplate_prefixes = (
            "The following images were",
            "The following image was",
            "These images can be copied",
            "<image_files>",
        )
        for raw in s.replace("\r", "").split("\n"):
            ln = raw.strip()
            if not ln:
                continue
            # 跳过纯 XML/HTML 标签行（如 <user_query>, </user_query>, <system_reminder>）
            if ln.startswith("<") and ln.endswith(">") and " " not in ln:
                continue
            # 跳过占位符（图片/附件/编号行）
            if ln in ("[Image]", "[Attachment]") or ln.startswith("![image]"):
                continue
            if _re.match(r"^\d+\.\s+[A-Z]:", ln):  # "1. C:\Users\..." 这种文件列表
                continue
            if any(ln.startswith(p) for p in _boilerplate_prefixes):
                continue
            clean = ln.replace("`", "").lstrip("#").strip()
            if clean:
                return clean[:limit] + ("…" if len(clean) > limit else "")
        return ""

    # 子会话标题：纯规则 —— 生成「用户XX：正文」(≤40 字)；失败时 fallback 到 60 字首行
    _summary = ""
    try:
        try:
            from .cot_title_generator import generate_intent_summary  # 包内
        except Exception:
            from cot_title_generator import generate_intent_summary  # 脚本直跑
        _tool_names = [
            (c.get("name") or c.get("tool_name") or "")
            for c in tool_calls
            if isinstance(c, dict)
        ]
        _intent = generate_intent_summary(
            user_query=user_query,
            final_response_preview=(final_response_text or "")[:400],
            tool_names=[n for n in _tool_names if n][:10],
        )
        if _intent:
            _summary = _intent
    except Exception:
        # 任何异常都静默回退，确保不破坏主流程
        pass

    if not _summary:
        _summary = (
            _first_line_clean(user_query)
            or _first_line_clean(final_response_text)
            or f"Turn {turn_index}"
        )

    # 简单质量分：0~1，三档信号
    # - final_response 存在：+0.25 基础分
    # - 每次错误恢复：-0.20（最多扣 0.45）
    # - 每次策略转换：-0.05（最多扣 0.15）
    # - 无任何推理/工具调用（纯噪声回合）：-0.20
    q = 0.75
    q_signals: Dict[str, Any] = {}
    if final_response_text:
        q += 0.25
        q_signals["has_final_response"] = True
    else:
        q_signals["has_final_response"] = False
    q -= min(0.45, 0.20 * error_recovery_count)
    q_signals["error_recovery_count"] = error_recovery_count
    q -= min(0.15, 0.05 * strategy_shifts)
    q_signals["strategy_shifts"] = strategy_shifts
    if len(tool_calls) == 0 and thinking_depth == 0:
        q -= 0.20
        q_signals["empty_turn"] = True
    q = max(0.0, min(1.0, q))
    _quality = round(q, 2)
    q_signals["score"] = _quality

    # turn_start_time：优先 user_msg_ts；它为空时用我们刚算出的 turn_start_ms
    # （从 step.observed_at_ms 兜底回来的）反推一个 ISO 字符串。让 OTel 层 / 前
    # 端不再依赖一个永远 None 的字段。
    final_turn_start_time = user_msg_ts
    if not final_turn_start_time and turn_start_ms:
        from datetime import datetime, timezone
        final_turn_start_time = datetime.fromtimestamp(
            turn_start_ms / 1000.0, tz=timezone.utc
        ).isoformat()

    return TurnCoT(
        turn_index=turn_index,
        user_query=user_query,
        steps=steps,
        tool_calls=tool_calls,
        strategy_shifts=strategy_shifts,
        thinking_depth=thinking_depth,
        total_steps=len(steps),
        has_error_recovery=has_error_recovery,
        final_response=final_response_text,
        usage=usage_agg,
        complexity_score=complexity_score,
        turn_start_time=final_turn_start_time,
        turn_duration_ms=turn_duration,
        interaction_summary=_summary,
        turn_quality_score=_quality,
        quality_signals=q_signals,
    )


# ─── 行为可观测性增强 ──────────────────────────────────────

def _compute_context_hash(step: ThoughtStep) -> str:
    """计算步骤的上下文 hash（用于追踪状态变化）"""
    import hashlib
    content = f"{step.step_type}:{step.tool_name}:{step.content[:200]}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()[:12]


def _infer_evidence_summary(step: ThoughtStep) -> str:
    """从步骤中推断新增证据摘要"""
    if step.step_type == StepType.TOOL_EXECUTION:
        is_err = step.metadata.get("is_error", False)
        result_len = step.metadata.get("result_len", len(step.content))
        if is_err:
            return f"工具执行失败，返回错误信息（{result_len} chars）"
        content_preview = step.content[:80].replace('\n', ' ')
        return f"获得工具结果：{content_preview}（{result_len} chars）"
    elif step.step_type == StepType.USER_INPUT:
        return f"用户提出新需求：{step.content[:60]}"
    elif step.step_type == StepType.FINAL_RESPONSE:
        return f"生成最终回复（{len(step.content)} chars）"
    elif step.step_type == StepType.PRE_TOOL_REASONING:
        return f"Agent 表达意图：{step.content[:80]}"
    return ""


def _safe_tool_input(meta_or_step) -> Dict:
    """Return ``metadata['tool_input']`` as a dict, or ``{}`` when missing.

    Cursor / Claude transcripts occasionally carry the field as a raw
    JSON *string* (typically when a tool call was streamed and reached
    us before being fully parsed). Every downstream consumer in this
    file expects a mapping, so we centralise the coercion here.

    Accepts either a ``ThoughtStep`` or a plain ``metadata`` dict.

    v0.19.5 加固：之前如果 ``tool_input`` 是 JSON 字符串就直接返回 ``{}``，
    会让 ``_build_plan_timeline`` 等消费者拿不到 ``todos`` —— 这是 Claude /
    CodeBuddy 在不同 IDE 下偶发的写入形态。现在多尝试一步 ``json.loads``：
    解析成 dict 才用，仍然不是 dict 才退回 ``{}``。
    """
    meta = meta_or_step.metadata if hasattr(meta_or_step, "metadata") else meta_or_step
    if not isinstance(meta, dict):
        return {}
    raw = meta.get("tool_input")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip().startswith(("{", "[")):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _infer_action_schema(step: ThoughtStep) -> str:
    """推断步骤的 action schema"""
    if step.step_type == StepType.TOOL_DECISION:
        tool = step.tool_name or "unknown"
        params = list(_safe_tool_input(step).keys())
        return f"CALL({tool}, [{', '.join(params[:5])}])"
    elif step.step_type == StepType.TOOL_EXECUTION:
        return "RECEIVE_RESULT"
    elif step.step_type == StepType.FINAL_RESPONSE:
        return "EMIT_RESPONSE"
    elif step.step_type == StepType.USER_INPUT:
        return "RECEIVE_INPUT"
    elif step.step_type == StepType.STRATEGY_SHIFT:
        return f"SHIFT({step.metadata.get('from_tool', '?')} → {step.metadata.get('to_tool', '?')})"
    elif step.step_type == StepType.ERROR_RECOVERY:
        return "ERROR_RECOVERY"
    elif step.step_type == StepType.PRE_TOOL_REASONING:
        return "REASON_BEFORE_TOOL"
    return step.step_type.upper()


def _infer_termination_check(step: ThoughtStep, is_last: bool) -> str:
    """推断终止条件检查结果"""
    if step.step_type == StepType.FINAL_RESPONSE:
        return "✅ 任务完成，生成最终回复"
    if step.step_type == StepType.TOOL_DECISION:
        return "❌ 需要更多信息，继续调用工具"
    if step.step_type == StepType.ERROR_RECOVERY:
        return "❌ 遇到错误，需要恢复"
    if step.step_type == StepType.STRATEGY_SHIFT:
        return "❌ 当前策略不可行，切换策略"
    if is_last:
        return "⏸️ Turn 结束（等待工具结果或用户输入）"
    return "❌ 继续执行"


def _generate_reasoning_digest(
    step: ThoughtStep,
    prev_step: Optional[ThoughtStep],
    next_step: Optional[ThoughtStep],
    user_query: str,
    all_steps: List[ThoughtStep],
) -> ReasoningDigest:
    """为单个步骤生成摘要式推理（不依赖 LLM，纯规则推断）"""
    why = ""
    evidence = ""
    basis = ""
    next_plan = ""

    if step.step_type == StepType.USER_INPUT:
        why = "接收用户输入，开始新的任务处理"
        evidence = f"用户请求：{step.content[:60]}"
        basis = "用户发起了新的对话轮次"
        next_plan = "分析用户需求，决定执行策略"

    elif step.step_type == StepType.TOOL_DECISION:
        tool = step.tool_name or "unknown"
        # 推断为什么选择这个工具
        # Cursor / Claude transcripts occasionally surface tool_input as a
        # raw JSON string instead of a parsed object (e.g. when the model
        # streams a partial call). Normalise once so downstream
        # ``.keys()`` / ``.items()`` calls in this function are safe.
        raw_tool_input = step.metadata.get("tool_input", {})
        tool_input = raw_tool_input if isinstance(raw_tool_input, dict) else {}
        param_keys = list(tool_input.keys())

        if prev_step and prev_step.step_type == StepType.TOOL_EXECUTION:
            if prev_step.metadata.get("is_error"):
                why = f"上一步工具执行失败，尝试使用 {tool} 进行恢复"
                evidence = f"上一步错误：{prev_step.content[:60]}"
            else:
                why = f"基于上一步结果，继续使用 {tool} 推进任务"
                evidence = f"上一步结果：{prev_step.content[:60]}"
        elif prev_step and prev_step.step_type == StepType.USER_INPUT:
            why = f"根据用户需求，选择 {tool} 作为第一步操作"
            evidence = f"用户需求：{user_query[:60]}"
        elif prev_step and prev_step.step_type == StepType.PRE_TOOL_REASONING:
            why = f"经过推理分析后，决定调用 {tool}"
            evidence = f"推理说明：{prev_step.content[:60]}"
        else:
            why = f"决定调用 {tool} 执行操作"
            evidence = f"当前任务上下文"

        basis = f"工具 {tool} 适合处理当前步骤，参数：{', '.join(param_keys[:3])}"
        next_plan = "等待工具执行结果，根据结果决定下一步"

    elif step.step_type == StepType.TOOL_EXECUTION:
        is_err = step.metadata.get("is_error", False)
        if is_err:
            why = "工具执行完成，但返回了错误"
            evidence = f"错误内容：{step.content[:80]}"
            basis = "需要分析错误原因并决定恢复策略"
            next_plan = "尝试错误恢复或切换策略"
        else:
            why = "工具执行成功，获得了新的信息"
            evidence = f"执行结果：{step.content[:80]}"
            basis = "结果提供了推进任务所需的信息"
            if next_step and next_step.step_type == StepType.TOOL_DECISION:
                next_plan = f"基于结果，继续调用 {next_step.tool_name or '下一个工具'}"
            elif next_step and next_step.step_type == StepType.FINAL_RESPONSE:
                next_plan = "信息充足，准备生成最终回复"
            else:
                next_plan = "分析结果，决定下一步操作"

    elif step.step_type == StepType.PRE_TOOL_REASONING:
        why = "在调用工具前进行推理分析"
        evidence = f"推理内容：{step.content[:80]}"
        basis = "需要明确工具调用的目的和预期结果"
        next_plan = "基于推理结果选择合适的工具"

    elif step.step_type == StepType.STRATEGY_SHIFT:
        from_tool = step.metadata.get("from_tool", "?")
        to_tool = step.metadata.get("to_tool", "?")
        why = f"当前策略（{from_tool}）不够有效，切换到 {to_tool}"
        evidence = f"策略转换信号：{step.content[:80]}"
        basis = "前序步骤的结果表明需要调整方法"
        next_plan = f"使用新策略（{to_tool}）继续推进"

    elif step.step_type == StepType.ERROR_RECOVERY:
        why = "检测到错误，启动恢复机制"
        evidence = f"错误信息：{step.metadata.get('error_content', step.content)[:80]}"
        basis = "错误需要被处理才能继续任务"
        next_plan = "分析错误原因，尝试替代方案"

    elif step.step_type == StepType.FINAL_RESPONSE:
        why = "所有必要信息已收集完毕，生成最终回复"
        # 统计之前的工具调用
        tool_steps = [s for s in all_steps if s.step_type == StepType.TOOL_DECISION and s.step_index < step.step_index]
        evidence = f"经过 {len(tool_steps)} 次工具调用后，信息充足"
        basis = "任务目标已达成或信息已充分"
        next_plan = "任务完成，等待用户下一轮输入"

    elif step.step_type in (StepType.THINKING_INTER, StepType.THINKING_EXPLICIT):
        why = "进行中间推理，分析当前状态"
        evidence = f"思考内容：{step.content[:80]}"
        basis = "需要在行动前进行深入分析"
        next_plan = "基于推理结果决定下一步行动"

    return ReasoningDigest(why=why, evidence=evidence, basis=basis, next_plan=next_plan)


def _generate_decision_trace(
    step: ThoughtStep,
    prev_step: Optional[ThoughtStep],
    next_step: Optional[ThoughtStep],
    user_query: str,
) -> Optional[DecisionTrace]:
    """为工具调用步骤生成决策轨迹"""
    if step.step_type != StepType.TOOL_DECISION:
        return None

    tool = step.tool_name or "unknown"
    raw_tool_input = step.metadata.get("tool_input", {})
    # See _generate_reasoning_digest: streamed partial calls can land
    # here as a string. Coerce to dict so .items()/.get() are safe.
    tool_input = raw_tool_input if isinstance(raw_tool_input, dict) else {}

    # 触发上下文
    if prev_step:
        if prev_step.step_type == StepType.USER_INPUT:
            trigger = f"用户请求触发：{prev_step.content[:60]}"
        elif prev_step.step_type == StepType.TOOL_EXECUTION:
            if prev_step.metadata.get("is_error"):
                trigger = f"上一步工具执行失败，需要恢复：{prev_step.content[:60]}"
            else:
                trigger = f"上一步工具结果需要后续处理：{prev_step.content[:60]}"
        elif prev_step.step_type == StepType.PRE_TOOL_REASONING:
            trigger = f"推理分析后决定调用：{prev_step.content[:60]}"
        else:
            trigger = f"前序步骤（{prev_step.step_type}）触发"
    else:
        trigger = "Turn 开始，首次工具调用"

    # 工具选择原因
    explore_tools = {"Read", "Bash", "Glob", "Grep", "LS", "WebSearch", "WebFetch"}
    execute_tools = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
    if tool in explore_tools:
        selection = f"选择 {tool}（探索类工具）用于收集信息"
    elif tool in execute_tools:
        selection = f"选择 {tool}（执行类工具）用于修改/创建内容"
    else:
        selection = f"选择 {tool} 执行特定操作"

    # 参数推断
    param_desc_parts = []
    for k, v in list(tool_input.items())[:5]:
        v_str = str(v)[:50]
        param_desc_parts.append(f"{k}={v_str}")
    param_inference = f"参数推断：{'; '.join(param_desc_parts)}" if param_desc_parts else "无参数"

    # 后续决策链
    if next_step:
        if next_step.step_type == StepType.TOOL_EXECUTION:
            continuation = "等待工具执行结果"
        elif next_step.step_type == StepType.TOOL_DECISION:
            continuation = f"结果后继续调用 {next_step.tool_name or '下一个工具'}"
        else:
            continuation = f"结果后进入 {next_step.step_type}"
    else:
        continuation = "Turn 结束，等待后续输入"

    return DecisionTrace(
        trigger_context=trigger,
        tool_selection_reason=selection,
        param_inference=param_inference,
        continuation_reason=continuation,
    )


def _enrich_observability(
    steps: List[ThoughtStep],
    user_query: str,
    final_response: str,
) -> None:
    """
    为所有步骤生成行为可观测性增强数据（原地修改）。

    包括：
    1. 摘要式推理（ReasoningDigest）
    2. 决策轨迹（DecisionTrace）
    3. 状态演化（StateEvolution）
    4. 错误形成路径（ErrorTrace）
    """
    if not steps:
        return

    # ── 1. 为每个步骤生成摘要式推理和决策轨迹 ──
    for i, step in enumerate(steps):
        prev_step = steps[i - 1] if i > 0 else None
        next_step = steps[i + 1] if i < len(steps) - 1 else None

        # 摘要式推理
        step.reasoning_digest = _generate_reasoning_digest(
            step, prev_step, next_step, user_query, steps
        )

        # 决策轨迹（仅 tool_decision 类型）
        step.decision_trace = _generate_decision_trace(
            step, prev_step, next_step, user_query
        )

        # 状态演化
        step.state_evolution = StateEvolution(
            context_hash=_compute_context_hash(step),
            evidence_summary=_infer_evidence_summary(step),
            action_schema=_infer_action_schema(step),
            termination_check=_infer_termination_check(step, i == len(steps) - 1),
        )

    # ── 2. 错误形成路径分析 ──
    # 找出所有错误步骤
    error_origins: List[int] = []  # 错误起源步骤的索引
    for i, step in enumerate(steps):
        if step.step_type == StepType.TOOL_EXECUTION and step.metadata.get("is_error"):
            error_origins.append(i)
            step.error_trace = ErrorTrace(
                is_error_origin=True,
                error_step_index=step.step_index,
            )

    # 分析错误传播
    for err_idx in error_origins:
        err_step = steps[err_idx]
        err_tool_use_id = err_step.tool_use_id
        err_content_lower = err_step.content.lower()[:100]

        # 检查后续步骤是否引用了这个错误
        for j in range(err_idx + 1, len(steps)):
            later_step = steps[j]
            # 如果后续步骤引用了同一个 tool_use_id
            if later_step.tool_use_id == err_tool_use_id and later_step.step_type != StepType.TOOL_EXECUTION:
                if err_step.error_trace:
                    err_step.error_trace.referenced_by.append(later_step.step_index)

            # 如果后续步骤是 error_recovery，说明有纠正机会
            if later_step.step_type == StepType.ERROR_RECOVERY:
                if later_step.metadata.get("error_tool_use_id") == err_tool_use_id:
                    # 已经纠正
                    pass
                else:
                    # 有纠正机会但没纠正这个错误
                    if err_step.error_trace:
                        err_step.error_trace.correction_opportunity = True

        # 检查错误是否与最终回复矛盾
        if final_response and err_step.content:
            # 简单检查：如果错误内容中的关键词出现在最终回复中
            err_keywords = [w for w in err_content_lower.split() if len(w) > 4][:3]
            final_lower = final_response.lower()
            if any(kw in final_lower for kw in err_keywords):
                if err_step.error_trace:
                    err_step.error_trace.contradicts_final = True


# ─── v0.7.0: Cursor 细粒度事件（cot-stream.js 产出）合并 ──────

_EVENT_PROVIDER_PREFIXES = ("codebuddy-",)


def _load_cursor_events(session_id: str) -> List[Dict]:
    """读取 cot-stream*.js 流式写入的 events.jsonl。

    候选路径（v0.18.5 起按优先级查找，第一个存在的就用，剩下的也会合并读取
    以兼容 mid-version 升级时新老路径并存的过渡场景）：

    1. ``$AGENT_COT_DATA_ROOT/events/<sid>/events.jsonl``
       —— ``agent-cot start`` 注入；wheel 安装态默认值。
    2. ``~/.agent-cot/data/events/<sid>/events.jsonl``
       —— v0.18.5 起 cot-stream.js / transcript_watcher.py 的统一默认。
       即便 ``AGENT_COT_DATA_ROOT`` 没注入（手动跑 extract_cot.py
       而不是通过 agent-cot start）也能命中。
    3. ``<COT_EXTRACTOR_ROOT>/output/events/<sid>/events.jsonl``
       —— v0.17 ~ v0.18.4 的旧路径；保留只为了不让升级前生成的旧 events
       立即消失。源码态 dev 跑 extractor 也走这一档。

    多 IDE 接入后，CodeBuddy 会给目录加 provider 前缀，这里会同时
    尝试裸 sid 与已知前缀目录，避免 hook 已落盘但 extractor 读不到。
    返回按 wall-clock 升序排列的事件列表。文件不存在或解析失败都返回空列表，
    不抛异常——这一步失败只意味着"没有实时流数据"，不应该影响主提取。
    """
    import os

    bases: List[Path] = []
    env_data_root = os.environ.get("AGENT_COT_DATA_ROOT")
    if env_data_root:
        bases.append(Path(env_data_root).expanduser() / "events")
    # 用户级默认：跟 cot-stream.js / transcript_watcher.py / backend 同一个真值
    bases.append(Path.home() / ".agent-cot" / "data" / "events")
    # 旧路径兜底：源码态开发 + v0.18.4 及更早遗留
    legacy_root = os.environ.get("COT_EXTRACTOR_ROOT") or str(Path(__file__).resolve().parent.parent)
    bases.append(Path(legacy_root) / "output" / "events")

    events: List[Dict] = []
    candidates: List[Path] = []
    seen: set[str] = set()
    for base in bases:
        for p in [base / session_id / "events.jsonl"]:
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(p)
        if not any(session_id.startswith(prefix) for prefix in _EVENT_PROVIDER_PREFIXES):
            for prefix in _EVENT_PROVIDER_PREFIXES:
                p = base / f"{prefix}{session_id}" / "events.jsonl"
                key = str(p)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(p)
    try:
        # utf-8-sig swallows any leading BOM left by PowerShell or other
        # editors that touched events.jsonl out-of-band; cot-stream.js
        # itself never writes one, but it's cheap insurance.
        for events_path in candidates:
            if not events_path.exists():
                continue
            with open(events_path, "r", encoding="utf-8-sig",
                      errors="replace") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        event = json.loads(raw)
                        event.setdefault("_events_path", str(events_path))
                        events.append(event)
                    except Exception:
                        continue
    except Exception:
        return []
    events.sort(key=lambda e: e.get("t", 0))
    return events


def _detect_agent_type_from_events(events: List[Dict]) -> Optional[str]:
    """Infer agent type from hook-stream events when transcript signals are weak."""
    if not events:
        return None
    codebuddy_events = {
        "PostToolUse",
        "PostToolUseFailure", "PermissionRequest", "PermissionDenied",
        "SubagentStart", "SubagentStop", "TaskCreated", "TaskCompleted",
        "StopFailure", "PreCompact", "PostCompact",
    }
    # v0.18.7: Cursor 自家 hook 写出来的事件名（lower-camel-case，区别于
    # Claude / CodeBuddy 的 PascalCase）。Cursor v2.6+ transcript 不再含
    # tool_use 块，所以 _detect_agent_type(msgs) 会返回 unknown，导致 OTel
    # enricher / 前端 KPI bar 把 cursor session 标成 unknown agent。这里
    # 用事件名指纹补上。
    cursor_events = {
        "afterAgentThought",
        "afterAgentResponse",
        "beforeShellExecution",
        "afterShellExecution",
        "beforeReadFile",
        "afterReadFile",
        "beforeFileEdit",
        "afterFileEdit",
        "beforeMCPExecution",
        "afterMCPExecution",
        "beforeSubmitPrompt",
        "stop",
    }
    for e in events[:200]:
        provider = str(e.get("provider") or "").lower()
        cid = str(e.get("cid") or "").lower()
        event = str(e.get("event") or "")
        payload = e.get("payload") if isinstance(e.get("payload"), dict) else {}
        blob = " ".join(
            str(x).lower()
            for x in (
                provider, cid, event,
                payload.get("source"), payload.get("origin"),
                payload.get("agent"), payload.get("service"),
            )
            if x
        )
        if "codebuddy" in blob or event in codebuddy_events:
            return "codebuddy"
        if "cursor" in blob or "cursor_version" in payload or event in cursor_events:
            return "cursor"
    return None


def _extract_events_only_session(
    *,
    transcript_path: Path,
    session_id: str,
    events: List[Dict],
) -> Optional["SessionCoT"]:
    """Build a minimal CodeBuddy session when hooks exist but transcript is absent.

    This is intentionally conservative: it preserves the event stream in a
    normal CoT shape so the dashboard can show prompt/tool/session activity
    while a richer transcript parser is still unknown.
    """
    if not events:
        return None
    from datetime import datetime, timezone

    agent_type = _detect_agent_type_from_events(events) or "unknown"
    steps: List[ThoughtStep] = []
    tool_calls: List[str] = []
    user_query = ""
    step_idx = 1
    for e in events:
        event = str(e.get("event") or "event")
        t_ms = e.get("t")
        metadata = {
            "observed_at_ms": t_ms,
            "observed_source": "codebuddy_events" if agent_type == "codebuddy" else "agent_events",
            "event": event,
            "cid": e.get("cid"),
            "provider": e.get("provider"),
            "payload": e.get("payload"),
        }
        brief_in = e.get("brief_input") if isinstance(e.get("brief_input"), dict) else {}
        brief_out = e.get("brief_output") if isinstance(e.get("brief_output"), dict) else {}
        tool = str(e.get("tool") or "")

        if event == "UserPromptSubmit":
            content = str(brief_in.get("prompt") or brief_in.get("prompt_probe") or "")
            user_query = user_query or content
            step_type = StepType.USER_INPUT
        elif e.get("thinking_probe"):
            content = str(e.get("thinking_probe"))
            step_type = StepType.THINKING_EXPLICIT
        elif event in ("PreToolUse",):
            content = f"准备调用工具 {tool}: {json.dumps(brief_in, ensure_ascii=False)[:500]}"
            step_type = StepType.TOOL_DECISION
            if tool:
                tool_calls.append(tool)
        elif event in ("PostToolUse", "PostToolUseFailure"):
            content = str(
                brief_out.get("stdout")
                or brief_out.get("stderr")
                or json.dumps(brief_out or e.get("payload") or {}, ensure_ascii=False)[:1000]
            )
            step_type = StepType.TOOL_EXECUTION
        elif event in ("Stop", "StopFailure"):
            content = str(brief_out.get("stdout") or "")
            step_type = StepType.FINAL_RESPONSE if content else "agent_event"
        else:
            content = json.dumps({
                "event": event,
                "tool": tool,
                "brief_input": brief_in,
                "brief_output": brief_out,
            }, ensure_ascii=False)[:1000]
            step_type = "agent_event"

        steps.append(ThoughtStep(
            step_index=step_idx,
            turn_index=1,
            step_type=step_type,
            content=content,
            metadata=metadata,
            tool_name=tool if step_type in (StepType.TOOL_DECISION, StepType.TOOL_EXECUTION) else "",
            tool_use_id=str((e.get("payload") or {}).get("tool_use_id") or f"{event}:{step_idx}")
            if isinstance(e.get("payload"), dict) else f"{event}:{step_idx}",
        ))
        step_idx += 1

    turn = TurnCoT(
        turn_index=1,
        user_query=user_query or "(events-only session)",
        steps=steps,
        tool_calls=tool_calls,
        thinking_depth=sum(1 for s in steps if s.step_type == StepType.THINKING_EXPLICIT),
        total_steps=len(steps),
        final_response=next((s.content for s in reversed(steps)
                             if s.step_type == StepType.FINAL_RESPONSE), ""),
        turn_start_ms_observed=min((int(e.get("t")) for e in events
                                    if isinstance(e.get("t"), (int, float))), default=None),
        turn_end_ms_observed=max((int(e.get("t")) for e in events
                                  if isinstance(e.get("t"), (int, float))), default=None),
    )
    if turn.turn_start_ms_observed is not None and turn.turn_end_ms_observed is not None:
        turn.turn_duration_ms_observed = max(0, turn.turn_end_ms_observed - turn.turn_start_ms_observed)

    observed_stats = {
        "events_total": len(events),
        "events_only": True,
        "agent_type_from_events": agent_type,
        "providers_observed": sorted({
            str(e.get("provider") or "").strip()
            for e in events if e.get("provider")
        }),
        "events_paths": sorted({
            str(e.get("_events_path") or "")
            for e in events if e.get("_events_path")
        }),
        "thought_events": sum(1 for e in events if e.get("thinking_probe")),
        "thought_injected": sum(1 for e in events if e.get("thinking_probe")),
        "thought_orphan": 0,
    }
    return SessionCoT(
        session_id=session_id,
        transcript_path=str(transcript_path),
        extracted_at=datetime.now(timezone.utc).isoformat(),
        turns=[turn],
        total_tool_calls=len(tool_calls),
        total_strategy_shifts=0,
        total_thinking_steps=turn.thinking_depth,
        tool_call_distribution={t: tool_calls.count(t) for t in set(tool_calls)},
        avg_steps_per_turn=float(len(steps)),
        avg_complexity=round(0.3 * len(tool_calls) + 0.2 * turn.thinking_depth, 2),
        observed_events=observed_stats,
        agent_type=agent_type,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CodeBuddy native transcript → SessionCoT
# ─────────────────────────────────────────────────────────────────────────────
#
# CodeBuddy stores conversations under
# %LOCALAPPDATA%\CodeBuddyExtension\Data\<machine>\CodeBuddyIDE\<machine>\
# history\<workspace>\<sid>\{index.json, messages\<msg_id>.json}
#
# The schema is double-JSON-encoded (each message file's "message" field is
# itself a JSON string). We delegate the file-level parsing to
# ``codebuddy_transcript`` so this module only needs to know about the
# already-normalized CoT shape. See codebuddy_transcript.py for the schema
# walkthrough.
# ─── CodeBuddy ↔ Cursor 工具名归一（v0.19.6 真正落地） ──────────────────
#
# CodeBuddy 的 transcript 写的是 snake_case 工具名（``read_file`` / ``todo_write``
# / ``execute_command`` 等），跟 Cursor / Claude Code 的 PascalCase 不一致。
# 0.19.4 的 release notes 承诺过统一，但 helper 当时只写在 CHANGELOG 没真正
# 落地代码 —— 0.19.6 在这里补回去。
#
# 规则：只翻译"两边都存在且语义对等"的工具（user 原话："不一样的不要统一"），
# CodeBuddy 独有的 ``task`` / ``attempt_completion`` 保持原状。
# ``todo_write`` 必须翻译，因为 ``_build_plan_timeline`` 只认 ``TodoWrite``——
# 不翻译会让 plan_timeline 永远空、前端 Plan 卡片消失（bug 1 的根因）。
_CODEBUDDY_TO_CURSOR_TOOL: Dict[str, str] = {
    "read_file":        "Read",
    "write_to_file":    "Write",
    "replace_in_file":  "Edit",
    "execute_command":  "Shell",
    "search_content":   "Grep",
    "search_file":      "Glob",
    "list_files":       "LS",
    "todo_write":       "TodoWrite",
}


def _normalize_codebuddy_tool_name(name: str) -> str:
    """Map CodeBuddy snake_case tool names to Cursor PascalCase equivalents.

    Returns the original name unchanged for CodeBuddy-only tools (e.g. ``task``)
    or anything not in the curated mapping. Idempotent: calling on an already
    PascalCase name returns it unchanged.
    """
    if not isinstance(name, str) or not name:
        return name
    return _CODEBUDDY_TO_CURSOR_TOOL.get(name, name)


def _coerce_codebuddy_todos(tool_name: str, args: Any) -> Any:
    """For ``TodoWrite``-equivalent calls, parse ``todos`` from JSON string → list.

    CodeBuddy writes ``tool_input.todos`` as a JSON-encoded string
    (``'[{"id":"1","status":"in_progress","content":"..."}]'``), whereas
    ``_build_plan_timeline`` requires ``todos`` to be a ``List[Dict]``. Without
    this coercion, plan_timeline stays empty even after tool name normalization
    (this is the second half of bug 1).

    Only mutates ``args`` when:
      - ``tool_name == "TodoWrite"`` (post-normalization), AND
      - ``args`` is a dict, AND
      - ``args["todos"]`` is a non-empty string that parses to a JSON list.

    Returns a (possibly new) dict; never raises.
    """
    if tool_name != "TodoWrite":
        return args
    if not isinstance(args, dict):
        return args
    todos = args.get("todos")
    if isinstance(todos, list):
        return args  # already a list, nothing to do
    if not isinstance(todos, str) or not todos.strip():
        return args
    try:
        parsed = json.loads(todos)
    except (ValueError, TypeError):
        return args
    if not isinstance(parsed, list):
        return args
    # Shallow copy + replace; never mutate the caller's dict in place.
    new_args = dict(args)
    new_args["todos"] = parsed
    return new_args


def _extract_codebuddy_session_from_transcript(
    *,
    index_path: Path,
    session_id: str,
    events: List[Dict],
) -> Optional["SessionCoT"]:
    """Build a full SessionCoT from a CodeBuddy native transcript.

    This is the rich path — it surfaces ``reasoning`` (Hunyuan & friends'
    Chain-of-Thought, *unredacted* in CodeBuddy's local store), assistant
    text, tool-call → tool-result pairs, and per-request token usage.

    Falls back to ``_extract_events_only_session`` if the index is empty.
    """
    from datetime import datetime, timezone
    try:
        from codebuddy_transcript import (  # type: ignore
            aggregate_token_usage,
            collect_models,
            extract_user_text,
            index_request_summaries,
            load_transcript,
            split_assistant_blocks,
            split_tool_results,
            stringify_tool_input,
            stringify_tool_result,
        )
    except Exception:  # pragma: no cover — soft import
        return None

    index_blob, messages = load_transcript(index_path)
    if not messages:
        return None

    # Token usage per request_id, used to attach `usage` to each turn.
    requests = index_request_summaries(index_blob)
    usage_by_request = {
        r["request_id"]: {
            "input_tokens": int(r.get("usage", {}).get("inputTokens") or 0),
            "output_tokens": int(r.get("usage", {}).get("outputTokens") or 0),
            "total_tokens": int(r.get("usage", {}).get("totalTokens") or 0),
            "last_tokens": int(r.get("usage", {}).get("lastTokens") or 0),
        }
        for r in requests
        if r.get("request_id")
    }
    started_by_request = {
        r["request_id"]: r.get("started_at_ms")
        for r in requests
        if r.get("request_id")
    }

    # Walk messages, segmenting on every user role. Each segment becomes
    # one TurnCoT. Tool messages always belong to the *current* turn (they
    # are the result of the previous assistant's tool-calls).
    turns: List[TurnCoT] = []
    cur_turn: Optional[TurnCoT] = None
    cur_request_id: Optional[str] = None
    global_step_idx = 0
    tool_calls_total: List[str] = []

    def _new_turn(user_msg) -> TurnCoT:
        nonlocal cur_request_id
        cur_request_id = user_msg.request_id
        return TurnCoT(
            turn_index=len(turns) + 1,
            user_query=extract_user_text(user_msg) or "(no user text)",
        )

    # Pre-walk: build per-msg "real" timestamps. assistant has its own
    # ``generated_at_ms`` from responseId; user/tool inherit the surrounding
    # assistant ts (user = next assistant gen ts; tool = previous assistant
    # gen ts) so the frontend can still order them on the timeline. None
    # values are kept None — never fabricate.
    asst_ts_seq: List[Tuple[int, Optional[int]]] = []   # (msg_index, generated_at_ms)
    for i, m in enumerate(messages):
        if m.role == "assistant" and isinstance(m.generated_at_ms, int):
            asst_ts_seq.append((i, m.generated_at_ms))

    def _msg_ts_ms(idx: int) -> Optional[int]:
        m = messages[idx]
        if m.role == "assistant" and isinstance(m.generated_at_ms, int):
            return m.generated_at_ms
        if m.role == "tool":
            # tool 结果产生在「前一个 assistant 决策完之后、下一个 assistant 拿到结果之前」
            # 取前一个 assistant ts 作锚点（最准的近似，且与 tool_decision 同时刻）
            prev = [t for j, t in asst_ts_seq if j < idx]
            if prev:
                return prev[-1]
            return None
        if m.role == "user":
            # 第一条 user 用 request startedAt（毫秒级真值）；后续 user 用其 request startedAt
            rid = m.request_id
            if rid and isinstance(started_by_request.get(rid), (int, float)):
                return int(started_by_request[rid])
            return None
        return None

    def _next_asst_ts_after(idx: int, *, same_request_id: Optional[str] = None) -> Optional[int]:
        """下一条 assistant 的 generated_at_ms。如果传 same_request_id，
        只在同 request 内查找——避免 tool 横跨到下一 turn 把 idle 算进时长。
        """
        for j, t in asst_ts_seq:
            if j <= idx:
                continue
            if same_request_id is not None:
                if messages[j].request_id != same_request_id:
                    return None
            return t
        return None

    for mi, msg in enumerate(messages):
        msg_ts = _msg_ts_ms(mi)
        msg_ts_iso = (
            datetime.fromtimestamp(msg_ts / 1000, tz=timezone.utc).isoformat()
            if isinstance(msg_ts, int) else None
        )

        if msg.role == "user":
            if cur_turn is not None:
                cur_turn.total_steps = len(cur_turn.steps)
                turns.append(cur_turn)
            cur_turn = _new_turn(msg)
            if isinstance(msg_ts, int):
                cur_turn.turn_start_ms_observed = int(msg_ts)
            global_step_idx += 1
            cur_turn.steps.append(ThoughtStep(
                step_index=global_step_idx,
                turn_index=cur_turn.turn_index,
                step_type=StepType.USER_INPUT,
                content=cur_turn.user_query,
                timestamp=msg_ts_iso,
                metadata={
                    "msg_id": msg.msg_id,
                    "request_id": msg.request_id,
                    "model_id": msg.model_id,
                    "trace_id": msg.trace_id,
                    "observed_source": "codebuddy_transcript",
                    "observed_at_ms": msg_ts,
                },
            ))
            continue

        if cur_turn is None:
            # Defensive: assistant/tool message with no preceding user.
            # Synthesize an empty turn so we don't drop content.
            cur_turn = TurnCoT(turn_index=1, user_query="(unknown user prompt)")

        if msg.role == "assistant":
            buckets = split_assistant_blocks(msg)
            cur_request_id = msg.request_id or cur_request_id
            for r in buckets["reasoning"]:
                global_step_idx += 1
                txt = str(r.get("text") or "").strip()
                if not txt:
                    continue
                cur_turn.steps.append(ThoughtStep(
                    step_index=global_step_idx,
                    turn_index=cur_turn.turn_index,
                    step_type=StepType.THINKING_EXPLICIT,
                    content=txt,
                    timestamp=msg_ts_iso,
                    metadata={
                        "msg_id": msg.msg_id,
                        "request_id": msg.request_id,
                        "model_id": msg.model_id,
                        "model_name": msg.model_name,
                        "trace_id": msg.trace_id,
                        "response_id": msg.response_id,
                        "observed_source": "codebuddy_reasoning",
                        "observed_at_ms": msg_ts,
                        "thought_chars": len(txt),
                    },
                ))
                cur_turn.thinking_depth += 1
            # text blocks: usually a single human-facing reply per message;
            # if more text follows tool-calls in the SAME message, treat
            # earlier ones as pre_tool_reasoning and the very last as a
            # candidate final_response (overridden if more assistant
            # messages follow in the same turn).
            text_blocks = buckets["text"]
            for i, t in enumerate(text_blocks):
                txt = str(t.get("text") or "").strip()
                if not txt:
                    continue
                global_step_idx += 1
                is_last_in_msg = (i == len(text_blocks) - 1)
                will_call_tools = bool(buckets["tool_calls"])
                step_type = (
                    StepType.PRE_TOOL_REASONING
                    if (will_call_tools and is_last_in_msg) or not is_last_in_msg
                    else StepType.FINAL_RESPONSE
                )
                cur_turn.steps.append(ThoughtStep(
                    step_index=global_step_idx,
                    turn_index=cur_turn.turn_index,
                    step_type=step_type,
                    content=txt,
                    timestamp=msg_ts_iso,
                    metadata={
                        "msg_id": msg.msg_id,
                        "request_id": msg.request_id,
                        "model_id": msg.model_id,
                        "model_name": msg.model_name,
                        "trace_id": msg.trace_id,
                        "response_id": msg.response_id,
                        "observed_source": "codebuddy_text",
                        "observed_at_ms": msg_ts,
                    },
                ))
                if step_type == StepType.FINAL_RESPONSE:
                    cur_turn.final_response = txt
            for tc in buckets["tool_calls"]:
                # v0.19.6: 把 CodeBuddy 原生 snake_case 工具名映射到 Cursor 风格
                # PascalCase（todo_write → TodoWrite 等），同时把 todos 字符串
                # 解析成 list —— 见 _CODEBUDDY_TO_CURSOR_TOOL / _coerce_codebuddy_todos。
                raw_tool_name = str(tc.get("toolName") or "unknown")
                tool_name = _normalize_codebuddy_tool_name(raw_tool_name)
                args = tc.get("args") or tc.get("input") or {}
                args = _coerce_codebuddy_todos(tool_name, args)
                global_step_idx += 1
                cur_turn.steps.append(ThoughtStep(
                    step_index=global_step_idx,
                    turn_index=cur_turn.turn_index,
                    step_type=StepType.TOOL_DECISION,
                    content=stringify_tool_input(args),
                    tool_name=tool_name,
                    tool_use_id=str(tc.get("toolCallId") or ""),
                    timestamp=msg_ts_iso,
                    metadata={
                        "msg_id": msg.msg_id,
                        "request_id": msg.request_id,
                        "model_id": msg.model_id,
                        "model_name": msg.model_name,
                        "trace_id": msg.trace_id,
                        "tool_input": args,
                        # 留一份原始 CodeBuddy 工具名，方便日后审计 / debug——
                        # 不影响前端（前端只读 tool_name）。
                        "tool_name_raw": raw_tool_name,
                        "observed_source": "codebuddy_tool_call",
                        "observed_at_ms": msg_ts,
                    },
                ))
                cur_turn.tool_calls.append(tool_name)
                tool_calls_total.append(tool_name)
            continue

        if msg.role == "tool":
            # 工具实际执行的真实墙钟窗口：[前一 assistant ts, 下一 assistant ts]
            # 限制在同 request_id 内查找下一 assistant —— 防止跨 turn 把 idle
            # 时间（用户停下不动）算成 tool 时长。
            window_start_ms = msg_ts                # 前一 asst ts
            window_end_ms = _next_asst_ts_after(mi, same_request_id=msg.request_id)
            tool_dur_ms: Optional[int] = None
            if isinstance(window_start_ms, int) and isinstance(window_end_ms, int):
                if window_end_ms > window_start_ms:
                    tool_dur_ms = window_end_ms - window_start_ms

            for tr in split_tool_results(msg):
                # v0.19.6: tool_execution 工具名也走同一套归一规则，保证 D-E 配对
                # 用相同字符串（前端 pairToolDecisionExecution 按名字 + tool_use_id
                # 配对，名字不一致就掉对）。
                raw_tool_name = str(tr.get("toolName") or "")
                tool_name = _normalize_codebuddy_tool_name(raw_tool_name)
                tool_use_id = str(tr.get("toolCallId") or "")
                is_error = bool(tr.get("isError"))
                content = stringify_tool_result(tr.get("result"))
                global_step_idx += 1
                cur_turn.steps.append(ThoughtStep(
                    step_index=global_step_idx,
                    turn_index=cur_turn.turn_index,
                    step_type=StepType.TOOL_EXECUTION,
                    content=content,
                    tool_name=tool_name,
                    tool_use_id=tool_use_id,
                    timestamp=msg_ts_iso,
                    duration_ms=tool_dur_ms,
                    metadata={
                        "msg_id": msg.msg_id,
                        "is_error": is_error,
                        "raw_result": tr.get("result"),
                        "tool_name_raw": raw_tool_name,
                        "observed_source": "codebuddy_tool_result",
                        "observed_at_ms": msg_ts,
                        "tool_window_ms": (
                            {"start_ms": window_start_ms, "end_ms": window_end_ms}
                            if isinstance(window_start_ms, int) and isinstance(window_end_ms, int)
                            else None
                        ),
                        "result_chars": len(content) if isinstance(content, str) else 0,
                        # 老前端 SubNode 用 result_len —— 同步一份避免 UI 显示 0 chars
                        "result_len": len(content) if isinstance(content, str) else 0,
                        # 标识：本步数据来自 CodeBuddy 原生 transcript，非合成占位。
                        # 让前端可以贴一个 "↑ transcript" 徽章对齐 cursor 的 ↑ upgraded。
                        "captured_from": "codebuddy_transcript",
                    },
                ))
            continue

    if cur_turn is not None:
        cur_turn.total_steps = len(cur_turn.steps)
        turns.append(cur_turn)

    # ── 把 assistant message 内的多个连续 step 也分配真实 step 级 duration ──
    #
    # 同一条 assistant message 里的所有 step (reasoning + text + tool_decision)
    # 共享一个 generated_at_ms。它们之间的"内部顺序耗时"transcript 没记，但整条
    # message 的 LLM 生成总耗时 ≈ T_this_asst - T_prev_asst（前一条已结束到这条
    # 出来的 wall-clock 间隔）。把这个总耗时分到这条 message 的所有 step 上：
    #
    #   - 唯一 step 直接吃总耗时；
    #   - 多 step 时按 thought_chars / tool_input 长度等权分摊；
    #   - 没有前一条 assistant 时（首条）退化用 request startedAt 当锚点。
    #
    # 这是「能从 transcript 真实推出的最细粒度」，绝不再外推 idle 时间到 step 上。
    asst_ts_in_order = [t for _, t in asst_ts_seq]
    asst_msg_in_order = [messages[i] for i, _ in asst_ts_seq]
    # 每条 assistant message 的"前锚点"——优先级：
    #   1) 同 request 内的前一条 assistant ts（同 request → 没有跨 turn idle）
    #   2) 否则用本 request 的 startedAt（每 turn 的第一条 assistant 必走这一支，
    #      避免拿到上一 turn 的 ts 把 idle 也吃进 step duration）
    prev_anchor_by_asst: List[Optional[int]] = []
    last_ts_by_request: Dict[str, int] = {}
    for i, m in enumerate(asst_msg_in_order):
        rid = m.request_id
        anchor: Optional[int] = None
        if rid and rid in last_ts_by_request:
            anchor = last_ts_by_request[rid]
        elif rid and isinstance(started_by_request.get(rid), (int, float)):
            anchor = int(started_by_request[rid])
        prev_anchor_by_asst.append(anchor)
        if rid:
            last_ts_by_request[rid] = asst_ts_in_order[i]

    for i, asst in enumerate(asst_msg_in_order):
        gen_ms = asst_ts_in_order[i]
        anchor = prev_anchor_by_asst[i]
        if not (isinstance(gen_ms, int) and isinstance(anchor, int) and gen_ms > anchor):
            continue
        total_ms = gen_ms - anchor
        # 收集本条 message 产生的所有 step（msg_id 是稳定外键）
        steps_of_msg: List["ThoughtStep"] = []
        for tt in turns:
            for s in tt.steps:
                md = s.metadata or {}
                if md.get("msg_id") == asst.msg_id and s.step_type != StepType.TOOL_EXECUTION:
                    steps_of_msg.append(s)
        if not steps_of_msg:
            continue
        weights = []
        for s in steps_of_msg:
            if s.step_type in (StepType.THINKING_EXPLICIT, StepType.PRE_TOOL_REASONING,
                               StepType.FINAL_RESPONSE):
                w = max(1, len(s.content or ""))
            else:  # tool_decision
                w = max(1, len((s.metadata or {}).get("tool_input") and
                              json.dumps((s.metadata or {}).get("tool_input"),
                                         ensure_ascii=False) or "{}"))
            weights.append(w)
        wsum = sum(weights) or 1
        # 整数毫秒分配，余数补到最后一项，杜绝累计漂移
        allocated = 0
        for j, s in enumerate(steps_of_msg):
            if j == len(steps_of_msg) - 1:
                s.duration_ms = total_ms - allocated
            else:
                d = int(round(total_ms * weights[j] / wsum))
                s.duration_ms = d
                allocated += d

    # ── 真实 turn 时长 = 本 turn 内最后一条 assistant ts - request startedAt ──
    #
    # 这是 CodeBuddy 给我们的真值上限：用户按下回车（startedAt） → 模型最后一次
    # 回应（last assistant generated_at_ms）。中间是 LLM 生成 + 工具执行的全部
    # 真实活跃时间，不含「下一次 user 输入前的 idle」。
    for turn in turns:
        rid = next((s.metadata.get("request_id") for s in turn.steps
                    if isinstance(s.metadata, dict) and s.metadata.get("request_id")), None)
        if rid and rid in usage_by_request:
            turn.usage = usage_by_request[rid]
        ts = started_by_request.get(rid) if rid else None
        if isinstance(ts, (int, float)):
            turn.turn_start_ms_observed = int(ts)
            turn.turn_start_time = datetime.fromtimestamp(
                int(ts) / 1000, tz=timezone.utc).isoformat()
        # 找本 turn 内最后一条 assistant generated_at_ms
        last_asst_ts = max(
            (int(md.get("observed_at_ms"))
             for s in turn.steps
             if (md := (s.metadata or {})).get("observed_source") in
                ("codebuddy_reasoning", "codebuddy_text", "codebuddy_tool_call")
                and isinstance(md.get("observed_at_ms"), int)),
            default=None,
        )
        if last_asst_ts and turn.turn_start_ms_observed and last_asst_ts > turn.turn_start_ms_observed:
            turn.turn_end_ms_observed = int(last_asst_ts)
            dur = int(last_asst_ts) - int(turn.turn_start_ms_observed)
            turn.turn_duration_ms_observed = dur
            turn.turn_duration_ms = dur

    # ── Session-level rollups ────────────────────────────────────────────
    totals = aggregate_token_usage(index_blob)
    models = collect_models(messages)
    trace_ids = sorted({m.trace_id for m in messages if m.trace_id})

    # ── Per-turn 真实 model 检测（多模型/同 session 切换都能正确识别） ──
    #
    # 数据真值：每条 assistant 消息的 ``msg.model_id``（CodeBuddy extra.modelId
    # 字段，写到 step.metadata.model_id）。我们直接从 turn.steps 里读，按出现
    # 频次取主导 —— 用户中途切了模型时，主导会自然落到该 turn 实际用得最多
    # 的那个 model_id 上，不需要任何模型名硬编码。
    #
    # 同时记录 ``turn_models``：本 turn 用过的所有 model_id（去重，按出现顺序）。
    # 当 turn 内多次切换模型，前端可以选择展示主导 + 全名单。
    def _turn_model_signal(t: TurnCoT) -> Tuple[Optional[str], List[str], Optional[str]]:
        """Return (dominant_model_id, distinct_models_in_order, model_name).

        Reads strictly from ``step.metadata.model_id`` —— assistant 消息上
        必然带，user/tool step 可能没有，统计时跳过 None。
        """
        seq: List[str] = []
        seen_set: set = set()
        seen_order: List[str] = []
        name_for: Dict[str, str] = {}
        for s in t.steps:
            md = s.metadata or {}
            mid = md.get("model_id")
            if not isinstance(mid, str) or not mid:
                continue
            seq.append(mid)
            if mid not in seen_set:
                seen_set.add(mid)
                seen_order.append(mid)
            mname = md.get("model_name")
            if isinstance(mname, str) and mname and mid not in name_for:
                name_for[mid] = mname
        if not seq:
            return None, [], None
        # 出现频次最多的就是本 turn 的主导 model
        from collections import Counter
        dominant = Counter(seq).most_common(1)[0][0]
        return dominant, seen_order, name_for.get(dominant)

    # 每条 assistant message 的 model_name（前端可读人话）也单独存下来
    model_name_by_id: Dict[str, str] = {}
    for m in messages:
        if m.model_id and m.model_name and m.model_id not in model_name_by_id:
            model_name_by_id[m.model_id] = m.model_name

    # 给每个 turn 标 model（写到 step.metadata.turn_model 也行，但主路径是
    # turn.otel；这里先收集，后面 host_step 注入时会用）
    turn_dominant_model: Dict[int, Optional[str]] = {}
    turn_models_seen: Dict[int, List[str]] = {}
    turn_model_display_name: Dict[int, Optional[str]] = {}
    for t in turns:
        dom, seen, name = _turn_model_signal(t)
        turn_dominant_model[t.turn_index] = dom
        turn_models_seen[t.turn_index] = seen
        turn_model_display_name[t.turn_index] = name

    # session 级 dominant_model = 出现频次累加最多的 model_id（不是首条！）
    from collections import Counter as _Counter
    _session_model_counter: _Counter = _Counter()
    for t in turns:
        for m in turn_models_seen.get(t.turn_index, []) or []:
            # 用每 turn 各自的频次累加—— turn 内频次已经经过 dominant 提取，
            # 但 distinct list 不带 count；这里以"该 turn 主导 +1"近似累加，
            # 会议长 turn 主导权会自然胜出，避免短 turn 个别 model 把全局拉偏。
            pass
        d = turn_dominant_model.get(t.turn_index)
        if d:
            _session_model_counter[d] += 1
    if _session_model_counter:
        session_dominant_model: Optional[str] = _session_model_counter.most_common(1)[0][0]
    elif models:
        session_dominant_model = models[0]
    else:
        session_dominant_model = None

    # session 真实活跃时长 = 各 turn 实际费时之和（不含 idle 间隔）。
    # session 起止时间窗仍然记录全量（startedAt[0] 到最后 assistant ts），
    # 但前端展示的 "总耗时" 用 active 累加值。
    active_session_ms = sum(
        int(t.turn_duration_ms_observed or 0)
        for t in turns
        if isinstance(t.turn_duration_ms_observed, int)
    ) or None

    started_values = [int(v) for v in started_by_request.values()
                      if isinstance(v, (int, float))]
    session_start_ms = min(started_values) if started_values else None
    last_asst_global = max(
        (int(t.turn_end_ms_observed) for t in turns
         if isinstance(t.turn_end_ms_observed, int)),
        default=None,
    )
    session_end_ms = last_asst_global

    client_name = next(
        (str((e.get("payload") or {}).get("client"))
         for e in events
         if isinstance(e.get("payload"), dict) and (e.get("payload") or {}).get("client")),
        None,
    )
    client_version = next(
        (str((e.get("payload") or {}).get("version"))
         for e in events
         if isinstance(e.get("payload"), dict) and (e.get("payload") or {}).get("version")),
        None,
    )
    # session_wall_ms = 真实墙钟跨度（含 idle）；session_active_ms = 累加 turn 真实费时
    session_wall_ms = (
        int(session_end_ms) - int(session_start_ms)
        if (isinstance(session_start_ms, int) and isinstance(session_end_ms, int)
            and session_end_ms > session_start_ms)
        else None
    )
    # model_timeline：[{turn_index, model_id, model_name, models_seen}, ...]
    # 让前端能直接看到「session 中模型何时切换、每 turn 实际用啥」。
    model_timeline = [
        {
            "turn_index": t.turn_index,
            "model_id": turn_dominant_model.get(t.turn_index),
            "model_name": turn_model_display_name.get(t.turn_index),
            "models_seen": turn_models_seen.get(t.turn_index, []),
        }
        for t in turns
    ]
    session_meta: Dict[str, Any] = {
        "agent_type": "codebuddy",
        "models": models,
        # session 主导 model = 各 turn 主导 model 的众数（v0.17.2，不再是 first-seen）
        "model_id": session_dominant_model,
        "model_names": dict(model_name_by_id),
        "model_timeline": model_timeline,
        "trace_ids": trace_ids,
        "transcript_path": str(index_path),
        "session_start_ms_observed": session_start_ms,
        "session_end_ms_observed": session_end_ms,
        # 前端 SpanTree.totalDuration 直接 sum(turn.turn_duration_ms_observed)，
        # 已经只算活跃时间。这两个字段单纯做 transparency：
        "session_active_ms_observed": active_session_ms,
        "session_wall_ms_observed": session_wall_ms,
        # 兼容老前端（cursor 用的字段名）。值取 active —— 跟 turn 累加一致。
        "session_duration_ms_observed": active_session_ms,
        "client": client_name,
        "client_version": client_version,
        "hook_events_observed": _summarize_event_counts(events),
    }

    observed_stats = {
        "events_total": len(events),
        "events_only": False,
        "transcript_format": "codebuddy_index_json",
        "agent_type_from_events": _detect_agent_type_from_events(events),
        "providers_observed": sorted({
            str(e.get("provider") or "").strip()
            for e in events if e.get("provider")
        }),
        "events_paths": sorted({
            str(e.get("_events_path") or "")
            for e in events if e.get("_events_path")
        }),
        "thought_events": sum(1 for s in (st for tt in turns for st in tt.steps)
                              if s.step_type == StepType.THINKING_EXPLICIT),
        "thought_injected": sum(1 for s in (st for tt in turns for st in tt.steps)
                                if s.step_type == StepType.THINKING_EXPLICIT),
        "thought_orphan": 0,
        "codebuddy_messages_total": len(messages),
        "codebuddy_requests_total": len(requests),
        "codebuddy_token_usage_total": totals,
    }

    total_steps = sum(len(t.steps) for t in turns)
    total_thinking = sum(t.thinking_depth for t in turns)

    # ── v0.17.1: synth otel_view + invocation_stats + per-step token_usage ──
    #
    # CodeBuddy session 不会再走 cot_otel_enricher（return rich, offset 早退），
    # 但前端 SessionList / SessionDetail / SpanTree 的 KPI 面板全靠 cot.otel_view
    # / cot.invocation_stats / step.otel.token_usage 三个字段。这里直接从
    # transcript 真实数据合成它们，让 codebuddy 的展示丰富度对齐 cursor。
    #
    # ⚠ 仅在 codebuddy 路径生效；不会触碰 cursor / claude 走的 enricher 链路。
    invocation_stats = InvocationStats(
        llm_calls=len(requests),
        rag_queries=0,
        web_searches=0,
        llm_call_distribution={"codebuddy": len(requests)} if requests else {},
        rag_query_distribution={},
    )

    cb_agent_name = client_name or "CodeBuddyIDE"
    # session 主导 model 仅作 fallback 显示用；per-turn 的 step.otel / turn.otel
    # 永远用 turn 自身实际识别到的 model，不再硬塞 session model。
    session_model_for_view = session_dominant_model or "unknown"
    session_provider_for_view = _provider_from_codebuddy_model(session_dominant_model)

    in_tok_total = int(totals.get("input_tokens") or 0)
    out_tok_total = int(totals.get("output_tokens") or 0)

    actual_token_usage = {
        "input_tokens": in_tok_total,
        "output_tokens": out_tok_total,
        # CodeBuddy transcript 没显式 cache 命中字段；保持 0 让前端不画 cache 块
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "agent_response_count": len(requests),
        "full_price_cost_usd": None,
        "source": "codebuddy_transcript",
        "assistant_messages_counted": sum(1 for m in messages if m.role == "assistant"),
    }

    # otel_view 的 duration_ms 取 active —— 跟 turn 累加一致，避免 "152 min" 误导。
    # 同时单独保留 wall_ms 给 power user 看真实跨度（前端可选展示）。
    duration_ms_session = active_session_ms

    # model_distribution 真实计数：每条 assistant 消息一个 model_id 计 1
    model_dist: Dict[str, int] = {}
    for m in messages:
        if m.role == "assistant" and m.model_id:
            model_dist[m.model_id] = model_dist.get(m.model_id, 0) + 1

    otel_view: Dict[str, Any] = {
        "schema": "codebuddy_synth_v1",
        "model": session_model_for_view,
        "model_source": "codebuddy_transcript",
        "models_seen": models,
        "model_distribution": model_dist,
        # 用 codebuddy_ 前缀避开 cursor 已有的 model_timeline 类型（Array<{t_ms,model}>）
        "codebuddy_model_timeline": model_timeline,  # [{turn_index, model_id, model_name, models_seen}]
        "provider": session_provider_for_view,
        "agent_name": cb_agent_name,
        "actual_cost_usd": None,         # 没有官方 hunyuan pricing
        "totals": {
            "input_tokens": in_tok_total,
            "output_tokens": out_tok_total,
            "cost_usd": None,
        },
        "actual_token_usage": actual_token_usage,
        "client_runtime": {
            "cursor_version": client_version,   # 复用同 KPI 行；前端只是当作版本展示
            "events_count": len(events),
            "events_path": next(
                (str(e.get("_events_path")) for e in events if e.get("_events_path")),
                None,
            ),
        },
        "duration_ms": duration_ms_session,
        "wall_duration_ms": session_wall_ms,
        "trace_ids": trace_ids,
        "missing_signals": [
            # 前端会把这些列出来作为"unavailable"提示
            {"field": "actual_cost_usd",
             "reason": "no_pricing_table",
             "detail": f"无 {session_model_for_view} 的 pricing 表，cost 无法换算 USD"},
            {"field": "cache_read_tokens",
             "reason": "transcript_lacks_cache_field",
             "detail": "CodeBuddy transcript 不区分 cache/非 cache 输入"},
        ],
    }

    # ── 把每 turn 的 token usage + 真实 model 注入到该 turn 的首个 LLM-like step ──
    # 让 SpanTree turnOtel 聚合能算出每轮 in/out tokens、亮起 ↧/↥ chip，并
    # 让前端能拿到 turn 真实使用的 model（不是套到所有 turn 的 session 主导值）。
    for turn in turns:
        if not turn.usage:
            continue
        # 该 turn 的真实 model（自动从 step.metadata.model_id 检测）
        turn_model_id = turn_dominant_model.get(turn.turn_index) or session_model_for_view
        turn_model_name = (
            turn_model_display_name.get(turn.turn_index)
            or model_name_by_id.get(turn_model_id)
            or turn_model_id
        )
        turn_provider = _provider_from_codebuddy_model(turn_model_id)

        # 也把 turn-level otel 落到 dataclass 上（下游 OTLP exporter / 报表都能直接用）
        turn.otel = {
            "model": turn_model_id,
            "model_name": turn_model_name,
            "model_source": "codebuddy_transcript",
            "models_seen": turn_models_seen.get(turn.turn_index, []),
            "provider": turn_provider,
            "agent_name": cb_agent_name,
            "operation_name": "invoke_agent",
            "duration_ms": turn.turn_duration_ms_observed,
            "token_usage": {
                "input_tokens": int(turn.usage.get("input_tokens") or 0),
                "output_tokens": int(turn.usage.get("output_tokens") or 0),
                "cost_usd": None,
                "cost_reason": "no_pricing_table",
            },
            "attributes": {
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.request.model": turn_model_id,
                "gen_ai.response.model": turn_model_id,
                "gen_ai.provider.name": turn_provider,
                "gen_ai.usage.input_tokens": int(turn.usage.get("input_tokens") or 0),
                "gen_ai.usage.output_tokens": int(turn.usage.get("output_tokens") or 0),
            },
        }

        host_step = None
        for s in turn.steps:
            if s.step_type in (
                StepType.THINKING_EXPLICIT,
                StepType.PRE_TOOL_REASONING,
                StepType.FINAL_RESPONSE,
            ):
                host_step = s
                break
        if host_step is None:
            continue
        attrs: Dict[str, Any] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": turn_model_id,
            "gen_ai.response.model": turn_model_id,
            "gen_ai.provider.name": turn_provider,
            "gen_ai.usage.input_tokens": int(turn.usage.get("input_tokens") or 0),
            "gen_ai.usage.output_tokens": int(turn.usage.get("output_tokens") or 0),
        }
        if isinstance(turn.turn_duration_ms_observed, int):
            attrs["gen_ai.client.operation.duration"] = float(turn.turn_duration_ms_observed)
        host_step.otel = {
            "step_kind": "llm_call",
            "operation_name": "chat",
            "model": turn_model_id,
            "model_source": "codebuddy_transcript",
            "provider": turn_provider,
            "token_usage": {
                "input_tokens": int(turn.usage.get("input_tokens") or 0),
                "output_tokens": int(turn.usage.get("output_tokens") or 0),
                "cost_usd": None,
                "cost_reason": "no_pricing_table",
            },
            "attributes": attrs,
        }

    # v0.19.6: CodeBuddy session 早退（line 5482 的 ``return rich, offset``）后
    # 不会再走 run_extract 主流程，所以 plan_timeline / mode_transitions 必须
    # 在这里就地生成，否则前端 Plan 卡片对 CodeBuddy 永远是空（bug 1 后半段）。
    # 任一构建失败都软退出，不影响 SessionCoT 主体返回。
    cb_plan_timeline: List[PlanSnapshot] = []
    cb_mode_transitions: List[ModeTransition] = []
    cb_plan_proposals: List[PlanProposal] = []
    try:
        cb_plan_timeline = _build_plan_timeline(turns)
    except Exception:
        cb_plan_timeline = []
    try:
        cb_mode_transitions, cb_plan_proposals = _build_mode_transitions(turns)
    except Exception:
        cb_mode_transitions = []
        cb_plan_proposals = []

    return SessionCoT(
        session_id=session_id,
        transcript_path=str(index_path),
        extracted_at=datetime.now(timezone.utc).isoformat(),
        turns=turns,
        total_tool_calls=len(tool_calls_total),
        total_strategy_shifts=0,
        total_thinking_steps=total_thinking,
        tool_call_distribution={t: tool_calls_total.count(t) for t in set(tool_calls_total)},
        avg_steps_per_turn=round(total_steps / max(1, len(turns)), 2),
        avg_complexity=round(0.3 * len(tool_calls_total) + 0.2 * total_thinking, 2),
        observed_events=observed_stats,
        agent_type="codebuddy",
        session_meta=session_meta,
        invocation_stats=invocation_stats,
        otel_view=otel_view,
        plan_timeline=cb_plan_timeline,
        mode_transitions=cb_mode_transitions,
        plan_proposals=cb_plan_proposals,
    )


def _provider_from_codebuddy_model(model_id: Optional[str]) -> str:
    """Map a CodeBuddy model_id → vendor / provider for the OTel view.

    CodeBuddy IDE 自带模型多用厂内别名（hy3-…=hunyuan, gpt-4-…, claude-…），
    这里只覆盖能从前缀稳定判断的几族；其它一律落 ``codebuddy``，避免误标。
    Cursor 的 _provider_from_model 不动，省得动到它已稳定的 Cursor 路径。
    """
    if not model_id:
        return "codebuddy"
    m = model_id.lower()
    if m.startswith("hy") or "hunyuan" in m:
        return "tencent"
    if m.startswith(("claude-", "claude_")):
        return "anthropic"
    if m.startswith(("gpt-", "o1", "o3", "o4")) or "codex" in m:
        return "openai"
    if "gemini" in m:
        return "google"
    if "deepseek" in m or m.startswith("ds-"):
        return "deepseek"
    if "qwen" in m or "tongyi" in m:
        return "alibaba"
    if "glm" in m or "zhipu" in m:
        return "zhipu"
    return "codebuddy"


def _summarize_event_counts(events: List[Dict]) -> Dict[str, int]:
    """Group hook events by event name → count, for session_meta."""
    out: Dict[str, int] = {}
    for e in events or ():
        name = str(e.get("event") or "")
        if not name:
            continue
        out[name] = out.get(name, 0) + 1
    return out


# ─── v0.19.6: 用户消息 <timestamp> 作为 turn 全覆盖时间窗 ───
#
# 背景：Cursor 压缩（compaction）会重写 transcript，晚近 turn 的 tool_use
# 块被剥掉后，这些 turn 一个带 observed_at_ms 的锚点 step 都不剩。事件
# 路由（_inject_agent_thoughts 的窗口兜底 + _attach_cursor_events 的
# _pick_turn_for）依赖锚点推断 turn 窗口——无锚 turn 对路由完全不可见，
# 后续所有事件塌陷进最后一个有锚的 turn（实测会话 83811387：今天 373
# 条事件全部倒进 turn 1，turn 4/5 只剩 user_input + final_response）。
# 用户消息正文里的 <timestamp>（IDE 注入）是压缩也拿不走的真实 turn
# 边界，用它构建 [start_i, start_{i+1}) 全覆盖窗口，优先于锚点窗口。

_USER_TS_RE = re.compile(
    r"<timestamp>\s*(?:[A-Za-z]+,\s*)?"
    r"([A-Za-z]{3})\w*\s+(\d{1,2}),\s*(\d{4}),\s*(\d{1,2}):(\d{2})\s*([AP]M)"
    r"\s*\(\s*UTC\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?\s*\)"
)

_TS_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_user_ts_ms(content: Optional[str]) -> Optional[int]:
    """解析 user_input 正文里的 <timestamp> 标签 → epoch ms（失败 None）。

    月份按静态表映射，不走 strptime %b——后者依赖系统 locale，中文
    Windows 上匹配不了英文月份缩写。
    """
    if not content:
        return None
    m = _USER_TS_RE.search(str(content))
    if not m:
        return None
    month = _TS_MONTHS.get(m.group(1).lower())
    if month is None:
        return None
    try:
        from datetime import datetime, timedelta, timezone
        hour = int(m.group(4)) % 12
        if m.group(6).upper() == "PM":
            hour += 12
        sign = 1 if m.group(7) == "+" else -1
        off_min = sign * (int(m.group(8)) * 60
                          + (int(m.group(9)) if m.group(9) else 0))
        dt = datetime(int(m.group(3)), month, int(m.group(2)),
                      hour, int(m.group(5)),
                      tzinfo=timezone(timedelta(minutes=off_min)))
        return int(dt.timestamp() * 1000)
    except (ValueError, OverflowError):
        return None


def _turn_user_start_ms(turn: "TurnCoT") -> Optional[int]:
    for s in turn.steps:
        if s.step_type == StepType.USER_INPUT:
            return _parse_user_ts_ms(s.content)
    return None


def _user_ts_turn_windows(
    turns_cot: List["TurnCoT"],
) -> Optional[List[Tuple[Optional[int], Optional[int], "TurnCoT"]]]:
    """[(lo_ms, hi_ms|None, turn)]：hi=None 表示 +∞。全无时间戳返回 None。

    窗口半开 [lo, hi)，相邻 turn 无缝衔接，任何事件都有唯一归属；无
    时间戳的 turn 窗口记 (None, None)，交回调用方的旧逻辑兜底。
    """
    starts = [_turn_user_start_ms(t) for t in turns_cot]
    if not any(v is not None for v in starts):
        return None
    windows: List[Tuple[Optional[int], Optional[int], "TurnCoT"]] = []
    n = len(turns_cot)
    for i, turn in enumerate(turns_cot):
        lo = starts[i]
        if lo is None:
            windows.append((None, None, turn))
            continue
        hi = None
        for j in range(i + 1, n):
            if starts[j] is not None:
                hi = starts[j]
                break
        windows.append((lo, hi, turn))
    return windows


def _match_user_ts_window(
    windows: Optional[List[Tuple[Optional[int], Optional[int], "TurnCoT"]]],
    t_ms: float,
) -> Optional["TurnCoT"]:
    """在全覆盖窗口里找 t_ms 的归属 turn；找不到返回 None。"""
    if not windows:
        return None
    for lo, hi, turn in windows:
        if lo is not None and lo <= t_ms and (hi is None or t_ms < hi):
            return turn
    return None


def _attach_cursor_events(
    turns_cot: List["TurnCoT"],
    events: List[Dict],
) -> Dict[str, int]:
    """把 events.jsonl 里真实的 shell stdout/stderr/exit_code 回灌到对应的
    tool_execution step。Cursor transcript 不回传 tool_result，CoT 里
    tool_execution 是 synthetic 占位（content 为空）——这一步用实时观测
    数据替换掉那些空占位，让前端看到**真实**的命令、返回码和输出内容。

    匹配策略（保守启发式，避免乱配）:
    - 仅处理 `after*` 事件 + 已知工具名（Shell / CallMcpTool / MCP）
    - 同一工具名下，按时间顺序 zip tool_execution 与 events 一一对齐
    - 未匹配的条目静默跳过；已注入的步骤把 metadata.synthetic 置 False
      并加 `synthetic_upgraded: True` 方便前端区分

    返回统计信息（命中 / 总 tool_execution / 总 events）。
    """
    stats = {
        "events_total": len(events),
        "injected": 0,
        "tool_executions_total": 0,
        "shell_events": 0,
        "mcp_events": 0,
        "file_edit_events": 0,
        "file_edit_injected": 0,
        # v0.13.x: 新增三条通道的统计
        "read_events": 0,           # beforeReadFile 总数（带 content）
        "read_injected": 0,         # 命中 ReadFile/Read 的 step 数
        "tool_result_events": 0,    # agentToolResult 总数
        "tool_result_injected": 0,  # 命中 Glob/Grep/Delete 等的 step 数
    }
    if not events or not turns_cot:
        return stats

    after_events = [
        e for e in events
        if str(e.get("event", "")).startswith("after")
        or e.get("event") in ("PostToolUse", "PostToolUseFailure")
    ]
    # v0.13.x：beforeReadFile 是唯一一个 hook 在 *before* 阶段就携带 result
    # （Cursor 会把读到的文件 content 整段塞进 payload）的事件，所以单独抽出来。
    before_read_events = [e for e in events
                          if e.get("event") == "beforeReadFile"]
    # v0.13.x: transcript_watcher / backfill_results 注入的"重放结果"事件
    tool_result_events = [e for e in events
                          if e.get("event") == "agentToolResult"]

    # 收集所有 tool_execution step
    tool_execs: List[ThoughtStep] = []
    for turn in turns_cot:
        for s in turn.steps:
            if s.step_type == StepType.TOOL_EXECUTION:
                tool_execs.append(s)
    stats["tool_executions_total"] = len(tool_execs)

    # 把 cot-stream.js 的 event.tool 映射到 transcript 的 tool_name
    def _event_to_tool_name(ev: Dict) -> Optional[str]:
        t = (ev.get("tool") or "").strip()
        name = str(ev.get("event") or "")
        if name in ("afterShellExecution", "beforeShellExecution"):
            return "Shell"
        if name in ("afterMCPExecution", "beforeMCPExecution"):
            return "CallMcpTool"
        if name in ("PostToolUse", "PostToolUseFailure", "PreToolUse"):
            if t in ("Bash", "Shell"):
                return "Shell"
            if t in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
                return t
            if t in ("Read", "ReadFile"):
                return "Read"
            return t or None
        # afterFileEdit / afterTabFileEdit 不进 tool 桶，下面单独走 file 通道
        if name in ("afterFileEdit", "afterTabFileEdit"):
            return None
        return t or None

    # ── 通道 1：Shell / MCP 按 tool_name 分桶（保持原行为）──
    buckets: Dict[str, List[Dict]] = {}
    file_edit_events: List[Dict] = []
    for e in after_events:
        ev_name = str(e.get("event") or "")
        if ev_name in ("afterFileEdit", "afterTabFileEdit"):
            file_edit_events.append(e)
            continue
        tool_name = _event_to_tool_name(e)
        if not tool_name:
            continue
        if tool_name == "Shell":
            stats["shell_events"] += 1
        elif tool_name == "CallMcpTool":
            stats["mcp_events"] += 1
        buckets.setdefault(tool_name, []).append(e)
    for ev_list in buckets.values():
        ev_list.sort(key=lambda e: e.get("t", 0))
    file_edit_events.sort(key=lambda e: e.get("t", 0))
    stats["file_edit_events"] = len(file_edit_events)

    # 对每个 tool，把同名 tool_execution 与 events 一一对齐
    by_tool_exec: Dict[str, List[ThoughtStep]] = {}
    for s in tool_execs:
        key = (s.tool_name or "").strip()
        if key in buckets:
            by_tool_exec.setdefault(key, []).append(s)
    for execs in by_tool_exec.values():
        execs.sort(key=lambda s: s.step_index)

    for tool_name, ev_list in buckets.items():
        execs = by_tool_exec.get(tool_name) or []
        if not execs:
            continue
        # 关键：events.jsonl 的时间窗从 cot-stream.js 挂上去那一刻才开始，
        # 所以它里面的 N 条事件对应的一定是整个 session **最后 N 条**同名
        # tool_execution —— 而不是最前 N 条。按尾对齐避免把早期正确步骤覆盖。
        n = len(ev_list)
        aligned_execs = execs[-n:] if n <= len(execs) else execs
        for s, e in zip(aligned_execs, ev_list):
            brief_in = e.get("brief_input") or {}
            brief_out = e.get("brief_output") or {}

            # 拼人类可读的 content，替换 synthetic 空占位
            parts: List[str] = []
            if brief_in.get("command"):
                parts.append(f"$ {brief_in['command']}")
                if brief_in.get("cwd"):
                    parts.append(f"(cwd: {brief_in['cwd']})")
            if "exit_code" in brief_out:
                parts.append(f"exit_code = {brief_out['exit_code']}")
            if "duration_ms" in brief_out:
                parts.append(f"duration = {brief_out['duration_ms']} ms")
            if brief_out.get("stdout"):
                parts.append("── stdout ──\n" + str(brief_out["stdout"]).rstrip())
            if brief_out.get("stderr"):
                parts.append("── stderr ──\n" + str(brief_out["stderr"]).rstrip())
            new_content = "\n".join(parts).strip()

            if new_content:
                s.content = new_content
            md = dict(s.metadata) if isinstance(s.metadata, dict) else {}
            md["observed_input"] = brief_in
            md["observed_output"] = brief_out
            md["observed_at_ms"] = e.get("t")
            md["observed_source"] = "cursor_events"
            # v0.13.x: 把 generation_id 显式提到 metadata 顶层，方便
            # _inject_agent_thoughts 用同一把锁定位 thought 所属的 turn。
            # brief_input 不带 gid（Cursor hook 把它放在 payload 里），
            # 所以这里直接从原始 event.payload 取。
            # v0.14.2: payload 在 cot-stream.js 里超 20480 char 时会被替换成
            # 字符串 ``"[truncated-total]"``（见 cot-stream.js MAX_TOTAL）。
            # 这里必须容错：不是 dict 就别调 .get，否则整条 _attach_cursor_events
            # 直接挂掉，间接吞掉 _inject_agent_thoughts / _attach_lifecycle_events。
            _payload = e.get("payload")
            payload_gid = _payload.get("generation_id") if isinstance(_payload, dict) else None
            if payload_gid:
                md["generation_id"] = payload_gid
            if md.get("synthetic"):
                md["synthetic"] = False
                md["synthetic_upgraded"] = True
            # exit_code 非 0 时，带上 is_error 让下游 error_recovery 检测可用
            if isinstance(brief_out.get("exit_code"), int) and brief_out["exit_code"] != 0:
                md["is_error"] = True
            s.metadata = md
            stats["injected"] += 1

    # ── 通道 2：afterFileEdit 按 file_path 分桶 → 注入对应文件写工具的 execution ──
    # 这一步让用户在前端能看到："第 N 步真的写出去了，写了几次、加了多少行"，
    # 配合 cot_script_tracker 的 Shell 反查就是完整的 L5 Execution Trace。
    if file_edit_events:
        def _norm(p: Any) -> str:
            s = str(p or "").replace("\\", "/").strip()
            if len(s) >= 2 and s[1] == ":":
                s = s[0].lower() + s[1:]
            return s

        edit_tool_names = ("Write", "StrReplace", "MultiEdit", "Edit")
        # 按 file_path 分桶（events 端）
        events_by_path: Dict[str, List[Dict]] = {}
        for e in file_edit_events:
            brief_in = e.get("brief_input") or {}
            _ep = e.get("payload")
            _ep = _ep if isinstance(_ep, dict) else {}
            fp = brief_in.get("file_path") or _ep.get("file_path")
            if not fp:
                continue
            events_by_path.setdefault(_norm(fp), []).append(e)

        # 按 file_path 分桶（execution 端：tool_execution + 它的 sibling tool_decision 拿 path）
        # 找 path 的来源是 sibling tool_decision.metadata.tool_input.path
        decisions_by_id: Dict[str, ThoughtStep] = {}
        for turn in turns_cot:
            for s in turn.steps:
                if s.step_type == StepType.TOOL_DECISION and (s.tool_name or "") in edit_tool_names:
                    if s.tool_use_id:
                        decisions_by_id[s.tool_use_id] = s

        execs_by_path: Dict[str, List[ThoughtStep]] = {}
        for s in tool_execs:
            if (s.tool_name or "") not in edit_tool_names:
                continue
            dec = decisions_by_id.get(s.tool_use_id or "")
            if not dec:
                continue
            ti = _safe_tool_input(dec)
            fp = ti.get("path") or ti.get("file_path")
            if not fp:
                continue
            execs_by_path.setdefault(_norm(fp), []).append(s)
        for execs in execs_by_path.values():
            execs.sort(key=lambda s: s.step_index)

        for path_key, ev_list in events_by_path.items():
            execs = execs_by_path.get(path_key) or []
            if not execs:
                continue
            ev_list.sort(key=lambda e: e.get("t", 0))
            n = len(ev_list)
            aligned = execs[-n:] if n <= len(execs) else execs
            for s, e in zip(aligned, ev_list):
                brief_in = e.get("brief_input") or {}
                md = dict(s.metadata) if isinstance(s.metadata, dict) else {}
                # observed_input：带 file_path / edits_count / added_lines / removed_lines / generation_id / model
                md["observed_input"] = brief_in
                md["observed_output"] = {
                    "edits_count": brief_in.get("edits_count"),
                    "added_lines": brief_in.get("added_lines"),
                    "removed_lines": brief_in.get("removed_lines"),
                    "file_path": brief_in.get("file_path"),
                    "generation_id": brief_in.get("generation_id"),
                    "model": brief_in.get("model"),
                }
                md["observed_at_ms"] = e.get("t")
                md["observed_source"] = "cursor_events_file"
                if md.get("synthetic"):
                    md["synthetic"] = False
                    md["synthetic_upgraded"] = True
                # 在 content 上贴一段人类可读摘要（如果原 content 还是空占位）
                if not (s.content or "").strip():
                    parts2: List[str] = []
                    if brief_in.get("file_path"):
                        parts2.append(f"📝 {brief_in['file_path']}")
                    if brief_in.get("edits_count") is not None:
                        parts2.append(f"edits = {brief_in['edits_count']}")
                    if brief_in.get("added_lines") is not None or brief_in.get("removed_lines") is not None:
                        parts2.append(
                            f"+{brief_in.get('added_lines', 0)} / -{brief_in.get('removed_lines', 0)} lines"
                        )
                    if brief_in.get("model"):
                        parts2.append(f"model = {brief_in['model']}")
                    if parts2:
                        s.content = "\n".join(parts2)
                s.metadata = md
                stats["injected"] += 1
                stats["file_edit_injected"] += 1

    # ── 通道 3：beforeReadFile.payload.content → 注入到 ReadFile/Read step ──
    # Cursor 的 beforeReadFile 是个特殊存在：它在 hook *before* 阶段就把
    # 文件全文塞进 payload.content。但 cot-stream.js 的早期版本没把这块往
    # events.jsonl 落，新版会落 brief_input.content。这里把它当成 result
    # 注入，让前端 Read 步骤不再显示 "no result"。
    stats["read_events"] = len(before_read_events)
    if before_read_events:
        def _norm_path(p: Any) -> str:
            t = str(p or "").replace("\\", "/").strip()
            if len(t) >= 2 and t[1] == ":":
                t = t[0].lower() + t[1:]
            return t

        # 仅匹配本身就是 ReadFile / Read 的 tool_execution
        read_tools = ("ReadFile", "Read")
        decisions_by_id_read: Dict[str, ThoughtStep] = {}
        for turn in turns_cot:
            for s in turn.steps:
                if s.step_type == StepType.TOOL_DECISION and (s.tool_name or "") in read_tools:
                    if s.tool_use_id:
                        decisions_by_id_read[s.tool_use_id] = s

        # events 端按 path 分桶
        read_by_path: Dict[str, List[Dict]] = {}
        for e in before_read_events:
            brief_in = e.get("brief_input") or {}
            payload = e.get("payload")
            payload = payload if isinstance(payload, dict) else {}
            fp = brief_in.get("path") or brief_in.get("file_path") \
                or payload.get("path") or payload.get("file_path")
            if not fp:
                continue
            read_by_path.setdefault(_norm_path(fp), []).append(e)

        # execution 端按 path 分桶（path 来自 sibling tool_decision）
        execs_by_read_path: Dict[str, List[ThoughtStep]] = {}
        for s in tool_execs:
            if (s.tool_name or "") not in read_tools:
                continue
            dec = decisions_by_id_read.get(s.tool_use_id or "")
            ti = _safe_tool_input(dec) if dec else {}
            fp = ti.get("path") or ti.get("file_path")
            if not fp:
                continue
            execs_by_read_path.setdefault(_norm_path(fp), []).append(s)

        for path_key, ev_list in read_by_path.items():
            execs = execs_by_read_path.get(path_key) or []
            if not execs:
                continue
            ev_list.sort(key=lambda e: e.get("t", 0))
            n = len(ev_list)
            aligned = execs[-n:] if n <= len(execs) else execs
            for s, e in zip(aligned, ev_list):
                payload = e.get("payload")
                payload = payload if isinstance(payload, dict) else {}
                brief_in = e.get("brief_input") or {}
                content = (payload.get("content") or brief_in.get("content")
                           or "")
                content = str(content)
                # 乱码兜底：hook 落盘(events.jsonl)发生编码事故时 content 里
                # 的 CJK 已被烤坏；beforeReadFile 事件恰好带着磁盘路径，直接
                # 从磁盘复读原文换回干净内容。重读失败/仍然乱码则保留原文。
                rescued = False
                if _looks_garbled(content):
                    roots_raw = (payload.get("workspace_roots")
                                 or payload.get("workspaceRoots")
                                 or e.get("workspace_roots")
                                 or e.get("workspaceRoots") or [])
                    roots = roots_raw if isinstance(roots_raw, list) else []
                    clean = _reread_file_utf8(
                        str(brief_in.get("path") or payload.get("path") or ""),
                        max(len(content), 1),
                        roots,
                    )
                    if clean is not None and not _looks_garbled(clean):
                        content = clean
                        rescued = True
                        stats["read_content_rescued"] = (
                            stats.get("read_content_rescued", 0) + 1
                        )
                # 大文件 content 会很长；预览取前 64 KB，全文还在 events.jsonl
                preview = content[:64 * 1024]
                md = dict(s.metadata) if isinstance(s.metadata, dict) else {}
                md["observed_input"] = brief_in
                md["observed_output"] = {
                    "content_chars": len(content),
                    "content_preview_chars": len(preview),
                    "path": brief_in.get("path") or payload.get("path"),
                }
                md["observed_at_ms"] = e.get("t")
                md["observed_source"] = "cursor_events_read"
                if rescued:
                    md["read_content_rescued"] = True
                if md.get("synthetic"):
                    md["synthetic"] = False
                    md["synthetic_upgraded"] = True
                if preview and (not (s.content or "").strip()
                                or _is_synthetic_placeholder(s.content)):
                    s.content = (
                        f"📖 {brief_in.get('path') or path_key} "
                        f"({len(content)} chars)\n"
                        f"── content (preview) ──\n{preview}"
                    )
                s.metadata = md
                stats["injected"] += 1
                stats["read_injected"] += 1

    # ── 通道 4：agentToolResult → 注入到 Glob / Grep / Delete 等 step ──
    # 这些事件由 transcript_watcher 的"确定性重放层"或者 backfill_results.py
    # 写入。它们带的是 reproduce() 出来的真实文件系统结果，不是 LLM 推理。
    #
    # 对齐策略：Cursor 的 transcript tool_use.id 实际为 None（与 Anthropic API
    # 直连不同），cot_extractor 给它合成了 `cursor:tN:sM:<tool>` 风格的 id，
    # 所以无法用 tool_use_id 直接 join。改成与通道 1 (Shell/MCP) 完全一致的
    # **按工具名分桶 + 时序 zip** 策略：
    #   - tool_result_events 按 tool 名分桶并按 t 升序
    #   - tool_execs 按 tool_name 分桶并按 step_index 升序
    #   - 每个桶里 zip 对齐；如果数量不一致取尾对齐（同 Shell/MCP 处理）
    stats["tool_result_events"] = len(tool_result_events)
    if tool_result_events:
        # 工具名归一：rg ≡ Grep（cot_extractor 内部把它们当不同 tool_name 看，
        # 但前端显示一致；这里我们也按各自原 name 分桶，因为 step.tool_name
        # 也是 'rg' 还是 'Grep' 取决于 transcript 怎么写的）
        result_buckets: Dict[str, List[Dict]] = {}
        for e in tool_result_events:
            tn = e.get("tool") or "(unknown)"
            result_buckets.setdefault(tn, []).append(e)
        for ev_list in result_buckets.values():
            ev_list.sort(key=lambda e: e.get("t", 0))

        exec_buckets: Dict[str, List[ThoughtStep]] = {}
        for s in tool_execs:
            tn = (s.tool_name or "").strip()
            if tn in result_buckets:
                exec_buckets.setdefault(tn, []).append(s)
        for execs in exec_buckets.values():
            execs.sort(key=lambda s: s.step_index)

        for tool_name, ev_list in result_buckets.items():
            execs = exec_buckets.get(tool_name) or []
            if not execs:
                continue
            n = len(ev_list)
            # 与通道 1 同样的尾对齐：events.jsonl 只在 watcher 启动后才记，
            # 所以 events 数 ≤ executions 数；少的那侧贴在末尾
            aligned = execs[-n:] if n <= len(execs) else execs
            for s, e in zip(aligned, ev_list):
                payload = e.get("payload")
                payload = payload if isinstance(payload, dict) else {}
                result = payload.get("result") or {}
                brief_out = e.get("brief_output") or {}
                md = dict(s.metadata) if isinstance(s.metadata, dict) else {}
                md["observed_input"] = (e.get("brief_input")
                                        or {"tool": tool_name})
                md["observed_output"] = brief_out
                # agentToolResult 的 ``t`` 是**重放时刻**（提取器/backfill
                # re-run Glob/Grep 的 wall-clock），不是工具真实执行时刻。
                # 写进 observed_at_ms 会污染 _inject_agent_thoughts 的时间
                # 锚——全部 thinking 被判成"早于第一个锚点"，扎堆排到 turn
                # 开头（实测 6b0c4dd8 turn1 的 17 条 thinking 全部前置）。
                # 只记 reproduced_at_ms 留档，不参与时间锚。
                md["reproduced_at_ms"] = e.get("t")
                md["observed_source"] = "reproduced"
                md["reproduction_elapsed_ms"] = brief_out.get("elapsed_ms")
                if md.get("synthetic"):
                    md["synthetic"] = False
                    md["synthetic_upgraded"] = True
                if not (s.content or "").strip() or _is_synthetic_placeholder(s.content):
                    s.content = _format_reproduced_result(
                        tool_name, result, brief_out
                    )
                s.metadata = md
                stats["injected"] += 1
                stats["tool_result_injected"] += 1

    # ── 通道 5（v0.18.7）：未消费的 cursor 风格 tool 事件 → 合成新 step ──
    #
    # 背景：Cursor v2.6+ 把 transcript 里的 ``tool_use`` block 整块移到 hook
    # event 流里（agent transcript 只剩 ``type:"text"`` 的 thinking）。结果
    # _build_turns_from_transcript 阶段产出的 ``turns_cot`` 里**没有任何**
    # ``TOOL_DECISION`` / ``TOOL_EXECUTION`` 骨架，于是上面通道 1 / 2 / 3 /
    # 4 都跑空（``tool_execs == []``），events.jsonl 里 N 个 Shell / Read /
    # Edit / MCP 事件**无处可注**，前端就只看得到 thinking、看不到 tool。
    #
    # 这里兜底：把任何"没有被注入到任何 step"的 cursor 风格 tool event
    # 直接合成成一个 ``ThoughtStep(step_type=TOOL_EXECUTION)``，按 ``t`` 时
    # 间戳分配到匹配的 turn，并按时间重排该 turn 的 steps，让 thinking 与
    # tool 行为按真实顺序交错。``observed_source`` 标 ``cursor_events_synthesised``
    # 方便前端区分"合成的"和"transcript 原生的"两类 step。
    #
    # 选择"after* + beforeReadFile"做合成的原因：after 阶段事件携带完整
    # input + output，beforeReadFile 是唯一一个 hook 在 before 就携带 result
    # 的特殊 event。其它 before* 事件信息不完整，跳过避免重复（同 (cmd,
    # cwd, generation_id) 的 before/after 已经在 after 一侧拿到了）。
    consumed_event_ts: set = set()
    for turn in turns_cot:
        for s in turn.steps:
            if isinstance(s.metadata, dict):
                t_marker = s.metadata.get("observed_at_ms")
                if isinstance(t_marker, (int, float)):
                    consumed_event_ts.add(int(t_marker))

    SYNTH_EVENT_NAMES = {
        "afterShellExecution",
        "afterMCPExecution", "afterMcpExecution",
        "afterFileEdit", "afterTabFileEdit",
        "beforeReadFile",
    }
    orphan_events: List[Dict] = []
    for e in events:
        name = str(e.get("event") or "")
        if name not in SYNTH_EVENT_NAMES:
            continue
        t_val = e.get("t")
        if isinstance(t_val, (int, float)) and int(t_val) in consumed_event_ts:
            continue
        orphan_events.append(e)

    if orphan_events and turns_cot:
        # v0.19.6：优先用用户消息 <timestamp> 的全覆盖窗口——压缩后晚近
        # turn 没有锚点 step，锚点窗口对它们完全失明；用户消息时间戳是
        # 压缩也拿不走的真实边界。命中直接返回，未命中再走锚点推断。
        user_ts_windows = _user_ts_turn_windows(turns_cot)
        # 用每个 turn 已有 step 的 observed_at_ms 推断 turn 的 wall-clock 窗口；
        # 用于把 orphan event 分配到正确的 turn（thinking 已经被 _inject_agent_thoughts
        # 注入过 observed_at_ms，所以多 turn 场景下窗口足够区分）。
        turn_windows: List[Tuple[Optional[int], Optional[int], "TurnCoT"]] = []
        for turn in turns_cot:
            ts_in_turn: List[int] = []
            for s in turn.steps:
                if isinstance(s.metadata, dict):
                    tv = s.metadata.get("observed_at_ms")
                    if isinstance(tv, (int, float)):
                        ts_in_turn.append(int(tv))
            if ts_in_turn:
                turn_windows.append((min(ts_in_turn), max(ts_in_turn), turn))
            else:
                turn_windows.append((None, None, turn))

        def _pick_turn_for(t_ms: Optional[int]) -> "TurnCoT":
            if t_ms is not None:
                hit = _match_user_ts_window(user_ts_windows, t_ms)
                if hit is not None:
                    return hit
            if t_ms is None or not turn_windows:
                return turns_cot[-1]
            # 1) 落在某个 turn 的精确窗口里
            for lo, hi, turn in turn_windows:
                if lo is not None and hi is not None and lo <= t_ms <= hi:
                    return turn
            # 2) 比所有 turn 都早 → 第一个 turn
            valid = [(lo, turn) for lo, hi, turn in turn_windows if lo is not None]
            if not valid:
                return turns_cot[0]
            valid.sort(key=lambda x: x[0])
            if t_ms < valid[0][0]:
                return valid[0][1]
            # 3) 落在 turn 之间或之后 → 最近的前一个 turn
            chosen = valid[0][1]
            for lo, turn in valid:
                if lo <= t_ms:
                    chosen = turn
            return chosen

        next_step_idx = 1 + max(
            (s.step_index for turn in turns_cot for s in turn.steps),
            default=0,
        )

        for e in sorted(orphan_events, key=lambda ev: ev.get("t", 0) or 0):
            name = str(e.get("event") or "")
            brief_in = e.get("brief_input") if isinstance(e.get("brief_input"), dict) else {}
            brief_out = e.get("brief_output") if isinstance(e.get("brief_output"), dict) else {}
            payload = e.get("payload") if isinstance(e.get("payload"), dict) else {}

            if name == "afterShellExecution":
                tool_name_syn = "Shell"
            elif name in ("afterMCPExecution", "afterMcpExecution"):
                tool_name_syn = "CallMcpTool"
            elif name in ("afterFileEdit", "afterTabFileEdit"):
                tool_name_syn = "Edit"
            elif name == "beforeReadFile":
                tool_name_syn = "Read"
            else:
                tool_name_syn = str(e.get("tool") or "Tool")

            # 拼一段人类可读 content（与通道 1/2/3 注入逻辑保持类似风格，
            # 让前端 SpanTree / DetailPanel 不需要额外 if 分支）。
            parts_syn: List[str] = []
            if tool_name_syn == "Shell":
                if brief_in.get("command"):
                    parts_syn.append(f"$ {brief_in['command']}")
                if brief_in.get("cwd"):
                    parts_syn.append(f"(cwd: {brief_in['cwd']})")
                if "exit_code" in brief_out:
                    parts_syn.append(f"exit_code = {brief_out['exit_code']}")
                if "duration_ms" in brief_out:
                    parts_syn.append(f"duration = {brief_out['duration_ms']} ms")
                if brief_out.get("stdout"):
                    parts_syn.append(
                        "── stdout ──\n" + str(brief_out["stdout"]).rstrip()
                    )
                if brief_out.get("stderr"):
                    parts_syn.append(
                        "── stderr ──\n" + str(brief_out["stderr"]).rstrip()
                    )
            elif tool_name_syn == "Read":
                fp_r = (brief_in.get("file_path") or brief_in.get("path")
                        or payload.get("file_path") or payload.get("path"))
                if fp_r:
                    parts_syn.append(f"📄 {fp_r}")
                content_r = brief_in.get("content") or payload.get("content")
                if isinstance(content_r, str) and content_r:
                    parts_syn.append(f"({content_r.count(chr(10)) + 1} lines)")
            elif tool_name_syn == "Edit":
                fp_e = brief_in.get("file_path") or payload.get("file_path")
                if fp_e:
                    parts_syn.append(f"📝 {fp_e}")
                if brief_in.get("edits_count") is not None:
                    parts_syn.append(f"edits = {brief_in['edits_count']}")
                if (brief_in.get("added_lines") is not None
                        or brief_in.get("removed_lines") is not None):
                    parts_syn.append(
                        f"+{brief_in.get('added_lines', 0)} / "
                        f"-{brief_in.get('removed_lines', 0)} lines"
                    )
                if brief_in.get("model"):
                    parts_syn.append(f"model = {brief_in['model']}")
            else:
                # MCP / 其它兜底：dump brief_in/out
                parts_syn.append(json.dumps(
                    {"input": brief_in, "output": brief_out},
                    ensure_ascii=False,
                )[:600])
            content_syn = "\n".join(p for p in parts_syn if p).strip() or f"({tool_name_syn})"

            t_ms_val = e.get("t") if isinstance(e.get("t"), (int, float)) else None
            target_turn = _pick_turn_for(int(t_ms_val) if t_ms_val is not None else None)

            md_syn: Dict[str, Any] = {
                "observed_input": brief_in,
                "observed_output": brief_out,
                "observed_at_ms": e.get("t"),
                "observed_source": "cursor_events_synthesised",
                "synthetic": False,
                "synthesised_from_events": True,
                "event": name,
            }
            if payload.get("generation_id"):
                md_syn["generation_id"] = payload["generation_id"]
            if payload.get("model"):
                md_syn["model"] = payload["model"]
            if isinstance(brief_out.get("exit_code"), int) and brief_out["exit_code"] != 0:
                md_syn["is_error"] = True

            target_turn.steps.append(ThoughtStep(
                step_index=next_step_idx,
                turn_index=target_turn.turn_index,
                step_type=StepType.TOOL_EXECUTION,
                content=content_syn,
                metadata=md_syn,
                tool_name=tool_name_syn,
                tool_use_id=f"{name}:{int(t_ms_val) if t_ms_val is not None else next_step_idx}",
            ))
            target_turn.tool_calls.append(tool_name_syn)
            target_turn.total_steps = len(target_turn.steps)
            next_step_idx += 1
            stats["injected"] += 1
            stats["synthesised_from_events"] = stats.get("synthesised_from_events", 0) + 1

        # 按 observed_at_ms 重排每个 turn 的 steps，让 thinking 与合成的 tool
        # 按真实 wall-clock 时间交错。无时间戳的 step 继承最近锚点的时间
        # （前向填充，队首向后取），保持 tool_decision→tool_result 这类相邻
        # 对不被拆散——直接给 0 会把它们全部沉到带锚步骤之前，打乱真实执行
        # 顺序。排序全程稳定，相等键保持原相对顺序。
        def _resort_turn_steps(turn: "TurnCoT") -> None:
            steps = turn.steps
            n = len(steps)
            if n <= 1:
                return
            raw: List[Optional[int]] = []
            for s in steps:
                md = s.metadata if isinstance(s.metadata, dict) else {}
                t_s = md.get("observed_at_ms")
                raw.append(int(t_s) if isinstance(t_s, (int, float)) else None)
            # 前向填充：无锚 step 继承前一个锚点时间（贴住上文语境）
            eff: List[Optional[int]] = list(raw)
            last: Optional[int] = None
            for i in range(n):
                if eff[i] is not None:
                    last = eff[i]
                else:
                    eff[i] = last
            # 队首无锚段：继承后一个锚点时间
            nxt: Optional[int] = None
            for i in range(n - 1, -1, -1):
                if raw[i] is not None:
                    nxt = raw[i]
                elif eff[i] is None:
                    eff[i] = nxt

            big = 10 ** 18

            def _key(item: Tuple[int, "ThoughtStep"]) -> Tuple[int, int]:
                i, s = item
                if s.step_type == StepType.USER_INPUT:
                    return (-big, i)  # 永远最前
                if s.step_type == StepType.FINAL_RESPONSE:
                    return (big, i)  # 永远最后
                return (eff[i] if eff[i] is not None else 0, i)

            turn.steps = [s for _, s in sorted(enumerate(steps), key=_key)]

        for turn in turns_cot:
            _resort_turn_steps(turn)

    return stats


_SYNTHETIC_PLACEHOLDER_HINTS = (
    "tool_result block",
    "未记录",
    "synthetic",
    "no result",
)


def _is_synthetic_placeholder(content: str) -> bool:
    """Return True if ``content`` looks like one of cot_extractor's synthetic
    "this tool's result was not recorded" stubs (rather than real captured
    output). Used to decide whether overwriting it with a reproduced result
    is safe.
    """
    if not content:
        return True
    snippet = content.strip()
    if not snippet:
        return True
    # Stubs are short by construction; never wipe a real multi-line output.
    if len(snippet) > 400:
        return False
    low = snippet.lower()
    return any(h in low for h in _SYNTHETIC_PLACEHOLDER_HINTS)


def _format_reproduced_result(tool: str, result: Dict, brief: Dict) -> str:
    """Pretty-print a reproduced tool result for display in step.content.

    Kept local to avoid leaking reproduction internals into the step model.
    """
    if not isinstance(result, dict):
        return str(result)
    if tool == "Glob":
        matches = result.get("matches") or []
        n = result.get("match_count", len(matches))
        head = "\n".join(f"  {m}" for m in matches[:30])
        tail = "" if n <= 30 else f"\n  …{n - 30} more"
        pat = result.get("pattern_normalized", "?")
        return f"🔍 Glob {pat} → {n} matches\n{head}{tail}"
    if tool in ("Grep", "rg"):
        matches = result.get("matches") or []
        n = result.get("match_count", len(matches))
        files = result.get("file_count", 0)
        head_lines: List[str] = []
        for m in matches[:20]:
            path = m.get("path", "?")
            ln = m.get("line_number", "?")
            txt = (m.get("line") or "").rstrip()[:200]
            head_lines.append(f"  {path}:{ln}: {txt}")
        tail = "" if n <= 20 else f"\n  …{n - 20} more"
        via = result.get("via", "?")
        return (f"🔎 Grep → {n} matches in {files} files (via {via})\n"
                + "\n".join(head_lines) + tail)
    if tool == "Delete":
        path = result.get("checked_path", "?")
        gone = not result.get("still_exists")
        flag = "✅ deleted" if gone else "⚠️ still exists"
        return f"🗑 Delete {path} → {flag}"
    # Generic fallback
    elapsed = brief.get("elapsed_ms")
    return f"{tool} → ok={brief.get('ok')} ({elapsed} ms)"


# ─────────────────────────────────────────────────────────────────────────────
# v0.13.x: 通道 5 —— afterAgentThought 注入（thinking_explicit 步骤）
#
# Cursor transcript jsonl 的 content blocks 只有 ``tool_use`` 和 ``text`` 两类，
# **没有** Anthropic API 那种 ``type:"thinking"`` block。所以 cot_extractor
# 原本是用 transcript 里的 text block 反推 thinking_inter（碎片化、零散）。
#
# 真正的"长篇思考流"（Cursor UI 里折叠的 *Thinking* / *Exploring* 面板，
# 单条经常上千字、单 turn 几十次）走的是另一条管道：``afterAgentThought``
# hook。它的 payload 里塞的是模型 reasoning 的完整原文。cot-stream.js 早就
# 把它落到 events.jsonl 了——只是 cot_extractor 一直没把它接到 cot.json，
# 导致前端 timeline 完全看不见这部分推理。
#
# 这个函数把每条 afterAgentThought 转成一个 ``thinking_explicit`` 步骤，
# **按时间插入到对应 turn 的正确位置**，让 dashboard 上能完整看到 agent
# "想 → 干 → 想 → 干"的真实节奏。
# ─────────────────────────────────────────────────────────────────────────────

def _inject_agent_thoughts(
    turns_cot: List["TurnCoT"],
    events: List[Dict],
) -> Dict[str, int]:
    """Insert ``thinking_explicit`` steps from ``afterAgentThought`` events.

    Strategy
    --------
    Step A: build ``gen_id -> turn_index`` from anchored ``tool_execution``
            steps' ``metadata.generation_id`` (populated by channel 1).
            This gives an exact, no-heuristic mapping for any turn that has
            at least one Shell/MCP/FileEdit hook event — i.e. any turn the
            agent actually executed real work in.

    Step B: for thoughts whose ``generation_id`` isn't on the gid map (turn
            had only Glob/Grep/Read or hooks didn't fire), fall back to a
            **time window** match: each turn's window is the [min, max] of
            its anchored steps' ``observed_at_ms``; the thought's ``t``
            falls into the unique containing turn.

    Step C: within the chosen turn, insert the new step right BEFORE the
            first existing step whose **effective anchor time** is greater
            than the thought's ``t``. Effective anchor time = the step's
            own ``observed_at_ms`` if present, otherwise inherited from
            the next anchored step (since unanchored steps like
            ``tool_decision`` precede their paired ``tool_execution``).

    Step D: re-index ``step_index`` globally and update each turn's
            ``thinking_depth`` / ``total_steps`` counters so downstream
            stats stay accurate.
    """
    stats = {
        "thought_events": 0,
        "thought_injected": 0,
        "thought_orphan": 0,
    }
    thoughts = [
        e for e in events
        if e.get("event") == "afterAgentThought" or e.get("thinking_probe")
    ]
    stats["thought_events"] = len(thoughts)
    if not thoughts or not turns_cot:
        return stats

    # ── Step A: 用 channel 1 注入的 generation_id 锚点构建 gid -> turn 映射 ──
    # 注意：channel 1 的尾对齐有时会让 1-2 个 step 的 generation_id 越过 turn
    # 边界（比如 turn N+1 的第一个 Shell 被错配到 turn N 的最后一个 Shell
    # tool_execution 上）。所以这里用**多数票**而不是"第一次出现"——既兼容
    # 偶发的边界泄漏，又对干净的 session 保持等价。
    from collections import Counter as _Counter
    gid_turn_votes: Dict[str, "_Counter"] = {}
    turn_window: Dict[int, Tuple[float, float]] = {}
    for turn in turns_cot:
        anchors: List[float] = []
        for s in turn.steps:
            md = s.metadata if isinstance(s.metadata, dict) else {}
            gid = md.get("generation_id")
            if gid:
                gid_turn_votes.setdefault(gid, _Counter())[turn.turn_index] += 1
            oms = md.get("observed_at_ms")
            if isinstance(oms, (int, float)):
                anchors.append(float(oms))
        if anchors:
            turn_window[turn.turn_index] = (min(anchors), max(anchors))
    gid_to_turn: Dict[str, int] = {
        gid: votes.most_common(1)[0][0]
        for gid, votes in gid_turn_votes.items()
    }

    # ── Step B: 把每条 thought 落到 turn ──
    thoughts_by_turn: Dict[int, List[Dict]] = {}
    # v0.19.6：用户消息 <timestamp> 全覆盖窗口——压缩后无锚 turn 的
    # thinking 全靠它归位（否则 thought_orphan 直接丢弃）。
    user_ts_windows = _user_ts_turn_windows(turns_cot)
    # 时间窗会扩 60s 以容忍 thinking 在第一次 tool 之前 / 最后一次之后的情况
    # （turn 第一句往往是纯思考、最后一句也常常是 wrap-up 思考）
    SLACK_S = 60.0
    for th in thoughts:
        gid = (th.get("payload") or {}).get("generation_id")
        ti = gid_to_turn.get(gid) if gid else None
        if ti is None and user_ts_windows:
            t = th.get("t")
            if isinstance(t, (int, float)):
                hit = _match_user_ts_window(user_ts_windows, float(t))
                if hit is not None:
                    ti = hit.turn_index
        if ti is None:
            t = th.get("t")
            if isinstance(t, (int, float)):
                t_s = float(t) / 1000.0
                # pick the unique turn whose [lo-slack, hi+slack] covers t
                hits: List[int] = []
                for tidx, (lo_ms, hi_ms) in turn_window.items():
                    lo = lo_ms / 1000.0 - SLACK_S
                    hi = hi_ms / 1000.0 + SLACK_S
                    if lo <= t_s <= hi:
                        hits.append(tidx)
                if len(hits) == 1:
                    ti = hits[0]
                elif len(hits) > 1:
                    # multiple turns overlap on slack: pick the one whose
                    # max anchor is closest to t (i.e. thought belongs to
                    # the most-recently-active turn at thought time)
                    ti = min(hits, key=lambda i: abs(
                        turn_window[i][1] / 1000.0 - t_s))
        if ti is None:
            stats["thought_orphan"] += 1
            continue
        thoughts_by_turn.setdefault(ti, []).append(th)

    if not thoughts_by_turn:
        return stats

    # ── Step C: 在 turn 内按时间插入 ──
    for turn in turns_cot:
        ths = thoughts_by_turn.get(turn.turn_index)
        if not ths:
            continue
        ths.sort(key=lambda e: e.get("t", 0))

        original = list(turn.steps)
        n_orig = len(original)
        # effective_t 反向扫一遍：unanchored step 继承下一个 anchored 的时间
        # (因为 tool_decision 紧挨 tool_execution；thinking_inter 紧挨下一个
        # 工具调用)
        eff_t: List[Optional[float]] = [None] * n_orig
        next_anchor: Optional[float] = None
        for i in range(n_orig - 1, -1, -1):
            md = original[i].metadata if isinstance(original[i].metadata, dict) else {}
            oms = md.get("observed_at_ms")
            if isinstance(oms, (int, float)):
                next_anchor = float(oms)
            eff_t[i] = next_anchor

        merged: List[ThoughtStep] = []
        th_iter = iter(ths)
        next_th: Optional[Dict] = next(th_iter, None)
        for i in range(n_orig):
            # user_input 永远是 turn 的第一个元素：thinking 注入绝不能排到
            # 用户提问之前。无锚点场景下（effective time 全部 None）while
            # 会把所有 thought 倒到 step 0 前面，形成"用户还没问就先想了
            # 一屏"的时序错乱。
            if original[i].step_type == StepType.USER_INPUT:
                merged.append(original[i])
                continue
            et = eff_t[i]
            # 把所有 t 早于本步骤 effective time 的 thought 插到本步骤前
            while next_th is not None and (et is None
                                           or float(next_th.get("t", 0)) < et):
                merged.append(_make_thinking_step(next_th, turn.turn_index))
                stats["thought_injected"] += 1
                next_th = next(th_iter, None)
            merged.append(original[i])
        # 剩余 thought 全部接到 turn 末尾（说明都在最后一个 anchor 之后）
        while next_th is not None:
            merged.append(_make_thinking_step(next_th, turn.turn_index))
            stats["thought_injected"] += 1
            next_th = next(th_iter, None)

        turn.steps = merged
        turn.total_steps = len(merged)
        # thinking_depth = 中间 + 显式两类思考之和
        # （StepType 类只定义了 THINKING_INTER / THINKING_EXPLICIT 两个常量；
        # 'thinking_intermediate' 字符串在 frontend StepType 联合类型里也有，
        # 写裸字符串避免 AttributeError 把整个 inject 流程吞掉。）
        _think_types = (
            StepType.THINKING_INTER,
            StepType.THINKING_EXPLICIT,
            "thinking_intermediate",
        )
        turn.thinking_depth = sum(
            1 for s in merged if s.step_type in _think_types
        )

    # ── Step D: 去重 ──
    # Cursor 把模型 reasoning 同时写进 hook（payload.text）**和**
    # transcript 的 text block，所以一条思考会同时变成相邻的 thinking_explicit
    # 和 thinking_inter，文本几乎一字不差（行尾换行差 1 字符）。前端会显示成
    # 视觉上的"重复行"，把这些 inter 直接清掉，只留 explicit（后者带 hook
    # 元数据：observed_at_ms / generation_id / duration_ms / model）。
    # 先清 hook 双写（explicit vs explicit），再清跨通道重复（inter vs
    # explicit）。顺序不能反：双写留下的两条 explicit 会让 inter 去重的
    # 比对集合里出现重复项，虽然结果一样但白跑一遍。
    double_written = _dedupe_double_written_thoughts(turns_cot)
    stats["thinking_double_written_deduped"] = double_written
    deduped = _dedupe_redundant_thinking_inter(turns_cot)
    stats["thinking_inter_deduped"] = deduped

    # ── Step E: 全局 step_index 重排 ──
    next_idx = 1
    for turn in turns_cot:
        for s in turn.steps:
            s.step_index = next_idx
            next_idx += 1

    return stats


def _norm_thought_text(s: Optional[str]) -> str:
    """归一化思考文本以便比对：折叠空白、首尾去空。"""
    if not s:
        return ""
    return " ".join(str(s).split()).strip()


_GARBLE_CJK_Q_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]\?")


def _ascii_skeleton(text: str) -> str:
    """把正文压成 ASCII 骨架，用于跨编码判重。

    Why: hook 落盘（events.jsonl）若发生编码事故（GBK→UTF-8 误读等），
    thinking 正文里的 CJK 会被烤成乱码，但 ASCII 字母/数字/结构通常原样
    存活；transcript 侧的 inter 文本始终是干净 UTF-8。剥掉非 ASCII 后
    两边骨架一致，即可认定同一条思考。
    """
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _looks_garbled(text: str) -> bool:
    """True 当文本疑似编码事故产物。

    信号：U+FFFD replacement char，或「CJK 字符后紧跟 ASCII ?」出现 ≥3 次
    （"—"→"鈥?"、"✅"→"鉁?" 这类 GBK 误读的典型残留；正常中文标点用的是
    全角 ？，不会触发）。宁严勿宽——只在疑似时触发兜底，避免误伤正常文本。
    """
    if not text:
        return False
    if "\ufffd" in text:
        return True
    return len(_GARBLE_CJK_Q_RE.findall(text)) >= 3


def _reread_file_utf8(path: str, max_chars: int,
                      roots: Optional[List[str]] = None) -> Optional[str]:
    """beforeReadFile 乱码兜底：从磁盘以 UTF-8 重读文件，换回干净内容。

    hook 落盘若发生编码事故，event content 里的 CJK 已被烤坏；但事件同时
    携带文件路径，磁盘原件仍是干净的。重读失败（文件不存在/已变更/读取出
    错）返回 None，调用方保留原文。截断上限与 hook 原文对齐，避免把整份
    文件灌进 trace。
    """
    if not path:
        return None
    p = Path(path)
    candidates = [p]
    if not p.is_absolute():
        for root in roots or []:
            candidates.append(Path(str(root)) / path)
    for cand in candidates:
        try:
            if not cand.is_file():
                continue
            with open(cand, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read(max_chars + 1)[:max_chars]
        except Exception:
            continue
    return None


# ─── v0.17.x: cursor afterAgentResponse → turn.usage 回填 ───
#
# 背景：Cursor transcript 里 assistant message 的 usage 字段是 **模型 SDK
# 写什么我们就有什么**。Anthropic SDK 会写完整的
# ``{input_tokens, output_tokens, cache_creation_input_tokens,
# cache_read_input_tokens}``；OpenAI / GPT-5 系列在 Cursor 里**完全不写**
# usage —— assistant.message 顶层只剩 ``content``。结果是 GPT 模型跑出来的
# turn.usage 全是 0，前端 DetailPanel "所在 Turn Token" 也就显示 0/0。
#
# Cursor 的 hook 反而把每一次 LLM 调用的真值 token 写进了 ``afterAgentResponse``
# payload（``input_tokens`` / ``output_tokens`` / ``cache_read_tokens`` /
# ``cache_write_tokens`` + ``model`` + ``generation_id``）。cot_otel_enricher
# 已经用这份数据做 OTel 视图，但**没有反向回填到 TurnCoT.usage**，所以前端
# DetailPanel / SpanTree token chip 看的全是 transcript 那份 0。
#
# 这个函数只做一件事：当 turn.usage 在 transcript 阶段是空的（全 0），就用
# events.jsonl 的 afterAgentResponse 真值回填；transcript 已经给了真值
# （Claude）的 turn 一律不动，避免重复计数。

def _attach_cursor_token_usage(
    turns_cot: List["TurnCoT"],
    events: List[Dict],
) -> Dict[str, int]:
    """把 cursor ``afterAgentResponse`` 事件里的 token usage 回填到 turn.usage。

    按时间窗匹配 turn：优先用 ``turn_start_ms_observed`` /
    ``turn_end_ms_observed`` 这对 hook 真值时间戳；缺失时 fallback 到
    ``turn_start_time`` (ISO) → ms。落不进任何窗的事件归到最后一个 turn
    （sessionEnd 之后的孤儿 hook 不可能出现，但仍兜底）。

    多次 ``afterAgentResponse`` 命中同一 turn 时：``output_tokens`` 累加，
    ``input_tokens`` 取 max（cumulative，与 transcript 原逻辑一致），
    cache 字段累加。

    Returns:
        诊断 stats，写入 ``observed_stats["token_usage_backfill"]``。
    """
    if not turns_cot or not events:
        return {"after_agent_response_events": 0, "turns_patched": 0}

    after_resps = [e for e in events if e.get("event") == "afterAgentResponse"]
    if not after_resps:
        return {"after_agent_response_events": 0, "turns_patched": 0}

    # 构 turn 时间窗
    def _window(t: "TurnCoT") -> Tuple[Optional[int], Optional[int]]:
        s = t.turn_start_ms_observed
        e_ms = t.turn_end_ms_observed
        if s is None and t.turn_start_time:
            s_f = _ts_to_ms(t.turn_start_time)
            s = int(s_f) if s_f is not None else None
        return s, e_ms

    windows: List[Tuple["TurnCoT", Optional[int], Optional[int]]] = []
    for idx, t in enumerate(turns_cot):
        s, e_ms = _window(t)
        if e_ms is None and idx + 1 < len(turns_cot):
            # 用下一个 turn 的开始时刻作为右边界
            ns, _ = _window(turns_cot[idx + 1])
            if ns is not None:
                e_ms = ns - 1
        windows.append((t, s, e_ms))

    def _has_real_tokens(usage: Optional[Dict]) -> bool:
        if not usage:
            return False
        return bool(int(usage.get("input_tokens") or 0) or int(usage.get("output_tokens") or 0))

    # 先按 turn 聚合，再一次性 commit，避免循环里反复改 usage
    pending: Dict[int, Dict[str, int]] = {}
    pending_models: Dict[int, str] = {}
    for ev in after_resps:
        t_ms = ev.get("t")
        if not isinstance(t_ms, (int, float)):
            continue
        payload = ev.get("payload")
        if not isinstance(payload, dict):
            continue
        in_tok = int(payload.get("input_tokens") or 0)
        out_tok = int(payload.get("output_tokens") or 0)
        cache_r = int(payload.get("cache_read_tokens") or 0)
        cache_w = int(payload.get("cache_write_tokens") or 0)
        if not (in_tok or out_tok or cache_r or cache_w):
            continue

        # 找窗
        target: Optional["TurnCoT"] = None
        for t, s, e_ms in windows:
            if s is None:
                continue
            if e_ms is None:
                if t_ms >= s:
                    target = t  # 最后一个开放区间
            elif s <= t_ms <= e_ms:
                target = t
                break
        if target is None:
            target = turns_cot[-1]

        agg = pending.setdefault(target.turn_index, {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        })
        agg["input_tokens"] = max(agg["input_tokens"], in_tok)
        agg["output_tokens"] += out_tok
        agg["cache_read_input_tokens"] += cache_r
        agg["cache_creation_input_tokens"] += cache_w
        m = payload.get("model")
        if isinstance(m, str) and m and target.turn_index not in pending_models:
            pending_models[target.turn_index] = m

    patched = 0
    skipped_existing = 0
    for t in turns_cot:
        agg = pending.get(t.turn_index)
        if not agg:
            continue
        if _has_real_tokens(t.usage):
            # transcript 已经给了真值（Claude）—— 不动
            skipped_existing += 1
            continue
        new_usage = dict(t.usage or {})
        for k, v in agg.items():
            new_usage[k] = int(new_usage.get(k) or 0) + int(v) if k != "input_tokens" else max(int(new_usage.get(k) or 0), int(v))
        # source 标记便于前端 / 调试区分
        new_usage["source"] = "afterAgentResponse"
        t.usage = new_usage
        patched += 1
        # 顺手把 model 写到 turn.otel.model（如果 otel 还没建）以便后续视图能看出 GPT
        m = pending_models.get(t.turn_index)
        if m and not t.otel:
            t.otel = {"model": m, "model_source": "afterAgentResponse"}

    return {
        "after_agent_response_events": len(after_resps),
        "turns_patched": patched,
        "turns_kept_transcript_truth": skipped_existing,
    }


def _recount_turn_steps(turn: "TurnCoT", keep: List["ThoughtStep"]) -> None:
    """删完步骤后同步 turn 上的计数，否则前端 KPI 会跟树对不上。"""
    turn.steps = keep
    turn.total_steps = len(keep)
    _think_types_local = (
        StepType.THINKING_INTER,
        StepType.THINKING_EXPLICIT,
        "thinking_intermediate",
    )
    turn.thinking_depth = sum(1 for s in keep if s.step_type in _think_types_local)


# 同一条 thinking 被 hook 推两次的时间间隔上限。实测中位数 163ms、最大 1.3s，
# 5s 留足余量；模型真的重复思考同一段话时间隔在分钟级，不会被误判。
_DUP_THOUGHT_WINDOW_MS = 5000

# 短于这个长度的思考正文不足以当身份标识，不参与文本判重。
_DUP_THOUGHT_MIN_CHARS = 24

# 整条正文就是一个方括号/尖括号占位符的形态，例如 ``[REDACTED]``。
_PLACEHOLDER_THOUGHT_RE = re.compile(r"^[\[\<\(][A-Za-z_\s\.…-]{2,30}[\]\>\)]$")


def _is_dedupe_safe_thought(norm: str) -> bool:
    """这段正文能不能拿来做「同一条思考」的判定依据。

    去重是按正文相同来判的，所以正文本身必须具备区分度。两类文本不具备：

    * 占位符 —— transcript 里被抹掉的思考统一写成 ``[REDACTED]``，一个 turn
      里能出现二十多条，它们是**不同**的思考，只是正文看不见了。按文本判重
      会把它们合成一条，等于凭空删掉真实步骤（实测一条会话里 48 组全是这种）。
    * 过短文本 —— 十几个字符的巧合重复概率不低，不值得为它冒删错的风险。

    宁可漏删几条重复，也不能删掉真实步骤：重复只是看着啰嗦，删错则是数据丢失。
    """
    if len(norm) < _DUP_THOUGHT_MIN_CHARS:
        return False
    return not _PLACEHOLDER_THOUGHT_RE.match(norm)


def _dedupe_double_written_thoughts(turns_cot: List["TurnCoT"]) -> int:
    """同一条 thinking 被 afterAgentThought hook 写了两次时只留一条。

    Why: Cursor 对同一段 reasoning 会推两次 hook —— 一次带裸的
    ``generation_id``（``<uuid>``），一次带按思考序号加了后缀的
    （``<uuid>-3-yn8q``）。两次的正文一字不差，只有 ``generation_id`` 和
    ``observed_at_ms`` 不同，于是前端 Thinking Phase 里每条思考都显示两遍，
    ``thinking_depth`` 也翻倍。

    判定条件是「同一 turn 内正文完全相同」且「时间差在
    ``_DUP_THOUGHT_WINDOW_MS`` 之内」，两个条件缺一不可：只看正文会误删
    模型真的把同一句话想了两遍的情况，只看时间会误删两条不同的短思考。

    被丢掉那条的 ``generation_id`` 记进保留条的
    ``metadata.duplicate_generation_ids``——hook 双写是上游行为，留个凭据
    方便日后回查，也免得看起来像我们凭空少了数据。
    """
    removed = 0
    for turn in turns_cot:
        steps = turn.steps
        if not steps:
            continue
        keep: List[ThoughtStep] = []
        # 归一化正文 → 已保留的那条 step，用于判重
        seen: Dict[str, ThoughtStep] = {}
        for s in steps:
            if s.step_type != StepType.THINKING_EXPLICIT:
                keep.append(s)
                continue
            norm = _norm_thought_text(s.content)
            if not norm or not _is_dedupe_safe_thought(norm):
                keep.append(s)
                continue
            prev = seen.get(norm)
            if prev is not None:
                t_prev = (prev.metadata or {}).get("observed_at_ms") or 0
                t_cur = (s.metadata or {}).get("observed_at_ms") or 0
                close = (
                    isinstance(t_prev, (int, float))
                    and isinstance(t_cur, (int, float))
                    and abs(int(t_cur) - int(t_prev)) <= _DUP_THOUGHT_WINDOW_MS
                )
                if close:
                    gid = (s.metadata or {}).get("generation_id")
                    if gid:
                        dupes = prev.metadata.setdefault("duplicate_generation_ids", [])
                        if gid not in dupes:
                            dupes.append(gid)
                    removed += 1
                    continue
            seen[norm] = s
            keep.append(s)
        if len(keep) != len(steps):
            _recount_turn_steps(turn, keep)
    return removed


def _dedupe_redundant_thinking_inter(turns_cot: List["TurnCoT"]) -> int:
    """删除跟同轮 thinking_explicit 文本几乎相同的 thinking_inter。

    Why: Cursor 同时把 reasoning 流写入两条独立通道：
    - afterAgentThought hook → events.jsonl → 通道 5 → thinking_explicit
    - transcript text block → extract_turn_cot → thinking_inter

    两边落盘的内容来自**同一**模型输出，几乎一字不差（差距通常是行尾的
    1–10 字符）。同时显示会让 timeline 出现视觉重复（参 a185f20b 实测
    19/120 条 explicit 都有 inter 副本）。

    去重规则：
    - 在**整个 turn 内**找文本几乎相同的 thinking_explicit（完全相等，或
      长度差 ≤10 且其中一者是另一者的前缀）；
    - 命中的 thinking_inter 整行删掉；
    - 保留 explicit 的原因：它带完整 hook 元数据（observed_at_ms /
      generation_id / model / duration_ms），下游 OTel / observability
      模块都依赖这些字段。

    早先这里只扫 ±2 个 step 的窗口，但两条通道的注入位置由各自的时间戳
    决定，实测同一条思考的 inter 与 explicit 常常相隔几十甚至两百多个
    step（一条 cursor 会话里 782 组重复因此全部漏掉）。既然判定依据是
    「正文几乎相同」而不是「挨得近」，窗口就没有存在意义。

    乱码兜底（ASCII 骨架判重）：hook 落盘若发生编码事故，explicit 正文
    里的 CJK 被烤成乱码，norm/前缀规则全部失效。此时改用 ASCII 骨架
    （剥掉非 ASCII 后的字母数字串，≥24 字符才参与）判同一条思考；命中
    且 explicit 疑似乱码、inter 干净时，用 inter 文本升级 explicit 正文
    ——metadata（generation_id / observed_at_ms / model）留在 explicit
    上不动，只是正文换成干净副本。
    """
    removed = 0
    upgraded = 0
    for turn in turns_cot:
        steps = turn.steps
        if not steps:
            continue
        # (归一化正文, ascii 骨架, 原 step)；乱码 explicit 的 norm 与干净
        # inter 对不上，但 ASCII 骨架通常对得上。
        exp_entries: List[Tuple[str, str, "ThoughtStep"]] = []
        for s in steps:
            if s.step_type != StepType.THINKING_EXPLICIT:
                continue
            n = _norm_thought_text(s.content)
            if n and _is_dedupe_safe_thought(n):
                exp_entries.append((n, _ascii_skeleton(n), s))
        if not exp_entries:
            continue
        exp_exact = {n for n, _, _ in exp_entries}
        exp_skeletons = {sk for _, sk, _ in exp_entries if len(sk) >= 24}
        keep: List[ThoughtStep] = []
        for s in steps:
            if s.step_type != StepType.THINKING_INTER:
                keep.append(s)
                continue
            inter_norm = _norm_thought_text(s.content)
            if not inter_norm or not _is_dedupe_safe_thought(inter_norm):
                keep.append(s)
                continue
            if inter_norm in exp_exact:
                removed += 1
                continue
            # 容忍尾部省略号 / 标点差异：长度差 ≤10 且前缀对得上
            matched = False
            for exp_norm, _, _ in exp_entries:
                if abs(len(exp_norm) - len(inter_norm)) > 10:
                    continue
                short, long = sorted([exp_norm, inter_norm], key=len)
                head_len = max(0, len(short) - 5)
                if head_len > 0 and long.startswith(short[:head_len]):
                    matched = True
                    break
            if not matched:
                # 乱码兜底：hook 侧正文被编码事故烤坏时 norm/前缀规则全部
                # 失效，改用 ASCII 骨架判同一条思考。先精确命中；否则放宽到
                # 「一方骨架是另一方的子串」——乱码对 CJK 前导语的破坏不
                # 对称（一边的 ASCII 幸存片段另一边没有），严格相等会漏。
                # ≥24 字符地板 + 同 turn 作用域保持保守。命中后若 explicit
                # 疑似乱码、inter 干净，用 inter 的干净文本升级 explicit
                # 正文——metadata（generation_id / observed_at_ms / model）
                # 留在 explicit 上不动。
                sk = _ascii_skeleton(inter_norm)
                matched_sk: Optional[str] = None
                if len(sk) >= 24:
                    if sk in exp_skeletons:
                        matched_sk = sk
                    else:
                        for _, exp_sk, _ in exp_entries:
                            if (len(exp_sk) >= 24
                                    and (sk in exp_sk or exp_sk in sk)):
                                matched_sk = exp_sk
                                break
                if matched_sk is not None:
                    matched = True
                    for _, exp_sk, exp_step in exp_entries:
                        if exp_sk == matched_sk:
                            if (_looks_garbled(exp_step.content)
                                    and not _looks_garbled(s.content)):
                                exp_step.content = s.content
                                upgraded += 1
                            break
            if matched:
                removed += 1
                continue
            keep.append(s)
        if len(keep) != len(steps):
            _recount_turn_steps(turn, keep)
    return removed


def _make_thinking_step(thought_event: Dict, turn_index: int) -> "ThoughtStep":
    """Convert one ``afterAgentThought`` event to a ``thinking_explicit``
    step. Step index is set to 0 here and rewritten by the caller after
    all insertions are done.
    """
    payload = thought_event.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    brief_out = thought_event.get("brief_output") or {}
    text = str(
        thought_event.get("thinking_probe")
        or payload.get("text")
        or payload.get("thinking")
        or payload.get("thought")
        or payload.get("reasoning")
        or ""
    )
    duration_ms = brief_out.get("duration_ms")
    source = "afterAgentThought" if thought_event.get("event") == "afterAgentThought" else "thinking_probe"
    return ThoughtStep(
        step_index=0,
        turn_index=turn_index,
        step_type=StepType.THINKING_EXPLICIT,
        content=text,
        tool_name="",
        tool_use_id="",
        metadata={
            "observed_at_ms": thought_event.get("t"),
            "observed_source": source,
            "observed_event": thought_event.get("event"),
            "generation_id": payload.get("generation_id"),
            "model": payload.get("model"),
            "synthetic": False,
            # tokens hint — Cursor doesn't always report it on this hook,
            # so keep it nullable; the OTel enricher will try to fill in
            # later if a sibling response carries token usage.
            "thought_chars": len(text),
            "thought_duration_ms": duration_ms,
        },
        timestamp=None,
        duration_ms=float(duration_ms) if isinstance(duration_ms, (int, float)) else None,
        tokens=0,
    )


def _todo_key(t: Dict) -> str:
    """Todo 的稳定 key：优先 id，没有就用 content 归一化。"""
    tid = str(t.get("id") or "").strip()
    if tid:
        return f"id::{tid}"
    return f"c::{str(t.get('content') or '').strip()}"


def _diff_todos(
    prev: Optional[List[Dict]],
    curr: List[Dict],
    *,
    merge_mode: bool = False,
) -> Dict:
    """两次 TodoWrite 之间的 status 迁移。

    返回:
        {
          "newly_completed": [{id, content}, ...],   # pending/in_progress → completed
          "newly_started":   [{id, content}, ...],   # pending/<missing>  → in_progress
          "newly_added":     [{id, content, status}],
          "removed":         [{id, content}],
          "status_changes":  [{id, content, from, to}],   # 兜底其他迁移
        }

    v0.14.1：
    - ``curr`` 项目缺 ``content`` 时，回退到同 id 的 ``prev`` 的 content。
      因为 ``TodoWrite(merge=True)`` 只传 ``{id, status}``，不带 content，
      之前直接用 ``t.get('content')`` 会写出一堆 ``content: null`` 的 diff
      记录，前端渲染就只剩个空壳。
    - ``merge_mode=True`` 时，``curr`` 不在的 prev 项不算 ``removed``——
      因为 merge 模式只是局部更新，没在 curr 出现的 prev 项应该被保留。
    """
    diff = {
        "newly_completed": [],
        "newly_started": [],
        "newly_added": [],
        "removed": [],
        "status_changes": [],
    }
    prev_map: Dict[str, Dict] = {}
    if isinstance(prev, list):
        for t in prev:
            if isinstance(t, dict):
                prev_map[_todo_key(t)] = t

    def _content_for(t: Dict, prev_t: Optional[Dict]) -> Any:
        """优先用 curr 的 content，缺了用 prev 的 content。"""
        c = t.get("content")
        if isinstance(c, str) and c.strip():
            return c
        if prev_t is not None:
            pc = prev_t.get("content")
            if isinstance(pc, str) and pc.strip():
                return pc
        return c  # 还是返回原值（可能是 None / 空串）

    curr_keys = set()
    for t in curr:
        if not isinstance(t, dict):
            continue
        k = _todo_key(t)
        curr_keys.add(k)
        cs = str(t.get("status") or "").lower()
        prev_t = prev_map.get(k)
        content_eff = _content_for(t, prev_t)
        if prev_t is None:
            diff["newly_added"].append({
                "id": t.get("id"), "content": content_eff, "status": cs,
            })
            if cs == "in_progress":
                diff["newly_started"].append({"id": t.get("id"), "content": content_eff})
            elif cs == "completed":
                diff["newly_completed"].append({"id": t.get("id"), "content": content_eff})
            continue
        ps = str(prev_t.get("status") or "").lower()
        # merge 模式下 status 字段可能没传 → 视为「保持原状态」
        if not cs and merge_mode:
            cs = ps
        if ps == cs:
            continue
        if cs == "completed" and ps != "completed":
            diff["newly_completed"].append({"id": t.get("id"), "content": content_eff})
        elif cs == "in_progress" and ps != "in_progress":
            diff["newly_started"].append({"id": t.get("id"), "content": content_eff})
        else:
            diff["status_changes"].append({
                "id": t.get("id"), "content": content_eff, "from": ps, "to": cs,
            })

    if not merge_mode:
        for k, prev_t in prev_map.items():
            if k not in curr_keys:
                diff["removed"].append({
                    "id": prev_t.get("id"),
                    "content": prev_t.get("content"),
                })
    return diff


def _resolve_effective_todos(
    prev_full: Optional[List[Dict]],
    curr_input: List[Dict],
    *,
    merge_mode: bool,
) -> List[Dict]:
    """把一次 TodoWrite 调用解析成「完整有效快照」。

    Cursor 的 TodoWrite 有两种模式：
    - ``merge=False`` → curr_input 就是新的完整列表，**整体覆盖** 旧 plan。
    - ``merge=True``  → curr_input 只是 ``[{id, status}, ...]`` 的局部状态
       更新（content 缺省！），需要用 prev_full 做底再 patch。

    旧版本（v0.10.0~v0.14.0）默认 curr_input 总是完整列表，导致
    ``merge=True`` 后所有快照丢内容、前端 plan 弹层一片空。

    返回的列表条目形如：
        ``{"id", "content", "status", "idx"}``，``idx`` 反映在最终列表里
    的顺序，方便前端按原始 plan 顺序展示。
    """
    def _norm(item: Dict, fallback: Optional[Dict] = None) -> Optional[Dict]:
        if not isinstance(item, dict):
            return None
        cid = item.get("id")
        # content：优先 curr，回退 prev
        content = item.get("content")
        if not (isinstance(content, str) and content.strip()) and fallback is not None:
            content = fallback.get("content")
        content = (content or "")
        if isinstance(content, str):
            content = content.strip()
        if not content:
            return None
        # status：优先 curr，缺了回退 prev
        status_raw = item.get("status")
        if not (isinstance(status_raw, str) and status_raw.strip()) and fallback is not None:
            status_raw = fallback.get("status")
        status = str(status_raw or "").lower()
        return {"id": cid, "content": content, "status": status}

    if not merge_mode:
        out: List[Dict] = []
        for idx, t in enumerate(curr_input):
            n = _norm(t)
            if n is None:
                continue
            n["idx"] = idx
            out.append(n)
        return out

    # ---- merge mode ----
    # 1) 用 prev_full 建 id → 项 的映射（保持原顺序）
    prev_list: List[Dict] = []
    prev_by_id: Dict[Any, Dict] = {}
    seen_ids = set()
    if isinstance(prev_full, list):
        for t in prev_full:
            if not isinstance(t, dict):
                continue
            cid = t.get("id")
            entry = {
                "id": cid,
                "content": t.get("content"),
                "status": t.get("status"),
            }
            prev_list.append(entry)
            if cid is not None and cid not in seen_ids:
                prev_by_id[cid] = entry
                seen_ids.add(cid)

    # 2) 把 curr 里的更新 patch 到对应 prev 项；新 id 视作「新增」放到末尾
    new_items: List[Dict] = []
    patched_ids = set()
    for t in curr_input:
        if not isinstance(t, dict):
            continue
        cid = t.get("id")
        target = prev_by_id.get(cid) if cid is not None else None
        if target is not None:
            # patch in place
            new_status = t.get("status")
            if isinstance(new_status, str) and new_status.strip():
                target["status"] = new_status
            new_content = t.get("content")
            if isinstance(new_content, str) and new_content.strip():
                target["content"] = new_content
            patched_ids.add(cid)
        else:
            new_items.append(t)

    # 3) 输出：按 prev 原顺序 + 新增项追加在末尾
    out_seq: List[Dict] = []
    for entry in prev_list:
        n = _norm(entry)
        if n is not None:
            out_seq.append(n)
    for item in new_items:
        n = _norm(item)
        if n is not None:
            out_seq.append(n)

    for idx, item in enumerate(out_seq):
        item["idx"] = idx
    return out_seq


def _build_plan_timeline(turns_cot: List["TurnCoT"]) -> List[PlanSnapshot]:
    """从所有 turn 的 TodoWrite 调用里抽取 plan 演进快照。

    TodoWrite 工具的 `todos` 参数直接包含当前完整的任务清单（带 status），
    多次调用就是 agent 的 plan 演进轨迹——这是 L4 计划层最直接的数据源。

    v0.10.0 增量：
    - PlanSnapshot 同时保留完整顺序 todos 列表 + 与上一条快照的 diff
    - 在 source step.metadata 里写 ``plan_snapshot_idx`` / ``plan_diff``，前端
      渲染 TodoWrite step 时不用回头去 timeline 里查

    v0.14.1 修复：
    - ``TodoWrite(merge=True)`` 只发 ``[{id, status}, ...]``（无 content）
      时，把 curr 当作 patch 而不是完整列表，与 prev 快照合并出真正的
      完整 todos，避免前端弹层里 完成/启动/新增/删除 全空。
    - prev_todos_raw 改为存「合并后」的快照而不是原始 ``tool_input``，
      这样下一次再来 merge 时也能正确累积。
    """
    timeline: List[PlanSnapshot] = []
    prev_todos_full: Optional[List[Dict]] = None
    snap_idx = 0
    # 每个 turn 独立的 plan 序号（前端展示的 "#N"）——用户明确要求：新的
    # 用户会话（= 新 turn）里的 plan 编号应该从 1 起，不应把全 session 内已
    # 出现过的 plan 数量继续叠加。跨 turn 的连续性由内部 tasks/todos 列表
    # 自己 tracking，跟前端展示编号解耦。
    per_turn_idx = 0
    prev_turn_index: Optional[int] = None

    for turn in turns_cot:
        if prev_turn_index is None or turn.turn_index != prev_turn_index:
            per_turn_idx = 0
            prev_turn_index = turn.turn_index
        for s in turn.steps:
            if s.step_type != StepType.TOOL_DECISION:
                continue
            if (s.tool_name or "") != "TodoWrite":
                continue
            tool_input = _safe_tool_input(s)
            todos = tool_input.get("todos")
            if not isinstance(todos, list):
                continue
            merge_mode = bool(tool_input.get("merge"))

            # 解析出真正的完整快照（merge=True 会和 prev 合并）
            full_todos = _resolve_effective_todos(
                prev_todos_full, todos, merge_mode=merge_mode,
            )
            if not full_todos:
                # 既没 prev 又没 content，跳过这次调用，免得污染 timeline
                continue

            snap = PlanSnapshot(
                at_step=s.step_index,
                turn_index=turn.turn_index,
                timestamp=s.timestamp,
                total=len(full_todos),
                snapshot_index=snap_idx,
            )
            for entry in full_todos:
                content = entry["content"]
                status = entry.get("status") or ""
                if status == "in_progress":
                    snap.in_progress.append(content)
                elif status == "completed":
                    snap.completed.append(content)
                elif status == "cancelled":
                    snap.cancelled.append(content)
                else:
                    snap.pending.append(content)
            snap.todos = full_todos
            # diff 用「合并后的完整快照」对比上一次「合并后的完整快照」
            snap.diff = _diff_todos(
                prev_todos_full, full_todos, merge_mode=merge_mode,
            )

            # 把 snapshot 索引 + diff + 完整 todos 同步写到 source step 的 metadata 上。
            # plan_full_todos 是「解析后」的完整列表（merge=True 时，上一次快照的
            # content 会被搬过来），前端不再依赖原始 tool_input.todos —— 那对
            # merge=True 调用而言只有 ``[{id, status}, ...]``，缺 content，渲染
            # 出来「状态/内容/id」表格里"内容"列就是空的。
            md = s.metadata if isinstance(s.metadata, dict) else {}
            md["plan_snapshot_idx"] = per_turn_idx
            md["plan_diff"] = snap.diff
            md["plan_total"] = len(full_todos)
            md["plan_completed_count"] = len(snap.completed)
            md["plan_in_progress_count"] = len(snap.in_progress)
            md["plan_pending_count"] = len(snap.pending)
            md["plan_merge_mode"] = merge_mode
            md["plan_full_todos"] = full_todos
            s.metadata = md

            timeline.append(snap)
            prev_todos_full = full_todos
            snap_idx += 1
            per_turn_idx += 1

    # v0.19.5: Claude Internal 用 TaskCreate / TaskUpdate 维护任务清单，跟
    # TodoWrite 是不同的两套 tool（不能强行重命名归一，因为语义不同——TodoWrite
    # 一次性覆盖整个列表，TaskCreate/TaskUpdate 是增量 add/patch）。如果上面
    # 没扫到任何 TodoWrite snapshot，就再用 Task* 这套抽一遍 plan timeline。
    if not timeline:
        timeline = _build_plan_timeline_from_task_tools(turns_cot)

    # v0.14.3: 末态滞后推断 + 把推断结果回灌到对应 step.metadata
    _reconcile_plan_timeline(turns_cot, timeline)
    return timeline


def _build_plan_timeline_from_task_tools(
    turns_cot: List["TurnCoT"],
) -> List[PlanSnapshot]:
    """Claude Internal 的 plan timeline——基于 ``TaskCreate`` / ``TaskUpdate``。

    Claude Internal（Sonnet / Opus internal）不使用 ``TodoWrite``，而是用一对
    工具维护任务清单：

    - ``TaskCreate {subject, description, activeForm}`` —— 追加一条新任务。
      task_id **不在 tool_input 里**，由 IDE 隐式地按创建顺序从 ``"1"`` 起编。
    - ``TaskUpdate {taskId, status}``                    —— 更新已存在任务的状态。

    把这两类调用按时间顺序回放就能得到完整的 plan 演进；每发生一次调用就发
    一个 PlanSnapshot，最终形态与 TodoWrite 路径完全对齐（前端 Plan 卡片读的
    是相同的 ``PlanSnapshot.todos`` / ``diff`` / metadata 回灌字段，所以不需要
    动前端）。

    设计要点：

    - 只在 TodoWrite-based timeline 为空时调用（见 ``_build_plan_timeline``
      尾部），避免对真正用 TodoWrite 的会话产生干扰。
    - 不重命名 ``TaskCreate`` / ``TaskUpdate`` 为 ``TodoWrite``：用户明确要求
      "不一样的 tool 不要统一"，这两套工具语义本来就不同。
    - 状态值规范化：``"completed"`` / ``"in_progress"`` / ``"cancelled"`` 三类
      已知；其余落回 ``pending``。
    """
    timeline: List[PlanSnapshot] = []
    # Claude IDE 在整个 session（含 subagent）里使用全局递增的 task id：
    # taskId="11" 指的是 "session 内第 11 个被 TaskCreate 创建的任务"，并不
    # 是 "当前 turn 第 11 个"。所以 tasks 清单**跨 turn 共享**；如果某次
    # TaskUpdate 引用的 taskId 在我们的 tasks 列表里找不到（subagent 的
    # 内部任务有时不会作为顶层 tool_use 出现），直接静默跳过即可。
    tasks: List[Dict] = []
    tasks_by_id: Dict[str, Dict] = {}
    snap_idx = 0
    # 每个 turn 独立的 plan 序号（前端展示 "#N"）；跨 turn 时重置到 0，避免
    # 一个 session 里 turn N 出现 "Plan #10"、turn N+1 出现 "Plan #11" 的
    # 累加错觉。内部 tasks 列表跨 turn 保留是必需的（TaskUpdate 要引用之前
    # TaskCreate 出的 id），跟展示编号无关。
    per_turn_idx = 0
    prev_turn_index: Optional[int] = None
    current_turn_task_ids: List[str] = []
    prev_todos_full: Optional[List[Dict]] = None

    for turn in turns_cot:
        if prev_turn_index is None or turn.turn_index != prev_turn_index:
            per_turn_idx = 0
            prev_turn_index = turn.turn_index
            current_turn_task_ids = []
            prev_todos_full = None
        for s in turn.steps:
            if s.step_type != StepType.TOOL_DECISION:
                continue
            tn = s.tool_name or ""
            if tn not in ("TaskCreate", "TaskUpdate"):
                continue
            ti = _safe_tool_input(s)

            if tn == "TaskCreate":
                new_id = str(len(tasks) + 1)
                content = ""
                for key in ("subject", "activeForm", "description"):
                    val = ti.get(key)
                    if isinstance(val, str) and val.strip():
                        content = val.strip()
                        break
                if not content:
                    continue
                task = {
                    "source_id": new_id,
                    "content": content,
                    "status": "pending",
                }
                tasks.append(task)
                tasks_by_id[new_id] = task
                current_turn_task_ids.append(new_id)
            else:  # TaskUpdate
                raw_id = ti.get("taskId")
                if raw_id is None:
                    continue
                target_id = str(raw_id)
                target = tasks_by_id.get(target_id)
                if target is None:
                    continue
                if target_id not in current_turn_task_ids:
                    current_turn_task_ids.append(target_id)
                new_status = ti.get("status")
                if isinstance(new_status, str) and new_status.strip():
                    target["status"] = new_status.strip().lower()
                for key in ("subject", "activeForm"):
                    val = ti.get(key)
                    if isinstance(val, str) and val.strip():
                        target["content"] = val.strip()
                        break

            full_todos: List[Dict] = [
                {
                    "id": str(idx + 1),
                    "source_id": source_id,
                    "content": t["content"],
                    "status": t.get("status", "pending"),
                    "idx": idx,
                }
                for idx, source_id in enumerate(current_turn_task_ids)
                for t in [tasks_by_id[source_id]]
            ]
            snap = PlanSnapshot(
                at_step=s.step_index,
                turn_index=turn.turn_index,
                timestamp=s.timestamp,
                total=len(full_todos),
                snapshot_index=snap_idx,
            )
            for entry in full_todos:
                content = entry["content"]
                status = (entry.get("status") or "").lower()
                if status == "in_progress":
                    snap.in_progress.append(content)
                elif status == "completed":
                    snap.completed.append(content)
                elif status == "cancelled":
                    snap.cancelled.append(content)
                else:
                    snap.pending.append(content)
            snap.todos = full_todos
            try:
                snap.diff = _diff_todos(prev_todos_full, full_todos, merge_mode=False)
            except Exception:
                snap.diff = None

            md = s.metadata if isinstance(s.metadata, dict) else {}
            raw_task_id = str(ti.get("taskId") or "")
            if tn == "TaskCreate":
                raw_task_id = new_id
            if raw_task_id:
                md["plan_source_task_id"] = raw_task_id
                if raw_task_id in current_turn_task_ids:
                    md["plan_display_task_id"] = str(current_turn_task_ids.index(raw_task_id) + 1)
            md["plan_snapshot_idx"] = per_turn_idx
            md["plan_diff"] = snap.diff
            md["plan_total"] = len(full_todos)
            md["plan_completed_count"] = len(snap.completed)
            md["plan_in_progress_count"] = len(snap.in_progress)
            md["plan_pending_count"] = len(snap.pending)
            md["plan_merge_mode"] = False
            md["plan_full_todos"] = full_todos
            s.metadata = md

            timeline.append(snap)
            prev_todos_full = full_todos
            snap_idx += 1
            per_turn_idx += 1

    return timeline


def _reconcile_plan_timeline(
    turns_cot: List["TurnCoT"],
    timeline: List[PlanSnapshot],
) -> None:
    """v0.14.3：给每个 turn 的最后一个 plan snapshot 做"滞后推断"。

    背景：agent 经常完成多个 todo 后只攒一次 TodoWrite 打勾。当用户在前端
    看最后一帧 plan 时，会出现 ``4/9 完成`` 但实际所有 9 项都做完了的错觉。
    后端没法严格判断"是不是真的做完了"，但可以用四个事实信号做软推断：

    阈值都偏保守，宁可不推断也不误推。

    Returns:
        就地 patch ``snap.is_likely_stale`` / ``snap.lag_steps_to_turn_end`` /
        ``snap.inferred_completed_ids`` / ``snap.stale_reason`` 与对应 step
        ``metadata.plan_inferred_completed`` / ``plan_is_likely_stale`` /
        ``plan_lag_steps``。
    """
    if not timeline or not turns_cot:
        return

    # 同 turn 的多条 snapshot 找出"最后一条"
    last_snap_per_turn: Dict[int, PlanSnapshot] = {}
    for snap in timeline:
        last_snap_per_turn[snap.turn_index] = snap  # 顺序遍历，最后一条胜出

    # 每个 turn 末尾是否出了 final_response（agent 已经收尾了）
    turns_by_index: Dict[int, "TurnCoT"] = {t.turn_index: t for t in turns_cot}
    final_signal_re = (
        "全部完成", "全部已完成", "都已完成", "已完成", "搞定", "完成所有",
        "all done", "completed", "all finished", "wrapped up", "finished all",
        "done", "✅", "🎉",
    )

    for turn_idx, snap in last_snap_per_turn.items():
        turn = turns_by_index.get(turn_idx)
        if turn is None:
            continue
        # 算"snapshot 之后这个 turn 还做了多少步、其中有多少不是 TodoWrite"
        post_steps = [s for s in turn.steps if s.step_index > snap.at_step]
        # 排除继续打勾的 TodoWrite / Claude TaskCreate / TaskUpdate —— 那些都是
        # 合法 plan 推进，不算"滞后"（v0.19.5：加 Task* 识别，避免 Claude
        # Internal 路径误触发 stale 推断）
        _plan_tools = {"TodoWrite", "TaskCreate", "TaskUpdate"}
        non_todo_post_steps = [
            s for s in post_steps
            if not (s.step_type == StepType.TOOL_DECISION
                    and (s.tool_name or "") in _plan_tools)
        ]
        snap.lag_steps_to_turn_end = len(non_todo_post_steps)

        # 判定 stale 的三种情况（任一即可）：
        # 1) 这个 turn 已经给出 final_response（agent 主动收尾了）
        # 2) snapshot 之后还干了 > 5 步实事但没再打勾
        # 3) final_response 文本含强信号词
        in_progress_ids = [t.get("id") for t in (snap.todos or [])
                           if (t.get("status") or "").lower() == "in_progress"]
        pending_ids = [t.get("id") for t in (snap.todos or [])
                       if (t.get("status") or "").lower() == "pending"]
        if not in_progress_ids and not pending_ids:
            # 全 completed/cancelled，没东西可推断
            continue

        reason: Optional[str] = None
        fr = (turn.final_response or "").lower()
        if fr and any(k.lower() in fr for k in final_signal_re):
            reason = "final_response_signal"
        elif (turn.final_response or "").strip():
            reason = "turn_finalized"
        elif snap.lag_steps_to_turn_end > 5:
            reason = "lag_too_many_steps"

        if reason is None:
            continue

        # 推断完成范围：
        # - lag_too_many_steps / turn_finalized：仅 in_progress（保守，不动 pending）
        # - final_response_signal：in_progress + pending（强信号，agent 自己宣告
        #   做完了，残留 pending 大概率也都做了 / 取消了）
        if reason == "final_response_signal":
            inferred = list(in_progress_ids) + list(pending_ids)
        else:
            inferred = list(in_progress_ids)

        # 没东西可补就别标"可能滞后"——保留 turn_finalized 但 inferred 为空的
        # 那种快照不画警告，避免前端"⚠ 滞后但又啥都没多算"看着困惑。
        # 这种情况一般是 agent 给出 final_response 但 plan 里还留着真正的 pending
        # 项目（明确没做的待办，不是"做了没打勾"），这是合法的 plan 末态。
        if not inferred:
            continue

        snap.is_likely_stale = True
        snap.stale_reason = reason
        snap.inferred_completed_ids = inferred

        # 回灌到 source step.metadata
        for s in turn.steps:
            if s.step_index != snap.at_step:
                continue
            md = s.metadata if isinstance(s.metadata, dict) else {}
            md["plan_is_likely_stale"] = True
            md["plan_stale_reason"] = reason
            md["plan_lag_steps"] = snap.lag_steps_to_turn_end
            md["plan_inferred_completed"] = list(snap.inferred_completed_ids)
            s.metadata = md
            break


def _build_mode_transitions(
    turns_cot: List["TurnCoT"],
) -> Tuple[List[ModeTransition], List[PlanProposal]]:
    """扫 ``SwitchMode`` / ``CreatePlan`` 工具调用，构建模式转换 + plan 文档时间线。

    工作流（Cursor agent 真实行为）：

    1. agent 主动调 ``SwitchMode(target_mode_id="plan")`` → 显式进入 plan 模式
    2. agent 调 ``CreatePlan(name, overview, plan)`` → 在 plan 模式里产出文档
    3. **用户在 UI 确认 plan** → 系统**隐式切回 agent 模式**（不会有 SwitchMode 调用）
    4. 下一条 user 消息开始，agent 进入"执行模式" → 推断为 ``implicit_back_to_agent``

    返回:
        (transitions, plan_proposals)
    """
    transitions: List[ModeTransition] = []
    proposals: List[PlanProposal] = []
    prev_mode: Optional[str] = None
    in_plan_mode = False
    pending_back_to_agent_at_turn: Optional[int] = None

    for turn in turns_cot:
        # 上一次 plan 完成后，第一条新 turn 推断为切回 agent
        if pending_back_to_agent_at_turn is not None and turn.turn_index >= pending_back_to_agent_at_turn:
            first_step = turn.steps[0] if turn.steps else None
            if first_step is not None and prev_mode == "plan":
                tr = ModeTransition(
                    at_step=first_step.step_index,
                    turn_index=turn.turn_index,
                    target_mode_id="agent",
                    explanation="用户确认 plan 后系统自动切回 agent 模式（推断）",
                    timestamp=first_step.timestamp,
                    prev_mode_id=prev_mode,
                    trigger="implicit_back_to_agent",
                )
                transitions.append(tr)
                prev_mode = "agent"
                in_plan_mode = False
            pending_back_to_agent_at_turn = None

        for s in turn.steps:
            if s.step_type != StepType.TOOL_DECISION:
                continue
            tool = (s.tool_name or "")

            if tool == "SwitchMode":
                tool_input = _safe_tool_input(s)
                target = str(tool_input.get("target_mode_id") or "").strip().lower()
                if not target:
                    continue
                tr = ModeTransition(
                    at_step=s.step_index,
                    turn_index=turn.turn_index,
                    target_mode_id=target,
                    explanation=tool_input.get("explanation") or None,
                    timestamp=s.timestamp,
                    prev_mode_id=prev_mode,
                    trigger="switch_mode",
                )
                transitions.append(tr)

                md = s.metadata if isinstance(s.metadata, dict) else {}
                md["mode_switch"] = {
                    "target_mode_id": target,
                    "prev_mode_id": prev_mode,
                    "explanation": tr.explanation,
                    "trigger": "switch_mode",
                }
                s.metadata = md

                prev_mode = target
                in_plan_mode = (target == "plan")

            elif tool == "CreatePlan":
                # 在 plan 模式下提交正式 plan 文档
                tool_input = _safe_tool_input(s)
                proposal = PlanProposal(
                    at_step=s.step_index,
                    turn_index=turn.turn_index,
                    name=str(tool_input.get("name") or "").strip(),
                    overview=str(tool_input.get("overview") or "").strip(),
                    plan=str(tool_input.get("plan") or "").strip(),
                    timestamp=s.timestamp,
                )
                proposals.append(proposal)

                md = s.metadata if isinstance(s.metadata, dict) else {}
                md["plan_proposal"] = {
                    "name": proposal.name,
                    "overview_preview": proposal.overview[:200],
                    "plan_chars": len(proposal.plan),
                }
                s.metadata = md

                # 一旦 CreatePlan 出现，就标记"plan 阶段结束、下一 turn 起隐式回 agent"
                if in_plan_mode or prev_mode == "plan":
                    pending_back_to_agent_at_turn = turn.turn_index + 1

    return transitions, proposals


def _propagate_invocation_to_executions(turns_cot: List["TurnCoT"]) -> None:
    """把 ``tool_decision.metadata.invocation_category`` 同步到对应的
    ``tool_execution`` step 上，并对 ``rag_query`` / ``web_search`` 类
    步骤抽召回片段（observed_output / content / inline tool_use_result）。

    必须在 ``_attach_cursor_events()`` 之后调用 —— 这样 ``observed_output``
    才已经回灌到 metadata 上，能拿到真实的 stdout 当召回内容。
    """
    if _classify_invocation is None and _extract_invocation_recall is None:
        return

    decision_meta_by_id: Dict[str, Dict] = {}
    for turn in turns_cot:
        for s in turn.steps:
            if s.step_type != StepType.TOOL_DECISION:
                continue
            cat = (s.metadata or {}).get("invocation_category")
            if cat:
                decision_meta_by_id[s.tool_use_id or ""] = s.metadata or {}

    if not decision_meta_by_id:
        return

    for turn in turns_cot:
        for s in turn.steps:
            if s.step_type != StepType.TOOL_EXECUTION:
                continue
            dec_meta = decision_meta_by_id.get(s.tool_use_id or "")
            if not dec_meta:
                continue
            cat = dec_meta.get("invocation_category")
            if not cat:
                continue
            md = dict(s.metadata) if isinstance(s.metadata, dict) else {}
            md["invocation_category"] = cat
            # 仅对 RAG 类抽召回（web_search 也走 RAG 视觉，召回内容也展示）
            if cat in ("rag_query", "web_search") and _extract_invocation_recall is not None:
                try:
                    recall = _extract_invocation_recall(
                        tool_input=dec_meta.get("tool_input"),
                        observed_output=md.get("observed_output"),
                        content=s.content,
                    )
                    if recall:
                        md["recall_preview"] = recall
                    elif _diagnose_recall_unavailable is not None:
                        # 抽不到就告诉前端为什么——避免用户看到一片空白还在猜
                        reason = _diagnose_recall_unavailable(
                            observed_output=md.get("observed_output"),
                            synthetic=bool(md.get("synthetic")),
                        )
                        if reason:
                            md["recall_unavailable_reason"] = reason
                except Exception:
                    pass
            # 把 decision 上的 prompt_preview / prompt_full_chars 同步过来
            # —— 让 execution step 在前端也能直接看到"这次调用送了什么 prompt"
            for k in ("prompt_preview", "prompt_full_chars"):
                if dec_meta.get(k) is not None and md.get(k) is None:
                    md[k] = dec_meta.get(k)
            # 把 decision 上的完整 tool_input 拷一份到 execution，方便前端
            # 在"返回"区直接对照"入参"
            if dec_meta.get("tool_input") is not None and md.get("decision_tool_input") is None:
                md["decision_tool_input"] = dec_meta.get("tool_input")
            s.metadata = md


# v0.14.5：相邻 step 间隔 > 这个阈值就视为"用户离开 IDE"的空闲，
# 从 active 时长里剔除。5 分钟够大，正常 agent 干活的 step 间隔
# 几秒到几十秒不等，几乎不会触发；但跨日留 IDE 的 ~16 小时 gap
# 会被精准排除。
_TURN_IDLE_GAP_THRESHOLD_MS = 5 * 60_000


def _compute_active_duration(
    obs_list_sorted: List[float],
) -> Tuple[float, float, float]:
    """从有序的 observed_at_ms 列表里算出三个时长指标。

    Returns
    -------
    (active_ms, wallclock_span_ms, idle_ms)
        active_ms       : 累加所有 < ``_TURN_IDLE_GAP_THRESHOLD_MS`` 的相邻间隔
        wallclock_span  : max - min（含 idle）
        idle_ms         : 被剔除的累计 idle 时长（>= threshold 的 gap 之和）
    """
    if len(obs_list_sorted) < 2:
        return 0.0, 0.0, 0.0
    wallclock = obs_list_sorted[-1] - obs_list_sorted[0]
    active = 0.0
    idle = 0.0
    for i in range(1, len(obs_list_sorted)):
        gap = obs_list_sorted[i] - obs_list_sorted[i - 1]
        if gap < _TURN_IDLE_GAP_THRESHOLD_MS:
            active += gap
        else:
            idle += gap
    return active, wallclock, idle


def _recompute_turn_durations(turns_cot: List["TurnCoT"]) -> None:
    """v0.14.4 / v0.14.5：用 step.metadata.observed_at_ms 给 turn 时间补救。

    背景：transcript 里 user_msg 和绝大多数 step 都没有 timestamp，导致
    extract_turn_cot 算出来的 turn_duration_ms / turn_start_time 几乎全是
    None。但 cot-stream.js 实时回灌的 ``observed_at_ms`` 是真值时间戳。
    在 _attach_cursor_events 跑完后扫一遍，对 turn_duration_ms 还是 None
    的 turn 用 step.observed_at_ms 兜底。

    v0.14.5 关键改动：把"max - min"改成"active 时长（剔除 >5min idle gap）"，
    因为用户经常下班 IDE 不关、第二天接着干同一个 turn，wall-clock 会爆出
    1183.9 min 这种荒谬数字。

    同时仍然记录 wall-clock 跨度到 ``turn_wallclock_span_ms``，前端走
    tooltip 展示，避免把"agent 真正干活时间"和"用户挂着 IDE 时间"混淆。

    注意：``turn_duration_ms_observed`` 是 _attach_lifecycle_events 用
    beforeSubmitPrompt + stop 算出来的"真·真值"，优先级最高，这里不动。
    """
    from datetime import datetime, timezone

    for turn in turns_cot:
        # 收集 step 上的 observed_at_ms（哪怕已经有 transcript-derived
        # turn_duration_ms 也算 wall-clock，让前端能显示"实际跨度"）
        obs_list: List[float] = []
        for s in turn.steps:
            md = s.metadata if isinstance(s.metadata, dict) else {}
            ms = md.get("observed_at_ms")
            if isinstance(ms, (int, float)) and ms > 0:
                obs_list.append(float(ms))
        if len(obs_list) < 2:
            continue

        obs_list.sort()
        active_ms, wallclock_ms, idle_ms = _compute_active_duration(obs_list)

        # wall-clock 跨度永远写
        turn.turn_wallclock_span_ms = round(wallclock_ms, 2)
        turn.turn_idle_ms = round(idle_ms, 2)

        # turn_duration_ms：transcript 已经给过就不动；否则用 active
        if turn.turn_duration_ms is None:
            if active_ms > 0:
                turn.turn_duration_ms = round(active_ms, 2)
            elif wallclock_ms > 0:
                # 只有 1 个 obs 点 / 全是 idle 的极端情况，回退给 wallclock
                turn.turn_duration_ms = round(wallclock_ms, 2)

        if not turn.turn_start_time:
            turn.turn_start_time = datetime.fromtimestamp(
                obs_list[0] / 1000.0, tz=timezone.utc
            ).isoformat()


def _attach_lifecycle_events(
    turns_cot: List["TurnCoT"],
    events: List[Dict],
) -> Tuple[Optional[Dict], List[Dict]]:
    """v0.14.2 通道 6：消费 6 个新订阅的 Cursor hook 事件，给 session/turn 打真值时间锚。

    新订阅的 hook 事件（v0.14.2 起 cot-stream 采集）：
    - ``sessionStart`` / ``sessionEnd``：整个 IDE 会话的真实起止 ms + cursor_version /
      user_email / workspace_roots（transcript 完全无此信息）。
    - ``beforeSubmitPrompt``：用户回车那一刻的真值 ms + 用户原始 prompt 字符数预览。
    - ``stop``：一轮对话被模型主动结束（end_turn）那一刻的真值 ms。
    - ``beforeTabFileRead`` / ``afterTabFileEdit``：用户在 IDE tab 里**手动**读 / 改文件，
      与 agent 的 Read/Edit 工具完全分开。后者尤其重要：用户在 agent 之后还得手动改一遍
      → 「agent 没干完活，用户自己补」的强信号，是"agent 质量"的天然 ground truth。

    Returns:
        ``(session_meta, user_activity)``，分别写到 ``SessionCoT.session_meta``
        / ``SessionCoT.user_activity``；同时**就地** patch ``TurnCoT.turn_start_ms_observed``
        / ``turn_end_ms_observed`` / ``turn_duration_ms_observed``。
    """
    if not events:
        return None, []

    # 按事件类型分桶（按 t 升序）
    def _by(name: str) -> List[Dict]:
        return sorted(
            (e for e in events if e.get("event") == name),
            key=lambda x: x.get("t", 0),
        )

    sess_start_evs = _by("sessionStart")
    sess_end_evs = _by("sessionEnd")
    submit_evs = _by("beforeSubmitPrompt")
    stop_evs = _by("stop")
    tab_read_evs = _by("beforeTabFileRead")
    tab_edit_evs = _by("afterTabFileEdit")

    if not (sess_start_evs or sess_end_evs or submit_evs or stop_evs
            or tab_read_evs or tab_edit_evs):
        return None, []

    # ---- session_meta ----
    session_meta: Dict = {
        "hook_events_observed": {
            "sessionStart": len(sess_start_evs),
            "sessionEnd": len(sess_end_evs),
            "beforeSubmitPrompt": len(submit_evs),
            "stop": len(stop_evs),
            "beforeTabFileRead": len(tab_read_evs),
            "afterTabFileEdit": len(tab_edit_evs),
        },
    }

    if sess_start_evs:
        first = sess_start_evs[0]
        bi = first.get("brief_input") or {}
        pl = first.get("payload")
        pl = pl if isinstance(pl, dict) else {}
        session_meta["session_start_ms_observed"] = int(first.get("t") or 0)
        for k in ("cursor_version", "user_email"):
            v = bi.get(k) or pl.get(k)
            if v:
                session_meta[k] = v
        roots = bi.get("workspace_roots") or pl.get("workspace_roots")
        if isinstance(roots, list) and roots:
            session_meta["workspace_roots"] = roots[:8]
        tp = pl.get("transcript_path")
        if tp:
            session_meta["transcript_path"] = tp

    if sess_end_evs:
        last = sess_end_evs[-1]
        session_meta["session_end_ms_observed"] = int(last.get("t") or 0)

    if "session_start_ms_observed" in session_meta and "session_end_ms_observed" in session_meta:
        dur = session_meta["session_end_ms_observed"] - session_meta["session_start_ms_observed"]
        if dur >= 0:
            session_meta["session_duration_ms_observed"] = dur

    # ---- per-turn 真值时间锚 ----
    # 策略：温和 zip。若 N(submit) == N(turns) 完美 zip；否则按时间窗试着匹配。
    # 错配宁可不写，也不写错——这些字段都是可选的 *_observed，前端缺则回退原 turn_start_time。
    n_turns = len(turns_cot)
    if submit_evs and n_turns:
        usable = submit_evs[:n_turns] if len(submit_evs) >= n_turns else submit_evs
        for turn, ev in zip(turns_cot, usable):
            t_ms = ev.get("t")
            if isinstance(t_ms, (int, float)):
                turn.turn_start_ms_observed = int(t_ms)
    if stop_evs and n_turns:
        usable = stop_evs[:n_turns] if len(stop_evs) >= n_turns else stop_evs
        for turn, ev in zip(turns_cot, usable):
            t_ms = ev.get("t")
            if isinstance(t_ms, (int, float)):
                turn.turn_end_ms_observed = int(t_ms)
    for turn in turns_cot:
        if turn.turn_start_ms_observed and turn.turn_end_ms_observed \
           and turn.turn_end_ms_observed >= turn.turn_start_ms_observed:
            turn.turn_duration_ms_observed = float(
                turn.turn_end_ms_observed - turn.turn_start_ms_observed
            )

    # ---- user_activity ----
    user_activity: List[Dict] = []

    for ev in submit_evs:
        bi = ev.get("brief_input") or {}
        entry = {
            "kind": "submit_prompt",
            "t": int(ev.get("t") or 0),
        }
        if bi.get("prompt_chars") is not None:
            entry["prompt_chars"] = bi.get("prompt_chars")
        if bi.get("prompt_preview"):
            entry["prompt_preview"] = bi.get("prompt_preview")
        if bi.get("generation_id"):
            entry["generation_id"] = bi.get("generation_id")
        user_activity.append(entry)

    for ev in tab_read_evs:
        bi = ev.get("brief_input") or {}
        entry = {
            "kind": "tab_read",
            "t": int(ev.get("t") or 0),
        }
        if bi.get("file_path"):
            entry["file_path"] = bi.get("file_path")
        user_activity.append(entry)

    for ev in tab_edit_evs:
        bi = ev.get("brief_input") or {}
        entry = {
            "kind": "tab_edit",
            "t": int(ev.get("t") or 0),
        }
        for k in ("file_path", "edits_count", "added_lines", "removed_lines",
                  "generation_id", "model"):
            if bi.get(k) is not None:
                entry[k] = bi.get(k)
        user_activity.append(entry)

    user_activity.sort(key=lambda x: x.get("t", 0))

    # 最多 500 条，避免极端长 session 把 cot.json 撑爆
    if len(user_activity) > 500:
        user_activity = user_activity[:500] + [{
            "kind": "_truncated",
            "t": int(user_activity[499].get("t") or 0),
            "note": f"+{len(user_activity) - 500} more activity entries truncated",
        }]

    return session_meta, user_activity


def _aggregate_invocation_stats(turns_cot: List["TurnCoT"]) -> InvocationStats:
    """把每个 turn 的 tool_decision invocation_category 汇总成 session 级统计。"""
    stats = InvocationStats()
    for turn in turns_cot:
        for s in turn.steps:
            if s.step_type != StepType.TOOL_DECISION:
                continue
            cat = (s.metadata or {}).get("invocation_category")
            if not cat:
                continue
            key = s.tool_name or "?"
            if cat == "llm_call":
                stats.llm_calls += 1
                stats.llm_call_distribution[key] = stats.llm_call_distribution.get(key, 0) + 1
            elif cat == "rag_query":
                stats.rag_queries += 1
                stats.rag_query_distribution[key] = stats.rag_query_distribution.get(key, 0) + 1
            elif cat == "web_search":
                stats.web_searches += 1
    return stats


# ─── v0.15.0: Claude / Cursor 识别 + Claude turn 边界修正 ───
#
# 背景：原 extract_session_cot 把每个 user 角色 tool_result 消息当作 turn 边界
# （flush() 一次新建 turn），这套规则是 Cursor transcript 的早期形态决定的，
# 在 Claude transcript 上完全失效——Claude 一个用户指令里可能有十几次
# tool_use ↔ tool_result 来回，每一对都被切成独立 turn，导致：
#   * 前端看到 9 个 turn，其中 8 个是没有用户输入的"伪 turn"
#   * 只有最后一个有 final_response，前 N-1 个都缺；
#   * usage / 复杂度 / 工具分布 没汇总到 logical turn 上。
#
# 修法策略：保持原 splitter 不动（避免动 Cursor 那条已稳定路径），加一个
# 后处理 _merge_claude_continuation_turns —— 把所有"没有 user_input step"的
# 伪 turn 合并回最近一个真 turn，把 step / token / final_response 统一归位。
# 这保证 Cursor 一侧零回归，同时把 Claude 那侧切回正确的 logical turn。

def _detect_agent_type(msgs: List[Dict]) -> str:
    """根据 transcript 顶层字段特征判定 IDE 来源。

    判定信号（用一个轻量加权，避免少量噪声造成误判）：

      Claude 信号：
        - 出现 ``type="permission-mode"`` 整行（Cursor 没有）
        - 出现 ``type="attachment"`` 且 ``attachment.hookEvent`` 非空
        - 出现 ``isSidechain`` 字段
        - assistant message 顶层 ``model`` 以 ``claude-`` 开头
        - usage 含 ``cache_creation_input_tokens`` / ``cache_read_input_tokens``

      Cursor 信号：
        - assistant message 含 OpenAI 风格 ``tool_calls`` 数组
        - 行级出现 ``cursor_request_id`` / ``cursor_session_id``
        - user message 内嵌 ``<hooks_context>`` 标签

      CodeBuddy 信号（v0.17.0 加，待 Phase 3 实物校准）：
        - 行级出现 ``codebuddy_*`` 前缀字段
        - source/origin 含 ``codebuddy`` / ``code-buddy`` / 腾讯云相关标志

    保守策略：所有 IDE 都没强信号时回 "unknown"，让前端按现状降级。
    """
    if not msgs:
        return "unknown"
    claude_score = 0
    cursor_score = 0
    codebuddy_score = 0

    def _shallow_text(o, depth=0) -> str:
        """递归收集前 ~3 层 dict/list 的字符串，用来做关键字匹配（轻量、避免栈深）。"""
        if depth > 3:
            return ""
        if isinstance(o, str):
            return o.lower()
        if isinstance(o, dict):
            return " ".join(
                f"{k}={_shallow_text(v, depth + 1)}" for k, v in list(o.items())[:30]
            )
        if isinstance(o, list):
            return " ".join(_shallow_text(x, depth + 1) for x in o[:20])
        return ""

    for m in msgs[:60]:  # 只扫前 60 条足以判定
        t = m.get("type")
        # ── Claude ──
        if t == "permission-mode":
            claude_score += 3
        if t == "attachment":
            att = m.get("attachment") or {}
            if att.get("hookEvent"):
                claude_score += 2
        if "isSidechain" in m:
            claude_score += 1
        # ── Cursor ──
        if any(k in m for k in ("cursor_request_id", "cursor_session_id", "generation_id")):
            cursor_score += 2
        # ── CodeBuddy ──
        if any(k in m for k in (
            "codebuddy_session_id", "codebuddySessionId",
            "codebuddy_request_id", "codebuddyRequestId",
        )):
            codebuddy_score += 3

        msg = m.get("message") or {}
        model = msg.get("model")
        if isinstance(model, str):
            ml = model.lower()
            if ml.startswith("claude-"):
                claude_score += 1
        usage = msg.get("usage") or {}
        if isinstance(usage, dict) and (
            "cache_creation_input_tokens" in usage
            or "cache_read_input_tokens" in usage
        ):
            claude_score += 1
        if isinstance(msg.get("tool_calls"), list):
            cursor_score += 1

        # source / origin 兜底（行级 IDE 标识）
        haystack = _shallow_text({k: m.get(k) for k in ("source", "origin", "agent", "service") if k in m})
        if haystack:
            if "codebuddy" in haystack or "code-buddy" in haystack:
                codebuddy_score += 1
            if "cursor" in haystack:
                cursor_score += 1
            if "claude" in haystack:
                claude_score += 1

    # 选最高分；阈值 ≥3 才认（防误判）
    scores = {
        "claude": claude_score,
        "cursor": cursor_score,
        "codebuddy": codebuddy_score,
    }
    winner = max(scores, key=lambda k: scores[k])
    if scores[winner] >= 3:
        # 必须严格高于第二名（防止平局误判）
        runner_up = max((v for k, v in scores.items() if k != winner), default=0)
        if scores[winner] > runner_up:
            return winner
    return "unknown"


def _merge_claude_continuation_turns(turns_cot: List[TurnCoT]) -> List[TurnCoT]:
    """把"无 user_input 起点"的伪 turn 合并回前一个真 turn。

    Claude transcript 的标准模式：
        user_prompt → asst(tool_use) → user(tool_result) → asst(tool_use)
                    → user(tool_result) → asst(end_turn)
    原 splitter 见到 ``user(tool_result)`` 就 flush()，会把 1 个逻辑 turn
    切成 N 个物理 turn。本函数把没有 user_input step 的 turn 合并到
    上一个真 turn（即第一个 step 不是 user_input 的）：

      * 步骤 list 串接，turn_index 重新归一
      * tool_calls / thinking_depth / strategy_shifts 累加
      * usage 数值字段相加，非数值字段在缺失时填补
      * final_response 取最新一个非空值（伪 turn 的 final_response 优先）
      * turn_duration_ms 求和（缺一个就退而保留另一个）
      * complexity_score 取 max
    其余字段（user_query / final_thought 等）以"被合入 turn"的为准。

    Cursor 路径 **不调用** 本函数 —— 老逻辑里 Cursor 的"伪 turn" 是有
    意义的（它们标识了 cot-stream 切回的回合边界），动了会回归。
    """
    if not turns_cot or len(turns_cot) <= 1:
        return turns_cot
    merged: List[TurnCoT] = []
    # StepType 在本仓库里是 plain class，``StepType.USER_INPUT`` 已经是字符串
    # ``"user_input"``——直接做字符串比较就行（dataclass 反序列化后也是 str）。
    user_input_kind = StepType.USER_INPUT
    for t in turns_cot:
        has_user_input = any(
            (getattr(s, "step_type", None) == user_input_kind)
            for s in t.steps
        )
        if has_user_input or not merged:
            merged.append(t)
            continue
        prev = merged[-1]
        for s in t.steps:
            s.turn_index = prev.turn_index
        prev.steps.extend(t.steps)
        if t.tool_calls:
            prev.tool_calls = (prev.tool_calls or []) + list(t.tool_calls)
        prev.thinking_depth = (prev.thinking_depth or 0) + (t.thinking_depth or 0)
        prev.strategy_shifts = (prev.strategy_shifts or 0) + (t.strategy_shifts or 0)
        prev.has_error_recovery = bool(prev.has_error_recovery or t.has_error_recovery)
        if t.usage:
            if not prev.usage:
                prev.usage = {}
            for k, v in t.usage.items():
                if isinstance(v, (int, float)):
                    prev.usage[k] = (prev.usage.get(k) or 0) + v
                elif k not in prev.usage:
                    prev.usage[k] = v
        if (t.final_response or "").strip():
            prev.final_response = t.final_response
        if t.turn_duration_ms is not None:
            prev.turn_duration_ms = (prev.turn_duration_ms or 0) + t.turn_duration_ms
        if t.complexity_score and t.complexity_score > (prev.complexity_score or 0):
            prev.complexity_score = t.complexity_score
    # ── v0.15.0：去除被原 splitter 误生成的孤立 synthetic tool_execution ──
    #
    # 原 extract_turn_cot 在 _has_real_exec=False 的 turn 里会给每个
    # tool_decision 紧跟插一个 ``synthetic=True`` 占位，这是为 Cursor 那条
    # 不写 tool_result 的 transcript 路径准备的。Claude 其实有 tool_result，
    # 它们在原 splitter 的"伪 turn"里被实化了；merge 之后两者并存：
    #   tool_decision(A) → tool_exec_synth(A) → ... → tool_exec_real(A)
    # 只保留 real 那一条，前端不再看到"结果未记录"的占位。
    for t in merged:
        real_ids: set = set()
        for s in t.steps:
            if (
                getattr(s, "step_type", None) == StepType.TOOL_EXECUTION
                and not (s.metadata or {}).get("synthetic")
                and s.tool_use_id
            ):
                real_ids.add(s.tool_use_id)
        if real_ids:
            t.steps = [
                s for s in t.steps
                if not (
                    getattr(s, "step_type", None) == StepType.TOOL_EXECUTION
                    and (s.metadata or {}).get("synthetic")
                    and s.tool_use_id in real_ids
                )
            ]
            t.total_steps = len(t.steps)
    return merged


# ─── v0.15.0/v0.15.1: Claude stream hook events.jsonl 注入 ───
#
# 由 claude-code/hooks/claude_stream_hook.py 在每个 hook 触发时追加一行
# JSONL 到 ``~/.claude/state/events/<session_id>/events.jsonl``。本函数
# 把这份文件按 hook_event 分流到 SessionCoT 的 5 条 Claude 专属时间线，
# 并把 Claude 27 个 hook **全部**事件计数到 ``session_meta.hook_events_observed``。
#
#   compact_events       ← PreCompact / PostCompact
#   subagent_timeline    ← SubagentStart / SubagentStop / TaskCreated / TaskCompleted
#   permission_events    ← PermissionRequest / PermissionDenied / Elicitation* + transcript permissionMode 行
#   notification_events  ← Notification / TeammateIdle / StopFailure
#   environment_events   ← ConfigChange / InstructionsLoaded / CwdChanged / FileChanged
#                          / WorktreeCreate / WorktreeRemove / Setup
#   ─ 仅计数（已有 transcript 真值，不重复落 step）：SessionStart / SessionEnd / Stop /
#     UserPromptSubmit / PreToolUse / PostToolUse / PostToolUseFailure
#
# 设计原则：
#   * 只读，绝不写——events.jsonl 是 hook 真值，不动它
#   * 文件不存在 = 用户没装 claude_stream_hook = 静默 no-op，前端拿到空数组
#   * 解析失败的行 skip，不影响其他行（hook payload 有时被截断）
#   * **transcript-first**（见 iwiki §12）：tool / session / prompt 信号不从 hook 重抽，
#     只补 transcript 不存在的事件类（subagent / compact / permission / notification / env）
#   * tool 真值时间戳通过 ``observed_at_ms`` / ``observed_duration_ms`` 回填到对应 step，
#     不替换 transcript 已有的 timestamp 字段

def _claude_events_paths(session_id: str) -> List[Path]:
    """Return all Claude hook event streams for OSS and internal wrappers."""
    home = Path.home()
    return [
        home / ".claude" / "state" / "events" / session_id / "events.jsonl",
        home / ".claude-internal" / "state" / "events" / session_id / "events.jsonl",
        home / ".claude-inertnal" / "state" / "events" / session_id / "events.jsonl",
    ]


def _claude_events_path(session_id: str) -> Path:
    """Backward-compatible primary path for callers that expect one file."""
    return _claude_events_paths(session_id)[0]


# 5 条时间线 + "仅计数" 两类的 hook 事件名分类常量。**变了任何一条都
# 必须同步 frontend types/index.ts 里的 hook_events_observed 渲染。**
_CLAUDE_HOOK_BUCKET_COMPACT      = ("PreCompact", "PostCompact")
_CLAUDE_HOOK_BUCKET_SUBAGENT     = (
    "SubagentStart", "SubagentStop", "TaskCreated", "TaskCompleted"
)
_CLAUDE_HOOK_BUCKET_PERMISSION   = (
    "PermissionRequest", "PermissionDenied", "Elicitation", "ElicitationResult"
)
_CLAUDE_HOOK_BUCKET_NOTIFICATION = ("Notification", "TeammateIdle", "StopFailure")
_CLAUDE_HOOK_BUCKET_ENV          = (
    "ConfigChange", "InstructionsLoaded", "CwdChanged", "FileChanged",
    "WorktreeCreate", "WorktreeRemove", "Setup",
)
# 这批 hook 是 transcript 已经能给出真值的（model / token / tool_input /
# tool_result / final_response / user_prompt 等）。我们只把 events.jsonl 里
# 它们的触发计数填进 hook_events_observed 给前端展示，**不**抽内容免得跟
# transcript-first 原则打架。tool 类的真值时间戳通过 _merge_tool_truth_ts
# 单独回填到 step.metadata。
_CLAUDE_HOOK_BUCKET_COUNTONLY = (
    "SessionStart", "SessionEnd", "Stop",
    "UserPromptSubmit",
    "PreToolUse", "PostToolUse", "PostToolUseFailure",
)


def _attach_claude_hook_events(session_cot: "SessionCoT") -> None:
    """读 events.jsonl + transcript 的环境信号，写到 session_cot 5 条时间线 + meta。

    副作用：直接 mutate
        - session_cot.compact_events
        - session_cot.subagent_timeline
        - session_cot.permission_events
        - session_cot.notification_events
        - session_cot.environment_events
        - session_cot.session_meta['hook_events_observed']
        - 各 turn.steps[*].metadata.observed_at_ms / observed_duration_ms（tool 真值时）
    """
    sid = getattr(session_cot, "session_id", "") or ""
    if not sid:
        return

    compact: List[Dict] = []
    subagent: List[Dict] = []
    permission: List[Dict] = []
    notification: List[Dict] = []
    environment: List[Dict] = []
    permission_mode_seen: List[Dict] = []
    # 27 hook 触发计数（含 0 次的，按集合枚举给前端）
    hook_counts: Dict[str, int] = {}
    # tool_use_id -> { observed_at_ms, observed_duration_ms } 用于回填到 step.metadata
    tool_truth_by_id: Dict[str, Dict[str, Any]] = {}

    def _bump(ev: str) -> None:
        hook_counts[ev] = hook_counts.get(ev, 0) + 1

    # 1) events.jsonl 来自 claude_stream_hook（用户没装就跳过——transcript 扫描照跑）
    for p in _claude_events_paths(sid):
        if not p.exists():
            continue
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                    except Exception:
                        continue
                    ev = rec.get("hook_event") or "Unknown"
                    t_ms = rec.get("t_ms")
                    if not isinstance(t_ms, (int, float)):
                        continue
                    payload = rec.get("payload") if isinstance(rec.get("payload"), dict) else {}
                    _bump(ev)

                    if ev in _CLAUDE_HOOK_BUCKET_COMPACT:
                        compact.append({
                            "t_ms": int(t_ms),
                            "trigger": payload.get("trigger") or (
                                "auto" if ev == "PreCompact" else None
                            ),
                            "before_tokens": payload.get("before_tokens"),
                            "after_tokens": payload.get("after_tokens"),
                            "saved_tokens": payload.get("saved_tokens"),
                            "summary_chars": payload.get("summary_chars"),
                            "phase": "before" if ev == "PreCompact" else "after",
                        })

                    elif ev in _CLAUDE_HOOK_BUCKET_SUBAGENT:
                        sub_id = (
                            payload.get("sub_agent_id")
                            or payload.get("subagent_id")
                            or payload.get("task_id")
                            or rec.get("tool_use_id")
                            or ""
                        )
                        subagent.append({
                            "t_ms": int(t_ms),
                            "sub_agent_id": str(sub_id) if sub_id else "",
                            "parent_tool_use_id": rec.get("tool_use_id") or payload.get("tool_use_id"),
                            "agent_type": payload.get("subagent_type") or payload.get("agent_type"),
                            "prompt_preview": (payload.get("prompt") or "")[:240] if isinstance(payload.get("prompt"), str) else None,
                            "model": payload.get("model"),
                            "summary": (payload.get("summary") or "")[:600] if isinstance(payload.get("summary"), str) else None,
                            "duration_ms": payload.get("duration_ms"),
                            "status": (
                                "running" if ev in ("SubagentStart", "TaskCreated")
                                else "completed" if ev in ("SubagentStop", "TaskCompleted")
                                else None
                            ),
                            "phase": ev,
                        })

                    elif ev in _CLAUDE_HOOK_BUCKET_PERMISSION:
                        permission.append({
                            "t_ms": int(t_ms),
                            "kind": ev,
                            "tool_name": rec.get("tool_name") or payload.get("tool_name"),
                            "tool_use_id": rec.get("tool_use_id") or payload.get("tool_use_id"),
                            "reason": payload.get("reason") or payload.get("message"),
                            "decision": payload.get("decision"),
                        })

                    elif ev in _CLAUDE_HOOK_BUCKET_NOTIFICATION:
                        notification.append({
                            "t_ms": int(t_ms),
                            "kind": ev,
                            "message": (payload.get("message") or "")[:600] if isinstance(payload.get("message"), str) else "",
                            "tool_name": rec.get("tool_name") or payload.get("tool_name"),
                            "tool_use_id": rec.get("tool_use_id") or payload.get("tool_use_id"),
                        })

                    elif ev in _CLAUDE_HOOK_BUCKET_ENV:
                        # 环境层事件——根据 ev 各取自己的关键字段，**不丢**未识别字段
                        # （挂在 details 上让前端摊开）
                        env_entry: Dict[str, Any] = {
                            "t_ms": int(t_ms),
                            "kind": ev,
                        }
                        if ev == "CwdChanged":
                            env_entry["before"] = payload.get("before") or payload.get("from")
                            env_entry["after"]  = payload.get("after")  or payload.get("to") or payload.get("cwd")
                        elif ev == "FileChanged":
                            env_entry["path"] = payload.get("path") or payload.get("file_path")
                            env_entry["change_kind"] = payload.get("change_kind") or payload.get("kind")
                            env_entry["is_user_initiated"] = payload.get("is_user_initiated")
                        elif ev in ("WorktreeCreate", "WorktreeRemove"):
                            env_entry["worktree_path"] = payload.get("worktree_path") or payload.get("path")
                            env_entry["branch"] = payload.get("branch")
                        elif ev == "ConfigChange":
                            env_entry["key"] = payload.get("key")
                            env_entry["before"] = payload.get("before")
                            env_entry["after"]  = payload.get("after")
                        elif ev == "InstructionsLoaded":
                            files = payload.get("instruction_files") or payload.get("files")
                            env_entry["instruction_files"] = files if isinstance(files, list) else []
                        elif ev == "Setup":
                            env_entry["setup_args"] = payload.get("setup_args")
                            env_entry["claude_version"] = payload.get("claude_version")
                        # 兜底：原 payload 摘要也带上
                        env_entry["details"] = {
                            k: v for k, v in payload.items()
                            if k not in ("hook_event_name", "session_id")
                            and not isinstance(v, (dict, list))
                        }
                        environment.append(env_entry)

                    elif ev in _CLAUDE_HOOK_BUCKET_COUNTONLY:
                        # 计数已经在 _bump 完成；这里再针对 PreToolUse/PostToolUse
                        # 抽 hook 真值时间戳，便于回填到对应 step.metadata。
                        if ev in ("PreToolUse", "PostToolUse", "PostToolUseFailure"):
                            tu_id = (
                                rec.get("tool_use_id")
                                or payload.get("tool_use_id")
                                or ""
                            )
                            if tu_id:
                                bucket = tool_truth_by_id.setdefault(str(tu_id), {})
                                if ev == "PreToolUse":
                                    bucket["pre_t_ms"] = int(t_ms)
                                else:
                                    bucket["post_t_ms"] = int(t_ms)
                                    if ev == "PostToolUseFailure":
                                        bucket["failure"] = True
                                # 同时挂工具名（前端 fallback 用）
                                tn = rec.get("tool_name") or payload.get("tool_name")
                                if tn and "tool_name" not in bucket:
                                    bucket["tool_name"] = tn
                    # 其他未知 hook：只计数，不分流（未来 Anthropic 加新 hook 不会丢）
        except Exception:
            # events.jsonl 解析失败不影响 transcript permission-mode 扫描
            pass

    # 2) transcript 顶层 ``type='permission-mode'`` 行也归到 permission_events。
    # 字段名是 camelCase 的 ``permissionMode``（不是 mode / permission_mode），
    # 这是 Claude Code transcript 的稳定写法；时间戳通常缺失，t_ms 给 0 占位。
    try:
        tr_path = getattr(session_cot, "transcript_path", "") or ""
        if tr_path and Path(tr_path).is_file():
            with open(tr_path, "r", encoding="utf-8", errors="replace") as f:
                prev_mode = None
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                    except Exception:
                        continue
                    if rec.get("type") != "permission-mode":
                        continue
                    mode = (
                        rec.get("permissionMode")
                        or rec.get("mode")
                        or rec.get("permission_mode")
                        or "default"
                    )
                    ts = rec.get("timestamp")
                    t_ms = _ts_to_ms(ts) if ts else None
                    permission_mode_seen.append({
                        "t_ms": int(t_ms) if t_ms else 0,
                        "kind": "PermissionMode",
                        "mode": mode,
                        "prev_mode": prev_mode,
                        "source": "transcript",
                    })
                    prev_mode = mode
    except Exception:
        pass

    if permission_mode_seen:
        permission.extend(permission_mode_seen)

    # 排序，写回
    compact.sort(key=lambda x: x.get("t_ms") or 0)
    subagent.sort(key=lambda x: x.get("t_ms") or 0)
    permission.sort(key=lambda x: x.get("t_ms") or 0)
    notification.sort(key=lambda x: x.get("t_ms") or 0)
    environment.sort(key=lambda x: x.get("t_ms") or 0)

    try:
        session_cot.compact_events     = compact
        session_cot.subagent_timeline  = subagent
        session_cot.permission_events  = permission
        session_cot.notification_events = notification
        session_cot.environment_events = environment
    except Exception:
        # 字段还没声明（老 SessionCoT），忽略
        pass

    # 3) tool 真值时间戳回填：按 tool_use_id 找到对应 step.metadata。
    if tool_truth_by_id:
        try:
            for turn in (session_cot.turns or []):
                for step in (turn.steps or []):
                    tid = (
                        getattr(step, "tool_use_id", None)
                        or (step.metadata or {}).get("tool_use_id")
                        or ""
                    )
                    if not tid:
                        continue
                    truth = tool_truth_by_id.get(str(tid))
                    if not truth:
                        continue
                    md = step.metadata if isinstance(step.metadata, dict) else {}
                    pre_t = truth.get("pre_t_ms")
                    post_t = truth.get("post_t_ms")
                    if pre_t is not None:
                        md["observed_pre_t_ms"] = pre_t
                        md["observed_at_ms"] = pre_t  # 跟 cursor 那边的字段名对齐
                    if post_t is not None:
                        md["observed_post_t_ms"] = post_t
                    if pre_t is not None and post_t is not None and post_t >= pre_t:
                        md["observed_duration_ms"] = post_t - pre_t
                    if truth.get("failure"):
                        md["observed_tool_failure"] = True
                    if truth.get("tool_name") and not (md.get("tool_name") or step.tool_name):
                        md["tool_name_observed"] = truth["tool_name"]
                    step.metadata = md
        except Exception:
            pass

    # 4) hook_events_observed：填到 session_meta，前端可一眼看到 27 hook 实际触发分布。
    if hook_counts:
        try:
            sm = session_cot.session_meta
            if not isinstance(sm, dict):
                sm = {}
            existing = sm.get("hook_events_observed")
            if isinstance(existing, dict):
                # 跟 transcript 推断的 hook_events_observed 合并（取 max），不互相覆盖
                merged = dict(existing)
                for k, v in hook_counts.items():
                    merged[k] = max(int(merged.get(k, 0)), int(v))
                sm["hook_events_observed"] = merged
            else:
                sm["hook_events_observed"] = dict(hook_counts)
            session_cot.session_meta = sm
        except Exception:
            pass


def extract_session_cot(
    transcript_path: Path,
    session_id: str,
    offset: int = 0,
) -> Tuple[Optional[SessionCoT], int]:
    """
    从 transcript 文件中提取整个 session 的 CoT。

    Args:
        transcript_path: transcript .jsonl 文件路径
        session_id: 会话 ID
        offset: 读取起始偏移量（用于增量读取）

    Returns:
        (SessionCoT, new_offset)，如果没有新内容则返回 (None, offset)
    """
    from datetime import datetime, timezone

    # ── CodeBuddy fast-path ─────────────────────────────────────────────
    # Two ways the caller may have routed us here:
    #   1. transcript_path already points at a CodeBuddy ``index.json``
    #      (extract_cot.py CLI / watcher discovered it directly), or
    #   2. transcript_path is the events.jsonl placeholder because the
    #      Cursor-style discovery couldn't find anything — but the events
    #      themselves come from cot-stream-codebuddy.js and carry the
    #      real CodeBuddy index path inside payload.transcript_path.
    # Either way, prefer the rich transcript parser before falling back
    # to events-only synthesis.
    _events_for_codebuddy: Optional[List[Dict]] = None
    _codebuddy_index: Optional[Path] = None
    try:
        from codebuddy_transcript import (  # type: ignore
            _exists_long as _cb_exists_long,
            transcript_path_from_events as _cb_transcript_path_from_events,
            find_transcript_by_session_id as _cb_find_by_sid,
        )
        _cb_imports_ok = True
    except Exception:
        _cb_exists_long = None  # type: ignore
        _cb_transcript_path_from_events = None  # type: ignore
        _cb_find_by_sid = None  # type: ignore
        _cb_imports_ok = False

    if (transcript_path.name.lower() == "index.json"
            and _cb_imports_ok and _cb_exists_long(transcript_path)):
        _codebuddy_index = transcript_path
        _events_for_codebuddy = _load_cursor_events(session_id)
    elif _cb_imports_ok:
        _events_for_codebuddy = _load_cursor_events(session_id)
        if _detect_agent_type_from_events(_events_for_codebuddy) == "codebuddy":
            _codebuddy_index = (
                _cb_transcript_path_from_events(_events_for_codebuddy)
                or _cb_find_by_sid(session_id)
            )
    if _codebuddy_index is not None and _cb_imports_ok and _cb_exists_long(_codebuddy_index):
        rich = _extract_codebuddy_session_from_transcript(
            index_path=_codebuddy_index,
            session_id=session_id,
            events=_events_for_codebuddy or [],
        )
        if rich is not None and rich.turns:
            return rich, offset

    if not transcript_path.exists():
        events_only = _extract_events_only_session(
            transcript_path=transcript_path,
            session_id=session_id,
            events=_events_for_codebuddy if _events_for_codebuddy is not None
                   else _load_cursor_events(session_id),
        )
        return (events_only, offset) if events_only is not None else (None, offset)

    # 读取 transcript
    try:
        with open(transcript_path, "rb") as f:
            f.seek(offset)
            chunk = f.read()
            new_offset = f.tell()
    except Exception as e:
        return None, offset

    if not chunk:
        return None, offset

    text = chunk.decode("utf-8", errors="replace")
    msgs = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            msgs.append(json.loads(line))
        except Exception:
            continue

    if not msgs:
        return None, offset

    # 按 user turn 分组
    turns_cot: List[TurnCoT] = []
    current_user: Optional[Dict] = None
    current_user_ts: Optional[str] = None
    assistant_msgs: List[Dict] = []
    assistant_msg_tss: List[Optional[str]] = []
    turn_index = 0
    global_step_offset = 0

    # v0.14.1: Cursor 长会话中会注入两类**伪 user 消息**，过去都被错误
    # 当作 turn 边界，导致同一次用户指令下的工作被拆成 #N、#N+1 多个 span：
    #
    # 1) **重复 user_query**：在指令长度 / 上下文压力大时，Cursor 会把
    #    最近一次 user_query 完整重放一遍（观察到 idx 361 / 362 / 417 三次
    #    出现 len=214 一字不差的文本）。
    # 2) **纯 hooks_context 注入**：v0.14 起 Cursor hooks 会单独发一条
    #    user 角色消息，里面只有 ``<hooks_context>...</hooks_context>``
    #    （memory_context / observations 等），完全没有 ``<user_query>``。
    #    它本质是给模型刷一段背景，不是用户新指令，不应当作 turn 边界。
    #
    # 这里用 `seen_user_texts` + `_is_pure_injection` 两层防护：命中其一
    # 就跳过 flush，让后续 assistant 继续累在当前 turn 里。
    # 安全阈值：dedup 只对 ≥30 char 的「实质性消息」生效，避免把多条
    # 「好的」「ok」「继续」误并；tool_result 走另外的分支不受影响。
    seen_user_texts: set = set()
    DEDUP_MIN_LEN = 30

    def _norm_user_text(m: Dict) -> str:
        c = _get_content(m)
        text = _extract_text_only(c) if not isinstance(c, str) else c
        return (text or "").strip()

    def _is_pure_injection(text: str) -> bool:
        """判断 user 角色消息是否是纯系统/hook 注入（无真实用户输入）。

        判定规则（保守，宁可漏判也不误判）：
        - 文本以 ``<hooks_context`` / ``<system_reminder`` /
          ``<session-start>`` / ``<memory_context`` 等系统标签开头；
        - **且** 不包含 ``<user_query>`` 块。

        这样 Cursor 把 hooks_context 拼进真正 user_query 的混合消息
        （header 是 hooks_context、尾部带 ``<user_query>``）依然被当
        成真实用户输入处理。
        """
        if not text:
            return True
        s = text.lstrip()
        injection_prefixes = (
            "<hooks_context",
            "<memory_context",
            "<system_reminder",
            "<session-start",
            "<session_summary",
        )
        if s.startswith(injection_prefixes) and "<user_query>" not in s:
            return True
        return False

    def flush():
        nonlocal current_user, current_user_ts, assistant_msgs, assistant_msg_tss, turn_index, global_step_offset
        if current_user is None or not assistant_msgs:
            return
        turn_index += 1
        turn_cot = extract_turn_cot(
            user_msg=current_user,
            assistant_msgs=assistant_msgs,
            turn_index=turn_index,
            global_step_offset=global_step_offset,
            user_msg_ts=current_user_ts,
            assistant_msg_timestamps=assistant_msg_tss,
        )
        turns_cot.append(turn_cot)
        global_step_offset += turn_cot.total_steps

    def _merge_tool_result_into_current(prev: Dict, cur: Dict) -> None:
        """把 cur（tool_result-only user msg）的 tool_result 块并入 prev。

        v0.15.0：Claude transcript 里同一个 ``tool_use_id`` 经常被记录 2 次
        （API 原生输出 + 某些 hook 把结果再回灌一遍），原 splitter 见到
        每条都 flush()，导致只有最后一条留下 current_user，前面的
        tool_result 全部丢失。本函数在 assistant_msgs 为空、current_user
        本就是 tool_result-only 时把新块并进去（同 tool_use_id 取最新）。

        ``prev`` / ``cur`` 都是 transcript 顶层 dict；这里直接 mutate prev
        的 message.content 列表（msgs 列表只在本函数生命周期里使用，无
        外部副作用）。Cursor 路径上这个函数从不会被走进来——它的入口
        条件 ``_is_tool_result_msg(prev)`` 只在 Claude 顺序的 tool_result
        连发场景才成立。
        """
        prev_msg = prev.get("message") or {}
        cur_msg = cur.get("message") or {}
        prev_content = prev_msg.get("content")
        cur_content = cur_msg.get("content")
        if not isinstance(prev_content, list) or not isinstance(cur_content, list):
            return
        # 同 tool_use_id 替换为最新；其余追加
        existing_idx: Dict[str, int] = {}
        for i, b in enumerate(prev_content):
            if isinstance(b, dict) and b.get("type") == "tool_result":
                bid = b.get("tool_use_id") or ""
                if bid:
                    existing_idx[bid] = i
        for b in cur_content:
            if not isinstance(b, dict) or b.get("type") != "tool_result":
                continue
            bid = b.get("tool_use_id") or ""
            if bid and bid in existing_idx:
                prev_content[existing_idx[bid]] = b
            else:
                prev_content.append(b)
                if bid:
                    existing_idx[bid] = len(prev_content) - 1

    for msg in msgs:
        role = _get_role(msg)
        if _is_tool_result_msg(msg):
            # v0.15.0：Claude transcript 把同一个工具结果重复发 2 次 +
            # 一个 user prompt 引出 N 次工具调用循环，每次都被 flush 掉。
            # 这里只在「真有 assistant 工作要刷出去」或「前一个 user 不是
            # tool_result」时才 flush，否则 **合并** 进当前 user 的 content
            # 列表，让 extract_turn_cot 一次拿到所有 tool_result。
            if assistant_msgs or current_user is None or not _is_tool_result_msg(current_user):
                flush()
                current_user = msg
                current_user_ts = _get_timestamp(msg)
                assistant_msgs = []
                assistant_msg_tss = []
            else:
                _merge_tool_result_into_current(current_user, msg)
                # 时间戳取最新一条 tool_result 的时间，duration 计算更准
                _new_ts = _get_timestamp(msg)
                if _new_ts:
                    current_user_ts = _new_ts
        elif role == "user":
            user_text = _norm_user_text(msg)
            # ① 纯 hooks/system 注入抑制：current_user 已建立时直接跳过，
            #    防止把 agent 在同一指令下的后续工作分到一个新 turn 里。
            if (
                current_user is not None
                and _is_pure_injection(user_text)
            ):
                continue
            # ② 重复 user_query 抑制：长度足够 + 文本已见过 → 跳过 flush
            #    （把后续 assistant 累到当前 turn）。如果 current_user
            #    还没建立，仍然要建立——这是该 user_query 第一次出现。
            if (
                user_text
                and len(user_text) >= DEDUP_MIN_LEN
                and user_text in seen_user_texts
                and current_user is not None
            ):
                continue
            flush()
            current_user = msg
            current_user_ts = _get_timestamp(msg)
            assistant_msgs = []
            assistant_msg_tss = []
            if user_text and len(user_text) >= DEDUP_MIN_LEN:
                seen_user_texts.add(user_text)
        elif role == "assistant":
            if current_user is not None:
                assistant_msgs.append(msg)
                assistant_msg_tss.append(_get_timestamp(msg))

    flush()

    if not turns_cot:
        events_only = _extract_events_only_session(
            transcript_path=transcript_path,
            session_id=session_id,
            events=_load_cursor_events(session_id),
        )
        return (events_only, new_offset) if events_only is not None else (None, new_offset)

    # ─── v0.15.0: 识别 IDE 来源；Claude 修正 turn 边界 ───
    # 必须在汇总统计前跑——合并完成后 turn 数目 / 工具分布 / 步骤数都会变。
    agent_type = _detect_agent_type(msgs)
    if agent_type == "claude":
        try:
            turns_cot = _merge_claude_continuation_turns(turns_cot)
        except Exception as _merge_err:
            # 失败则保留原 splitter 结果，不阻塞主链路；但把根因写到 stderr
            # 方便日志巡检（没有正式 logger，避免引入新依赖）。
            try:
                import sys as _sys
                _sys.stderr.write(
                    f"[cot_extractor] _merge_claude_continuation_turns failed: "
                    f"{type(_merge_err).__name__}: {str(_merge_err)[:200]}\n"
                )
            except Exception:
                pass

    # 汇总统计
    total_tool_calls = sum(len(t.tool_calls) for t in turns_cot)
    total_strategy_shifts = sum(t.strategy_shifts for t in turns_cot)
    total_thinking_steps = sum(t.thinking_depth for t in turns_cot)

    tool_dist: Dict[str, int] = {}
    for t in turns_cot:
        for tool in t.tool_calls:
            tool_dist[tool] = tool_dist.get(tool, 0) + 1

    avg_steps = round(
        sum(t.total_steps for t in turns_cot) / len(turns_cot), 2
    ) if turns_cot else 0.0

    avg_complexity = round(
        sum(t.complexity_score for t in turns_cot) / len(turns_cot), 2
    ) if turns_cot else 0.0

    # v0.7.0: 合并 cot-stream.js 实时事件 + 抽取 plan_timeline —— 都做成软失败。
    observed_stats: Optional[Dict] = None
    session_meta: Optional[Dict] = None
    user_activity: List[Dict] = []
    try:
        _cursor_events = _load_cursor_events(session_id)
        if _cursor_events:
            event_agent_type = _detect_agent_type_from_events(_cursor_events)
            if event_agent_type and agent_type in ("unknown", "cursor"):
                agent_type = event_agent_type
            observed_stats = _attach_cursor_events(turns_cot, _cursor_events)
            if observed_stats is not None:
                observed_stats["agent_type_from_events"] = event_agent_type
                observed_stats["providers_observed"] = sorted({
                    str(e.get("provider") or "").strip()
                    for e in _cursor_events if e.get("provider")
                })
                observed_stats["events_paths"] = sorted({
                    str(e.get("_events_path") or "")
                    for e in _cursor_events if e.get("_events_path")
                })
            # v0.13.x: 通道 5 —— afterAgentThought → thinking_explicit step。
            # 必须在 _attach_cursor_events 之后跑，因为它依赖 channel 1 注入
            # 上去的 metadata.generation_id / observed_at_ms 作为锚点。
            # 失败把错误记到 stats 里，方便前端 / 运维看到，而不是默默吞掉。
            try:
                thought_stats = _inject_agent_thoughts(turns_cot, _cursor_events)
                if observed_stats is not None and isinstance(thought_stats, dict):
                    observed_stats.update(thought_stats)
            except Exception as _e:
                if observed_stats is not None:
                    observed_stats["thought_error"] = (
                        f"{type(_e).__name__}: {str(_e)[:200]}"
                    )
            # v0.14.2: 通道 6 —— sessionStart / sessionEnd / beforeSubmitPrompt / stop /
            # beforeTabFileRead / afterTabFileEdit 的真值时间锚 + 用户活动时间线。
            try:
                session_meta, user_activity = _attach_lifecycle_events(
                    turns_cot, _cursor_events,
                )
                if observed_stats is not None and session_meta is not None:
                    observed_stats["lifecycle"] = {
                        "session_meta_present": True,
                        "user_activity_count": len(user_activity),
                        "hook_events_observed": session_meta.get(
                            "hook_events_observed") or {},
                    }
            except Exception as _e:
                if observed_stats is not None:
                    observed_stats["lifecycle_error"] = (
                        f"{type(_e).__name__}: {str(_e)[:200]}"
                    )
            # v0.17.x: 通道 7 —— afterAgentResponse → turn.usage 真值回填。
            # 必须在 _attach_lifecycle_events 之后跑，因为依赖
            # turn_start_ms_observed / turn_end_ms_observed 做时间窗匹配。
            # GPT 模型在 cursor transcript 里不写 message.usage，这一步是
            # 让前端 "所在 Turn Token" 不显示 0 的唯一来源。Claude 已有真值的
            # turn 会被跳过，绝不重复计数。
            try:
                token_stats = _attach_cursor_token_usage(turns_cot, _cursor_events)
                if observed_stats is not None and isinstance(token_stats, dict):
                    observed_stats["token_usage_backfill"] = token_stats
            except Exception as _e:
                if observed_stats is not None:
                    observed_stats["token_usage_backfill_error"] = (
                        f"{type(_e).__name__}: {str(_e)[:200]}"
                    )
    except Exception:
        observed_stats = None

    # v0.14.4：用 cot-stream.js 注入的 step.metadata.observed_at_ms 给那些
    # transcript ts 缺失的 turn 补一份真值耗时。必须在 _attach_cursor_events
    # 之后跑（之前 turn_duration_ms 几乎全是 None）。
    try:
        _recompute_turn_durations(turns_cot)
    except Exception:
        pass

    # v0.14.7：本地 MCP 代理流量回填——把 ~/.agent-cot/mcp-traffic/ 下的
    # JSONL 里 result.content[].text 写到对应 CallMcpTool step 的 observed_output。
    # Hook 那一侧给 MCP 工具的脱水 metadata 经常空白，代理是 wire 字节直采的
    # 唯一可靠路径。模块自带"绝不覆盖已有真实数据"约束，软失败。
    try:
        # 兼容两种调用方式：作为 package 子模块 import（ pytest / -m 调用）
        # 和作为顶层脚本（scripts/extract_cot.py 直接 sys.path 注入）。
        try:
            from .mcp_traffic_reader import attach_mcp_traffic
        except ImportError:
            from mcp_traffic_reader import attach_mcp_traffic  # type: ignore
        # 时间窗口尽量精确，少扫日志：取所有 turn 的 observed_at_ms 极值
        all_obs_ms: List[int] = []
        for _t in turns_cot:
            for _s in _t.steps:
                _md = _s.metadata if isinstance(_s.metadata, dict) else None
                if _md:
                    _v = _md.get("observed_at_ms")
                    if isinstance(_v, (int, float)):
                        all_obs_ms.append(int(_v))
        _mcp_stats = attach_mcp_traffic(
            turns_cot,
            session_start_ms=min(all_obs_ms) if all_obs_ms else None,
            session_end_ms=max(all_obs_ms) if all_obs_ms else None,
        )
        if observed_stats is not None and isinstance(_mcp_stats, dict):
            observed_stats["mcp_traffic"] = _mcp_stats
    except Exception as _e:
        if observed_stats is not None:
            observed_stats["mcp_traffic_error"] = (
                f"{type(_e).__name__}: {str(_e)[:200]}"
            )

    plan_timeline: List[PlanSnapshot] = []
    try:
        plan_timeline = _build_plan_timeline(turns_cot)
    except Exception:
        plan_timeline = []

    # v0.10.0: 模式转换时间线（plan ↔ agent ↔ debug ↔ ask）+ CreatePlan 文档
    mode_transitions: List[ModeTransition] = []
    plan_proposals: List[PlanProposal] = []
    try:
        mode_transitions, plan_proposals = _build_mode_transitions(turns_cot)
    except Exception:
        mode_transitions = []
        plan_proposals = []

    # v0.8.0: 在 _attach_cursor_events 之后做"决策→执行"分类传播 +
    # 召回抽取，再聚合一次 session 级统计。任一步骤失败都软退出。
    try:
        _propagate_invocation_to_executions(turns_cot)
    except Exception:
        pass

    invocation_stats: Optional[InvocationStats] = None
    try:
        invocation_stats = _aggregate_invocation_stats(turns_cot)
    except Exception:
        invocation_stats = None

    # v0.9.0: L5 Execution Trace —— 扫 Write/StrReplace/Delete + Shell 反查
    # 产出 ScriptArtifact 列表与 session 级 stats，并就地给 step.metadata 写
    # ``file_op`` / ``executed_artifact`` 子对象，前端用来高亮临时脚本。
    script_artifacts_serialized: List[Dict] = []
    script_stats_dict: Optional[Dict] = None
    if _build_script_artifacts is not None:
        try:
            tracker = _build_script_artifacts(turns_cot)
            arts = tracker.get("artifacts") or []
            script_artifacts_serialized = [
                a.to_dict() if hasattr(a, "to_dict") else a for a in arts
            ]
            stats_obj = tracker.get("stats")
            if stats_obj is not None:
                script_stats_dict = (
                    stats_obj.to_dict() if hasattr(stats_obj, "to_dict") else stats_obj
                )
        except Exception:
            script_artifacts_serialized = []
            script_stats_dict = None

    # v0.18.7: 通道 5（_attach_cursor_events 内 events→step 合成）+ 通道 5'
    # （_inject_agent_thoughts 把 afterAgentThought 注入为 thinking_explicit）
    # 都在前面的 ``汇总统计`` 区段之后才跑过，所以这里必须**重算**所有 session
    # 级聚合，否则 SessionCoT 仍然带着合成前的冻结值（``total_tool_calls=0``，
    # ``tool_call_distribution={}``），前端 KPI bar 就会出现 "thinking 一大堆但
    # tool=0 / cost=0 / 模型=unknown" 的明显割裂感。
    total_tool_calls = sum(len(t.tool_calls) for t in turns_cot)
    total_strategy_shifts = sum(t.strategy_shifts for t in turns_cot)
    total_thinking_steps = sum(t.thinking_depth for t in turns_cot)
    tool_dist = {}
    for _t_ in turns_cot:
        for _tool_ in _t_.tool_calls:
            tool_dist[_tool_] = tool_dist.get(_tool_, 0) + 1
    avg_steps = round(
        sum(t.total_steps for t in turns_cot) / len(turns_cot), 2
    ) if turns_cot else 0.0
    avg_complexity = round(
        sum(t.complexity_score for t in turns_cot) / len(turns_cot), 2
    ) if turns_cot else 0.0

    session_cot = SessionCoT(
        session_id=session_id,
        transcript_path=str(transcript_path),
        extracted_at=datetime.now(timezone.utc).isoformat(),
        turns=turns_cot,
        total_tool_calls=total_tool_calls,
        total_strategy_shifts=total_strategy_shifts,
        total_thinking_steps=total_thinking_steps,
        tool_call_distribution=tool_dist,
        avg_steps_per_turn=avg_steps,
        avg_complexity=avg_complexity,
        plan_timeline=plan_timeline,
        observed_events=observed_stats,
        invocation_stats=invocation_stats,
        script_artifacts=script_artifacts_serialized,
        script_stats=script_stats_dict,
        mode_transitions=mode_transitions,
        plan_proposals=plan_proposals,
        session_meta=session_meta,
        user_activity=user_activity,
        agent_type=agent_type,
    )

    # v0.16.2: tool_execution 顶层 / metadata 里补 tool_name。
    # 单 turn 内做不到（tool_result 在当前 turn 的 user_msg 里、对应 tool_decision
    # 在前一 turn 的 assistant_msgs 里），所以放在 session 级做一次跨 turn 的
    # tool_use_id → tool_name 字典扫描后填回。这样前端 SpanTree 能直接显示
    # "Tool Execution → Read"，并且能按 use_id 配对 D-E 重排。
    try:
        _name_by_use_id: Dict[str, str] = {}
        for _t in session_cot.turns:
            for _s in _t.steps:
                if _s.step_type == StepType.TOOL_DECISION and _s.tool_use_id and _s.tool_name:
                    _name_by_use_id.setdefault(_s.tool_use_id, _s.tool_name)
        for _t in session_cot.turns:
            for _s in _t.steps:
                if _s.step_type != StepType.TOOL_EXECUTION:
                    continue
                _uid = _s.tool_use_id or (_s.metadata or {}).get("tool_use_id", "")
                if not _uid:
                    continue
                _name = _name_by_use_id.get(_uid)
                if not _name:
                    continue
                if not _s.tool_name:
                    _s.tool_name = _name
                if _s.metadata is None:
                    _s.metadata = {}
                _s.metadata.setdefault("tool_name", _name)
                _s.metadata.setdefault("tool_use_id", _uid)
    except Exception:
        pass

    # v0.15.0：Claude path —— 把 claude_stream_hook 落地的 events.jsonl
    # 拆解成四条时间线，注入到新字段。文件不存在时保持空 list（前端零回归）。
    if agent_type == "claude":
        try:
            _attach_claude_hook_events(session_cot)
        except Exception:
            pass

    # v0.11.0: OTel GenAI 视图（client-side 合成 trace/span/token/cost/messages/retrieval/eval）
    # 软失败：任何异常都不影响主提取，session 仍完整返回。
    if _enrich_otel is not None:
        try:
            _enrich_otel(session_cot)
        except Exception:
            pass

    return session_cot, new_offset
