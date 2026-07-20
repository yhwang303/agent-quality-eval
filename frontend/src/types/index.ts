// TypeScript 类型定义

// v0.11.2：SessionList 行内可直接渲染的 OTel KPI 子集（来自 cot.otel_view）
export interface SessionOtelKpi {
  model: string;
  model_source?: string | null;
  agent_name?: string | null;
  provider?: string | null;
  cost_usd: number | null;
  full_price_cost_usd?: number | null;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  cache_hit_rate: number | null;
  cursor_version?: string | null;
  events_count?: number | null;
  has_actual_usage: boolean;
}

export interface SessionOverview {
  session_id: string;
  topic: string;
  extracted_at: string;
  transcript_path: string;
  total_turns: number;
  total_tool_calls: number;
  total_thinking_steps: number;
  total_strategy_shifts: number;
  avg_complexity: number;
  avg_steps_per_turn: number;
  tool_call_distribution: Record<string, number>;
  has_response_report: boolean;
  has_transcript: boolean;
  response_score: number | null;
  project_name?: string;
  project_path?: string;
  project_id?: string;
  project_source?: 'workspace_roots' | 'transcript_path' | 'fallback' | string;
  // 父子 session 字段
  is_parent?: boolean;
  sub_sessions?: SubSessionInfo[];
  sub_session_count?: number;
  // v0.11.2：OTel 关键 KPI（自动检测；未启用 cot-stream / 老 session 时为 null）
  otel?: SessionOtelKpi | null;
  // v0.18.0：CoT Uplink 来源标记
  //   owner = null  → 本机 session（『我自己』）
  //   owner = 'zhangsan' → 来自小组同事的 central uplink，session_id 形如 'zhangsan::abc-123'
  owner?: string | null;
  source?: 'local' | 'uplink' | string;
  host?: string;             // 同事的 hostname（central 来源才有）
  received_at?: string;      // 中央服务收到这份 cot.json 的时间
  // v0.20.7: cot_extractor._detect_agent_type 写的 IDE 来源（前端 SessionList
  // 用它给每个 session 卡片打不同颜色的 IDE 徽章）。
  agent_type?: 'cursor' | 'claude' | 'codex' | 'codebuddy' | string | null;
}

// v0.18.0：中央上行用户列表（GET /api/uplink/users）
export interface EvalEvent {
  id: number;
  created_at: string;
  event_type: 'trace' | 'reference' | 'ab' | 'regression' | 'gold' | string;
  project_id?: string | null;
  project_name?: string | null;
  project_path?: string | null;
  session_id?: string | null;
  turn_index?: number | null;
  baseline_session_id?: string | null;
  baseline_turn_index?: number | null;
  candidate_session_id?: string | null;
  candidate_turn_index?: number | null;
  has_gold: boolean;
  gold_hash?: string | null;
  verdict?: string | null;
  winner?: string | null;
  summary: Record<string, any>;
  target: Record<string, any>;
}

export interface UplinkUserSummary {
  user_id: string;
  session_count: number;
  last_uploaded: string | null;
  total_bytes: number;
}

export interface SubSessionInfo {
  sub_session_id: string;
  topic: string;
  extracted_at: string;
  total_turns: number;
  total_tool_calls: number;
  avg_complexity?: number;
  tool_call_distribution?: Record<string, number>;
}

export type StepType =
  | 'user_input'
  | 'tool_result_input'
  | 'thinking_inter'
  | 'thinking_intermediate'
  | 'thinking_explicit'
  | 'pre_tool_reasoning'
  | 'tool_decision'
  | 'tool_execution'
  | 'strategy_shift'
  | 'error_recovery'
  | 'final_response';

export interface ReasoningDigest {
  why: string;           // 简短理由
  evidence: string;      // 证据引用
  basis: string;         // 决策依据
  next_plan: string;     // 下一步计划
}

export interface DecisionTrace {
  trigger_context: string;       // 触发调用的上下文
  tool_selection_reason: string; // 工具选择原因
  param_inference: string;       // 参数推断
  continuation_reason: string;   // 后续决策链
}

export interface StateEvolution {
  context_hash: string;       // 上下文 hash
  evidence_summary: string;   // 新增证据摘要
  action_schema: string;      // action 类型
  termination_check: string;  // 终止条件检查
}

export interface ErrorTrace {
  is_error_origin: boolean;       // 是否错误起源
  error_step_index: number;       // 关联错误步骤
  referenced_by: number[];        // 被引用的步骤
  correction_opportunity: boolean; // 未纠正机会
  contradicts_final: boolean;     // 与最终答案矛盾
}

export interface ThoughtStep {
  step_index: number;
  turn_index: number;
  step_type: StepType;
  content: string;
  metadata: Record<string, any>;
  tool_name: string;
  tool_use_id: string;
  tokens: number;
  timestamp?: string;
  duration_ms?: number;
  // 行为可观测性增强字段
  reasoning_digest?: ReasoningDigest | null;
  decision_trace?: DecisionTrace | null;
  state_evolution?: StateEvolution | null;
  error_trace?: ErrorTrace | null;
  // v0.11.0：client-side 合成的 OTel 视图
  otel?: OtelStepView | null;
}

export interface TurnCoT {
  turn_index: number;
  user_query: string;
  steps: ThoughtStep[];
  tool_calls: string[];
  strategy_shifts: number;
  thinking_depth: number;
  total_steps: number;
  has_error_recovery: boolean;
  final_response: string;
  usage: {
    input_tokens: number;
    output_tokens: number;
    cache_creation_input_tokens: number;
    cache_read_input_tokens: number;
  };
  complexity_score: number;
  turn_start_time?: string;
  turn_duration_ms?: number;          // v0.14.5: 活跃时长（剔除 >5min idle gap）
  turn_wallclock_span_ms?: number;    // v0.14.5: wall-clock 总跨度（含 idle）
  turn_idle_ms?: number;              // v0.14.5: 被剔除的 idle 累计
  // v0.14.2: 由 beforeSubmitPrompt / stop hook 直接给的真值时间戳（毫秒）。
  // 比 turn_start_time / turn_duration_ms 更可信，前端优先用这两个。
  turn_start_ms_observed?: number;
  turn_end_ms_observed?: number;
  turn_duration_ms_observed?: number;
  cot_summary?: string;
  // ── 子会话级字段（由 cot_extractor 填充，一个 turn = 一次用户→AI 交互 = 一个"子会话"）──
  interaction_summary?: string;         // 子会话一句话摘要（来自 user_query 首行）
  turn_quality_score?: number;          // 本轮质量分 0~1（response-verifier 风格的单 turn 指标）
  quality_signals?: {
    has_final_response?: boolean;
    error_recovery_count?: number;
    strategy_shifts?: number;
    score?: number;
    [k: string]: any;
  };
  // 子 session 标记（合并视图中使用，legacy）
  _sub_session_id?: string;
  _interaction_time?: string;
  _interaction_topic?: string;
  // v0.11.0：client-side 合成的 OTel 视图（turn 级 root span + token + finish reasons + ...）
  otel?: OtelTurnView | null;
  // v0.11.0：response-verifier / turn quality 派生的评估指标
  eval?: OtelEval | null;
}

// 单条 todo（顺序保留 & 状态完整）
export interface TodoItem {
  id?: string;
  content: string;
  status: 'pending' | 'in_progress' | 'completed' | 'cancelled' | string;
  idx: number;
}

// 两次 TodoWrite 之间的 status 迁移
export interface PlanDiff {
  newly_completed: { id?: string; content?: string }[];   // X → completed
  newly_started: { id?: string; content?: string }[];     // X → in_progress
  newly_added: { id?: string; content?: string; status?: string }[];
  removed: { id?: string; content?: string }[];
  status_changes: { id?: string; content?: string; from?: string; to?: string }[];
}

// Plan 演进快照：从 TodoWrite 每次调用抽取，串起来就是 agent 的 L4 计划层
// 时间线，补「CoT 里纯工具但看不到 plan 演进」的盲点。
//
// v0.10.0 增量：
// - todos：完整顺序的 todo 列表（带 id/content/status/idx）
// - diff： 与上一次同 session 快照的 status 迁移
// v0.14.5: turn 上的多种时长字段
//   turn_duration_ms          : 活跃时长（剔除 idle gap）—— 前端默认展示
//   turn_wallclock_span_ms    : wall-clock 总跨度（含 idle）—— tooltip 展示
//   turn_idle_ms              : 被剔除的 idle 累计
//   turn_duration_ms_observed : hook 真值（beforeSubmitPrompt → stop）—— 优先级最高


export interface PlanSnapshot {
  at_step: number;           // 全局 step_index
  turn_index: number;
  timestamp?: string | null;
  in_progress: string[];
  completed: string[];
  pending: string[];
  cancelled: string[];
  total: number;
  todos?: TodoItem[];                 // v0.10.0
  diff?: PlanDiff | null;             // v0.10.0
  snapshot_index?: number;            // v0.10.0：全局第几次 TodoWrite
  // v0.14.3：plan 滞后推断（agent 没及时打勾 → 后端补一个推断完成清单）
  inferred_completed_ids?: string[];
  lag_steps_to_turn_end?: number;
  is_likely_stale?: boolean;
  stale_reason?: 'lag_too_many_steps' | 'turn_finalized' | 'final_response_signal' | string | null;
}

// v0.10.0：Cursor 模式转换（plan / agent / debug / ask）
export type ModeId = 'plan' | 'agent' | 'debug' | 'ask' | string;
export type ModeTrigger = 'switch_mode' | 'create_plan' | 'implicit_back_to_agent' | string;

export interface ModeTransition {
  at_step: number;
  turn_index: number;
  target_mode_id: ModeId;
  explanation?: string | null;
  timestamp?: string | null;
  prev_mode_id?: ModeId | null;
  trigger: ModeTrigger;
}

// CreatePlan 工具调用产出的正式 plan 文档
export interface PlanProposal {
  at_step: number;
  turn_index: number;
  name: string;
  overview: string;
  plan: string;             // markdown 全文
  timestamp?: string | null;
}

// cot-stream.js 实时事件合并统计
export interface ObservedEvents {
  events_total: number;              // events.jsonl 总条数
  injected: number;                  // 成功回灌到 tool_execution 的条数
  tool_executions_total: number;     // 整个 session 的 tool_execution 总数
  shell_events: number;
  mcp_events: number;
  file_edit_events?: number;         // v0.9.0: afterFileEdit 事件总数
  file_edit_injected?: number;       // v0.9.0: 成功回灌到 Write/StrReplace tool_execution 的条数
}

// v0.9.0: L5 Execution Trace —— 临时脚本/文件产物追踪
export type FileOpKind = 'create' | 'modify' | 'delete';

export interface FileEditEvent {
  step_index: number;
  turn_index: number;
  timestamp?: string | null;
  kind: FileOpKind;
  tool_name: string;            // Write / StrReplace / Delete / MultiEdit / Edit
  edits_count: number;
  added_lines: number;
  removed_lines: number;
  content_chars: number;
}

export type ScriptLifecycle =
  | 'created'
  | 'created_modified'
  | 'created_executed'
  | 'created_executed_deleted'
  | 'created_modified_executed'
  | 'created_deleted'
  | 'modified_only'
  | 'deleted_only'
  | 'modified';

export interface ScriptArtifact {
  path: string;
  basename: string;
  extension: string;            // .py / .cjs / .md / ...
  language: string;             // python / javascript / markdown / ...
  is_temp: boolean;             // 启发式：临时验证脚本 / 草稿
  purpose_hint?: string;        // 文件名 / 首行注释推断的用途
  lifecycle: ScriptLifecycle;
  first_seen_step: number;
  last_seen_step: number;
  created_at_step?: number | null;
  deleted_at_step?: number | null;
  edit_count: number;           // Write + StrReplace + MultiEdit
  delete_count: number;
  executed_at_steps: number[];  // Shell 命令里反查到的执行 step_index
  total_added_lines: number;
  total_removed_lines: number;
  last_content_chars: number;
  timeline: FileEditEvent[];
}

export interface ScriptStats {
  total_artifacts: number;
  total_writes: number;
  total_strreplaces: number;
  total_deletes: number;
  total_executions: number;
  temp_scripts: number;
  executed_temp_scripts: number;
  deleted_temp_scripts: number;
  extensions: Record<string, number>;
  languages: Record<string, number>;
}

// 写入 step.metadata.file_op 的载荷（前端只读消费）
export interface FileOpMeta {
  kind: FileOpKind;
  tool_name: string;
  path: string;
  basename: string;
  extension: string;
  language: string;
  is_temp: boolean;
  edits_count: number;
  added_lines: number;
  removed_lines: number;
  content_chars: number;
  artifact_key: string;
}

// 写入 Shell step.metadata.executed_artifact 的载荷
export interface ExecutedArtifactMeta {
  path: string;
  basename: string;
  language: string;
  extension: string;
  is_temp: boolean;
  artifact_key: string;
}

// v0.8.0: LLM / RAG / Web Search 调用分类（hybrid：白名单 + 启发式）
// 由 cot_invocation_classifier.py 在后端打标，前端只是消费。
export type InvocationCategory = 'llm_call' | 'rag_query' | 'web_search';

export interface InvocationStats {
  llm_calls: number;
  rag_queries: number;
  web_searches: number;
  llm_call_distribution: Record<string, number>;
  rag_query_distribution: Record<string, number>;
}

// 后端真正写到 metadata 上的 v0.8.0 字段（运行时可选；类型上保持 any 兼容
// 现有代码，但加上下列 alias 方便 IDE 跳转 / autocomplete）
export interface InvocationMetadataExt {
  invocation_category?: InvocationCategory;
  prompt_preview?: string;
  prompt_full_chars?: number;
  recall_preview?: string;
  // 召回内容拿不到时的解释（来自 cot_invocation_classifier.diagnose_recall_unavailable）
  recall_unavailable_reason?: string;
  // _propagate_invocation_to_executions 把 decision 的 tool_input 也复制到 execution
  decision_tool_input?: any;
  // v0.8.2 起前端能用的更多 metadata 字段
  // tool_decision: 一行版 tool_input 摘要
  input_summary?: string;
  // user_input: text / image / tool_result 等
  block_type?: string;
  // final_response: 是否启发式推断的 final（Cursor 不返 stop_reason 时）
  inferred_final?: boolean;
  inferred_reason?: string;
  // tool_execution: cot-stream.js 事件墙钟时间
  observed_at_ms?: number;
  output_tokens?: number;
  // v0.9.0: L5 Execution Trace —— 文件操作元数据
  file_op?: FileOpMeta;
  executed_artifact?: ExecutedArtifactMeta;
  // v0.10.0：Plan / Mode 元数据（注入到 TodoWrite / SwitchMode / CreatePlan 的 step.metadata）
  plan_snapshot_idx?: number;
  plan_diff?: PlanDiff;
  plan_total?: number;
  plan_completed_count?: number;
  plan_in_progress_count?: number;
  plan_pending_count?: number;
  mode_switch?: {
    target_mode_id: ModeId;
    prev_mode_id?: ModeId | null;
    explanation?: string | null;
    trigger?: ModeTrigger;
  };
  plan_proposal?: {
    name?: string;
    overview_preview?: string;
    plan_chars?: number;
  };
}

export interface SessionCoT {
  session_id: string;
  transcript_path: string;
  extracted_at: string;
  turns: TurnCoT[];
  total_tool_calls: number;
  total_strategy_shifts: number;
  total_thinking_steps: number;
  tool_call_distribution: Record<string, number>;
  avg_steps_per_turn: number;
  avg_complexity: number;
  // 父子 session 元数据
  is_parent?: boolean;
  sub_sessions?: SubSessionInfo[];
  // v0.7.0：Plan 演进 + 实时事件观测
  plan_timeline?: PlanSnapshot[];
  observed_events?: ObservedEvents | null;
  // v0.8.0：LLM / RAG / Web Search 调用聚合
  invocation_stats?: InvocationStats | null;
  // v0.9.0：L5 Execution Trace —— 临时脚本与文件产物
  script_artifacts?: ScriptArtifact[];
  script_stats?: ScriptStats | null;
  // v0.10.0：模式转换 + Plan 文档
  mode_transitions?: ModeTransition[];
  plan_proposals?: PlanProposal[];
  // v0.11.0：client-side 合成的 OTel GenAI 视图（顶层）
  otel_view?: OtelSessionView | null;
  resource_attributes?: OtelResourceAttributes | null;
  // v0.14.2：Cursor IDE 真值生命周期（sessionStart / End / stop / beforeSubmitPrompt /
  // beforeTabFileRead / afterTabFileEdit hook 抽出来的事实）
  session_meta?: SessionLifecycleMeta | null;
  // v0.14.2：用户在 IDE 里手动操作（区别于 agent 的工具调用）
  user_activity?: UserActivityEntry[];
  // v0.15.0：Claude / Cursor 区分标识 + Claude 独有时间线
  // agent_type 由后端 cot_extractor._detect_agent_type 自动判定。
  // 前端按 agent 枚举切换徽章颜色和图标。
  // 'claude' / 'codex' / 'cursor' / 'codebuddy' / 'unknown'
  agent_type?: 'claude' | 'codex' | 'cursor' | 'codebuddy' | 'unknown' | string | null;
  // 以下五条 Claude 专属时间线由 cot_extractor / claude_stream_hook 注入；
  // Cursor session 始终为空数组（不打破 schema）。
  subagent_timeline?: SubagentEvent[];
  permission_events?: PermissionEvent[];
  compact_events?: CompactEvent[];
  notification_events?: NotificationEvent[];
  environment_events?: EnvironmentEvent[];
}

// ─── v0.15.0 Claude 专属时间线类型 ─────────────────────────────
//
// 这些字段都是来自 transcript 原生数据 / Claude hooks 的事实，前端只读消费。
// transcript-first 原则：只挂 Claude 真实发生过的事件，不做推断。

// Task 子代理（isSidechain 消息族）
export interface SubagentEvent {
  t_ms: number;
  sub_agent_id: string;            // 子 agent 的 session/uuid
  parent_tool_use_id?: string;     // 触发它的 Task 调用 tool_use_id
  agent_type?: string;              // 'generalPurpose' | 'explore' | 'shell' | ...
  prompt_preview?: string;          // 主 agent 给子 agent 的初始指令前 N 字符
  model?: string;                   // 子 agent 用的 model（可能跟主 agent 不同）
  step_count?: number;              // 子 agent 跑了多少步
  tool_calls?: number;              // 子 agent 调了多少次工具
  summary?: string;                 // 子 agent 返回给主 agent 的最终摘要
  duration_ms?: number;
  status?: 'running' | 'completed' | 'failed' | 'cancelled' | string;
}

// permission-mode 切换（Claude transcript 顶层 type='permission-mode'）
export interface PermissionEvent {
  t_ms: number;
  mode: 'plan' | 'acceptEdits' | 'bypassPermissions' | 'default' | string;
  prev_mode?: string | null;
  source?: 'user' | 'auto' | 'hook' | string;
  reason?: string;
}

// 上下文压缩（PreCompact / SubagentCompact hook）
export interface CompactEvent {
  t_ms: number;
  phase?: 'before' | 'after' | string;
  trigger?: 'auto' | 'manual' | 'subagent_threshold' | string;
  source?: string;
  turn_index?: number | null;
  before_tokens?: number;
  after_tokens?: number;
  saved_tokens?: number;
  summary_chars?: number;
  summary?: string;
  summary_preview?: string;
}

// Notification hook（用户级通知）
export interface NotificationEvent {
  t_ms: number;
  kind: 'Notification' | 'TeammateIdle' | 'StopFailure' | string;
  message: string;
  tool_name?: string;
  tool_use_id?: string;
}

// v0.15.1：第 5 条时间线 environment_events——IDE / 环境层事件，跟 agent
// 行为不直接相关但对回放上下文极有价值。
//   * CwdChanged          工作目录切换
//   * FileChanged         外部对文件做了改动（非 agent 工具）
//   * WorktreeCreate/Remove git worktree 操作
//   * ConfigChange        settings 改了哪个 key
//   * InstructionsLoaded  全局/项目指令文件被加载
//   * Setup               Claude Code 启动参数
export type EnvironmentEventKind =
  | 'CwdChanged' | 'FileChanged'
  | 'WorktreeCreate' | 'WorktreeRemove'
  | 'ConfigChange' | 'InstructionsLoaded' | 'Setup'
  | string;

export interface EnvironmentEvent {
  t_ms: number;
  kind: EnvironmentEventKind;
  // CwdChanged / ConfigChange 共享
  before?: string | null;
  after?: string | null;
  // FileChanged
  path?: string;
  change_kind?: string;
  is_user_initiated?: boolean;
  // WorktreeCreate/Remove
  worktree_path?: string;
  branch?: string;
  // ConfigChange
  key?: string;
  // InstructionsLoaded
  instruction_files?: string[];
  // Setup
  setup_args?: any;
  claude_version?: string;
  // 兜底：未识别字段
  details?: Record<string, any>;
}

// v0.14.2：sessionStart / End hook 给出的会话级元数据
export interface SessionLifecycleMeta {
  cursor_version?: string;
  user_email?: string;
  workspace_roots?: string[];
  session_start_ms_observed?: number;
  session_end_ms_observed?: number;
  session_duration_ms_observed?: number;
  transcript_path?: string;
  hook_events_observed?: Record<string, number>;
}

// v0.14.2：用户活动时间线（不是 agent 的工具调用！）
//   kind = 'submit_prompt' → 用户回车
//   kind = 'tab_read'      → 用户在 IDE 里点开了某个文件查看
//   kind = 'tab_edit'      → 用户**手动**修改了某个文件（关键质量信号）
export interface UserActivityEntry {
  kind: 'submit_prompt' | 'tab_read' | 'tab_edit' | '_truncated';
  t: number;                      // wall-clock ms
  file_path?: string;
  prompt_chars?: number;
  prompt_preview?: string;
  edits_count?: number;
  added_lines?: number;
  removed_lines?: number;
  generation_id?: string;
  model?: string;
  note?: string;                  // _truncated 时用
}

// ─── v0.11.0 OpenTelemetry GenAI 视图（client-side 合成） ──────

// 单个 OTel span 共享字段
export interface OtelTraceContext {
  trace_id: string;       // 32 hex chars
  span_id: string;        // 16 hex chars
  parent_span_id: string; // 16 hex chars
}

// gen_ai.usage —— token + cost
export interface OtelTokenUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cost_usd: number | null;
  /**
   * cost_usd 为 null 时的解释（也用于"可信度"判定）：
   *   - 'ok'                            → 正常算出
   *   - 'ok_true_usage_per_call'        → v0.20.7 Claude per-message API 真值 ✓
   *   - 'ok_apportioned_from_turn_real' → v0.20.7 Cursor/CodeBuddy turn 真值按 char 分摊 ≈
   *   - 'shared_with_anchor'            → v0.20.7 同 message 的非首 step (避免重复计数)
   *   - 'unknown_model'                 → model 未配置
   *   - 'no_pricing'                    → model 已知但 pricing 表里没有
   *   - 'non_llm_step'                  → 这个 step 不经过 LLM（host / user / 合成）
   */
  cost_reason?:
    | 'ok' | 'ok_true_usage_per_call' | 'ok_apportioned_from_turn_real'
    | 'shared_with_anchor' | 'unknown_model' | 'no_pricing' | 'non_llm_step'
    | string;
  currency: string;
  is_estimate: boolean;
  model_key: string | null;
  /**
   * v0.20.7 新增：token 数据的"出处"，用于前端展示可信度图标：
   *   - 'transcript_per_message'  → Claude per-call API 原生真值
   *   - 'turn_real_apportioned'   → 来自 turn 真值按字符比例的分摊
   *   - 'shared_with_anchor'      → 同 message 拆出的非首 step（已并入 anchor）
   *   - 'char_estimate'           → char/4 启发式（兜底）
   *   - 'non_llm'                 → 非 LLM step
   *   - 'unknown'                 → 未标注（老数据兼容）
   */
  source?:
    | 'transcript_per_message' | 'turn_real_apportioned' | 'shared_with_anchor'
    | 'char_estimate' | 'non_llm' | 'unknown' | string;
  cache_read_tokens?: number;
  cache_creation_tokens?: number;
}

// gen_ai.input.messages / gen_ai.output.messages
export interface OtelMessagePart {
  type: string;            // 'text' | 'image' | 'tool_call' | ...
  content: string;
}

export interface OtelStructuredMessage {
  role: string;            // 'user' | 'assistant' | 'system' | 'tool' | ...
  parts: OtelMessagePart[];
  finish_reason?: string;  // assistant 输出有这个字段
}

// OpenInference RetrievalSpan documents
export interface OtelRetrievalDocument {
  id: string;
  content: string;
  score: number | null;
  metadata: Record<string, any>;
}

// gen_ai 评估指标
export interface OtelEval {
  metric_name: string;
  score: number | null;
  label: string | null;
  details?: Record<string, any>;
  scores?: Record<string, number>;
  summary?: string;
  checked_at?: string;
}

/**
 * step 的语义分类（v0.11.1+）：
 *   - 'llm_call'    → thinking / decision / final_response 等真正经过 LLM 的步骤
 *   - 'host_tool'   → tool_execution（Shell / Read / Write / Grep ...），Cursor host runtime 在跑，**不**走 LLM
 *   - 'user_input'  → 客户端输入
 *   - 'agent_event' → strategy_shift / mode_transition / plan_update 等合成事件
 */
export type OtelStepKind = 'llm_call' | 'host_tool' | 'user_input' | 'agent_event' | string;

// model 字段的来源（v0.11.1+），用来在 model = unknown 时给出修复 hint
export type OtelModelSource =
  | 'events'      // v0.11.2：来自 cot-stream.js 实时 hook events.jsonl（自动检测，0 硬编码）
  | 'env'         // 来自 COT_DEFAULT_MODEL 环境变量（兜底）
  | 'transcript'  // 从 transcript metadata 扫到
  | 'host'        // host_tool / 非 LLM step
  | 'client'      // user_input
  | 'synthetic'   // agent_event
  | 'unknown'     // 都没拿到
  | string;

// v0.11.2：来自 Cursor hook 的真实 token 用量（cache-aware）
export interface OtelActualTokenUsage {
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  agent_response_count: number;
  model: string;
  model_key: string | null;
  cost_usd: number | null;
  cost_breakdown?: {
    non_cache_input_usd?: number;
    cache_write_usd?: number;
    cache_read_usd?: number;
    output_usd?: number;
    non_cache_input_tokens?: number;
  } | null;
  // 没 cache 折扣的全价（仅用于前端对比展示「cache 帮你省了多少」）
  full_price_cost_usd?: number | null;
  cost_reason?: string;
  is_estimate: false;
  source: string;
}

// v0.11.2：来自 cot-stream hook 的客户端运行时上下文
export interface OtelClientRuntime {
  cursor_version?: string | null;
  user_email?: string | null;
  events_count?: number | null;
  events_path?: string | null;
  model_distribution?: Record<string, number> | null;
}

// step 级 OTel 视图
export interface OtelStepView extends OtelTraceContext {
  kind: string;                       // 'internal' | 'server' | 'client'
  step_kind?: OtelStepKind;           // v0.11.1：四档分类
  operation_name: string;             // 'execute_tool' | 'chat' | 'invoke_agent' | ...
  model: string;
  provider: string;
  model_source?: OtelModelSource;     // v0.11.1
  // v0.17：与 OtelTurnView 对齐（OtelPanel 把 step 也传给 TokenCostCard）
  models_seen?: string[];
  finish_reason: string;
  finish_reasons: string[];
  token_usage: OtelTokenUsage;
  input_messages: OtelStructuredMessage[];
  output_messages: OtelStructuredMessage[];
  retrieval_documents: OtelRetrievalDocument[];
  attributes: Record<string, any>;
}

// turn 级 OTel 视图
export interface OtelTurnView {
  trace_id: string;
  span_id: string;
  parent_span_id: string;
  operation_name: string;            // 'invoke_agent'
  agent_name: string;
  model: string;
  provider: string;
  model_source?: OtelModelSource;    // v0.11.1
  // v0.17：多模型 session 里 turn 内可能跨多个 model（少见但合法），
  // OtelPanel 用它做 TokenCost 卡片的 model 切换提示。
  models_seen?: string[];
  finish_reasons: string[];
  token_usage: OtelTokenUsage;
  conversation_id: string;
  turn_index: number;
  request_params: {
    temperature: number | null;
    top_p: number | null;
    max_tokens: number | null;
    seed: number | null;
    stop_sequences: string[] | null;
  };
  response: {
    id: string | null;
    model: string;
  };
}

// hint 给前端展示「为啥 unknown / 怎么补」
export interface OtelHint {
  level: 'warn' | 'info' | 'error' | string;
  code: string;
  message: string;
}

// session 级 OTel 视图
export interface OtelSessionView {
  schema: string;
  trace_id: string;
  root_span_id: string;
  service: { name: string; version: string };
  model: string;
  provider: string;
  model_source?: OtelModelSource;    // v0.11.1
  // v0.13.x：本 session 内出现过的所有 model（多模型场景）
  models_seen?: string[];
  // 模型切换时间轴（renderer.log 解出的 buildRequestedModel 时间序列）
  model_timeline?: Array<{ t_ms: number; model: string }>;
  agent_name?: string;               // v0.11.1
  session_id: string;
  conversation_id: string;
  totals: {
    turns: number;
    steps: number;
    tool_calls: number;
    input_tokens: number;
    output_tokens: number;
    cost_usd: number | null;
  };
  token_usage: OtelTokenUsage;
  // v0.11.2：cache-aware 的真值 token 用量（来自 cot-stream events.jsonl）
  actual_token_usage?: OtelActualTokenUsage | null;
  actual_cost_usd?: number | null;
  client_runtime?: OtelClientRuntime | null;
  eval: OtelEval | null;
  request_params: {
    temperature: number | null;
    top_p: number | null;
    max_tokens: number | null;
    seed: number | null;
    stop_sequences: string[] | null;
    _note?: string;
  };
  missing_signals: string[];
  hints?: OtelHint[];                // v0.11.1
  generated_at_ms: number;
}

// OTel resource attributes（service.* / host.* / telemetry.sdk.*）
export interface OtelResourceAttributes {
  'service.name': string;
  'service.version': string;
  'service.namespace'?: string;
  'deployment.environment'?: string;
  'host.name'?: string;
  'host.arch'?: string;
  'host.os.type'?: string;
  'host.os.version'?: string;
  'telemetry.sdk.name'?: string;
  'telemetry.sdk.language'?: string;
  'telemetry.sdk.version'?: string;
  'process.runtime.name'?: string;
  'process.runtime.version'?: string;
  [key: string]: any;
}

export interface ResponseReport {
  session_id: string;
  checked_at: string;
  scores: Record<string, number>;
  details: Record<string, any>;
  summary: string;
}

export interface TurnEvalScore {
  key: string;
  name?: string;
  label_zh: string;
  label_en: string;
  score: number;
  weight: number;
  threshold: number;
  passed: boolean;
  reason_zh: string;
  reason_en: string;
  reason?: string;
}

export interface TurnEvalTaskProfile {
  primary?: string;
  labels?: string[];
  confidence?: number;
  rationale?: string;
  [key: string]: any;
}

export interface TurnEvalAssertionResult {
  key: string;
  name?: string;
  label_zh?: string;
  label_en?: string;
  type?: string;
  source?: string;
  category?: string;
  severity?: 'critical' | 'high' | 'medium' | 'low' | string;
  score: number;
  threshold?: number;
  passed: boolean;
  skipped?: boolean;
  binary?: boolean;
  quantitative?: boolean;
  reason?: string;
  reason_zh?: string;
  reason_en?: string;
  evidence?: any;
  [key: string]: any;
}

export interface TurnEvalAssertionGroup {
  key: string;
  label: string;
  passed: number;
  total: number;
  items: TurnEvalAssertionResult[];
}

export interface TurnEvalPipelineInfo {
  automation_ready?: {
    ready?: boolean;
    dataset_case_shape?: string;
    recommended_trials?: number;
    pass_threshold?: number;
    [key: string]: any;
  };
  ab_testing?: {
    unit?: string;
    primary_metric?: string;
    secondary_metrics?: string[];
    dimensions?: string[];
    [key: string]: any;
  };
  [key: string]: any;
}

export interface TurnEvalPanelDimension {
  key: string;
  label_zh: string;
  label_en: string;
  verdict: string;
  review?: string;
  source?: string;
  allowed?: string[];
  [key: string]: any;
}

export interface TurnEvalPanel {
  mode?: string;
  method?: string;
  overall_verdict?: string;
  core_dimensions?: TurnEvalPanelDimension[];
  diagnostics?: {
    efficiency?: Record<string, any>;
    reliability?: Record<string, any>;
    [key: string]: any;
  };
  safety_gate?: {
    status?: string;
    items?: Array<{ key: string; label_zh: string; hit: boolean; detail?: string }>;
    [key: string]: any;
  };
  assertion_pass_rate?: number;
  notes?: string[];
  [key: string]: any;
}

export interface TurnEvalReport {
  report_id: string;
  session_id: string;
  turn_index: number;
  created_at: string;
  passed: boolean;
  overall_score: number;
  quality_score: number;
  eval_version?: string;
  assertion_pass_rate?: number;
  task_profile?: TurnEvalTaskProfile;
  assertion_set?: {
    version?: string;
    source?: string;
    default_assertions?: string[];
    specialized_assertions?: string[];
    total_assertions?: number;
    [key: string]: any;
  };
  assertion_results?: TurnEvalAssertionResult[];
  assertion_groups?: TurnEvalAssertionGroup[];
  critical_failures?: TurnEvalAssertionResult[];
  judge?: {
    status?: string;
    provider?: string;
    model?: string;
    reason?: string;
    config_path?: string;
    [key: string]: any;
  };
  pipeline?: TurnEvalPipelineInfo;
  eval_panel?: TurnEvalPanel;
  score_breakdown?: {
    score?: number;
    method?: string;
    assertion_pass_rate?: number;
    judge_used?: boolean;
    weights?: Record<string, number>;
    components?: TurnEvalScore[];
    notes?: string[];
    [key: string]: any;
  };
  score_formula?: {
    description_zh?: string;
    description_en?: string;
    weights?: Record<string, number>;
  };
  metrics: {
    user_query?: string;
    final_response_chars?: number;
    has_final_response?: boolean;
    input_tokens?: number;
    output_tokens?: number;
    cache_read_tokens?: number;
    cache_write_tokens?: number;
    total_tokens?: number;
    duration_ms?: number | null;
    tokens_per_second?: number | null;
    tokens_per_second_basis?: string;
    tool_count?: number;
    unique_tool_count?: number;
    tool_calls?: string[];
    step_count?: number;
    step_type_counts?: Record<string, number>;
    strategy_shifts?: number;
    plan_update_count?: number;
    error_count?: number;
    tool_error_count?: number;
    error_terms?: string[];
    pii_or_secret_risk?: boolean;
    pii_or_secret_hits?: string[];
    trace_fields_present?: Record<string, boolean>;
    missing_duration_steps?: number;
    [key: string]: any;
  };
  scores?: TurnEvalScore[];
  summary?: {
    zh?: string;
    en?: string;
    strongest?: Array<{ key: string; label_zh: string; label_en: string; score: number }>;
    needs_attention?: Array<{ key: string; label_zh: string; label_en: string; score: number }>;
    tokens_per_second?: number | null;
    [key: string]: any;
  };
  ab_ready_dimensions?: string[];
  lineage?: {
    letsgoagenteval_retained?: string[];
    implementation_note_zh?: string;
    implementation_note_en?: string;
    [key: string]: any;
  };
  source?: Record<string, boolean>;
}
