import { useState, useMemo, useEffect, useRef } from 'react';
import type { SessionCoT, SessionOverview, ResponseReport, TurnCoT, ThoughtStep, TurnEvalReport } from '../types';
import { ClaudeToolStatsCompact } from './ClaudeOtelPanel';
import {
  api,
  saveDownloadedFile,
  type ClaudeOtelData,
  type TraceEvent,
} from '../hooks/api';

// 思考 → 工具 → 更多思考 —— 用 step_index 的原顺序渲染，
// 让 tool_decision/tool_execution 紧跟在触发它的 thinking 步骤之后。

interface Props {
  cot: SessionCoT;
  session: SessionOverview | null;
  report: ResponseReport | null;
  selectedNode: SelectedNode | null;
  onSelectNode: (node: SelectedNode) => void;
  onEvalTurn?: (turn: TurnCoT, cot: SessionCoT) => void;
  onReferenceEvalTurn?: (turn: TurnCoT, cot: SessionCoT) => void;
  turnEvalReports?: Record<string, TurnEvalReport>;
  turnEvalLoadingKey?: string | null;
  liveCritic?: any | null;
  liveCriticTurns?: Record<string, any>;
  abBaseline?: TurnRef | null;
  abCandidate?: TurnRef | null;
}

type TurnRef = { session_id: string; turn_index: number };
type AbTurnRole = 'baseline' | 'candidate' | 'both' | null;

export type SelectedNode =
  | { kind: 'session'; cot: SessionCoT; session: SessionOverview | null; report: ResponseReport | null }
  | { kind: 'turn'; turn: TurnCoT; cot: SessionCoT }
  | { kind: 'step'; step: ThoughtStep; turn: TurnCoT };

function sameTurnRef(ref: TurnRef | null | undefined, sessionId: string, turnIndex: number): boolean {
  return !!ref && ref.session_id === sessionId && ref.turn_index === turnIndex;
}

function getAbTurnRole(
  sessionId: string,
  turnIndex: number,
  baseline?: TurnRef | null,
  candidate?: TurnRef | null,
): AbTurnRole {
  const isBaseline = sameTurnRef(baseline, sessionId, turnIndex);
  const isCandidate = sameTurnRef(candidate, sessionId, turnIndex);
  if (isBaseline && isCandidate) return 'both';
  if (isBaseline) return 'baseline';
  if (isCandidate) return 'candidate';
  return null;
}

function AbTurnStamp({ role, compact = false }: { role: AbTurnRole; compact?: boolean }) {
  if (!role) return null;
  const label = role === 'baseline' ? 'BASE' : role === 'candidate' ? 'CANDIDATE' : 'BASE + CANDIDATE';
  const title = role === 'baseline'
    ? 'A/B baseline turn'
    : role === 'candidate'
      ? 'A/B candidate turn'
      : 'This turn is both A/B baseline and candidate';
  return (
    <span className={`ab-turn-stamp ab-turn-stamp-${role} ${compact ? 'ab-turn-stamp-compact' : ''}`} title={title}>
      {label}
    </span>
  );
}

// ─── v0.11.2 OTel inline 工具函数 ────────────────────────
function fmtOtelCost(usd: number | null | undefined): string {
  if (usd == null) return '';
  if (Math.abs(usd) < 0.001) return `$${usd.toExponential(1)}`;
  if (Math.abs(usd) < 1) return `$${usd.toFixed(3)}`;
  return `$${usd.toFixed(2)}`;
}
function fmtOtelTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}
function shortOtelModel(m?: string | null): string {
  if (!m || m === 'unknown') return 'unknown';
  return m.replace(/^claude-/, 'cla-').replace(/^gpt-/, 'gpt-').slice(0, 16);
}

// ─── 耗时格式化 ──────────────────────────────────────────
function fmtDuration(ms?: number): string {
  if (ms == null || ms <= 0) return '';
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(2)}s`;
  return `${(ms / 60000).toFixed(1)}min`;
}

// ─── v0.20.7: 每个 step 的 token 用量徽章 ───────────────
// 数据来源优先级：
//   1) step.otel.token_usage —— cot_otel_enricher 注入的标准 GenAI 视图
//      （Cursor / Claude / CodeBuddy 三家 extractor 全部已经在写）
//   2) step.metadata.output_tokens / step.metadata.input_tokens —— 原始
//      extractor 直接抄 transcript usage 字段时挂的兜底
//   3) step.tokens —— 早期老 cot.json 兼容
//
// 渲染规则：
//   * llm_call / 含 thinking / final_response / pre_tool_reasoning：
//       显示 ↓in ↑out · $cost（任一不为 0 才显示该项）
//   * tool_execution / 其他 host_tool / user_input 无 token：整段不渲染
//   * cost 为估算（is_estimate=true）时前缀 `~`，hover tooltip 说明原因
function _readStepTokens(step: ThoughtStep): {
  input: number; output: number; cost: number | null;
  is_estimate: boolean; cost_reason?: string; source?: string; model?: string;
  is_llm_call: boolean;
  is_shared: boolean;
  cache_read?: number; cache_creation?: number;
} | null {
  const otel: any = (step as any)?.otel;
  const tu = otel?.token_usage;
  if (tu) {
    const reason: string = tu.cost_reason || '';
    const source: string = tu.source || '';
    const input = Number(tu.input_tokens) || 0;
    const output = Number(tu.output_tokens) || 0;
    const rawCost = (typeof tu.cost_usd === 'number') ? tu.cost_usd : null;
    // cost 只在真正经过 LLM API 时展示（user_input / tool_execution 的
    // cost 字段总是 null+non_llm_step），但 input/output token 数本身对
    // 「user 提示词多大、工具结果多长（=下一次 LLM 的隐性输入）」也是
    // 重要信号，所以 non_llm_step 时只剥掉 cost、保留 token 数字。
    const is_llm = reason !== 'non_llm_step';
    const is_shared = source === 'shared_with_anchor';
    // v0.20.7: shared step 真值已归并到同 message 的 anchor，本身展示 0/0；
    // 为了让前端能区分"真的 0"和"shared 折叠"，保留 is_shared 标志。
    const cost = is_llm ? rawCost : null;
    if (input === 0 && output === 0 && !cost && !is_shared) return null;
    return {
      input, output, cost,
      is_estimate: !!tu.is_estimate,
      cost_reason: reason,
      source,
      model: otel.model,
      is_llm_call: is_llm,
      is_shared,
      cache_read: Number(tu.cache_read_tokens) || 0,
      cache_creation: Number(tu.cache_creation_tokens) || 0,
    };
  }
  // metadata 兜底（早期 extractor / 没跑过 enricher 的 cot.json）
  const md: any = step.metadata || {};
  const mdIn = Number(md.input_tokens) || 0;
  const mdOut = Number(md.output_tokens) || 0;
  const legacy = Number((step as any).tokens) || 0;
  if (mdIn || mdOut) {
    return {
      input: mdIn, output: mdOut, cost: null, is_estimate: true,
      is_llm_call: true, is_shared: false, source: 'unknown',
    };
  }
  if (legacy > 0) {
    return {
      input: 0, output: legacy, cost: null, is_estimate: true,
      is_llm_call: true, is_shared: false, source: 'unknown',
    };
  }
  return null;
}


function _fmtTokenCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 10_000) return `${(n / 1000).toFixed(0)}K`;
  if (n >= 1_000) return `${(n / 1000).toFixed(1)}K`;
  return String(n);
}

function _fmtTokenCost(usd: number): string {
  if (Math.abs(usd) < 0.0001) return '$<0.0001';
  if (Math.abs(usd) < 0.01) return `$${usd.toFixed(4)}`;
  if (Math.abs(usd) < 1) return `$${usd.toFixed(3)}`;
  return `$${usd.toFixed(2)}`;
}

// v0.20.7: turn 级 token 用量解析 ── 跟 DetailPanel.readTurnUsage 同源。
// fallback 链：hook 真值 → 累加 step.otel.token_usage 里 LLM 调用 → enricher。
// 用于 Turn header 右上角那枚 ``14.2K`` 小徽章——之前直接读 turn.usage，
// Cursor afterAgentResponse hook 没触发时整个 turn 显示 0K 不出徽章。
function _readTurnTotalTokens(turn: TurnCoT): number {
  const u: any = turn.usage || {};
  const uIn = Number(u.input_tokens) || 0;
  const uOut = Number(u.output_tokens) || 0;
  if (uIn > 0 || uOut > 0) return uIn + uOut;

  // step LLM 累加（跟 SpanTree chip 上的 $0.341 等真值对齐）
  let llmTot = 0; let hasLlm = false;
  for (const s of turn.steps || []) {
    const stu: any = (s as any)?.otel?.token_usage;
    if (!stu || stu.cost_reason === 'non_llm_step') continue;
    llmTot += (Number(stu.input_tokens) || 0) + (Number(stu.output_tokens) || 0);
    hasLlm = true;
  }
  if (hasLlm && llmTot > 0) return llmTot;

  const o: any = (turn as any)?.otel?.token_usage || {};
  return (Number(o.input_tokens) || 0) + (Number(o.output_tokens) || 0);
}

function StepTokenBadge({ step }: { step: ThoughtStep }) {
  const t = _readStepTokens(step);
  if (!t) return null;
  // shared step：已折叠到上一步 anchor，chip 只显示一个折叠图标，避免每个
  // tool_decision 都重复 0/0 干扰阅读。
  if (t.is_shared) {
    return (
      <span
        className="tree-step-tokens-group tree-step-tokens-shared"
        title="该 step 与同一次 LLM message 的上一步合并显示，避免重复计算。"
      >
        <span className="tree-step-tokens-merged">↗ merged</span>
      </span>
    );
  }
  const tipParts: string[] = [];
  if (t.input)  tipParts.push(`input  ${t.input.toLocaleString()} tok`);
  if (t.output) tipParts.push(`output ${t.output.toLocaleString()} tok`);
  if (t.cache_read) tipParts.push(`cache_read ${t.cache_read.toLocaleString()} tok`);
  if (t.cache_creation) tipParts.push(`cache_write ${t.cache_creation.toLocaleString()} tok`);
  if (t.cost != null) {
    tipParts.push(`cost   $${t.cost.toFixed(6)} USD`);
  }
  if (t.model) tipParts.push(`model  ${t.model}`);
  if (!t.is_llm_call) {
    tipParts.push('(非 LLM 调用 — 这些 token 会作为下一次 LLM 调用的输入)');
  }
  const title = tipParts.join('\n');
  // 非 LLM step 的 token chip 用更淡的描边，跟 LLM step 视觉上分层。
  const dimCls = t.is_llm_call ? '' : ' tree-step-tokens-dim';
  return (
    <span className={`tree-step-tokens-group${dimCls}`} title={title}>
      {t.input > 0 && (
        <span className="tree-step-tokens-in">↓{_fmtTokenCount(t.input)}</span>
      )}
      {t.output > 0 && (
        <span className="tree-step-tokens-out">↑{_fmtTokenCount(t.output)}</span>
      )}
      {t.cost != null && t.cost > 0 && (
        <span className="tree-step-tokens-cost">
          {_fmtTokenCost(t.cost)}
        </span>
      )}
    </span>
  );
}

// ─── Turn 行程标签（Turn header 用）───────────────────────
// 分隔带已展示"子会话主题"（interaction_summary），
// Turn header 聚焦"这一轮执行了什么" —— 工具聚合 + 步骤数。
function getTurnLabel(turn: TurnCoT): { icon: string; label: string; sub: string } {
  const tools = turn.tool_calls.filter(Boolean);
  const uniqueTools = [...new Set(tools)];

  if (tools.length > 0) {
    const label = uniqueTools.length === 1
      ? `调用 ${uniqueTools[0]}`
      : `${uniqueTools[0]} + ${uniqueTools.length - 1} 个工具`;
    const sub = uniqueTools.length === 1
      ? `${tools.length} 次调用 · ${turn.total_steps} 步`
      : `${tools.length} 次 / ${uniqueTools.length} 种 · ${turn.total_steps} 步`;
    return { icon: '⚙️', label, sub };
  }

  if (turn.final_response) {
    return { icon: '📝', label: '直接回复', sub: `无工具 · ${turn.total_steps} 步` };
  }

  return { icon: '🔄', label: `Turn ${turn.turn_index}`, sub: `${turn.total_steps} 步` };
}

// ─── Step 类型配置 ────────────────────────────────────────
const STEP_CFG: Record<string, { icon: string; color: string; label: string }> = {
  user_input:           { icon: '💬', color: '#3b82f6', label: 'User Input' },
  tool_result_input:    { icon: '📥', color: '#6366f1', label: 'Tool Result' },
  thinking_inter:       { icon: '🧠', color: '#8b5cf6', label: 'Thinking' },
  thinking_intermediate:{ icon: '🧠', color: '#8b5cf6', label: 'Thinking' },
  thinking_explicit:    { icon: '🧠', color: '#8b5cf6', label: 'Thinking' },
  // v0.20.11: 决策说明 → 紫色大脑统一
  // 之前 pre_tool_reasoning 单独用 💡 黄色"决策说明"标签，但用户反馈这跟 🧠
  // thinking 的内容几乎重叠（都是 LLM 在调工具前的推理输出），看起来像两类
  // 不同事件其实是一类，反而干扰阅读。改成跟 thinking_inter/explicit 同款
  // 紫色大脑 + 'Thinking' label，再由 getStepNodeLabel 用首句内容覆盖标签
  // （比如 "我发现新加的 if (step.step_type =..."），让用户一眼读到推理内容。
  pre_tool_reasoning:   { icon: '🧠', color: '#8b5cf6', label: 'Thinking' },
  // v0.20.10: tool_decision 视觉上=一次 LLM API 调用（CC 源码 llm_request span
  // 等价），它有真实 input/output token；改用紫色大脑 + "LLM Thinking" 标签，
  // 让用户立刻读到"模型在推理然后决定调用工具"的语义；getStepNodeLabel
  // 仍会拼上 "→ <tool_name>"，渲染出 "🧠 LLM Thinking → Bash"。
  tool_decision:        { icon: '🧠', color: '#a78bfa', label: 'LLM Thinking' },
  tool_execution:       { icon: '⚙️', color: '#10b981', label: 'Tool Execution' },
  strategy_shift:       { icon: '🔄', color: '#f59e0b', label: 'Strategy Shift' },
  error_recovery:       { icon: '⚠️', color: '#ef4444', label: 'Error Recovery' },
  final_response:       { icon: '📝', color: '#06b6d4', label: 'Final Response' },
};

// v0.8.0: 调用分类视觉。后端打了 invocation_category 时覆盖默认配色。
// v0.20.11：tool_decision = LLM API 调用。所有 invocation 子类都共用同一套
// 主视觉（紫色大脑 + "LLM Thinking"），文案在 getStepNodeLabel 里统一拼成
// "LLM Thinking → Use Tool <tool>"，子类信息靠后缀的工具名识别即可。之前的
// 避免让用户误以为"金=LLM、青=RAG 是三类不同的事"。tool_execution 不读这张
// 表（getStepCfg 里 tool_execution 走自己的 host 视觉），所以这里不影响它。
const INVOCATION_CFG: Record<string, { icon: string; color: string; label: string }> = {
  llm_call:   { icon: '🧠', color: '#a78bfa', label: 'LLM Thinking' },
  rag_query:  { icon: '🧠', color: '#a78bfa', label: 'LLM Thinking' },
  web_search: { icon: '🧠', color: '#a78bfa', label: 'LLM Thinking' },
};

// v0.10.0: Plan 类工具单独视觉（让 TodoWrite/SwitchMode/CreatePlan 从普通工具流里
// 脱离出来 —— 它们不是"agent 在干活"，而是"agent 在 plan/调度任务"）
// 这三个工具在 SpanTree 上不再写"Tool Decision → TodoWrite"，而是各自有专属符号。
//
// v0.19.5：Claude Internal 用一对增量式 plan 工具——``TaskCreate`` 追加一条
// 任务、``TaskUpdate`` 改单条状态——跟 ``TodoWrite``（整列表覆盖）语义本来就
// 不同，保留各自工具名 + 各自 label（"Plan 新增" / "Plan 推进"），但用同金色
// 视觉，让用户一眼能识别"这都是 plan 类节点"。后端 ``_build_plan_timeline_from_
// task_tools`` 已经把 plan_total / plan_completed_count / plan_diff / plan_
// full_todos 写到这两类 step 的 metadata，所以下面的 Plan 标签 / diff 徽标 /
// DetailPanel 快照卡片都能直接复用同一套 metadata。
const PLAN_TOOL_CFG: Record<string, { icon: string; color: string; label: string; execIcon?: string; execLabel?: string }> = {
  TodoWrite:  { icon: '🗺️', color: '#fbbf24', label: 'Plan 更新', execIcon: '🗺️', execLabel: 'Plan 已记录' },
  TaskCreate: { icon: '🗺️', color: '#fbbf24', label: 'Plan 新增', execIcon: '🗺️', execLabel: 'Plan 已新增' },
  TaskUpdate: { icon: '🗺️', color: '#fbbf24', label: 'Plan 推进', execIcon: '🗺️', execLabel: 'Plan 已推进' },
  SwitchMode: { icon: '🔀', color: '#c084fc', label: '模式切换',  execIcon: '🔀', execLabel: '模式已切换' },
  CreatePlan: { icon: '📋', color: '#fbbf24', label: 'Plan 文档',  execIcon: '📋', execLabel: 'Plan 已提交' },
};

// 在指定 turn 列表里按顺序拢出某 invocation 类型的所有 tool_decision step。
// 用来支持"点 RAG 徽章直接跳到首个 RAG step"这种导航。
function collectInvocationSteps(
  turns: TurnCoT[],
  cat: 'llm_call' | 'rag_query' | 'web_search',
): { turn: TurnCoT; step: ThoughtStep }[] {
  const out: { turn: TurnCoT; step: ThoughtStep }[] = [];
  for (const t of turns) {
    for (const s of t.steps) {
      if (s.step_type !== 'tool_decision') continue;
      if (s.metadata?.invocation_category === cat) {
        out.push({ turn: t, step: s });
      }
    }
  }
  return out;
}

function getStepCfg(step: ThoughtStep) {
  if (step.step_type === 'tool_execution') {
    if (step.metadata?.is_error) {
      return { icon: '❌', color: '#ef4444', label: 'Tool Error' };
    }
    // Cursor transcript 不记录 tool_result —— cot_extractor 合成的占位节点
    if (step.metadata?.synthetic) {
      // 但如果是 plan 类工具，仍然给它专属视觉，不用 ◌
      const tool = step.tool_name || step.metadata?.tool_name;
      if (tool && PLAN_TOOL_CFG[tool]) {
        const c = PLAN_TOOL_CFG[tool];
        return { icon: c.execIcon || c.icon, color: c.color, label: c.execLabel || c.label };
      }
      return { icon: '◌', color: '#64748b', label: 'Tool Executed (no result)' };
    }
    // v0.20.11：RAG / Web / LLM 子调用的 tool_execution 用专属图标，
    // 跟普通工具的绿色齿轮区分开（注释见 line 332-339 的视觉决策说明）。
    // RAG 召回结果 → 📚 课本（"知识检索回来的资料"语义最贴）
    // Web 搜索结果 → 🌐
    // LLM 子调用响应 → 💬
    const cat = step.metadata?.invocation_category as string | undefined;
    if (cat === 'rag_query') {
      return { icon: '📚', color: '#10b981', label: 'RAG Result' };
    }
    if (cat === 'web_search') {
      return { icon: '🌐', color: '#10b981', label: 'Web Result' };
    }
    if (cat === 'llm_call') {
      return { icon: '💬', color: '#10b981', label: 'LLM Response' };
    }
  }
  // v0.10.0: Plan 类工具优先（TodoWrite / SwitchMode / CreatePlan）
  const tool = step.tool_name || step.metadata?.tool_name;
  if (tool && PLAN_TOOL_CFG[tool]) {
    const c = PLAN_TOOL_CFG[tool];
    if (step.step_type === 'tool_execution') {
      return { icon: c.execIcon || c.icon, color: c.color, label: c.execLabel || c.label };
    }
    return { icon: c.icon, color: c.color, label: c.label };
  }
  // v0.20.10: 调用分类覆盖现在只作用在 tool_decision 上 —— 因为它代表
  // "LLM 在思考 / 决定调哪个工具"，跟 INVOCATION_CFG 的"LLM Thinking · RAG /
  // 检索 / 子调用" 紫色大脑视觉天然匹配；而 tool_execution（host runtime
  // 真正执行工具）必须走 STEP_CFG 默认的绿色齿轮 ⚙️，否则 RAG 召回结果
  // 也会被涂成紫色，跟它的"host 执行"语义冲突。
  // 之前版本对 tool_execution 也套 INVOCATION_CFG，靠 getStepNodeLabel 在
  // label 层把文本改成"RAG 召回结果"——文字对但图标/颜色错。这里把分类
  // 视觉限制在 decision 上，让两类 step 视觉上各司其职。
  const cat = step.metadata?.invocation_category as string | undefined;
  if (cat && INVOCATION_CFG[cat] && step.step_type === 'tool_decision') {
    return INVOCATION_CFG[cat];
  }
  return STEP_CFG[step.step_type] || { icon: '•', color: '#64748b', label: step.step_type };
}

function getStepNodeLabel(step: ThoughtStep): string {
  const cfg = getStepCfg(step);
  const tool = step.tool_name || step.metadata?.tool_name;

  // v0.10.0: Plan 类工具不写"Tool Decision → TodoWrite"，而是直接装内容摘要
  if (tool && PLAN_TOOL_CFG[tool]) {
    // v0.19.5: TodoWrite / TaskCreate / TaskUpdate 三个 plan 工具都用同一套
    // plan_total / plan_completed_count / plan_snapshot_idx 元数据
    // （由后端 _build_plan_timeline / _build_plan_timeline_from_task_tools 统一回灌），
    // 所以这里的进度标签逻辑对它们共用——只是 cfg.label 自身已经"Plan 更新 /
    // Plan 新增 / Plan 推进"区分了，不需要再做分支。
    const isPlanSnapshot = (
      (tool === 'TodoWrite' || tool === 'TaskCreate' || tool === 'TaskUpdate')
      && step.step_type === 'tool_decision'
    );
    if (isPlanSnapshot) {
      const total = step.metadata?.plan_total ?? 0;
      const done = step.metadata?.plan_completed_count ?? 0;
      const ip = step.metadata?.plan_in_progress_count ?? 0;
      const idx = step.metadata?.plan_snapshot_idx;
      const idxLabel = typeof idx === 'number' ? ` #${idx + 1}` : '';
      // v0.14.3: 推断完成数（agent 没及时打勾时）
      const inferredIds = (step.metadata?.plan_inferred_completed as string[] | undefined) || [];
      const isStale = step.metadata?.plan_is_likely_stale === true;
      // "🗺️ Plan 更新 #5 · 3/8 完成 · 1 进行中"
      // 滞后情况："🗺️ Plan 更新 #29 · 4/9 完成 +⚡5 · ⚠ 滞后"
      let stat = '';
      if (total > 0) {
        stat = ` · ${done}/${total} 完成`;
        if (isStale && inferredIds.length > 0) {
          stat += ` +⚡${inferredIds.length}`;
        }
        if (ip > 0) stat += ` · ${ip} 进行中`;
        if (isStale) stat += ` · ⚠ 可能滞后`;
      }
      // v0.19.5：TaskCreate / TaskUpdate 个体调用还带 subject / taskId，挂在
      // tool_input 上；附在主标签后面，方便扫读"这一步具体改了哪条"。
      let suffix = '';
      if (tool === 'TaskCreate') {
        const subj = step.metadata?.tool_input?.subject
          || step.metadata?.tool_input?.activeForm;
        if (typeof subj === 'string' && subj.trim()) {
          suffix = ` · ${subj.trim().slice(0, 24)}${subj.length > 24 ? '…' : ''}`;
        }
      } else if (tool === 'TaskUpdate') {
        const tid = step.metadata?.plan_display_task_id ?? step.metadata?.tool_input?.taskId;
        const st = step.metadata?.tool_input?.status;
        if (tid != null || (typeof st === 'string' && st)) {
          suffix = ` · #${tid ?? '?'} → ${st ?? '?'}`;
        }
      }
      return `${cfg.label}${idxLabel}${stat}${suffix}`;
    }
    if (tool === 'SwitchMode' && step.step_type === 'tool_decision') {
      const ms = step.metadata?.mode_switch as { target_mode_id?: string } | undefined;
      const target = ms?.target_mode_id || step.metadata?.tool_input?.target_mode_id;
      return target ? `${cfg.label} → ${target}` : cfg.label;
    }
    if (tool === 'CreatePlan' && step.step_type === 'tool_decision') {
      const name = step.metadata?.tool_input?.name;
      return name ? `${cfg.label} · ${name}` : cfg.label;
    }
    // tool_execution 仍走默认 cfg.label
    return cfg.label;
  }

  if (tool) {
    // v0.14.4：RAG / LLM / Web 调用的 decision 和 execution 都被 invocation_category
    // 覆盖了同一个 cfg.label（如 'RAG Query'），看起来像"同一个调用生成了两个 span"。
    // 这里在 execution 阶段加一个明确的"召回结果 / 工具响应"前缀，让两个 span 各
    // 司其职：decision = 发出去的请求，execution = 收回来的真实返回。
    const cat = step.metadata?.invocation_category;
    if (step.step_type === 'tool_execution' && cat) {
      const respLabel = cat === 'rag_query' ? 'RAG 召回结果'
        : cat === 'web_search' ? 'Web 搜索结果'
        : 'LLM 响应';
      return `${respLabel} ← ${tool}`;
    }
    // v0.20.11：tool_decision 一律渲染成 "LLM Thinking → Use Tool <tool>"
    // 不论 invocation_category 是 rag_query / web_search / llm_call 还是普通工具，
    // 视觉口径统一。tool_execution 已在上面分流，到这里的 tool_execution 是
    // 普通工具（无 invocation_category），保留原 "Tool Execution → <tool>" 格式。
    if (step.step_type === 'tool_decision') {
      return `${cfg.label} → Use Tool ${tool}`;
    }
    return `${cfg.label} → ${tool}`;
  }
  // 对 thinking 类型截取内容前 28 字
  // v0.20.11: pre_tool_reasoning 一并走这条路径——它跟 thinking_* 的内容
  // 性质完全一致（LLM 在调工具前的推理），不再保留单独的"决策说明"标签。
  if (
    (step.step_type.startsWith('thinking') || step.step_type === 'pre_tool_reasoning')
    && step.content
  ) {
    const c = step.content.replace(/[\r\n]+/g, ' ').trim();
    return c.slice(0, 28) + (c.length > 28 ? '…' : '');
  }
  // v0.8.2: final_response 截 28 字预览，便于扫读
  if (step.step_type === 'final_response' && step.content) {
    const c = step.content.replace(/[\r\n]+/g, ' ').trim();
    return `${cfg.label}: ${c.slice(0, 28)}${c.length > 28 ? '…' : ''}`;
  }
  return cfg.label;
}

// ─── 单个 Step 节点（支持展开子节点）────────────────────
function StepNode({
  step, turn, isSelected, onSelect, treatAsThinking = false,
}: {
  step: ThoughtStep; turn: TurnCoT; isSelected: boolean;
  onSelect: (n: SelectedNode) => void;
  // v0.16.2: 当 turn 级判定为"Claude 未启用 Extended Thinking"时，把
  // 单独出现的 pre_tool_reasoning 渲染成紫色 🧠 Pre-tool Thinking，
  // 跟 Cursor 那边的 thinking 视觉一致；不再保留之前那个 💡 决策说明。
  treatAsThinking?: boolean;
}) {
  const cfg = getStepCfg(step);
  // v0.20.11: pre_tool_reasoning 已在 STEP_CFG 默认就用紫色 🧠 + 'Thinking' 标签，
  // getStepNodeLabel 也会返回内容首句。这里 treatAsThinking 仅用于"内容为空"
  // 的兜底场景：用 extractThoughtTitleSpanTree 取更长的标题（80 字而非 28），
  // 标签彻底空时回退到 'Pre-tool Thinking'，保持视觉不出现纯空气节点。
  void treatAsThinking; // v1.0.4: prop 暂未在 body 内消费，但保留 API 不破坏调用方
  let stepLabel = getStepNodeLabel(step);
  if (step.step_type === 'pre_tool_reasoning') {
    const thoughtTitle = extractThoughtTitleSpanTree(step.content || '', 80);
    if (thoughtTitle) {
      stepLabel = thoughtTitle;
    } else if (!stepLabel || stepLabel === 'Thinking') {
      stepLabel = 'Pre-tool Thinking';
    }
  }
  const dur = fmtDuration(step.duration_ms);

  // 判断是否有可展开的子节点
  const hasChildren = (
    (step.step_type === 'tool_decision' && step.metadata?.tool_input) ||
    (step.step_type === 'tool_execution' && step.content) ||
    (step.step_type === 'error_recovery' && step.metadata?.error_content) ||
    (step.step_type === 'strategy_shift')
  );
  const [expanded, setExpanded] = useState(false);

  // v0.10.0: Plan 类工具加专属 className，左侧色带让它们从普通 tool 流里
  // 视觉上脱出（金色：TodoWrite/CreatePlan，紫色：SwitchMode）
  const planTool = step.tool_name || step.metadata?.tool_name;
  const planCls = planTool && PLAN_TOOL_CFG[planTool]
    ? ` tree-step-plantool tree-step-plantool-${planTool.toLowerCase()}`
    : '';

  return (
    <div className="tree-step-wrapper" data-step-index={step.step_index}>
      <div
        className={`tree-step ${isSelected ? 'selected' : ''}${planCls}`}
        onClick={() => {
          onSelect({ kind: 'step', step, turn });
          if (hasChildren) setExpanded(e => !e);
        }}
      >
        <div className="tree-step-connector" />
        {hasChildren && (
          <span className="tree-step-chevron">{expanded ? '▾' : '▸'}</span>
        )}
        <div className="tree-step-dot" style={{ background: cfg.color }} />
        <span className="tree-step-icon">{cfg.icon}</span>
        <span className="tree-step-label">{stepLabel}</span>
        <div className="tree-step-right">
          {/* v0.8.2: final_response 启发式推断的小角标 */}
          {step.step_type === 'final_response' && step.metadata?.inferred_final && (
            <span
              className="tree-step-inferred"
              title={`启发式推断 final（${step.metadata?.inferred_reason || 'unknown'}）`}
            >
              ⚠️ inferred
            </span>
          )}
          {/* v0.8.2: synthetic 占位执行节点的小角标 */}
          {step.step_type === 'tool_execution' && step.metadata?.synthetic && !step.metadata?.synthetic_upgraded && (
            <span className="tree-step-synthetic" title="结果未捕获，cot_extractor 合成的占位">
              ◌ no result
            </span>
          )}
          {/* v0.8.2: synthetic ↑ upgraded 标记，被 cot-stream.js 实时回灌后 */}
          {step.step_type === 'tool_execution' && step.metadata?.synthetic_upgraded && (
            <span className="tree-step-upgraded" title="synthetic 占位已被 cot-stream.js 实时回灌升级为真值">
              ↑ upgraded
            </span>
          )}
          {/* v0.17.1: CodeBuddy 真值徽章 —— 工具结果直接来自 CodeBuddy 原生
              transcript（不是 synthetic 占位、也不是 cot-stream 回灌），等价于
              cursor 的"↑ upgraded" 但更明确地标注数据来源是 native transcript。
              Cursor / Claude session 不会渲染（依赖 captured_from 字段）。 */}
          {step.step_type === 'tool_execution'
            && (step.metadata as any)?.captured_from === 'codebuddy_transcript' && (
            <span
              className="tree-step-upgraded"
              title="CodeBuddy 原生 transcript 真值（不是占位，含完整 result 内容）"
            >
              ↑ transcript
            </span>
          )}
          {/* v0.17.1: CodeBuddy 工具异常 ⚠️ 徽章（is_error=true 时显示） */}
          {step.step_type === 'tool_execution'
            && (step.metadata as any)?.captured_from === 'codebuddy_transcript'
            && step.metadata?.is_error && (
            <span
              className="tree-step-inferred"
              title="CodeBuddy 报告 isError=true"
            >
              ⚠️ tool error
            </span>
          )}
          {/* v0.16.5: OTel 通道独有工具（subagent 内部嵌套调用，主 transcript 看不见） */}
          {(step.metadata as any)?._otel_orphan && (
            <span
              className="tree-step-otel-orphan"
              title="来自 Claude 进程级 OTel：subagent 内部嵌套调用——主 transcript 把它们折叠在 Agent 工具的 result 文本里看不见，这里通过 OTel 通道补全"
            >
              🌀 otel
            </span>
          )}
          {/* v0.9.0: file_op 徽标 —— 让用户在树上一眼看到"哪一步在写文件"
              create=📜（临时变💛）/ modify=✏️ / delete=🗑️ */}
          {step.metadata?.file_op && (
            (() => {
              const fop = step.metadata!.file_op as { kind: string; basename: string; is_temp: boolean; language: string };
              const icon = fop.kind === 'create' ? (fop.is_temp ? '📜' : '📄')
                : fop.kind === 'modify' ? '✏️' : '🗑️';
              const cls = fop.kind === 'create' ? (fop.is_temp ? 'tree-step-fileop-temp' : 'tree-step-fileop-create')
                : fop.kind === 'modify' ? 'tree-step-fileop-modify' : 'tree-step-fileop-delete';
              return (
                <span className={`tree-step-fileop ${cls}`} title={`${fop.kind} ${fop.basename}`}>
                  {icon} {fop.basename.length > 18 ? fop.basename.slice(0, 16) + '…' : fop.basename}
                </span>
              );
            })()
          )}
          {/* v0.9.0: Shell step 命中执行某个 artifact 时贴个 ▶ 标 */}
          {step.metadata?.executed_artifact && (
            (() => {
              const ea = step.metadata!.executed_artifact as { basename: string; is_temp: boolean };
              return (
                <span
                  className={`tree-step-execart ${ea.is_temp ? 'tree-step-execart-temp' : ''}`}
                  title={`运行脚本 ${ea.basename}`}
                >
                  ▶ {ea.basename.length > 16 ? ea.basename.slice(0, 14) + '…' : ea.basename}
                </span>
              );
            })()
          )}
          {/* v0.10.0: SwitchMode 徽标 —— "🔀 → plan/agent" */}
          {step.metadata?.mode_switch && (
            (() => {
              const ms = step.metadata!.mode_switch as { target_mode_id: string; trigger?: string };
              const cls = ms.target_mode_id === 'plan'
                ? 'tree-step-mode-plan'
                : ms.target_mode_id === 'agent'
                  ? 'tree-step-mode-agent'
                  : 'tree-step-mode-other';
              const arrow = ms.trigger === 'implicit_back_to_agent' ? '⤷' : '→';
              return (
                <span
                  className={`tree-step-mode ${cls}`}
                  title={`模式转换: ${arrow} ${ms.target_mode_id}${ms.trigger === 'implicit_back_to_agent' ? '（隐式）' : ''}`}
                >
                  🔀 {arrow} {ms.target_mode_id}
                </span>
              );
            })()
          )}
          {/* v0.10.0: CreatePlan 徽标 —— 醒目的 📋 plan 文档 */}
          {step.metadata?.plan_proposal && (
            (() => {
              const pp = step.metadata!.plan_proposal as { name?: string; plan_chars?: number };
              return (
                <span
                  className="tree-step-plan-doc"
                  title={`Plan 文档「${pp.name || '未命名'}」(${pp.plan_chars || 0} 字)`}
                >
                  📋 plan 文档
                </span>
              );
            })()
          )}
          {/* v0.10.0: TodoWrite step diff 徽标 —— "✅+2 ▶+1 ＋3"
              v0.19.5: 同样适用 Claude Internal 的 TaskCreate / TaskUpdate，它们
              的 step.metadata.plan_diff 也是 _diff_todos(prev_full, curr_full)
              的标准输出，前端不用区分。 */}
          {step.metadata?.plan_diff && (() => {
            const tn = step.tool_name || step.metadata?.tool_name;
            return tn === 'TodoWrite' || tn === 'TaskCreate' || tn === 'TaskUpdate';
          })() && (
            (() => {
              const d = step.metadata!.plan_diff as {
                newly_completed?: any[]; newly_started?: any[];
                newly_added?: any[]; removed?: any[];
              };
              const nc = d.newly_completed?.length || 0;
              const ns = d.newly_started?.length || 0;
              const na = d.newly_added?.length || 0;
              const nr = d.removed?.length || 0;
              if (nc + ns + na + nr === 0) return null;
              return (
                <span
                  className="tree-step-plan-diff"
                  title={`本次 TodoWrite 变化: 完成 ${nc} / 启动 ${ns} / 新增 ${na} / 删除 ${nr}`}
                >
                  {nc > 0 && <span className="tree-plan-diff-done">✅+{nc}</span>}
                  {ns > 0 && <span className="tree-plan-diff-start">▶+{ns}</span>}
                  {na > 0 && <span className="tree-plan-diff-add">＋{na}</span>}
                  {nr > 0 && <span className="tree-plan-diff-remove">−{nr}</span>}
                </span>
              );
            })()
          )}
          {/* v0.20.7: per-step token + cost 徽章。读 step.otel.token_usage
              （所有 IDE 通用，由 cot_otel_enricher 注入），fallback 到
              metadata.{input,output}_tokens 与早期 step.tokens 兼容字段。
              tool_execution / non_llm_step 自动隐藏，避免 0 token 噪声。
              v0.20.10: Claude Code session 下 host_tool / user_input 在
              enricher 阶段已经强制 0/0+non_llm，自动不出 chip。 */}
          <StepTokenBadge step={step} />
          {dur && <span className="tree-step-dur">{dur}</span>}
        </div>
      </div>

      {/* 子节点：工具参数 / 工具结果 / 错误详情 */}
      {hasChildren && expanded && (
        <div className="tree-sub-nodes">
          {step.step_type === 'tool_decision' && step.metadata?.tool_input && (
            <SubNode icon="📋" label="工具参数" color="#a855f7" />
          )}
          {step.step_type === 'tool_execution' && (
            step.metadata?.synthetic ? (
              <SubNode
                icon="◌"
                label="结果未记录 (Cursor)"
                color="#64748b"
                sub="transcript 不含 tool_result"
              />
            ) : (
              <SubNode
                icon={step.metadata?.is_error ? '❌' : '✅'}
                label={step.metadata?.is_error ? '执行失败' : '执行成功'}
                color={step.metadata?.is_error ? '#ef4444' : '#10b981'}
                sub={`${step.metadata?.result_len || 0} chars`}
              />
            )
          )}
          {step.step_type === 'error_recovery' && (
            <SubNode icon="🔧" label="恢复策略" color="#f59e0b" />
          )}
          {step.step_type === 'strategy_shift' && (
            <SubNode
              icon="🔄"
              label={`${step.metadata?.from_tool || '?'} → ${step.metadata?.to_tool || '?'}`}
              color="#f59e0b"
            />
          )}
        </div>
      )}
    </div>
  );
}

// ─── 子节点（叶子级别）───────────────────────────────────
function SubNode({ icon, label, color, sub }: { icon: string; label: string; color: string; sub?: string }) {
  return (
    <div className="tree-sub-node">
      <div className="tree-sub-connector" />
      <div className="tree-sub-dot" style={{ background: `${color}66` }} />
      <span className="tree-sub-icon">{icon}</span>
      <span className="tree-sub-label" style={{ color }}>{label}</span>
      {sub && <span className="tree-sub-meta">{sub}</span>}
    </div>
  );
}

// ─── 线性时序渲染辅助 ─────────────────────────────────────
// 让 tool_decision + tool_execution 作为一个逻辑单元（缩进在 thinking 之下）。
// 我们把 steps 按原始 step_index 顺序分组：
//   - thinking/user_input/final_response 等 → "思考段"（顶级节点）
//   - 紧随其后的连续 tool_decision / tool_execution / strategy_shift / error_recovery
//     → 作为上一个"思考段"的工具链子节点
interface RenderGroup {
  leader: ThoughtStep | null;  // 顶级思考节点（可能是 user_input / thinking / final_response）
  tools: ThoughtStep[];        // 紧跟其后的工具 + 异常步骤
}

// Thinking Phase：把多组"leader=thinking_explicit、无 tools"的连续 RenderGroup
// 折叠为一个黄色阶段节点。点开后下面是子树，每条 thought 是一行可独立展开的
// 小卡。设计动机：单 turn 里经常出现 30-50 条连续 afterAgentThought，平铺
// 会把树撑得没法看；折叠后能把"模型在思考"vs"模型在执行"的节奏一眼看清。
interface ThinkingPhaseGroup {
  kind: 'phase';
  thoughts: ThoughtStep[];
  firstIndex: number;
  lastIndex: number;
}

type AnyGroup = RenderGroup | ThinkingPhaseGroup;

// v0.16.5: 取 step 真实时间戳（ms）。tool_execution 的 step.timestamp 经常是
// user_message 时间（远晚于实际执行），所以优先 metadata.observed_at_ms。
function stepTsMs(s: ThoughtStep): number {
  const md: any = (s as any)?.metadata || {};
  if (typeof md.observed_at_ms === 'number' && md.observed_at_ms > 0) return md.observed_at_ms;
  if (typeof md._t_ms === 'number' && md._t_ms > 0) return md._t_ms;
  const ts = (s as any)?.timestamp;
  if (typeof ts === 'string' && ts) {
    const t = Date.parse(ts);
    if (!isNaN(t)) return t;
  }
  return 0;
}

// 把后端 /api/sessions/<sid>/timeline 的有序事件流还原成渲染分组。
//
// 前端不再自己排序。顺序（含 thinking 折叠、D-E 配对、并行事件穿插位置）
// 全部由后端 agent_cot.trace 算好，导出用的是同一份结果——UI 和导出因此
// 不可能漂移。这里只做两件事：
//   1. 用 step_index 把事件 join 回完整的 ThoughtStep（事件本身是投影，
//      不带 metadata / otel / reasoning_digest，DetailPanel 还要用）；
//   2. 按 group_id / role / phase_id 重建 RenderGroup 与 ThinkingPhaseGroup。
function groupsFromTimeline(
  events: TraceEvent[],
  turn: TurnCoT,
  agentType: string,
): AnyGroup[] {
  const byStepIndex = new Map<number, ThoughtStep>();
  for (const s of turn.steps) byStepIndex.set(s.step_index, s);

  const out: AnyGroup[] = [];
  let currentGroup: RenderGroup | null = null;
  let currentGroupId: number | null = null;
  let currentPhase: ThinkingPhaseGroup | null = null;
  let currentPhaseId: number | null = null;
  let virtIdx = 0;

  for (const ev of events) {
    if (ev.turn !== turn.turn_index) continue;
    if (ev.type === 'turn_start') continue;
    // 后端已经标好「当前 UI 不渲染」的事件（环境事件、权限档位切换、
    // 落在 turn 时间窗外的、以及无时间戳排不进顺序的）
    if (ev.in_ui === false || ev.ordered === false) continue;

    const step = resolveTimelineStep(ev, byStepIndex, agentType, virtIdx);
    if (!step) continue;
    if (ev.step_index == null || !byStepIndex.has(ev.step_index)) virtIdx++;

    if (ev.phase_id != null) {
      if (!currentPhase || currentPhaseId !== ev.phase_id) {
        currentPhase = {
          kind: 'phase',
          thoughts: [],
          firstIndex: step.step_index,
          lastIndex: step.step_index,
        };
        out.push(currentPhase);
        currentPhaseId = ev.phase_id;
        currentGroup = null;
        currentGroupId = null;
      }
      currentPhase.thoughts.push(step);
      currentPhase.lastIndex = step.step_index;
      continue;
    }
    currentPhase = null;
    currentPhaseId = null;

    if (currentGroup === null || ev.group_id !== currentGroupId || ev.role === 'leader') {
      currentGroup = { leader: ev.role === 'child' ? null : step, tools: [] };
      currentGroupId = ev.group_id ?? null;
      out.push(currentGroup);
      if (ev.role === 'child') currentGroup.tools.push(step);
      continue;
    }
    currentGroup.tools.push(step);
  }
  return out;
}

// 事件 → ThoughtStep。三种来源：
//   * 主步流：按 step_index 取回完整对象
//   * hook 事件（子代理 / 权限 / 通知 / 压缩）：合成虚拟 step 走 ClaudeEventInline
//   * OTel 孤儿工具调用：subagent 内部的调用，主 transcript 里没有，合成一个
function resolveTimelineStep(
  ev: TraceEvent,
  byStepIndex: Map<number, ThoughtStep>,
  agentType: string,
  virtIdx: number,
): ThoughtStep | null {
  if (ev.step_index != null) {
    const real = byStepIndex.get(ev.step_index);
    if (real) return real;
  }
  if (CLAUDE_EVENT_KINDS.has(ev.type)) {
    return makeClaudeEventStep(
      { kind: ev.type as ClaudeEventKind, payload: ev.payload, t_ms: ev.t_ms || 0, agentType },
      virtIdx,
    );
  }
  if (ev.type === 'tool_call' || ev.type === 'tool_result') {
    return {
      step_index: 1_000_000 + virtIdx,
      turn_index: ev.turn ?? 0,
      step_type: ev.type === 'tool_call' ? 'tool_decision' : 'tool_execution',
      content: ev.content || '',
      tool_name: ev.tool || 'tool',
      tool_use_id: ev.tool_use_id || '',
      metadata: {
        observed_at_ms: ev.t_ms,
        tool_name: ev.tool,
        tool_use_id: ev.tool_use_id,
        tool_input: ev.payload,
        is_error: ev.is_error,
        _otel_orphan: ev.otel_orphan,
      } as any,
      timestamp: ev.ts || '',
      duration_ms: ev.duration_ms ?? null,
      tokens: ev.tokens ?? 0,
    } as any;
  }
  return null;
}

// 跟 DetailPanel 用的同款首句抽取，复制一份避免循环依赖
function extractThoughtTitleSpanTree(text: string, maxLen = 100): string {
  if (!text) return '';
  const cleaned = text.replace(/^[\s\u3000]+/, '');
  const m = cleaned.match(/^([^\.\!\?。！？\n]{6,}?)(?:[\.\!\?。！？\n]|$)/);
  let s = m ? m[1] : cleaned.slice(0, maxLen);
  s = s.trim();
  if (s.length > maxLen) s = s.slice(0, maxLen).trim() + '…';
  return s;
}

// ─── Claude 时间线事件 ───────────────────────────────────
// 4 类 hook events（subagent / perm / notif / compact）不是主步流的一部分，
// 后端 agent_cot.trace 已经按真实时刻把它们穿插进事件流的正确位置，这里只负责
// 把单条事件转成虚拟 ThoughtStep：渲染时识别 step_type.startsWith('claude_event_')
// 走 ClaudeEventInline 而不是 StepNode；data-claude-section 让 turn header 上的
// 徽章仍能 scrollIntoView。

type ClaudeEventKind = 'subagent' | 'permission' | 'notification' | 'compact';

const CLAUDE_EVENT_KINDS = new Set<string>([
  'subagent', 'permission', 'notification', 'compact',
]);

interface ClaudeEventEntry {
  kind: ClaudeEventKind;
  payload: any;
  t_ms: number;
  agentType?: string;
}

function makeClaudeEventStep(ev: ClaudeEventEntry, idx: number): ThoughtStep {
  // 用足够负的 step_index 避免跟真步号冲突；同时 step_index 唯一，
  // 让 React key / selectedNode.step_index 比对都不会撞车。
  return {
    step_index: -1_000_000 - idx,
    turn_index: 0,
    step_type: (`${ev.agentType === 'codex' ? 'codex' : 'claude'}_event_` + ev.kind) as any,
    content: ev.kind === 'compact'
      ? String(ev.payload?.summary || ev.payload?.summary_preview || '')
      : '',
    tool_name: ev.kind,
    metadata: {
      _claude_event_payload: ev.payload,
      _claude_event_kind: ev.kind,
      _t_ms: ev.t_ms,
      observed_at_ms: ev.t_ms,
    } as any,
    timestamp: ev.t_ms ? new Date(ev.t_ms).toISOString() : '',
    duration_ms: typeof ev.payload?.duration_ms === 'number' ? ev.payload.duration_ms : null,
    tokens: 0,
  } as any;
}

function isClaudeEventStep(s: ThoughtStep): boolean {
  return typeof (s as any)?.step_type === 'string'
    && (
      (s as any).step_type.startsWith('claude_event_')
      || (s as any).step_type.startsWith('codex_event_')
    );
}

// 连续相同工具折叠（v0.16.5）：
// 把 RenderGroup.tools 里"连续 ≥2 个相同 tool_name 的 tool_decision（含其紧邻
// tool_execution + 中间夹的 permission inline）"压成一个可展开的 ToolBatch。
// 用户场景：subagent / hook 大量重复 Read / Grep / Bash 时把树撑爆，折叠成
// "Read ×N" 一行能极大压缩视觉，展开后还能看到每条调用的细节。
type ToolBatchEntry = {
  kind: 'tool-batch';
  toolName: string;
  steps: ThoughtStep[];
  count: number;
  firstTs: number;
  lastTs: number;
  permCount: number;
};
type ToolNodeEntry = { kind: 'tool-node'; step: ThoughtStep };
type ToolRenderEntry = ToolBatchEntry | ToolNodeEntry;

function buildToolRenderEntries(tools: ThoughtStep[]): ToolRenderEntry[] {
  const out: ToolRenderEntry[] = [];
  let i = 0;
  const isPermInline = (s: ThoughtStep): boolean =>
    isClaudeEventStep(s) && (s as any).step_type === 'claude_event_permission';
  // v0.18.14: 入口扩展 —— 现在 tool_decision *和* tool_execution 都能起头一批。
  //   背景：Cursor v2.6+ events.jsonl 合成出来的 TOOL_EXECUTION 步骤没有
  //   配对的 tool_decision（cot_extractor "Channel 5: events synthesis"），
  //   导致 50+ 条连续的 "Tool Execution → Read" 在前端铺平 → 树爆炸。
  //   折叠门槛：
  //     · decision-led batch（带 tool_decision）：≥2 个 decision 即折
  //     · execution-only batch（无 decision，纯合成）：≥3 个连续同名 execution 才折
  const entryToolName = (s: ThoughtStep): string => {
    if (s.step_type === 'tool_decision' || s.step_type === 'tool_execution') {
      return ((s.tool_name || (s.metadata as any)?.tool_name || '') as string);
    }
    return '';
  };
  while (i < tools.length) {
    const s = tools[i];
    const tn = entryToolName(s);
    if (!tn) {
      out.push({ kind: 'tool-node', step: s });
      i += 1;
      continue;
    }
    // 尝试沿着 tools 找连续的同名工具调用（decision-led 或 execution-only）
    const batch: ThoughtStep[] = [];
    let permCnt = 0;
    let j = i;
    while (j < tools.length) {
      const tj = tools[j];
      const tjName = entryToolName(tj);
      if (tjName !== tn) break;

      if (tj.step_type === 'tool_decision') {
        batch.push(tj);
        // 下一条若是同 use_id 的 tool_execution，归入本批
        const useId = (tj.tool_use_id || (tj.metadata as any)?.tool_use_id || '') as string;
        if (
          j + 1 < tools.length
          && tools[j + 1].step_type === 'tool_execution'
          && useId
          && (tools[j + 1].tool_use_id || (tools[j + 1].metadata as any)?.tool_use_id) === useId
        ) {
          batch.push(tools[j + 1]);
          j += 2;
        } else {
          j += 1;
        }
      } else if (tj.step_type === 'tool_execution') {
        // execution-only 入口：直接吃一条
        batch.push(tj);
        j += 1;
      } else {
        break;
      }

      // 紧贴的 permission inline 也并入本批
      while (j < tools.length && isPermInline(tools[j])) {
        batch.push(tools[j]);
        permCnt += 1;
        j += 1;
      }
    }
    // 折叠门槛
    const decisionCount = batch.filter(b => b.step_type === 'tool_decision').length;
    const executionCount = batch.filter(b => b.step_type === 'tool_execution').length;
    const shouldBatch =
      decisionCount >= 2 || (decisionCount === 0 && executionCount >= 3);
    if (shouldBatch) {
      const tsList = batch.map(stepTsMs).filter(t => t > 0);
      out.push({
        kind: 'tool-batch',
        toolName: tn,
        steps: batch,
        count: decisionCount > 0 ? decisionCount : executionCount,
        firstTs: tsList[0] || 0,
        lastTs: tsList[tsList.length - 1] || 0,
        permCount: permCnt,
      });
      i = j;
    } else {
      out.push({ kind: 'tool-node', step: s });
      i += 1;
    }
  }
  return out;
}

// ─── Turn 块（线性时序 + 工具嵌在 thinking 之下） ─────────
function TurnNode({
  turn, cot, timeline, selectedNode, onSelect, abRole,
}: {
  turn: TurnCoT; cot: SessionCoT; timeline: TraceEvent[] | null;
  selectedNode: SelectedNode | null;
  onSelect: (n: SelectedNode) => void;
  abRole?: AbTurnRole;
}) {
  const [expanded, setExpanded] = useState(true);
  const { icon, label, sub } = getTurnLabel(turn);
  const isTurnSelected = selectedNode?.kind === 'turn' && selectedNode.turn.turn_index === turn.turn_index;

  // v0.14.6：当从徽章/外部跳转选中本 turn 内的某个 step 时，自动展开本 turn，
  // 否则 step DOM 不在树里，scrollIntoView 也滚不过去。
  useEffect(() => {
    if (!selectedNode) return;
    if (selectedNode.kind === 'step' && selectedNode.turn.turn_index === turn.turn_index) {
      setExpanded(true);
    } else if (selectedNode.kind === 'turn' && selectedNode.turn.turn_index === turn.turn_index) {
      setExpanded(true);
    }
  }, [selectedNode, turn.turn_index]);

  // v0.16.4: 徽章点击 → 展开 turn + 滚到对应事件区块。
  // 之前所有 Claude 4 类徽章 onClick 都只是 onSelect({kind:'turn'})，跳到
  // turn 详情但不滚到具体节点；现在用 data-claude-section + 双 rAF + setTimeout
  // 等 expanded 状态在 DOM 上落地后再 scrollIntoView，并加 flash 高亮。
  const scrollToClaudeSection = (sectionKey: string) => {
    setExpanded(true);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      setTimeout(() => {
        const sel = `[data-turn-index="${turn.turn_index}"] [data-claude-section="${sectionKey}"]`;
        const el = document.querySelector(sel) as HTMLElement | null;
        if (!el) return;
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        el.classList.add('tree-scroll-flash');
        setTimeout(() => el.classList.remove('tree-scroll-flash'), 1200);
      }, 60);
    }));
  };
  // v0.20.7: hook 真值 → otel enricher 兜底（详见 _readTurnTotalTokens 注释）
  const totalTokens = _readTurnTotalTokens(turn);
  // v0.14.2: 优先用 hook 真值时长，回退到 transcript 估算
  const dur = fmtDuration(turn.turn_duration_ms_observed ?? turn.turn_duration_ms);

  // 本轮是否发生了 TodoWrite（Plan 变更）
  const turnPlanCount = (cot.plan_timeline || []).filter(p => p.turn_index === turn.turn_index).length;

  // v0.8.0: 本轮 LLM / RAG / Web Search 计数（按 tool_decision 上的
  // invocation_category 聚合；turn header 直接显示徽章方便定位）
  const turnInvocation = useMemo(() => {
    let llm = 0, rag = 0, web = 0;
    for (const s of turn.steps) {
      if (s.step_type !== 'tool_decision') continue;
      const cat = s.metadata?.invocation_category;
      if (cat === 'llm_call') llm++;
      else if (cat === 'rag_query') rag++;
      else if (cat === 'web_search') web++;
    }
    return { llm, rag, web };
  }, [turn.steps]);

  // v0.10.0: 本轮模式转换 + plan 提案
  const turnModeInfo = useMemo(() => {
    const transitions = (cot.mode_transitions || []).filter(m => m.turn_index === turn.turn_index);
    const proposals = (cot.plan_proposals || []).filter(p => p.turn_index === turn.turn_index);
    const finalMode: string | undefined = transitions.length > 0
      ? transitions[transitions.length - 1].target_mode_id
      : undefined;
    return { transitions, proposals, finalMode };
  }, [cot.mode_transitions, cot.plan_proposals, turn.turn_index]);

  // v0.9.0: 本轮文件操作 + 临时脚本计数（按 step.metadata.file_op 聚合）
  const turnFileOps = useMemo(() => {
    let create = 0, modify = 0, del = 0, temp = 0, exec = 0;
    for (const s of turn.steps) {
      const fop = s.metadata?.file_op;
      if (fop) {
        if (fop.kind === 'create') create++;
        else if (fop.kind === 'modify') modify++;
        else if (fop.kind === 'delete') del++;
        if (fop.is_temp && fop.kind === 'create') temp++;
      }
      if (s.metadata?.executed_artifact && s.step_type === 'tool_decision') exec++;
    }
    return { create, modify, del, temp, exec };
  }, [turn.steps]);

  // v0.15.1：本轮内 Subagent / Compact 计数（按时间窗口与本 turn 匹配）。
  // 用于在 turn header 显示徽章；点击跳到 turn 详情滚动到对应 Section。
  // v0.16.4：现在不只统计计数，连同**事件 list 本身**也返回；turn 展开后
  // 这些 list 会被渲染成可点击节点，让用户真的看见这些事件，不再只是数字。
  const turnClaudeBadges = useMemo(() => {
    const agentType = cot?.agent_type || '';
    if (agentType !== 'claude' && agentType !== 'codex') {
      return { subagent: 0, compact: 0, permission: 0, notification: 0,
        subagentList: [] as any[], compactList: [] as any[],
        permissionList: [] as any[], notificationList: [] as any[] };
    }
    const t0 = turn.turn_start_ms_observed
      || (turn.turn_start_time ? new Date(turn.turn_start_time).getTime() : 0);
    const t1 = turn.turn_end_ms_observed
      || (t0 && turn.turn_duration_ms ? t0 + turn.turn_duration_ms : Number.MAX_SAFE_INTEGER);
    const inWin = (t: number | undefined): boolean => {
      if (!t || t <= 0) return false;
      return t >= t0 && t <= t1;
    };
    const sa = (cot.subagent_timeline || []).filter((e: any) => inWin(e.t_ms));
    // subagent 一对 start+stop 算 1 个，按 sub_agent_id 去重
    const saIds = new Set(sa.map((e: any) => e.sub_agent_id || `t${e.t_ms}`));
    const cmp = (cot.compact_events || []).filter((e: any) => inWin(e.t_ms));
    // compact PreCompact + PostCompact 算 1 次（按 phase=before 计数）
    const cmpCount = cmp.filter((e: any) => e.phase === 'before').length || cmp.length;
    const perm = (cot.permission_events || []).filter((e: any) =>
      e.kind !== 'PermissionMode' && inWin(e.t_ms),
    );
    const notif = (cot.notification_events || []).filter((e: any) => inWin(e.t_ms));
    return {
      subagent: saIds.size,
      compact: cmpCount,
      permission: perm.length,
      notification: notif.length,
      subagentList: sa,
      compactList: cmp,
      permissionList: perm,
      notificationList: notif,
    };
  }, [
    cot?.agent_type, cot?.subagent_timeline, cot?.compact_events,
    cot?.permission_events, cot?.notification_events,
    turn.turn_start_ms_observed, turn.turn_end_ms_observed,
    turn.turn_start_time, turn.turn_duration_ms,
  ]);

  // v0.15.0：Claude 未启用 ext-thinking 时把 pre_tool_reasoning 按思考气泡
  // 渲染（与 DetailPanel 一致），给前端补上"思考阶段"的观感。
  // v0.17.1：CodeBuddy 永远启用——它的 pre_tool_reasoning 装的是模型在
  // tool_use 之前的自然语言推理，按 💡 决策说明渲染会把真实推理藏进折叠器。
  // 这纯粹是展示口径，与事件顺序无关（顺序由后端定）。
  const treatPreToolAsThinking = useMemo(() => {
    const at = (cot?.agent_type || '');
    if (at === 'codebuddy') return true;
    if (at !== 'claude') return false;
    const hasExtThinking = turn.steps.some(
      s => s.step_type === 'thinking_explicit'
        || s.step_type === 'thinking_inter'
        || s.step_type === 'thinking_intermediate',
    );
    return !hasExtThinking;
  }, [turn.steps, cot?.agent_type]);

  // 渲染顺序完全来自后端。timeline 未就绪时先不渲染步骤（而不是退回本地
  // 排序）——两套排序实现正是这次改造要消灭的东西，宁可空一帧也不能让
  // UI 显示一个与导出不一致的顺序。
  const groups = useMemo(() => {
    if (!timeline) return [] as AnyGroup[];
    return groupsFromTimeline(timeline, turn, cot?.agent_type || '');
  }, [timeline, turn, cot?.agent_type]);

  // v0.11.2：本轮聚合 step.otel.token_usage（input/output/cost）
  // v1.0.4: Cursor / CodeBuddy 在 1.0.3 起 step.usage 全是 0/0（协议没暴露 per-step 真值），
  // 累加结果会落到 0/0 + 0 cost，让 chip 显示 "↧0/↥0"、tooltip 显示 "输入 0 · 输出 0"，
  // 误导用户。turn.usage 一直存的是 hook 真值（cursor renderer.log per-turn 总额 /
  // codebuddy index.json::requests[i].usage 真值），所以这里改成：**优先用 turn.usage
  // 真值**，缺时才落到 step 累加（保留对老 cot.json 的兼容路径）。
  // Claude session 走 step 累加路径完全不变（per-message transcript_per_message 真值
  // 加起来 == turn.usage，二选一结果一致）。
  const turnOtel = useMemo(() => {
    // 先看 turn.usage（hook / transcript per-turn 真值，三家 IDE 都已落到这里）
    const tUsage: any = (turn as any).usage || {};
    const turnIn = Number(tUsage.input_tokens) || 0;
    const turnOut = Number(tUsage.output_tokens) || 0;
    const turnSource = typeof tUsage.source === 'string' ? tUsage.source : '';
    let llm = 0;
    let stepInTok = 0, stepOutTok = 0, costSum = 0, hasCost = false;
    for (const s of turn.steps) {
      const tu: any = (s as any).otel?.token_usage;
      if (!tu) continue;
      if ((s as any).otel?.step_kind === 'llm_call') llm++;
      // non_llm_step（tool_execution / user_input）的 token 是工具入参和结果
      // 的字符估算，模型从没产出过它们，cost_usd 也是 null。把它们加进来会让
      // 这枚徽章自相矛盾：一条 cursor turn 实测 181 个 LLM step 是 4.4K/26.0K，
      // 混进 94 个工具步骤后变成 9.1K/33.7K，而旁边的 cost 仍只按 LLM 算，
      // 同时 eval 面板报的又是 LLM-only 的数——同一轮交互三个答案。
      if (tu.cost_reason === 'non_llm_step') continue;
      stepInTok += tu.input_tokens || 0;
      stepOutTok += tu.output_tokens || 0;
      if (typeof tu.cost_usd === 'number') {
        costSum += tu.cost_usd;
        hasCost = true;
      }
    }
    // 三层兜底，跟 DetailPanel.readTurnUsage 和后端 _extract_turn_usage 完全同链：
    //   1) turn.usage（hook per-turn 真值）
    //   2) LLM step 累加
    //   3) turn.otel.token_usage（enricher turn 级）
    // 第 3 层是后补的：部分 cursor 会话每个 step 的用量都是 0（协议没暴露
    // per-step 真值），前两层加出来是 0，chip 就会显示 ↧0/↥0，而右侧详情面板
    // 走到第 3 层显示 256/792 —— 同一轮两个数字，且 0 那个明显是假的。
    const enricher: any = (turn as any)?.otel?.token_usage || {};
    const enIn = Number(enricher.input_tokens) || 0;
    const enOut = Number(enricher.output_tokens) || 0;

    let inTok: number, outTok: number, truthSource: string;
    if (turnIn > 0 || turnOut > 0) {
      inTok = turnIn; outTok = turnOut; truthSource = turnSource || 'turn_truth';
    } else if (stepInTok > 0 || stepOutTok > 0) {
      inTok = stepInTok; outTok = stepOutTok; truthSource = 'step_aggregate';
    } else {
      inTok = enIn; outTok = enOut; truthSource = 'otel_enricher';
      if (!hasCost && typeof enricher.cost_usd === 'number' && enricher.cost_usd > 0) {
        costSum = enricher.cost_usd;
        hasCost = true;
      }
    }
    return { inTok, outTok, costSum, hasCost, llm, truthSource };
  }, [turn.steps, turn]);

  // v0.17.2 (codebuddy only)：每 turn 真实使用的 model 徽章。优先级：
  //   turn.otel.model → step.otel.model → step.metadata.model_id（首个 assistant）
  // 多模型切换时用 turn.otel.models_seen 做 tooltip。Cursor/Claude 不动，避免
  // 影响它们已稳定的 turn header 渲染。
  const turnModelBadge = useMemo<{
    id: string;
    name: string;
    seen: string[];
    isSession: boolean;
  } | null>(() => {
    if ((cot?.agent_type || '') !== 'codebuddy') return null;
    const turnOtelObj: any = (turn as any).otel || null;
    let modelId: string | null = null;
    let modelName: string | null = null;
    let modelsSeen: string[] = [];
    if (turnOtelObj && typeof turnOtelObj.model === 'string' && turnOtelObj.model) {
      modelId = String(turnOtelObj.model);
      modelName = (typeof turnOtelObj.model_name === 'string' && turnOtelObj.model_name)
        ? String(turnOtelObj.model_name) : null;
      if (Array.isArray(turnOtelObj.models_seen)) {
        modelsSeen = turnOtelObj.models_seen.filter((x: any) => typeof x === 'string');
      }
    }
    if (!modelId) {
      for (const s of turn.steps) {
        const m = ((s as any).metadata || {}).model_id;
        const n = ((s as any).metadata || {}).model_name;
        if (typeof m === 'string' && m) {
          modelId = m;
          if (!modelName && typeof n === 'string' && n) modelName = n;
          break;
        }
      }
    }
    if (!modelId) return null;
    const sessionMain: string | null = ((cot as any)?.otel_view?.model
      ?? (cot as any)?.session_meta?.model_id) ?? null;
    return {
      id: modelId,
      name: modelName || modelId,
      seen: modelsSeen.length ? modelsSeen : [modelId],
      isSession: !!sessionMain && sessionMain === modelId,
    };
  }, [cot?.agent_type, (cot as any)?.otel_view?.model, (cot as any)?.session_meta?.model_id, turn]);

  return (
    <div className="tree-turn" data-turn-index={turn.turn_index}>
      <div
        className={`tree-turn-header ${isTurnSelected ? 'selected' : ''}`}
        onClick={() => {
          onSelect({ kind: 'turn', turn, cot });
          setExpanded(e => !e);
        }}
      >
        <span className="tree-turn-chevron">{expanded ? '▾' : '▸'}</span>
        <span className="tree-turn-icon">{icon}</span>
        <div className="tree-turn-text">
          <span className="tree-turn-label">{label}</span>
          {sub && <span className="tree-turn-sub">{sub}</span>}
        </div>
        <div className="tree-turn-badges">
          <AbTurnStamp role={abRole ?? null} compact />
          {/* v0.17.2 (codebuddy)：本轮真实使用的 model 徽章 */}
          {turnModelBadge && (
            <span
              className="tree-otel-chip tree-otel-chip-model"
              title={
                `本轮模型：${turnModelBadge.name}\nmodel_id：${turnModelBadge.id}` +
                (turnModelBadge.seen.length > 1
                  ? `\n本轮共出现：${turnModelBadge.seen.join(', ')}`
                  : '') +
                (turnModelBadge.isSession ? '' : '\n（与 session 主导模型不同）')
              }
            >
              🧬 {turnModelBadge.name}
              {turnModelBadge.seen.length > 1 && (
                <span className="tree-otel-chip-sub">×{turnModelBadge.seen.length}</span>
              )}
            </span>
          )}
          {/* v0.11.2：本轮 OTel 聚合 —— LLM step / token / cost
              v1.0.4：tooltip 区分 turn 真值 vs step 累加 */}
          {turnOtel.llm > 0 && (
            <span
              className="tree-otel-chip"
              title={
                `本轮 LLM step ×${turnOtel.llm}\n` +
                `输入 ${fmtOtelTokens(turnOtel.inTok)} · 输出 ${fmtOtelTokens(turnOtel.outTok)}\n` +
                (turnOtel.truthSource === 'step_aggregate'
                  ? `来源：step.otel.token_usage 累加（仅 LLM 调用，不含工具步骤）`
                  : turnOtel.truthSource === 'otel_enricher'
                    ? `来源：turn.otel.token_usage（enricher 估算；step 级无真值）`
                    : `来源：turn 真值（${turnOtel.truthSource}）`) +
                (turnOtel.hasCost
                  ? `\n估算 cost ${fmtOtelCost(turnOtel.costSum)}（按字符 token，非 cache-aware）`
                  : '')
              }
            >
              ↧{fmtOtelTokens(turnOtel.inTok)}/↥{fmtOtelTokens(turnOtel.outTok)}
            </span>
          )}
          {turnOtel.hasCost && turnOtel.costSum > 0 && (
            <span
              className="tree-otel-chip tree-otel-chip-cost"
              title="本轮估算 cost（聚合自 step.otel.token_usage.cost_usd）"
            >
              {fmtOtelCost(turnOtel.costSum)}
            </span>
          )}
          {turn.has_error_recovery && <span className="dot dot-red" title="错误恢复" />}
          {turn.strategy_shifts > 0 && <span className="dot dot-yellow" title="策略转换" />}
          {turn.cot_summary && <span className="dot dot-cyan" title="LLM CoT 摘要" />}
          {turnPlanCount > 0 && (
            <span className="tree-turn-plan" title={`本轮更新了 plan ${turnPlanCount} 次`}>
              🗺️ plan ×{turnPlanCount}
            </span>
          )}
          {/* v0.10.0: 本轮发生 mode 转换则显著标识 */}
          {turnModeInfo.transitions.map((m, i) => (
            <span
              key={`mode-${i}`}
              className={`tree-turn-mode tree-turn-mode-${m.target_mode_id}`}
              title={
                m.trigger === 'implicit_back_to_agent'
                  ? `隐式切回 agent 模式（用户确认 plan 后自动）`
                  : `模式切换 → ${m.target_mode_id}\n${m.explanation || ''}`
              }
              onClick={(e) => {
                e.stopPropagation();
                const found = turn.steps.find(s => s.step_index === m.at_step);
                if (found) onSelect({ kind: 'step', step: found, turn });
              }}
            >
              🔀 {m.trigger === 'implicit_back_to_agent' ? '⤷' : '→'} {m.target_mode_id}
            </span>
          ))}
          {/* v0.10.0: 本轮 CreatePlan 提案 */}
          {turnModeInfo.proposals.length > 0 && (
            <span
              className="tree-turn-plan-proposal tree-badge-clickable"
              title={`本轮提交了 ${turnModeInfo.proposals.length} 份 plan 文档（点击跳到首份）`}
              onClick={(e) => {
                e.stopPropagation();
                const target = turn.steps.find(s => s.step_index === turnModeInfo.proposals[0].at_step);
                if (target) onSelect({ kind: 'step', step: target, turn });
              }}
            >
              📋 plan 文档 ×{turnModeInfo.proposals.length}
            </span>
          )}
          {/* v0.15.1: Claude 4 条新时间线在本 turn 的命中徽章。
              点击全部跳到 turn 详情（不区分到具体 step，因为时间线事件没有 step_index）。 */}
          {turnClaudeBadges.subagent > 0 && (
            <span
              className="tree-badge-clickable"
              style={{ background: 'rgba(168,85,247,0.18)', color: '#d8b4fe', padding: '1px 7px', borderRadius: 4, fontSize: 11 }}
              title={`本轮触发了 ${turnClaudeBadges.subagent} 个 Subagent / Task（点击跳到详情节点）`}
              onClick={(e) => { e.stopPropagation(); scrollToClaudeSection('subagent'); }}
            >
              🧬 Subagent ×{turnClaudeBadges.subagent}
            </span>
          )}
          {turnClaudeBadges.compact > 0 && (
            <span
              className="tree-badge-clickable"
              style={{ background: 'rgba(6,182,212,0.18)', color: '#67e8f9', padding: '1px 7px', borderRadius: 4, fontSize: 11 }}
              title={`本轮发生了 ${turnClaudeBadges.compact} 次上下文压缩（点击跳到详情节点）`}
              onClick={(e) => { e.stopPropagation(); scrollToClaudeSection('compact'); }}
            >
              📦 Compact ×{turnClaudeBadges.compact}
            </span>
          )}
          {turnClaudeBadges.permission > 0 && (
            <span
              className="tree-badge-clickable"
              style={{ background: 'rgba(245,158,11,0.18)', color: '#fcd34d', padding: '1px 7px', borderRadius: 4, fontSize: 11 }}
              title={`本轮 ${turnClaudeBadges.permission} 次权限请求/拒绝（点击跳到详情节点）`}
              onClick={(e) => { e.stopPropagation(); scrollToClaudeSection('permission'); }}
            >
              🔐 Perm ×{turnClaudeBadges.permission}
            </span>
          )}
          {turnClaudeBadges.notification > 0 && (
            <span
              className="tree-badge-clickable"
              style={{ background: 'rgba(251,191,36,0.18)', color: '#fcd34d', padding: '1px 7px', borderRadius: 4, fontSize: 11 }}
              title={`本轮 ${turnClaudeBadges.notification} 条通知（点击跳到详情节点）`}
              onClick={(e) => { e.stopPropagation(); scrollToClaudeSection('notification'); }}
            >
              🔔 ×{turnClaudeBadges.notification}
            </span>
          )}
          {turnInvocation.llm > 0 && (
            <span
              className="tree-turn-llm tree-badge-clickable"
              title={`本轮显式调用 LLM 的次数（点击跳到首个 LLM step）`}
              onClick={(e) => {
                e.stopPropagation();
                const list = collectInvocationSteps([turn], 'llm_call');
                if (list.length) onSelect({ kind: 'step', step: list[0].step, turn });
              }}
            >
              🧠 LLM ×{turnInvocation.llm}
            </span>
          )}
          {turnInvocation.rag > 0 && (
            <span
              className="tree-turn-rag tree-badge-clickable"
              title={`本轮 RAG / 知识库查询次数（点击跳到首个 RAG step）`}
              onClick={(e) => {
                e.stopPropagation();
                const list = collectInvocationSteps([turn], 'rag_query');
                if (list.length) onSelect({ kind: 'step', step: list[0].step, turn });
              }}
            >
              📚 RAG ×{turnInvocation.rag}
            </span>
          )}
          {turnInvocation.web > 0 && (
            <span
              className="tree-turn-web tree-badge-clickable"
              title={`本轮 Web Search 次数（点击跳到首个 Web step）`}
              onClick={(e) => {
                e.stopPropagation();
                const list = collectInvocationSteps([turn], 'web_search');
                if (list.length) onSelect({ kind: 'step', step: list[0].step, turn });
              }}
            >
              🔎 Web ×{turnInvocation.web}
            </span>
          )}
          {/* v0.9.0: 本轮文件操作摘要 —— +N 创建 / ✏️ M 改 / 🗑 K 删 */}
          {(turnFileOps.create + turnFileOps.modify + turnFileOps.del) > 0 && (
            <span
              className="tree-turn-fileops"
              title={`本轮文件改动: +${turnFileOps.create} 创建 / ${turnFileOps.modify} 修改 / ${turnFileOps.del} 删除${turnFileOps.exec > 0 ? ` / ▶ ${turnFileOps.exec} 执行脚本` : ''}`}
            >
              {turnFileOps.create > 0 && <span>📝{turnFileOps.create}</span>}
              {turnFileOps.modify > 0 && <span>✏️{turnFileOps.modify}</span>}
              {turnFileOps.del > 0 && <span>🗑{turnFileOps.del}</span>}
              {turnFileOps.exec > 0 && <span>▶{turnFileOps.exec}</span>}
            </span>
          )}
          {turnFileOps.temp > 0 && (
            <span
              className="tree-turn-script tree-badge-clickable"
              title={`本轮 agent 创建了 ${turnFileOps.temp} 个临时脚本（点击跳到首个）`}
              onClick={(e) => {
                e.stopPropagation();
                const target = turn.steps.find(
                  s => s.metadata?.file_op?.kind === 'create' && s.metadata?.file_op?.is_temp
                );
                if (target) onSelect({ kind: 'step', step: target, turn });
              }}
            >
              📜 临时 ×{turnFileOps.temp}
            </span>
          )}
          {/* v0.14.4：本轮使用的模型（OTel 解析到的 dominant model；
              session 跨多个模型时这里就能各 turn 标对） */}
          {(turn as any)?.otel?.model && (turn as any).otel.model !== 'unknown' && (
            <span
              className="tree-turn-model"
              title={`本轮模型：${(turn as any).otel.model}\nprovider=${(turn as any).otel.provider || 'unknown'}\nsource=${(turn as any).otel.model_source || 'unknown'}`}
            >
              🤖 {shortOtelModel((turn as any).otel.model)}
            </span>
          )}
          {dur && <span className="tree-turn-dur" title="本轮总耗时（首条事件到末条事件 wall-clock）">{dur}</span>}
          {totalTokens > 0 && (
            <span className="tree-turn-token">{(totalTokens / 1000).toFixed(1)}K</span>
          )}
        </div>
      </div>

      {expanded && (
        <div className="tree-steps-list">
          {groups.map((g, gi) => {
            // Thinking Phase 折叠节点
            if ('kind' in g && g.kind === 'phase') {
              return (
                <PhaseNode
                  key={`phase-${g.firstIndex}`}
                  phase={g}
                  turn={turn}
                  selectedNode={selectedNode}
                  onSelect={onSelect}
                  treatAsThinking={treatPreToolAsThinking}
                />
              );
            }
            const rg = g as RenderGroup;
            return (
              <div key={rg.leader?.step_index ?? `implicit-${gi}`} className="tree-step-group">
                {rg.leader && (
                  <StepNode
                    step={rg.leader}
                    turn={turn}
                    isSelected={
                      selectedNode?.kind === 'step' &&
                      selectedNode.step.step_index === rg.leader.step_index
                    }
                    onSelect={onSelect}
                    treatAsThinking={treatPreToolAsThinking}
                  />
                )}
                {rg.tools.length > 0 && (
                  <div className={`tree-tool-chain ${rg.leader ? '' : 'tree-tool-chain-orphan'}`}>
                    {buildToolRenderEntries(rg.tools).map((ent, ei) => {
                      if (ent.kind === 'tool-batch') {
                        return (
                          <ToolBatchNode
                            key={`batch-${rg.leader?.step_index ?? gi}-${ei}`}
                            entry={ent}
                            turn={turn}
                            selectedNode={selectedNode}
                            onSelect={onSelect}
                            treatAsThinking={treatPreToolAsThinking}
                          />
                        );
                      }
                      const t = ent.step;
                      if (isClaudeEventStep(t)) {
                        return (
                          <ClaudeEventInline
                            key={t.step_index}
                            step={t}
                            turn={turn}
                            isSelected={
                              selectedNode?.kind === 'step' &&
                              selectedNode.step.step_index === t.step_index
                            }
                            onSelect={onSelect}
                          />
                        );
                      }
                      return (
                        <StepNode
                          key={t.step_index}
                          step={t}
                          turn={turn}
                          isSelected={
                            selectedNode?.kind === 'step' &&
                            selectedNode.step.step_index === t.step_index
                          }
                          onSelect={onSelect}
                          treatAsThinking={treatPreToolAsThinking}
                        />
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}

          {/* v0.16.5: 4 类 hook events（subagent / permission / notification /
              compact）改成按时间线穿插到对应 RenderGroup.tools 里——见
              interleaveClaudeEventsIntoGroups。这里不再渲染末尾大区块。
              turn header 的徽章 onClick → scrollToClaudeSection 仍能命中
              data-claude-section="<kind>" 的首条 inline 节点。 */}
        </div>
      )}
    </div>
  );
}

// ─── Claude Event Inline（v0.16.5）──────────────────────
// 替代 v0.16.4 末尾大区块；按时间线穿插的虚拟 step 在 tools 流里渲染成
// 单行内嵌行，带 emoji 徽章 + 时间戳。data-claude-section 给 turn header
// 徽章 onClick 用来 scrollIntoView 跳到首条对应类型的事件。
function ClaudeEventInline({
  step, turn, isSelected, onSelect,
}: {
  step: ThoughtStep;
  turn: TurnCoT;
  isSelected: boolean;
  onSelect: (n: SelectedNode) => void;
}) {
  const md: any = (step.metadata as any) || {};
  const kind: ClaudeEventKind = md._claude_event_kind;
  const payload: any = md._claude_event_payload || {};
  const t_ms: number = md._t_ms || 0;
  const ts = t_ms ? new Date(t_ms).toLocaleTimeString('zh-CN', { hour12: false }) : '';

  const cfg = (() => {
    if (kind === 'subagent') {
      const phase = payload.phase || '';
      return {
        icon: '🧬', color: '#a855f7', bg: 'rgba(168,85,247,0.10)',
        tag: phase === 'SubagentStart' ? 'Sub▶start' : phase === 'SubagentStop' ? 'Sub◼stop' : (phase || 'Subagent'),
        tagCls: phase === 'SubagentStart' ? 'cl-evt-tag-start' : 'cl-evt-tag-stop',
        main: `${payload.agent_type || 'Subagent'}${payload.sub_agent_id ? ` · ${String(payload.sub_agent_id).slice(0, 8)}` : ''}`,
        sub: payload.summary
          || (payload.duration_ms ? `${(payload.duration_ms / 1000).toFixed(2)}s` : '')
          || (payload.status || ''),
      };
    }
    if (kind === 'permission') {
      return {
        icon: '🔐', color: '#f59e0b', bg: 'rgba(245,158,11,0.10)',
        tag: payload.kind || 'Permission',
        tagCls: payload.kind === 'PermissionDenied' ? 'cl-evt-tag-denied' : 'cl-evt-tag-perm',
        main: payload.tool_name || payload.message || payload.mode || 'permission event',
        sub: payload.source || '',
      };
    }
    if (kind === 'notification') {
      return {
        icon: '🔔', color: '#fbbf24', bg: 'rgba(251,191,36,0.10)',
        tag: payload.kind || 'Notification',
        tagCls: 'cl-evt-tag-notif',
        main: payload.message || '(无消息)',
        sub: payload.tool_name || '',
      };
    }
    // compact
    return {
      icon: '📦', color: '#06b6d4', bg: 'rgba(6,182,212,0.10)',
      tag: payload.phase === 'before' ? 'PreCompact' : payload.phase === 'after' ? 'PostCompact' : 'Compact',
      tagCls: payload.phase === 'before' ? 'cl-evt-tag-pre' : 'cl-evt-tag-post',
      main: `trigger=${payload.trigger || 'auto'}`,
      sub: payload.before_tokens && payload.after_tokens
        ? `${payload.before_tokens}t → ${payload.after_tokens}t`
        : (payload.summary_chars ? `summary ${payload.summary_chars} chars` : ''),
    };
  })();

  return (
    <div
      className={`tree-step-wrapper tree-claude-inline${isSelected ? ' selected' : ''}`}
      data-claude-section={kind}
      data-step-index={step.step_index}
      style={{ background: cfg.bg }}
      onClick={() => onSelect({ kind: 'step', step, turn })}
    >
      <div className={`tree-step tree-claude-inline-row${isSelected ? ' selected' : ''}`}>
        <div className="tree-step-connector" />
        <div className="tree-step-dot" style={{ background: cfg.color }} />
        <span className="tree-step-icon">{cfg.icon}</span>
        <span className="tree-step-label">
          <span
            className={`tree-claude-inline-tag ${cfg.tagCls}`}
            style={{ color: cfg.color, borderColor: cfg.color }}
          >
            {cfg.tag}
          </span>
          <span className="tree-claude-inline-main">{cfg.main}</span>
          {cfg.sub && <span className="tree-claude-inline-sub">{cfg.sub}</span>}
        </span>
        <div className="tree-step-right">
          {ts && <span className="tree-step-dur" title="hook 触发时刻">{ts}</span>}
        </div>
      </div>
    </div>
  );
}

// ─── Tool Batch Node（v0.16.5）──────────────────────────
// 同一组连续 ≥2 个相同 tool_name 的 tool_decision（含其紧邻 tool_execution
// 与中间夹的 permission inline）会被压成一行可展开的批：
//   "🔁 Read ×N · 1.2s..3.4s · 🔐2"
// 用户场景：subagent 大量重复调 Read/Grep/Bash 时不再撑爆树。
function ToolBatchNode({
  entry, turn, selectedNode, onSelect, treatAsThinking,
}: {
  entry: ToolBatchEntry;
  turn: TurnCoT;
  selectedNode: SelectedNode | null;
  onSelect: (n: SelectedNode) => void;
  treatAsThinking: boolean;
}) {
  const [expanded, setExpanded] = useState(false);

  // 如果选中的 step 落在本批，自动展开
  useEffect(() => {
    if (!selectedNode || selectedNode.kind !== 'step') return;
    const idx = selectedNode.step.step_index;
    if (entry.steps.some(s => s.step_index === idx)) setExpanded(true);
  }, [selectedNode, entry.steps]);

  const tsText = entry.firstTs && entry.lastTs && entry.lastTs > entry.firstTs
    ? fmtDuration(entry.lastTs - entry.firstTs)
    : '';

  return (
    <div className="tree-step-wrapper tree-tool-batch">
      <div
        className={`tree-step tree-tool-batch-leader${expanded ? ' tree-tool-batch-open' : ''}`}
        onClick={() => setExpanded(e => !e)}
      >
        <div className="tree-step-connector" />
        <span className="tree-step-chevron">{expanded ? '▾' : '▸'}</span>
        <div className="tree-step-dot tree-tool-batch-dot" />
        <span className="tree-step-icon">🔁</span>
        <span className="tree-step-label">
          <span className="tree-tool-batch-tag">batch</span>
          <span className="tree-tool-batch-name">{entry.toolName}</span>
          <span className="tree-tool-batch-count">×{entry.count}</span>
          {entry.permCount > 0 && (
            <span className="tree-tool-batch-perm" title={`本批夹杂 ${entry.permCount} 个权限事件`}>
              🔐{entry.permCount}
            </span>
          )}
        </span>
        <div className="tree-step-right">
          {tsText && <span className="tree-step-dur">{tsText}</span>}
        </div>
      </div>
      {expanded && (
        <div className="tree-tool-batch-children">
          {entry.steps.map(t => {
            if (isClaudeEventStep(t)) {
              return (
                <ClaudeEventInline
                  key={t.step_index}
                  step={t}
                  turn={turn}
                  isSelected={
                    selectedNode?.kind === 'step' &&
                    selectedNode.step.step_index === t.step_index
                  }
                  onSelect={onSelect}
                />
              );
            }
            return (
              <StepNode
                key={t.step_index}
                step={t}
                turn={turn}
                isSelected={
                  selectedNode?.kind === 'step' &&
                  selectedNode.step.step_index === t.step_index
                }
                onSelect={onSelect}
                treatAsThinking={treatAsThinking}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}

// ─── Thinking Phase 节点（黄色徽章 + 折叠子树）────────────
// 设计动机：连续 ≥2 条 thinking_explicit 在 SpanTree 里平铺会让"思考海"
// 把整棵树撑爆。这里折成一行带徽章的阶段节点：标题 = 首条 thought 首句，
// 元数据 = N 条 / 总时长 / 总字数 / 步号区间。点开后下方显示子树，每条
// thought 都是一个独立的可点击节点，单击会和普通 step 一样把右侧详情切到
// 那条 thought（用 onSelect 推送 SelectedNode.step）。
function PhaseNode({
  phase, turn, selectedNode, onSelect, treatAsThinking = false,
}: {
  phase: ThinkingPhaseGroup;
  turn: TurnCoT;
  selectedNode: SelectedNode | null;
  onSelect: (n: SelectedNode) => void;
  // v0.16.2: 整段 phase 都是 pre_tool_reasoning（Claude 未启用 Extended
  // Thinking 时）时，顶节点用紫色 🧠 而非 💡，跟 Cursor 视觉一致。
  treatAsThinking?: boolean;
}) {
  const [expanded, setExpanded] = useState(false);

  // v0.14.6：被选中的 step 如果在本 phase 折叠组内，自动展开 phase，
  // 让外层 scrollIntoView 能命中真实 DOM。
  useEffect(() => {
    if (!selectedNode || selectedNode.kind !== 'step') return;
    const idx = selectedNode.step.step_index;
    if (phase.thoughts.some(t => t.step_index === idx)) {
      setExpanded(true);
    }
  }, [selectedNode, phase.thoughts]);

  const totalChars = phase.thoughts.reduce(
    (s, t) => s + (t.metadata?.thought_chars ?? t.content?.length ?? 0),
    0,
  );
  const totalDur = phase.thoughts.reduce(
    (s, t) => s + (typeof t.duration_ms === 'number' ? t.duration_ms : 0),
    0,
  );
  const phaseTitle = extractThoughtTitleSpanTree(
    phase.thoughts[0]?.content || '', 90,
  );

  // v0.16.2: 整段 phase 都是 pre_tool_reasoning 且 turn 判定 treatAsThinking 时，
  // 走"Pre-tool Thinking"视觉（紫色大脑）；混合 / 真 ext-thinking 还是黄色💡
  const isPreToolPhase = treatAsThinking && phase.thoughts.every(
    t => t.step_type === 'pre_tool_reasoning',
  );
  const phaseIcon = isPreToolPhase ? '🧠' : '💡';
  const phaseBadgeText = isPreToolPhase ? 'Pre-tool Thinking' : 'Thinking Phase';
  const phaseGroupCls = `tree-step-group tree-phase-group${isPreToolPhase ? ' tree-phase-group-pretool' : ''}`;

  return (
    <div className={phaseGroupCls}>
      <div
        className={`tree-step tree-phase-leader ${expanded ? 'tree-phase-open' : ''}`}
        onClick={() => setExpanded(e => !e)}
      >
        <div className="tree-step-connector" />
        <span className="tree-step-chevron tree-phase-chevron">
          {expanded ? '▾' : '▸'}
        </span>
        <div className="tree-step-dot tree-phase-dot" />
        <span className="tree-step-icon">{phaseIcon}</span>
        <span className="tree-step-label tree-phase-label">
          <span className="tree-phase-badge">{phaseBadgeText}</span>
          {phaseTitle && (
            <span className="tree-phase-title">"{phaseTitle}"</span>
          )}
        </span>
        <div className="tree-step-right">
          <span className="tree-phase-meta">
            {phase.thoughts.length}× thoughts
            {totalDur > 0 && ` · ${fmtDuration(totalDur)}`}
            {` · ${totalChars.toLocaleString()}c`}
          </span>
          <span className="tree-phase-range">
            #{phase.firstIndex}–{phase.lastIndex}
          </span>
        </div>
      </div>
      {expanded && (
        <div className="tree-phase-children">
          {phase.thoughts.map(t => (
            <StepNode
              key={t.step_index}
              step={t}
              turn={turn}
              isSelected={
                selectedNode?.kind === 'step' &&
                selectedNode.step.step_index === t.step_index
              }
              onSelect={onSelect}
              treatAsThinking={treatAsThinking}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ─── 子会话分隔带 ────────────────────────────────────────
// 每一次"用户按回车 → AI 完成回答"对应一个子会话 = 一个 turn。
// 这里渲染一条醒目的分隔带 + 子会话概括 + 单轮质量分，
// 让整棵树读起来像"第 N 次交互的小传"，而不是一堆"调用 Shell"并列。
function SubSessionDivider({
  turn,
  cot,
  onEvalTurn,
  onReferenceEvalTurn,
  evalReport,
  isEvalLoading,
  abRole,
}: {
  turn: TurnCoT;
  cot: SessionCoT;
  onEvalTurn?: (turn: TurnCoT, cot: SessionCoT) => void;
  onReferenceEvalTurn?: (turn: TurnCoT, cot: SessionCoT) => void;
  evalReport?: TurnEvalReport;
  isEvalLoading?: boolean;
  liveCritic?: any | null;
  liveCriticTurn?: any | null;
  abRole?: AbTurnRole;
}) {
  const reportStatus = String(
    evalReport?.judge?.status
    || (evalReport?.eval_panel as any)?.agent_critic?.status
    || ''
  ).toLowerCase();
  const reportStartedRaw = String((evalReport?.judge as any)?.created_at || evalReport?.created_at || '').trim();
  const reportStarted = reportStartedRaw ? Date.parse(reportStartedRaw) : Number.NaN;
  const isStaleRunningReport = (
    (reportStatus === 'queued' || reportStatus === 'running')
    && Number.isFinite(reportStarted)
    && Date.now() - reportStarted >= 10 * 60 * 1000
  );
  const isReportRunning = (reportStatus === 'queued' || reportStatus === 'running') && !isStaleRunningReport;
  const evalState = evalReport
    ? (isReportRunning ? 'running' : (isStaleRunningReport || reportStatus === 'interrupted' ? 'missing' : 'done'))
    : (isEvalLoading ? 'running' : 'missing');
  const evalVerdict = evalReport?.eval_panel?.overall_verdict;
  const color = evalReport
    ? (evalVerdict === 'pass' ? '#22c55e' : '#f59e0b')
    : (evalState === 'running' ? '#38bdf8' : '#94a3b8');
  const evalLabel = evalState === 'done'
    ? 'Eval 完成'
    : (evalState === 'running' ? 'Eval 生成中' : 'Eval 评估');
  const evalTitle = evalState === 'done'
    ? 'Agent Critic 已生成本轮评估。点击可手动重跑。'
    : (evalState === 'running'
      ? 'Agent Critic 正在生成本轮评估报告。'
      : '对当前 trace 执行 Agent Critic 评估。');
  // v0.14.2: 真值时间戳优先，回退到 transcript ts
  const startMs = turn.turn_start_ms_observed;
  const time = startMs
    ? new Date(startMs).toLocaleString('zh-CN', {
        month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
      })
    : (turn.turn_start_time
      ? new Date(turn.turn_start_time).toLocaleString('zh-CN', {
          month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
        })
      : '');
  // v0.14.5：active 时长（已剔除 >5min idle gap）。tooltip 给 wall-clock 跨度
  // 让用户知道两个数字的关系——例如"agent 实际干活 4 分钟，但你 IDE 留了一夜
  // 没关，wall-clock 跨度 1180 分钟"。
  const activeMs = turn.turn_duration_ms_observed ?? turn.turn_duration_ms;
  const wallclockMs = (turn as any).turn_wallclock_span_ms;
  const idleMs = (turn as any).turn_idle_ms;
  const dur = fmtDuration(activeMs);
  const durTitle = (() => {
    const lines = ['本轮活跃耗时（已剔除 >5min 的 IDE 闲置间隙）'];
    if (typeof activeMs === 'number') lines.push(`• 活跃: ${fmtDuration(activeMs)}`);
    if (typeof wallclockMs === 'number' && wallclockMs > (activeMs ?? 0)) {
      lines.push(`• wall-clock 跨度: ${fmtDuration(wallclockMs)}`);
    }
    if (typeof idleMs === 'number' && idleMs > 0) {
      lines.push(`• 闲置剔除: ${fmtDuration(idleMs)}`);
    }
    return lines.join('\n');
  })();
  const toolCount = turn.tool_calls.filter(Boolean).length;
  const topic = turn.interaction_summary
    || (turn.user_query ? turn.user_query.slice(0, 40) : `Turn ${turn.turn_index}`);
  const planCount = (cot.plan_timeline || []).filter(p => p.turn_index === turn.turn_index).length;

  return (
    <div className="tree-subsession-divider">
      {/* v0.14.5：去掉两侧渐变线，让 chip 撑满容器宽度 */}
      <div className="tree-subsession-chip">
        <span className="tree-subsession-index">#{turn.turn_index}</span>
        <span className="tree-subsession-topic">💬 {topic}</span>
        <AbTurnStamp role={abRole ?? null} />
        <TurnTraceExportButton
          sessionId={cot.session_id}
          turnIndex={turn.turn_index}
        />
        <button
          type="button"
          className={`tree-subsession-eval tree-subsession-eval-${evalState} ${evalReport && evalState === 'done' ? 'tree-subsession-eval-done' : ''}`}
          style={{ color, borderColor: `${color}66`, background: `${color}14` }}
          title={evalTitle}
          disabled={isEvalLoading || evalState === 'running'}
          onClick={(e) => {
            e.stopPropagation();
            onEvalTurn?.(turn, cot);
          }}
        >
          <span className="tree-subsession-quality-dot" style={{ background: color }} />
          {evalLabel}
        </button>
        <button
          type="button"
          className="tree-subsession-reference"
          title="为这条 trace 上传答案或评判资料"
          onClick={(e) => {
            e.stopPropagation();
            onReferenceEvalTurn?.(turn, cot);
          }}
        >
          Gold
        </button>
        {planCount > 0 && (
          <span className="tree-subsession-plan" title={`本轮更新 plan ${planCount} 次`}>
            🗺️ ×{planCount}
          </span>
        )}
        {toolCount > 0 && (
          <span className="tree-subsession-metric">🔧 {toolCount}</span>
        )}
        {dur && (
          <span
            className="tree-subsession-metric"
            title={durTitle}
          >
            ⏱ {dur}
          </span>
        )}
        {(turn as any)?.otel?.model && (turn as any).otel.model !== 'unknown' && (
          <span
            className="tree-subsession-model"
            title={`本轮模型：${(turn as any).otel.model}\nprovider=${(turn as any).otel.provider || 'unknown'}\nsource=${(turn as any).otel.model_source || 'unknown'}`}
          >
            🤖 {shortOtelModel((turn as any).otel.model)}
          </span>
        )}
        {time && <span className="tree-subsession-time">{time}</span>}
      </div>
    </div>
  );
}

// ─── 单轮导出按钮 ─────────────────────────────────────────
// 挂在每张交互卡片上，导的就是这一轮的事件。导出顺序与树上看到的一致——
// 两者读的是同一份后端 /timeline 结果。
function TurnTraceExportButton({
  sessionId, turnIndex,
}: {
  sessionId: string; turnIndex: number;
}) {
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState('');

  const download = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setBusy(true);
    setFailed('');
    try {
      await saveDownloadedFile(await api.downloadTurnTrace(sessionId, turnIndex));
    } catch (err: any) {
      // 失败必须看得见。一个点了毫无反应的按钮比报错更难排查。
      const msg = err?.message || String(err);
      setFailed(msg);
      if (msg !== '已取消保存') window.alert(`导出第 ${turnIndex} 轮 trace 失败：${msg}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <button
      type="button"
      className={`tree-subsession-export${failed ? ' tree-subsession-export-failed' : ''}`}
      disabled={busy}
      title={failed
        ? `导出失败：${failed}`
        : `导出第 ${turnIndex} 轮的完整 trace（jsonl：thinking / 工具调用 / plan / 权限，一行一个事件）`}
      onClick={download}
    >
      {busy ? '⏳' : '⬇'} {failed ? '导出失败' : '导出'}
    </button>
  );
}

// ─── 主组件 ───────────────────────────────────────────────
export default function SpanTree({
  cot: rawCot,
  session,
  report,
  selectedNode,
  onSelectNode,
  onEvalTurn,
  onReferenceEvalTurn,
  turnEvalReports = {},
  turnEvalLoadingKey,
  liveCritic,
  liveCriticTurns = {},
  abBaseline,
  abCandidate,
}: Props) {
  const isSessionSelected = selectedNode?.kind === 'session';

  // v0.16.5：Claude 会话顶层拉一次 OTel data，把 OTel 通道独有的工具调用
  // （subagent 内部嵌套的，主 transcript 看不见的）以"虚拟 step"形式注入
  // 主时间线。这样 ToolBatchNode + interleaveClaudeEventsIntoGroups 可以
  // 把上百次 Read/Grep/Bash 自动按时间线穿插并折叠成 "🔁 Read ×N" 的批。
  const [otelData, setOtelData] = useState<ClaudeOtelData | null>(null);
  useEffect(() => {
    if ((rawCot?.agent_type || '') !== 'claude' || !rawCot?.session_id) {
      setOtelData(null);
      return;
    }
    let cancelled = false;
    api.getClaudeOtel(rawCot.session_id).then(d => {
      if (!cancelled) setOtelData(d);
    }).catch(() => {
      // OTel 数据缺失时静默跳过（只影响 ClaudeToolStatsCompact，不影响主链路）
    });
    return () => { cancelled = true; };
  }, [rawCot?.agent_type, rawCot?.session_id]);

  const cot = rawCot;

  // 有序事件流：唯一的顺序真值。后端已经把主步流和 5 条并行时间线按真实
  // 时刻合并好，也把 Claude subagent 内部那些主 transcript 看不见的工具
  // 调用（原生 OTel 通道独有）合了进来。
  const [timeline, setTimeline] = useState<TraceEvent[] | null>(null);
  const [timelineError, setTimelineError] = useState<string>('');
  useEffect(() => {
    if (!cot?.session_id) {
      setTimeline(null);
      return;
    }
    let cancelled = false;
    setTimeline(null);
    setTimelineError('');
    api.getSessionTimeline(cot.session_id).then(d => {
      if (!cancelled) setTimeline(d.events || []);
    }).catch(e => {
      if (!cancelled) setTimelineError(e?.message || String(e));
    });
    return () => { cancelled = true; };
  }, [cot?.session_id]);

  const timelineByTurn = useMemo(() => {
    const map = new Map<number, TraceEvent[]>();
    for (const ev of (timeline || [])) {
      if (typeof ev.turn !== 'number') continue;
      const bucket = map.get(ev.turn);
      if (bucket) bucket.push(ev);
      else map.set(ev.turn, [ev]);
    }
    return map;
  }, [timeline]);

  // v0.14.6：徽章点击/外部跳转后把 SpanTree 滚到对应节点。
  // 思路：StepNode/TurnNode 上挂 data-step-index / data-turn-index，
  //   选中变化后用 rAF 等 React 提交完成（包括 TurnNode/PhaseNode 的
  //   "自动展开" useEffect 跑完）再 scrollIntoView。
  // 用 lastKey 去掉相同节点的重复滚动；若 step DOM 还没出现就重试 5 次。
  const lastSelKeyRef = useRef<string | null>(null);
  useEffect(() => {
    if (!selectedNode) return;
    let key: string;
    let selector: string;
    if (selectedNode.kind === 'step') {
      // v0.19.5: claude hook 事件（subagent / permission / notification / compact）
      // 的虚拟 step_index 用 `-1_000_000 - virtIdx`，virtIdx 在每个 turn 内独立
      // 从 0 计数，所以多 turn 场景下不同 turn 的同号虚拟 step_index 在 DOM 里
      // **同时存在**。document.querySelector 只返回文档里最靠前的那个匹配，
      // 表现就是"点击 subagent / perm / notification 行总是跳到第一个 turn 的
      // 那一行"——看着像"跳顶"。修法：把 selector scope 到当前 turn。
      // （普通 tool step 的 step_index 是后端全局重编号过的，并不会撞车，但
      // 即便加上 turn scope 也不会变更行为，纯防御加固。）
      key = `step:${selectedNode.turn.turn_index}:${selectedNode.step.step_index}`;
      selector = `[data-turn-index="${selectedNode.turn.turn_index}"] [data-step-index="${selectedNode.step.step_index}"]`;
    } else if (selectedNode.kind === 'turn') {
      key = `turn:${selectedNode.turn.turn_index}`;
      selector = `[data-turn-index="${selectedNode.turn.turn_index}"]`;
    } else {
      return;
    }
    if (lastSelKeyRef.current === key) return;
    lastSelKeyRef.current = key;

    let cancelled = false;
    let attempts = 0;
    const tryScroll = () => {
      if (cancelled) return;
      const el = document.querySelector(selector) as HTMLElement | null;
      if (el) {
        // 已经在视口内（用户刚在树里点的）不滚不闪，
        // 只有"目标不在视口"才滚动并高亮闪一下，提示用户跳到这里了。
        const rect = el.getBoundingClientRect();
        const inView = rect.top >= 0 && rect.bottom <= (window.innerHeight || document.documentElement.clientHeight);
        if (!inView) {
          el.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' });
          el.classList.add('tree-scroll-flash');
          setTimeout(() => el.classList.remove('tree-scroll-flash'), 1200);
        }
        return;
      }
      if (attempts < 5) {
        attempts++;
        setTimeout(tryScroll, 80);
      }
    };
    // 双 rAF：等 React commit + 自动展开 useEffect 跑完
    requestAnimationFrame(() => requestAnimationFrame(tryScroll));
    return () => { cancelled = true; };
  }, [selectedNode]);

  // 计算会话总用时
  const totalDuration = useMemo(() => {
    const totalMs = cot.turns.reduce((sum, t) =>
      sum + (t.turn_duration_ms_observed ?? t.turn_duration_ms ?? 0), 0);
    return fmtDuration(totalMs);
  }, [cot.turns]);

  const isParent = cot.is_parent || (cot as any).sub_sessions?.length > 0;
  const subCount = (cot as any).sub_sessions?.length || 0;

  // v0.11.2：从 cot.otel_view 抽 model + cost + cache 三个最重要的微指标
  const otelView = (cot as any).otel_view;
  const otelModel: string | undefined = otelView?.model;
  const otelCost: number | null = otelView?.actual_cost_usd ?? otelView?.totals?.cost_usd ?? null;
  const otelActual = otelView?.actual_token_usage;
  const otelCacheRate = (() => {
    if (!otelActual) return 0;
    const inT = otelActual.input_tokens || 0;
    const cr = otelActual.cache_read_tokens || 0;
    const cw = otelActual.cache_write_tokens || 0;
    const denom = Math.max(0, inT - cr - cw) + cr + cw;
    return denom > 0 ? cr / denom : 0;
  })();

  return (
    <div className="span-tree">
      {/* Session 根节点 */}
      <div
        className={`tree-session-root ${isSessionSelected ? 'selected' : ''}`}
        onClick={() => onSelectNode({ kind: 'session', cot, session, report })}
      >
        <span className="tree-session-icon">🖥️</span>
        <div className="tree-session-text">
          <span className="tree-session-title">{session?.topic || cot.turns[0]?.user_query || 'Claude Code Session'}</span>
          <span className="tree-session-id">{cot.session_id.slice(0, 8)}…</span>
        </div>
        <div className="tree-session-stats">
          {/* v0.11.2：OTel 自动检测 chip —— model + cost + cache hit */}
          {otelModel && (
            <span
              className={`tree-otel-chip ${otelModel === 'unknown' ? 'tree-otel-chip-warn' : ''}`}
              title={`OTel: model=${otelModel}\nsource=${otelView.model_source || '—'}\nprovider=${otelView.provider || '—'}\nagent=${otelView.agent_name || '—'}`}
            >
              🤖 {shortOtelModel(otelModel)}
            </span>
          )}
          {otelCost != null && otelCost > 0 && (
            <span
              className="tree-otel-chip tree-otel-chip-cost"
              title={otelActual?.full_price_cost_usd && otelActual.full_price_cost_usd > otelCost
                ? `OTel cache-aware cost ${fmtOtelCost(otelCost)}\n全价 ${fmtOtelCost(otelActual.full_price_cost_usd)}（cache 省了 ${fmtOtelCost(otelActual.full_price_cost_usd - otelCost)}）`
                : `OTel session 累计 cost`}
            >
              💰 {fmtOtelCost(otelCost)}
            </span>
          )}
          {otelCacheRate > 0 && (
            <span
              className="tree-otel-chip tree-otel-chip-cache"
              title={`cache 命中率 ${(otelCacheRate * 100).toFixed(1)}%\ncache_read ${otelActual?.cache_read_tokens ?? 0} · cache_write ${otelActual?.cache_write_tokens ?? 0}`}
            >
              ⚡ {Math.round(otelCacheRate * 100)}%
            </span>
          )}
          {isParent && <span className="session-parent-badge">📂 {subCount} 次交互</span>}
          {(cot.plan_timeline?.length ?? 0) > 0 && (
            <span
              className="tree-session-plan"
              title={`点我查看 ${cot.plan_timeline!.length} 次 Plan 演进`}
            >
              🗺️ Plan ×{cot.plan_timeline!.length}
            </span>
          )}
          {/* v0.10.0: 模式转换徽标 —— 显示总次数；点击跳到首条 SwitchMode step */}
          {(cot.mode_transitions?.length ?? 0) > 0 && (
            <span
              className="tree-session-mode tree-badge-clickable"
              title={`本会话发生了 ${cot.mode_transitions!.length} 次模式转换（plan ↔ agent）— 点击跳到首条`}
              onClick={(e) => {
                e.stopPropagation();
                const first = cot.mode_transitions![0];
                for (const t of cot.turns) {
                  const found = t.steps.find(s => s.step_index === first.at_step);
                  if (found) {
                    onSelectNode({ kind: 'step', step: found, turn: t });
                    return;
                  }
                }
              }}
            >
              🔀 Mode ×{cot.mode_transitions!.length}
            </span>
          )}
          {/* v0.10.0: Plan 文档徽标 —— CreatePlan 调用次数；点击跳到首份 plan 文档 */}
          {(cot.plan_proposals?.length ?? 0) > 0 && (
            <span
              className="tree-session-plan-doc tree-badge-clickable"
              title={`本会话提交了 ${cot.plan_proposals!.length} 份 plan 文档（CreatePlan）— 点击跳到首份`}
              onClick={(e) => {
                e.stopPropagation();
                const first = cot.plan_proposals![0];
                for (const t of cot.turns) {
                  const found = t.steps.find(s => s.step_index === first.at_step);
                  if (found) {
                    onSelectNode({ kind: 'step', step: found, turn: t });
                    return;
                  }
                }
              }}
            >
              📋 Plan 文档 ×{cot.plan_proposals!.length}
            </span>
          )}
          {(cot.observed_events?.injected ?? 0) > 0 && (
            <span
              className="tree-session-observed"
              title={`cot-stream.js 回灌 ${cot.observed_events!.injected} 条真实输出`}
            >
              📡 {cot.observed_events!.injected}
            </span>
          )}
          {(cot.invocation_stats?.llm_calls ?? 0) > 0 && (
            <span
              className="tree-session-llm tree-badge-clickable"
              title={`显式 LLM 调用 ${cot.invocation_stats!.llm_calls} 次（点击跳到首个 LLM step）`}
              onClick={(e) => {
                e.stopPropagation();
                const list = collectInvocationSteps(cot.turns, 'llm_call');
                if (list.length) onSelectNode({ kind: 'step', step: list[0].step, turn: list[0].turn });
              }}
            >
              🧠 LLM ×{cot.invocation_stats!.llm_calls}
            </span>
          )}
          {(cot.invocation_stats?.rag_queries ?? 0) > 0 && (
            <span
              className="tree-session-rag tree-badge-clickable"
              title={`RAG / 知识库查询 ${cot.invocation_stats!.rag_queries} 次（点击跳到首个 RAG step）`}
              onClick={(e) => {
                e.stopPropagation();
                const list = collectInvocationSteps(cot.turns, 'rag_query');
                if (list.length) onSelectNode({ kind: 'step', step: list[0].step, turn: list[0].turn });
              }}
            >
              📚 RAG ×{cot.invocation_stats!.rag_queries}
            </span>
          )}
          {(cot.invocation_stats?.web_searches ?? 0) > 0 && (
            <span
              className="tree-session-web tree-badge-clickable"
              title={`Web Search ${cot.invocation_stats!.web_searches} 次（点击跳到首个 Web step）`}
              onClick={(e) => {
                e.stopPropagation();
                const list = collectInvocationSteps(cot.turns, 'web_search');
                if (list.length) onSelectNode({ kind: 'step', step: list[0].step, turn: list[0].turn });
              }}
            >
              🔎 Web ×{cot.invocation_stats!.web_searches}
            </span>
          )}
          {/* v0.9.0: 临时脚本计数徽章；点击跳到首个临时脚本的创建 step */}
          {(cot.script_stats?.temp_scripts ?? 0) > 0 && (
            <span
              className="tree-session-script tree-badge-clickable"
              title={`本会话内 agent 创建了 ${cot.script_stats!.temp_scripts} 个临时验证脚本（点击跳到首个）`}
              onClick={(e) => {
                e.stopPropagation();
                const arts = (cot.script_artifacts || []).filter(a => a.is_temp && a.created_at_step);
                if (arts.length === 0) return;
                const targetStep = arts[0].created_at_step!;
                for (const t of cot.turns) {
                  const found = t.steps.find(s => s.step_index === targetStep);
                  if (found) {
                    onSelectNode({ kind: 'step', step: found, turn: t });
                    return;
                  }
                }
              }}
            >
              📜 临时 ×{cot.script_stats!.temp_scripts}
            </span>
          )}
          {totalDuration && <span className="tree-session-duration">⏱ {totalDuration}</span>}
          <span>{cot.turns.length} 子会话</span>
          <span>{cot.total_tool_calls} tools</span>
        </div>
      </div>

      {/* v0.16.4: Claude 会话顶部嵌入 OTel 工具调用统计 ——
          让用户在主 SpanTree 里就看到 OTel 视野下的工具总数（含 subagent 嵌套）；
          这块对 Cursor / 非 Claude 会话不渲染。
          v0.16.5: 复用顶层 otelData，避免重复 fetch。 */}
      {(cot?.agent_type || '') === 'claude' && cot.session_id && (
        <ClaudeToolStatsCompact sessionId={cot.session_id} prefetched={otelData} />
      )}

      {/* 时间线是步骤树的顺序来源，取不到就整棵树都画不出来，
          所以这个错误必须显式告诉用户，而不是留一片空白让人以为没数据。 */}
      {timelineError && (
        <div className="tree-timeline-error">
          ⚠️ 时间线加载失败，步骤树与导出均不可用：{timelineError}
        </div>
      )}

      {/* 子会话流：每个 turn = 一次 user→AI 交互。每个 turn 前面都放一条醒目分隔带。 */}
      <div className="tree-turns-list">
        {cot.turns.map((turn, idx) => {
          const abRole = getAbTurnRole(cot.session_id, turn.turn_index, abBaseline, abCandidate);
          return (
            <div key={turn.turn_index} className="tree-subsession-wrap" data-first={idx === 0}>
              <SubSessionDivider
                turn={turn}
                cot={cot}
                onEvalTurn={onEvalTurn}
                onReferenceEvalTurn={onReferenceEvalTurn}
                evalReport={turnEvalReports[`${cot.session_id}:${turn.turn_index}`]}
                isEvalLoading={turnEvalLoadingKey === `${cot.session_id}:${turn.turn_index}`}
                liveCritic={liveCritic}
                liveCriticTurn={liveCriticTurns[`${cot.session_id}:${turn.turn_index}`]}
                abRole={abRole}
              />
              <TurnNode
                turn={turn}
                cot={cot}
                timeline={timelineByTurn.get(turn.turn_index) ?? (timeline ? [] : null)}
                selectedNode={selectedNode}
                onSelect={onSelectNode}
                abRole={abRole}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
