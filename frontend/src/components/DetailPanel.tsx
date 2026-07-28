import { useState, type ReactNode } from 'react';
import type { SelectedNode } from './SpanTree';
import type {
  ThoughtStep, InvocationCategory, ScriptArtifact, SessionCoT, TurnCoT,
  OtelTokenUsage, TurnEvalReport,
} from '../types';
import { api, saveDownloadedFile } from '../hooks/api';
// v0.16.1: 旧 OtlpExportDialog（hero "🚀 导出到 OTel" 按钮）已从 SessionDetail 移除。
// ClaudeOtelPanel 已搬入右侧 OTel tab（OtelPanel 内）；这里不再需要它们的 import。

interface Props {
  node: SelectedNode | null;
  onSelectNode?: (n: SelectedNode) => void;
  turnEvalReports?: Record<string, TurnEvalReport>;
  liveCritic?: any | null;
  turnEvalLoadingKey?: string | null;
  turnEvalError?: string | null;
}

// ─── 耗时格式化 ──────────────────────────────────────────
function fmtDur(ms?: number): string {
  if (ms == null || ms <= 0) return '';
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(2)}s`;
  return `${(ms / 60000).toFixed(1)}min`;
}

// ─── v0.20.7: 统一 turn 级 token 用量解析 ──────────────────
// 历史问题：DetailPanel 之前只读 turn.usage（来自 Cursor afterAgentResponse
// hook / Claude transcript usage 字段），但：
//   1. Cursor 当 afterAgentResponse hook 未触发时 → turn.usage = {0,0,0,0}
//   2. CodeBuddy 当 turn 未走完时 → 同样可能 = {0,0,0,0}
//   3. Claude 中间 turn 偶发同问题
// 结果：右侧详情面板「Token 消耗」/「所在 Turn Token」一片 0，但 SpanTree
// 上每步明明已经显示了各自 token chip。
//
// 三层 fallback 链（实测数据对比后定的顺序）：
//   1. turn.usage（hook 真值，含 cache 细分，最准 — 当 hook 触发到了）
//   2. 累加 step.otel.token_usage 里 cost_reason != 'non_llm_step' 的所有
//      LLM 调用 — 这跟 SpanTree 上 chip 显示的逐步累加完全一致（$0.341
//      跟 turn header 显示一致），是 turn-level enricher 值不准时的 best-
//      effort 真值。
//   3. turn.otel.token_usage（enricher turn-level 字段，实测发现它有时
//      只反映某次集成式 LLM call 而非整轮累加 — Turn 1 显示 338/1810 但
//      step 累加才是 1950/4159 — 留作 step 全无 OTel 的最后兜底）
type TurnUsageView = {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens: number;
  cache_read_input_tokens: number;
  cost_usd: number | null;
  source: 'hook' | 'step-sum' | 'otel-enricher';
  is_estimate: boolean;
};
function readTurnUsage(turn: TurnCoT): TurnUsageView | null {
  const u: any = turn.usage || {};
  const uIn = Number(u.input_tokens) || 0;
  const uOut = Number(u.output_tokens) || 0;

  // —— 层 1：hook 真值
  if (uIn > 0 || uOut > 0) {
    const o: any = (turn as any)?.otel?.token_usage || {};
    return {
      input_tokens: uIn,
      output_tokens: uOut,
      cache_creation_input_tokens: Number(u.cache_creation_input_tokens) || 0,
      cache_read_input_tokens: Number(u.cache_read_input_tokens) || 0,
      cost_usd: (typeof o.cost_usd === 'number') ? o.cost_usd : null,
      source: 'hook',
      is_estimate: false,
    };
  }

  // —— 层 2：step.otel.token_usage LLM 累加（hook 没触发时的 best-effort 真值）
  let llmIn = 0, llmOut = 0, llmCost = 0;
  let hasLlmStep = false, hasLlmCost = false;
  for (const s of turn.steps || []) {
    const stu: any = (s as any)?.otel?.token_usage;
    if (!stu) continue;
    if (stu.cost_reason === 'non_llm_step') continue;
    llmIn += Number(stu.input_tokens) || 0;
    llmOut += Number(stu.output_tokens) || 0;
    if (typeof stu.cost_usd === 'number') { llmCost += stu.cost_usd; hasLlmCost = true; }
    hasLlmStep = true;
  }
  if (hasLlmStep && (llmIn > 0 || llmOut > 0)) {
    return {
      input_tokens: llmIn,
      output_tokens: llmOut,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
      cost_usd: hasLlmCost ? llmCost : null,
      source: 'step-sum',
      is_estimate: true,
    };
  }

  // —— 层 3：enricher turn-level（数据存在但可能与逐步累加对不上，留作末路兜底）
  const o: any = (turn as any)?.otel?.token_usage || {};
  const oIn = Number(o.input_tokens) || 0;
  const oOut = Number(o.output_tokens) || 0;
  if (oIn > 0 || oOut > 0 || (typeof o.cost_usd === 'number' && o.cost_usd > 0)) {
    return {
      input_tokens: oIn,
      output_tokens: oOut,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
      cost_usd: (typeof o.cost_usd === 'number') ? o.cost_usd : null,
      source: 'otel-enricher',
      is_estimate: !!o.is_estimate,
    };
  }
  return null;
}

// ─── v0.20.7: 单 step token 用量解析（给 StepDetail 「本步 Token」用）─
// 跟 SpanTree.StepTokenBadge 同源逻辑（保持显示一致）。
type StepTokenView = {
  input: number; output: number; cost: number | null;
  is_estimate: boolean; cost_reason?: string; source?: string;
  is_llm_call: boolean;
  is_shared: boolean;
  cache_read?: number; cache_creation?: number;
};
function readStepTokens(step: ThoughtStep): StepTokenView | null {
  const otel: any = (step as any)?.otel;
  const tu = otel?.token_usage;
  if (tu) {
    const reason: string = tu.cost_reason || '';
    const source: string = tu.source || '';
    const input = Number(tu.input_tokens) || 0;
    const output = Number(tu.output_tokens) || 0;
    const rawCost = (typeof tu.cost_usd === 'number') ? tu.cost_usd : null;
    const is_llm = reason !== 'non_llm_step';
    const is_shared = source === 'shared_with_anchor';
    const cost = is_llm ? rawCost : null;
    if (input === 0 && output === 0 && !cost && !is_shared) return null;
    return {
      input, output, cost,
      is_estimate: !!tu.is_estimate,
      cost_reason: reason,
      source,
      is_llm_call: is_llm,
      is_shared,
      cache_read: Number(tu.cache_read_tokens) || 0,
      cache_creation: Number(tu.cache_creation_tokens) || 0,
    };
  }
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

// ─── Step 类型配置（与 SpanTree 保持一致）────────────────
type StepCfgItem = { icon: string; color: string; label: string; desc: string };
const STEP_CFG: Record<string, StepCfgItem> = {
  user_input: { icon: '💬', color: '#3b82f6', label: 'User Input', desc: '用户输入' },
  tool_result_input: { icon: '📥', color: '#6366f1', label: 'Tool Result', desc: '工具返回结果' },
  thinking_inter: { icon: '🧠', color: '#8b5cf6', label: 'Thinking', desc: '中间推理过程' },
  thinking_intermediate: { icon: '🧠', color: '#8b5cf6', label: 'Thinking', desc: '中间推理过程' },
  thinking_explicit: { icon: '🧠', color: '#8b5cf6', label: 'Extended Thinking', desc: '显式思考内容' },
  // v0.20.11: 决策说明 → 紫色大脑统一（跟 SpanTree 视觉一致）
  // 之前用 💡 黄色"决策说明"标签，但跟 thinking_inter/explicit 内容性质完全
  // 一致（都是 LLM 在调工具前的推理输出）。改成 🧠 + 紫色 + 'Thinking'，
  // 配合内容首句作为 SpanTree 节点标签，让用户一眼读到推理内容。
  pre_tool_reasoning: { icon: '🧠', color: '#8b5cf6', label: 'Thinking', desc: '调用工具前的推理' },
  // v0.20.10: tool_decision = 一次完整的 LLM API 调用（CC 源码 llm_request 等价），
  // 它有真实的 input/output token。改成 🧠 紫色大脑 + "LLM Thinking"，让用户
  // 立刻读到"模型在推理然后决定调用工具"，跟 SpanTree 主视觉保持一致。
  tool_decision: { icon: '🧠', color: '#a78bfa', label: 'LLM Thinking', desc: 'LLM 推理后决定调用工具' },
  tool_execution: { icon: '⚙️', color: '#10b981', label: 'Tool Execution', desc: '工具执行结果' },
  strategy_shift: { icon: '🔄', color: '#f59e0b', label: 'Strategy Shift', desc: '策略发生转换' },
  error_recovery: { icon: '⚠️', color: '#ef4444', label: 'Error Recovery', desc: '检测到错误，尝试恢复' },
  final_response: { icon: '📝', color: '#06b6d4', label: 'Final Response', desc: '最终回复' },
};

function getStepCfg(step: ThoughtStep, opts?: { treatPreToolAsThinking?: boolean }) {
  if (step.step_type === 'tool_execution' && step.metadata?.is_error) {
    return { icon: '❌', color: '#ef4444', label: 'Tool Error', desc: '工具执行失败' };
  }
  // v0.15.0: Claude 路径下没启用 Extended Thinking 时，把 pre_tool_reasoning
  // 渲染成 "🧠 Pre-tool Thinking"——这是 transcript 里 Claude 在 tool_use
  // 之前的 text content，本质就是它"轻量思考"的全部。改成 🧠 + 紫色让用户
  // 视觉上能看出来 agent 在想什么，而不是只看到一堆"决策说明"小字。
  if (
    step.step_type === 'pre_tool_reasoning'
    && opts?.treatPreToolAsThinking
  ) {
    return {
      icon: '🧠',
      color: '#a78bfa',
      label: 'Pre-tool Thinking',
      desc: '调用工具前的推理（Claude 未启用 Extended Thinking 时）',
    };
  }
  return STEP_CFG[step.step_type] || { icon: '•', color: '#64748b', label: step.step_type, desc: '' };
}

// ─── v0.11.2 OTel KPI 条 ────────────────────────────────
// SessionDetail / TurnDetail 顶部展示自动检测到的 model · cost · cache · cursor.version
// 数据来源：
//   - session 级：cot.otel_view（actual_token_usage / actual_cost_usd / client_runtime / hints）
//   - turn 级：聚合 turn.steps[*].otel.token_usage 拿 token，model 继承 session
function fmtUsd(usd: number | null | undefined, dp = 4): string {
  if (usd == null) return '—';
  if (usd === 0) return '$0';
  // 小额费用也用普通小数显示（避免 7.50e-4 这种写法）
  if (Math.abs(usd) < 1) {
    const digits = Math.max(dp, 6);
    const s = usd.toFixed(digits).replace(/\.?0+$/, '');
    return `$${s}`;
  }
  return `$${usd.toFixed(2)}`;
}
function fmtNum(n: number | null | undefined): string {
  if (n == null) return '—';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}
function modelSourceLabel(src?: string | null): { text: string; cls: string } {
  switch (src) {
    case 'events': return { text: 'auto · cot-stream', cls: 'otel-badge-good' };
    case 'env': return { text: 'env (.env)', cls: 'otel-badge-info' };
    case 'transcript': return { text: 'transcript', cls: 'otel-badge-info' };
    case 'host': return { text: 'host_tool', cls: 'otel-badge-neutral' };
    case 'client': return { text: 'user_input', cls: 'otel-badge-neutral' };
    case 'synthetic': return { text: 'synthetic', cls: 'otel-badge-neutral' };
    case 'unknown': return { text: 'unknown', cls: 'otel-badge-warn' };
    default: return { text: src || '—', cls: 'otel-badge-neutral' };
  }
}

function SessionOtelKpiBar({ cot }: { cot: SessionCoT }) {
  const otel = (cot as any).otel_view;
  if (!otel) return null;

  const actual = otel.actual_token_usage;
  const runtime = otel.client_runtime;
  const totals = otel.totals || {};

  const inTok = actual?.input_tokens ?? totals.input_tokens ?? 0;
  const outTok = actual?.output_tokens ?? totals.output_tokens ?? 0;
  const cacheRead = actual?.cache_read_tokens ?? 0;
  const cacheWrite = actual?.cache_write_tokens ?? 0;
  const nonCacheIn = Math.max(0, inTok - cacheRead - cacheWrite);
  const denom = nonCacheIn + cacheRead + cacheWrite;
  const cacheHitRate = denom > 0 ? cacheRead / denom : 0;

  const cost = otel.actual_cost_usd ?? totals.cost_usd;
  const fullPrice = actual?.full_price_cost_usd;
  const savings = (fullPrice && cost && fullPrice > cost) ? fullPrice - cost : null;

  const ms = modelSourceLabel(otel.model_source);

  // v0.15.0：IDE 徽章——一眼看出这条 session 来自哪个 IDE。
  // agent_type 由 cot_extractor._detect_agent_type 在 transcript 里识别后写
  // 进 cot.json，前端只是消费。
  // v0.17.0：扩展支持 codebuddy。
  const agentTypeRaw = (cot as any).agent_type;
  const agentBadge = (() => {
    switch (agentTypeRaw) {
      case 'claude':    return { text: '🤖 Claude',    cls: 'otel-badge-info' };
      case 'cursor':    return { text: '🅒 Cursor',    cls: 'otel-badge-good' };
      case 'codebuddy': return { text: '🐤 CodeBuddy', cls: 'otel-badge-codebuddy' };
      case 'unknown':
      case null:
      case undefined:
        return null;
      default:        return { text: `IDE: ${agentTypeRaw}`, cls: 'otel-badge-neutral' };
    }
  })();

  // v0.15.1：Claude 27 hook 触发总数徽章——给一眼能看出 stream hook 装没装。
  const hookKpiBadge = (() => {
    if (agentTypeRaw !== 'claude') return null;
    const heo = ((cot as any).session_meta || {}).hook_events_observed || {};
    const total = Object.values(heo).reduce(
      (s: number, v: any) => s + (Number(v) || 0), 0,
    );
    const distinct = Object.keys(heo).filter(
      k => (heo as any)[k] > 0,
    ).length;
    if (total === 0) {
      return { text: '🪝 hook ×0（未装 stream hook）', cls: 'otel-badge-warn' };
    }
    return { text: `🪝 ${distinct}/27 hook · ${total} 触发`, cls: 'otel-badge-good' };
  })();

  return (
    <div className="dp-otel-kpi">
      <div className="dp-otel-kpi-title">
        🛰️ OpenTelemetry · 自动检测
        <span className={`otel-badge ${ms.cls}`} style={{ marginLeft: 8 }}>{ms.text}</span>
        {agentBadge && (
          <span className={`otel-badge ${agentBadge.cls}`} style={{ marginLeft: 6 }}>
            {agentBadge.text}
          </span>
        )}
        {hookKpiBadge && (
          <span className={`otel-badge ${hookKpiBadge.cls}`} style={{ marginLeft: 6 }}>
            {hookKpiBadge.text}
          </span>
        )}
      </div>
      <div className="dp-otel-kpi-grid">
        <div className="dp-otel-kpi-row">
          <span className="dp-otel-kpi-k">model</span>
          <span className="dp-otel-kpi-v">{otel.model || 'unknown'}</span>
        </div>
        <div className="dp-otel-kpi-row">
          <span className="dp-otel-kpi-k">provider</span>
          <span className="dp-otel-kpi-v">{otel.provider || '—'}</span>
        </div>
        <div className="dp-otel-kpi-row">
          <span className="dp-otel-kpi-k">agent</span>
          <span className="dp-otel-kpi-v">{otel.agent_name || '—'}</span>
        </div>
        <div className="dp-otel-kpi-row">
          <span className="dp-otel-kpi-k">cost · 实际</span>
          <span className="dp-otel-kpi-v" title={actual ? `cache-aware（cache_write 1.25x · cache_read 0.1x · output 1.0x）` : '估算（无 cache 折扣，按字符近似）'}>
            {fmtUsd(cost)}
            {actual && <span style={{ marginLeft: 6, fontSize: 10, color: '#6ee7b7' }}>cache-aware</span>}
          </span>
        </div>
        <div className="dp-otel-kpi-row">
          <span className="dp-otel-kpi-k">输入 tokens</span>
          <span className="dp-otel-kpi-v" title={`非 cache ${fmtNum(nonCacheIn)} · cache_read ${fmtNum(cacheRead)} · cache_write ${fmtNum(cacheWrite)}`}>
            {fmtNum(inTok)}
          </span>
        </div>
        <div className="dp-otel-kpi-row">
          <span className="dp-otel-kpi-k">输出 tokens</span>
          <span className="dp-otel-kpi-v">{fmtNum(outTok)}</span>
        </div>
        {cacheHitRate > 0 && (
          <div className="dp-otel-kpi-row">
            <span className="dp-otel-kpi-k">cache 命中</span>
            <span className="dp-otel-kpi-v" style={{ color: '#d8b4fe' }}>
              ⚡ {(cacheHitRate * 100).toFixed(1)}%
            </span>
          </div>
        )}
        {actual?.agent_response_count != null && (
          <div className="dp-otel-kpi-row">
            <span className="dp-otel-kpi-k">LLM 回复次数</span>
            <span className="dp-otel-kpi-v">{actual.agent_response_count}</span>
          </div>
        )}
        {runtime?.cursor_version && (
          <div className="dp-otel-kpi-row">
            <span className="dp-otel-kpi-k">cursor.version</span>
            <span className="dp-otel-kpi-v">{runtime.cursor_version}</span>
          </div>
        )}
        {runtime?.events_count != null && (
          <div className="dp-otel-kpi-row">
            <span className="dp-otel-kpi-k">events</span>
            <span className="dp-otel-kpi-v" title={runtime.events_path || ''}>
              {runtime.events_count} 条 · cot-stream
            </span>
          </div>
        )}
      </div>
      {savings != null && (
        <div className="dp-otel-kpi-savings">
          ⚡ cache 帮你省了 <code>{fmtUsd(savings, 2)}</code>（全价 {fmtUsd(fullPrice, 2)} → 实际 {fmtUsd(cost, 2)}）
        </div>
      )}
    </div>
  );
}

// v0.12.0：step 级 OTel KPI 条 —— 把每一步的 gen_ai.* 标准 attribute 直接渲染成
// 一眼能看懂的卡片。重点显示本轮新补齐的字段：
//   - gen_ai.tool.name / .call.id / .type  （host_tool / tool_decision）
//   - gen_ai.client.operation.duration       （ms，从 step.duration_ms 透出）
//   - gen_ai.request.model / response.model / finish_reasons
//   - gen_ai.usage.input_tokens / output_tokens / cost.usd
function StepOtelKpiBar({ step }: { step: ThoughtStep }) {
  const otel: any = (step as any).otel;
  if (!otel) return null;

  const attrs: Record<string, any> = otel.attributes || {};
  const tu: OtelTokenUsage | undefined = otel.token_usage;
  const stepKind: string | undefined = otel.step_kind;
  const opName: string = otel.operation_name || attrs['gen_ai.operation.name'] || '—';
  const ms = modelSourceLabel(otel.model_source);

  // 标准 attribute 取数（缺时回退到 step.duration_ms / metadata）
  const model = attrs['gen_ai.request.model'] || otel.model;
  const provider = attrs['gen_ai.provider.name'] || otel.provider;
  const finish = (attrs['gen_ai.response.finish_reasons'] as string[] | undefined)?.[0]
    || otel.finish_reason
    || (otel.finish_reasons as string[] | undefined)?.[0];
  const toolName = attrs['gen_ai.tool.name'] || step.tool_name;
  const toolCallId = attrs['gen_ai.tool.call.id'] || step.tool_use_id;
  const toolType = attrs['gen_ai.tool.type'];
  const durMs = attrs['gen_ai.client.operation.duration']
    ?? step.duration_ms;
  const inTok = tu?.input_tokens ?? attrs['gen_ai.usage.input_tokens'] ?? 0;
  const outTok = tu?.output_tokens ?? attrs['gen_ai.usage.output_tokens'] ?? 0;

  // 让 step kind 决定标题颜色 / icon
  const kindMeta: Record<string, { icon: string; label: string; tone: string }> = {
    llm_call:    { icon: '🧠', label: 'LLM call',     tone: '#a78bfa' },
    host_tool:   { icon: '🔧', label: '宿主工具',     tone: '#f59e0b' },
    user_input:  { icon: '👤', label: '用户输入',     tone: '#06b6d4' },
    agent_event: { icon: '🛰️', label: 'Agent event',  tone: '#94a3b8' },
  };
  const km = kindMeta[stepKind || ''] || { icon: '🛰️', label: 'OpenTelemetry · 本步', tone: '#94a3b8' };

  return (
    <div className="dp-otel-kpi dp-otel-kpi-onecol">
      <div className="dp-otel-kpi-title" style={{ color: km.tone }}>
        {km.icon} OpenTelemetry · {km.label}
        <span className={`otel-badge ${ms.cls}`} style={{ marginLeft: 8 }}>{ms.text}</span>
      </div>
      <div className="dp-otel-kpi-grid">
        <div className="dp-otel-kpi-row" title="OTel 标准 attribute：gen_ai.operation.name">
          <span className="dp-otel-kpi-k">gen_ai.operation.name</span>
          <span className="dp-otel-kpi-v">{opName}</span>
        </div>
        {/* host_tool / tool_decision 才有 tool.* —— 突出显示本轮新补齐的标准属性 */}
        {toolName && (
          <div className="dp-otel-kpi-row" title="OTel 标准 attribute：gen_ai.tool.name">
            <span className="dp-otel-kpi-k">gen_ai.tool.name</span>
            <span className="dp-otel-kpi-v">{toolName}</span>
          </div>
        )}
        {toolCallId && (
          <div className="dp-otel-kpi-row" title="OTel 标准 attribute：gen_ai.tool.call.id">
            <span className="dp-otel-kpi-k">tool.call.id</span>
            <span className="dp-otel-kpi-v dp-otel-kpi-mono">
              {String(toolCallId).slice(0, 12)}…
            </span>
          </div>
        )}
        {toolType && (
          <div className="dp-otel-kpi-row" title="OTel 标准 attribute：gen_ai.tool.type">
            <span className="dp-otel-kpi-k">tool.type</span>
            <span className="dp-otel-kpi-v">{toolType}</span>
          </div>
        )}
        {/* LLM 字段（host_tool 上不会有 model） */}
        {stepKind === 'llm_call' && (
          <>
            <div className="dp-otel-kpi-row">
              <span className="dp-otel-kpi-k">gen_ai.request.model</span>
              <span className="dp-otel-kpi-v">{model || 'unknown'}</span>
            </div>
            <div className="dp-otel-kpi-row">
              <span className="dp-otel-kpi-k">provider.name</span>
              <span className="dp-otel-kpi-v">{provider || '—'}</span>
            </div>
          </>
        )}
        {finish && (
          <div className="dp-otel-kpi-row">
            <span className="dp-otel-kpi-k">finish_reason</span>
            <span className="dp-otel-kpi-v">{finish}</span>
          </div>
        )}
        {/* 本轮新补：duration（ms）—— OTel GenAI metrics 的 client.operation.duration */}
        {durMs != null && Number(durMs) > 0 && (
          <div
            className="dp-otel-kpi-row"
            title="OTel GenAI metric：gen_ai.client.operation.duration（毫秒）"
          >
            <span className="dp-otel-kpi-k">operation.duration</span>
            <span className="dp-otel-kpi-v">
              {Number(durMs) >= 1000
                ? `${(Number(durMs) / 1000).toFixed(2)} s`
                : `${Math.round(Number(durMs))} ms`}
            </span>
          </div>
        )}
        {(inTok > 0 || outTok > 0) && (
          <>
            <div className="dp-otel-kpi-row">
              <span className="dp-otel-kpi-k">usage.input_tokens</span>
              <span className="dp-otel-kpi-v">{fmtNum(Number(inTok))}</span>
            </div>
            <div className="dp-otel-kpi-row">
              <span className="dp-otel-kpi-k">usage.output_tokens</span>
              <span className="dp-otel-kpi-v">{fmtNum(Number(outTok))}</span>
            </div>
          </>
        )}
        {tu && tu.cost_usd != null && (
          <div className="dp-otel-kpi-row" title={`cost_reason: ${tu.cost_reason || 'ok'}`}>
            <span className="dp-otel-kpi-k">usage.cost.usd</span>
            <span className="dp-otel-kpi-v">{fmtUsd(tu.cost_usd)}</span>
          </div>
        )}
        {tu && tu.cost_usd == null && tu.cost_reason && tu.cost_reason !== 'non_llm_step' && (
          <div className="dp-otel-kpi-row">
            <span className="dp-otel-kpi-k">cost.reason</span>
            <span className="dp-otel-kpi-v" style={{ color: '#fbbf24' }}>{tu.cost_reason}</span>
          </div>
        )}
      </div>

      {/* 完整 attribute 折叠（给真正想看 raw OTLP 数据的人） */}
      {Object.keys(attrs).length > 0 && (
        <details className="dp-otel-attrs">
          <summary>查看全部 {Object.keys(attrs).length} 项 OTel attributes</summary>
          <div className="dp-otel-attrs-table">
            {Object.entries(attrs).map(([k, v]) => (
              <div className="dp-otel-attrs-row" key={k}>
                <span className="dp-otel-attrs-k">{k}</span>
                <span className="dp-otel-attrs-v">
                  {typeof v === 'object' ? JSON.stringify(v) : String(v)}
                </span>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

function TurnOtelKpiBar({ cot, turn }: { cot: SessionCoT; turn: TurnCoT }) {
  const sessionOtel = (cot as any).otel_view;
  if (!sessionOtel) return null;

  let inTok = 0, outTok = 0, costSum = 0;
  let hasCost = false, llmCount = 0;
  for (const s of turn.steps) {
    const tu: OtelTokenUsage | undefined = (s as any).otel?.token_usage;
    if (!tu) continue;
    if ((s as any).otel?.step_kind === 'llm_call') llmCount++;
    // 跟 readTurnUsage 用同一条规则：non_llm_step 的 token 是工具入参与结果的
    // 字符估算，模型没产出过，混进来会让本轮 token 虚高（实测一条 cursor turn
    // 从 4.4K/26.0K 涨到 9.1K/33.7K）。
    if ((tu as any).cost_reason === 'non_llm_step') continue;
    inTok += tu.input_tokens || 0;
    outTok += tu.output_tokens || 0;
    if (typeof tu.cost_usd === 'number') {
      costSum += tu.cost_usd;
      hasCost = true;
    }
  }

  const ms = modelSourceLabel(sessionOtel.model_source);

  return (
    <div className="dp-otel-kpi">
      <div className="dp-otel-kpi-title">
        🛰️ OpenTelemetry · 本轮
        <span className={`otel-badge ${ms.cls}`} style={{ marginLeft: 8 }}>{ms.text}</span>
      </div>
      <div className="dp-otel-kpi-grid">
        <div className="dp-otel-kpi-row">
          <span className="dp-otel-kpi-k">model</span>
          <span className="dp-otel-kpi-v">{sessionOtel.model || 'unknown'}</span>
        </div>
        <div className="dp-otel-kpi-row">
          <span className="dp-otel-kpi-k">provider</span>
          <span className="dp-otel-kpi-v">{sessionOtel.provider || '—'}</span>
        </div>
        <div className="dp-otel-kpi-row">
          <span className="dp-otel-kpi-k">本轮输入</span>
          <span className="dp-otel-kpi-v">{fmtNum(inTok)}</span>
        </div>
        <div className="dp-otel-kpi-row">
          <span className="dp-otel-kpi-k">本轮输出</span>
          <span className="dp-otel-kpi-v">{fmtNum(outTok)}</span>
        </div>
        <div className="dp-otel-kpi-row">
          <span className="dp-otel-kpi-k">本轮估算 cost</span>
          <span className="dp-otel-kpi-v" title="按字符级 token 估算的 LLM 调用费用（非 cache-aware；session 级才有真实 cache 价）">
            {hasCost ? fmtUsd(costSum) : '—'}
          </span>
        </div>
        <div className="dp-otel-kpi-row">
          <span className="dp-otel-kpi-k">LLM step</span>
          <span className="dp-otel-kpi-v">{llmCount}</span>
        </div>
      </div>
    </div>
  );
}

// ─── 工具函数 ─────────────────────────────────────────────
function ScoreBar({ label, value }: { label: string; value: number }) {
  const pct = Math.min(100, Math.max(0, value * 100));
  const color = pct >= 80 ? '#10b981' : pct >= 60 ? '#f59e0b' : '#ef4444';
  return (
    <div className="dp-score-row">
      <div className="dp-score-meta">
        <span className="dp-score-label">{label}</span>
        <span className="dp-score-val" style={{ color }}>{pct.toFixed(1)}%</span>
      </div>
      <div className="dp-score-track">
        <div className="dp-score-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

function Kv({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="dp-kv">
      <span className="dp-kv-key">{k}</span>
      <span className="dp-kv-val">{v}</span>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="dp-section">
      <div className="dp-section-title">{title}</div>
      {children}
    </div>
  );
}

// ─── eval 结果导出 ────────────────────────────────────────
// 导的是这一轮报告的全量内容：断言明细、维度打分、safety gate、以及 hook
// 阶段 Agent Critic 的 provenance。用 JSON 而不是 trace 那边的 jsonl ——
// 报告是一个嵌套文档，不是按时间推进的事件流。
function TurnEvalExportButton({
  sessionId, turnIndex,
}: {
  sessionId: string; turnIndex: number;
}) {
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState('');

  const download = async () => {
    setBusy(true);
    setFailed('');
    try {
      await saveDownloadedFile(await api.downloadTurnEval(sessionId, turnIndex));
    } catch (e: any) {
      // 失败必须看得见。一个点了毫无反应的按钮比报错更难排查。
      const msg = e?.message || String(e);
      setFailed(msg);
      if (msg !== '已取消保存') window.alert(`导出评估结果失败：${msg}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <button
      type="button"
      className={`dp-eval-export${failed ? ' dp-eval-export-failed' : ''}`}
      disabled={busy}
      title={failed
        ? `导出失败：${failed}`
        : '导出本轮完整评估结果（断言明细 / 维度面板 / hook 阶段评审）'}
      onClick={download}
    >
      {busy ? '⏳' : '⬇'} {failed ? '导出失败' : '导出'}
    </button>
  );
}

function fmtEvalPct(value?: number | null): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return '--';
  return `${(Math.max(0, Math.min(1, value)) * 100).toFixed(1)}%`;
}

function fmtEvalNum(value?: number | null, digits = 1): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return '--';
  return value.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: value % 1 === 0 ? 0 : Math.min(digits, 2),
  });
}

function EvalScorePill({ value, passed, binary }: { value?: number | null; passed?: boolean; binary?: boolean }) {
  if (binary) {
    return (
      <span className={`dp-eval-binary-icon ${passed ? 'is-pass' : 'is-fail'}`} title={passed ? '通过' : '未通过'}>
        {passed ? '✓' : '×'}
      </span>
    );
  }
  const pct = typeof value === 'number' ? Math.max(0, Math.min(1, value)) * 100 : 0;
  const tone = pct >= 80 ? '#10b981' : pct >= 60 ? '#f59e0b' : '#ef4444';
  return (
    <span className={`dp-eval-pill ${passed ? 'dp-eval-pill-pass' : 'dp-eval-pill-risk'}`} style={{ borderColor: `${tone}66`, color: tone }}>
      {fmtEvalPct(value)}
    </span>
  );
}

/* function LegacyTurnEvalReportCard({
  report,
  isLoading,
  error,
}: {
  report?: TurnEvalReport;
  isLoading?: boolean;
  error?: string | null;
}) {
  if (isLoading && !report) {
    return (
      <Section title="Eval 评估 / Evaluation">
        <div className="dp-eval-panel dp-eval-panel-loading">
          <div className="dp-eval-spinner" />
          <div>
            <div className="dp-eval-title">正在评估当前交互</div>
            <div className="dp-eval-sub">Scoring the current turn trace, tools, tokens, and final output.</div>
          </div>
        </div>
      </Section>
    );
  }

  if (error && !report) {
    return (
      <Section title="Eval 评估 / Evaluation">
        <div className="dp-eval-panel dp-eval-panel-error">
          <div className="dp-eval-title">评估失败 / Eval failed</div>
          <div className="dp-eval-sub">{error}</div>
        </div>
      </Section>
    );
  }

  if (!report) {
    return (
      <Section title="Eval 评估 / Evaluation">
        <div className="dp-eval-panel dp-eval-empty">
          <div className="dp-eval-title">尚未评估这一轮</div>
          <div className="dp-eval-sub">点击左侧紫色交互条上的 Eval，即可在这里生成当前 #N 的评估报告。</div>
        </div>
      </Section>
    );
  }

  const metrics = report.metrics || {};
  const totalTokens = Number(metrics.total_tokens || 0);
  const inputTokens = Number(metrics.input_tokens || 0);
  const outputTokens = Number(metrics.output_tokens || 0);
  const cacheRead = Number(metrics.cache_read_tokens || 0);
  const cacheWrite = Number(metrics.cache_write_tokens || 0);
  const tps = typeof metrics.tokens_per_second === 'number' ? metrics.tokens_per_second : null;
  const toolCount = Number(metrics.tool_count || 0);
  const uniqueToolCount = Number(metrics.unique_tool_count || 0);
  const mcpToolCount = Number(metrics.mcp_tool_count || 0);
  const ragToolCount = Number(metrics.rag_tool_count || 0);
  const retrievalToolCount = Number(metrics.retrieval_tool_count || 0);
  const searchToolCount = Number(metrics.search_tool_count || 0);
  const passRate = typeof report.assertion_pass_rate === 'number' ? report.assertion_pass_rate : report.overall_score;
  const isV3 = report.eval_version === 'v3' || Boolean(report.assertion_results?.length);
  const assertionGroups = report.assertion_groups || [];
  const criticalFailures = report.critical_failures || [];
  const sortedScores = [...(report.scores || [])].sort((a, b) => (a.score ?? 0) - (b.score ?? 0));
  const attention = sortedScores.slice(0, 4);
  const strongest = [...sortedScores].reverse().slice(0, 3);

  if (!isV3) {
    return (
      <Section title="Eval 评估 / Evaluation">
        <div className="dp-eval-panel dp-eval-empty">
          <div className="dp-eval-title">旧版评估缓存 / Legacy eval cache</div>
          <div className="dp-eval-sub">
            当前保存的是旧版 Quality Score 报告，已不再作为 V3 结果展示。点击左侧本轮的 Eval 评估重新生成 assertion-based report。
          </div>
        </div>
      </Section>
    );
  }

  return (
    <Section title="Eval 评估 / Evaluation">
      <div className="dp-eval-panel">
        <div className="dp-eval-hero">
          <div>
            <div className="dp-eval-eyebrow">Turn #{report.turn_index} {isV3 ? 'V3 Eval Verdict' : 'Eval Verdict'}</div>
            <div className="dp-eval-score">{fmtEvalPct(passRate)}</div>
            <div className="dp-eval-status">
              {report.passed ? 'Pass' : 'Needs attention'} · {isV3 ? 'Assertion pass rate' : 'Score'}
            </div>
          </div>
          <div className="dp-eval-ring" style={{ ['--score' as any]: `${Math.max(0, Math.min(100, passRate * 100))}%` }}>
            <span>{Math.round(passRate * 100)}</span>
          </div>
        </div>

        {isV3 && (
          <div className="dp-eval-profile">
            <div className="dp-eval-profile-main">
              <span>Task Profile</span>
              <strong>{report.task_profile?.primary || 'general'}</strong>
              {typeof report.task_profile?.confidence === 'number' && (
                <em>{fmtEvalPct(report.task_profile.confidence)} confidence</em>
              )}
            </div>
            <div className="dp-eval-tags">
              {taskLabels.map(label => <span key={label}>{label}</span>)}
              {report.assertion_set?.version && <span>{report.assertion_set.version}</span>}
              {report.assertion_set?.total_assertions != null && <span>{report.assertion_set.total_assertions} assertions</span>}
            </div>
          </div>
        )}

        <div className="dp-eval-note">
          <div>{report.summary?.zh || 'V3 Eval 基于声明式断言评分，可用于自动化评估和 A/B 对比。'}</div>
          <div>{report.summary?.en || 'V3 Eval scores trace assertions for automated eval and A/B comparison.'}</div>
        </div>

        <div className="dp-eval-kpis">
          <div className="dp-eval-kpi">
            <span>Input Tokens</span>
            <strong>{fmtEvalNum(inputTokens, 0)}</strong>
          </div>
          <div className="dp-eval-kpi">
            <span>Output Tokens</span>
            <strong>{fmtEvalNum(outputTokens, 0)}</strong>
          </div>
          <div className="dp-eval-kpi">
            <span>Total Tokens</span>
            <strong>{fmtEvalNum(totalTokens, 0)}</strong>
          </div>
          <div className="dp-eval-kpi">
            <span>Tokens/s</span>
            <strong>{fmtEvalNum(tps, 2)}</strong>
          </div>
          <div className="dp-eval-kpi">
            <span>Cache Read</span>
            <strong>{fmtEvalNum(cacheRead, 0)}</strong>
          </div>
          <div className="dp-eval-kpi">
            <span>Cache Write</span>
            <strong>{fmtEvalNum(cacheWrite, 0)}</strong>
          </div>
        </div>

        {criticalFailures.length > 0 && (
          <div className="dp-eval-critical">
            <div className="dp-eval-list-title">Blocking Failures</div>
            {criticalFailures.map(item => (
              <div key={item.key} className="dp-eval-failure-row">
                <strong>{item.key}</strong>
                <span>{item.severity}</span>
                <p>{item.reason || item.reason_en || item.reason_zh}</p>
              </div>
            ))}
          </div>
        )}

        {isV3 && assertionGroups.length > 0 ? (
          <div className="dp-eval-breakdown">
            {assertionGroups.map(group => (
              <div className="dp-eval-group" key={group.key}>
                <div className="dp-eval-group-head">
                  <div>
                    <div className="dp-eval-list-title">{group.label}</div>
                    <div className="dp-eval-dim-en">{group.key}</div>
                  </div>
                  <strong>{group.passed}/{group.total}</strong>
                </div>
                <div className="dp-eval-assertions">
                  {group.items.map(item => (
                    <div className={`dp-eval-assertion ${item.passed ? 'is-pass' : item.skipped ? 'is-skip' : 'is-fail'}`} key={item.key}>
                      <div className="dp-eval-assertion-top">
                        <span>{item.key}</span>
                        <div className="dp-eval-assertion-badges">
                          <em>{item.skipped ? 'SKIP' : item.passed ? 'PASS' : 'FAIL'}</em>
                          <em>{item.severity || 'medium'}</em>
                          <EvalScorePill value={item.skipped ? null : item.score} passed={item.passed || item.skipped} binary={Boolean(item.binary) && !item.quantitative} />
                        </div>
                      </div>
                      <div className="dp-eval-reason">{item.reason || item.reason_en || item.reason_zh}</div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="dp-eval-breakdown">
            {(report.scores || []).map(item => (
              <div className="dp-eval-dim" key={item.key}>
                <div className="dp-eval-dim-head">
                  <div>
                    <div className="dp-eval-dim-title">{item.label_zh}</div>
                    <div className="dp-eval-dim-en">{item.label_en}</div>
                  </div>
                  <EvalScorePill value={item.score} passed={item.passed} />
                </div>
                <div className="dp-eval-track">
                  <div className="dp-eval-fill" style={{ width: `${Math.max(0, Math.min(1, item.score)) * 100}%` }} />
                </div>
                <div className="dp-eval-reason">
                  <span>{item.reason_zh}</span>
                  <span>{item.reason_en}</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {!isV3 && (
          <div className="dp-eval-columns">
            <div>
              <div className="dp-eval-list-title">优势 / Strengths</div>
              {strongest.map(item => (
                <div key={item.key} className="dp-eval-mini-row">
                  <span>{item.label_zh}</span>
                  <strong>{fmtEvalPct(item.score)}</strong>
                </div>
              ))}
            </div>
            <div>
              <div className="dp-eval-list-title">风险 / Watchlist</div>
              {attention.map(item => (
                <div key={item.key} className="dp-eval-mini-row">
                  <span>{item.label_zh}</span>
                  <strong>{fmtEvalPct(item.score)}</strong>
                </div>
              ))}
            </div>
          </div>
        )}

        {false && isV3 && (
          <div className="dp-eval-pipeline">
            <div className="dp-eval-pipeline-block">
              <div className="dp-eval-list-title">Automation</div>
              <div className="dp-eval-mini-row"><span>Ready</span><strong>{automationReady?.ready ? 'yes' : 'no'}</strong></div>
              <div className="dp-eval-mini-row"><span>Trials</span><strong>{automationReady?.recommended_trials ?? '--'}</strong></div>
              <div className="dp-eval-mini-row"><span>Threshold</span><strong>{fmtEvalPct(automationReady?.pass_threshold)}</strong></div>
            </div>
            <div className="dp-eval-pipeline-block">
              <div className="dp-eval-list-title">A/B Testing</div>
              <div className="dp-eval-mini-row"><span>Metric</span><strong>{abTesting?.primary_metric || 'assertion_pass_rate'}</strong></div>
              <div className="dp-eval-tags">
                {(abTesting?.secondary_metrics || []).map(dim => <span key={dim}>{dim}</span>)}
              </div>
            </div>
            <div className="dp-eval-pipeline-block">
              <div className="dp-eval-list-title">A/B Metadata</div>
              <div className="dp-eval-mini-row"><span>Status</span><strong>{abTesting?.requires_pair ? 'ready' : 'off'}</strong></div>
              {(abTesting?.secondary_metrics || []).slice(0, 4).map(rule => (
                <div key={rule} className="dp-eval-rule">{rule}</div>
              ))}
            </div>
          </div>
        )}

        {!isV3 && report.ab_ready_dimensions && report.ab_ready_dimensions.length > 0 && (
          <div className="dp-eval-ab">
            <div className="dp-eval-list-title">A/B Dimensions</div>
            <div className="dp-eval-tags">
              {report.ab_ready_dimensions.map(dim => <span key={dim}>{dim}</span>)}
            </div>
          </div>
        )}

        {isV3 && report.judge && (
          <div className="dp-eval-ab">
            <div className="dp-eval-list-title">Agent Critic</div>
            <div className="dp-eval-mini-row"><span>Status</span><strong>{report.judge.status || 'unknown'}</strong></div>
            {report.judge.model && <div className="dp-eval-mini-row"><span>Model</span><strong>{report.judge.model}</strong></div>}
            {report.judge.reason && <div className="dp-eval-sub">{report.judge.reason}</div>}
          </div>
        )}

        {report.lineage && (
          <div className="dp-eval-lineage">
            <div>{report.lineage.implementation_note_zh}</div>
            <div>{report.lineage.implementation_note_en}</div>
            {report.lineage.letsgoagenteval_retained?.length ? (
              <div className="dp-eval-tags">
                {report.lineage.letsgoagenteval_retained.map(item => <span key={item}>{item}</span>)}
              </div>
            ) : null}
          </div>
        )}
      </div>
    </Section>
  );
}

*/
function TurnEvalReportCard({
  report,
  isLoading,
  error,
  liveCritic,
  turn,
}: {
  report?: TurnEvalReport;
  isLoading?: boolean;
  error?: string | null;
  liveCritic?: any | null;
  turn?: TurnCoT;
}) {
  if (isLoading && !report) {
    return (
      <Section title="评估">
        <div className="dp-eval-panel dp-eval-panel-loading">
          <div className="dp-eval-spinner" />
          <div>
            <div className="dp-eval-title">正在评估当前交互</div>
            <div className="dp-eval-sub">正在检查最终回答、执行完整性、工具使用、Token 与 Agent Critic 评审。</div>
          </div>
        </div>
      </Section>
    );
  }

  if (error && !report) {
    return (
      <Section title="评估">
        <div className="dp-eval-panel dp-eval-panel-error">
          <div className="dp-eval-title">评估失败</div>
          <div className="dp-eval-sub">{error}</div>
        </div>
      </Section>
    );
  }

  if (!report) {
    return (
      <Section title="评估">
        <div className="dp-eval-panel dp-eval-empty">
          <div className="dp-eval-pending-body">
            <div className="dp-eval-title">尚未评估当前 trace</div>
            <div className="dp-eval-sub">点击左侧本轮 Eval 生成 Agent Critic 结果。</div>
          </div>
        </div>
      </Section>
    );
  }

  const metrics = report.metrics || {};
  const totalTokens = Number(metrics.total_tokens || 0);
  const inputTokens = Number(metrics.input_tokens || 0);
  const outputTokens = Number(metrics.output_tokens || 0);
  const cacheRead = Number(metrics.cache_read_tokens || 0);
  const cacheWrite = Number(metrics.cache_write_tokens || 0);
  const tps = typeof metrics.tokens_per_second === 'number' ? metrics.tokens_per_second : null;
  const toolCount = Number(metrics.tool_count || 0);
  const toolKindCount = Number(metrics.tool_kind_count || 0);
  const mcpToolCount = Number(metrics.mcp_tool_count || 0);
  const ragToolCount = Number(metrics.rag_tool_count || 0);
  const retrievalToolCount = Number(metrics.retrieval_tool_count || 0);
  const searchToolCount = Number(metrics.search_tool_count || 0);
  const toolNameCounts = (metrics.tool_name_counts || {}) as Record<string, number>;
  const toolNameRows = Object.entries(toolNameCounts).filter(([, count]) => Number(count) > 0).slice(0, 20);
  const errorBreakdown = (metrics.error_breakdown || {}) as Record<string, number>;
  const errorSamples = Array.isArray(metrics.error_samples) ? metrics.error_samples : [];
  const toolErrorByTool = (metrics.tool_error_by_tool || {}) as Record<string, number>;
  const passRate = typeof report.assertion_pass_rate === 'number' ? report.assertion_pass_rate : report.overall_score;
  const isV3 = report.eval_version === 'v3' || Boolean(report.assertion_results?.length);
  const assertionGroups = report.assertion_groups || [];
  const criticalFailures = report.critical_failures || [];
  const judgeStatus = String(report.judge?.status || '').toLowerCase();
  const judgeIncomplete = judgeStatus === 'queued' || judgeStatus === 'running' || judgeStatus === 'interrupted';
  const judgeStructured = judgeIncomplete ? {} : report.judge?.structured;
  const liveSupervisor = (report.judge as any)?.live_supervisor || liveCritic;
  const criticSummary = judgeIncomplete
    ? ''
    : String(judgeStructured?.summary_conclusion || report.judge?.summary_conclusion || '').trim();
  const rawJudgeReview = String(judgeStructured?.review_markdown || judgeStructured?.review || (judgeIncomplete ? '' : report.judge?.reason) || '')
    .replace(/部分解决/g, '仍有待核实的缺口')
    .replace(/已解决/g, '交付路径较完整')
    .replace(/未解决/g, '交付证据不足')
    .trim();
  const judgeReview = normalizeCriticReview(rawJudgeReview, criticSummary);
  const provenanceText = buildEvalProvenance(report, liveSupervisor, turn);

  const severityLabel = (severity?: string) => {
    if (severity === 'critical') return '阻断';
    if (severity === 'high') return '高';
    if (severity === 'medium') return '中';
    if (severity === 'low') return '低';
    return severity || '中';
  };
  if (!isV3) {
    return (
      <Section title="评估">
        <div className="dp-eval-panel dp-eval-empty">
          <div className="dp-eval-title">旧版评估缓存</div>
          <div className="dp-eval-sub">当前保存的是旧版评估缓存。请点击左侧本轮 Eval 重新生成新的评估报告。</div>
        </div>
      </Section>
    );
  }

  return (
    <Section title="评估">
      <div className="dp-eval-panel">
        <div className="dp-eval-dashboard-head">
          <div>
            <div className="dp-eval-eyebrow">第 {report.turn_index} 轮</div>
            <div className="dp-eval-title">
              Agent 评估维度面板
              <TurnEvalExportButton
                sessionId={report.session_id}
                turnIndex={report.turn_index}
              />
            </div>
            <div className="dp-eval-status">{provenanceText}</div>
          </div>
        </div>

        <div className="dp-eval-kpis">
          <div className="dp-eval-kpi"><span>输入 Token</span><strong>{fmtEvalNum(inputTokens, 0)}</strong></div>
          <div className="dp-eval-kpi"><span>输出 Token</span><strong>{fmtEvalNum(outputTokens, 0)}</strong></div>
          <div className="dp-eval-kpi"><span>总 Token</span><strong>{fmtEvalNum(totalTokens, 0)}</strong></div>
          <div className="dp-eval-kpi"><span>Tokens/s</span><strong>{fmtEvalNum(tps, 2)}</strong></div>
          <div className="dp-eval-kpi"><span>缓存读取</span><strong>{fmtEvalNum(cacheRead, 0)}</strong></div>
          <div className="dp-eval-kpi"><span>缓存写入</span><strong>{fmtEvalNum(cacheWrite, 0)}</strong></div>
          <div className="dp-eval-kpi"><span>工具调用</span><strong>{fmtEvalNum(toolCount, 0)}</strong></div>
          <div className="dp-eval-kpi"><span>工具种类</span><strong>{fmtEvalNum(toolKindCount, 0)}</strong></div>
          <div className="dp-eval-kpi"><span>断言通过率</span><strong>{fmtEvalPct(passRate)}</strong></div>
          <div className="dp-eval-kpi"><span>MCP</span><strong>{fmtEvalNum(mcpToolCount, 0)}</strong></div>
          <div className="dp-eval-kpi"><span>RAG</span><strong>{fmtEvalNum(ragToolCount, 0)}</strong></div>
          <div className="dp-eval-kpi"><span>检索</span><strong>{fmtEvalNum(retrievalToolCount, 0)}</strong></div>
          <div className="dp-eval-kpi"><span>搜索</span><strong>{fmtEvalNum(searchToolCount, 0)}</strong></div>
        </div>

        {toolNameRows.length > 0 && (
          <details className="dp-eval-details" open>
            <summary>工具使用明细</summary>
            <div className="dp-eval-tool-list">
              {toolNameRows.map(([name, count]) => (
                <div className="dp-eval-tool-row" key={name}>
                  <span>{name}</span>
                  <strong>{fmtEvalNum(Number(count), 0)}</strong>
                </div>
              ))}
            </div>
          </details>
        )}

        {criticalFailures.length > 0 && (
          <div className="dp-eval-critical">
            <div className="dp-eval-list-title">阻断问题</div>
            {criticalFailures.map(item => (
              <div key={item.key} className="dp-eval-failure-row">
                <strong>{item.label_zh || item.key}</strong>
                <span>{severityLabel(item.severity)}</span>
                <p>{item.reason_zh || item.reason}</p>
              </div>
            ))}
          </div>
        )}

        {assertionGroups.length > 0 && (
          <div className="dp-eval-breakdown">
            {assertionGroups.map(group => (
              <div className="dp-eval-group" key={group.key}>
                <div className="dp-eval-group-head">
                  <div>
                    <div className="dp-eval-list-title">{group.label}</div>
                    <div className="dp-eval-dim-en">{group.key}</div>
                  </div>
                  <strong>{group.passed}/{group.total}</strong>
                </div>
                <div className="dp-eval-assertions">
                  {group.items.map(item => (
                    <div className={`dp-eval-assertion ${item.passed ? 'is-pass' : item.skipped ? 'is-skip' : 'is-fail'}`} key={item.key}>
                      <div className="dp-eval-assertion-top">
                        <span>{item.label_zh || item.name || item.key}</span>
                        <div className="dp-eval-assertion-badges">
                          <em>{item.skipped ? '跳过' : item.passed ? '通过' : '失败'}</em>
                          <em>{severityLabel(item.severity)}</em>
                          {item.skipped ? (
                            <span className="dp-eval-skip-pill">跳过</span>
                          ) : (
                            <EvalScorePill value={item.score} passed={item.passed} binary={Boolean(item.binary) && !item.quantitative} />
                          )}
                        </div>
                      </div>
                      <div className="dp-eval-reason">{item.reason_zh || item.reason}</div>
                      {item.type === 'no-error' && Object.keys(errorBreakdown).length > 0 && (
                        <details className="dp-eval-evidence">
                          <summary>错误明细</summary>
                          <div className="dp-eval-mini-grid">
                            {Object.entries(errorBreakdown).map(([key, count]) => (
                              <div className="dp-eval-mini-row" key={key}>
                                <span>{({
                                  tool_errors: '工具执行错误',
                                  error_recovery_steps: '错误恢复步骤',
                                  turn_error_recovery_flag: '本轮恢复标记',
                                  final_response_error_terms: '最终回答错误词',
                                  step_text_error_terms: '步骤文本错误词',
                                } as Record<string, string>)[key] || key}</span>
                                <strong>{fmtEvalNum(Number(count), 0)}</strong>
                              </div>
                            ))}
                          </div>
                          {Object.keys(toolErrorByTool).length > 0 && (
                            <div className="dp-eval-tool-list">
                              {Object.entries(toolErrorByTool).map(([name, count]) => (
                                <div className="dp-eval-tool-row" key={name}>
                                  <span>{name}</span>
                                  <strong>{fmtEvalNum(Number(count), 0)}</strong>
                                </div>
                              ))}
                            </div>
                          )}
                          {errorSamples.length > 0 && (
                            <div className="dp-eval-error-samples">
                              {errorSamples.slice(0, 6).map((sample: any, idx: number) => (
                                <div className="dp-eval-error-sample" key={`${sample.step_index || idx}-${idx}`}>
                                  <strong>{sample.source === 'recovery' ? '恢复' : '错误'} · {sample.tool || 'unknown'} · step {sample.step_index ?? '-'}</strong>
                                  <p>{sample.message || '未捕获到错误文本'}</p>
                                </div>
                              ))}
                            </div>
                          )}
                        </details>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {report.judge && (
          <div className="dp-eval-ab">
            <div className="dp-eval-list-title">Agent Critic</div>
            {report.judge.verdict && <div className="dp-eval-mini-row"><span>结论</span><strong>{report.judge.verdict}</strong></div>}
            {report.judge.model && <div className="dp-eval-mini-row"><span>模型</span><strong>{report.judge.model}</strong></div>}
            {report.judge.status && <div className="dp-eval-mini-row"><span>状态</span><strong>{report.judge.status}</strong></div>}
            {liveSupervisor && <LiveCriticCard liveCritic={liveSupervisor} />}
            {criticSummary && <div className="dp-critic-summary-box">{criticSummary}</div>}
            {judgeReview ? (
              <div className="dp-judge-review-box">
                <JudgeReviewMarkdown
                  text={judgeReview}
                  evidenceByDimension={collectCriticEvidence(judgeStructured)}
                />
              </div>
            ) : report.judge.status !== 'completed' ? (
              <div className="dp-eval-sub">{report.judge.reason}</div>
            ) : null}
            <CriticClaimsCard claims={(judgeStructured as any)?.claims} />
          </div>
        )}
      </div>
    </Section>
  );
}

const CRITIC_EVIDENCE_DIMENSIONS: Array<{ key: string; label: string }> = [
  { key: 'task_completion', label: '任务完成' },
  { key: 'tool_use', label: '工具使用' },
  { key: 'reasoning', label: '推理路径' },
  { key: 'instruction_following', label: '指令遵循' },
  { key: 'faithfulness', label: '忠实度' },
  { key: 'efficiency', label: '效率' },
  { key: 'reliability', label: '可靠性' },
];

type CriticEvidence = { ref?: string; quote?: string; source?: string };

function collectCriticEvidence(structured: any): Map<string, CriticEvidence[]> {
  const out = new Map<string, CriticEvidence[]>();
  if (!structured || typeof structured !== 'object') return out;
  for (const { key, label } of CRITIC_EVIDENCE_DIMENSIONS) {
    const dim = structured[key];
    if (!dim || typeof dim !== 'object') continue;
    const ev = Array.isArray(dim.evidence)
      ? (dim.evidence as CriticEvidence[]).filter((e: any) => e && (e.ref || e.quote))
      : [];
    if (!ev.length) continue;
    out.set(label, ev.slice(0, 6));
  }
  return out;
}

function CriticClaimsCard({ claims }: { claims: any }) {
  if (!Array.isArray(claims) || !claims.length) return null;
  const verifiedLabel = (v: unknown): { label: string; tone: string } => {
    if (v === true) return { label: '已验证', tone: 'ok' };
    if (v === false) return { label: '反例', tone: 'fail' };
    return { label: '未确证', tone: 'unknown' };
  };
  const typeLabel = (t: unknown) => {
    const text = String(t || '').toLowerCase();
    if (text === 'factual') return '事实';
    if (text === 'process') return '流程';
    if (text === 'quality') return '质量';
    return '其他';
  };
  return (
    <details className="dp-critic-claims" open>
      <summary>Agent 声称核对（共 {claims.length} 条）</summary>
      <div className="dp-critic-claims-list">
        {claims.slice(0, 12).map((c: any, idx: number) => {
          const v = verifiedLabel(c?.verified);
          const evidence = Array.isArray(c?.evidence) ? c.evidence.filter((e: any) => e && (e.ref || e.quote)) : [];
          return (
            <div className={`dp-critic-claim is-${v.tone}`} key={idx}>
              <div className="dp-critic-claim-head">
                <span className="dp-critic-claim-type">{typeLabel(c?.type)}</span>
                <span className={`dp-critic-claim-verdict is-${v.tone}`}>{v.label}</span>
              </div>
              <div className="dp-critic-claim-text">{c?.claim || ''}</div>
              {evidence.length > 0 && (
                <div className="dp-critic-claim-evidence">
                  {evidence.slice(0, 3).map((e: any, eidx: number) => (
                    <div className="dp-critic-evidence-item" key={eidx}>
                      {e.ref && <code className="dp-critic-evidence-ref">{e.ref}</code>}
                      {e.source && <span className="dp-critic-evidence-source">{e.source}</span>}
                      {e.quote && <span className="dp-critic-evidence-quote">{e.quote}</span>}
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </details>
  );
}

function LiveCriticCard({ liveCritic, compact = false }: { liveCritic: any; compact?: boolean }) {
  const observations = Array.isArray(liveCritic?.observations) ? liveCritic.observations.slice(-6).reverse() : [];
  const snapshots = Array.isArray(liveCritic?.model_snapshots) ? liveCritic.model_snapshots.slice(-2).reverse() : [];
  return (
    <div className={`dp-live-critic ${compact ? 'is-compact' : ''}`}>
      <div className="dp-eval-list-title">Live Critic Supervisor</div>
      <div className="dp-eval-mini-row"><span>Status</span><strong>{liveCritic.status || 'running'}</strong></div>
      <div className="dp-eval-mini-row"><span>Risk</span><strong>{liveCritic.risk_level || 'low'}</strong></div>
      <div className="dp-eval-sub">{liveCritic.live_summary || 'Live supervisor state is available.'}</div>
      {snapshots.map((item: any, idx: number) => (
        <div className="dp-live-critic-note" key={`snap-${idx}`}>
          <strong>{item.note || 'Live model note'}</strong>
          {item.watch_next && <span>{item.watch_next}</span>}
        </div>
      ))}
      {observations.length > 0 && (
        <div className="dp-live-critic-events">
          {observations.map((item: any, idx: number) => (
            <div className={`dp-live-critic-event is-${item.severity || 'info'}`} key={`${item.event || 'event'}-${idx}`}>
              <span>{item.event || 'event'}</span>
              <p>{item.message || ''}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CodeBlock({ data }: { data: any }) {
  return (
    <pre className="dp-code">{typeof data === 'string' ? data : JSON.stringify(data, null, 2)}</pre>
  );
}

function stripDuplicateCriticConclusion(review: string, summary: string): string {
  const text = String(review || '').trim();
  if (!text || !String(summary || '').trim()) return text;
  const blocks = text.split(/\n{2,}/);
  const first = (blocks[0] || '').trim();
  if (/^\*\*结论\*\*[:：]/.test(first) || /^结论[:：]/.test(first)) {
    return blocks.slice(1).join('\n\n').trim();
  }
  return text
    .replace(/^\s*\*\*结论\*\*[:：].*(?:\r?\n){1,2}/, '')
    .replace(/^\s*结论[:：].*(?:\r?\n){1,2}/, '')
    .trim();
}

function stripFinalVerdictSection(review: string): string {
  return String(review || '')
    .replace(/(?:^|\n)\s*(?:#{1,6}\s*)?(?:\*\*)?最终判[定断](?:\*\*)?\s*(?:[:：])?\s*[\s\S]*$/m, '')
    .trim();
}

function normalizeCriticReview(review: string, summary: string): string {
  return stripFinalVerdictSection(stripDuplicateCriticConclusion(review, summary));
}

function buildEvalProvenance(report: TurnEvalReport, liveSupervisor: any, turn?: TurnCoT): string {
  const judge = (report.judge || {}) as any;
  const status = String(judge.status || '').toLowerCase();
  const sourceEvent = String(judge.source_event || judge.structured?.source_event || '').trim();
  const inputSourcesRaw = judge.input_sources || judge.structured?.input_sources || [];
  const inputSources = Array.isArray(inputSourcesRaw)
    ? inputSourcesRaw
        .map((item: any) => String(item?.label || item?.path_name || '').trim())
        .filter(Boolean)
        .slice(0, 2)
    : [];
  const hasTraceHookSignal = Boolean(
    liveSupervisor
    || turn?.turn_start_ms_observed
    || turn?.turn_end_ms_observed
    || turn?.turn_duration_ms_observed
  );
  const hookText = hasTraceHookSignal ? 'Trace 来源：hook 已触发并写入' : 'Trace 来源：未确认 hook 写入';
  const sourceText = (() => {
    if (sourceEvent === 'api-rerun') {
      return '\u0045\u0076\u0061\u006c \u6765\u6e90\uff1a\u65e7\u7248\u975e hook report\uff08\u5df2\u4e0d\u4f5c\u4e3a\u6709\u6548\u7ed3\u679c\uff09';
    }
    if (sourceEvent === 'manual-hook-report-recovery') {
      return hasTraceHookSignal
        ? '\u0045\u0076\u0061\u006c \u6765\u6e90\uff1ahook/live supervisor \u5df2\u89e6\u53d1\uff0c\u6700\u7ec8 hook report \u66fe\u7f3a\u5931\uff1b\u672c\u6b21\u7531\u70b9\u51fb Eval \u8865\u751f\u6210'
        : '\u0045\u0076\u0061\u006c \u6765\u6e90\uff1a\u68c0\u6d4b\u5230 hook report \u7f3a\u5931\uff0c\u672c\u6b21\u7531\u70b9\u51fb Eval \u8865\u751f\u6210';
    }
    if (sourceEvent === 'manual-agent-critic-fallback') {
      return hasTraceHookSignal
        ? '\u0045\u0076\u0061\u006c \u6765\u6e90\uff1ahook/live \u4fe1\u53f7\u5b58\u5728\uff0c\u4f46 hook report \u7f3a\u5931\u6216\u62a5\u9519\uff1b\u672c\u6b21\u5df2\u964d\u7ea7\u4e3a\u70b9\u51fb Eval \u540e\u7684 Agent Critic \u624b\u52a8\u515c\u5e95'
        : '\u0045\u0076\u0061\u006c \u6765\u6e90\uff1a\u672a\u786e\u8ba4 hook \u89e6\u53d1\uff1b\u672c\u6b21\u7531\u70b9\u51fb Eval \u540e\u7684 Agent Critic \u624b\u52a8\u515c\u5e95\u4ea7\u751f';
    }
    if (status === 'queued' || status === 'running') {
      return sourceEvent
        ? `\u0045\u0076\u0061\u006c \u6765\u6e90\uff1ahook \u9636\u6bb5 Agent Critic \u6b63\u5728\u751f\u6210\uff08${sourceEvent}\uff09`
        : '\u0045\u0076\u0061\u006c \u6765\u6e90\uff1ahook \u9636\u6bb5 Agent Critic \u6b63\u5728\u751f\u6210';
    }
    if (status === 'interrupted') {
      return sourceEvent
        ? `\u0045\u0076\u0061\u006c \u6765\u6e90\uff1ahook \u9636\u6bb5 Agent Critic \u5df2\u89e6\u53d1\u4f46\u672a\u5b8c\u6210\uff08${sourceEvent}\uff09\uff0c\u70b9\u51fb Eval \u53ef\u8865\u751f\u6210 report`
        : '\u0045\u0076\u0061\u006c \u6765\u6e90\uff1ahook \u9636\u6bb5 Agent Critic \u672a\u5b8c\u6210\uff0c\u70b9\u51fb Eval \u53ef\u8865\u751f\u6210 report';
    }
    return sourceEvent
      ? `\u0045\u0076\u0061\u006c \u6765\u6e90\uff1ahook \u9636\u6bb5 Agent Critic report\uff08${sourceEvent}\uff09`
      : '\u0045\u0076\u0061\u006c \u6765\u6e90\uff1a\u5c1a\u672a\u53d1\u73b0 hook \u9636\u6bb5 Agent Critic report';
  })();
  const inputsText = inputSources.length ? `输入：${inputSources.join(' + ')}` : '';
  return [hookText, sourceText, inputsText].filter(Boolean).join(' · ');
}

// ═══════════════════════════════════════════════════════════
//  v0.15.1: Claude 5 条专属时间线 + 27 hook 计数 Section 组件
//
//  设计原则：
//   * 数据为空（数组长度=0）时**整段不渲染**，Cursor session 视觉零回归
//   * 一律展示 t_ms 真值（来自 hook），格式化成 HH:MM:SS.mmm
//   * 时间戳为 0 = transcript 推断的事件（如 permissionMode）—— 标 "transcript"
// ═══════════════════════════════════════════════════════════

function JudgeReviewMarkdown({
  text,
  evidenceByDimension,
}: {
  text: string;
  evidenceByDimension?: Map<string, CriticEvidence[]>;
}) {
  const renderInline = (line: string) => {
    const nodes: ReactNode[] = [];
    let cursor = 0;
    const regex = /\*\*(.+?)\*\*/g;
    let match: RegExpExecArray | null;
    while ((match = regex.exec(line))) {
      if (match.index > cursor) nodes.push(line.slice(cursor, match.index));
      nodes.push(<strong key={`${match.index}-${match[1]}`}>{match[1]}</strong>);
      cursor = match.index + match[0].length;
    }
    if (cursor < line.length) nodes.push(line.slice(cursor));
    return nodes.length ? nodes : line;
  };

  // Extract the dimension label from a heading like "**任务完成** · partial".
  // We match against the full prefix between the first pair of ** so the label
  // never accidentally includes the verdict suffix.
  const headingLabel = (line: string): string | null => {
    const match = line.match(/^\*\*(.+?)\*\*/);
    return match ? match[1].trim() : null;
  };

  const lines = String(text || '').split(/\r?\n/);
  const nodes: ReactNode[] = [];
  let lastHeadingLabel: string | null = null;
  let blockBuffer: ReactNode[] = [];
  let blockKey = 0;

  const flushBlock = () => {
    if (blockBuffer.length) {
      nodes.push(<div className="dp-judge-md-block" key={`block-${blockKey++}`}>{blockBuffer}</div>);
      blockBuffer = [];
    }
    if (lastHeadingLabel && evidenceByDimension?.has(lastHeadingLabel)) {
      const ev = evidenceByDimension.get(lastHeadingLabel) || [];
      if (ev.length) {
        nodes.push(
          <div className="dp-judge-md-evidence" key={`ev-${blockKey++}`}>
            {ev.map((e, idx) => (
              <div className="dp-critic-evidence-item" key={idx}>
                {e.ref && <code className="dp-critic-evidence-ref">{e.ref}</code>}
                {e.source && <span className="dp-critic-evidence-source">{e.source}</span>}
                {e.quote && <span className="dp-critic-evidence-quote">{e.quote}</span>}
              </div>
            ))}
          </div>,
        );
      }
    }
    lastHeadingLabel = null;
  };

  for (let idx = 0; idx < lines.length; idx++) {
    const trimmed = lines[idx].trim();
    if (!trimmed) {
      blockBuffer.push(<div className="dp-judge-md-gap" key={`gap-${idx}`} />);
      continue;
    }
    const heading = trimmed.match(/^#{1,6}\s+(.+)$/);
    const isMdHeading = !!heading;
    const isStrongHeading = !isMdHeading && /^\*\*.+?\*\*/.test(trimmed);
    if (isMdHeading || isStrongHeading) {
      flushBlock();
      lastHeadingLabel = isMdHeading ? heading![1].trim().replace(/^\*\*|\*\*$/g, '') : headingLabel(trimmed);
      blockBuffer.push(
        <div className="dp-judge-md-heading" key={idx}>
          {renderInline(isMdHeading ? heading![1].trim() : trimmed)}
        </div>,
      );
      continue;
    }
    const bullet = trimmed.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      blockBuffer.push(
        <div className="dp-judge-md-bullet" key={idx}>
          <span />
          <p>{renderInline(bullet[1])}</p>
        </div>,
      );
      continue;
    }
    const ordered = trimmed.match(/^\d+[.)]\s+(.+)$/);
    if (ordered) {
      blockBuffer.push(
        <div className="dp-judge-md-bullet" key={idx}>
          <span />
          <p>{renderInline(ordered[1])}</p>
        </div>,
      );
      continue;
    }
    blockBuffer.push(
      <div className="dp-judge-md-line" key={idx}>
        {renderInline(trimmed)}
      </div>,
    );
  }
  flushBlock();

  return <>{nodes}</>;
}

function _fmtHookTime(t_ms?: number): string {
  if (!t_ms || t_ms <= 0) return '—';
  try {
    const d = new Date(t_ms);
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    const ss = String(d.getSeconds()).padStart(2, '0');
    const ms = String(d.getMilliseconds()).padStart(3, '0');
    return `${hh}:${mm}:${ss}.${ms}`;
  } catch {
    return '—';
  }
}

function _fmtTokens(n?: number | null): string {
  if (n == null) return '—';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

// ─── 27 hook 计数面板（KPI 风格）─────────────
// 用一行 chip 列出本 session 27 hook 实际触发分布。chip 颜色按事件分类
// 区分（compact / subagent / permission / notification / env / transcript-first）。
const _CLAUDE_HOOK_BUCKETS: Record<string, { tone: string; tonebg: string }> = {
  // session 生命周期 / transcript-first（仅计数）
  SessionStart:     { tone: '#94a3b8', tonebg: '#94a3b81a' },
  SessionEnd:       { tone: '#94a3b8', tonebg: '#94a3b81a' },
  Stop:             { tone: '#94a3b8', tonebg: '#94a3b81a' },
  UserPromptSubmit: { tone: '#94a3b8', tonebg: '#94a3b81a' },
  PreToolUse:       { tone: '#94a3b8', tonebg: '#94a3b81a' },
  PostToolUse:      { tone: '#94a3b8', tonebg: '#94a3b81a' },
  PostToolUseFailure: { tone: '#ef4444', tonebg: '#ef44441a' },
  StopFailure:      { tone: '#ef4444', tonebg: '#ef44441a' },
  // compact
  PreCompact:  { tone: '#06b6d4', tonebg: '#06b6d41a' },
  PostCompact: { tone: '#06b6d4', tonebg: '#06b6d41a' },
  // subagent
  SubagentStart: { tone: '#a855f7', tonebg: '#a855f71a' },
  SubagentStop:  { tone: '#a855f7', tonebg: '#a855f71a' },
  TaskCreated:   { tone: '#a855f7', tonebg: '#a855f71a' },
  TaskCompleted: { tone: '#a855f7', tonebg: '#a855f71a' },
  // permission
  PermissionRequest: { tone: '#f59e0b', tonebg: '#f59e0b1a' },
  PermissionDenied:  { tone: '#f59e0b', tonebg: '#f59e0b1a' },
  Elicitation:       { tone: '#f59e0b', tonebg: '#f59e0b1a' },
  ElicitationResult: { tone: '#f59e0b', tonebg: '#f59e0b1a' },
  // notification
  Notification: { tone: '#fbbf24', tonebg: '#fbbf241a' },
  TeammateIdle: { tone: '#fbbf24', tonebg: '#fbbf241a' },
  // environment
  CwdChanged:        { tone: '#10b981', tonebg: '#10b9811a' },
  FileChanged:       { tone: '#10b981', tonebg: '#10b9811a' },
  WorktreeCreate:    { tone: '#10b981', tonebg: '#10b9811a' },
  WorktreeRemove:    { tone: '#10b981', tonebg: '#10b9811a' },
  ConfigChange:      { tone: '#10b981', tonebg: '#10b9811a' },
  InstructionsLoaded:{ tone: '#10b981', tonebg: '#10b9811a' },
  Setup:             { tone: '#10b981', tonebg: '#10b9811a' },
};
const _CLAUDE_27_HOOKS = Object.keys(_CLAUDE_HOOK_BUCKETS);

// ─── v0.15.2: Claude 会话快览卡片 ─────────────────────────────────
//
// 让 Claude 会话即使没有 extended thinking 也能"看起来很富"——
// 这是 v0.15.x 现场反馈："4.6 sonnet/opus 非 thinking 变体导致 transcript
// 里 thinking 块 = 0，前端只剩 tool_use 一片紫，体感糟糕"的对策。
//
// 渲染内容：
//   1. 模型分布 + thinking 检测（无则给切换教程）
//   2. 第一句用户问题 + 最后一段模型回复（让"对话感"立刻显现）
//   3. 执行思路浓缩条：把每个 tool_decision 的 tool_intent 串成时间线
//
// 仅在 agent_type=claude 时渲染，不影响 Cursor 会话。
function ClaudeQuickGlance({ cot }: { cot: SessionCoT }) {
  if ((cot.agent_type || '') !== 'claude') return null;

  // 1. 模型 + thinking 检测
  const modelDist: Record<string, number> = {};
  let thinkingExplicitCount = 0;
  let preToolReasoningCount = 0;
  let toolIntentCount = 0;
  const intents: { turn: number; tool: string; intent: string }[] = [];
  let firstUserQuery = '';
  let lastFinalResp = '';

  for (const t of cot.turns) {
    if (!firstUserQuery && t.user_query) firstUserQuery = t.user_query;
    if (t.final_response) lastFinalResp = t.final_response;
    for (const s of t.steps) {
      if (s.step_type === 'thinking_explicit') thinkingExplicitCount++;
      if (s.step_type === 'pre_tool_reasoning') preToolReasoningCount++;
      const md = (s as any).metadata || {};
      if (md.model && typeof md.model === 'string') {
        modelDist[md.model] = (modelDist[md.model] || 0) + 1;
      }
      if (s.step_type === 'tool_decision' && md.tool_intent) {
        toolIntentCount++;
        if (intents.length < 12) {
          intents.push({
            turn: t.turn_index,
            tool: s.tool_name || md.tool_name || '?',
            intent: String(md.tool_intent),
          });
        }
      }
    }
  }
  // 兜底：从 session_meta.models 取
  if (Object.keys(modelDist).length === 0) {
    const sm: any = (cot as any).session_meta || {};
    const fromOtel: any = (cot as any).otel_view?.actual_token_usage;
    const m = sm.model || fromOtel?.model;
    if (typeof m === 'string') modelDist[m] = 1;
  }
  const modelEntries = Object.entries(modelDist).sort((a, b) => b[1] - a[1]);
  const hasThinking = thinkingExplicitCount > 0;
  const hasAnyReasoning = hasThinking || preToolReasoningCount > 0 || toolIntentCount > 0;

  // 都没有就别渲染（极端空会话）
  if (!firstUserQuery && !lastFinalResp && !hasAnyReasoning && modelEntries.length === 0) {
    return null;
  }

  // 简单 markdown → 纯文本预览（不引入 markdown 库，避免 bundle 涨）
  const previewMd = (s: string, n: number) => {
    if (!s) return '';
    return s.length > n ? s.slice(0, n) + '…' : s;
  };

  return (
    <Section title="🤖 Claude 会话快览">
      <div className="dp-claude-glance">
        {/* 第 1 行：模型 + thinking 状态 */}
        <div className="dp-claude-glance-row">
          {modelEntries.map(([m, n]) => (
            <span key={m} className="dp-claude-glance-model" title={`${m}：${n} 条 assistant 消息`}>
              <span className="dp-claude-glance-model-name">{m}</span>
              <span className="dp-claude-glance-model-count">×{n}</span>
            </span>
          ))}
          {hasThinking ? (
            <span className="dp-claude-glance-think dp-claude-glance-think-on" title="本次会话产出了 type:thinking 块">
              🧠 Extended Thinking · {thinkingExplicitCount} 块
            </span>
          ) : (
            <span className="dp-claude-glance-think dp-claude-glance-think-off"
                  title="非 *-thinking 模型变体不会产出 type:thinking 块">
              ⚠️ 未启用 Extended Thinking
            </span>
          )}
          <span className="dp-claude-glance-stat" title="从 tool_input.description / .prompt 抽取的 Claude 行级意图">
            💭 {toolIntentCount} 条 intent
          </span>
          {preToolReasoningCount > 0 && (
            <span className="dp-claude-glance-stat" title="工具调用前的 text 块">
              📝 {preToolReasoningCount} 条 pre-tool 推理
            </span>
          )}
        </div>

        {/* 严格遵守 transcript-first 原则：tool_intent 是从 tool_input.description
            抽出来的真实字段，不是合成或推理。下面这段提示明确告诉用户
            transcript 里 thinking = 0 的原因和它的"真实 vs 推理"边界，
            不夸大也不暗示这些是 thinking。 */}
        {!hasThinking && (
          <div className="dp-claude-glance-tip">
            <div className="dp-claude-glance-tip-head">
              ℹ️ 本会话 transcript 里没有 <code>type:&quot;thinking&quot;</code> 块
            </div>
            <div className="dp-claude-glance-tip-body">
              这是 API 层事实：当前模型 SKU 不支持 Extended Thinking，
              即使 prompt 里写 <code>ultrathink</code> / <code>think hard</code>
              也不会让模型产出 thinking 块（这些关键词只对带
              <code>-thinking</code> 后缀的 SKU 生效）。如果你的渠道（如
              claude-internal）不暴露这种 SKU，那就**没有 thinking 数据**
              可以采集——transcript 是模型 API 的原始输出，前端只展示真实
              数据，绝不会从工具调用反推或合成 thinking。
              <br />
              <br />
              下方紫色的 <span className="dp-claude-glance-tip-badge">💭 Claude&apos;s intent</span>
              卡片是 Claude 在工具调用 <code>tool_input.description</code> /
              <code>.prompt</code> 里**自己写的**一句话目的，**这是真实字段**，
              不是推理；它解释"为什么调用这个工具"，但不等于 thinking。
              当前会话共 <b>{toolIntentCount}</b> 条 intent。
            </div>
          </div>
        )}

        {/* 第 2 行：用户首问 + 模型尾答 */}
        {(firstUserQuery || lastFinalResp) && (
          <div className="dp-claude-glance-talk">
            {firstUserQuery && (
              <div className="dp-claude-glance-talk-row dp-claude-glance-talk-user">
                <div className="dp-claude-glance-talk-tag">USER</div>
                <div className="dp-claude-glance-talk-body">
                  {previewMd(firstUserQuery, 360)}
                </div>
              </div>
            )}
            {lastFinalResp && (
              <div className="dp-claude-glance-talk-row dp-claude-glance-talk-assistant">
                <div className="dp-claude-glance-talk-tag">CLAUDE</div>
                <div className="dp-claude-glance-talk-body">
                  {previewMd(lastFinalResp, 600)}
                </div>
              </div>
            )}
          </div>
        )}

        {/* 第 3 行：tool_intent 时间线浓缩 */}
        {intents.length > 0 && (
          <div className="dp-claude-glance-intents">
            <div className="dp-claude-glance-intents-head">
              💡 执行思路（按调用顺序前 {intents.length} 步意图）
            </div>
            <ol className="dp-claude-glance-intents-list">
              {intents.map((it, i) => (
                <li key={i} className="dp-claude-glance-intents-item">
                  <span className="dp-claude-glance-intents-idx">#{i + 1}</span>
                  <span className="dp-claude-glance-intents-tool">{it.tool}</span>
                  <span className="dp-claude-glance-intents-text">{it.intent}</span>
                </li>
              ))}
              {toolIntentCount > intents.length && (
                <li className="dp-claude-glance-intents-more">
                  …还有 {toolIntentCount - intents.length} 条 intent，展开下方 turn / step 查看完整链
                </li>
              )}
            </ol>
          </div>
        )}
      </div>
    </Section>
  );
}

function ClaudeHookKpiSection({ cot }: { cot: SessionCoT }) {
  if ((cot.agent_type || '') !== 'claude') return null;
  const sm = (cot as any).session_meta || {};
  const heo: Record<string, number> = sm.hook_events_observed || {};
  const total = Object.values(heo).reduce((s: number, v: any) => s + (Number(v) || 0), 0);
  if (total === 0) return null;

  const triggered = _CLAUDE_27_HOOKS.filter(h => (heo[h] || 0) > 0);
  const silent = _CLAUDE_27_HOOKS.filter(h => !(heo[h] || 0));

  return (
    <Section title={`🪝 Claude Hook 触发分布 · 27 个事件中 ${triggered.length} 个有数据 / 共 ${total} 次`}>
      <div className="dp-claude-hookkpi">
        <div className="dp-claude-hookkpi-row">
          {triggered.map(h => {
            const cfg = _CLAUDE_HOOK_BUCKETS[h];
            return (
              <span
                key={h}
                className="dp-claude-hookchip"
                style={{ borderColor: cfg.tone, background: cfg.tonebg, color: cfg.tone }}
                title={`${h}: 触发 ${heo[h]} 次`}
              >
                {h} <b style={{ marginLeft: 4 }}>×{heo[h]}</b>
              </span>
            );
          })}
        </div>
        {silent.length > 0 && (
          <details className="dp-claude-hookkpi-silent">
            <summary>本次未触发的 {silent.length} 个 hook（点击展开）</summary>
            <div className="dp-claude-hookkpi-row" style={{ opacity: 0.55, marginTop: 6 }}>
              {silent.map(h => (
                <span
                  key={h}
                  className="dp-claude-hookchip"
                  style={{ borderColor: '#475569', color: '#94a3b8' }}
                >{h}</span>
              ))}
            </div>
          </details>
        )}
      </div>
    </Section>
  );
}

// ─── Subagent 时间线 ───────────────
function ClaudeSubagentSection({ cot }: { cot: SessionCoT }) {
  const evs = (cot as any).subagent_timeline as any[] | undefined;
  if (!evs || evs.length === 0) return null;
  // 按 sub_agent_id / parent_tool_use_id 配对成 group：start + stop -> 一行
  const groups: Record<string, any> = {};
  for (const e of evs) {
    const key = e.sub_agent_id || e.parent_tool_use_id || `t${e.t_ms}`;
    const g = groups[key] || { events: [] };
    g.events.push(e);
    groups[key] = g;
  }
  const rows = Object.entries(groups).map(([key, g]: any) => {
    const start = g.events.find((x: any) => x.phase === 'SubagentStart' || x.phase === 'TaskCreated');
    const stop  = g.events.find((x: any) => x.phase === 'SubagentStop'  || x.phase === 'TaskCompleted');
    const any   = g.events[0];
    const t0 = start?.t_ms ?? any.t_ms;
    const t1 = stop?.t_ms;
    const duration = (t1 && t0) ? t1 - t0 : (any.duration_ms || null);
    return {
      key,
      sub_agent_id: any.sub_agent_id || '—',
      parent_tool_use_id: any.parent_tool_use_id,
      agent_type: any.agent_type,
      model: any.model,
      prompt_preview: start?.prompt_preview,
      summary: stop?.summary,
      t_ms: t0,
      duration_ms: duration,
      status: stop ? 'completed' : 'running',
      events_count: g.events.length,
    };
  });
  return (
    <Section title={`🧬 Subagent / Task 时间线（×${rows.length}）`}>
      <div className="dp-claude-subagent-list">
        {rows.map(r => (
          <div key={r.key} className="dp-claude-subagent-row">
            <div className="dp-claude-subagent-head">
              <span className="dp-claude-badge dp-claude-badge-subagent">
                {r.status === 'completed' ? '✅ Completed' : '🔄 Running'}
              </span>
              {r.agent_type && <span className="dp-claude-meta">type: <code>{r.agent_type}</code></span>}
              {r.model && <span className="dp-claude-meta">model: <code>{r.model}</code></span>}
              {r.duration_ms != null && (
                <span className="dp-claude-meta">⏱ {fmtDur(r.duration_ms)}</span>
              )}
              <span className="dp-claude-meta">{_fmtHookTime(r.t_ms)}</span>
            </div>
            {r.prompt_preview && (
              <div className="dp-claude-subagent-prompt">
                <span className="dp-claude-label">prompt</span>
                <span className="dp-claude-text">{r.prompt_preview}</span>
              </div>
            )}
            {r.summary && (
              <div className="dp-claude-subagent-summary">
                <span className="dp-claude-label">summary</span>
                <span className="dp-claude-text">{r.summary}</span>
              </div>
            )}
            <div className="dp-claude-subagent-foot">
              <code>id={r.sub_agent_id}</code>
              {r.parent_tool_use_id && <code>parent={r.parent_tool_use_id.slice(0, 16)}…</code>}
              <span className="dp-claude-dim">{r.events_count} hook events</span>
            </div>
          </div>
        ))}
      </div>
    </Section>
  );
}

// ─── Compact 时间线（横条对比 token before -> after）─────────
function ClaudeCompactSection({ cot }: { cot: SessionCoT }) {
  const evs = (cot as any).compact_events as any[] | undefined;
  if (!evs || evs.length === 0) return null;
  // pair PreCompact -> PostCompact 按 t_ms 顺序成对
  const pairs: any[] = [];
  let pending: any = null;
  for (const e of evs) {
    if (e.phase === 'before') pending = e;
    else if (e.phase === 'after') {
      if (pending) {
        pairs.push({ pre: pending, post: e });
        pending = null;
      } else {
        pairs.push({ pre: null, post: e });
      }
    }
  }
  if (pending) pairs.push({ pre: pending, post: null });

  return (
    <Section title={`📦 上下文压缩时间线（×${pairs.length}）`}>
      <div className="dp-claude-compact-list">
        {pairs.map((p, idx) => {
          const before = p.pre?.before_tokens;
          const after  = p.post?.after_tokens;
          const saved  = p.post?.saved_tokens
            ?? (before != null && after != null ? before - after : null);
          const t = p.pre?.t_ms ?? p.post?.t_ms;
          const trigger = p.pre?.trigger;
          const dur = (p.pre?.t_ms && p.post?.t_ms) ? p.post.t_ms - p.pre.t_ms : null;
          return (
            <div key={idx} className="dp-claude-compact-row">
              <div className="dp-claude-compact-head">
                <span className="dp-claude-badge dp-claude-badge-compact">📦 Compact #{idx + 1}</span>
                {trigger && <span className="dp-claude-meta">trigger: <code>{trigger}</code></span>}
                {dur != null && <span className="dp-claude-meta">⏱ {fmtDur(dur)}</span>}
                <span className="dp-claude-meta">{_fmtHookTime(t)}</span>
              </div>
              <div className="dp-claude-compact-bar">
                <span className="dp-claude-compact-before">
                  before: <b>{_fmtTokens(before)}</b>
                </span>
                <span className="dp-claude-compact-arrow">⤍</span>
                <span className="dp-claude-compact-after">
                  after: <b>{_fmtTokens(after)}</b>
                </span>
                {saved != null && saved > 0 && (
                  <span className="dp-claude-compact-saved">
                    省 <b>{_fmtTokens(saved)}</b> tokens
                  </span>
                )}
              </div>
              {p.post?.summary_chars != null && (
                <div className="dp-claude-compact-summary">
                  压缩摘要长度：{p.post.summary_chars.toLocaleString()} chars
                </div>
              )}
              {p.post?.summary && (
                <details className="dp-claude-compact-summary">
                  <summary>Compact summary content</summary>
                  <pre>{p.post.summary}</pre>
                </details>
              )}
            </div>
          );
        })}
      </div>
    </Section>
  );
}

// ─── Permission 时间线 ───────────────
function ClaudePermissionSection({ cot }: { cot: SessionCoT }) {
  const evs = (cot as any).permission_events as any[] | undefined;
  if (!evs || evs.length === 0) return null;
  return (
    <Section title={`🔐 Permission / Mode 时间线（×${evs.length}）`}>
      <div className="dp-claude-perm-list">
        {evs.map((e, idx) => {
          const isMode = e.kind === 'PermissionMode';
          return (
            <div key={idx} className="dp-claude-perm-row">
              <span className={`dp-claude-badge ${isMode ? 'dp-claude-badge-mode' : 'dp-claude-badge-perm'}`}>
                {isMode ? '🎚 PermissionMode' : `🔐 ${e.kind}`}
              </span>
              {isMode && (
                <span className="dp-claude-meta">
                  {e.prev_mode ? <><code>{e.prev_mode}</code> → </> : null}
                  <code style={{ color: '#fbbf24' }}>{e.mode}</code>
                </span>
              )}
              {e.tool_name && <span className="dp-claude-meta">tool: <code>{e.tool_name}</code></span>}
              {e.decision && <span className="dp-claude-meta">decision: <code>{e.decision}</code></span>}
              {e.reason && <span className="dp-claude-text" style={{ flex: 1, minWidth: 0 }}>{e.reason}</span>}
              <span className="dp-claude-meta dp-claude-dim">
                {e.t_ms ? _fmtHookTime(e.t_ms) : (e.source === 'transcript' ? 'transcript' : '—')}
              </span>
            </div>
          );
        })}
      </div>
    </Section>
  );
}

// ─── Notification 时间线 ───────────────
function ClaudeNotificationSection({ cot }: { cot: SessionCoT }) {
  const evs = (cot as any).notification_events as any[] | undefined;
  if (!evs || evs.length === 0) return null;
  return (
    <Section title={`🔔 Notification / Idle 时间线（×${evs.length}）`}>
      <div className="dp-claude-notif-list">
        {evs.map((e, idx) => (
          <div key={idx} className="dp-claude-notif-row">
            <span className="dp-claude-badge dp-claude-badge-notif">🔔 {e.kind}</span>
            {e.tool_name && <span className="dp-claude-meta">tool: <code>{e.tool_name}</code></span>}
            {e.message && <span className="dp-claude-text" style={{ flex: 1, minWidth: 0 }}>{e.message}</span>}
            <span className="dp-claude-meta dp-claude-dim">{_fmtHookTime(e.t_ms)}</span>
          </div>
        ))}
      </div>
    </Section>
  );
}

// ─── Environment 时间线 ───────────────
function ClaudeEnvironmentSection({ cot }: { cot: SessionCoT }) {
  const evs = (cot as any).environment_events as any[] | undefined;
  if (!evs || evs.length === 0) return null;
  return (
    <Section title={`🌐 Environment Events（×${evs.length}）`}>
      <div className="dp-claude-env-list">
        {evs.map((e, idx) => {
          let body: React.ReactNode = null;
          if (e.kind === 'CwdChanged') {
            body = <><code>{e.before || '?'}</code> → <code>{e.after}</code></>;
          } else if (e.kind === 'FileChanged') {
            body = <>
              <code>{e.path}</code>
              {e.change_kind && <span className="dp-claude-meta">{e.change_kind}</span>}
              {e.is_user_initiated && <span className="dp-claude-badge dp-claude-badge-user">user</span>}
            </>;
          } else if (e.kind === 'WorktreeCreate' || e.kind === 'WorktreeRemove') {
            body = <>
              <code>{e.worktree_path}</code>
              {e.branch && <span className="dp-claude-meta">branch: <code>{e.branch}</code></span>}
            </>;
          } else if (e.kind === 'ConfigChange') {
            body = <>
              <code>{e.key}</code>: <code>{String(e.before ?? '∅')}</code> → <code>{String(e.after ?? '∅')}</code>
            </>;
          } else if (e.kind === 'InstructionsLoaded') {
            body = <>
              {(e.instruction_files || []).slice(0, 4).map((f: string, i: number) => (
                <code key={i} style={{ marginRight: 4 }}>{f}</code>
              ))}
              {(e.instruction_files || []).length > 4 && (
                <span className="dp-claude-dim">+{(e.instruction_files || []).length - 4} more</span>
              )}
            </>;
          } else if (e.kind === 'Setup') {
            body = <>
              {e.claude_version && <span className="dp-claude-meta">v{e.claude_version}</span>}
              {e.setup_args && <code>{JSON.stringify(e.setup_args)}</code>}
            </>;
          } else {
            body = <code className="dp-claude-dim">{JSON.stringify(e.details || {})}</code>;
          }
          return (
            <div key={idx} className="dp-claude-env-row">
              <span className="dp-claude-badge dp-claude-badge-env">🌐 {e.kind}</span>
              <span className="dp-claude-text" style={{ flex: 1, minWidth: 0 }}>{body}</span>
              <span className="dp-claude-meta dp-claude-dim">{_fmtHookTime(e.t_ms)}</span>
            </div>
          );
        })}
      </div>
    </Section>
  );
}

// v0.8.0: prompt / 召回片段折叠展示。默认显示前 800 字，超长时附"展开完整"按钮。
// 不用 <details>，因为 details 默认是收起 → 用户看不到任何内容；这里是
// "默认显示精简版 + 一键展开"更符合阅读习惯。
function CollapsedTextBlock({
  text,
  threshold = 800,
}: {
  text: string;
  threshold?: number;
}) {
  const [expanded, setExpanded] = useState(false);
  const len = text.length;
  if (len <= threshold) {
    return <CodeBlock data={text} />;
  }
  const display = expanded ? text : text.slice(0, threshold) + '\n…';
  return (
    <>
      <CodeBlock data={display} />
      <button
        type="button"
        className="dp-collapse-btn"
        onClick={() => setExpanded(e => !e)}
      >
        {expanded ? `收起（共 ${len} 字）` : `展开完整内容（${len} 字）`}
      </button>
    </>
  );
}

// v0.8.0: 调用分类元信息（与 SpanTree.tsx 的 INVOCATION_CFG 视觉对齐）
const INVOCATION_PROMPT_TITLE: Record<InvocationCategory, string> = {
  llm_call:   '🧠 完整 Prompt（送给 LLM）',
  rag_query:  '📚 RAG 查询 Prompt',
  web_search: '🔎 Web Search Query',
};
const INVOCATION_PROMPT_HINT: Record<InvocationCategory, string> = {
  llm_call:   '此步骤显式调用 LLM，下面是被打包发送的完整 prompt（messages / system / 上下文）。',
  rag_query:  '此步骤是一次 RAG / 知识库 / 向量库查询；下面是查询字符串本身，可用于评估检索精度。',
  web_search: '此步骤是一次在线搜索；下面是查询字符串。',
};

// ─── Thinking Phase 分组 ───────────────────────────────────
// 一连串相邻的 thinking_explicit 经常是模型一次"思考爆发"——前端把它们
// 一条条全部渲染会把 timeline 撑得很长（参见 a185f20b：连续 47 条思考夹
// 在两次 Shell 调用之间）。这里把 ≥2 条相邻的 thinking_explicit 折叠为
// 一个 ThinkingPhase 节点：默认收起、显示阶段标题 + 总时长 + 总字数；
// 点开是子树，每条 thought 又是一个独立可展开的小折叠。
type ThinkingPhase = {
  kind: 'thinking_phase';
  thoughts: ThoughtStep[];
  firstIndex: number;
  lastIndex: number;
};
type TimelineItem = ThoughtStep | ThinkingPhase;

function groupThinkingPhases(
  steps: ThoughtStep[],
  opts?: { treatPreToolAsThinking?: boolean },
): TimelineItem[] {
  const out: TimelineItem[] = [];
  let buf: ThoughtStep[] = [];
  const flush = () => {
    if (buf.length === 0) return;
    if (buf.length === 1) {
      out.push(buf[0]);
    } else {
      out.push({
        kind: 'thinking_phase',
        thoughts: buf,
        firstIndex: buf[0].step_index,
        lastIndex: buf[buf.length - 1].step_index,
      });
    }
    buf = [];
  };
  // v0.15.0: Claude 未启用 Extended Thinking 时把 pre_tool_reasoning 也算
  // 进 thinking 阶段折叠——多条相邻可读得更紧凑，不再淹没在 timeline 里。
  const isThinkingLike = (s: ThoughtStep): boolean => {
    if (s.step_type === 'thinking_explicit') return true;
    if (opts?.treatPreToolAsThinking && s.step_type === 'pre_tool_reasoning') return true;
    return false;
  };
  for (const s of steps) {
    if (isThinkingLike(s)) {
      buf.push(s);
    } else {
      flush();
      out.push(s);
    }
  }
  flush();
  return out;
}

// 从一段 thinking 文本里抽一句话当标题：第一句（句号 / 换行 / 标点切分），
// 截断在 100 字以内，去掉前导空白和句尾标点。
function extractThoughtTitle(text: string, maxLen = 100): string {
  if (!text) return '';
  const cleaned = text.replace(/^[\s\u3000]+/, '');
  // 中英文句末标点 + 换行都作为切分点，取首句
  const m = cleaned.match(/^([^\.\!\?。！？\n]{6,}?)(?:[\.\!\?。！？\n]|$)/);
  let s = m ? m[1] : cleaned.slice(0, maxLen);
  s = s.trim();
  if (s.length > maxLen) s = s.slice(0, maxLen).trim() + '…';
  return s;
}

// ─── 思考流程时间线（Turn 详情核心）─────────────────────
function ThinkingTimeline({
  steps,
  treatPreToolAsThinking = false,
}: {
  steps: ThoughtStep[];
  // v0.15.0: Claude 未启用 Extended Thinking 时，把 pre_tool_reasoning 视
  // 同 thinking_explicit 处理（折叠 + 紫色 🧠 视觉），由调用方判断后传入。
  treatPreToolAsThinking?: boolean;
}) {
  if (!steps.length) return <div className="tl-empty">无步骤数据</div>;

  const grouped = groupThinkingPhases(steps, { treatPreToolAsThinking });

  return (
    <div className="tl-container">
      {grouped.map((item, idx) => {
        const isLast = idx === grouped.length - 1;
        if ('kind' in item && item.kind === 'thinking_phase') {
          return (
            <ThinkingPhaseRow
              key={`phase-${item.firstIndex}`}
              phase={item}
              isLast={isLast}
              treatPreToolAsThinking={treatPreToolAsThinking}
            />
          );
        }
        const step = item as ThoughtStep;
        const cfg = getStepCfg(step, { treatPreToolAsThinking });
        const isError = step.step_type === 'tool_execution' && step.metadata?.is_error;
        const tool = step.tool_name || step.metadata?.tool_name;

        return (
          <div key={step.step_index} className={`tl-item ${isError ? 'tl-error' : ''}`}>
            {/* 时间线轴 */}
            <div className="tl-axis">
              <div className="tl-dot" style={{ background: cfg.color, boxShadow: `0 0 6px ${cfg.color}66` }}>
                <span className="tl-dot-icon">{cfg.icon}</span>
              </div>
              {!isLast && <div className="tl-line" />}
            </div>

            {/* 内容区 */}
            <div className="tl-body">
              {/* 标题行 */}
              <div className="tl-header">
                <span className="tl-type" style={{ color: cfg.color }}>{cfg.label}</span>
                {tool && <span className="tl-tool">→ {tool}</span>}
                {step.tokens > 0 && <span className="tl-tokens">{step.tokens}t</span>}
                {step.duration_ms != null && step.duration_ms > 0 && (
                  <span className="tl-dur">{fmtDur(step.duration_ms)}</span>
                )}
                <span className="tl-step-num">#{step.step_index}</span>
              </div>

              {/* 描述 */}
              {cfg.desc && (
                <div className="tl-desc">{cfg.desc}</div>
              )}

              {/* 内容 —— Extended Thinking 单独走一条折叠路径，
                  原因：单条 afterAgentThought 经常是 1k–4k 字，直出 timeline
                  会把整个时间线撑爆；折叠掉可以让用户先扫节奏再展开看细节，
                  跟 Cursor 自己 UI 的 "Thinking" 折叠面板一致。 */}
              {step.content && step.step_type === 'thinking_explicit' ? (
                (() => {
                  const chars = step.metadata?.thought_chars
                    ?? step.content.length;
                  const isLong = chars > 280;
                  const preview = step.content.slice(0, 220).replace(/\s+/g, ' ');
                  return (
                    <details className="tl-think-details" open={!isLong}>
                      <summary className="tl-think-summary">
                        <span className="tl-think-badge">🧠 Extended Thinking</span>
                        <span className="tl-think-meta">
                          {chars.toLocaleString()} chars
                          {step.duration_ms != null && step.duration_ms > 0 && (
                            <> · {fmtDur(step.duration_ms)}</>
                          )}
                        </span>
                        {isLong && (
                          <span className="tl-think-preview">
                            “{preview}{preview.length < step.content.length ? '…' : ''}”
                          </span>
                        )}
                      </summary>
                      <div className="tl-think-content">{step.content}</div>
                    </details>
                  );
                })()
              ) : (
                step.content && (
                  <div className={`tl-content ${isError ? 'tl-content-error' : ''}`}>
                    {step.content}
                  </div>
                )
              )}

              {/* 工具调用的 input 详情 */}
              {step.step_type === 'tool_decision' && step.metadata?.tool_input && (
                <details className="tl-details">
                  <summary className="tl-details-summary">查看工具参数</summary>
                  <CodeBlock data={step.metadata.tool_input} />
                </details>
              )}

              {/* 错误恢复的原始错误 */}
              {step.step_type === 'error_recovery' && step.metadata?.error_content && (
                <div className="tl-error-box">
                  <span className="tl-error-label">原始错误</span>
                  <div className="tl-error-content">{step.metadata.error_content}</div>
                </div>
              )}

              {/* 策略转换的 from/to */}
              {step.step_type === 'strategy_shift' && (
                <div className="tl-shift-box">
                  <span className="tl-shift-from">{step.metadata?.from_tool || '?'}</span>
                  <span className="tl-shift-arrow">→</span>
                  <span className="tl-shift-to">{step.metadata?.to_tool || '?'}</span>
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─── Thinking Phase 卡片 + 子树 ────────────────────────────
// 设计：黄色 💡 徽章 + 自动总结（取首条 thought 的第一句）+ N 条 / 总时长 /
// 总字数。点开后下面是 sub-tree，每条 thought 又是一个折叠小卡（默认收起，
// 标题就是该 thought 的首句）。
function ThinkingPhaseRow({
  phase,
  isLast,
  treatPreToolAsThinking = false,
}: {
  phase: ThinkingPhase;
  isLast: boolean;
  // v0.15.0: 当 phase 是由 pre_tool_reasoning 折叠出来的（Claude 未启用
  // ext-thinking），徽章改成 🧠 Pre-tool Thinking 区别真正的 Extended
  // Thinking phase。
  treatPreToolAsThinking?: boolean;
}) {
  const { thoughts } = phase;
  const totalChars = thoughts.reduce(
    (s, t) => s + (t.metadata?.thought_chars ?? t.content?.length ?? 0),
    0,
  );
  const totalDur = thoughts.reduce(
    (s, t) => s + (typeof t.duration_ms === 'number' ? t.duration_ms : 0),
    0,
  );
  // 用首条 thought 的首句做整段阶段的标题
  const phaseTitle = extractThoughtTitle(thoughts[0]?.content || '', 110);

  // 判断该 phase 内全部为 pre_tool_reasoning（Claude 未启用 ext-thinking
  // 时的轻量推理） vs. 真 Extended Thinking。仅当**整段都是** pre-tool 时
  // 切换徽章；混合段落仍按 Extended Thinking 走（保持兼容）。
  const isPreToolPhase = treatPreToolAsThinking && thoughts.every(
    t => t.step_type === 'pre_tool_reasoning',
  );
  const badgeText = isPreToolPhase ? '🧠 Pre-tool Thinking' : '💡 Thinking Phase';
  const dotIcon = isPreToolPhase ? '🧠' : '💡';

  return (
    <div className="tl-item tl-phase-item">
      <div className="tl-axis">
        <div className="tl-dot tl-phase-dot">
          <span className="tl-dot-icon">{dotIcon}</span>
        </div>
        {!isLast && <div className="tl-line" />}
      </div>

      <div className="tl-body">
        <details className="tl-phase-details">
          <summary className="tl-phase-summary">
            <span className="tl-phase-badge">{badgeText}</span>
            <span className="tl-phase-meta">
              {thoughts.length} thoughts
              {totalDur > 0 && <> · {fmtDur(totalDur)}</>}
              {' · '}
              {totalChars.toLocaleString()} chars
            </span>
            <span className="tl-phase-range">
              #{phase.firstIndex}–{phase.lastIndex}
            </span>
            {phaseTitle && (
              <span className="tl-phase-title">"{phaseTitle}"</span>
            )}
          </summary>

          {/* 子树：每条 thought 一行，可独立展开看完整原文 */}
          <div className="tl-phase-children">
            {thoughts.map((t, i) => (
              <ThinkingPhaseChild
                key={t.step_index}
                thought={t}
                isLast={i === thoughts.length - 1}
              />
            ))}
          </div>
        </details>
      </div>
    </div>
  );
}

function ThinkingPhaseChild({
  thought,
  isLast,
}: {
  thought: ThoughtStep;
  isLast: boolean;
}) {
  const chars =
    thought.metadata?.thought_chars ?? thought.content?.length ?? 0;
  const title = extractThoughtTitle(thought.content || '', 80);

  return (
    <details className={`tl-phase-child ${isLast ? 'tl-phase-child-last' : ''}`}>
      <summary className="tl-phase-child-summary">
        <span className="tl-phase-child-marker" />
        <span className="tl-phase-child-title">{title || '(empty)'}</span>
        <span className="tl-phase-child-meta">
          {chars.toLocaleString()}c
          {typeof thought.duration_ms === 'number' && thought.duration_ms > 0 && (
            <> · {fmtDur(thought.duration_ms)}</>
          )}
        </span>
        <span className="tl-phase-child-num">#{thought.step_index}</span>
      </summary>
      <div className="tl-phase-child-content">{thought.content}</div>
    </details>
  );
}

// ─── v0.9.0: 临时脚本与文件产物可视化 ─────────────────────
// 描述：把"agent 在执行任务过程中自己写出来的脚本/草稿/产物"汇成一张表，
// 同时高亮"这一组里哪些是 _verify*.py / _audit*.cjs 之类的临时验证脚本"。
// 列表里每条 = 一个 ScriptArtifact，整个生命周期（创建→改→执行→删）一行可见。

const LANGUAGE_TONE: Record<string, string> = {
  python: '#3b82f6', javascript: '#f59e0b', typescript: '#06b6d4',
  shell: '#10b981', powershell: '#0ea5e9',
  markdown: '#8b5cf6', json: '#a855f7', yaml: '#a855f7',
  text: '#64748b', unknown: '#64748b',
};

const LIFECYCLE_LABEL: Record<string, { text: string; tone: string }> = {
  created:                     { text: '已创建',           tone: '#10b981' },
  created_modified:            { text: '创建+修改',         tone: '#06b6d4' },
  created_executed:            { text: '创建+执行',         tone: '#f59e0b' },
  created_modified_executed:   { text: '创建+改+执行',      tone: '#f59e0b' },
  created_executed_deleted:    { text: '创建→执行→删除',   tone: '#ef4444' },
  created_deleted:             { text: '创建→删除',         tone: '#ef4444' },
  modified_only:               { text: '仅修改（已存在文件）', tone: '#06b6d4' },
  deleted_only:                { text: '仅删除',             tone: '#ef4444' },
  modified:                    { text: '修改',               tone: '#06b6d4' },
};

function ScriptArtifactCard({
  art, onJump,
}: {
  art: ScriptArtifact;
  onJump?: (stepIndex: number) => void;
}) {
  const lc = LIFECYCLE_LABEL[art.lifecycle] || { text: art.lifecycle, tone: '#64748b' };
  const tone = LANGUAGE_TONE[art.language] || '#64748b';
  const created = art.created_at_step;
  const deleted = art.deleted_at_step;
  const execs = art.executed_at_steps || [];
  const dim = art.basename.length > 40 ? art.basename : '';
  return (
    <div
      className={`dp-artifact ${art.is_temp ? 'dp-artifact-temp' : ''}`}
      style={{ borderLeftColor: tone }}
    >
      <div className="dp-artifact-head">
        <span className="dp-artifact-icon">{art.is_temp ? '📜' : '📄'}</span>
        <span className="dp-artifact-name" title={art.path || dim}>{art.basename}</span>
        <span className="dp-artifact-lang" style={{ borderColor: `${tone}55`, color: tone, background: `${tone}14` }}>
          {art.language || art.extension || 'file'}
        </span>
        <span
          className="dp-artifact-lifecycle"
          style={{ color: lc.tone, borderColor: `${lc.tone}55`, background: `${lc.tone}14` }}
        >
          {lc.text}
        </span>
        {art.is_temp && (
          <span className="dp-artifact-temp-pill" title="启发式判定为临时验证脚本">
            临时
          </span>
        )}
      </div>
      <div className="dp-artifact-meta">
        {art.purpose_hint && (
          <span className="dp-artifact-purpose" title={art.purpose_hint}>
            “{art.purpose_hint}”
          </span>
        )}
      </div>
      <div className="dp-artifact-path-row">
        <span className="dp-artifact-path" title={art.path}>{art.path}</span>
      </div>
      <div className="dp-artifact-stats">
        <span className="dp-artifact-stat">
          ✏️ {art.edit_count} 次写
        </span>
        <span className="dp-artifact-stat">
          +{art.total_added_lines} / -{art.total_removed_lines} 行
        </span>
        {art.last_content_chars > 0 && (
          <span className="dp-artifact-stat">
            {art.last_content_chars}ch
          </span>
        )}
      </div>
      <div className="dp-artifact-timeline">
        {created != null && (
          <button
            type="button"
            className="dp-artifact-step dp-artifact-step-create"
            onClick={() => onJump?.(created)}
            title="跳转到创建该文件的 step"
          >
            ➕ 创建于 step #{created}
          </button>
        )}
        {execs.map((idx, i) => (
          <button
            type="button"
            key={i}
            className="dp-artifact-step dp-artifact-step-exec"
            onClick={() => onJump?.(idx)}
            title="跳转到执行该脚本的 Shell step"
          >
            ▶ 执行于 step #{idx}
          </button>
        ))}
        {deleted != null && (
          <button
            type="button"
            className="dp-artifact-step dp-artifact-step-delete"
            onClick={() => onJump?.(deleted)}
            title="跳转到删除该文件的 step"
          >
            🗑 删除于 step #{deleted}
          </button>
        )}
      </div>
    </div>
  );
}

function ScriptArtifactsSection({
  artifacts, stats, onJump,
}: {
  artifacts: ScriptArtifact[];
  stats?: any;
  onJump?: (stepIndex: number) => void;
}) {
  const [showAll, setShowAll] = useState(false);
  const tempCount = stats?.temp_scripts ?? artifacts.filter(a => a.is_temp).length;
  const total = stats?.total_artifacts ?? artifacts.length;
  // 默认只展示临时脚本（用户最关心的"验证脚本"）；点"显示全部"再展开
  const list = showAll ? artifacts : artifacts.filter(a => a.is_temp);
  return (
    <Section title={`🛠️ 临时脚本与文件产物（×${total}）`}>
      <div className="dp-artifact-desc">
        Agent 在本会话内创建 / 修改 / 删除的所有文件产物。其中
        <b className="dp-artifact-desc-em"> {tempCount} 个</b>
        被启发式判定为临时验证脚本（命名带
        <code> _ </code> 前缀、含 <code>verify</code>/<code>audit</code>/<code>tmp</code>{' '}
        等关键词，或在 session 内创建后又被删除）。
      </div>
      {stats && (
        <div className="dp-artifact-stat-row">
          <span className="dp-artifact-stat-chip">写 {stats.total_writes}</span>
          <span className="dp-artifact-stat-chip">改 {stats.total_strreplaces}</span>
          <span className="dp-artifact-stat-chip">删 {stats.total_deletes}</span>
          <span className="dp-artifact-stat-chip dp-artifact-stat-chip-emph">
            ▶ 执行 {stats.total_executions}
          </span>
          {stats.executed_temp_scripts > 0 && (
            <span className="dp-artifact-stat-chip dp-artifact-stat-chip-warn">
              📜 执行临时 {stats.executed_temp_scripts}
            </span>
          )}
          {stats.deleted_temp_scripts > 0 && (
            <span className="dp-artifact-stat-chip dp-artifact-stat-chip-danger">
              🗑 删除临时 {stats.deleted_temp_scripts}
            </span>
          )}
        </div>
      )}
      <div className="dp-artifact-list">
        {list.map((art, i) => (
          <ScriptArtifactCard key={i} art={art} onJump={onJump} />
        ))}
        {list.length === 0 && (
          <div className="dp-artifact-empty">
            没有命中临时脚本，点下面按钮看全部 {total} 个文件产物。
          </div>
        )}
      </div>
      {total > tempCount && (
        <button
          type="button"
          className="dp-artifact-toggle"
          onClick={() => setShowAll(s => !s)}
        >
          {showAll ? `仅看临时脚本（${tempCount}）` : `显示全部 ${total} 个文件产物`}
        </button>
      )}
    </Section>
  );
}

// ─── Session 详情 ─────────────────────────────────────────
// v0.9.0: 把"哪些步骤创建了哪些临时脚本"做成"🛠️ 临时脚本与文件产物"section。
// 每个 artifact 卡片可点击直接跳转到首次创建的 step（用 onSelectNode）。
function SessionDetail({
  node, onSelectNode,
}: {
  node: Extract<SelectedNode, { kind: 'session' }>;
  onSelectNode?: (n: SelectedNode) => void;
}) {
  const { cot, session, report } = node;
  // v0.20.7: 累加每个 turn 用同一个 readTurnUsage 兜底链——避免某些 turn 因
  // hook 未触发而 usage = {0,0}，让会话级 "总 Token" 显示 0K。
  const totalTokens = cot.turns.reduce((s, t) => {
    const tu = readTurnUsage(t);
    return s + (tu?.input_tokens || 0) + (tu?.output_tokens || 0);
  }, 0);
  const isParent = cot.is_parent || (cot as any).sub_sessions?.length > 0;
  const subSessions = (cot as any).sub_sessions || session?.sub_sessions || [];

  // 工具：根据 step_index 找 (step, turn) 然后调 onSelectNode 跳转。
  const jumpToStep = (stepIndex: number) => {
    if (!onSelectNode || !stepIndex) return;
    for (const t of cot.turns) {
      const found = t.steps.find(s => s.step_index === stepIndex);
      if (found) {
        onSelectNode({ kind: 'step', step: found, turn: t });
        return;
      }
    }
  };

  // v0.16.1: 旧的 OTLP 导出弹窗（"🚀 导出到 OTel"）已下线——它是把 cot.json
  // 重放/合成 OTLP 推后端，跟 Claude Code 原生 OTel 真值通道概念混淆。Claude
  // session 现在统一在右侧 OTel tab 里看真实 OTel 数据并下载（OtelPanel 内）。

  return (
    <>
      <div className="dp-hero">
        <div className="dp-hero-icon">🖥️</div>
        <div className="dp-hero-title">{session?.topic || 'Claude Code Session'}</div>
        <div className="dp-hero-id">{cot.session_id}</div>
        {isParent && (
          <div className="dp-hero-sub">📂 包含 {subSessions.length} 次交互</div>
        )}
      </div>

      {/* v0.11.2：OTel KPI 条 —— 自动检测的 model / cost / cache / cursor.version */}
      <SessionOtelKpiBar cot={cot} />

      {/* v0.15.2：Claude 会话快览 —— 让非 thinking 变体的 4.x 会话也能"看起来富"，
          展示模型 / thinking 状态 / 用户首问 / 模型尾答 / 执行意图链 */}
      <ClaudeQuickGlance cot={cot} />

      <Section title="概览统计">
        <div className="dp-grid-2">
          <Kv k="Turns" v={cot.turns.length} />
          <Kv k="工具调用" v={cot.total_tool_calls} />
          <Kv k="思考步骤" v={cot.total_thinking_steps} />
          <Kv k="策略转换" v={cot.total_strategy_shifts} />
          <Kv k="总 Token" v={`${(totalTokens / 1000).toFixed(1)}K`} />
          <Kv k="平均复杂度" v={cot.avg_complexity.toFixed(3)} />
          {typeof (cot as any).avg_steps_per_turn === 'number' && (
            <Kv k="平均步数/Turn" v={(cot as any).avg_steps_per_turn.toFixed(2)} />
          )}
        </div>
      </Section>

      {/* v0.14.2: Cursor IDE 会话真值生命周期（来自 sessionStart / sessionEnd 等 hook） */}
      {cot.session_meta && Object.keys(cot.session_meta).length > 0 && (
        <Section title="🟢 IDE 会话生命周期（来自 Cursor Hook 真值）">
          <div className="dp-grid-2">
            {cot.session_meta.cursor_version && (
              <Kv k="Cursor 版本" v={<span className="dp-mono-sm">{cot.session_meta.cursor_version}</span>} />
            )}
            {cot.session_meta.user_email && (
              <Kv k="用户" v={<span className="dp-mono-sm">{cot.session_meta.user_email}</span>} />
            )}
            {typeof cot.session_meta.session_duration_ms_observed === 'number' && (
              <Kv k="实际 session 时长"
                  v={fmtDur(cot.session_meta.session_duration_ms_observed)} />
            )}
            {typeof cot.session_meta.session_start_ms_observed === 'number' && (
              <Kv k="session 开始" v={
                <span className="dp-mono-sm" title={String(cot.session_meta.session_start_ms_observed)}>
                  {new Date(cot.session_meta.session_start_ms_observed).toLocaleString('zh-CN', {
                    month: '2-digit', day: '2-digit',
                    hour: '2-digit', minute: '2-digit', second: '2-digit',
                  })}
                </span>
              } />
            )}
            {typeof cot.session_meta.session_end_ms_observed === 'number' && (
              <Kv k="session 结束" v={
                <span className="dp-mono-sm" title={String(cot.session_meta.session_end_ms_observed)}>
                  {new Date(cot.session_meta.session_end_ms_observed).toLocaleString('zh-CN', {
                    month: '2-digit', day: '2-digit',
                    hour: '2-digit', minute: '2-digit', second: '2-digit',
                  })}
                </span>
              } />
            )}
          </div>
          {Array.isArray(cot.session_meta.workspace_roots) && cot.session_meta.workspace_roots.length > 0 && (
            <div className="dp-session-roots">
              <strong>工作区根目录：</strong>
              <ul>
                {cot.session_meta.workspace_roots.map((r, i) => (
                  <li key={i}><span className="dp-mono-sm">{r}</span></li>
                ))}
              </ul>
            </div>
          )}
          {cot.session_meta.hook_events_observed && (
            <div className="dp-session-hooks">
              <strong>Hook 事件计数（cot-stream 实时采集）：</strong>
              <div className="dp-grid-3">
                {Object.entries(cot.session_meta.hook_events_observed).map(([k, v]) => (
                  <span key={k} className={`dp-hook-tag ${v > 0 ? 'dp-hook-active' : 'dp-hook-idle'}`}>
                    {k}: <strong>{v}</strong>
                  </span>
                ))}
              </div>
            </div>
          )}
        </Section>
      )}

      {/* v0.14.2: 用户在 IDE 里手动操作的时间线（区别于 agent 的工具调用） */}
      {Array.isArray(cot.user_activity) && cot.user_activity.length > 0 && (
        <Section title={`👤 用户手动操作 ×${cot.user_activity.length}（IDE Tab 行为，与 agent 无关）`}>
          <div className="dp-user-activity-hint">
            <strong>说明</strong>：这里是用户**自己**在 Cursor 里点开/编辑文件的真实记录——
            和 agent 的 <code>Read</code> / <code>StrReplace</code> 工具调用<strong>不是同一个东西</strong>。
            <code>tab_edit</code> 高频出现，通常意味着「agent 没干完，用户自己补了」。
          </div>
          <div className="dp-user-activity-list">
            {cot.user_activity.slice(0, 80).map((a, i) => {
              const ts = new Date(a.t).toLocaleString('zh-CN', {
                month: '2-digit', day: '2-digit',
                hour: '2-digit', minute: '2-digit', second: '2-digit',
              });
              if (a.kind === '_truncated') {
                return <div key={i} className="dp-user-activity-trunc">⋯ {a.note}</div>;
              }
              const icon = a.kind === 'submit_prompt' ? '✉️'
                : a.kind === 'tab_read' ? '👁️'
                : a.kind === 'tab_edit' ? '✏️' : '·';
              const label = a.kind === 'submit_prompt' ? '回车发送'
                : a.kind === 'tab_read' ? '打开/查看'
                : a.kind === 'tab_edit' ? '手动编辑' : a.kind;
              return (
                <div key={i} className={`dp-user-activity-row dp-ua-${a.kind}`}>
                  <span className="dp-ua-icon">{icon}</span>
                  <span className="dp-ua-kind">{label}</span>
                  {a.file_path && (
                    <span className="dp-ua-path dp-mono-sm" title={a.file_path}>
                      {a.file_path.length > 60 ? '…' + a.file_path.slice(-58) : a.file_path}
                    </span>
                  )}
                  {typeof a.added_lines === 'number' && a.added_lines > 0 && (
                    <span className="dp-ua-added">+{a.added_lines}</span>
                  )}
                  {typeof a.removed_lines === 'number' && a.removed_lines > 0 && (
                    <span className="dp-ua-removed">−{a.removed_lines}</span>
                  )}
                  {typeof a.prompt_chars === 'number' && (
                    <span className="dp-ua-chars">{a.prompt_chars} chars</span>
                  )}
                  <span className="dp-ua-time dp-mono-sm">{ts}</span>
                </div>
              );
            })}
            {cot.user_activity.length > 80 && (
              <div className="dp-user-activity-more">
                ⋯ 还有 {cot.user_activity.length - 80} 条未显示
              </div>
            )}
          </div>
        </Section>
      )}

      {/* v0.15.1 ─ Claude 5 条专属时间线 + 27 hook 计数。
          只在有数据时渲染（Cursor session 这些字段都是空数组，不会出现）。 */}
      <ClaudeHookKpiSection cot={cot} />
      <ClaudeSubagentSection cot={cot} />
      <ClaudeCompactSection cot={cot} />
      <ClaudePermissionSection cot={cot} />
      <ClaudeNotificationSection cot={cot} />
      <ClaudeEnvironmentSection cot={cot} />

      {/* 子 Session 列表 */}
      {subSessions.length > 0 && (
        <Section title="交互历史">
          <div className="dp-sub-sessions">
            {subSessions.map((sub: any, idx: number) => (
              <div key={sub.sub_session_id || idx} className="dp-sub-session-item">
                <div className="dp-sub-session-topic">
                  #{idx + 1} {sub.topic || '未知主题'}
                </div>
                <div className="dp-sub-session-meta">
                  <span>🔄 {sub.total_turns} turns</span>
                  <span>🔧 {sub.total_tool_calls} tools</span>
                </div>
                {sub.extracted_at && (
                  <div className="dp-sub-session-time">
                    {new Date(sub.extracted_at).toLocaleString('zh-CN')}
                  </div>
                )}
              </div>
            ))}
          </div>
        </Section>
      )}

      <Section title="工具调用分布">
        <div className="dp-tool-dist">
          {Object.entries(cot.tool_call_distribution).map(([tool, count]) => (
            <div key={tool} className="dp-tool-chip">
              <span className="dp-tool-name">{tool}</span>
              <span className="dp-tool-count">×{count}</span>
            </div>
          ))}
        </div>
      </Section>

      {/* v0.7.0: Plan 演进时间线（TodoWrite diff） —— Session 级聚合视图
          每次 TodoWrite 一次快照，这里是"跨所有子会话"汇总。
          细粒度 diff 请在具体子会话的「本轮 Plan 演进」里看。 */}
      {Array.isArray((cot as any).plan_timeline) && (cot as any).plan_timeline.length > 0 && (
        <Section title={`🗺️ 全 Session Plan 演进（×${(cot as any).plan_timeline.length}）`}>
          <div className="dp-plan-hint">
            全会话 plan 快照聚合视图。单轮详情请进入子会话 (#N) 看「本轮 Plan 演进」。
          </div>
          <div className="dp-plan-timeline">
            {(cot as any).plan_timeline.map((snap: any, idx: number) => (
              <div key={idx} className="dp-plan-snap">
                <div className="dp-plan-snap-head">
                  <span className="dp-plan-snap-idx">#{idx + 1}</span>
                  <span className="dp-plan-snap-meta">
                    Turn {snap.turn_index} · Step {snap.at_step} · 共 {snap.total} 项
                  </span>
                </div>
                {snap.in_progress?.length > 0 && (
                  <div className="dp-plan-group dp-plan-ip">
                    <span className="dp-plan-tag">⏳ in_progress</span>
                    <ul>{snap.in_progress.map((x: string, i: number) => <li key={i}>{x}</li>)}</ul>
                  </div>
                )}
                {snap.completed?.length > 0 && (
                  <div className="dp-plan-group dp-plan-done">
                    <span className="dp-plan-tag">✅ completed</span>
                    <ul>{snap.completed.map((x: string, i: number) => <li key={i}>{x}</li>)}</ul>
                  </div>
                )}
                {snap.pending?.length > 0 && (
                  <div className="dp-plan-group dp-plan-pending">
                    <span className="dp-plan-tag">🕓 pending</span>
                    <ul>{snap.pending.map((x: string, i: number) => <li key={i}>{x}</li>)}</ul>
                  </div>
                )}
                {snap.cancelled?.length > 0 && (
                  <div className="dp-plan-group dp-plan-cancel">
                    <span className="dp-plan-tag">✖ cancelled</span>
                    <ul>{snap.cancelled.map((x: string, i: number) => <li key={i}>{x}</li>)}</ul>
                  </div>
                )}
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* v0.10.0: 模式转换时间线 —— Cursor agent 的 plan↔agent 模式切换。
          数据来自 SwitchMode 工具调用 + CreatePlan 后的隐式回 agent 推断。 */}
      {Array.isArray((cot as any).mode_transitions) && (cot as any).mode_transitions.length > 0 && (
        <Section title={`🔀 模式转换时间线（×${(cot as any).mode_transitions.length}）`}>
          <div className="dp-mode-hint">
            <strong>Cursor agent 在执行复杂任务时</strong>会先切到 <code>plan</code> 模式起草 plan，
            和用户确认后再切回 <code>agent</code> 模式开始执行。下表是本会话发生的全部
            模式转换（点击单条可跳转到对应 step）。
          </div>
          <div className="dp-mode-timeline">
            {(cot as any).mode_transitions.map((m: any, idx: number) => {
              const isImplicit = m.trigger === 'implicit_back_to_agent';
              return (
                <div
                  key={idx}
                  className={`dp-mode-row dp-mode-target-${m.target_mode_id} ${isImplicit ? 'dp-mode-implicit' : ''}`}
                  onClick={() => jumpToStep(m.at_step)}
                >
                  <div className="dp-mode-arrow">
                    <span className="dp-mode-from">{m.prev_mode_id || 'agent'}</span>
                    <span className="dp-mode-glyph">{isImplicit ? '⤷' : '→'}</span>
                    <span className="dp-mode-to">{m.target_mode_id}</span>
                  </div>
                  <div className="dp-mode-meta">
                    <span className="dp-mode-step">step #{m.at_step}</span>
                    <span className="dp-mode-turn">turn {m.turn_index}</span>
                    {isImplicit
                      ? <span className="dp-mode-trigger-implicit">隐式（用户确认 plan）</span>
                      : <span className="dp-mode-trigger-explicit">SwitchMode 工具</span>}
                  </div>
                  {m.explanation && (
                    <div className="dp-mode-explanation">{m.explanation}</div>
                  )}
                </div>
              );
            })}
          </div>
        </Section>
      )}

      {/* v0.10.0: Plan 提案 —— CreatePlan 工具产出的正式 plan 文档（与 TodoWrite 不同：
          TodoWrite 是滚动 checklist，CreatePlan 是一次性的正式蓝图）。 */}
      {Array.isArray((cot as any).plan_proposals) && (cot as any).plan_proposals.length > 0 && (
        <Section title={`📋 Plan 文档提案（×${(cot as any).plan_proposals.length}）`}>
          <div className="dp-plan-proposal-hint">
            <code>CreatePlan</code> 工具调用产出的正式 plan markdown 文档；
            通常是 agent 在 <strong>plan 模式</strong>下提交给用户审阅的"完整蓝图"。
          </div>
          {(cot as any).plan_proposals.map((p: any, idx: number) => (
            <div key={idx} className="dp-plan-proposal-card" onClick={() => jumpToStep(p.at_step)}>
              <div className="dp-plan-proposal-head">
                <span className="dp-plan-proposal-icon">📋</span>
                <span className="dp-plan-proposal-name">{p.name || '未命名 plan'}</span>
                <span className="dp-plan-proposal-meta">
                  step #{p.at_step} · turn {p.turn_index} · {p.plan?.length || 0} 字
                </span>
              </div>
              {p.overview && (
                <div className="dp-plan-proposal-overview">{p.overview}</div>
              )}
              {p.plan && (
                <details className="dp-plan-proposal-details">
                  <summary>展开完整 plan markdown</summary>
                  <pre className="dp-plan-proposal-md">{p.plan}</pre>
                </details>
              )}
            </div>
          ))}
        </Section>
      )}

      {/* v0.7.0: 实时事件观测统计（cot-stream.js 合并） */}
      {(cot as any).observed_events && (
        <Section title="📡 实时事件观测">
          <div className="dp-obs-events">
            <div className="dp-obs-events-desc">
              来自 <code>~/.cursor/hooks/cot-stream.js</code> 的实时事件流，已把真实
              shell stdout/stderr/exit_code 回灌到下方 tool_execution 步骤。
            </div>
            <div className="dp-grid-2" style={{ marginTop: 8 }}>
              <Kv k="事件总数" v={(cot as any).observed_events.events_total} />
              <Kv k="成功回灌" v={(cot as any).observed_events.injected} />
              <Kv k="Shell 事件" v={(cot as any).observed_events.shell_events} />
              <Kv k="MCP 事件" v={(cot as any).observed_events.mcp_events} />
              {typeof (cot as any).observed_events.file_edit_events === 'number' && (
                <Kv k="文件编辑事件" v={(cot as any).observed_events.file_edit_events} />
              )}
              {typeof (cot as any).observed_events.file_edit_injected === 'number' && (
                <Kv k="文件编辑回灌" v={(cot as any).observed_events.file_edit_injected} />
              )}
              <Kv k="Tool Execution 总数" v={(cot as any).observed_events.tool_executions_total} />
            </div>
          </div>
        </Section>
      )}

      {/* v0.9.0: 临时脚本与文件产物 —— L5 Execution Trace 在 session 级的最直观体现 */}
      {Array.isArray((cot as any).script_artifacts) && (cot as any).script_artifacts.length > 0 && (
        <ScriptArtifactsSection
          artifacts={(cot as any).script_artifacts as ScriptArtifact[]}
          stats={(cot as any).script_stats}
          onJump={jumpToStep}
        />
      )}

      {report && (
        <Section title="Response 准确度">
          {(report as any).summary && (() => {
            const s = (report as any).summary;
            return (
              <>
                <div className="dp-scores">
                  {['avg_ocs', 'avg_ccr', 'avg_pmr', 'avg_es'].map(k =>
                    typeof s[k] === 'number'
                      ? <ScoreBar key={k} label={k.replace('avg_', '').toUpperCase()} value={s[k]} />
                      : null
                  )}
                </div>
                {s.verdict_distribution && (
                  <div className="dp-grid-2" style={{ marginTop: 10 }}>
                    {Object.entries(s.verdict_distribution).map(([k, v]) => (
                      <Kv key={k} k={k} v={String(v)} />
                    ))}
                  </div>
                )}
              </>
            );
          })()}
        </Section>
      )}

      <Section title="提取元信息">
        <Kv k="提取时间" v={<span className="dp-mono">{cot.extracted_at}</span>} />
        {(cot as any).transcript_path && (
          <Kv
            k="数据来源"
            v={<span className="dp-mono-sm dp-path-trim" title={(cot as any).transcript_path}>{(cot as any).transcript_path}</span>}
          />
        )}
      </Section>
    </>
  );
}

// ─── v0.10.0: 本轮 Plan 进度面板 ─────────────────────────
// 把"快照列表"重塑成"最终 todo 清单 + diff 时间线"两段式
function TurnPlanProgress({ turnPlans }: { turnPlans: any[] }) {
  const [showAllDiffs, setShowAllDiffs] = useState(false);
  const lastSnap = turnPlans[turnPlans.length - 1];
  const finalTodos: any[] = Array.isArray(lastSnap?.todos) ? lastSnap.todos : [];

  // 兼容老 snapshot（没有 todos[]，回退用 in_progress/completed/pending 重建）
  const buildFallback = (snap: any) => {
    const out: any[] = [];
    (snap.in_progress || []).forEach((c: string) => out.push({ content: c, status: 'in_progress' }));
    (snap.completed || []).forEach((c: string) => out.push({ content: c, status: 'completed' }));
    (snap.pending || []).forEach((c: string) => out.push({ content: c, status: 'pending' }));
    (snap.cancelled || []).forEach((c: string) => out.push({ content: c, status: 'cancelled' }));
    return out;
  };
  const todoList: any[] = finalTodos.length > 0 ? finalTodos : buildFallback(lastSnap || {});
  const completedCount = todoList.filter(t => t.status === 'completed').length;
  const totalCount = todoList.length;
  const progressPct = totalCount > 0 ? Math.round((completedCount / totalCount) * 100) : 0;

  const statusIcon = (s: string) => {
    if (s === 'completed') return '✅';
    if (s === 'in_progress') return '🔄';
    if (s === 'cancelled') return '✖';
    return '⏳';
  };
  const statusClass = (s: string) => `dp-todo-${s || 'pending'}`;

  return (
    <Section title={`🗺️ 本轮 Plan 演进（×${turnPlans.length}）`}>
      <div className="dp-plan-hint">
        <strong>最终 todo 清单</strong>是本轮结束时的状态；下面"演进时间线"展示
        每次 plan 类工具（<code>TodoWrite</code> / <code>TaskCreate</code> /{' '}
        <code>TaskUpdate</code>）调用相对于上一次的变化（完成 / 启动 / 新增 / 删除）。
      </div>

      {/* 最终 todo 清单 + 进度条 */}
      <div className="dp-todo-final">
        <div className="dp-todo-progress">
          <div className="dp-todo-progress-text">
            <span className="dp-todo-progress-label">本轮完成度</span>
            <span className="dp-todo-progress-num">{completedCount}/{totalCount}</span>
            <span className="dp-todo-progress-pct">{progressPct}%</span>
          </div>
          <div className="dp-todo-progress-track">
            <div
              className="dp-todo-progress-fill"
              style={{ width: `${progressPct}%` }}
            />
          </div>
        </div>
        <div className="dp-todo-list">
          {todoList.map((t, i) => (
            <div key={i} className={`dp-todo-item ${statusClass(t.status)}`}>
              <span className="dp-todo-icon">{statusIcon(t.status)}</span>
              <span className="dp-todo-content">{t.content}</span>
              {t.id && <span className="dp-todo-id">{t.id}</span>}
            </div>
          ))}
        </div>
      </div>

      {/* 演进时间线（每次 TodoWrite 调用的 diff） */}
      <div className="dp-plan-diff-timeline">
        <div className="dp-plan-diff-head">
          <span className="dp-plan-diff-title">📜 演进时间线 — 每次 plan 调用</span>
          {turnPlans.length > 3 && (
            <button
              type="button"
              className="dp-collapse-btn"
              onClick={() => setShowAllDiffs(s => !s)}
            >
              {showAllDiffs ? '只看最近 3 次' : `展开全部 ${turnPlans.length} 次`}
            </button>
          )}
        </div>
        {(showAllDiffs ? turnPlans : turnPlans.slice(-3)).map((snap: any, idx: number) => {
          const realIdx = (showAllDiffs ? 0 : Math.max(0, turnPlans.length - 3)) + idx;
          const d = snap.diff || {};
          const empty = (
            (d.newly_completed?.length || 0) +
            (d.newly_started?.length || 0) +
            (d.newly_added?.length || 0) +
            (d.removed?.length || 0) +
            (d.status_changes?.length || 0)
          ) === 0;
          return (
            <div key={realIdx} className="dp-plan-diff-row">
              <div className="dp-plan-diff-row-head">
                <span className="dp-plan-diff-row-idx">#{realIdx + 1}</span>
                <span className="dp-plan-diff-row-meta">
                  step {snap.at_step} · 共 {snap.total} 项
                </span>
                {realIdx === 0 && empty && (d.newly_added?.length || 0) === 0 && (
                  <span className="dp-plan-diff-tag-init">初始 plan</span>
                )}
              </div>
              {(d.newly_completed?.length || 0) > 0 && (
                <div className="dp-plan-diff-section dp-plan-diff-completed">
                  <span className="dp-plan-diff-tag">✅ 完成 {d.newly_completed.length}</span>
                  <ul>{d.newly_completed.map((x: any, i: number) => (
                    <li key={i}>{x.content}{x.id && <span className="dp-todo-id">{x.id}</span>}</li>
                  ))}</ul>
                </div>
              )}
              {(d.newly_started?.length || 0) > 0 && (
                <div className="dp-plan-diff-section dp-plan-diff-started">
                  <span className="dp-plan-diff-tag">▶ 启动 {d.newly_started.length}</span>
                  <ul>{d.newly_started.map((x: any, i: number) => (
                    <li key={i}>{x.content}{x.id && <span className="dp-todo-id">{x.id}</span>}</li>
                  ))}</ul>
                </div>
              )}
              {(d.newly_added?.length || 0) > 0 && (
                <div className="dp-plan-diff-section dp-plan-diff-added">
                  <span className="dp-plan-diff-tag">＋ 新增 {d.newly_added.length}</span>
                  <ul>{d.newly_added.map((x: any, i: number) => (
                    <li key={i}>{x.content} <span className="dp-todo-id">{x.status}</span></li>
                  ))}</ul>
                </div>
              )}
              {(d.removed?.length || 0) > 0 && (
                <div className="dp-plan-diff-section dp-plan-diff-removed">
                  <span className="dp-plan-diff-tag">− 删除 {d.removed.length}</span>
                  <ul>{d.removed.map((x: any, i: number) => (
                    <li key={i}>{x.content}</li>
                  ))}</ul>
                </div>
              )}
              {(d.status_changes?.length || 0) > 0 && (
                <div className="dp-plan-diff-section dp-plan-diff-changes">
                  <span className="dp-plan-diff-tag">⇄ 状态变化 {d.status_changes.length}</span>
                  <ul>{d.status_changes.map((x: any, i: number) => (
                    <li key={i}>
                      {x.content}{' '}
                      <span className="dp-todo-id">{x.from} → {x.to}</span>
                    </li>
                  ))}</ul>
                </div>
              )}
              {empty && (d.newly_added?.length || 0) === 0 && (
                <div className="dp-plan-diff-empty">无变化（重复提交）</div>
              )}
            </div>
          );
        })}
      </div>
    </Section>
  );
}

// ─── Turn 详情（核心：思考流程时间线）────────────────────
function TurnDetail({
  node,
  evalReport,
  isEvalLoading,
  evalError,
  liveCritic,
}: {
  node: Extract<SelectedNode, { kind: 'turn' }>;
  evalReport?: TurnEvalReport;
  isEvalLoading?: boolean;
  evalError?: string | null;
  liveCritic?: any | null;
}) {
  const { turn, cot } = node;
  // v0.20.7: hook 真值 → otel enricher 兜底（详见 readTurnUsage 注释）
  const turnUsage = readTurnUsage(turn);
  const totalTokens = (turnUsage?.input_tokens || 0) + (turnUsage?.output_tokens || 0);
  // 本轮内发生的 plan 更新（TodoWrite 调用）
  const turnPlans = (cot?.plan_timeline || []).filter(p => p.turn_index === turn.turn_index);

  // 统计各类型步骤数量
  const stepTypeCounts = turn.steps.reduce((acc, s) => {
    acc[s.step_type] = (acc[s.step_type] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  return (
    <>
      <div className="dp-hero">
        <div className="dp-hero-icon">💬</div>
        <div className="dp-hero-title">
          {/* v0.19.6: 修 bug 3 —— CodeBuddy 这条提取路径不写
              ``interaction_summary``（cot_extractor 里只在 Cursor/Claude 走
              ``_summarize_interaction``），所以原 fallback 直接掉到
              ``tool_calls.join(' · ')``，hero 标题被渲染成 "todo_write · task · …"，
              看起来像是工具列表当成 prompt 在展示。
              加 ``user_query`` 作为 interaction_summary 缺失时的优先 fallback：
              它就是用户那条原始 prompt，正好是 hero 标题想表达的"这一轮在干嘛"。
              截到 120 字符避免把整段超长 prompt 全摊到顶部。*/}
          {turn.interaction_summary
            || (turn.user_query
                ? (turn.user_query.length > 120
                    ? turn.user_query.slice(0, 120) + '…'
                    : turn.user_query)
                : '')
            || turn.tool_calls.filter(Boolean).join(' · ')
            || (turn.final_response ? '最终回复' : `Turn ${turn.turn_index}`)}
        </div>
        <div className="dp-hero-sub">
          子会话 #{turn.turn_index} · {turn.total_steps} 步
        </div>
      </div>

      <TurnEvalReportCard report={evalReport} isLoading={isEvalLoading} error={evalError} liveCritic={liveCritic} turn={turn} />

      {/* v0.11.2：本轮 OTel KPI（model 继承 session，token/cost 聚合自 step.otel） */}
      {cot && <TurnOtelKpiBar cot={cot} turn={turn} />}

      {/* 用户输入 */}
      {turn.user_query && (
        <Section title="用户输入">
          <div className="dp-prose">{turn.user_query}</div>
        </Section>
      )}

      {/* v0.10.0: 本轮 Plan 演进 —— 重构为「最终 todo 清单 + 每次 diff 时间线」。
          上面：本轮结束时 todo 的最终状态（顺序保留），✅ / ▶ / ⏳ / ✖ 分别用图标
          下面：每一次 TodoWrite 调用的 diff（完成了哪条 / 启动了哪条 / 新增/删除）
          这样既能一眼看到"最终结果"，又能滚动看到"是怎么一步步推进的"。 */}
      {turnPlans.length > 0 && <TurnPlanProgress turnPlans={turnPlans} />}

      {/* v0.10.0: 本轮 plan 文档（CreatePlan）—— 如果该 turn 在 plan 模式下产出过文档，
          把整份 plan 摆在 user_query 后、最显眼的位置 */}
      {(() => {
        const turnProps = ((cot as any)?.plan_proposals || []).filter(
          (p: any) => p.turn_index === turn.turn_index
        );
        if (turnProps.length === 0) return null;
        return (
          <Section title="📋 本轮 Plan 文档（plan 模式提交）">
            {turnProps.map((p: any, idx: number) => (
              <div key={idx} className="dp-plan-proposal-card dp-plan-proposal-inline">
                <div className="dp-plan-proposal-head">
                  <span className="dp-plan-proposal-icon">📋</span>
                  <span className="dp-plan-proposal-name">{p.name || '未命名 plan'}</span>
                  <span className="dp-plan-proposal-meta">
                    step #{p.at_step} · {p.plan?.length || 0} 字
                  </span>
                </div>
                {p.overview && <div className="dp-plan-proposal-overview">{p.overview}</div>}
                {p.plan && (
                  <details className="dp-plan-proposal-details">
                    <summary>展开完整 plan markdown</summary>
                    <pre className="dp-plan-proposal-md">{p.plan}</pre>
                  </details>
                )}
              </div>
            ))}
          </Section>
        );
      })()}

      {/* v0.10.0: 本轮 mode 转换 —— 如果该 turn 内有 SwitchMode 调用 */}
      {(() => {
        const turnTransitions = ((cot as any)?.mode_transitions || []).filter(
          (m: any) => m.turn_index === turn.turn_index
        );
        if (turnTransitions.length === 0) return null;
        return (
          <Section title={`🔀 本轮模式转换（×${turnTransitions.length}）`}>
            <div className="dp-mode-timeline">
              {turnTransitions.map((m: any, idx: number) => {
                const isImplicit = m.trigger === 'implicit_back_to_agent';
                return (
                  <div
                    key={idx}
                    className={`dp-mode-row dp-mode-target-${m.target_mode_id} ${isImplicit ? 'dp-mode-implicit' : ''}`}
                  >
                    <div className="dp-mode-arrow">
                      <span className="dp-mode-from">{m.prev_mode_id || 'agent'}</span>
                      <span className="dp-mode-glyph">{isImplicit ? '⤷' : '→'}</span>
                      <span className="dp-mode-to">{m.target_mode_id}</span>
                    </div>
                    <div className="dp-mode-meta">
                      <span className="dp-mode-step">step #{m.at_step}</span>
                      {isImplicit
                        ? <span className="dp-mode-trigger-implicit">隐式</span>
                        : <span className="dp-mode-trigger-explicit">SwitchMode</span>}
                    </div>
                    {m.explanation && (
                      <div className="dp-mode-explanation">{m.explanation}</div>
                    )}
                  </div>
                );
              })}
            </div>
          </Section>
        );
      })()}

      {/* 思考流程时间线
          v0.15.0：Claude session **未启用 Extended Thinking** 时，
          transcript 里 thinking_explicit / thinking_inter 都是 0，
          唯一的"思考"信号就是 pre_tool_reasoning（Claude 在 tool_use 之前
          的 text block）。这种情况下把它们当 Thinking Phase 折叠 + 紫色 🧠
          展示，避免前端看起来"完全没有 thinking"的错觉。
          Cursor session 维持原渲染（pre_tool_reasoning 与 thinking_inter
          并存时，硬合并会丢真 thinking 信息）。 */}
      {(() => {
        const at = (cot?.agent_type || '');
        const isCodeBuddy = at === 'codebuddy';
        const isClaude = at === 'claude';
        const noExtThinking = !turn.steps.some(
          s => s.step_type === 'thinking_explicit'
            || s.step_type === 'thinking_inter'
            || s.step_type === 'thinking_intermediate',
        );
        // v0.17.1：CodeBuddy 永远启用 pre_tool_reasoning→thinking 视觉，让
        // Hunyuan/混元的"我来帮你…"那段决策直接显示标题，不再被"决策说明"
        // 折叠器吞掉。Cursor 路径维持空。
        const treatPreToolAsThinking = isCodeBuddy || (isClaude && noExtThinking);
        return (
          <Section title="完整思考流程">
            <ThinkingTimeline
              steps={turn.steps}
              treatPreToolAsThinking={treatPreToolAsThinking}
            />
          </Section>
        );
      })()}

      {/* 最终回复（本轮 AI 的综合输出）放在思考流程之后，符合时间轴
          v0.8.2：把 final_response step 上的 inferred_final/reason 抽出来当徽章
          —— 让用户一眼看出"这是真 final，还是因为 Cursor 没返 stop_reason 而启发式推断的" */}
      {turn.final_response && (() => {
        const finalStep = turn.steps.find(s => s.step_type === 'final_response');
        const inferred = !!finalStep?.metadata?.inferred_final;
        const inferredReason = finalStep?.metadata?.inferred_reason as string | undefined;
        return (
          <Section title="本轮最终输出">
            {inferred && (
              <div className="dp-inferred-banner" title="此最终回复并非 Cursor 显式标记，而是由 cot_extractor 启发式推断">
                <span className="dp-inferred-icon">⚠️</span>
                <span className="dp-inferred-text">
                  启发式推断的 final（reason: <span className="dp-mono-sm">{inferredReason || 'unknown'}</span>）—
                  Cursor transcript 未显式标记 stop_reason，由 cot_extractor 基于结构猜测。
                </span>
              </div>
            )}
            <div className="dp-prose">{turn.final_response}</div>
          </Section>
        );
      })()}

      {/* LLM 生成的 CoT 摘要 */}
      {turn.cot_summary && (
        <Section title="🧠 AI 思维链分析">
          <div className="dp-cot-summary">{turn.cot_summary}</div>
        </Section>
      )}


      {/* Turn 统计 */}
      <Section title="Turn 统计">
        <div className="dp-grid-2">
          <Kv k="步骤数" v={turn.total_steps} />
          <Kv k="工具调用" v={turn.tool_calls.filter(Boolean).length} />
          <Kv k="思考深度" v={turn.thinking_depth} />
          <Kv k="策略转换" v={turn.strategy_shifts} />
          <Kv k="复杂度" v={turn.complexity_score.toFixed(3)} />
          <Kv k="错误恢复" v={turn.has_error_recovery ? '是 ⚠️' : '否'} />
        </div>
        {Object.keys(stepTypeCounts).length > 0 && (
          <div className="dp-step-type-dist" style={{ marginTop: 8 }}>
            {Object.entries(stepTypeCounts).map(([type, count]) => {
              const cfg = STEP_CFG[type];
              return (
                <span key={type} className="dp-step-type-chip" style={{ borderColor: `${cfg?.color || '#64748b'}44`, color: cfg?.color || '#64748b' }}>
                  {cfg?.icon || '•'} {cfg?.label || type} ×{count}
                </span>
              );
            })}
          </div>
        )}
      </Section>

      {/* Token 消耗（v0.20.7: hook 真值 → otel enricher 兜底） */}
      {turnUsage && (
        <Section title="Token 消耗">
          <div className="dp-grid-2">
            <Kv k="输入" v={turnUsage.input_tokens.toLocaleString()} />
            <Kv k="输出" v={turnUsage.output_tokens.toLocaleString()} />
            <Kv k="缓存创建" v={turnUsage.cache_creation_input_tokens.toLocaleString()} />
            <Kv k="缓存命中" v={turnUsage.cache_read_input_tokens.toLocaleString()} />
            <Kv k="合计" v={`${(totalTokens / 1000).toFixed(1)}K`} />
            {turnUsage.cost_usd != null && turnUsage.cost_usd > 0 && (
              <Kv
                k="成本 (USD)"
                v={`${turnUsage.is_estimate ? '≈ ' : ''}$${turnUsage.cost_usd.toFixed(4)}`}
              />
            )}
            <Kv
              k="数据来源"
              v={
                turnUsage.source === 'hook'
                  ? 'IDE hook 真值（含 cache 细分）'
                  : turnUsage.source === 'step-sum'
                    ? '逐步 OTel 累加（hook 未触发）'
                    : 'OTel enricher 估算（末路兜底）'
              }
            />
          </div>
        </Section>
      )}
    </>
  );
}

// ─── Step 详情 ────────────────────────────────────────────
const STEP_COLORS: Record<string, string> = {
  user_input: '#3b82f6', tool_result_input: '#6366f1',
  thinking_inter: '#8b5cf6', thinking_intermediate: '#8b5cf6', thinking_explicit: '#8b5cf6',
  pre_tool_reasoning: '#c084fc',
  tool_decision: '#a855f7', tool_execution: '#10b981',
  strategy_shift: '#f59e0b', error_recovery: '#ef4444',
  final_response: '#06b6d4',
};

function StepDetail({ node }: { node: Extract<SelectedNode, { kind: 'step' }> }) {
  const { step, turn } = node;
  const color = step.metadata?.is_error ? '#ef4444' : (STEP_COLORS[step.step_type] || '#64748b');
  const cfg = getStepCfg(step);

  // v0.8.0: 调用分类元信息（LLM / RAG / Web Search）。这些字段由 cot_extractor
  // 在 tool_decision 步骤里打上；tool_execution 步骤通过
  // _propagate_invocation_to_executions 继承。
  const invCat = step.metadata?.invocation_category as InvocationCategory | undefined;
  const promptPreview = step.metadata?.prompt_preview as string | undefined;
  const promptFullChars = step.metadata?.prompt_full_chars as number | undefined;
  const recallPreview = step.metadata?.recall_preview as string | undefined;
  const recallReason = step.metadata?.recall_unavailable_reason as string | undefined;
  const decisionToolInput = step.metadata?.decision_tool_input ?? step.metadata?.tool_input;

  // ── 上下文链路：在同一 turn 内找前后 step，把"前一步思考 / 当前 RAG 调用 /
  //    返回 / 后一步如何使用召回结果"串起来展示
  const sortedSteps = [...turn.steps].sort((a, b) => a.step_index - b.step_index);
  const myIdx = sortedSteps.findIndex(s => s.step_index === step.step_index);
  const prevStep = myIdx > 0 ? sortedSteps[myIdx - 1] : null;
  void (myIdx >= 0 && myIdx < sortedSteps.length - 1 ? sortedSteps[myIdx + 1] : null); // nextStep（保留供后续 UI 扩展）
  // 向后找首个 thinking / final_response —— 这就是"agent 拿到 RAG 结果后真实做的下一步"
  let postStep: ThoughtStep | null = null;
  for (let i = myIdx + 1; i < sortedSteps.length; i++) {
    const t = sortedSteps[i].step_type;
    if (t === 'thinking_inter' || t === 'thinking_intermediate'
        || t === 'thinking_explicit' || t === 'final_response'
        || t === 'pre_tool_reasoning' || t === 'tool_decision') {
      postStep = sortedSteps[i];
      break;
    }
  }

  const observedInput = step.metadata?.observed_input as Record<string, any> | undefined;
  const observedOutput = step.metadata?.observed_output as Record<string, any> | undefined;
  const hasRealObserved = !!(observedInput || observedOutput);
  const hasRichInvocation = !!invCat;

  // v0.9.0: file_op 元数据（cot_script_tracker 写在 Write/StrReplace/Delete 上）
  const fileOp = step.metadata?.file_op as
    | { kind: 'create' | 'modify' | 'delete'; tool_name: string; path: string; basename: string;
        extension: string; language: string; is_temp: boolean;
        edits_count: number; added_lines: number; removed_lines: number;
        content_chars: number; artifact_key: string }
    | undefined;
  // v0.9.0: executed_artifact 元数据（cot_script_tracker 写在 Shell tool_decision/execution 上）
  const executedArtifact = step.metadata?.executed_artifact as
    | { path: string; basename: string; language: string; extension: string;
        is_temp: boolean; artifact_key: string }
    | undefined;

  const fileOpKindCfg: Record<string, { icon: string; label: string; tone: string }> = {
    create: { icon: '📜', label: '创建文件',  tone: '#10b981' },
    modify: { icon: '✏️', label: '修改文件',  tone: '#06b6d4' },
    delete: { icon: '🗑️', label: '删除文件',  tone: '#ef4444' },
  };

  return (
    <>
      <div className="dp-hero">
        <div className="dp-hero-type-badge" style={{ background: `${color}22`, borderColor: `${color}55`, color }}>
          {step.step_type}
        </div>
        <div className="dp-hero-title" style={{ color }}>
          {step.tool_name ? `${cfg.label} → ${step.tool_name}` : cfg.label}
        </div>
        <div className="dp-hero-sub">Turn {step.turn_index} · Step #{step.step_index}</div>
        {cfg.desc && <div className="dp-hero-desc">{cfg.desc}</div>}
        {invCat && (
          <div className={`dp-hero-inv-pill dp-hero-inv-${invCat}`}>
            {INVOCATION_PROMPT_TITLE[invCat]}
          </div>
        )}
      </div>

      {/* v0.12.0：本步 OTel GenAI KPI —— 直接显示 gen_ai.tool.* / duration / usage / cost.usd
          使得每一个小步骤的 OTel 标准 attribute 在主详情页就能看到（不再需要切到 OTel tab）。 */}
      <StepOtelKpiBar step={step} />

      {/* v0.9.0: file_op 横幅 —— Write/StrReplace/Delete 的 tool_decision 顶部
          直接展示"创建/修改/删除哪个文件、几行"，临时脚本会高亮成黄色边框。 */}
      {fileOp && (
        <div
          className={`dp-file-op-banner dp-file-op-${fileOp.kind} ${fileOp.is_temp ? 'dp-file-op-temp' : ''}`}
        >
          <div className="dp-file-op-head">
            <span className="dp-file-op-icon">{fileOpKindCfg[fileOp.kind]?.icon || '📁'}</span>
            <span className="dp-file-op-label" style={{ color: fileOpKindCfg[fileOp.kind]?.tone }}>
              {fileOpKindCfg[fileOp.kind]?.label || fileOp.kind}
            </span>
            <span className="dp-file-op-name">{fileOp.basename}</span>
            {fileOp.is_temp && (
              <span className="dp-file-op-temp-pill">📜 临时验证脚本</span>
            )}
          </div>
          <div className="dp-file-op-path" title={fileOp.path}>{fileOp.path}</div>
          <div className="dp-file-op-stats">
            {fileOp.kind !== 'delete' && (
              <>
                <span className="dp-file-op-stat">+{fileOp.added_lines} 行</span>
                {fileOp.removed_lines > 0 && (
                  <span className="dp-file-op-stat">-{fileOp.removed_lines} 行</span>
                )}
                {fileOp.content_chars > 0 && (
                  <span className="dp-file-op-stat">{fileOp.content_chars}ch</span>
                )}
              </>
            )}
            <span className="dp-file-op-stat dp-file-op-stat-lang">
              {fileOp.language || fileOp.extension || 'file'}
            </span>
          </div>
        </div>
      )}

      {/* v0.9.0: Shell step 如果命中执行某个 artifact，给出醒目提示
          —— 让用户一眼看到"这一步运行了之前创建的 _verify_v080.py"。 */}
      {executedArtifact && step.step_type !== 'tool_decision' && (
        <div className={`dp-exec-artifact-banner ${executedArtifact.is_temp ? 'dp-exec-artifact-temp' : ''}`}>
          <span className="dp-exec-artifact-icon">▶</span>
          <span className="dp-exec-artifact-label">
            执行了脚本 <b>{executedArtifact.basename}</b>
          </span>
          <span className="dp-exec-artifact-lang">{executedArtifact.language}</span>
          {executedArtifact.is_temp && (
            <span className="dp-exec-artifact-temp">📜 临时</span>
          )}
        </div>
      )}
      {executedArtifact && step.step_type === 'tool_decision' && (
        <div className={`dp-exec-artifact-banner ${executedArtifact.is_temp ? 'dp-exec-artifact-temp' : ''}`}>
          <span className="dp-exec-artifact-icon">▶</span>
          <span className="dp-exec-artifact-label">
            将运行脚本 <b>{executedArtifact.basename}</b>
          </span>
          <span className="dp-exec-artifact-lang">{executedArtifact.language}</span>
          {executedArtifact.is_temp && (
            <span className="dp-exec-artifact-temp">📜 临时</span>
          )}
        </div>
      )}

      {/* v0.10.0: SwitchMode 醒目横幅 + explanation */}
      {step.metadata?.mode_switch && (
        (() => {
          const ms = step.metadata!.mode_switch as {
            target_mode_id: string; prev_mode_id?: string | null;
            explanation?: string | null; trigger?: string;
          };
          const isImplicit = ms.trigger === 'implicit_back_to_agent';
          return (
            <div className={`dp-mode-switch-banner dp-mode-target-${ms.target_mode_id} ${isImplicit ? 'dp-mode-implicit' : ''}`}>
              <div className="dp-mode-switch-head">
                <span className="dp-mode-switch-icon">🔀</span>
                <span className="dp-mode-switch-title">
                  模式切换 {ms.prev_mode_id || 'agent'} {isImplicit ? '⤷' : '→'} <b>{ms.target_mode_id}</b>
                </span>
                {isImplicit ? (
                  <span className="dp-mode-trigger-implicit">隐式（用户确认 plan 后自动）</span>
                ) : (
                  <span className="dp-mode-trigger-explicit">SwitchMode 工具</span>
                )}
              </div>
              {ms.explanation && (
                <div className="dp-mode-switch-explanation">
                  <span className="dp-mode-switch-key">原因：</span>
                  {ms.explanation}
                </div>
              )}
            </div>
          );
        })()
      )}

      {/* v0.10.0: CreatePlan 文档完整渲染 */}
      {(step.metadata?.plan_proposal || step.tool_name === 'CreatePlan') && (
        (() => {
          const ti = step.metadata?.tool_input as
            | { name?: string; overview?: string; plan?: string }
            | undefined;
          const pp = step.metadata?.plan_proposal as
            | { name?: string; overview_preview?: string; plan_chars?: number }
            | undefined;
          const name = ti?.name || pp?.name || '未命名 plan';
          const overview = ti?.overview || pp?.overview_preview || '';
          const plan = ti?.plan || '';
          return (
            <div className="dp-plan-proposal-card">
              <div className="dp-plan-proposal-head">
                <span className="dp-plan-proposal-icon">📋</span>
                <span className="dp-plan-proposal-name">{name}</span>
                <span className="dp-plan-proposal-meta">
                  {plan ? `${plan.length} 字` : (pp?.plan_chars ? `${pp.plan_chars} 字` : '')}
                </span>
              </div>
              {overview && (
                <div className="dp-plan-proposal-overview">{overview}</div>
              )}
              {plan && (
                <details className="dp-plan-proposal-details" open>
                  <summary>展开完整 plan markdown（{plan.length} 字）</summary>
                  <pre className="dp-plan-proposal-md">{plan}</pre>
                </details>
              )}
            </div>
          );
        })()
      )}

      {/* v0.10.0: TodoWrite step 专用面板 —— 顶部 diff 摘要 + todos 表格 */}
      {/* v0.14.1 修复：merge=true 的 TodoWrite 调用，原始 tool_input.todos
          只有 ``[{id, status}]``（缺 content），所以底部"状态/内容/id"表格
          长期是空的。后端 _build_plan_timeline 已经把解析后的 plan_full_todos
          写进 step.metadata，前端优先用它，原始 tool_input 只作为兜底。
          v0.19.5 扩展：Claude Internal 用 TaskCreate / TaskUpdate（不是 TodoWrite）
          维护 plan，但后端把 plan_full_todos / plan_diff / plan_total / plan_
          completed_count 等全部回灌到了 Task* 类 step 的 metadata，跟 TodoWrite
          完全同形。所以这里把门 enable 到任意带 plan_full_todos 的 plan 类
          tool_decision 即可——Plan 快照卡片在 Claude Internal 也会正常渲染。 */}
      {(['TodoWrite', 'TaskCreate', 'TaskUpdate'].includes(step.tool_name || '')) && step.step_type === 'tool_decision' && (
        (() => {
          const fullFromBackend = step.metadata?.plan_full_todos as any[] | undefined;
          const ti = step.metadata?.tool_input as { todos?: any[] } | undefined;
          const rawTodos = ti?.todos || [];
          const todos = (Array.isArray(fullFromBackend) && fullFromBackend.length > 0)
            ? fullFromBackend
            : rawTodos;
          const d = (step.metadata?.plan_diff || {}) as {
            newly_completed?: any[]; newly_started?: any[];
            newly_added?: any[]; removed?: any[]; status_changes?: any[];
          };
          // v0.14.3：plan 滞后推断 —— 这一帧 plan 是 turn 末状态时，agent 经常
          // 没及时打勾，导致 4/9 完成的假象。后端会在 step.metadata 里写出
          // 推断完成的 ids，前端拿来高亮 + 警告条 + 重新计数。
          const isLikelyStale = step.metadata?.plan_is_likely_stale === true;
          const staleReason = step.metadata?.plan_stale_reason as string | undefined;
          const lagSteps = step.metadata?.plan_lag_steps as number | undefined;
          const inferredIds = (step.metadata?.plan_inferred_completed as string[] | undefined) || [];
          const inferredSet = new Set(inferredIds);
          const total = todos.length;
          const realDone = todos.filter((t: any) => t.status === 'completed').length;
          const ip = todos.filter((t: any) => t.status === 'in_progress').length;
          // 推断完成的项数：与 inferredSet 相交的非 completed 项
          const inferredDone = todos.filter((t: any) =>
            t.status !== 'completed' && t.id != null && inferredSet.has(String(t.id))).length;
          const effectiveDone = realDone + inferredDone;
          const pct = total > 0 ? Math.round((realDone / total) * 100) : 0;
          const pctEff = total > 0 ? Math.round((effectiveDone / total) * 100) : 0;
          const staleReasonLabel = staleReason === 'final_response_signal'
              ? 'agent 在本轮 final_response 里宣告了"全部完成"'
              : staleReason === 'turn_finalized'
              ? 'agent 已经给出 final_response（本轮已收尾）'
              : staleReason === 'lag_too_many_steps'
              ? `agent 在最后一次 TodoWrite 后又干了 ${lagSteps ?? '?'} 步操作，没再打勾`
              : staleReason || '原因未知';
          // v0.19.5: Plan 快照卡片标题随真实 tool_name 切换，避免在 Claude
          // TaskCreate / TaskUpdate 上误显示 "TodoWrite 快照"——三种 plan 工具
          // 语义不同（whole-list / append / patch），label 也应当各异。
          const cardTitle = step.tool_name === 'TaskCreate'
            ? 'Plan 新增（TaskCreate）'
            : step.tool_name === 'TaskUpdate'
            ? 'Plan 推进（TaskUpdate）'
            : 'TodoWrite 快照';
          return (
            <div className={`dp-todowrite-card ${isLikelyStale ? 'dp-todowrite-stale' : ''}`}>
              <div className="dp-todowrite-head">
                <span className="dp-todowrite-icon">🗺️</span>
                <span className="dp-todowrite-title">{cardTitle}</span>
                <span className="dp-todowrite-meta">
                  {total} 项 · 完成 {realDone}
                  {inferredDone > 0 && (
                    <span className="dp-todowrite-meta-inferred"> + ⚡{inferredDone} 推断</span>
                  )}
                  {ip > 0 && ` / 进行中 ${ip}`}
                </span>
                {isLikelyStale && inferredDone > 0 ? (
                  <span className="dp-todowrite-progress dp-todowrite-progress-stale" title={`真实打勾 ${pct}%，含推断后 ${pctEff}%`}>
                    {pct}% <span className="dp-todowrite-progress-arrow">→</span> {pctEff}%
                  </span>
                ) : (
                  <span className="dp-todowrite-progress">{pct}%</span>
                )}
              </div>
              {/* v0.14.3：滞后警告条 */}
              {isLikelyStale && (
                <div className="dp-todowrite-stale-banner" title="这一帧 plan 状态由 agent 主动调用 TodoWrite 决定；agent 经常合并多次 todo 完成事件，导致前端看到的最后一帧 plan 落后于实际进度。">
                  <span className="dp-todowrite-stale-icon">⚠️</span>
                  <div className="dp-todowrite-stale-text">
                    <strong>plan 状态可能滞后于实际</strong>
                    <span className="dp-todowrite-stale-reason">{staleReasonLabel}</span>
                    {inferredDone > 0 && (
                      <span className="dp-todowrite-stale-recover">
                        下方 {inferredDone} 项已被<strong>推断为已完成</strong>（⚡ 角标 + 浅金背景）
                      </span>
                    )}
                  </div>
                </div>
              )}
              {/* 本次 TodoWrite 的 diff 摘要 */}
              {(((d.newly_completed?.length || 0) + (d.newly_started?.length || 0)
                + (d.newly_added?.length || 0) + (d.removed?.length || 0)
                + (d.status_changes?.length || 0)) > 0) && (
                <div className="dp-todowrite-diff">
                  {(d.newly_completed?.length || 0) > 0 && (
                    <div className="dp-todowrite-diff-row dp-plan-diff-completed">
                      <span className="dp-plan-diff-tag">✅ 完成 {d.newly_completed!.length}</span>
                      <ul>{d.newly_completed!.map((x: any, i: number) => (
                        <li key={i}>{x.content}</li>
                      ))}</ul>
                    </div>
                  )}
                  {(d.newly_started?.length || 0) > 0 && (
                    <div className="dp-todowrite-diff-row dp-plan-diff-started">
                      <span className="dp-plan-diff-tag">▶ 启动 {d.newly_started!.length}</span>
                      <ul>{d.newly_started!.map((x: any, i: number) => (
                        <li key={i}>{x.content}</li>
                      ))}</ul>
                    </div>
                  )}
                  {(d.newly_added?.length || 0) > 0 && (
                    <div className="dp-todowrite-diff-row dp-plan-diff-added">
                      <span className="dp-plan-diff-tag">＋ 新增 {d.newly_added!.length}</span>
                      <ul>{d.newly_added!.map((x: any, i: number) => (
                        <li key={i}>{x.content}</li>
                      ))}</ul>
                    </div>
                  )}
                  {(d.removed?.length || 0) > 0 && (
                    <div className="dp-todowrite-diff-row dp-plan-diff-removed">
                      <span className="dp-plan-diff-tag">− 删除 {d.removed!.length}</span>
                      <ul>{d.removed!.map((x: any, i: number) => (
                        <li key={i}>{x.content}</li>
                      ))}</ul>
                    </div>
                  )}
                </div>
              )}
              {/* 完整 todo 列表 */}
              <table className="dp-todowrite-table">
                <thead>
                  <tr>
                    <th style={{ width: 80 }}>状态</th>
                    <th>内容</th>
                    <th style={{ width: 120 }}>id</th>
                  </tr>
                </thead>
                <tbody>
                  {todos.map((t: any, i: number) => {
                    const isInferred = t.status !== 'completed'
                      && t.id != null && inferredSet.has(String(t.id));
                    const icon = t.status === 'completed' ? '✅'
                      : isInferred ? '⚡'
                      : t.status === 'in_progress' ? '🔄'
                      : t.status === 'cancelled' ? '✖' : '⏳';
                    const statusText = isInferred
                      ? `${t.status} → 推断完成`
                      : t.status;
                    return (
                      <tr key={i} className={`dp-todo-${t.status} ${isInferred ? 'dp-todo-inferred' : ''}`}>
                        <td className="dp-todowrite-status">
                          <span className="dp-todo-icon">{icon}</span>
                          <span className="dp-todo-status-text">{statusText}</span>
                        </td>
                        <td className="dp-todowrite-content">{t.content}</td>
                        <td className="dp-todowrite-id"><span className="dp-mono-sm">{t.id || '—'}</span></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          );
        })()
      )}

      <Section title="基本信息">
        <div className="dp-grid-2">
          <Kv k="Turn" v={step.turn_index} />
          <Kv k="Step" v={step.step_index} />
          {step.tool_name && <Kv k="工具" v={step.tool_name} />}
          {step.tokens > 0 && <Kv k="Tokens" v={step.tokens} />}
          {(step.metadata?.output_tokens ?? 0) > 0 && (
            <Kv k="output_tokens" v={step.metadata!.output_tokens} />
          )}
          {step.duration_ms != null && step.duration_ms > 0 && (
            <Kv k="耗时" v={fmtDur(step.duration_ms)} />
          )}
          {step.timestamp && (
            <Kv k="时间戳" v={<span className="dp-mono-sm">{step.timestamp}</span>} />
          )}
          {step.tool_use_id && <Kv k="Tool Use ID" v={<span className="dp-mono-sm">{step.tool_use_id}</span>} />}
          {invCat && (
            <Kv
              k="调用分类"
              v={<span className={`dp-mono-sm dp-inv-tag dp-inv-${invCat}`}>{invCat}</span>}
            />
          )}
          {step.step_type === 'user_input' && step.metadata?.block_type && (
            <Kv
              k="block_type"
              v={<span className="dp-mono-sm">{step.metadata.block_type}</span>}
            />
          )}
          {step.step_type === 'tool_execution' && typeof step.metadata?.observed_at_ms === 'number' && (
            <Kv
              k="实时事件时刻"
              v={
                <span className="dp-mono-sm" title={String(step.metadata.observed_at_ms)}>
                  {new Date(step.metadata.observed_at_ms).toLocaleString('zh-CN', {
                    month: '2-digit', day: '2-digit',
                    hour: '2-digit', minute: '2-digit', second: '2-digit',
                  })}
                </span>
              }
            />
          )}
        </div>
        {/* v0.15.1: Claude tool_intent 高亮卡片 —— 当 Claude 在 tool_input 里写了
            description / prompt（这是 Claude 4.x 非 thinking 变体唯一暴露的"行级思考"
            来源），把它单独抽成紫色卡片显眼展示，等同 thinking 替代品。
            来源 source 标签让用户知道是 description 还是 prompt 第一行。 */}
        {step.step_type === 'tool_decision' && step.metadata?.tool_intent && (
          <div className="dp-tool-intent">
            <div className="dp-tool-intent-head">
              <span className="dp-tool-intent-icon">💭</span>
              <span className="dp-tool-intent-label">Claude's intent</span>
              {step.metadata?.tool_intent_source && (
                <span className="dp-tool-intent-src">
                  · from <code>tool_input.{step.metadata.tool_intent_source}</code>
                </span>
              )}
            </div>
            <div className="dp-tool-intent-body">
              {String(step.metadata.tool_intent)}
            </div>
          </div>
        )}
        {/* v0.8.2: tool_decision 一行版 input_summary —— 给"扫读"用，
            完整 JSON 在下方 details 里继续保留 */}
        {step.step_type === 'tool_decision' && step.metadata?.input_summary && (
          <div className="dp-input-summary">
            <span className="dp-input-summary-key">入参摘要：</span>
            <span className="dp-input-summary-val dp-mono-sm">{step.metadata.input_summary}</span>
          </div>
        )}
        {/* v0.8.2: final_response 启发式推断标识 —— 让用户一眼看到这是
            "真 final" 还是 cursor_no_stop_reason 启发式补出来的 */}
        {step.step_type === 'final_response' && step.metadata?.inferred_final && (
          <div className="dp-inferred-banner" style={{ marginTop: 8 }}>
            <span className="dp-inferred-icon">⚠️</span>
            <span className="dp-inferred-text">
              此 final_response 是启发式推断（reason:
              <span className="dp-mono-sm">&nbsp;{step.metadata?.inferred_reason || 'unknown'}</span>
              ），并非 Cursor 显式 stop。
            </span>
          </div>
        )}
      </Section>

      {/* v0.8.0: Prompt 完整内容（金色 / 青色 / 绿色边框 by 分类）。
          紧跟"基本信息"，让用户一眼看到送给 LLM 的真实 prompt 或
          打到 RAG 的真实 query —— 这是评估 RAG 召回精度的核心证据。 */}
      {invCat && promptPreview && (
        <Section title={INVOCATION_PROMPT_TITLE[invCat]}>
          <div className={`dp-prompt-section dp-prompt-${invCat}`}>
            <div className="dp-prompt-hint">{INVOCATION_PROMPT_HINT[invCat]}</div>
            <div className="dp-prompt-meta">
              共 {promptFullChars ?? promptPreview.length} 字符
              {typeof promptFullChars === 'number' && promptFullChars > 1024
                ? '（截断到 1024 字预览）'
                : ''}
            </div>
            <CollapsedTextBlock text={promptPreview} threshold={800} />
          </div>
        </Section>
      )}

      {/* v0.8.0: RAG / Web Search 召回片段（青色边框）。
          数据来源优先 observed_output.result_text/stdout（实时回灌），其次 step.content。 */}
      {(invCat === 'rag_query' || invCat === 'web_search') && recallPreview && (
        <Section title="📥 召回片段（RAG / Web 返回的真实内容）">
          <div className="dp-recall-section">
            <div className="dp-prompt-hint">
              这是上述查询打回来的真实结果片段（来自 cot-stream.js 实时流）。对比 prompt 和召回内容，可以判断 RAG 是否检索到真正想要的信息。
            </div>
            <div className="dp-prompt-meta">{recallPreview.length} 字符预览</div>
            <CollapsedTextBlock text={recallPreview} threshold={800} />
          </div>
        </Section>
      )}

      {/* v0.20.11: 当 recall_preview 缺失但有 recall_unavailable_reason 时，
           显式渲染原因（覆盖"工具返回空数组"/"Cursor transcript 不带 tool_result"
           /"实时事件流缺失 after* 事件"三类）。前端不再让用户看到一片空白
           或字面量 [] / {} —— 直接用一句话告诉他为什么没有内容。 */}
      {(invCat === 'rag_query' || invCat === 'web_search') && !recallPreview && recallReason && (
        <Section title="📥 召回片段（RAG / Web 返回的真实内容）">
          <div className="dp-recall-section dp-recall-empty">
            <div className="dp-recall-empty-hint">
              ⚠️ {recallReason}
            </div>
          </div>
        </Section>
      )}

      {/* v0.14.5：之前在 RAG/Web 上还会再画一个独立的"召回片段（暂未捕获）"
          section，但它和下面"调用上下文链路 → 真实返回"区是同一件事的两份
          重复表述（用户能在前端看到两段一样的"无法显示真实召回内容…"），所以
          这里整个删掉，让用户只在"调用上下文链路"的真实返回区看一次。 */}

      {/* v0.8.1: ⭐ 上下文链路（仅 LLM/RAG/Web 调用）—— 把"上一步思考 → 这次调用的入参 →
            真实返回 → 下一步如何使用结果"四件事拼到同一个 section，让用户看 RAG / LLM
            调用时不用上下来回翻。这是"看清楚 prompt 和召回如何被消费"的核心视图。 */}
      {hasRichInvocation && (
        <Section title={`🔗 调用上下文链路（${invCat === 'rag_query' ? 'RAG' : invCat === 'web_search' ? 'Web' : 'LLM'}）`}>
          <div className="dp-chain">
            {prevStep && (prevStep.content || prevStep.step_type.startsWith('thinking')) && (
              <div className="dp-chain-block dp-chain-prev">
                <div className="dp-chain-block-head">
                  <span className="dp-chain-arrow">↑</span>
                  <span className="dp-chain-block-title">前一步：触发本次调用的思考 / 上下文</span>
                  <span className="dp-chain-step-pill">#{prevStep.step_index} · {prevStep.step_type}</span>
                </div>
                <div className="dp-chain-block-body">
                  {prevStep.content
                    ? <CollapsedTextBlock text={prevStep.content} threshold={400} />
                    : <span className="dp-chain-empty-hint">（前一步无文本内容）</span>}
                </div>
              </div>
            )}

            <div className="dp-chain-block dp-chain-curr">
              <div className="dp-chain-block-head">
                <span className="dp-chain-arrow">▶</span>
                <span className="dp-chain-block-title">本步入参（送给 {invCat === 'rag_query' ? 'RAG' : invCat === 'web_search' ? 'Web' : 'LLM'} 的内容）</span>
                {step.tool_name && <span className="dp-chain-step-pill">{step.tool_name}</span>}
              </div>
              <div className="dp-chain-block-body">
                {decisionToolInput
                  ? <CodeBlock data={decisionToolInput} />
                  : promptPreview
                    ? <CollapsedTextBlock text={promptPreview} threshold={400} />
                    : <span className="dp-chain-empty-hint">（无入参）</span>}
              </div>
            </div>

            {/* v0.14.4：真实返回区
                · tool_decision step 永远不可能有 observed_output（那写在配对的
                  tool_execution 上）→ 渲染时直接告诉用户"返回值在下一步"，不
                  画一个永远空的"未捕获到"框（之前那个 fallback 文本误导了好几位）。
                · tool_execution step 才检查真返回；若 result_text 是
                  '{"content":[{"type":"text","text":""}],"isError":false}' 这种
                  空壳 JSON，标识为"工具返回了空 payload"而不是当成真内容渲染。 */}
            <div className="dp-chain-block dp-chain-resp">
              <div className="dp-chain-block-head">
                <span className="dp-chain-arrow">◀</span>
                <span className="dp-chain-block-title">真实返回（{invCat === 'rag_query' || invCat === 'web_search' ? '召回内容' : '工具响应'}）</span>
                {step.step_type === 'tool_decision' && (
                  <span className="dp-chain-resp-source-pill" title="tool_decision 是发起调用的瞬间，返回值落在配对的 tool_execution 上">
                    入参 step（返回看下方 ⚙️）
                  </span>
                )}
                {step.step_type === 'tool_execution' && (() => {
                  // v0.14.7：tool_execution 的真实来源现在有两条：
                  //   - cot-stream.js hook（cursor_events）—— 经常对 MCP 拿不到完整 content
                  //   - 本地 MCP 代理（mcp_proxy）—— wire 字节直采，RAG 真值唯一可靠源
                  // observed_source 形如 "mcp_proxy" / "cursor_events" / "mcp_proxy+cursor_events"
                  const src = (step.metadata?.observed_source || '') as string;
                  const isProxy = src.startsWith('mcp_proxy');
                  if (isProxy) {
                    const trafficTs = step.metadata?.mcp_traffic_ts_ms;
                    const tooltipParts: string[] = ['返回值由本地 MCP 代理 wire 字节直采（mcp_traffic_proxy.js）'];
                    if (trafficTs) tooltipParts.push(`代理记录时间: ${new Date(trafficTs).toLocaleString('zh-CN')}`);
                    if (step.metadata?.mcp_traffic_elapsed_ms) tooltipParts.push(`上游响应耗时: ${step.metadata.mcp_traffic_elapsed_ms}ms`);
                    return (
                      <span
                        className="dp-chain-resp-source-pill dp-chain-resp-source-proxy"
                        title={tooltipParts.join('\n')}
                      >
                        📡 MCP 代理（wire 真值）
                      </span>
                    );
                  }
                  return (
                    <span className="dp-chain-resp-source-pill dp-chain-resp-source-exec" title="返回值来自 cot-stream.js 实时回灌（observed_output.result_text）">
                      实时回灌
                    </span>
                  );
                })()}
              </div>
              <div className="dp-chain-block-body">
                {(() => {
                  // tool_decision：返回值不在自己身上，直接给说明，不画 fallback
                  if (step.step_type === 'tool_decision') {
                    return (
                      <div className="dp-chain-empty-hint">
                        （此步是<strong>发起调用</strong>瞬间，真实返回写在配对的 <code>tool_execution</code> step 上 —— 点开 SpanTree 里下一行的 ⚙️ 可看到）
                      </div>
                    );
                  }
                  // 优先用 observed_output（实时回灌真值），不用 recall_preview，
                  // 因为 recall_preview 在历史调用上有时回填的是 content（synthetic 占位文本）
                  const realText: string = (
                    observedOutput?.result_text
                    ?? observedOutput?.stdout
                    ?? recallPreview
                    ?? ''
                  ) as string;
                  // 空壳 JSON 检测：MCP 经常返回 {"content":[{"type":"text","text":""}],"isError":false}
                  // 这是合法的"返回了，但是没内容"，不是 fallback hint。
                  const isEmptyShell = /^\s*\{\s*"content"\s*:\s*\[\s*\{\s*"type"\s*:\s*"text"\s*,\s*"text"\s*:\s*""\s*\}\s*\]\s*,\s*"isError"\s*:\s*false\s*\}\s*$/.test(
                    realText.trim()
                  );
                  if (isEmptyShell) {
                    return (
                      <div className="dp-chain-empty-shell">
                        <span className="dp-chain-empty-shell-tag">空 payload</span>
                        <span className="dp-chain-empty-shell-text">
                          MCP 返回了合法响应但 <code>content[0].text</code> 是空字符串
                          —— 这是工具自己什么都没回，不是采集层缺失。
                        </span>
                      </div>
                    );
                  }
                  if (realText.trim()) {
                    return <CollapsedTextBlock text={realText} threshold={400} />;
                  }
                  // 真没拿到，给最准确的 reason
                  return (
                    <div className="dp-chain-empty-hint">
                      {recallReason || '（未捕获到 tool_result；可能是 cot-stream.js 在此调用发生时未挂载，或工具本身没回 result block）'}
                    </div>
                  );
                })()}
              </div>
            </div>

            {postStep && (postStep.content || postStep.metadata?.tool_input) && (
              <div className="dp-chain-block dp-chain-next">
                <div className="dp-chain-block-head">
                  <span className="dp-chain-arrow">↓</span>
                  <span className="dp-chain-block-title">
                    下一步：拿到{invCat === 'rag_query' ? '召回' : invCat === 'web_search' ? '搜索结果' : 'LLM 输出'}后的下一次思考 / 决策
                  </span>
                  <span className="dp-chain-step-pill">#{postStep.step_index} · {postStep.step_type}</span>
                </div>
                <div className="dp-chain-block-body">
                  {postStep.content
                    ? <CollapsedTextBlock text={postStep.content} threshold={400} />
                    : postStep.metadata?.tool_input
                      ? <CodeBlock data={postStep.metadata.tool_input} />
                      : <span className="dp-chain-empty-hint">（无文本内容）</span>}
                </div>
              </div>
            )}
            {!postStep && (
              <div className="dp-chain-block dp-chain-next dp-chain-next-end">
                <div className="dp-chain-block-head">
                  <span className="dp-chain-arrow">⏹</span>
                  <span className="dp-chain-block-title">这是本轮最后一个被记录的动作（无后续）</span>
                </div>
              </div>
            )}
          </div>
        </Section>
      )}

      {/* "内容"区：一旦这步已经被 cot-stream.js / MCP 代理回灌真实数据，
          就别再在"内容"里把同一份输出重复贴一遍——改为精简说明，
          把完整 command/stdout/stderr 下移到专门的「实时观测」区。 */}
      {(() => {
        const src = (step.metadata?.observed_source || '') as string;
        const isInjected = src.startsWith('cursor_events') || src.startsWith('mcp_proxy');
        if (!step.content) return null;
        // v0.16.2: tool_execution 的"内容"是 transcript 里 tool_result.content 原文
        // 截断（≤2000 字符）。is_error=true 时，这段是工具失败的真实输出
        // （例如 stderr / Python traceback / WebFetch 文档预览），**不是**模型
        // 的 thinking。给加一条说明 banner 避免误读。
        const isToolExec = step.step_type === 'tool_execution';
        const isErr = !!step.metadata?.is_error;
        const resultLen = (step.metadata?.result_len as number | undefined) || 0;
        const truncated = !!step.metadata?.truncated;
        const fullLen = resultLen || step.content.length;
        if (!isInjected) {
          return (
            <Section title="内容">
              {isToolExec && (
                <div className={`dp-tool-output-banner${isErr ? ' dp-tool-output-banner-error' : ''}`}>
                  <span className="dp-tool-output-banner-icon">
                    {isErr ? '⚠️' : 'ℹ️'}
                  </span>
                  <div className="dp-tool-output-banner-text">
                    <strong>{isErr ? '工具执行失败的真实输出' : '工具执行的真实输出'}</strong>
                    <span className="dp-tool-output-banner-meta">
                      （来自 transcript 里这一步的 <code>tool_result.content</code> ——
                      可能是 stdout / stderr / 文件正文 / API 返回；
                      <strong>不是模型的 thinking</strong>）。
                    </span>
                    {truncated && (
                      <span className="dp-tool-output-banner-trunc">
                        · 已截断显示 2000 / {fullLen.toLocaleString()} 字符
                      </span>
                    )}
                  </div>
                </div>
              )}
              <div className={`dp-prose ${isErr ? 'dp-prose-error' : ''}`}>
                {step.content}
              </div>
            </Section>
          );
        }
        const sourceLabel = src.startsWith('mcp_proxy')
          ? '本地 MCP 代理（wire 字节直采）'
          : 'cot-stream.js 实时回灌';
        return (
          <Section title="内容">
            <div className="dp-obs-note">
              此 <code>tool_execution</code> 的真实命令与输出由 <code>{sourceLabel}</code>
              提供，完整数据展示在下方「实时观测」区。
            </div>
          </Section>
        );
      })()}

      {/* v0.7.0+：cot-stream.js 实时回灌的真实 shell / MCP 输出；
          v0.14.7：本地 MCP 代理也写到这里，标题根据来源切换。 */}
      {(() => {
        const src = (step.metadata?.observed_source || '') as string;
        const isInjected = src.startsWith('cursor_events') || src.startsWith('mcp_proxy');
        const show = isInjected || hasRealObserved;
        if (!show) return null;
        const isProxy = src.startsWith('mcp_proxy');
        const sectionTitle = isProxy
          ? '📡 实时观测（来自 MCP 代理 wire 抓包）'
          : '📡 实时观测（来自 cot-stream.js）';
        return (
        <Section title={sectionTitle}>
          <div className="dp-obs-box">
            <div className="dp-obs-head">
              <span className="dp-obs-pill dp-obs-ok">
                ✓ real I/O from Cursor hook
              </span>
              {step.metadata?.synthetic_upgraded && (
                <span className="dp-obs-pill dp-obs-upgrade">
                  synthetic ↑ upgraded
                </span>
              )}
              {typeof observedOutput?.exit_code === 'number' && (
                <span
                  className={
                    'dp-obs-pill ' +
                    (observedOutput.exit_code === 0 ? 'dp-obs-ok' : 'dp-obs-err')
                  }
                >
                  exit={observedOutput.exit_code}
                </span>
              )}
              {observedOutput?.is_error === true && (
                <span className="dp-obs-pill dp-obs-err">MCP isError</span>
              )}
              {typeof observedOutput?.duration_ms === 'number' && (
                <span className="dp-obs-pill">
                  {fmtDur(observedOutput.duration_ms)}
                </span>
              )}
              {typeof observedOutput?.result_text_chars === 'number' && (
                <span className="dp-obs-pill">{observedOutput.result_text_chars} chars</span>
              )}
            </div>

            {/* 入参区 —— Shell / MCP / Read / Edit 全覆盖 */}
            {observedInput?.command && (
              <>
                <div className="dp-obs-sub">$ command</div>
                <CodeBlock data={observedInput.command} />
              </>
            )}
            {observedInput?.cwd && (
              <Kv k="cwd" v={<span className="dp-mono-sm">{observedInput.cwd}</span>} />
            )}
            {observedInput?.file_path && (
              <Kv k="file" v={<span className="dp-mono-sm">{observedInput.file_path}</span>} />
            )}
            {observedInput?.url && (
              <Kv k="url" v={<span className="dp-mono-sm">{observedInput.url}</span>} />
            )}
            {observedInput?.query && (
              <Kv k="query" v={<span className="dp-mono-sm">{observedInput.query}</span>} />
            )}
            {observedInput?.mcp_server && (
              <Kv k="mcp_server" v={<span className="dp-mono-sm">{observedInput.mcp_server}</span>} />
            )}
            {observedInput?.tool_input && (
              <>
                <div className="dp-obs-sub">tool_input</div>
                <CodeBlock data={observedInput.tool_input} />
              </>
            )}

            {/* 返回区 */}
            {observedOutput?.result_text && (
              <>
                <div className="dp-obs-sub dp-obs-sub-result">── result_text (MCP) ──</div>
                <CollapsedTextBlock text={String(observedOutput.result_text)} threshold={1200} />
              </>
            )}
            {observedOutput?.stdout && (
              <>
                <div className="dp-obs-sub">── stdout ──</div>
                <CollapsedTextBlock text={String(observedOutput.stdout)} threshold={1200} />
              </>
            )}
            {observedOutput?.stderr && (
              <>
                <div className="dp-obs-sub dp-obs-sub-err">── stderr ──</div>
                <CodeBlock data={observedOutput.stderr} />
              </>
            )}
          </div>
        </Section>
        );
      })()}

      {/* v0.8.1: 执行元信息 —— synthetic / truncated / result_len 这些以前藏在 Metadata
          里翻不到，现在专门展开成一行 chips，方便用户一眼判断"这步结果是真的还是占位"。 */}
      {step.step_type === 'tool_execution' && (
        (step.metadata?.synthetic !== undefined
         || typeof step.metadata?.result_len === 'number'
         || step.metadata?.truncated
         || step.metadata?.synthetic_reason) && (
        <Section title="执行状态">
          <div className="dp-exec-state-row">
            {step.metadata?.synthetic === true && (
              <span className="dp-exec-chip dp-exec-chip-warn" title="此节点的结果是 cot_extractor 合成的占位，并非真实工具返回">
                ◌ synthetic（占位）
              </span>
            )}
            {step.metadata?.synthetic === false && (
              <span className="dp-exec-chip dp-exec-chip-ok">✓ real</span>
            )}
            {step.metadata?.synthetic_upgraded && (
              <span className="dp-exec-chip dp-exec-chip-upgrade" title="原本是 synthetic 占位，被 cot-stream.js 实时回灌后升级">
                ↑ upgraded by stream
              </span>
            )}
            {step.metadata?.is_error && (
              <span className="dp-exec-chip dp-exec-chip-err">❌ is_error</span>
            )}
            {typeof step.metadata?.result_len === 'number' && (
              <span className="dp-exec-chip">{step.metadata.result_len} chars</span>
            )}
            {step.metadata?.truncated && (
              <span className="dp-exec-chip dp-exec-chip-warn">truncated</span>
            )}
          </div>
          {step.metadata?.synthetic_reason && (
            <div className="dp-exec-reason">
              <span className="dp-exec-reason-key">synthetic_reason：</span>
              <span className="dp-exec-reason-val">{step.metadata.synthetic_reason}</span>
            </div>
          )}
        </Section>
      ))}

      {/* 摘要式推理 */}
      {step.reasoning_digest && (
        <Section title="🧠 摘要式推理">
          <div className="dp-reasoning-digest">
            <div className="dp-rd-item">
              <span className="dp-rd-label">💡 理由</span>
              <span className="dp-rd-value">{step.reasoning_digest.why}</span>
            </div>
            <div className="dp-rd-item">
              <span className="dp-rd-label">📋 证据</span>
              <span className="dp-rd-value">{step.reasoning_digest.evidence}</span>
            </div>
            <div className="dp-rd-item">
              <span className="dp-rd-label">⚖️ 依据</span>
              <span className="dp-rd-value">{step.reasoning_digest.basis}</span>
            </div>
            <div className="dp-rd-item">
              <span className="dp-rd-label">➡️ 下一步</span>
              <span className="dp-rd-value">{step.reasoning_digest.next_plan}</span>
            </div>
          </div>
        </Section>
      )}

      {/* 决策轨迹 */}
      {step.decision_trace && (
        <Section title="🎯 决策轨迹">
          <div className="dp-decision-trace">
            <div className="dp-dt-item">
              <span className="dp-dt-label">触发上下文</span>
              <span className="dp-dt-value">{step.decision_trace.trigger_context}</span>
            </div>
            <div className="dp-dt-item">
              <span className="dp-dt-label">工具选择</span>
              <span className="dp-dt-value">{step.decision_trace.tool_selection_reason}</span>
            </div>
            <div className="dp-dt-item">
              <span className="dp-dt-label">参数推断</span>
              <span className="dp-dt-value">{step.decision_trace.param_inference}</span>
            </div>
            <div className="dp-dt-item">
              <span className="dp-dt-label">后续决策</span>
              <span className="dp-dt-value">{step.decision_trace.continuation_reason}</span>
            </div>
          </div>
        </Section>
      )}

      {/* 状态演化 */}
      {step.state_evolution && (
        <Section title="📊 状态演化">
          <div className="dp-grid-2">
            <Kv k="上下文 Hash" v={<span className="dp-mono-sm">{step.state_evolution.context_hash}</span>} />
            <Kv k="Action" v={<span className="dp-mono-sm">{step.state_evolution.action_schema}</span>} />
          </div>
          {step.state_evolution.evidence_summary && (
            <div className="dp-prose" style={{ marginTop: 8, fontSize: 11 }}>
              {step.state_evolution.evidence_summary}
            </div>
          )}
          <div className="dp-termination-check" style={{ marginTop: 8 }}>
            <span className="dp-mono-sm">{step.state_evolution.termination_check}</span>
          </div>
        </Section>
      )}

      {/* 错误形成路径 */}
      {step.error_trace && step.error_trace.is_error_origin && (
        <Section title="🚨 错误形成路径">
          <div className="dp-error-trace">
            <div className="dp-et-item">
              <span className="dp-et-label">⚡ 错误起源</span>
              <span className="dp-et-value">Step #{step.error_trace.error_step_index}</span>
            </div>
            {step.error_trace.referenced_by.length > 0 && (
              <div className="dp-et-item">
                <span className="dp-et-label">📎 被引用</span>
                <span className="dp-et-value">
                  {step.error_trace.referenced_by.map(i => `#${i}`).join(', ')}
                </span>
              </div>
            )}
            {step.error_trace.correction_opportunity && (
              <div className="dp-et-item dp-et-warning">
                <span className="dp-et-label">⚠️ 未纠正</span>
                <span className="dp-et-value">有纠正机会但未纠正</span>
              </div>
            )}
            {step.error_trace.contradicts_final && (
              <div className="dp-et-item dp-et-danger">
                <span className="dp-et-label">❌ 矛盾</span>
                <span className="dp-et-value">与最终答案存在矛盾</span>
              </div>
            )}
          </div>
        </Section>
      )}

      {step.metadata && Object.keys(step.metadata).length > 0 && (
        <Section title="Metadata">
          <CodeBlock data={step.metadata} />
        </Section>
      )}

      {/* v0.20.7: 「本步 Token」—— 当前 step 自己的 OTel token_usage，
          同 turn 下不同 step 显示不同数字（跟 SpanTree chip 一致）。 */}
      {(() => {
        const st = readStepTokens(step);
        if (!st) return null;
        if (st.is_shared) {
          return (
            <Section title="本步 Token">
              <div className="dp-note dp-note-gray">
                此 step 与同一次 LLM API 调用的上一步合并显示，
                token 已挂在前一个 step 上以避免重复计数。
              </div>
            </Section>
          );
        }
        return (
          <Section title="本步 Token">
            <div className="dp-grid-2">
              <Kv k="输入" v={st.input.toLocaleString()} />
              <Kv k="输出" v={st.output.toLocaleString()} />
              {!!st.cache_read && (
                <Kv k="cache_read" v={st.cache_read.toLocaleString()} />
              )}
              {!!st.cache_creation && (
                <Kv k="cache_write" v={st.cache_creation.toLocaleString()} />
              )}
              {st.cost != null && st.cost > 0 && (
                <Kv k="本步成本" v={`$${st.cost.toFixed(6)}`} />
              )}
              <Kv
                k="类型"
                v={
                  st.is_llm_call
                    ? '🧠 LLM 调用'
                    : '⚙ 非 LLM（内容会作为下次 LLM input）'
                }
              />
              {/* v0.20.11: 来源可信度徽章（三家 IDE 都已支持 per-call 真值） */}
              {st.source && (
                <Kv
                  k="来源"
                  v={
                    st.source === 'transcript_per_message'
                      ? '✓ Claude API 真值（每次调用）'
                      : st.source === 'events_per_generation'
                      ? '✓ Cursor 真值（per-generation）'
                      : st.source === 'index_per_request'
                      ? '✓ CodeBuddy 真值（per-request）'
                      : st.source === 'turn_real_apportioned'
                      ? '≈ Turn 真值按字符分摊'
                      : st.source === 'shared_with_anchor'
                      ? '↗ 同 LLM call 已挂在前一步'
                      : st.source === 'char_estimate'
                      ? '~ 字符启发式估算'
                      : st.source === 'missing_transcript'
                      ? '⚠ transcript 缺失'
                      : st.source === 'non_llm'
                      ? '· 非 LLM step'
                      : st.source
                  }
                />
              )}
            </div>
          </Section>
        );
      })()}

      {/* v0.20.7: 「所在 Turn 累计」section 已下线 —— turn 级累加跨多个
          数据源（hook 真值 / step 累加 / enricher），在 Cursor afterAgentResponse
          hook 未触发的 turn 上始终无法给出权威值。整轮信息已经在 Turn
          header（SpanTree 上的徽章）和 Turn 节点的「Token 消耗」section
          里完整展示，这里仅保留「本步 Token」单步真值即可。 */}
    </>
  );
}

// ─── 主组件 ───────────────────────────────────────────────
export default function DetailPanel({
  node,
  onSelectNode,
  turnEvalReports = {},
  liveCritic = null,
  turnEvalLoadingKey,
  turnEvalError,
}: Props) {
  if (!node) {
    return (
      <div className="detail-panel">
        <div className="dp-empty">
          <div className="dp-empty-icon">←</div>
          <div className="dp-empty-text">点击左侧节点</div>
          <div className="dp-empty-sub">查看详细信息</div>
        </div>
      </div>
    );
  }

  return (
    <div className="detail-panel">
      {node.kind === 'session' && <SessionDetail node={node} onSelectNode={onSelectNode} />}
      {node.kind === 'turn'    && (
        <TurnDetail
          node={node}
          evalReport={turnEvalReports[`${node.cot.session_id}:${node.turn.turn_index}`]}
          isEvalLoading={turnEvalLoadingKey === `${node.cot.session_id}:${node.turn.turn_index}`}
          evalError={turnEvalError}
          liveCritic={liveCritic}
        />
      )}
      {node.kind === 'step'    && <StepDetail    node={node} />}
    </div>
  );
}
