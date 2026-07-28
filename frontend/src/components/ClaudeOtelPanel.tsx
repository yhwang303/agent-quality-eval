/**
 * ClaudeOtelPanel —— Claude Code 原生 OTel 数据展示。
 *
 * v0.16.0 引入。Claude Code 自带 OpenTelemetry，启用 CLAUDE_CODE_ENABLE_TELEMETRY=1
 * + OTEL_EXPORTER_OTLP_ENDPOINT 后，会把三类信号 POST 到我们后端：
 *
 *   logs/events  → user_prompt / api_request / tool_result / tool_decision / ...
 *   metrics      → token / cost / decision / lines_of_code 计数
 *   traces beta  → claude_code.interaction → llm_request / tool 树
 *
 * 这个 Panel 直接消费扁平化后的数据，不依赖 transcript / hook
 * （那些是另一条独立通道）。本 Panel 显示的是 **Claude Code 进程亲口
 * 上报的官方真值**，权威性最高。
 *
 * 渲染分四段：
 *   1. KPI 行：tokens / cost / events / spans / 模型
 *   2. Events 时间线：按 prompt.id 分组，每组一个折叠卡
 *   3. Metrics 总览：metric → 各 attribute 维度的最新值表格
 *   4. Traces 树：claude_code.interaction 根 → llm_request / tool 子节点
 */
import { useEffect, useState } from 'react';
import { api, saveDownloadedFile } from '../hooks/api';
import type {
  ClaudeOtelData,
  ClaudeOtelEvent,
  ClaudeOtelSpan,
} from '../hooks/api';

interface Props {
  sessionId: string;
  // v0.16.1: 当被 OtelPanel 嵌入时，外层已经提供 header + 下载按钮，这里就不
  // 再显示自己的 toolbar / 下载按钮，避免重复。Standalone 模式时仍然显示。
  hideInternalToolbar?: boolean;
}

function fmtNum(n: number | undefined | null): string {
  if (n == null || isNaN(Number(n))) return '0';
  const num = Number(n);
  if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(2)}M`;
  if (num >= 1_000) return `${(num / 1_000).toFixed(1)}K`;
  return String(num);
}

function fmtDur(ms?: number | null): string {
  if (ms == null) return '';
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(2)}s`;
  return `${(ms / 60000).toFixed(1)}min`;
}

function fmtTs(ts?: string | null): string {
  if (!ts) return '';
  // 只显示 HH:MM:SS.mmm，让时间线对齐更紧凑
  const m = ts.match(/T(\d{2}:\d{2}:\d{2}(?:\.\d+)?)/);
  return m ? m[1] : ts;
}

const saveJsonBlob = (blob: Blob, filename: string) =>
  saveDownloadedFile({ blob, filename });

// v0.16.2: 汇总 hook_execution_* 事件 —— 这是 Claude Code 27 个 hooks 的真实
// 触发记录（通过 OTel 通道，权威性高于自部署的 hook 脚本）。每对 start/complete
// 描述一次 hook 触发：hook_event = SessionStart / PreToolUse / PostToolUse /
// UserPromptSubmit / Stop / Notification / PreCompact / SubagentStop ...
interface HookStat {
  hook_event: string;        // 例 "PreToolUse"
  total: number;             // 总触发次数
  total_duration_ms: number; // 累计耗时（来自 complete 事件的 total_duration_ms 属性）
  num_success: number;
  num_blocking: number;
  num_cancelled: number;
  num_non_blocking_error: number;
  hook_names: Set<string>;   // 同一 hook_event 下挂的脚本名集合
  last_ts: string;
}

interface HookSummary {
  total_executions: number;       // hook_execution_complete 总数
  total_duration_ms: number;
  by_event: HookStat[];           // 按触发次数排序
}

// v0.16.3: 按 tool_name 聚合 OTel 看到的全部工具调用——这是"OTel 真值视野"，
// 包含主 agent + 所有 subagent 嵌套；通常远大于主 transcript 解析出的工具数
// （因为 subagent 内部的工具调用不进主 transcript），而 OTel 直接采样进程内
// 所有 prompt.id 的活动。
interface ToolStat {
  tool_name: string;
  total: number;        // 调用次数（来自 tool_decision events）
  success: number;      // tool_result.success === 'true'
  failure: number;      // tool_result.success === 'false'
  total_duration_ms: number;
  total_result_bytes: number;
  err_types: Map<string, number>;  // 按 error_type 分布
}

export interface ToolSummary {
  total_calls: number;          // OTel tool_decision events 总数
  unique_tools: number;
  by_tool: ToolStat[];          // 按调用次数排序
  total_duration_ms: number;
  total_result_bytes: number;
}

export function summarizeToolEvents(events: ClaudeOtelEvent[]): ToolSummary {
  const map = new Map<string, ToolStat>();
  let totalDur = 0;
  let totalBytes = 0;
  for (const e of events) {
    const ev = (e.event_name || (e.attributes as any)?.['event.name'] || '').toString();
    const a = (e.attributes || {}) as Record<string, any>;
    const toolName = String(a.tool_name || '');
    if (!toolName) continue;
    if (ev !== 'tool_decision' && ev !== 'tool_result') continue;
    let stat = map.get(toolName);
    if (!stat) {
      stat = {
        tool_name: toolName,
        total: 0,
        success: 0,
        failure: 0,
        total_duration_ms: 0,
        total_result_bytes: 0,
        err_types: new Map(),
      };
      map.set(toolName, stat);
    }
    if (ev === 'tool_decision') {
      stat.total += 1;
    } else if (ev === 'tool_result') {
      const dur = Number(a.duration_ms) || 0;
      const bytes = Number(a.tool_result_size_bytes) || 0;
      stat.total_duration_ms += dur;
      stat.total_result_bytes += bytes;
      totalDur += dur;
      totalBytes += bytes;
      const succ = String(a.success || '');
      if (succ === 'true') stat.success += 1;
      else if (succ === 'false') {
        stat.failure += 1;
        const et = String(a.error_type || 'error');
        stat.err_types.set(et, (stat.err_types.get(et) || 0) + 1);
      }
    }
  }
  const by_tool = Array.from(map.values()).sort((a, b) => b.total - a.total);
  return {
    total_calls: by_tool.reduce((s, x) => s + x.total, 0),
    unique_tools: by_tool.length,
    by_tool,
    total_duration_ms: totalDur,
    total_result_bytes: totalBytes,
  };
}

function summarizeHookEvents(events: ClaudeOtelEvent[]): HookSummary {
  const map = new Map<string, HookStat>();
  let total = 0;
  let totalDur = 0;
  for (const e of events) {
    const ev = (e.event_name || (e.attributes as any)?.['event.name'] || '').toString();
    if (ev !== 'hook_execution_complete') continue;
    const a = (e.attributes || {}) as Record<string, any>;
    const hookEvent = String(a.hook_event || a['hook_event'] || 'unknown');
    const dur = Number(a.total_duration_ms) || 0;
    total += 1;
    totalDur += dur;
    let stat = map.get(hookEvent);
    if (!stat) {
      stat = {
        hook_event: hookEvent,
        total: 0,
        total_duration_ms: 0,
        num_success: 0,
        num_blocking: 0,
        num_cancelled: 0,
        num_non_blocking_error: 0,
        hook_names: new Set<string>(),
        last_ts: '',
      };
      map.set(hookEvent, stat);
    }
    stat.total += 1;
    stat.total_duration_ms += dur;
    stat.num_success += Number(a.num_success) || 0;
    stat.num_blocking += Number(a.num_blocking) || 0;
    stat.num_cancelled += Number(a.num_cancelled) || 0;
    stat.num_non_blocking_error += Number(a.num_non_blocking_error) || 0;
    if (a.hook_name) stat.hook_names.add(String(a.hook_name));
    if (e.ts && (!stat.last_ts || e.ts > stat.last_ts)) stat.last_ts = e.ts;
  }
  const by_event = Array.from(map.values()).sort((a, b) => b.total - a.total);
  return { total_executions: total, total_duration_ms: totalDur, by_event };
}

// 把 events 按 prompt.id 分组（OTel 标准的关联属性，每个 user_prompt 对应一个）
interface PromptGroup {
  prompt_id: string;
  user_prompt?: ClaudeOtelEvent;
  api_requests: ClaudeOtelEvent[];
  tool_results: ClaudeOtelEvent[];
  others: ClaudeOtelEvent[];
}

function groupByPrompt(events: ClaudeOtelEvent[]): PromptGroup[] {
  const groups = new Map<string, PromptGroup>();
  const orphans: PromptGroup = {
    prompt_id: '__orphan__',
    api_requests: [],
    tool_results: [],
    others: [],
  };
  for (const e of events) {
    const pid = (e.attributes?.['prompt.id'] as string) || '';
    const target = pid ? (groups.get(pid) || (() => {
      const g: PromptGroup = {
        prompt_id: pid,
        api_requests: [],
        tool_results: [],
        others: [],
      };
      groups.set(pid, g);
      return g;
    })()) : orphans;
    const ev = (e.event_name || e.attributes?.['event.name'] || '').toString();
    if (ev === 'user_prompt') target.user_prompt = e;
    else if (ev === 'api_request' || ev === 'api_request_body') target.api_requests.push(e);
    else if (ev === 'tool_result') target.tool_results.push(e);
    else target.others.push(e);
  }
  const out = Array.from(groups.values());
  if (orphans.user_prompt || orphans.api_requests.length || orphans.tool_results.length || orphans.others.length) {
    out.push(orphans);
  }
  // 按第一个 user_prompt 的时间排序（缺失的丢到末尾）
  out.sort((a, b) => {
    const ta = a.user_prompt?.ts || a.api_requests[0]?.ts || '9999';
    const tb = b.user_prompt?.ts || b.api_requests[0]?.ts || '9999';
    return ta.localeCompare(tb);
  });
  return out;
}

// 把 spans 组织成树
interface SpanNode extends ClaudeOtelSpan {
  children: SpanNode[];
}

function buildSpanTree(spans: ClaudeOtelSpan[]): SpanNode[] {
  const byId = new Map<string, SpanNode>();
  spans.forEach(s => {
    if (s.span_id) byId.set(s.span_id, { ...s, children: [] });
  });
  const roots: SpanNode[] = [];
  byId.forEach(node => {
    const pid = node.parent_span_id;
    if (pid && byId.has(pid)) {
      byId.get(pid)!.children.push(node);
    } else {
      roots.push(node);
    }
  });
  // 子节点按 start_ts 排
  byId.forEach(n => n.children.sort((a, b) =>
    (a.start_ts || '').localeCompare(b.start_ts || '')));
  roots.sort((a, b) => (a.start_ts || '').localeCompare(b.start_ts || ''));
  return roots;
}

function SpanTreeNode({ node, depth = 0 }: { node: SpanNode; depth?: number }) {
  const dur = node.attributes?.duration_ms || node.attributes?.['interaction.duration_ms'];
  const isErr = node.status === 2; // OTel STATUS_CODE_ERROR
  const tone =
    node.name === 'claude_code.interaction' ? '#3b82f6' :
    node.name === 'claude_code.llm_request' ? '#a855f7' :
    node.name === 'claude_code.tool' ? '#10b981' :
    node.name === 'claude_code.hook' ? '#f59e0b' :
    '#64748b';
  const subtitle: string[] = [];
  if (node.attributes?.model) subtitle.push(String(node.attributes.model));
  if (node.attributes?.tool_name) subtitle.push(String(node.attributes.tool_name));
  if (node.attributes?.input_tokens) subtitle.push(`in:${fmtNum(node.attributes.input_tokens)}`);
  if (node.attributes?.output_tokens) subtitle.push(`out:${fmtNum(node.attributes.output_tokens)}`);
  if (node.attributes?.stop_reason) subtitle.push(`stop:${node.attributes.stop_reason}`);

  return (
    <div className="otel-span-node" style={{ marginLeft: depth * 16 }}>
      <div className={`otel-span-row ${isErr ? 'otel-span-err' : ''}`}>
        <span className="otel-span-name" style={{ color: tone, borderLeftColor: tone }}>
          {node.name}
        </span>
        {dur != null && (
          <span className="otel-span-dur">{fmtDur(Number(dur))}</span>
        )}
        {subtitle.length > 0 && (
          <span className="otel-span-sub">{subtitle.join(' · ')}</span>
        )}
        {node.start_ts && (
          <span className="otel-span-ts">{fmtTs(node.start_ts)}</span>
        )}
      </div>
      {node.children.map(c => (
        <SpanTreeNode key={c.span_id} node={c} depth={depth + 1} />
      ))}
    </div>
  );
}

export function ClaudeOtelPanel({ sessionId, hideInternalToolbar = false }: Props) {
  const [data, setData] = useState<ClaudeOtelData | null>(null);
  const [err, setErr] = useState<string>('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!sessionId) return;
    setLoading(true);
    setErr('');
    api.getClaudeOtel(sessionId)
      .then(d => setData(d))
      .catch(e => setErr(e?.message || String(e)))
      .finally(() => setLoading(false));
  }, [sessionId]);

  if (loading) {
    return <div className="cl-otel cl-otel-loading">加载 OTel 数据中…</div>;
  }
  if (err) {
    return (
      <div className="cl-otel cl-otel-empty">
        无法加载 OTel 数据：<code>{err}</code>
      </div>
    );
  }
  if (!data || (data.summary.events_total + data.summary.metrics_total + data.summary.spans_total) === 0) {
    return (
      <div className="cl-otel cl-otel-empty">
        <div className="cl-otel-empty-head">📡 本会话尚未收到 Claude Code OTel 数据</div>
        <div className="cl-otel-empty-body">
          要让 Claude Code 自己上报 OTel 真值，需要在启动前设置环境变量：
          <pre>{`# 写入 ~/.claude/settings.json 的 env 字段
{
  "env": {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
    "OTEL_LOGS_EXPORTER": "otlp",
    "OTEL_METRICS_EXPORTER": "otlp",
    "OTEL_EXPORTER_OTLP_PROTOCOL": "http/json",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:8765",
    "OTEL_LOG_USER_PROMPTS": "1",
    "OTEL_LOG_TOOL_DETAILS": "1"
  }
}`}</pre>
          配好后重启 Claude Code 进程，下一次 prompt 就会推到本后端。
          数据按 <code>session.id</code> 落到 <code>~/.claude/state/otel/&lt;sid&gt;/</code>。
        </div>
      </div>
    );
  }

  const { summary, events, metrics, spans } = data;
  const promptGroups = groupByPrompt(events);
  const spanTree = buildSpanTree(spans);
  const hookStats = summarizeHookEvents(events);
  const toolStats = summarizeToolEvents(events);

  // 把整个 OTel 数据序列化下载——这是 Claude Code 进程亲自上报的原始扁平 OTel
  // 数据（events.jsonl + metrics.jsonl + traces.jsonl 的解析合并版），
  // 可以离线塞 Phoenix / SigNoz / 自定义脚本。
  const handleDownload = () => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    saveJsonBlob(blob, `claude-otel-${sessionId.slice(0, 8)}.json`).catch(e => {
      setErr(e?.message || String(e));
    });
  };

  return (
    <div className="cl-otel">
      {/* ── 顶部操作条（嵌入 OtelPanel 时隐藏，由外层提供）── */}
      {!hideInternalToolbar && (
        <div className="otel-toolbar">
          <span className="otel-toolbar-label">
            📡 来自 Claude Code 进程的 OTLP 原始上报
            <span className="otel-toolbar-meta">
              {' '}· {summary.events_total} events · {summary.metrics_total} metrics
              {summary.spans_total > 0 ? ` · ${summary.spans_total} spans` : ''}
            </span>
          </span>
          <button
            className="otel-download-btn"
            onClick={handleDownload}
            title="导出当前 session 的完整 OTel 数据（events + metrics + spans + summary）为单个 JSON 文件"
          >
            📥 下载原始 OTel 数据
          </button>
        </div>
      )}

      {/* ── 1. KPI 行 ── */}
      <div className="otel-kpi-row">
        <div className="otel-kpi">
          <div className="otel-kpi-label">Input</div>
          <div className="otel-kpi-val">{fmtNum(summary.totals.input_tokens)}</div>
        </div>
        <div className="otel-kpi">
          <div className="otel-kpi-label">Output</div>
          <div className="otel-kpi-val">{fmtNum(summary.totals.output_tokens)}</div>
        </div>
        <div className="otel-kpi">
          <div className="otel-kpi-label">Cache R/W</div>
          <div className="otel-kpi-val">
            {fmtNum(summary.totals.cache_read_tokens)} / {fmtNum(summary.totals.cache_creation_tokens)}
          </div>
        </div>
        <div className="otel-kpi">
          <div className="otel-kpi-label">Cost</div>
          <div className="otel-kpi-val">${summary.totals.cost_usd.toFixed(4)}</div>
        </div>
        <div className="otel-kpi">
          <div className="otel-kpi-label">Events</div>
          <div className="otel-kpi-val">{summary.events_total}</div>
        </div>
        <div className="otel-kpi">
          <div className="otel-kpi-label">Metrics</div>
          <div className="otel-kpi-val">{summary.metrics_total}</div>
        </div>
        <div className="otel-kpi">
          <div className="otel-kpi-label">Spans</div>
          <div className="otel-kpi-val">{summary.spans_total}</div>
        </div>
      </div>

      {/* 模型分布 */}
      {Object.keys(summary.models).length > 0 && (
        <div className="otel-models-row">
          {Object.entries(summary.models).map(([m, total]) => (
            <span key={m} className="otel-model-chip">
              <b>{m}</b>
              <span style={{ marginLeft: 6, opacity: 0.75 }}>{fmtNum(total)} tokens</span>
            </span>
          ))}
        </div>
      )}

      {/* ── 2. Prompt 分组的事件时间线 ── */}
      {promptGroups.length > 0 && (
        <details open className="otel-section">
          <summary className="otel-section-head">
            🧵 Prompt 时间线（按 prompt.id 关联，共 {promptGroups.length} 个 prompt）
          </summary>
          <div className="otel-prompt-list">
            {promptGroups.map((g, idx) => {
              const promptText = (g.user_prompt?.attributes?.prompt as string) || '';
              const promptLen = (g.user_prompt?.attributes?.prompt_length as number) || 0;
              return (
                <div key={g.prompt_id || idx} className="otel-prompt-card">
                  <div className="otel-prompt-head">
                    <span className="otel-prompt-idx">#{idx + 1}</span>
                    <span className="otel-prompt-id">{g.prompt_id.slice(0, 8) || 'orphan'}</span>
                    <span className="otel-prompt-stats">
                      {g.api_requests.length} api · {g.tool_results.length} tool
                      {g.others.length > 0 ? ` · ${g.others.length} 其他` : ''}
                    </span>
                    {g.user_prompt?.ts && (
                      <span className="otel-prompt-ts">{fmtTs(g.user_prompt.ts)}</span>
                    )}
                  </div>
                  {promptText && (
                    <div className="otel-prompt-body" title={`prompt_length=${promptLen}`}>
                      {promptText.length > 400 ? promptText.slice(0, 400) + '…' : promptText}
                    </div>
                  )}
                  {!promptText && g.user_prompt && (
                    <div className="otel-prompt-redacted">
                      [prompt 内容已隐藏：未启用 OTEL_LOG_USER_PROMPTS=1]
                    </div>
                  )}
                  {g.api_requests.length > 0 && (
                    <div className="otel-event-list">
                      {g.api_requests.map((e, i) => (
                        <div key={i} className="otel-event-row otel-event-api">
                          <span className="otel-event-tag">API</span>
                          <span className="otel-event-model">{e.attributes?.model || '?'}</span>
                          <span className="otel-event-tokens">
                            in:{fmtNum(e.attributes?.input_tokens)} ·
                            out:{fmtNum(e.attributes?.output_tokens)}
                            {e.attributes?.cache_read_tokens ? ` · cache:${fmtNum(e.attributes.cache_read_tokens)}` : ''}
                          </span>
                          <span className="otel-event-cost">
                            {e.attributes?.cost_usd ? `$${Number(e.attributes.cost_usd).toFixed(4)}` : ''}
                          </span>
                          <span className="otel-event-dur">
                            {e.attributes?.duration_ms ? fmtDur(Number(e.attributes.duration_ms)) : ''}
                          </span>
                          <span className="otel-event-ts">{fmtTs(e.ts)}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  {g.tool_results.length > 0 && (
                    <div className="otel-event-list">
                      {g.tool_results.map((e, i) => (
                        <div key={i} className={`otel-event-row otel-event-tool ${e.attributes?.success === 'false' ? 'otel-event-fail' : ''}`}>
                          <span className="otel-event-tag">TOOL</span>
                          <span className="otel-event-tool-name">{e.attributes?.tool_name || '?'}</span>
                          <span className="otel-event-tokens">
                            {e.attributes?.tool_result_size_bytes
                              ? `${fmtNum(e.attributes.tool_result_size_bytes)}B`
                              : ''}
                          </span>
                          <span className="otel-event-success">
                            {e.attributes?.success === 'true' ? '✓' :
                             e.attributes?.success === 'false' ? `✗ ${e.attributes?.error_type || ''}` : ''}
                          </span>
                          <span className="otel-event-dur">
                            {e.attributes?.duration_ms ? fmtDur(Number(e.attributes.duration_ms)) : ''}
                          </span>
                          <span className="otel-event-ts">{fmtTs(e.ts)}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </details>
      )}

      {/* ── v0.16.3: 🛠️ 工具调用统计（OTel 真值视野）──
          这块的总数 **可能远大于** 主 SpanTree 看到的工具数。原因：主 SpanTree
          的数据源是 transcript（只记录主 agent 的 tool_use blocks），但 OTel
          通道是进程级采样，会把每一次 Agent (subagent) **内部嵌套**的
          Read/Bash/Grep/Glob 也都记下来。差值 ≈ subagent 替主 agent 干的活。 */}
      {toolStats.total_calls > 0 && (
        <details open className="otel-section">
          <summary className="otel-section-head">
            🛠️ 工具调用统计（OTel 视野：{toolStats.total_calls} 次调用 ·
            {' '}{toolStats.unique_tools} 种工具 · 总耗时 {fmtDur(toolStats.total_duration_ms)} ·
            {' '}总返回 {fmtNum(toolStats.total_result_bytes)}B）
          </summary>
          <div className="otel-tool-table">
            <div className="otel-tool-row otel-tool-row-head">
              <span className="otel-tool-name">Tool</span>
              <span className="otel-tool-cnt">调用</span>
              <span className="otel-tool-succ">成功 / 失败</span>
              <span className="otel-tool-dur">总耗时</span>
              <span className="otel-tool-avg">平均</span>
              <span className="otel-tool-bytes">返回大小</span>
              <span className="otel-tool-errs">错误类型</span>
            </div>
            {toolStats.by_tool.map(t => {
              const avg = t.total > 0 ? t.total_duration_ms / t.total : 0;
              const errEntries = Array.from(t.err_types.entries());
              return (
                <div
                  key={t.tool_name}
                  className={`otel-tool-row${t.failure > 0 ? ' otel-tool-row-fail' : ''}`}
                >
                  <span className="otel-tool-name">{t.tool_name}</span>
                  <span className="otel-tool-cnt">{t.total}</span>
                  <span className="otel-tool-succ">
                    <span className="otel-tool-ok">✓ {t.success}</span>
                    {t.failure > 0 && (
                      <>
                        {' / '}
                        <span className="otel-tool-bad">✗ {t.failure}</span>
                      </>
                    )}
                  </span>
                  <span className="otel-tool-dur">{fmtDur(t.total_duration_ms)}</span>
                  <span className="otel-tool-avg">{fmtDur(avg)}</span>
                  <span className="otel-tool-bytes">{fmtNum(t.total_result_bytes)}B</span>
                  <span className="otel-tool-errs">
                    {errEntries.length === 0
                      ? <span className="otel-tool-mute">—</span>
                      : errEntries.map(([k, v]) => (
                          <code key={k} className="otel-tool-err-chip">{k}×{v}</code>
                        ))}
                  </span>
                </div>
              );
            })}
          </div>
          <div className="otel-tool-foot">
            数据来源：进程级 OTel <code>tool_decision</code> + <code>tool_result</code> events。
            <strong>含 Agent (subagent) 内部嵌套调用</strong>——所以这里的总数可能远大于
            主时间线（左侧 SpanTree 只能看到主 agent 在 transcript 里的工具，
            subagent 内部调了几十次 Read 也不会出现在主 transcript 里）。
          </div>
        </details>
      )}

      {/* ── v0.16.2: Hooks 概览 ──
          Claude Code 把 27 个 hook（SessionStart / PreToolUse / PostToolUse /
          UserPromptSubmit / Stop / SubagentStop / Notification / PreCompact /
          ...）的每次触发都通过 OTel 上报为一对 hook_execution_start/complete
          事件。这块按 hook_event 聚合：触发次数 / 总耗时 / 成功 / blocking
          / cancelled。这是用户问的"27 hooks 的真实输出"——上报通道是 OTel
          而非自己部署的 stream_hook，但内容更权威（Claude 进程亲口说的）。 */}
      {hookStats.total_executions > 0 && (
        <details open className="otel-section">
          <summary className="otel-section-head">
            🪝 Hooks 触发概览（{hookStats.total_executions} 次执行 ·
            {' '}{hookStats.by_event.length} 种 hook ·
            {' '}总耗时 {fmtDur(hookStats.total_duration_ms)}）
          </summary>
          <div className="otel-hook-table">
            <div className="otel-hook-row otel-hook-row-head">
              <span className="otel-hook-event">Hook Event</span>
              <span className="otel-hook-cnt">触发</span>
              <span className="otel-hook-dur">总耗时</span>
              <span className="otel-hook-avg">平均</span>
              <span className="otel-hook-status">成功 / 阻断 / 取消 / 错误</span>
              <span className="otel-hook-names">注册脚本</span>
            </div>
            {hookStats.by_event.map(h => {
              const avg = h.total > 0 ? h.total_duration_ms / h.total : 0;
              const hasIssue = h.num_blocking + h.num_cancelled + h.num_non_blocking_error > 0;
              return (
                <div
                  key={h.hook_event}
                  className={`otel-hook-row${hasIssue ? ' otel-hook-row-warn' : ''}`}
                >
                  <span className="otel-hook-event">{h.hook_event}</span>
                  <span className="otel-hook-cnt">{h.total}</span>
                  <span className="otel-hook-dur">{fmtDur(h.total_duration_ms)}</span>
                  <span className="otel-hook-avg">{fmtDur(avg)}</span>
                  <span className="otel-hook-status">
                    <span className="otel-hook-ok">✓ {h.num_success}</span>
                    {' / '}
                    <span className={h.num_blocking ? 'otel-hook-bad' : 'otel-hook-mute'}>
                      ⛔ {h.num_blocking}
                    </span>
                    {' / '}
                    <span className={h.num_cancelled ? 'otel-hook-bad' : 'otel-hook-mute'}>
                      ⊗ {h.num_cancelled}
                    </span>
                    {' / '}
                    <span className={h.num_non_blocking_error ? 'otel-hook-bad' : 'otel-hook-mute'}>
                      ⚠ {h.num_non_blocking_error}
                    </span>
                  </span>
                  <span className="otel-hook-names" title={Array.from(h.hook_names).join('\n')}>
                    {Array.from(h.hook_names).slice(0, 3).map(n => (
                      <code key={n}>{n}</code>
                    ))}
                    {h.hook_names.size > 3 && (
                      <span className="otel-hook-more">+{h.hook_names.size - 3}</span>
                    )}
                  </span>
                </div>
              );
            })}
          </div>
          <div className="otel-hook-foot">
            数据来源：Claude Code 进程亲自上报的 <code>hook_execution_complete</code> events
            （{hookStats.total_executions} 条）。每对 <code>start/complete</code> 描述一次
            hook 触发；行内统计来自其 attributes。
          </div>
        </details>
      )}

      {/* ── 3. Metrics 总览（按 metric.name 折叠）── */}
      {metrics.length > 0 && (
        <details className="otel-section">
          <summary className="otel-section-head">
            📊 Metrics（{metrics.length} 个数据点 · 共 {Object.keys(summary.metrics_by_name).length} 种 metric）
          </summary>
          <div className="otel-metric-list">
            {Object.entries(summary.metrics_by_name)
              .sort((a, b) => b[1] - a[1])
              .map(([name, count]) => {
                const sample = metrics.filter(m => m.metric === name).slice(0, 3);
                return (
                  <div key={name} className="otel-metric-card">
                    <div className="otel-metric-head">
                      <span className="otel-metric-name">{name}</span>
                      <span className="otel-metric-count">{count} 点</span>
                      {sample[0]?.unit && <span className="otel-metric-unit">{sample[0].unit}</span>}
                    </div>
                    <div className="otel-metric-samples">
                      {sample.map((m, i) => (
                        <span key={i} className="otel-metric-sample">
                          {typeof m.value === 'number' ? m.value :
                           typeof m.value === 'object' ? JSON.stringify(m.value) : String(m.value)}
                          {Object.keys(m.attributes).length > 0 && (
                            <span className="otel-metric-attrs">
                              {' '}({Object.entries(m.attributes).map(([k, v]) => `${k}=${v}`).join(', ')})
                            </span>
                          )}
                        </span>
                      ))}
                      {count > 3 && <span className="otel-metric-more">…+{count - 3}</span>}
                    </div>
                  </div>
                );
              })}
          </div>
        </details>
      )}

      {/* ── 4. Traces 树 ── */}
      {spanTree.length > 0 && (
        <details className="otel-section">
          <summary className="otel-section-head">
            🌲 Traces（beta · {summary.spans_total} 个 span ·{' '}
            {Object.entries(summary.spans_by_name).map(([k, v]) => `${k}×${v}`).join(' · ')}）
          </summary>
          <div className="otel-trace-tree">
            {spanTree.map(root => (
              <SpanTreeNode key={root.span_id} node={root} />
            ))}
          </div>
        </details>
      )}

      {/* 数据来源说明 */}
      <div className="otel-source-note">
        ℹ️ 本面板数据来自 Claude Code 进程的官方 OTel 上报（OTLP/HTTP/JSON），
        落盘于 <code>~/.claude/state/otel/{sessionId.slice(0, 8)}…/</code>。
        与 transcript / hooks 是三条独立通道。
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// v0.16.4 ClaudeToolStatsCompact —— 紧凑版工具调用统计组件，
// 给左侧 SpanTree 顶部嵌入用。它自己拉 /api/sessions/<sid>/otel，
// 渲染一行 KPI（总调用 / 工具种数 / 主 vs subagent 拆分）+ 一张按 tool_name
// 聚合的小表。这样用户在主流程视图上就能一眼看到"OTel 真值视野"看到了
// 多少次工具调用，而不需要切到 OTel 标签去看。
// ─────────────────────────────────────────────────────────────────
export function ClaudeToolStatsCompact({ sessionId, prefetched }: { sessionId: string; prefetched?: ClaudeOtelData | null }) {
  const [data, setData] = useState<ClaudeOtelData | null>(prefetched || null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (prefetched) {
      setData(prefetched);
      return;
    }
    if (!sessionId) return;
    let cancelled = false;
    setLoading(true);
    api.getClaudeOtel(sessionId).then(d => {
      if (!cancelled) setData(d);
    }).catch(() => {
      // 静默失败 —— 没有 OTel 数据时不显示这块即可
    }).finally(() => {
      if (!cancelled) setLoading(false);
    });
    return () => { cancelled = true; };
  }, [sessionId, prefetched]);

  if (loading || !data) return null;
  const stats = summarizeToolEvents(data.events || []);
  if (stats.total_calls === 0) return null;

  // 估算 subagent 嵌套贡献：Agent 工具 N 次 → 它们内部调用是总数减去
  // 主 agent 在 transcript 里能看到的工具数；这里给一个简单展示——总数
  // 与 Agent 调用次数比，让用户感知"嵌套占比"。
  const agentCalls = stats.by_tool.find(t => t.tool_name === 'Agent')?.total || 0;

  return (
    <details open className="cl-tool-compact">
      <summary className="cl-tool-compact-head">
        <span className="cl-tool-compact-title">🛠️ OTel 工具调用统计</span>
        <span className="cl-tool-compact-meta">
          {stats.total_calls} 次 · {stats.unique_tools} 种工具 ·
          总耗时 {fmtDur(stats.total_duration_ms)} ·
          返回 {fmtNum(stats.total_result_bytes)}B
          {agentCalls > 0 && (
            <span className="cl-tool-compact-subagent" title="本会话派发了多少次 Subagent (Agent) 工具——它们内部还会嵌套调用 Read/Bash/Grep 等工具，这些都被 OTel 捕获了">
              · 含 Subagent ×{agentCalls}
            </span>
          )}
        </span>
      </summary>
      <div className="cl-tool-compact-list">
        {stats.by_tool.map(t => {
          const fail = t.failure;
          return (
            <div key={t.tool_name} className={`cl-tool-compact-chip${fail > 0 ? ' cl-tool-compact-chip-fail' : ''}`}>
              <span className="cl-tool-compact-chip-name">{t.tool_name}</span>
              <span className="cl-tool-compact-chip-cnt">×{t.total}</span>
              {fail > 0 && <span className="cl-tool-compact-chip-fail-cnt">✗{fail}</span>}
            </div>
          );
        })}
      </div>
      <div className="cl-tool-compact-foot">
        左侧主时间线展示的是 transcript 里主 agent 的 tool_use blocks；这里是
        Claude 进程级 OTel 真值，<strong>包含 Agent (subagent) 内部嵌套调用</strong>，
        所以总数通常比左侧主线多很多。完整明细见右侧 OTel 标签。
      </div>
    </details>
  );
}
