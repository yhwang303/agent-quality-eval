/**
 * OtelPanel — OpenTelemetry GenAI 视图（client-side 合成）
 *
 * 与 DetailPanel 并列存在于右侧 sidebar，通过 Tab 切换。当后端开启了
 * `cot_otel_enricher` 时（默认），每个 session/turn/step 都会带上一组按 OTel
 * `gen_ai.*` 语义约定整理过的字段，这个面板就把它们以"OTel trace 视角"展示。
 *
 * 三档：
 *   - SessionCoT 选中 → 展示 service / trace_id / token & cost 总览 / eval / missing_signals
 *   - TurnCoT    选中 → 展示 root_span_id / finish_reasons / token / request_params / eval
 *   - ThoughtStep 选中 → 展示 span 链 / attributes / input_messages / output_messages /
 *                         retrieval_documents / token & cost
 *
 * 字段缺失（Cursor / transcript 暴露不了的）一律渲染为 "—"，并在 footer 提示
 * "data not available from current transcript"，符合 OTel 语义里
 * "attribute may be unavailable" 的处理风格。
 */
import { useMemo, useState } from 'react';
import type {
  SessionCoT,
  SessionOverview,
  ResponseReport,
  TurnCoT,
  ThoughtStep,
  OtelStepView,
  OtelTurnView,
  OtelSessionView,
  OtelResourceAttributes,
  OtelStructuredMessage,
  OtelRetrievalDocument,
  OtelTokenUsage,
  OtelHint,
  OtelModelSource,
  OtelStepKind,
  OtelActualTokenUsage,
  OtelClientRuntime,
} from '../types';
import { api, saveDownloadedFile } from '../hooks/api';
// v0.16.1: Claude session 直接渲染真实 OTel 面板（来自 Claude 进程主动 OTLP 上报），
// 不走 cot_otel_enricher 的合成路径——那是为 Cursor 设计的 transcript-derived 视图。
import { ClaudeOtelPanel } from './ClaudeOtelPanel';

export type OtelSelectedNode =
  | { kind: 'session'; cot: SessionCoT; session: SessionOverview | null; report: ResponseReport | null }
  | { kind: 'turn'; turn: TurnCoT; cot: SessionCoT }
  | { kind: 'step'; step: ThoughtStep; turn: TurnCoT };

interface Props {
  node: OtelSelectedNode | null;
  // v0.14.8: 当前正在查看的 session id，用来生成 OTLP/JSON 协议下载文件
  // 没传值时按钮置灰（理论上 App.tsx 总会传，但有兜底更安全）
  sessionId?: string | null;
  // v0.16.1: 当前 session cot（用于判断 agent_type='claude' 走原生 OTel 路径）
  cot?: SessionCoT | null;
}

// ─── 公共 helpers ────────────────────────────────────────────

function fmtNum(n: number | null | undefined): string {
  if (n == null) return '—';
  if (Number.isFinite(n)) return n.toLocaleString();
  return '—';
}

function fmtCost(c: number | null | undefined): string {
  if (c == null) return '—';
  if (c < 1e-4) return `$${c.toFixed(6)}`;
  if (c < 1) return `$${c.toFixed(4)}`;
  return `$${c.toFixed(2)}`;
}

/** 把 cost_reason 翻译成展示用的「为啥 cost = —」文案 */
function costReasonLabel(reason?: string): string {
  switch (reason) {
    case 'unknown_model':
      return 'model unknown · 在 .env 设 COT_DEFAULT_MODEL 即可计费';
    case 'no_pricing':
      return 'no pricing entry · 给 _PRICING_USD_PER_1K 加条目';
    case 'non_llm_step':
      return 'N/A · 非 LLM 调用（host runtime / 用户输入 / 合成事件）';
    case 'ok':
    case undefined:
    case null:
      return '';
    default:
      return reason as string;
  }
}

function modelSourceLabel(src?: OtelModelSource | string): { text: string; tone: 'good' | 'info' | 'warn' | 'neutral' } {
  switch (src) {
    case 'events':
      return { text: 'auto · cot-stream', tone: 'good' };
    case 'env':
      return { text: 'from .env', tone: 'good' };
    case 'transcript':
      return { text: 'from transcript', tone: 'good' };
    case 'host':
      return { text: 'host runtime', tone: 'info' };
    case 'client':
      return { text: 'client input', tone: 'info' };
    case 'synthetic':
      return { text: 'synthetic event', tone: 'info' };
    case 'unknown':
      return { text: 'unknown · 需配置', tone: 'warn' };
    default:
      return { text: src ? String(src) : 'unknown', tone: 'neutral' };
  }
}

function stepKindLabel(k?: OtelStepKind | string): { text: string; tone: 'good' | 'info' | 'warn' | 'neutral' } {
  switch (k) {
    case 'llm_call':
      return { text: 'LLM call', tone: 'good' };
    case 'host_tool':
      return { text: 'host tool', tone: 'info' };
    case 'user_input':
      return { text: 'user input', tone: 'info' };
    case 'agent_event':
      return { text: 'agent event', tone: 'neutral' };
    default:
      return { text: k ? String(k) : '—', tone: 'neutral' };
  }
}

function shortId(id: string | undefined | null, len = 8): string {
  if (!id) return '—';
  if (id.length <= len) return id;
  return `${id.slice(0, len)}…${id.slice(-4)}`;
}

function CopyableId({ id, len = 12 }: { id: string | null | undefined; len?: number }) {
  const [copied, setCopied] = useState(false);
  if (!id) return <span className="otel-id otel-muted">—</span>;
  const display = id.length > len ? `${id.slice(0, len)}…${id.slice(-4)}` : id;
  return (
    <span
      className="otel-id"
      title={`${id}${copied ? ' (copied!)' : ' — click to copy'}`}
      onClick={() => {
        navigator.clipboard?.writeText(id).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        });
      }}
    >
      {display}
      {copied && <span className="otel-id-copied">✓</span>}
    </span>
  );
}

const saveJsonBlob = (blob: Blob, filename: string) =>
  saveDownloadedFile({ blob, filename });

function KV({ k, v, mono = true }: { k: string; v: React.ReactNode; mono?: boolean }) {
  return (
    <div className="otel-kv">
      <div className="otel-kv-k">{k}</div>
      <div className={`otel-kv-v ${mono ? 'otel-mono' : ''}`}>{v}</div>
    </div>
  );
}

function Badge({
  text,
  tone = 'neutral',
}: {
  text: string;
  tone?: 'neutral' | 'good' | 'warn' | 'bad' | 'info';
}) {
  return <span className={`otel-badge otel-badge-${tone}`}>{text}</span>;
}

/** cost 单行：值 + 原因 */
function CostLine({ tu }: { tu?: OtelTokenUsage | null }) {
  if (!tu) return <>—</>;
  const reasonText = costReasonLabel(tu.cost_reason);
  if (tu.cost_usd != null) {
    return <span className="otel-mono">{fmtCost(tu.cost_usd)}</span>;
  }
  // cost = null：显示 — + 原因
  const tone =
    tu.cost_reason === 'non_llm_step' ? 'info' :
    tu.cost_reason === 'unknown_model' ? 'warn' :
    'neutral';
  return (
    <span>
      <span className="otel-mono otel-muted">—</span>
      {reasonText && (
        <>
          {' '}<Badge text={reasonText} tone={tone as any} />
        </>
      )}
    </span>
  );
}

function CardHeader({ title, sub }: { title: string; sub?: string }) {
  return (
    <div className="otel-card-header">
      <span className="otel-card-title">{title}</span>
      {sub && <span className="otel-card-sub">{sub}</span>}
    </div>
  );
}

function Card({ title, sub, children }: { title: string; sub?: string; children: React.ReactNode }) {
  return (
    <section className="otel-card">
      <CardHeader title={title} sub={sub} />
      <div className="otel-card-body">{children}</div>
    </section>
  );
}

// ─── Token & cost ──────────────────────────────────────────

function TokenCostCard({
  tu,
  model,
  provider,
  modelSource,
  modelsSeen,
}: {
  tu?: OtelTokenUsage | null;
  model?: string;
  provider?: string;
  modelSource?: OtelModelSource | string;
  modelsSeen?: string[];
}) {
  const src = modelSourceLabel(modelSource);
  // 多模型场景：把所有出现过的模型都列出来，dominant 高亮
  const seen = (modelsSeen || []).filter(Boolean);
  const isMulti = seen.length > 1;
  return (
    <Card
      title="Token & Cost"
      sub={`gen_ai.usage · ${tu?.is_estimate ? 'estimate (char/4)' : 'real'} · ${
        tu?.model_key || model || 'unknown model'
      }`}
    >
      <KV k="input_tokens" v={fmtNum(tu?.input_tokens)} />
      <KV k="output_tokens" v={fmtNum(tu?.output_tokens)} />
      <KV k="total_tokens" v={fmtNum(tu?.total_tokens)} />
      <KV k="cost_usd" v={<CostLine tu={tu} />} />
      <KV k="provider" v={provider || '—'} />
      <KV
        k={isMulti ? 'models' : 'model'}
        v={
          isMulti ? (
            <span className="otel-models-list">
              {seen.map(m => (
                <span
                  key={m}
                  className={`otel-model-chip ${m === model ? 'otel-model-chip-dominant' : ''}`}
                  title={m === model ? '本 session 的 dominant model（出现次数最多）' : '本 session 内出现过该 model'}
                >
                  {m}
                </span>
              ))}
              {modelSource && (
                <Badge text={src.text} tone={src.tone as any} />
              )}
            </span>
          ) : (
            <span>
              <span className="otel-mono">{model || '—'}</span>
              {modelSource && (
                <>{' '}<Badge text={src.text} tone={src.tone as any} /></>
              )}
            </span>
          )
        }
      />
      {isMulti && (
        <div className="otel-note otel-note-info">
          ℹ 本 session 内检测到 <b>{seen.length}</b> 个不同 model：
          {seen.map((m, i) => (
            <span key={m}>
              {i > 0 && '、'}
              <code>{m}</code>
            </span>
          ))}
          。dominant 用于 session 级 cost / token 计算，每个 turn 会按
          <code> renderer.log </code> 时间轴标记当时实际生效的模型。
        </div>
      )}
      {modelSource === 'unknown' && (
        <div className="otel-note">
          ⚠ 未识别到 model。<b>推荐</b>挂上 <code>~/.cursor/hooks/cot-stream.js</code>（实时写入 events.jsonl）；
          GPT-5.x / Codex / GLM / Hunyuan 等非 Claude 模型 fallback 自动扫
          <code> Cursor renderer.log </code>。还都识别不出，可以在
          <code> cot-extractor/.env </code>加 <code>COT_DEFAULT_MODEL=...</code> 兜底。
        </div>
      )}
      {tu?.cost_reason === 'no_pricing' && (
        <div className="otel-note">
          ℹ model 已识别但 pricing 表里没有这个条目，可在
          <code> cot_otel_enricher._PRICING_USD_PER_1K </code>新增即可计费。
        </div>
      )}
    </Card>
  );
}

// ─── Request params (大多 unavailable) ─────────────────────

function RequestParamsCard({
  params,
}: {
  params?: {
    temperature: number | null;
    top_p: number | null;
    max_tokens: number | null;
    seed: number | null;
    stop_sequences: string[] | null;
    _note?: string;
  };
}) {
  if (!params) return null;
  const isAllNull =
    params.temperature == null &&
    params.top_p == null &&
    params.max_tokens == null &&
    params.seed == null &&
    (params.stop_sequences == null || (Array.isArray(params.stop_sequences) && params.stop_sequences.length === 0));
  return (
    <Card title="Request Params" sub="gen_ai.request.*">
      <KV k="temperature" v={params.temperature ?? <span className="otel-muted">unavailable</span>} />
      <KV k="top_p" v={params.top_p ?? <span className="otel-muted">unavailable</span>} />
      <KV k="max_tokens" v={params.max_tokens ?? <span className="otel-muted">unavailable</span>} />
      <KV k="seed" v={params.seed ?? <span className="otel-muted">unavailable</span>} />
      <KV
        k="stop_sequences"
        v={
          params.stop_sequences && params.stop_sequences.length
            ? params.stop_sequences.join(', ')
            : <span className="otel-muted">unavailable</span>
        }
      />
      {isAllNull && (
        <div className="otel-note">
          ⚠ Cursor / Claude Code transcript 不暴露 LLM 调用参数，本视图保留 OTel
          字段位但全部记为 unavailable —— 这是当前 transcript 已知盲点。
        </div>
      )}
      {params._note && <div className="otel-note">{params._note}</div>}
    </Card>
  );
}

// ─── Eval ──────────────────────────────────────────────────

function EvalCard({ eval_, title = 'Eval' }: { eval_?: any; title?: string }) {
  if (!eval_) return null;
  const tone =
    eval_.label === 'ok' ? 'good' : eval_.label === 'warn' ? 'warn' : eval_.label === 'fail' ? 'bad' : 'neutral';
  return (
    <Card title={title} sub={eval_.metric_name}>
      <KV
        k="score"
        v={
          <span>
            {eval_.score == null ? '—' : eval_.score}
            {eval_.label && <> · <Badge text={eval_.label} tone={tone as any} /></>}
          </span>
        }
      />
      {eval_.summary && <KV k="summary" v={eval_.summary} mono={false} />}
      {eval_.scores && (
        <div className="otel-subblock">
          <div className="otel-subblock-title">scores</div>
          <table className="otel-table">
            <tbody>
              {Object.entries(eval_.scores as Record<string, any>).map(([k, v]) => (
                <tr key={k}>
                  <td className="otel-mono">{k}</td>
                  <td className="otel-mono">{String(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {eval_.checked_at && <KV k="checked_at" v={eval_.checked_at} />}
    </Card>
  );
}

// ─── Resource attributes ──────────────────────────────────

function ResourceCard({ ra }: { ra?: OtelResourceAttributes | null }) {
  if (!ra) return null;
  const entries = Object.entries(ra);
  return (
    <Card title="Resource Attributes" sub="otel.resource.*">
      <table className="otel-table">
        <tbody>
          {entries.map(([k, v]) => (
            <tr key={k}>
              <td className="otel-mono otel-attr-key">{k}</td>
              <td className="otel-mono">{String(v ?? '—')}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

// ─── Attributes (gen_ai.*) ────────────────────────────────

function AttributesCard({ attrs }: { attrs?: Record<string, any> }) {
  if (!attrs) return null;
  const entries = Object.entries(attrs);
  if (!entries.length) return null;
  return (
    <Card title="Span Attributes" sub="gen_ai.* / tool.*">
      <table className="otel-table">
        <tbody>
          {entries.map(([k, v]) => (
            <tr key={k}>
              <td className="otel-mono otel-attr-key">{k}</td>
              <td className="otel-mono">
                {Array.isArray(v) ? v.join(', ') : (v == null ? '—' : String(v))}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

// ─── Structured messages ──────────────────────────────────

function MessagesCard({
  title,
  messages,
}: {
  title: string;
  messages?: OtelStructuredMessage[];
}) {
  if (!messages || !messages.length) return null;
  return (
    <Card title={title} sub={`${messages.length} message${messages.length > 1 ? 's' : ''} · OTel structured`}>
      {messages.map((m, i) => (
        <div key={i} className="otel-msg">
          <div className="otel-msg-head">
            <Badge text={m.role} tone={m.role === 'user' ? 'info' : m.role === 'assistant' ? 'good' : 'neutral'} />
            {m.finish_reason && <Badge text={`finish: ${m.finish_reason}`} tone="neutral" />}
            <span className="otel-muted otel-mono">parts={m.parts.length}</span>
          </div>
          {m.parts.map((p, j) => (
            <div key={j} className="otel-msg-part">
              <span className="otel-mono otel-muted">[{p.type}]</span>{' '}
              <span className="otel-msg-content">{p.content}</span>
            </div>
          ))}
        </div>
      ))}
    </Card>
  );
}

// ─── Retrieval documents ──────────────────────────────────

function RetrievalCard({ docs }: { docs?: OtelRetrievalDocument[] }) {
  if (!docs || !docs.length) return null;
  return (
    <Card title="Retrieval Documents" sub={`OpenInference RetrievalSpan · ${docs.length} doc${docs.length > 1 ? 's' : ''}`}>
      {docs.map((d, i) => (
        <div key={i} className="otel-doc">
          <div className="otel-doc-head">
            <Badge text={d.id} tone="info" />
            {d.score != null && <Badge text={`score=${d.score}`} tone="good" />}
            {Object.keys(d.metadata || {}).length > 0 && (
              <span className="otel-muted otel-mono">
                meta: {Object.keys(d.metadata).slice(0, 3).join(', ')}
              </span>
            )}
          </div>
          <div className="otel-doc-content">{d.content}</div>
        </div>
      ))}
    </Card>
  );
}

// ─── v0.11.2: Client Runtime（自动检测的 Cursor hook 上下文） ───

function ClientRuntimeCard({ rt }: { rt?: OtelClientRuntime | null }) {
  if (!rt) return null;
  const dist = rt.model_distribution || {};
  const distEntries = Object.entries(dist);
  return (
    <Card title="Client Runtime" sub="cot-stream events.jsonl · 自动检测，零硬编码">
      <KV
        k="cursor.version"
        v={rt.cursor_version ? <span className="otel-mono">{rt.cursor_version}</span> : <span className="otel-muted">unavailable</span>}
      />
      <KV
        k="user.email"
        v={rt.user_email ? <span className="otel-mono">{rt.user_email}</span> : <span className="otel-muted">unavailable</span>}
      />
      <KV
        k="events.count"
        v={
          rt.events_count != null
            ? <span><span className="otel-mono">{fmtNum(rt.events_count)}</span>{' '}<Badge text="hook events" tone="info" /></span>
            : '—'
        }
      />
      {distEntries.length > 0 && (
        <KV
          k="model_distribution"
          v={
            <span>
              {distEntries.map(([m, c]) => (
                <Badge key={m} text={`${m} × ${c}`} tone="good" />
              ))}
            </span>
          }
        />
      )}
      {rt.events_path && (
        <div className="otel-note otel-note-info">
          ℹ 数据源：<code className="otel-mono">{rt.events_path}</code>
          <br/>该文件由 <code>~/.cursor/hooks/cot-stream.js</code> 在 Cursor 每次 hook（shell / read / file edit / agent thought / agent response …）触发时实时写入，包含真实 <code>model</code> / <code>generation_id</code> / cache-aware token 计数。
        </div>
      )}
    </Card>
  );
}

function ActualUsageCard({ atu }: { atu?: OtelActualTokenUsage | null }) {
  if (!atu) return null;
  const totalCost = atu.full_price_cost_usd;
  const totalIn = atu.input_tokens || 0;
  const cacheRead = atu.cache_read_tokens || 0;
  const cacheWrite = atu.cache_write_tokens || 0;
  const nonCache = Math.max(0, totalIn - cacheRead - cacheWrite);
  const cacheHitRate = totalIn > 0 ? (cacheRead / totalIn) * 100 : 0;
  const bd = atu.cost_breakdown || {};
  return (
    <Card
      title="Actual Token & Cost"
      sub={`${atu.source} · cache-aware · ${atu.agent_response_count} agent response${atu.agent_response_count > 1 ? 's' : ''}`}
    >
      <KV
        k="input_tokens"
        v={
          <span>
            <span className="otel-mono">{fmtNum(totalIn)}</span>{' '}
            <Badge text="real" tone="good" />{' '}
            <span className="otel-muted otel-mono">
              ( {fmtNum(nonCache)} non-cache + {fmtNum(cacheWrite)} cache_write + {fmtNum(cacheRead)} cache_read )
            </span>
          </span>
        }
      />
      <KV k="output_tokens" v={<span><span className="otel-mono">{fmtNum(atu.output_tokens)}</span>{' '}<Badge text="real" tone="good" /></span>} />
      <KV
        k="cache_hit_rate"
        v={
          totalIn > 0 ? (
            <span>
              <span className="otel-mono">{cacheHitRate.toFixed(1)}%</span>{' '}
              <Badge
                text={cacheHitRate > 80 ? 'huge savings' : cacheHitRate > 50 ? 'good' : cacheHitRate > 20 ? 'fair' : 'low'}
                tone={cacheHitRate > 50 ? 'good' : cacheHitRate > 20 ? 'info' : 'neutral'}
              />
            </span>
          ) : '—'
        }
      />
      <KV
        k="cost_usd"
        v={
          atu.cost_usd != null ? (
            <span>
              <span className="otel-mono">{fmtCost(atu.cost_usd)}</span>{' '}
              <Badge text="cache-aware" tone="good" />
            </span>
          ) : (
            <span><span className="otel-mono otel-muted">—</span>{' '}<Badge text={costReasonLabel(atu.cost_reason)} tone="warn" /></span>
          )
        }
      />
      {Object.keys(bd).length > 0 && (
        <div className="otel-subblock">
          <div className="otel-subblock-title">cost breakdown</div>
          <table className="otel-table">
            <tbody>
              {bd.non_cache_input_usd != null && (
                <tr>
                  <td className="otel-mono">non_cache_input</td>
                  <td className="otel-mono">{fmtCost(bd.non_cache_input_usd)}</td>
                </tr>
              )}
              {bd.cache_write_usd != null && (
                <tr>
                  <td className="otel-mono">cache_write (1.25x)</td>
                  <td className="otel-mono">{fmtCost(bd.cache_write_usd)}</td>
                </tr>
              )}
              {bd.cache_read_usd != null && (
                <tr>
                  <td className="otel-mono">cache_read (0.10x)</td>
                  <td className="otel-mono">{fmtCost(bd.cache_read_usd)}</td>
                </tr>
              )}
              {bd.output_usd != null && (
                <tr>
                  <td className="otel-mono">output</td>
                  <td className="otel-mono">{fmtCost(bd.output_usd)}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
      {totalCost != null && atu.cost_usd != null && totalCost > atu.cost_usd && (
        <div className="otel-note otel-note-info">
          ℹ Anthropic prompt cache 帮你省了 <b>{fmtCost(totalCost - atu.cost_usd)}</b>
          ({(((totalCost - atu.cost_usd) / totalCost) * 100).toFixed(1)}%) ——
          全价应为 {fmtCost(totalCost)}，缓存折扣后 {fmtCost(atu.cost_usd)}。
        </div>
      )}
    </Card>
  );
}

// ─── Hints (修复指引) ──────────────────────────────────────

function HintsCard({ hints }: { hints?: OtelHint[] | null }) {
  if (!hints || !hints.length) return null;
  return (
    <Card title="Fix-up Hints" sub="如何把 unknown / 缺失字段补齐">
      <ul className="otel-hints">
        {hints.map((h, i) => (
          <li key={i} className={`otel-hint otel-hint-${h.level}`}>
            <Badge
              text={h.level}
              tone={h.level === 'warn' ? 'warn' : h.level === 'error' ? 'bad' : 'info'}
            />
            <span className="otel-mono otel-hint-code">{h.code}</span>
            <span className="otel-hint-msg">{h.message}</span>
          </li>
        ))}
      </ul>
    </Card>
  );
}

// ─── Missing signals ──────────────────────────────────────

function MissingSignalsCard({ items }: { items?: string[] }) {
  if (!items || !items.length) return null;
  return (
    <Card title="Known Gaps" sub="OTel signals 当前 transcript 拿不到">
      <ul className="otel-missing">
        {items.map((s, i) => (
          <li key={i}>
            <span className="otel-mono">{s}</span>
          </li>
        ))}
      </ul>
    </Card>
  );
}

// ─── Empty state ──────────────────────────────────────────

function Empty({ msg }: { msg: string }) {
  return (
    <div className="otel-empty">
      <div className="otel-empty-icon">⊘</div>
      <div className="otel-empty-msg">{msg}</div>
    </div>
  );
}

// ─── Session view ─────────────────────────────────────────

function SessionView({
  cot,
  session,
  report,
}: {
  cot: SessionCoT;
  session: SessionOverview | null;
  report: ResponseReport | null;
}) {
  const ov: OtelSessionView | null | undefined = cot.otel_view;
  const ra = cot.resource_attributes;
  if (!ov) return <Empty msg="此 session 暂无 OTel 视图（请重新提取 CoT）" />;
  const src = modelSourceLabel(ov.model_source);
  return (
    <>
      <HintsCard hints={ov.hints} />

      <Card
        title="Session OTel"
        sub={`${ov.schema} · gen_ai.conversation.id = ${shortId(ov.conversation_id, 12)}`}
      >
        <KV k="trace_id" v={<CopyableId id={ov.trace_id} len={16} />} />
        <KV k="root_span_id" v={<CopyableId id={ov.root_span_id} len={12} />} />
        <KV k="service" v={`${ov.service.name} · ${ov.service.version}`} />
        <KV k="agent.name" v={ov.agent_name || '—'} />
        <KV
          k="model"
          v={
            <span>
              <span className="otel-mono">{ov.model}</span>
              {ov.model_source && <>{' '}<Badge text={src.text} tone={src.tone as any} /></>}
            </span>
          }
        />
        <KV k="provider" v={ov.provider} />
        <KV k="generated_at" v={new Date(ov.generated_at_ms).toISOString()} />
      </Card>

      {/* v0.11.2：自动检测的 Cursor 客户端运行时（cot-stream hook） */}
      <ClientRuntimeCard rt={ov.client_runtime} />

      {/* v0.11.2：来自 Cursor hook 的真实 cache-aware token + cost */}
      <ActualUsageCard atu={ov.actual_token_usage} />

      <Card title="Totals" sub="session-level rollups · 估算口径（char/4 fallback）">
        <KV k="turns" v={fmtNum(ov.totals.turns)} />
        <KV k="steps" v={fmtNum(ov.totals.steps)} />
        <KV k="tool_calls" v={fmtNum(ov.totals.tool_calls)} />
        <KV k="input_tokens" v={fmtNum(ov.totals.input_tokens)} />
        <KV k="output_tokens" v={fmtNum(ov.totals.output_tokens)} />
        <KV k="cost_usd" v={fmtCost(ov.totals.cost_usd)} />
        {ov.actual_token_usage && (
          <div className="otel-note otel-note-info">
            ℹ 上面的数字是基于 transcript 的字符级估算；真实 cache-aware 数值见
            <b> Actual Token & Cost</b> 卡片（来自 Cursor hook events.jsonl）。
          </div>
        )}
      </Card>

      <TokenCostCard
        tu={ov.token_usage}
        model={ov.model}
        provider={ov.provider}
        modelSource={ov.model_source}
        modelsSeen={ov.models_seen}
      />

      <RequestParamsCard params={ov.request_params} />

      <EvalCard
        eval_={
          ov.eval ||
          (session?.response_score != null
            ? {
                metric_name: 'response_score (legacy)',
                score: session.response_score,
                label:
                  session.response_score >= 0.7
                    ? 'ok'
                    : session.response_score >= 0.4
                    ? 'warn'
                    : 'fail',
                summary: report?.summary || '',
              }
            : null)
        }
        title="Eval (response-verifier)"
      />

      <ResourceCard ra={ra} />

      <MissingSignalsCard items={ov.missing_signals} />
    </>
  );
}

// ─── Turn view ────────────────────────────────────────────

function TurnView({ turn }: { turn: TurnCoT }) {
  const ov: OtelTurnView | null | undefined = turn.otel;
  if (!ov) return <Empty msg="此 turn 暂无 OTel 视图（请重新提取 CoT）" />;
  return (
    <>
      <Card title={`Turn ${turn.turn_index} OTel`} sub={`gen_ai.operation.name = ${ov.operation_name}`}>
        <KV k="trace_id" v={<CopyableId id={ov.trace_id} len={16} />} />
        <KV k="span_id" v={<CopyableId id={ov.span_id} len={12} />} />
        <KV k="parent_span_id" v={<CopyableId id={ov.parent_span_id} len={12} />} />
        <KV k="agent.name" v={ov.agent_name} />
        <KV k="conversation.id" v={shortId(ov.conversation_id, 12)} />
        <KV
          k="finish_reasons"
          v={ov.finish_reasons?.length ? ov.finish_reasons.map((r) => <Badge key={r} text={r} tone="info" />) : '—'}
        />
        <KV k="response.id" v={ov.response.id ?? <span className="otel-muted">unavailable</span>} />
      </Card>

      <TokenCostCard
        tu={ov.token_usage}
        model={ov.model}
        provider={ov.provider}
        modelSource={ov.model_source}
        modelsSeen={ov.models_seen}
      />

      <RequestParamsCard params={ov.request_params} />

      {turn.eval && <EvalCard eval_={turn.eval} title="Turn Eval" />}
    </>
  );
}

// ─── Step view ────────────────────────────────────────────

function StepView({ step, turn }: { step: ThoughtStep; turn: TurnCoT }) {
  const ov: OtelStepView | null | undefined = step.otel;
  if (!ov) return <Empty msg="此 step 暂无 OTel 视图（请重新提取 CoT）" />;
  const skind = stepKindLabel(ov.step_kind);
  const msrc = modelSourceLabel(ov.model_source);
  return (
    <>
      <Card
        title={`Step ${step.step_index} OTel`}
        sub={`${ov.operation_name} · span.kind=${ov.kind} · ${step.step_type}`}
      >
        <KV
          k="step.kind"
          v={
            <span>
              <Badge text={skind.text} tone={skind.tone as any} />{' '}
              <span className="otel-muted otel-mono">({ov.step_kind || '—'})</span>
            </span>
          }
        />
        <KV k="trace_id" v={<CopyableId id={ov.trace_id} len={16} />} />
        <KV k="span_id" v={<CopyableId id={ov.span_id} len={12} />} />
        <KV k="parent_span_id" v={<CopyableId id={ov.parent_span_id} len={12} />} />
        <KV
          k="model"
          v={
            <span>
              <span className="otel-mono">{ov.model}</span>
              {ov.model_source && <>{' '}<Badge text={msrc.text} tone={msrc.tone as any} /></>}
            </span>
          }
        />
        <KV k="provider" v={ov.provider} />
        <KV
          k="finish_reasons"
          v={ov.finish_reasons?.length ? ov.finish_reasons.map((r) => <Badge key={r} text={r} tone="info" />) : '—'}
        />
        {step.tool_name && <KV k="tool.name" v={step.tool_name} />}
        {step.tool_use_id && <KV k="tool.call.id" v={shortId(step.tool_use_id, 14)} />}
      </Card>

      {ov.step_kind === 'host_tool' && (
        <div className="otel-note otel-note-info">
          ℹ 这是一个 <b>host runtime</b> 工具执行（Cursor 调本地工具，不经过 LLM）。
          按 OTel 规范，<code>gen_ai.request.model</code> 在这类 span 上不必填，
          所以 <code>model</code> 标为 <code>host:cursor</code>，<code>cost_usd</code> 标 <i>N/A</i>。
        </div>
      )}

      <TokenCostCard
        tu={ov.token_usage}
        model={ov.model}
        provider={ov.provider}
        modelSource={ov.model_source}
        modelsSeen={ov.models_seen}
      />

      <MessagesCard title="Input Messages" messages={ov.input_messages} />
      <MessagesCard title="Output Messages" messages={ov.output_messages} />

      <RetrievalCard docs={ov.retrieval_documents} />

      <AttributesCard attrs={ov.attributes} />

      <Card title="Turn Context" sub="所属 turn 的 root span">
        <KV k="turn_index" v={String(turn.turn_index)} />
        <KV k="turn.span_id" v={<CopyableId id={turn.otel?.span_id || null} len={12} />} />
      </Card>
    </>
  );
}

// ─── Top-level component ──────────────────────────────────

// v0.14.8: 一键导出整个 session 为 OTLP/JSON 协议文件 ──────────────────
//
// 设计原则：
//   - 这是 OTel 协议级别的"标准产物"，不是"后端推送"。后者是 OtlpExportDialog
//     干的活。这里就是简单 GET 后端，拿 Blob 触发浏览器下载。
//   - 只在 OTel 视图里出现。详情面板的"导出推送"在 DetailPanel 里挂了。
//   - 文件名后端会根据 session_id + trace_id 算好；前端只负责消费。
//   - 失败时给一个 ErrorState 不破坏右侧主面板（用户能继续看其它 step）。
// v0.16.1: Claude session 专用的下载按钮——把 Claude 进程主动上报的真实 OTel
// 数据（events/metrics/spans/summary 合并版）打包成单 JSON 触发浏览器下载。
//
// 这跟下面 ExportOtlpJsonButton 干的不是一回事：那个是把 cot.json 用 SDK
// 重放/合成 OTLP/JSON 协议产物（适合 Cursor，因 Cursor 没有原生 OTel）。
// Claude 这条路径直接拿 backend 已落盘的真值，不需要重放。
function NativeOtelDownloadButton({ sessionId, provider }: { sessionId: string; provider: 'claude' | 'codex' }) {
  const [busy, setBusy] = useState(false);
  const [okMsg, setOkMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const label = provider === 'codex' ? 'Codex' : 'Claude';

  const onClick = async () => {
    if (busy) return;
    setBusy(true); setErr(null); setOkMsg(null);
    try {
      const data = await api.getClaudeOtel(sessionId);
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const mode = await saveJsonBlob(blob, `${provider}-otel-${sessionId.slice(0, 8)}.json`);
      const sizeKb = (blob.size / 1024).toFixed(1);
      const prefix = mode === 'picked' ? '已保存' : '已下载到浏览器默认下载目录';
      setOkMsg(`${prefix} · ${data.summary.events_total + data.summary.metrics_total + data.summary.spans_total} 条 · ${sizeKb} KB`);
      setTimeout(() => setOkMsg(null), 4000);
    } catch (e: any) {
      setErr(e?.message || String(e));
      setTimeout(() => setErr(null), 6000);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="otel-export-row">
      <button
        className="otel-export-btn"
        disabled={busy}
        onClick={onClick}
        title={`下载 ${label} 进程亲自上报的完整 OTel 数据（events + metrics + spans + summary 合并 JSON）`}
      >
        {busy ? '导出中…' : '📥 下载原始 OTel 数据'}
      </button>
      {okMsg && <span className="otel-export-ok">{okMsg}</span>}
      {err && <span className="otel-export-err" title={err}>下载失败：{err.slice(0, 80)}</span>}
    </div>
  );
}

function ExportOtlpJsonButton({ sessionId }: { sessionId: string | null }) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [okMsg, setOkMsg] = useState<string | null>(null);

  const onClick = async () => {
    if (!sessionId || busy) return;
    setBusy(true);
    setErr(null);
    setOkMsg(null);
    try {
      const r = await api.downloadSessionOtlpJson(sessionId);
      const mode = await saveJsonBlob(r.blob, r.filename || `otlp-${sessionId.slice(0, 8)}.json`);
      const sizeKb = (r.blob.size / 1024).toFixed(1);
      const prefix = mode === 'picked' ? '已保存' : '已下载到浏览器默认下载目录';
      setOkMsg(`${prefix} · ${r.span_count} spans · ${sizeKb} KB`);
      setTimeout(() => setOkMsg(null), 4000);
    } catch (e: any) {
      setErr(e?.message || String(e));
      setTimeout(() => setErr(null), 6000);
    } finally {
      setBusy(false);
    }
  };

  const disabled = !sessionId || busy;
  return (
    <div className="otel-export-row">
      <button
        className="otel-export-btn"
        disabled={disabled}
        onClick={onClick}
        title={
          disabled && !sessionId
            ? '先选中一个 session'
            : 'Download OTLP/JSON · OpenTelemetry proto 协议格式，可喂给 Phoenix / Jaeger / SigNoz / otel-cli 任意兼容工具'
        }
      >
        {busy ? '导出中…' : '📥 导出 OTLP/JSON'}
      </button>
      {okMsg && <span className="otel-export-ok">{okMsg}</span>}
      {err && <span className="otel-export-err" title={err}>导出失败：{err.slice(0, 80)}</span>}
    </div>
  );
}

export default function OtelPanel({ node, sessionId, cot }: Props) {
  // ─── v0.16.1: Claude session 走原生 OTel 真值通道（不合成）──────────
  //
  // 凭据来源优先级：node 上的 cot > 顶层 cot prop。这样不管用户选 session/turn/step
  // 哪一级节点，都能解析到 agent_type，避免 Claude session 选 step 时 step.otel
  // 缺失导致原合成路径里 ov.* 字段 access 抛 TypeError → 整个 panel 黑屏。
  //
  // 对 Claude session：OTel 数据本来就是 session 级（Claude 进程按 session.id 推），
  // 不需要按 step 切片，所以无论选哪一层都展示同一份完整 OTel 数据。
  const resolvedCot: SessionCoT | null | undefined =
    cot ||
    (node?.kind === 'session' ? node.cot : null) ||
    (node?.kind === 'turn' ? node.cot : null) ||
    null;
  const agentType = (resolvedCot?.agent_type || '').toLowerCase();
  const nativeProvider = agentType === 'codex' ? 'codex' : agentType === 'claude' ? 'claude' : null;
  const nativeSid = sessionId || resolvedCot?.session_id || null;
  const nativeLabel = nativeProvider === 'codex' ? 'Codex' : 'Claude Code';
  const nativePathPrefix = nativeProvider === 'codex'
    ? '~/.claude/state/otel/codex-'
    : '~/.claude/state/otel/';

  // 注意：所有 hooks 必须在 early return 之前调用（React Rules of Hooks）。
  // useMemo 的代价是每次都计算一份合成内容；Claude 路径下虽然不渲染它但也只是
  // 一次轻量 JSX 构造，不影响性能。
  const content = useMemo(() => {
    if (!node) {
      return (
        <div className="otel-empty">
          <div className="otel-empty-icon">🛰️</div>
          <div className="otel-empty-msg">选中左侧 Span Tree 中的任意节点查看 OTel 视图</div>
          <div className="otel-empty-sub">支持 Session / Turn / Step 三级</div>
        </div>
      );
    }
    if (node.kind === 'session') {
      return <SessionView cot={node.cot} session={node.session} report={node.report} />;
    }
    if (node.kind === 'turn') {
      return <TurnView turn={node.turn} />;
    }
    return <StepView step={node.step} turn={node.turn} />;
  }, [node]);

  if (nativeProvider && nativeSid) {
    return (
      <div className="otel-panel">
        <div className="otel-panel-header">
          <div className="otel-panel-title">
            <span className="otel-panel-logo">📡</span>
            <span>{nativeLabel} 原生 OTel</span>
          </div>
          <div className="otel-panel-sub">
            来自 {nativeLabel} 进程主动 OTLP/HTTP 上报 · 官方真值 ·
            <span className="otel-mono"> {nativePathPrefix}{nativeSid.slice(0, 8)}…/</span>
          </div>
          <NativeOtelDownloadButton sessionId={nativeSid} provider={nativeProvider} />
        </div>
        <div className="otel-panel-body otel-panel-body-claude">
          <ClaudeOtelPanel sessionId={nativeSid} hideInternalToolbar />
        </div>
      </div>
    );
  }

  // ─── Cursor / 其他 agent：保持原合成 GenAI 视图（v0.11.2）──────────

  // 兜底：没显式传 sessionId 时尝试从当前选中节点上推导
  const effectiveSessionId =
    sessionId ||
    (node && node.kind === 'session' ? node.cot?.session_id : null) ||
    (node && node.kind === 'turn' ? node.cot?.session_id : null) ||
    null;

  return (
    <div className="otel-panel">
      <div className="otel-panel-header">
        <div className="otel-panel-title">
          <span className="otel-panel-logo">🛰️</span>
          <span>OpenTelemetry GenAI</span>
        </div>
        <div className="otel-panel-sub">
          schema: <span className="otel-mono">opentelemetry-genai/0.1</span> · client-side derived
        </div>
        <ExportOtlpJsonButton sessionId={effectiveSessionId} />
      </div>
      <div className="otel-panel-body">{content}</div>
    </div>
  );
}
