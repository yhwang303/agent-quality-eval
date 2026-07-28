import axios from 'axios';
import type { SessionOverview, SessionCoT, ResponseReport, UplinkUserSummary, TurnEvalReport, EvalEvent } from '../types';

// API base URL resolution.
//
// Why empty string by default:
//   In production (cursor-cot start), the FastAPI backend serves both
//   the SPA and the JSON API on the same origin (http://127.0.0.1:<port>).
//   Using a same-origin relative URL ("") means we don't have to know
//   the port at build time and the wheel bundle can be served on any
//   port the runtime picked.
//
// Why VITE_API_BASE override exists:
//   For corner cases (running the prebuilt SPA against a remote backend,
//   or a future "headless" deployment) you can override at build time:
//     VITE_API_BASE=https://my-backend npm run build
//
// Why dev (5173) does NOT need to set this:
//   `vite.config.ts` registers a proxy that forwards `/api/**` from
//   5173 to 127.0.0.1:8765. So `axios.get("/api/sessions")` works
//   identically in dev and prod.
const API_BASE = (import.meta.env.VITE_API_BASE ?? '').replace(/\/+$/, '');

// v0.12.0：OTLP 导出相关类型
export interface OtlpBackendPreset {
  id: string;
  label: string;
  endpoint: string;
  headers_hint?: Record<string, string>;
  doc?: string;
}

export interface OtlpExportRequest {
  endpoint?: string;
  headers?: Record<string, string>;
  service_name?: string;
  service_version?: string;
  deployment_environment?: string;
  dry_run?: boolean;
  timeout?: number;
}

export interface OtlpExportResult {
  ok: boolean;
  trace_id: string;
  endpoint?: string | null;
  service_name: string;
  span_count: number;
  dry_run: boolean;
  sample_spans?: any[];
  sample_total?: number;
}

// ─── v0.16.0: Claude Code 原生 OTel 数据类型 ────────────────────
//
// 后端 services/claude_otel_receiver.py 落到 ~/.claude/state/otel/<sid>/ 的
// events / metrics / traces 三种信号，扁平化后通过 GET /api/sessions/<sid>/otel
// 一次性返回。前端 ClaudeOtelPanel 直接消费这个结构。

export interface ClaudeOtelEvent {
  ts: string | null;
  observed_ts?: string | null;
  severity?: string | null;
  scope?: { name?: string; version?: string };
  event_name?: string;
  body?: any;
  attributes: Record<string, any>;
  resource: Record<string, any>;
}

export interface ClaudeOtelMetricPoint {
  ts: string | null;
  start_ts?: string | null;
  metric: string;
  unit: string;
  description?: string;
  type: 'sum' | 'gauge' | 'histogram' | 'exponentialHistogram' | 'summary' | string;
  value: number | { sum?: number; count?: number; min?: number; max?: number } | any;
  attributes: Record<string, any>;
  scope?: string;
  resource: Record<string, any>;
}

export interface ClaudeOtelSpan {
  trace_id?: string;
  span_id?: string;
  parent_span_id?: string;
  name: string;
  kind?: number;
  start_ts?: string | null;
  end_ts?: string | null;
  status?: number;
  status_message?: string;
  attributes: Record<string, any>;
  events?: { name: string; ts: string | null; attributes: Record<string, any> }[];
  scope?: string;
  resource: Record<string, any>;
}

export interface ClaudeOtelData {
  session_id: string;
  // 标识这条数据来自哪个 IDE 进程；'claude' 是默认值（向后兼容）。
  provider?: 'claude' | 'codex' | 'codebuddy' | string;
  events: ClaudeOtelEvent[];
  metrics: ClaudeOtelMetricPoint[];
  spans: ClaudeOtelSpan[];
  summary: {
    events_total: number;
    metrics_total: number;
    spans_total: number;
    first_ts?: string | null;
    last_ts?: string | null;
    metrics_by_name: Record<string, number>;
    events_by_name: Record<string, number>;
    spans_by_name: Record<string, number>;
    models: Record<string, number>;
    totals: {
      input_tokens: number;
      output_tokens: number;
      cache_read_tokens: number;
      cache_creation_tokens: number;
      cost_usd: number;
    };
  };
}

export interface OtelSessionListItem {
  session_id: string;
  // backend 解析 provider 前缀，暴露真实 IDE provider
  // 与去前缀的裸 session id；前端用 provider 决定徽章颜色 / 默认面板视图。
  bare_session_id?: string;
  provider?: 'claude' | 'codex' | 'codebuddy' | string;
  events_bytes: number;
  metrics_bytes: number;
  spans_bytes: number;
  last_modified: number;
}

// ─── 会话有序事件流 ────────────────────────────────────────
//
// 后端 agent_cot.trace 把主步流和 5 条并行时间线（权限 / 子代理 / 通知 /
// 压缩 / 环境）拍平成一条按真实执行顺序排列的事件流。顺序真值只有这一份：
// SpanTree 按它渲染，trace 导出用同一份结果，所以 UI 和导出不可能漂移。
//
// group_id / role / phase_id 是给 UI 重建折叠树用的；导出侧忽略它们。

export interface TraceEvent {
  seq: number;
  session_id: string;
  turn?: number;
  type: string;
  step_type?: string;
  step_index?: number;
  t_ms?: number;
  ts?: string;
  duration_ms?: number;
  title?: string;
  content?: string;
  tool?: string;
  tool_use_id?: string;
  payload?: any;
  plan?: Record<string, any>;
  tokens?: number;
  token_usage?: Record<string, any>;
  truncated?: boolean;
  original_len?: number;
  is_error?: boolean;
  otel_orphan?: boolean;
  // UI 重建用
  group_id?: number;
  role?: 'leader' | 'child';
  phase_id?: number;
  // 当前 UI 不渲染（落在 turn 时间窗之外、或权限模式切换这类记录）
  in_ui?: boolean;
  // 无可用时间戳、排不进执行顺序，被追加在有序流之后
  ordered?: boolean;
}

export interface SessionTimeline {
  schema: string;
  header: Record<string, any>;
  events: TraceEvent[];
}

export interface DownloadedFile {
  blob: Blob;
  filename: string;
}

// 后端在 Content-Disposition 里给了带会话/轮次的规范文件名，优先用它；
// 拿不到再退回调用方给的兜底名，别让用户存出一堆 "download" 之类的文件。
async function downloadAttachment(
  url: string,
  fallbackName: string,
): Promise<DownloadedFile> {
  const resp = await fetch(url);
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const j = await resp.json();
      if (j?.detail) detail = j.detail;
    } catch {
      // 非 JSON 错误体（例如反向代理返回的 HTML）就保留 HTTP 状态码
    }
    throw new Error(detail);
  }
  const blob = await resp.blob();
  const cd = resp.headers.get('Content-Disposition') || '';
  let filename = fallbackName;
  const m = cd.match(/filename\*?=(?:UTF-8''|")?([^";]+)"?/i);
  if (m && m[1]) {
    try { filename = decodeURIComponent(m[1]); } catch { filename = m[1]; }
  }
  return { blob, filename };
}

// 按扩展名给"另存为"对话框一个像样的文件类型过滤。
const _SAVE_TYPES: Record<string, { description: string; mime: string }> = {
  json: { description: 'JSON file', mime: 'application/json' },
  jsonl: { description: 'JSON Lines file', mime: 'application/x-ndjson' },
  md: { description: 'Markdown file', mime: 'text/markdown' },
};

/**
 * 把拿到的文件存到磁盘上。所有导出按钮都必须走这里。
 *
 * 三个看起来可有可无、实际缺一不可的细节，全都是踩过的坑：
 *
 * 1. `<a>` 必须先挂进 document 再 click。游离节点上的 click 在桌面壳
 *    （Electron/WebView）里会被直接忽略，表现就是"点了没反应"。
 * 2. `revokeObjectURL` 要延后。立刻回收会在下载真正开始前把 blob 抽掉，
 *    同样是点了没反应。
 * 3. 优先走 `showSaveFilePicker`，让用户自己选保存位置；桌面壳里有些实现
 *    暴露了这个 API 却禁用它，所以失败要回退到 `<a download>`，而不是
 *    把整个导出卡死。
 *
 * 这段逻辑之前在 OtelPanel 和 ClaudeOtelPanel 各抄了一份，新写的导出按钮
 * 又照着"想当然"的版本重写了第三份，于是同一个"点了没反应"复发。现在只留
 * 这一份，两个 OTel 面板也改成调它。
 */
export async function saveDownloadedFile(
  { blob, filename }: DownloadedFile,
): Promise<'picked' | 'downloaded'> {
  const ext = (filename.split('.').pop() || '').toLowerCase();
  const kind = _SAVE_TYPES[ext] || { description: 'File', mime: 'application/octet-stream' };

  const picker = (window as any).showSaveFilePicker;
  if (typeof picker === 'function') {
    try {
      const handle = await picker({
        suggestedName: filename,
        types: [{ description: kind.description, accept: { [kind.mime]: [`.${ext}`] } }],
      });
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
      return 'picked';
    } catch (e: any) {
      if (e?.name === 'AbortError') {
        throw new Error('已取消保存');
      }
      // 暴露了 API 但被禁用：继续走下面的普通下载，别让导出直接失败
    }
  }

  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
  return 'downloaded';
}

export const api = {
  getSessions: (): Promise<SessionOverview[]> =>
    axios.get(`${API_BASE}/api/sessions`).then(r => r.data),

  getSessionCot: (sessionId: string): Promise<SessionCoT> =>
    axios.get(`${API_BASE}/api/sessions/${sessionId}/cot`).then(r => r.data),

  getSessionReport: (sessionId: string): Promise<ResponseReport> =>
    axios.get(`${API_BASE}/api/sessions/${sessionId}/report`).then(r => r.data),

  evalTurn: (sessionId: string, turnIndex: number): Promise<TurnEvalReport> =>
    axios.post(`${API_BASE}/api/evals/session/${sessionId}/turn/${turnIndex}`).then(r => r.data),

  getLatestTurnEval: (sessionId: string, turnIndex: number): Promise<TurnEvalReport> =>
    axios.get(`${API_BASE}/api/evals/session/${sessionId}/turn/${turnIndex}/latest`).then(r => r.data),

  listTurnEvalReports: (sessionId?: string): Promise<{ turn_evals: TurnEvalReport[] }> => {
    const qs = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
    return axios.get(`${API_BASE}/api/evals/turn-reports${qs}`).then(r => r.data);
  },

  listEvalEvents: (params?: { event_type?: string; project_id?: string; has_gold?: boolean; limit?: number }): Promise<{ events: EvalEvent[] }> => {
    const qs = new URLSearchParams();
    if (params?.event_type) qs.set('event_type', params.event_type);
    if (params?.project_id) qs.set('project_id', params.project_id);
    if (typeof params?.has_gold === 'boolean') qs.set('has_gold', String(params.has_gold));
    if (params?.limit) qs.set('limit', String(params.limit));
    const suffix = qs.toString() ? `?${qs}` : '';
    return axios.get(`${API_BASE}/api/evals/events${suffix}`).then(r => r.data);
  },

  getLiveCritic: (sessionId: string): Promise<any> =>
    axios.get(`${API_BASE}/api/evals/live/${sessionId}`).then(r => r.data),

  getLiveCriticTurn: (sessionId: string, turnIndex: number): Promise<any> =>
    axios.get(`${API_BASE}/api/evals/live/${sessionId}/turn/${turnIndex}`).then(r => r.data),

  compareTurns: (
    baseline: { session_id: string; turn_index: number },
    candidate: { session_id: string; turn_index: number },
    mode: 'ab' | 'regression' = 'ab',
    reference_answer?: any,
    blind?: boolean,
  ): Promise<any> =>
    axios.post(`${API_BASE}/api/evals/turn-compare`, { baseline, candidate, mode, reference_answer, blind: !!blind }).then(r => r.data),

  uploadTrace: (body: { source: string; title?: string; trace: any; transcript?: string }): Promise<any> =>
    axios.post(`${API_BASE}/api/evals/upload-trace`, body).then(r => r.data),

  getTurnReferenceAnswer: (sessionId: string, turnIndex: number): Promise<any> =>
    axios.get(`${API_BASE}/api/evals/session/${sessionId}/turn/${turnIndex}/reference-answer`).then(r => r.data),

  saveTurnReferenceAnswer: (sessionId: string, turnIndex: number, confirmToken: string): Promise<any> =>
    axios.post(`${API_BASE}/api/evals/session/${sessionId}/turn/${turnIndex}/reference-answer`, { confirm_token: confirmToken }).then(r => r.data),

  deleteTurnReferenceAnswer: (sessionId: string, turnIndex: number): Promise<any> =>
    axios.delete(`${API_BASE}/api/evals/session/${sessionId}/turn/${turnIndex}/reference-answer`).then(r => r.data),

  normalizeReferenceAnswer: (
    filename: string,
    content: string,
    target?: { session_id: string; turn_index: number },
  ): Promise<any> =>
    axios.post(`${API_BASE}/api/evals/reference-answer/normalize`, { filename, content, ...(target || {}) }).then(r => r.data),

  listReferenceDatasets: (): Promise<{ datasets: any[] }> =>
    axios.get(`${API_BASE}/api/evals/reference-datasets`).then(r => r.data),

  getReferenceDataset: (datasetId: string): Promise<any> =>
    axios.get(`${API_BASE}/api/evals/reference-datasets/${encodeURIComponent(datasetId)}`).then(r => r.data),

  uploadReferenceDataset: (filename: string, content: string): Promise<any> =>
    axios.post(`${API_BASE}/api/evals/reference-datasets/upload`, { filename, content }).then(r => r.data),

  runReferenceEval: (body: { session_id: string; turn_index: number; dataset_id: string; case_id?: string | null }): Promise<any> =>
    axios.post(`${API_BASE}/api/evals/reference-eval`, body).then(r => r.data),

  getCriticSettings: (): Promise<any> =>
    axios.get(`${API_BASE}/api/evals/settings/critic`).then(r => r.data),

  saveCriticSettings: (body: any): Promise<any> =>
    axios.put(`${API_BASE}/api/evals/settings/critic`, body).then(r => r.data),

  getUiEvents: (): Promise<{ settings_open_seq: number }> =>
    axios.get(`${API_BASE}/api/evals/ui/events`).then(r => r.data),

  getHookHealth: (): Promise<any> =>
    axios.get(`${API_BASE}/api/evals/hook-health`).then(r => r.data),

  getLlmJudgeSettings: (): Promise<any> =>
    axios.get(`${API_BASE}/api/evals/settings/critic`).then(r => r.data),

  saveLlmJudgeSettings: (body: any): Promise<any> =>
    axios.put(`${API_BASE}/api/evals/settings/critic`, body).then(r => r.data),

  getSessionTranscript: (sessionId: string): Promise<any> =>
    axios.get(`${API_BASE}/api/sessions/${sessionId}/transcript`).then(r => r.data),

  deleteSession: (sessionId: string): Promise<any> =>
    axios.delete(`${API_BASE}/api/sessions/${sessionId}`).then(r => r.data),

  // v0.12.0：把 cot.json 重放为 OTLP/HTTP traces 推到任意 OTel 后端
  getOtlpPresets: (): Promise<{ presets: OtlpBackendPreset[]; error?: string }> =>
    axios.get(`${API_BASE}/api/otlp/presets`).then(r => r.data),

  exportSessionOtlp: (sessionId: string, body: OtlpExportRequest): Promise<OtlpExportResult> =>
    axios
      .post(`${API_BASE}/api/sessions/${sessionId}/export/otlp`, body)
      .then(r => r.data),

  // v0.14.8：把 cot.json 离线重放为 OTLP/JSON 协议文件，浏览器触发下载。
  // 后端响应是 application/json + Content-Disposition: attachment，
  // 这里 fetch 拿到 Blob 后做客户端下载，比 axios.get(responseType='blob') 简单
  // 且天然支持服务器给出的 filename。
  downloadSessionOtlpJson: async (
    sessionId: string,
    opts?: { service_name?: string; service_version?: string; deployment_environment?: string },
  ): Promise<{ blob: Blob; filename: string; trace_id: string; span_count: number }> => {
    const qs = new URLSearchParams();
    if (opts?.service_name) qs.set('service_name', opts.service_name);
    if (opts?.service_version) qs.set('service_version', opts.service_version);
    if (opts?.deployment_environment) qs.set('deployment_environment', opts.deployment_environment);
    const url = `${API_BASE}/api/sessions/${sessionId}/export/otlp.json${qs.toString() ? `?${qs}` : ''}`;
    const resp = await fetch(url);
    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`;
      try { const j = await resp.json(); if (j?.detail) detail = j.detail; } catch {}
      throw new Error(detail);
    }
    const blob = await resp.blob();
    // 解析 Content-Disposition 拿 filename；fallback 用 session id 拼一个
    const cd = resp.headers.get('Content-Disposition') || '';
    let filename = `otlp-${sessionId.slice(0, 8)}.json`;
    const m = cd.match(/filename\*?=(?:UTF-8''|")?([^";]+)"?/i);
    if (m && m[1]) {
      try { filename = decodeURIComponent(m[1]); } catch { filename = m[1]; }
    }
    return {
      blob,
      filename,
      trace_id: resp.headers.get('X-Cot-Trace-Id') || '',
      span_count: parseInt(resp.headers.get('X-Cot-Span-Count') || '0', 10) || 0,
    };
  },

  // ─── 会话有序事件流（SpanTree 的渲染顺序来源）──────────────
  getSessionTimeline: (sessionId: string): Promise<SessionTimeline> =>
    axios.get(`${API_BASE}/api/sessions/${sessionId}/timeline`).then(r => r.data),

  // 下载**一轮交互**的 trace。界面上每张交互卡片旁的导出按钮走这里。
  // 只出 jsonl：一行一个事件，顺序即行号，下游脚本直接可读。
  downloadTurnTrace: (sessionId: string, turnIndex: number) =>
    downloadAttachment(
      `${API_BASE}/api/sessions/${sessionId}/turns/${turnIndex}/export/trace.jsonl`,
      `trace-${sessionId.slice(0, 8)}-turn${turnIndex}.jsonl`,
    ),

  // 下载一轮的完整 eval 结果（断言明细 / 维度面板 / hook 阶段评审）。
  downloadTurnEval: (sessionId: string, turnIndex: number) =>
    downloadAttachment(
      `${API_BASE}/api/evals/session/${sessionId}/turn/${turnIndex}/export.json`,
      `eval-${sessionId.slice(0, 8)}-turn${turnIndex}.json`,
    ),

  // ─── v0.16.0: Claude Code 原生 OTel ────────────────────────
  getClaudeOtel: (sessionId: string): Promise<ClaudeOtelData> =>
    axios.get(`${API_BASE}/api/sessions/${sessionId}/otel`).then(r => r.data),

  listOtelSessions: (): Promise<{ sessions: OtelSessionListItem[] }> =>
    axios.get(`${API_BASE}/api/otel/sessions`).then(r => r.data),

  // ─── v0.18.0: 中央 CoT Uplink ──────────────────────────────
  // 列出所有有上行数据的同事（前端用户筛选下拉框数据源）
  listUplinkUsers: (): Promise<{ users: UplinkUserSummary[] }> =>
    axios.get(`${API_BASE}/api/uplink/users`).then(r => r.data),

  // 中央服务自检：当前 token 配置 / 落盘根 / 已收到的 user/session 总数
  getUplinkStatus: (): Promise<{
    central_root: string;
    central_root_exists: boolean;
    auth_required: boolean;
    user_count: number;
    total_session_count: number;
    users: UplinkUserSummary[];
  }> =>
    axios.get(`${API_BASE}/api/uplink/status`).then(r => r.data),
};
