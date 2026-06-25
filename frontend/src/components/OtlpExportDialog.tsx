/**
 * OtlpExportDialog — v0.12.0
 *
 * 把当前 session 的 cot.json 重放为 OTLP/HTTP traces 推到任意 OTel 兼容后端。
 *
 * 设计目标：让别人复用本项目时，能"一键导出"到他们已有的可观测平台
 *   - Phoenix / Langfuse / SigNoz / Jaeger / Datadog / Honeycomb / Grafana Tempo / ...
 * 同时本地 5173 网站完全不丢，OTLP 是『追加的便利通道』。
 */
import { useEffect, useMemo, useState } from 'react';
import {
  api,
  type OtlpBackendPreset,
  type OtlpExportResult,
} from '../hooks/api';

interface OtlpExportDialogProps {
  sessionId: string;
  open: boolean;
  onClose: () => void;
}

interface HeaderRow {
  k: string;
  v: string;
}

const STORAGE_KEY = 'cot-otlp-export-prefs';

export default function OtlpExportDialog({ sessionId, open, onClose }: OtlpExportDialogProps) {
  const [presets, setPresets] = useState<OtlpBackendPreset[]>([]);
  const [presetId, setPresetId] = useState<string>('otel-collector');
  const [endpoint, setEndpoint] = useState('http://localhost:4318/v1/traces');
  const [serviceName, setServiceName] = useState('cot-extractor');
  const [headers, setHeaders] = useState<HeaderRow[]>([]);
  const [dryRun, setDryRun] = useState<boolean>(false);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<OtlpExportResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // 加载 presets + 从 localStorage 恢复用户上次配置
  useEffect(() => {
    if (!open) return;
    setError(null);
    setResult(null);
    api.getOtlpPresets().then(r => {
      setPresets(r.presets || []);
    }).catch(() => setPresets([]));

    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const saved = JSON.parse(raw);
        if (saved.presetId) setPresetId(saved.presetId);
        if (saved.endpoint) setEndpoint(saved.endpoint);
        if (saved.serviceName) setServiceName(saved.serviceName);
        if (Array.isArray(saved.headers)) setHeaders(saved.headers);
      }
    } catch {
      // ignore
    }
  }, [open]);

  // 切 preset 时把 endpoint / headers hint 自动填入
  const onSelectPreset = (id: string) => {
    setPresetId(id);
    const p = presets.find(p => p.id === id);
    if (p) {
      setEndpoint(p.endpoint);
      const hint = p.headers_hint || {};
      const rows = Object.entries(hint).map(([k, v]) => ({ k, v }));
      setHeaders(rows);
    }
  };

  const currentPresetDoc = useMemo(
    () => presets.find(p => p.id === presetId)?.doc,
    [presets, presetId],
  );

  const updateHeaderRow = (i: number, k: string, v: string) => {
    setHeaders(prev => prev.map((h, idx) => (idx === i ? { k, v } : h)));
  };
  const addHeaderRow = () => setHeaders(prev => [...prev, { k: '', v: '' }]);
  const removeHeaderRow = (i: number) =>
    setHeaders(prev => prev.filter((_, idx) => idx !== i));

  const onSubmit = async () => {
    setSubmitting(true);
    setError(null);
    setResult(null);
    try {
      const headersDict: Record<string, string> = {};
      for (const h of headers) {
        if (h.k.trim()) headersDict[h.k.trim()] = h.v;
      }

      // 持久化偏好（不存 secret 类 header）
      try {
        localStorage.setItem(
          STORAGE_KEY,
          JSON.stringify({
            presetId,
            endpoint,
            serviceName,
            // 不持久化 secret —— 只存 key（v 留空）
            headers: headers.map(h => ({ k: h.k, v: '' })),
          }),
        );
      } catch {
        // ignore
      }

      const r = await api.exportSessionOtlp(sessionId, {
        endpoint: endpoint || undefined,
        headers: Object.keys(headersDict).length ? headersDict : undefined,
        service_name: serviceName,
        dry_run: dryRun,
      });
      setResult(r);
    } catch (e: any) {
      const msg = e?.response?.data?.detail || e?.message || String(e);
      setError(msg);
    } finally {
      setSubmitting(false);
    }
  };

  if (!open) return null;

  return (
    <div className="otlp-modal-backdrop" onClick={onClose}>
      <div className="otlp-modal" onClick={e => e.stopPropagation()}>
        <div className="otlp-modal-head">
          <div className="otlp-modal-title">
            <span className="otlp-modal-icon">🚀</span>
            导出到 OpenTelemetry 后端
          </div>
          <button className="otlp-modal-close" onClick={onClose}>×</button>
        </div>

        <div className="otlp-modal-body">
          <div className="otlp-modal-note">
            把这次 session 的完整 CoT 重放成标准 OTLP/HTTP traces 推到任意 OTel 兼容后端。
            本地 5173 网站不会受影响，这是『追加』通道。
          </div>

          {/* 预设 */}
          <div className="otlp-form-row">
            <label>后端预设</label>
            <select
              value={presetId}
              onChange={e => onSelectPreset(e.target.value)}
            >
              {presets.length === 0 && <option value="otel-collector">OTel Collector (本地)</option>}
              {presets.map(p => (
                <option key={p.id} value={p.id}>{p.label}</option>
              ))}
            </select>
            {currentPresetDoc && (
              <div className="otlp-form-hint">{currentPresetDoc}</div>
            )}
          </div>

          {/* endpoint */}
          <div className="otlp-form-row">
            <label>OTLP/HTTP Endpoint</label>
            <input
              type="text"
              value={endpoint}
              onChange={e => setEndpoint(e.target.value)}
              placeholder="http://localhost:4318/v1/traces"
            />
            <div className="otlp-form-hint">
              留空则用环境变量 <code>OTEL_EXPORTER_OTLP_ENDPOINT</code> 或默认值。
              没带 <code>/v1/traces</code> 时会自动补。
            </div>
          </div>

          {/* service.name */}
          <div className="otlp-form-row">
            <label>service.name</label>
            <input
              type="text"
              value={serviceName}
              onChange={e => setServiceName(e.target.value)}
              placeholder="cot-extractor"
            />
          </div>

          {/* headers */}
          <div className="otlp-form-row">
            <label>HTTP Headers (鉴权 / 项目 ID)</label>
            <div className="otlp-headers-table">
              {headers.map((h, i) => (
                <div className="otlp-header-row" key={i}>
                  <input
                    type="text"
                    placeholder="key"
                    value={h.k}
                    onChange={e => updateHeaderRow(i, e.target.value, h.v)}
                  />
                  <input
                    type="text"
                    placeholder="value"
                    value={h.v}
                    onChange={e => updateHeaderRow(i, h.k, e.target.value)}
                  />
                  <button
                    className="otlp-header-rm"
                    onClick={() => removeHeaderRow(i)}
                    title="删除这一行"
                  >×</button>
                </div>
              ))}
              <button className="otlp-header-add" onClick={addHeaderRow}>
                + 添加 header
              </button>
            </div>
            <div className="otlp-form-hint">
              常见：<code>Authorization: Bearer ...</code>（Langfuse），
              <code>x-honeycomb-team: ...</code>（Honeycomb），
              <code>signoz-access-token: ...</code>（SigNoz）。
            </div>
          </div>

          {/* dry-run */}
          <div className="otlp-form-row">
            <label className="otlp-checkbox">
              <input
                type="checkbox"
                checked={dryRun}
                onChange={e => setDryRun(e.target.checked)}
              />
              <span>Dry-run（不真发，只本地预览 span 树）</span>
            </label>
          </div>

          {/* 错误 / 结果 */}
          {error && (
            <div className="otlp-result otlp-result-err">
              <strong>导出失败：</strong>
              <pre>{error}</pre>
            </div>
          )}
          {result && (
            <div className="otlp-result otlp-result-ok">
              <div className="otlp-result-head">✅ 导出成功</div>
              <div className="otlp-result-grid">
                <div><span>trace_id</span><code>{result.trace_id}</code></div>
                <div><span>spans</span><code>{result.span_count}</code></div>
                <div><span>endpoint</span><code>{result.endpoint || '(dry-run)'}</code></div>
                <div><span>service</span><code>{result.service_name}</code></div>
                <div><span>dry_run</span><code>{String(result.dry_run)}</code></div>
              </div>
              {result.sample_spans && result.sample_spans.length > 0 && (
                <details className="otlp-result-spans">
                  <summary>
                    span 样本（{result.sample_spans.length} / {result.sample_total || result.sample_spans.length}）
                  </summary>
                  <pre>{JSON.stringify(result.sample_spans, null, 2)}</pre>
                </details>
              )}
              <div className="otlp-result-tip">
                打开你后端 UI（Phoenix / Langfuse / SigNoz / Jaeger…），用 trace_id 直接搜索就能看到这次 session。
              </div>
            </div>
          )}
        </div>

        <div className="otlp-modal-foot">
          <button className="otlp-btn-cancel" onClick={onClose} disabled={submitting}>取消</button>
          <button className="otlp-btn-submit" onClick={onSubmit} disabled={submitting}>
            {submitting ? '导出中…' : (dryRun ? '🔬 预览' : '🚀 导出')}
          </button>
        </div>
      </div>
    </div>
  );
}
