import { useState, useEffect, useRef, useCallback } from 'react';
import type { SessionOverview, SessionCoT, ResponseReport, TurnEvalReport, TurnCoT } from './types';
import type { SelectedNode } from './components/SpanTree';
import { api } from './hooks/api';
import SessionList from './components/SessionList';
import SpanTree from './components/SpanTree';
import DetailPanel from './components/DetailPanel';
import OtelPanel from './components/OtelPanel';
import type { OtelSelectedNode } from './components/OtelPanel';
import './App.css';

type RightTab = 'detail' | 'otel';

const INCOMPLETE_EVAL_STALE_MS = 10 * 60 * 1000;

function isIncompleteEvalFresh(report: TurnEvalReport | undefined): boolean {
  if (!report) return false;
  const status = String(report?.judge?.status || (report?.eval_panel as any)?.agent_critic?.status || '').toLowerCase();
  if (status !== 'queued' && status !== 'running') return false;
  const rawStarted = String((report.judge as any)?.created_at || report.created_at || '').trim();
  const started = rawStarted ? Date.parse(rawStarted) : Number.NaN;
  if (!Number.isFinite(started)) return true;
  return Date.now() - started < INCOMPLETE_EVAL_STALE_MS;
}

export default function App() {
  const [sessions, setSessions] = useState<SessionOverview[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [cot, setCot] = useState<SessionCoT | null>(null);
  const [report, setReport] = useState<ResponseReport | null>(null);
  const [selectedNode, setSelectedNode] = useState<SelectedNode | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [turnEvalReports, setTurnEvalReports] = useState<Record<string, TurnEvalReport>>({});
  const [liveCritic, setLiveCritic] = useState<any | null>(null);
  const [liveCriticTurns, setLiveCriticTurns] = useState<Record<string, any>>({});
  const [turnEvalLoadingKey, setTurnEvalLoadingKey] = useState<string | null>(null);
  const [turnEvalError, setTurnEvalError] = useState<string | null>(null);
  const [abBaseline, setAbBaseline] = useState<{ session_id: string; turn_index: number } | null>(null);
  const [abCandidate, setAbCandidate] = useState<{ session_id: string; turn_index: number } | null>(null);
  const [abResult, setAbResult] = useState<any | null>(null);
  const [abResultCache, setAbResultCache] = useState<Record<string, any>>({});
  const [abLoading, setAbLoading] = useState(false);
  const [abCompareOpen, setAbCompareOpen] = useState(false);
  const [abNotice, setAbNotice] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(() => new URLSearchParams(window.location.search).get('settings') === '1');
  const [hookHealthOpen, setHookHealthOpen] = useState(false);
  // v0.11.2：右侧栏 Tab 切换 —— 详情视图 ↔ OpenTelemetry GenAI 视图
  const [rightTab, setRightTab] = useState<RightTab>('detail');

  // ─── 拖拽分隔条状态 ────────────────────────────────────
  const [sidebarWidth, setSidebarWidth] = useState(260);
  const [detailWidth, setDetailWidth] = useState(380);
  const draggingRef = useRef<'left' | 'right' | null>(null);
  const startXRef = useRef(0);
  const startWidthRef = useRef(0);

  const handleMouseDown = useCallback((side: 'left' | 'right', e: React.MouseEvent) => {
    e.preventDefault();
    draggingRef.current = side;
    startXRef.current = e.clientX;
    startWidthRef.current = side === 'left' ? sidebarWidth : detailWidth;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, [sidebarWidth, detailWidth]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!draggingRef.current) return;
      const delta = e.clientX - startXRef.current;
      if (draggingRef.current === 'left') {
        setSidebarWidth(Math.max(180, Math.min(500, startWidthRef.current + delta)));
      } else {
        setDetailWidth(Math.max(260, Math.min(700, startWidthRef.current - delta)));
      }
    };
    const handleMouseUp = () => {
      if (draggingRef.current) {
        draggingRef.current = null;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      }
    };
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  // 加载 session 列表
  useEffect(() => {
    api.getSessions()
      .then(setSessions)
      .catch(() => setError('无法连接后端，请确认 API 服务已启动（python backend/main.py）'));
  }, []);

  // 选中 session 时加载详情
  useEffect(() => {
    if (!selectedId) return;
    setLoading(true);
    setCot(null);
    setReport(null);
    setSelectedNode(null);
    setTurnEvalReports({});
    setLiveCritic(null);
    setLiveCriticTurns({});
    setTurnEvalLoadingKey(null);
    setTurnEvalError(null);

    const cotPromise = api.getSessionCot(selectedId).then(setCot).catch(() => {});
    const reportPromise = api.getSessionReport(selectedId).then(setReport).catch(() => {});
    const turnEvalPromise = api.listTurnEvalReports(selectedId)
      .then(data => {
        const next: Record<string, TurnEvalReport> = {};
        for (const item of data.turn_evals || []) {
          if (item.eval_version !== 'v3' || !item.assertion_results?.length) continue;
          next[`${item.session_id}:${item.turn_index}`] = item;
        }
        setTurnEvalReports(next);
      })
      .catch(() => {});
    const liveCriticPromise = api.getLiveCritic(selectedId).then(setLiveCritic).catch(() => setLiveCritic(null));

    Promise.all([cotPromise, reportPromise, turnEvalPromise, liveCriticPromise]).finally(() => setLoading(false));
  }, [selectedId]);

  useEffect(() => {
    if (!selectedId) return;
    const loadLive = () => api.getLiveCritic(selectedId).then(setLiveCritic).catch(() => {});
    const timer = window.setInterval(loadLive, 3000);
    return () => window.clearInterval(timer);
  }, [selectedId]);

  useEffect(() => {
    if (!selectedId || selectedNode?.kind !== 'turn' || !selectedNode.cot) return;
    const sessionId = selectedNode.cot.session_id;
    const turnIndex = selectedNode.turn.turn_index;
    const key = `${sessionId}:${turnIndex}`;
    const loadTurnLive = () => {
      api.getLiveCriticTurn(sessionId, turnIndex)
        .then(data => setLiveCriticTurns(prev => ({ ...prev, [key]: data })))
        .catch(() => {});
    };
    loadTurnLive();
    const timer = window.setInterval(loadTurnLive, 3000);
    return () => window.clearInterval(timer);
  }, [selectedId, selectedNode]);

  useEffect(() => {
    if (!selectedId || !cot?.turns?.length) return;
    const loadReports = () => {
      api.listTurnEvalReports(selectedId)
        .then(data => {
          const next: Record<string, TurnEvalReport> = {};
          for (const item of data.turn_evals || []) {
            if (item.eval_version !== 'v3' || !item.assertion_results?.length) continue;
            next[`${item.session_id}:${item.turn_index}`] = item;
          }
          setTurnEvalReports(next);
        })
        .catch(() => {});
    };
    const needsRefresh = cot.turns.some(turn => {
      const report = turnEvalReports[`${cot.session_id}:${turn.turn_index}`] as any;
      if (!report) return turnEvalLoadingKey === `${cot.session_id}:${turn.turn_index}`;
      return isIncompleteEvalFresh(report);
    });
    if (!needsRefresh) return;
    const timer = window.setInterval(loadReports, 5000);
    return () => window.clearInterval(timer);
  }, [selectedId, cot, turnEvalReports, turnEvalLoadingKey]);

  const selectedSession = sessions.find(s => s.session_id === selectedId) || null;
  const selectedTurnLiveCritic = selectedNode?.kind === 'turn' && selectedNode.cot
    ? liveCriticTurns[`${selectedNode.cot.session_id}:${selectedNode.turn.turn_index}`]
    : null;
  const selectedTurnRef = selectedNode?.kind === 'turn' && selectedNode.cot
    ? { session_id: selectedNode.cot.session_id, turn_index: selectedNode.turn.turn_index }
    : null;
  const sameTurnRef = (
    left: { session_id: string; turn_index: number } | null,
    right: { session_id: string; turn_index: number } | null,
  ) => Boolean(left && right && left.session_id === right.session_id && left.turn_index === right.turn_index);
  const abPairKey = (
    baselineRef: { session_id: string; turn_index: number } | null,
    candidateRef: { session_id: string; turn_index: number } | null,
  ) => (
    baselineRef && candidateRef
      ? `${baselineRef.session_id}:${baselineRef.turn_index}__${candidateRef.session_id}:${candidateRef.turn_index}`
      : ''
  );
  const abRefLabel = (ref: { session_id: string; turn_index: number } | null) => (
    ref ? `${ref.session_id.slice(0, 8)} · 第 ${ref.turn_index} 轮` : '未选择'
  );
  const markBaseline = () => {
    if (!selectedTurnRef) return;
    setAbBaseline(selectedTurnRef);
    setAbNotice(`已设为基线：${abRefLabel(selectedTurnRef)}`);
  };
  const markCandidate = () => {
    if (!selectedTurnRef) return;
    setAbCandidate(selectedTurnRef);
    setAbNotice(`已设为候选：${abRefLabel(selectedTurnRef)}`);
  };
  const clearBaseline = () => {
    setAbBaseline(null);
    setAbNotice('已清空基线');
  };
  const clearCandidate = () => {
    setAbCandidate(null);
    setAbNotice('已清空候选');
  };
  const clearAbSelection = () => {
    setAbBaseline(null);
    setAbCandidate(null);
    setAbResult(null);
    setAbNotice('已清空 A/B 选择');
  };

  const handleDelete = (deletedId: string) => {
    setSessions(prev => prev.filter(s => s.session_id !== deletedId));
    if (selectedId === deletedId) {
      setSelectedId(null);
      setCot(null);
      setReport(null);
      setSelectedNode(null);
    }
  };

  const handleEvalTurn = useCallback((turn: TurnCoT, currentCot: SessionCoT) => {
    const sessionId = currentCot.session_id;
    const key = `${sessionId}:${turn.turn_index}`;
    setRightTab('detail');
    setSelectedNode({ kind: 'turn', turn, cot: currentCot });
    setTurnEvalError(null);
    setTurnEvalLoadingKey(key);
    api.evalTurn(sessionId, turn.turn_index)
      .then(report => {
        setTurnEvalReports(prev => ({ ...prev, [key]: report }));
      })
      .catch((err: any) => {
        const detail = err?.response?.data?.detail || err?.message || 'Eval failed';
        setTurnEvalError(String(detail));
      })
      .finally(() => {
        setTurnEvalLoadingKey(prev => (prev === key ? null : prev));
      });
  }, []);

  const compareAb = useCallback(() => {
    if (!abBaseline || !abCandidate) return;
    const pairKey = abPairKey(abBaseline, abCandidate);
    const cached = abResultCache[pairKey];
    if (cached) {
      setAbResult(cached);
      setAbCompareOpen(true);
      return;
    }
    setAbLoading(true);
    setAbCompareOpen(true);
    api.compareTurns(abBaseline, abCandidate)
      .then(result => {
        setAbResult(result);
        if (result?.llm_compare?.status === 'completed') {
          setAbResultCache(prev => ({ ...prev, [pairKey]: result }));
        }
      })
      .catch((err: any) => setAbResult({ error: err?.response?.data?.detail || err?.message || 'Compare failed' }))
      .finally(() => setAbLoading(false));
  }, [abBaseline, abCandidate, abResultCache]);

  return (
    <div className="app">
      {/* 顶部标题栏 */}
      <header className="app-header">
        <div className="header-left">
          <span className="header-logo">🤖</span>
          <span className="header-title">Agent Observation</span>
          <span className="header-sep">·</span>
          <span className="header-subtitle">Agent 观测平台</span>
        </div>
        <div className="header-right">
          <div className="ab-toolbar" aria-live="polite">
            <button
              className={`btn-refresh ab-action ${sameTurnRef(selectedTurnRef, abBaseline) ? 'is-active' : ''}`}
              disabled={!selectedTurnRef}
              onClick={markBaseline}
              title={abBaseline ? `当前基线：${abRefLabel(abBaseline)}` : '把当前选中的 turn 设为 A/B 基线'}
            >
              设为基线
            </button>
            <button
              className={`btn-refresh ab-action ${sameTurnRef(selectedTurnRef, abCandidate) ? 'is-active is-candidate' : ''}`}
              disabled={!selectedTurnRef}
              onClick={markCandidate}
              title={abCandidate ? `当前候选：${abRefLabel(abCandidate)}` : '把当前选中的 turn 设为 A/B 候选'}
            >
              设为候选
            </button>
            <div className="ab-ref-strip">
              <span className={`ab-ref-chip ${abBaseline ? 'is-set' : ''}`}>
                <span>基线：{abRefLabel(abBaseline)}</span>
                {abBaseline && <button className="ab-chip-clear" onClick={clearBaseline} title="清空基线">×</button>}
              </span>
              <span className={`ab-ref-chip ${abCandidate ? 'is-set is-candidate' : ''}`}>
                <span>候选：{abRefLabel(abCandidate)}</span>
                {abCandidate && <button className="ab-chip-clear" onClick={clearCandidate} title="清空候选">×</button>}
              </span>
              {abNotice && <span className="ab-notice">{abNotice}</span>}
            </div>
            <button className="btn-refresh ab-clear" disabled={!abBaseline && !abCandidate} onClick={clearAbSelection} title="清空基线和候选">
              清空
            </button>
            <button className={`btn-refresh ab-compare ${abBaseline && abCandidate ? 'is-ready' : ''}`} disabled={!abBaseline || !abCandidate || abLoading} onClick={compareAb}>
              {abLoading ? '对比中...' : 'A/B 对比'}
            </button>
          </div>
          <button className="btn-refresh" onClick={() => setSettingsOpen(true)}>
            API 设置
          </button>
          <button className="btn-refresh" onClick={() => setHookHealthOpen(true)}>
            IDE Hook 检查
          </button>
          <button className="btn-refresh" onClick={() => {
            api.getSessions().then(setSessions).catch(() => {});
          }}>
            ↺ 刷新
          </button>
        </div>
      </header>

      {error && (
        <div className="error-banner">
          ⚠️ {error}
        </div>
      )}

      {settingsOpen && <SettingsDialog onClose={() => setSettingsOpen(false)} />}
      {hookHealthOpen && <HookHealthDialog onClose={() => setHookHealthOpen(false)} />}
      {abCompareOpen && (
        <AbCompareDialog
          result={abResult}
          loading={abLoading}
          baseline={abBaseline}
          candidate={abCandidate}
          onClose={() => setAbCompareOpen(false)}
        />
      )}

      {/* 主体三栏布局 */}
      <div className="app-body">
        {/* 左栏：Session 列表 */}
        <aside className="sidebar" style={{ width: sidebarWidth }}>
          <SessionList
            sessions={sessions}
            selectedId={selectedId}
            onSelect={setSelectedId}
            onDelete={handleDelete}
          />
        </aside>

        {/* 左侧拖拽分隔条 */}
        <div
          className="resize-handle"
          onMouseDown={(e) => handleMouseDown('left', e)}
        />

        {/* 中栏：CoT 时间线 */}
        <main className="main-panel">
          {loading && (
            <div className="loading">
              <div className="loading-spinner" />
              <span>加载中…</span>
            </div>
          )}
          {!loading && !cot && !selectedId && (
            <div className="empty-state">
              <div className="empty-icon">📊</div>
              <div className="empty-text">选择左侧 Session 开始观测</div>
              <div className="empty-sub">共 {sessions.length} 个 Session</div>
            </div>
          )}
          {!loading && cot && (
            <SpanTree
              cot={cot}
              session={selectedSession}
              report={report}
              selectedNode={selectedNode}
              onSelectNode={setSelectedNode}
              onEvalTurn={handleEvalTurn}
              turnEvalReports={turnEvalReports}
              turnEvalLoadingKey={turnEvalLoadingKey}
              liveCritic={liveCritic}
              liveCriticTurns={liveCriticTurns}
            />
          )}
        </main>

        {/* 右侧拖拽分隔条 */}
        <div
          className="resize-handle"
          onMouseDown={(e) => handleMouseDown('right', e)}
        />

        {/* 右栏：详情 + OTel 视图（Tab 切换） */}
        <aside className="detail-sidebar" style={{ width: detailWidth }}>
          <div className="right-tabs">
            <button
              className={`right-tab ${rightTab === 'detail' ? 'right-tab-active' : ''}`}
              onClick={() => setRightTab('detail')}
              title="详情视图（CoT / Plan / Tools / Eval / ...）"
            >
              📋 详情
            </button>
            <button
              className={`right-tab ${rightTab === 'otel' ? 'right-tab-active' : ''}`}
              onClick={() => setRightTab('otel')}
              title="OpenTelemetry GenAI 视图（trace/span/token/cost/messages/retrieval）"
            >
              🛰️ OTel
            </button>
          </div>
          {rightTab === 'detail' ? (
            <DetailPanel
              node={selectedNode}
              onSelectNode={setSelectedNode}
              turnEvalReports={turnEvalReports}
              liveCritic={selectedTurnLiveCritic || liveCritic}
              turnEvalLoadingKey={turnEvalLoadingKey}
              turnEvalError={turnEvalError}
            />
          ) : (
            <OtelPanel
              node={selectedNode as OtelSelectedNode | null}
              sessionId={cot?.session_id || null}
              cot={cot}
            />
          )}
        </aside>
      </div>
    </div>
  );
}

function AbCompareDialog({
  result,
  loading,
  baseline,
  candidate,
  onClose,
}: {
  result: any | null;
  loading: boolean;
  baseline: { session_id: string; turn_index: number } | null;
  candidate: { session_id: string; turn_index: number } | null;
  onClose: () => void;
}) {
  const dimensionLabels: Record<string, string> = {
    task_completion: '任务完成',
    tool_use: '工具使用',
    reasoning: '推理路径',
    instruction_following: '指令遵循',
    faithfulness: '忠实度',
    efficiency: '效率',
    reliability: '可靠性',
  };
  const dimensions = Object.keys(dimensionLabels);
  const fmtPct = (value?: number | null) => {
    if (typeof value !== 'number' || Number.isNaN(value)) return '--';
    return `${(value * 100).toFixed(1)}%`;
  };
  const fmtNum = (value?: number | null, digits = 0) => {
    if (typeof value !== 'number' || Number.isNaN(value)) return '--';
    return value.toLocaleString(undefined, { maximumFractionDigits: digits });
  };
  const fmtDelta = (value?: number | null, digits = 0) => {
    if (typeof value !== 'number' || Number.isNaN(value)) return '--';
    if (Math.abs(value) < 1e-9) return '-';
    const formatted = Math.abs(value).toLocaleString(undefined, { maximumFractionDigits: digits });
    return `${value > 0 ? '+' : value < 0 ? '-' : ''}${formatted}`;
  };
  const fmtPctDelta = (value?: number | null) => {
    if (typeof value !== 'number' || Number.isNaN(value)) return '--';
    if (Math.abs(value) < 1e-9) return '-';
    return `${value > 0 ? '+' : ''}${(value * 100).toFixed(1)}%`;
  };
  const fmtDuration = (value?: number | null) => {
    if (typeof value !== 'number' || Number.isNaN(value)) return '--';
    if (value >= 1000) return `${(value / 1000).toFixed(1)}s`;
    return `${Math.round(value)}ms`;
  };
  const refLabel = (ref: { session_id: string; turn_index: number } | null) => (
    ref ? `${ref.session_id.slice(0, 12)} · Turn ${ref.turn_index}` : '未选择'
  );
  const baselineData = result?.baseline || {};
  const candidateData = result?.candidate || {};
  const baselineMeta = baselineData.trace_meta || {};
  const candidateMeta = candidateData.trace_meta || {};
  const summary = result?.summary || {};
  const llmCompare = result?.llm_compare || {};
  const diffs = Array.isArray(result?.diffs) ? result.diffs : [];
  const groupMap = new Map<string, any>();
  for (const group of baselineData.groups || []) {
    groupMap.set(group.key, { key: group.key, label: group.label, baseline: group.pass_rate, candidate: null });
  }
  for (const group of candidateData.groups || []) {
    const row = groupMap.get(group.key) || { key: group.key, label: group.label, baseline: null, candidate: null };
    row.candidate = group.pass_rate;
    row.label = group.label || row.label;
    groupMap.set(group.key, row);
  }
  const metricRows = [
    ['Total Tokens', 'total_tokens'],
    ['Input Tokens', 'input_tokens'],
    ['Output Tokens', 'output_tokens'],
    ['Tool Calls', 'tool_count'],
    ['Tool Kinds', 'tool_kind_count'],
    ['Shell', 'shell_tool_count'],
    ['Read', 'read_tool_count'],
    ['Write', 'write_tool_count'],
    ['Edit/Patch', 'edit_tool_count'],
    ['MCP', 'mcp_tool_count'],
    ['RAG', 'rag_tool_count'],
    ['Retrieval', 'retrieval_tool_count'],
    ['Search', 'search_tool_count'],
    ['Browser/GUI', 'browser_tool_count'],
    ['Tool Errors', 'tool_error_count'],
  ];
  const topDiffs = [...diffs].sort((a, b) => Math.abs(Number(b.delta || 0)) - Math.abs(Number(a.delta || 0))).slice(0, 12);
  const metricValue = (side: any, key: string) => {
    const value = side.metrics?.[key];
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  };
  const verdictText: Record<string, string> = {
    candidate_better: '候选更优',
    baseline_better: 'Base 更优',
    mixed: '各有优劣',
    no_material_difference: '无显著差异',
  };
  const winnerText: Record<string, string> = {
    baseline: 'Base',
    candidate: '候选',
    tie: '持平',
    unclear: '不明确',
  };
  const formatPairs = (pairs: any[]) => (
    Array.isArray(pairs) && pairs.length
      ? pairs.map(item => `${item.name}×${item.count}`).join(' / ')
      : ''
  );
  const formatTags = (tags: any[]) => (
    Array.isArray(tags) && tags.length ? tags.join(' / ') : ''
  );
  const renderMetaRows = (meta: any) => {
    const taskSignature = meta.task_signature || {};
    const responseSignature = meta.response_signature || {};
    const timelineSignature = meta.timeline_signature || {};
    const evidenceSignature = meta.evidence_signature || {};
    const rows = [
      ['Trace Fingerprint', meta.trace_fingerprint],
      ['Session', meta.session_id],
      ['Turn', meta.turn_index],
      ['Agent', meta.agent_type],
      ['Model', meta.model],
      ['Provider', meta.provider],
      ['Task Tags', formatTags(taskSignature.tags || meta.task_tags)],
      ['Request Signature', taskSignature.request_hash ? `${fmtNum(Number(taskSignature.request_chars || 0))} chars · ${taskSignature.request_hash}` : ''],
      ['Response Signature', responseSignature.final_hash ? `${fmtNum(Number(responseSignature.final_chars || 0))} chars · ${responseSignature.final_hash}` : ''],
      ['Response Preview', responseSignature.final_preview],
      ['Tool Signature', formatPairs(meta.tool_signature)],
      ['Timeline', timelineSignature.preview || formatPairs(timelineSignature.types)],
      ['Failed Assertions', Array.isArray(evidenceSignature.failed_assertions) ? evidenceSignature.failed_assertions.join(' / ') : ''],
      ['Judge', meta.judge_status],
      ['Source', meta.source_event],
      ['Hook', meta.hook_source ? 'yes' : 'no'],
      ['Duration', fmtDuration(Number(meta.duration_ms))],
      ['Tokens', fmtNum(Number(meta.total_tokens || 0))],
      ['Tools', fmtNum(Number(meta.tool_count || 0))],
      ['Tool Errors', fmtNum(Number(meta.tool_error_count || 0))],
      ['User Query Chars', fmtNum(Number(meta.user_query_chars || 0))],
      ['User Query Hash', meta.user_query_hash],
      ['Sources', [meta.has_cot ? 'cot' : '', meta.has_transcript ? 'transcript' : '', meta.has_otel ? 'otel' : '', meta.has_overview ? 'overview' : ''].filter(Boolean).join(' / ')],
    ].filter(([, value]) => value !== undefined && value !== null && value !== '');
    return rows.map(([label, value]) => (
      <div className="ab-meta-row" key={String(label)}>
        <span>{label}</span>
        <strong>{String(value)}</strong>
      </div>
    ));
  };
  const renderTraceCard = (role: 'baseline' | 'candidate', ref: { session_id: string; turn_index: number } | null, data: any, meta: any) => (
    <div className={`ab-trace-card is-${role}`}>
      <div className="ab-trace-role">{role === 'baseline' ? 'Base' : '候选'}</div>
      <strong>{meta.trace_title || refLabel(ref)}</strong>
      <span>{[refLabel(ref), meta.trace_fingerprint ? `#${meta.trace_fingerprint}` : '', formatTags(meta.task_tags)].filter(Boolean).join(' · ') || 'metadata pending'}</span>
      <div className="ab-trace-stats">
        <em>{fmtPct(data.assertion_pass_rate)}</em>
        <em>{fmtNum(Number(data.metrics?.total_tokens || 0))} tokens</em>
        <em>{fmtNum(Number(data.metrics?.tool_count || 0))} tools</em>
      </div>
      <div className="ab-meta-popover">{renderMetaRows(meta)}</div>
    </div>
  );
  const renderCompareReport = () => {
    if (!llmCompare || Object.keys(llmCompare).length === 0) return null;
    const completed = llmCompare.status === 'completed';
    return (
      <div className={`ab-llm-report ${completed ? 'is-completed' : 'is-muted'}`}>
        <div className="ab-llm-head">
          <div>
            <span>LLM A/B 对比分析</span>
            <strong>{verdictText[llmCompare.comparison_verdict] || llmCompare.comparison_verdict || '对比报告'}</strong>
          </div>
          <em>{llmCompare.cache_hit ? '已复用' : (llmCompare.status || 'unknown')}{llmCompare.model ? ` · ${llmCompare.model}` : ''}</em>
        </div>
        <p className="ab-llm-summary">{llmCompare.summary_conclusion || llmCompare.reason}</p>
        {!completed && llmCompare.reason && <p className="ab-llm-warning">{llmCompare.reason}</p>}
        <div className="ab-llm-dimensions">
          <div className="ab-llm-dimension is-wide">
            <span>用户诉求覆盖情况</span>
            <p>{llmCompare.user_request_coverage || '该部分缺少可展示内容，请结合总体结论和下方确定性指标复核。'}</p>
          </div>
          {dimensions.map(key => {
            const item = llmCompare[key] || {};
            return (
              <div className="ab-llm-dimension" key={key}>
                <span>{dimensionLabels[key]} · {winnerText[item.winner] || item.winner || '不明确'}</span>
                <strong>{item.verdict || 'unclear'}</strong>
                <p>{item.review || `该维度缺少可展示内容，请结合总体结论和下方确定性指标复核。`}</p>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="ab-modal-backdrop" onClick={onClose}>
      <div className="ab-modal" onClick={e => e.stopPropagation()}>
        <div className="ab-modal-head">
          <div>
            <strong>A/B 评估对比</strong>
            <span>逐项比较 baseline 与 candidate 的断言、分组和资源消耗</span>
          </div>
          <button className="settings-close" onClick={onClose}>×</button>
        </div>

        {loading && (
          <div className="ab-modal-loading">
            <div className="loading-spinner" />
            <span>正在生成 A/B 对比报告...</span>
          </div>
        )}

        {!loading && result?.error && (
          <div className="ab-modal-error">{result.error}</div>
        )}

        {!loading && !result?.error && (
          <>
            <div className="ab-trace-identity">
              {renderTraceCard('baseline', baseline, baselineData, baselineMeta)}
              {renderTraceCard('candidate', candidate, candidateData, candidateMeta)}
            </div>

            <div className="ab-summary-grid">
              <div className="ab-summary-card">
                <span>Base Verdict</span>
                <strong>{baselineData.eval_panel?.overall_verdict || '已评估'}</strong>
                <em>{fmtPct(baselineData.assertion_pass_rate)}</em>
              </div>
              <div className="ab-summary-card">
                <span>候选 Verdict</span>
                <strong>{candidateData.eval_panel?.overall_verdict || '已评估'}</strong>
                <em>{fmtPct(candidateData.assertion_pass_rate)}</em>
              </div>
              <div className="ab-summary-card">
                <span>断言变化</span>
                <strong>{fmtPctDelta(summary.pass_rate_delta)}</strong>
                <em>候选 - Base</em>
              </div>
              <div className="ab-summary-card">
                <span>差异统计</span>
                <strong>{summary.improvement_count || 0} 项提升 / {summary.decline_count || 0} 项下降</strong>
                <em>{summary.changed_count || 0} 项发生变化</em>
              </div>
            </div>

            {renderCompareReport()}

            <div className="ab-chart-section">
              <div className="ab-section-title">断言分组通过率</div>
              <div className="ab-chart-legend">
                <span className="is-baseline">Base</span>
                <span className="is-candidate">候选</span>
              </div>
              <div className="ab-vertical-chart">
                {[...groupMap.values()].map(group => {
                  const basePct = Math.max(0, Math.min(1, Number(group.baseline || 0))) * 100;
                  const candPct = Math.max(0, Math.min(1, Number(group.candidate || 0))) * 100;
                  return (
                    <div className="ab-vbar-card" key={group.key}>
                      <div className="ab-vbar-plot">
                        <div className="ab-vbar">
                          <b className="is-baseline" style={{ height: `${basePct}%` }} />
                          <em>{fmtPct(group.baseline)}</em>
                        </div>
                        <div className="ab-vbar">
                          <b className="is-candidate" style={{ height: `${candPct}%` }} />
                          <em>{fmtPct(group.candidate)}</em>
                        </div>
                      </div>
                      <strong>{group.label || group.key}</strong>
                      <span>{fmtPct(group.baseline)} → {fmtPct(group.candidate)}</span>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="ab-chart-section">
              <div className="ab-section-title">Token 与工具使用</div>
              <div className="ab-table-head">
                <span>指标</span>
                <strong className="is-baseline">Base</strong>
                <strong className="is-candidate">候选</strong>
                <strong>变化</strong>
              </div>
              <div className="ab-metric-grid">
                {metricRows.map(([label, key]) => {
                  const baseValue = metricValue(baselineData, key);
                  const candValue = metricValue(candidateData, key);
                  const delta = baseValue === null || candValue === null ? null : candValue - baseValue;
                  return (
                    <div className="ab-metric-row" key={key}>
                      <span>{label}</span>
                      <strong>{fmtNum(baseValue)}</strong>
                      <strong>{fmtNum(candValue)}</strong>
                      <b className={delta === null ? '' : delta > 0 ? 'is-positive' : delta < 0 ? 'is-negative' : ''}>{fmtDelta(delta)}</b>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="ab-chart-section">
              <div className="ab-section-title">断言差异明细</div>
              <div className="ab-diff-head">
                <span>断言</span>
                <strong className="is-baseline">Base</strong>
                <strong className="is-candidate">候选</strong>
                <strong>变化</strong>
              </div>
              <div className="ab-diff-table">
                {topDiffs.map((row: any) => (
                  <div className="ab-diff-row" key={row.key}>
                    <div>
                      <strong>{row.label_zh || row.key}</strong>
                      <span>{row.category} · {row.severity}</span>
                    </div>
                    <em title={row.baseline_reason || ''} className={row.baseline_passed ? 'is-pass' : 'is-fail'}>{row.baseline_passed ? '通过' : '失败'}</em>
                    <em title={row.candidate_reason || ''} className={row.candidate_passed ? 'is-pass' : 'is-fail'}>{row.candidate_passed ? '通过' : '失败'}</em>
                    <b className={Number(row.delta || 0) > 0 ? 'is-pass' : Number(row.delta || 0) < 0 ? 'is-fail' : ''}>{fmtPctDelta(Number(row.delta || 0))}</b>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function HookHealthDialog({ onClose }: { onClose: () => void }) {
  const [health, setHealth] = useState<any | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api.getHookHealth()
      .then(setHealth)
      .catch((err: any) => setError(String(err?.response?.data?.detail || err?.message || 'Hook health check failed')))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const agents = Array.isArray(health?.agents) ? health.agents : [];
  const statusText = (status?: string) => {
    if (status === 'ok') return '正常';
    if (status === 'warning') return '需处理';
    if (status === 'skipped') return '未检测到';
    if (status === 'error') return '异常';
    return status || '未知';
  };
  const boolText = (value: any) => (value ? '是' : '否');
  const latestText = (item: any) => {
    const latest = item?.recent_activity?.latest;
    if (!latest) return '暂无 hook 运行记录';
    return `${latest.ts || '-'} · ${latest.event || 'event'}`;
  };

  return (
    <div className="settings-backdrop" onClick={onClose}>
      <div className="settings-dialog hook-health-dialog" onClick={e => e.stopPropagation()}>
        <div className="settings-head">
          <div>
            <strong>IDE Hook 检查</strong>
            <span className="hook-health-sub">检查 hook 配置、脚本写入、runtime 和最近触发记录</span>
          </div>
          <button className="settings-close" onClick={onClose}>×</button>
        </div>

        <div className="hook-health-toolbar">
          <div className={`hook-health-overall is-${health?.overall_status || 'unknown'}`}>
            总体状态：{statusText(health?.overall_status)}
          </div>
          <button className="btn-refresh" onClick={load} disabled={loading}>
            {loading ? '检查中...' : 'Retest'}
          </button>
        </div>

        {error && <div className="hook-health-error">{error}</div>}
        {!error && loading && !health && (
          <div className="hook-health-loading">
            <div className="loading-spinner" />
            <span>正在检查 IDE hook...</span>
          </div>
        )}

        {health && (
          <>
            <div className="hook-health-runtime">
              <div><span>runtime.json</span><strong>{health.runtime?.exists ? '已写入' : '缺失'}</strong></div>
              <div><span>Python/Exe</span><strong>{health.runtime?.python_exists ? '可用' : '不可用'}</strong></div>
              <div><span>Extractor</span><strong>{health.runtime?.cot_extractor_exists ? '可用' : '不可用'}</strong></div>
              <code title={health.runtime?.python_executable || ''}>{health.runtime?.python_executable || health.current_python}</code>
            </div>

            <div className="hook-health-list">
              {agents.map((agent: any) => (
                <div className={`hook-health-card is-${agent.status || 'unknown'}`} key={agent.agent}>
                  <div className="hook-health-card-head">
                    <div>
                      <strong>{agent.display_name || agent.agent}</strong>
                      <span>{agent.agent}</span>
                    </div>
                    <em>{statusText(agent.status)}</em>
                  </div>
                  {!agent.installed ? (
                    <div className="hook-health-muted">{agent.reason || agent.error || '本机未检测到该 IDE。'}</div>
                  ) : (
                    <>
                      <div className="hook-health-grid">
                        <div><span>配置激活</span><strong>{boolText(agent.activated)}</strong></div>
                        <div><span>Runtime</span><strong>{boolText(agent.runtime_ok)}</strong></div>
                        <div><span>最近触发</span><strong>{agent.recent_activity?.available ? '有记录' : '暂无'}</strong></div>
                      </div>
                      <div className="hook-health-recent">{latestText(agent)}</div>
                      {(agent.targets || []).map((target: any, idx: number) => (
                        <div className="hook-health-target" key={`${agent.agent}-${idx}`}>
                          <div><span>配置</span><code>{target.config_path}</code></div>
                          <div><span>脚本</span><code>{target.assets_dir}</code></div>
                          <div className="hook-health-target-flags">
                            <span>config: {target.config_exists && target.config_valid_json ? 'ok' : 'missing'}</span>
                            <span>assets: {target.assets_written ? 'ok' : `缺 ${target.missing_assets?.length || 0}`}</span>
                            <span>entries: {target.entries_active ? 'ok' : `缺 ${target.missing_entry_count || 0}`}</span>
                          </div>
                        </div>
                      ))}
                    </>
                  )}
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function SettingsDialog({ onClose }: { onClose: () => void }) {
  const [enabled, setEnabled] = useState(false);
  const [provider, setProvider] = useState('timiai');
  const [model, setModel] = useState('gpt-4o-mini');
  const [apiKey, setApiKey] = useState('');
  const [hasApiKey, setHasApiKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const models: Record<string, string[]> = {
    timiai: ['gpt-4o-mini', 'gpt-5.4', 'gpt-4o'],
    deepseek: ['deepseek-chat', 'deepseek-reasoner'],
  };

  useEffect(() => {
    api.getCriticSettings().then(data => {
      setEnabled(Boolean(data.enabled));
      setProvider(data.provider || 'timiai');
      setModel(data.model || 'gpt-4o-mini');
      setHasApiKey(Boolean(data.has_api_key));
    }).catch(() => {});
  }, []);

  const save = () => {
    setSaving(true);
    api.saveCriticSettings({ enabled, provider, model, api_key: apiKey || undefined })
      .then(data => {
        setEnabled(Boolean(data.enabled));
        setProvider(data.provider || provider);
        setModel(data.model || model);
        setHasApiKey(Boolean(data.has_api_key));
        setApiKey('');
        onClose();
      })
      .finally(() => setSaving(false));
  };

  return (
    <div className="settings-backdrop" onClick={onClose}>
      <div className="settings-dialog" onClick={e => e.stopPropagation()}>
        <div className="settings-head">
          <strong>Critic 模型设置</strong>
          <button className="settings-close" onClick={onClose}>×</button>
        </div>
        <label className="settings-toggle">
          <input type="checkbox" checked={enabled} onChange={e => setEnabled(e.target.checked)} />
          <span>启用 Agent Critic sidecar eval</span>
        </label>
        <div className="settings-grid">
          <label>
            <span>服务提供商</span>
            <select value={provider} onChange={e => {
              const next = e.target.value;
              setProvider(next);
              setModel(models[next]?.[0] || '');
            }}>
              <option value="timiai">TIMIAI</option>
              <option value="deepseek">DeepSeek</option>
            </select>
          </label>
          <label>
            <span>模型</span>
            <select value={model} onChange={e => setModel(e.target.value)}>
              {(models[provider] || []).map(x => <option value={x} key={x}>{x}</option>)}
            </select>
          </label>
        </div>
        <label className="settings-field">
          <span>API Key {hasApiKey && !apiKey ? '(已保存)' : ''}</span>
          <input type="password" value={apiKey} placeholder={hasApiKey ? '********' : '输入 API Key'} onChange={e => setApiKey(e.target.value)} />
        </label>
        <div className="settings-actions">
          <button className="btn-refresh" onClick={onClose}>取消</button>
          <button className="btn-refresh" onClick={save} disabled={saving}>{saving ? '保存中...' : '保存'}</button>
        </div>
      </div>
    </div>
  );
}
