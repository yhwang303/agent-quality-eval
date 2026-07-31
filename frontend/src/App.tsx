import { useState, useEffect, useRef, useCallback } from 'react';
import type { SessionOverview, SessionCoT, ResponseReport, TurnEvalReport, TurnCoT, EvalEvent } from './types';
import type { SelectedNode } from './components/SpanTree';
import { api } from './hooks/api';
import SessionList from './components/SessionList';
import SpanTree from './components/SpanTree';
import DetailPanel from './components/DetailPanel';
import OtelPanel from './components/OtelPanel';
import type { OtelSelectedNode } from './components/OtelPanel';
import './App.css';

type RightTab = 'detail' | 'otel';
type TurnRef = { session_id: string; turn_index: number };

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

function stableStringify(value: unknown): string {
  if (value == null || typeof value !== 'object') return JSON.stringify(value) ?? 'undefined';
  if (Array.isArray(value)) return `[${value.map(item => stableStringify(item)).join(',')}]`;
  const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b));
  return `{${entries.map(([key, val]) => `${JSON.stringify(key)}:${stableStringify(val)}`).join(',')}}`;
}

function stableTextHash(value: unknown): string {
  const text = typeof value === 'string' ? value : stableStringify(value ?? {});
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16);
}

export default function App() {
  const [sessions, setSessions] = useState<SessionOverview[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [cot, setCot] = useState<SessionCoT | null>(null);
  const [report, setReport] = useState<ResponseReport | null>(null);
  const [selectedNode, setSelectedNode] = useState<SelectedNode | null>(null);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [turnEvalReports, setTurnEvalReports] = useState<Record<string, TurnEvalReport>>({});
  const [liveCritic, setLiveCritic] = useState<any | null>(null);
  const [liveCriticTurns, setLiveCriticTurns] = useState<Record<string, any>>({});
  const [turnEvalLoadingKey, setTurnEvalLoadingKey] = useState<string | null>(null);
  const [turnEvalError, setTurnEvalError] = useState<string | null>(null);
  const [abBaseline, setAbBaseline] = useState<TurnRef | null>(null);
  const [abCandidate, setAbCandidate] = useState<TurnRef | null>(null);
  const [abResult, setAbResult] = useState<any | null>(null);
  const [abResultCache, setAbResultCache] = useState<Record<string, any>>({});
  const [abLoading, setAbLoading] = useState(false);
  const [abCompareOpen, setAbCompareOpen] = useState(false);
  const [abNotice, setAbNotice] = useState<string | null>(null);
  const [abMode, setAbMode] = useState<'ab' | 'regression'>('ab');
  const [pendingTurnJump, setPendingTurnJump] = useState<TurnRef | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(() => new URLSearchParams(window.location.search).get('settings') === '1');
  const [hookHealthOpen, setHookHealthOpen] = useState(false);
  const [uploadTraceOpen, setUploadTraceOpen] = useState(false);
  const [evalLogOpen, setEvalLogOpen] = useState(false);
  const [referenceEvalTurn, setReferenceEvalTurn] = useState<TurnRef | null>(null);
  const [regressionGoldOpen, setRegressionGoldOpen] = useState(false);
  const [regressionReference, setRegressionReference] = useState<any | null>(null);
  const settingsEventSeqRef = useRef<number | null>(null);
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

  useEffect(() => {
    let stopped = false;
    const checkUiEvents = () => {
      api.getUiEvents()
        .then(data => {
          if (stopped) return;
          const nextSeq = Number(data.settings_open_seq || 0);
          if (settingsEventSeqRef.current === null) {
            settingsEventSeqRef.current = nextSeq;
            return;
          }
          if (nextSeq > settingsEventSeqRef.current) {
            settingsEventSeqRef.current = nextSeq;
            setSettingsOpen(true);
          }
        })
        .catch(() => {});
    };
    checkUiEvents();
    const timer = window.setInterval(checkUiEvents, 800);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, []);

  // 加载 session 列表
  useEffect(() => {
    api.getSessions()
      .then(setSessions)
      .catch(() => setError('无法连接后端，请确认 API 服务已启动。'));
  }, []);

  // 手动刷新 session 列表：带 loading 态，避免大量 trace 时刷新看起来「卡死」
  const handleRefresh = useCallback(() => {
    setRefreshing(true);
    api.getSessions()
      .then(setSessions)
      .catch(() => {})
      .finally(() => setRefreshing(false));
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
    left: TurnRef | null,
    right: TurnRef | null,
  ) => Boolean(left && right && left.session_id === right.session_id && left.turn_index === right.turn_index);
  const abPairKey = (
    baselineRef: TurnRef | null,
    candidateRef: TurnRef | null,
  ) => {
    if (!baselineRef || !candidateRef) return '';
    const goldKey = abMode === 'regression' && regressionReference?.reference_answer
      ? `__gold:${stableTextHash(regressionReference.reference_answer)}`
      : '';
    // A/B is always blind now; baked into the key so the cache shape is stable
    // even if the toggle is reintroduced later.
    const blindKey = abMode === 'ab' ? '__blind' : '';
    return `${abMode}__${baselineRef.session_id}:${baselineRef.turn_index}__${candidateRef.session_id}:${candidateRef.turn_index}${goldKey}${blindKey}`;
  };
  const abRefLabel = (ref: TurnRef | null) => (
    ref ? `${ref.session_id.slice(0, 8)} · 第 ${ref.turn_index} 轮` : '未选择'
  );
  const selectTurnInCot = useCallback((ref: TurnRef, currentCot: SessionCoT) => {
    const turn = currentCot.turns.find(item => item.turn_index === ref.turn_index);
    if (!turn) return false;
    setRightTab('detail');
    setSelectedNode({ kind: 'turn', turn, cot: currentCot });
    return true;
  }, []);
  const jumpToAbRef = useCallback((ref: TurnRef | null, roleLabel: string) => {
    if (!ref) return;
    setRightTab('detail');
    if (cot?.session_id === ref.session_id && selectTurnInCot(ref, cot)) {
      setPendingTurnJump(null);
      setAbNotice(`已跳转到${roleLabel}：${abRefLabel(ref)}`);
      return;
    }
    setPendingTurnJump(ref);
    setSelectedId(ref.session_id);
    setAbNotice(`正在跳转到${roleLabel}：${abRefLabel(ref)}`);
  }, [cot, selectTurnInCot]);

  useEffect(() => {
    if (!pendingTurnJump || !cot || loading || cot.session_id !== pendingTurnJump.session_id) return;
    if (selectTurnInCot(pendingTurnJump, cot)) {
      setAbNotice(`已跳转到：${abRefLabel(pendingTurnJump)}`);
    } else {
      setAbNotice(`未找到目标 trace：${abRefLabel(pendingTurnJump)}`);
    }
    setPendingTurnJump(null);
  }, [pendingTurnJump, cot, loading, selectTurnInCot]);
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
  const handleReferenceEvalTurn = useCallback((turn: TurnCoT, currentCot: SessionCoT) => {
    setRightTab('detail');
    setSelectedNode({ kind: 'turn', turn, cot: currentCot });
    setReferenceEvalTurn({ session_id: currentCot.session_id, turn_index: turn.turn_index });
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
    setAbResult(null);
    setAbCompareOpen(true);
    api.compareTurns(
      abBaseline,
      abCandidate,
      abMode,
      abMode === 'regression' ? regressionReference?.reference_answer : undefined,
      abMode === 'ab',  // A/B is always blind; regression keeps its own gold-standard pathway.
    )
      .then(result => {
        setAbResult(result);
        if (result?.llm_compare?.status === 'completed' || result?.regression_compare?.status === 'completed') {
          setAbResultCache(prev => ({ ...prev, [pairKey]: result }));
        }
      })
      .catch((err: any) => setAbResult({ error: err?.response?.data?.detail || err?.message || 'Compare failed' }))
      .finally(() => setAbLoading(false));
  }, [abBaseline, abCandidate, abMode, abResultCache, regressionReference]);

  const restoreEvalEvent = useCallback((event: EvalEvent) => {
    const target = event.target || {};
    const mode = event.event_type === 'regression' ? 'regression' : 'ab';
    if (event.event_type === 'ab' || event.event_type === 'regression') {
      const baseline = target.baseline || {
        session_id: event.baseline_session_id,
        turn_index: event.baseline_turn_index,
      };
      const candidate = target.candidate || {
        session_id: event.candidate_session_id,
        turn_index: event.candidate_turn_index,
      };
      if (baseline?.session_id && candidate?.session_id) {
        setAbBaseline({ session_id: baseline.session_id, turn_index: Number(baseline.turn_index || 0) });
        setAbCandidate({ session_id: candidate.session_id, turn_index: Number(candidate.turn_index || 0) });
        setAbMode(mode);
        const summary = event.summary || {};
        setAbResult({
          from_eval_log: true,
          compare_mode: mode,
          baseline: { session_id: baseline.session_id, turn_index: Number(baseline.turn_index || 0) },
          candidate: { session_id: candidate.session_id, turn_index: Number(candidate.turn_index || 0) },
          summary,
          llm_compare: mode === 'ab' ? {
            status: 'saved',
            comparison_verdict: summary.comparison_verdict || event.verdict,
            summary_conclusion: summary.summary_conclusion,
          } : undefined,
          regression_gate: mode === 'regression' ? {
            verdict: summary.gate_verdict || event.verdict,
            blocking_reasons: summary.blocking_reasons || [],
            warning_reasons: summary.warning_reasons || [],
          } : undefined,
          regression_compare: mode === 'regression' ? {
            status: 'saved',
            gate_verdict: summary.gate_verdict || event.verdict,
            summary_conclusion: summary.summary_conclusion,
            blocking_reasons: summary.blocking_reasons || [],
            warning_reasons: summary.warning_reasons || [],
          } : undefined,
        });
        setAbCompareOpen(true);
      }
      return;
    }
    const sessionId = target.session_id || event.session_id;
    const turnIndex = target.turn_index ?? event.turn_index;
    if (sessionId && turnIndex != null) {
      jumpToAbRef({ session_id: sessionId, turn_index: Number(turnIndex) }, 'Eval Log');
    }
  }, [jumpToAbRef]);

  return (
    <div className="app">
      {/* 顶部标题栏 */}
      <header className="app-header">
        <div className="header-left">
          <img className="header-logo" src="/logo.png" alt="Agent Observation" width={26} height={26} />
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
                <button
                  className="ab-ref-jump"
                  disabled={!abBaseline}
                  onClick={() => jumpToAbRef(abBaseline, '基线')}
                  title={abBaseline ? `跳转到基线 trace：${abRefLabel(abBaseline)}` : '尚未选择基线'}
                >
                  基线：{abRefLabel(abBaseline)}
                </button>
                {abBaseline && <button className="ab-chip-clear" onClick={clearBaseline} title="清空基线">×</button>}
              </span>
              <span className={`ab-ref-chip ${abCandidate ? 'is-set is-candidate' : ''}`}>
                <button
                  className="ab-ref-jump"
                  disabled={!abCandidate}
                  onClick={() => jumpToAbRef(abCandidate, '候选')}
                  title={abCandidate ? `跳转到候选 trace：${abRefLabel(abCandidate)}` : '尚未选择候选'}
                >
                  候选：{abRefLabel(abCandidate)}
                </button>
                {abCandidate && <button className="ab-chip-clear" onClick={clearCandidate} title="清空候选">×</button>}
              </span>
              {abNotice && <span className="ab-notice">{abNotice}</span>}
            </div>
            <button className="btn-refresh ab-clear" disabled={!abBaseline && !abCandidate} onClick={clearAbSelection} title="清空基线和候选">
              清空
            </button>
            <label className={`ab-mode-toggle ${abMode === 'regression' ? 'is-regression' : ''}`} title="使用回归门禁视角检查 candidate 是否破坏 baseline 已具备的能力">
              <input
                type="checkbox"
                checked={abMode === 'regression'}
                onChange={e => {
                  setAbMode(e.target.checked ? 'regression' : 'ab');
                  setAbResult(null);
                }}
              />
              <span>回归检测</span>
            </label>
            {abMode === 'regression' && (
              <button
                className={`btn-refresh regression-gold ${regressionReference ? 'is-active' : ''}`}
                disabled={!abBaseline || !abCandidate}
                onClick={() => setRegressionGoldOpen(true)}
                title="为本次回归检测上传答案或评判资料"
              >
                {regressionReference ? '已绑定' : 'Gold'}
              </button>
            )}
            <button className={`btn-refresh ab-compare ${abBaseline && abCandidate ? 'is-ready' : ''}`} disabled={!abBaseline || !abCandidate || abLoading} onClick={compareAb}>
              {abLoading ? '对比中...' : abMode === 'regression' ? '回归检测' : 'A/B 对比'}
            </button>
          </div>
          <button className="btn-refresh" onClick={() => setSettingsOpen(true)}>
            API 设置
          </button>
          <button className="btn-refresh" onClick={() => setUploadTraceOpen(true)}>
            上传 Trace
          </button>
          <button className="btn-refresh" onClick={() => setHookHealthOpen(true)}>
            IDE Hook 检查
          </button>
          <button className="btn-refresh" onClick={() => setEvalLogOpen(true)}>
            Eval Log
          </button>
          <button
            className={`btn-refresh ${refreshing ? 'is-refreshing' : ''}`}
            onClick={handleRefresh}
            disabled={refreshing}
            title={refreshing ? '正在刷新 trace 列表…' : '刷新 trace 列表'}
          >
            {refreshing ? '⏳ 刷新中…' : '↺ 刷新'}
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
      {uploadTraceOpen && (
        <UploadTraceDialog
          onClose={() => setUploadTraceOpen(false)}
          onUploaded={(sessionId) => {
            setUploadTraceOpen(false);
            api.getSessions().then(list => {
              setSessions(list);
              const target = list.find(s => s.session_id === sessionId);
              if (target) setSelectedId(target.session_id);
            }).catch(() => {});
          }}
        />
      )}
      {evalLogOpen && (
        <EvalLogDialog
          sessions={sessions}
          onClose={() => setEvalLogOpen(false)}
          onSelect={restoreEvalEvent}
        />
      )}
      {referenceEvalTurn && (
        <ReferenceEvalDialog
          mode="turn"
          selectedTurn={referenceEvalTurn}
          onClose={() => setReferenceEvalTurn(null)}
        />
      )}
      {regressionGoldOpen && (
        <ReferenceEvalDialog
          mode="regression"
          selectedTurn={null}
          initialReference={regressionReference}
          onClose={() => setRegressionGoldOpen(false)}
          onSaveRegressionReference={data => {
            setRegressionReference(data);
            setAbResult(null);
            setRegressionGoldOpen(false);
          }}
          onClearRegressionReference={() => {
            setRegressionReference(null);
            setAbResult(null);
          }}
        />
      )}
      {abCompareOpen && (
        <AbCompareDialog
          result={abResult}
          loading={abLoading}
          baseline={abBaseline}
          candidate={abCandidate}
          compareMode={abMode}
          onRefresh={compareAb}
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
              onReferenceEvalTurn={handleReferenceEvalTurn}
              turnEvalReports={turnEvalReports}
              turnEvalLoadingKey={turnEvalLoadingKey}
              liveCritic={liveCritic}
              liveCriticTurns={liveCriticTurns}
              abBaseline={abBaseline}
              abCandidate={abCandidate}
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

function EvalLogDialog({
  onClose,
  onSelect,
}: {
  sessions: SessionOverview[];
  onClose: () => void;
  onSelect: (event: EvalEvent) => void;
}) {
  const [events, setEvents] = useState<EvalEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    setLoading(true);
    setError(null);
    api.listEvalEvents({ limit: 100 })
      .then(data => setEvents(data.events || []))
      .catch((err: any) => setError(err?.response?.data?.detail || err?.message || 'Failed to load eval log'))
      .finally(() => setLoading(false));
  }, []);

  const visibleEvents = expanded ? events : events.slice(0, 10);

  const eventTitle = (event: EvalEvent) => {
    if (event.event_type === 'ab') return `A/B ${event.winner || event.verdict || 'mixed'}`;
    if (event.event_type === 'regression') return `Regression ${event.summary?.gate_verdict || event.verdict || 'WARN'}`;
    if (event.event_type === 'reference') return `Reference ${event.verdict || '-'}`;
    if (event.event_type === 'gold') return 'Gold bound';
    return `Trace ${event.verdict || '-'}`;
  };
  const eventSummary = (event: EvalEvent) => {
    const summary = event.summary || {};
    return summary.summary_conclusion
      || summary.blocking_reasons?.[0]
      || summary.warning_reasons?.[0]
      || (summary.final_score != null ? `score ${(Number(summary.final_score) * 100).toFixed(0)}%` : '')
      || (summary.quality_delta != null ? `quality delta ${Number(summary.quality_delta).toFixed(3)}` : '')
      || '';
  };
  const eventTarget = (event: EvalEvent) => {
    const summary = event.summary || {};
    const baseline = summary.baseline || event.target?.baseline;
    const candidate = summary.candidate || event.target?.candidate;
    if (baseline?.session_id && candidate?.session_id) {
      return `${String(baseline.session_id).slice(0, 8)} #${baseline.turn_index} -> ${String(candidate.session_id).slice(0, 8)} #${candidate.turn_index}`;
    }
    const sessionId = event.session_id || event.target?.session_id;
    const turnIndex = event.turn_index ?? event.target?.turn_index;
    return sessionId ? `${String(sessionId).slice(0, 8)} #${turnIndex ?? '-'}` : '';
  };
  const formatTime = (iso: string) => {
    try {
      return new Date(iso).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
    } catch {
      return String(iso || '').slice(0, 16);
    }
  };
  const handleSelect = (event: EvalEvent) => {
    onSelect(event);
    onClose();
  };

  return (
    <div className="eval-log-backdrop" onClick={onClose}>
      <div className="eval-log-dialog" onClick={e => e.stopPropagation()}>
        <div className="eval-log-head">
          <div>
            <h2>Eval Log</h2>
            <p>Recent eval activity</p>
          </div>
          <button className="eval-log-close" onClick={onClose}>×</button>
        </div>
        {loading && <div className="eval-log-empty">Loading...</div>}
        {error && <div className="eval-log-empty">{error}</div>}
        {!loading && !error && events.length === 0 && <div className="eval-log-empty">No eval events yet.</div>}
        <div className="eval-log-list">
          {visibleEvents.map(event => (
            <button key={event.id} className={`eval-log-item eval-log-${event.event_type}`} onClick={() => handleSelect(event)}>
              <div className="eval-log-main">
                <span className="eval-log-type">{event.event_type}</span>
                <div>
                  <strong>{eventTitle(event)}</strong>
                  {eventSummary(event) && <p>{eventSummary(event)}</p>}
                </div>
                <time>{formatTime(event.created_at)}</time>
              </div>
              <div className="eval-log-meta">
                <span>{event.project_name || 'Unknown Project'}</span>
                {eventTarget(event) && <span>{eventTarget(event)}</span>}
                <span>{event.has_gold ? 'Gold' : 'No gold'}</span>
              </div>
            </button>
          ))}
        </div>
        {events.length > 10 && (
          <button className="eval-log-more" onClick={() => setExpanded(prev => !prev)}>
            {expanded ? '收起' : `展开全部 ${events.length} 条`}
          </button>
        )}
      </div>
    </div>
  );
}

function ReferenceEvalDialog({
  mode,
  selectedTurn,
  initialReference,
  onClose,
  onSaveRegressionReference,
  onClearRegressionReference,
}: {
  mode: 'turn' | 'regression';
  selectedTurn: { session_id: string; turn_index: number } | null;
  initialReference?: any | null;
  onClose: () => void;
  onSaveRegressionReference?: (data: any) => void;
  onClearRegressionReference?: () => void;
}) {
  const [saving, setSaving] = useState(false);
  const [loadingExisting, setLoadingExisting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<any | null>(null);
  const [preview, setPreview] = useState<any | null>(null);
  const [pendingPreview, setPendingPreview] = useState<any | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const isRegression = mode === 'regression';
  const busy = saving || loadingExisting;
  const extractPreview = (data: any) => data?.reference_answer || data?.case || null;

  useEffect(() => {
    let cancelled = false;
    setError(null);

    if (isRegression) {
      setSaved(initialReference || null);
      setPreview(extractPreview(initialReference));
      setPendingPreview(null);
      return () => {
        cancelled = true;
      };
    }

    if (!selectedTurn) {
      setSaved(null);
      setPreview(null);
      setPendingPreview(null);
      return () => {
        cancelled = true;
      };
    }

    setLoadingExisting(true);
    api.getTurnReferenceAnswer(selectedTurn.session_id, selectedTurn.turn_index)
      .then(data => {
        if (cancelled) return;
        const reference = data?.bound ? data.reference_answer : null;
        setSaved(reference);
        setPreview(extractPreview(reference));
        setPendingPreview(null);
      })
      .catch(() => {
        if (cancelled) return;
        setSaved(null);
        setPreview(null);
        setPendingPreview(null);
      })
      .finally(() => {
        if (!cancelled) setLoadingExisting(false);
      });

    return () => {
      cancelled = true;
    };
  }, [isRegression, selectedTurn?.session_id, selectedTurn?.turn_index, initialReference]);

  const handleUpload = async (file: File | undefined) => {
    if (!file) return;
    setSaving(true);
    setError(null);
    setSaved(null);
    setPreview(null);
    setPendingPreview(null);
    try {
      const content = await file.text();
      const target = !isRegression && selectedTurn
        ? { session_id: selectedTurn.session_id, turn_index: selectedTurn.turn_index }
        : undefined;
      const data = await api.normalizeReferenceAnswer(file.name, content, target);
      setPendingPreview(data);
      setPreview(extractPreview(data?.canonical || data));
    } catch (err: any) {
      setError(String(err?.response?.data?.detail || err?.message || '评测资料规整失败'));
    } finally {
      setSaving(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleConfirm = async () => {
    if (!pendingPreview) return;
    setSaving(true);
    setError(null);
    try {
      if (isRegression) {
        onSaveRegressionReference?.(pendingPreview);
        setSaved(pendingPreview);
      } else if (selectedTurn) {
        const data = await api.saveTurnReferenceAnswer(
          selectedTurn.session_id,
          selectedTurn.turn_index,
          String(pendingPreview.confirm_token || ''),
        );
        setSaved(data);
        setPreview(extractPreview(data));
      }
      setPendingPreview(null);
    } catch (err: any) {
      setError(String(err?.response?.data?.detail || err?.message || '评测资料确认失败'));
    } finally {
      setSaving(false);
    }
  };

  const handleClear = async () => {
    setSaving(true);
    setError(null);
    try {
      if (isRegression) {
        onClearRegressionReference?.();
      } else if (selectedTurn) {
        await api.deleteTurnReferenceAnswer(selectedTurn.session_id, selectedTurn.turn_index);
      }
      setSaved(null);
      setPreview(null);
      setPendingPreview(null);
    } catch (err: any) {
      setError(String(err?.response?.data?.detail || err?.message || '清除评测资料失败'));
    } finally {
      setSaving(false);
    }
  };

  const targetText = isRegression
    ? '回归检测'
    : (selectedTurn ? `${selectedTurn.session_id.slice(0, 8)} · 第 ${selectedTurn.turn_index} 轮` : '未选择 trace');
  const expected = String(preview?.expected_answer || '');
  const keywords = Array.isArray(preview?.keywords) ? preview.keywords : [];
  const mappings = Array.isArray(pendingPreview?.mapping) ? pendingPreview.mapping : [];
  const warnings = Array.isArray(pendingPreview?.warnings) ? pendingPreview.warnings : [];
  const rawContent = String(pendingPreview?.raw?.content || '');

  return (
    <div className="reference-modal-backdrop" onClick={onClose}>
        <div className={`reference-modal ${pendingPreview ? '' : 'reference-modal-compact'}`} onClick={e => e.stopPropagation()}>
        <div className="reference-modal-head">
          <div>
            <span>{targetText}</span>
            <strong>评测标准</strong>
            <p>{isRegression ? '上传现有答案或评判资料，用于本次回归检测。' : '上传现有答案或评判资料，系统会自动识别并规整。'}</p>
          </div>
          <button className="settings-close" onClick={onClose}>×</button>
        </div>

        <div className="reference-modal-body">
          <div className="reference-control-card">
            <div className="reference-control-top">
              <div>
                <strong>答案或评判资料</strong>
                <span>无需模板。支持 JSON / JSONL / CSV / YAML / Markdown / TXT。</span>
              </div>
              <button className="reference-upload-btn" disabled={busy || (!isRegression && !selectedTurn)} onClick={() => fileInputRef.current?.click()}>
                {saving ? '处理中...' : loadingExisting ? '读取中...' : '选择文件'}
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".json,.jsonl,.csv,.yaml,.yml,.md,.markdown,.txt"
                style={{ display: 'none' }}
                onChange={e => handleUpload(e.target.files?.[0])}
              />
            </div>
            {loadingExisting && <div className="reference-hint">正在读取这条 trace 已绑定的评测资料...</div>}
            {error && <div className="reference-error">{error}</div>}
            {saved && (
              <div className="reference-saved">
                <div>
                  <strong>已保存</strong>
                  <span>{isRegression ? '运行回归检测时会带上这份评测标准。' : '回到主界面点击 Eval，即可按这份标准评估。'}</span>
                </div>
                <button className="reference-clear-btn" disabled={busy} onClick={handleClear}>
                  清除绑定
                </button>
              </div>
            )}
            {pendingPreview && (
              <>
                <div className="reference-normalization-status">
                  <strong>{pendingPreview.eval_mode === 'gold' ? 'Gold 评测依据已识别' : '未识别 Gold 依据，将使用通用 Trace Eval'}</strong>
                </div>
                <div className="reference-preview-grid">
                  <div className="reference-preview">
                    <span>用户原始文件</span>
                    <pre>{rawContent}</pre>
                  </div>
                  <div className="reference-preview">
                    <span>规整后的标准结构</span>
                    <pre>{JSON.stringify(pendingPreview.canonical?.reference_answer || preview, null, 2)}</pre>
                  </div>
                </div>
                <div className="reference-mapping-panel">
                  <strong>字段映射</strong>
                  {mappings.length > 0
                    ? mappings.map((item: any, index: number) => (
                      <div key={`${item.source_path}-${index}`}>
                        <code>{item.source_path}</code>
                        <span>→</span>
                        <code>{item.canonical_field}</code>
                      </div>
                    ))
                    : <span>未进行字段映射；内容将按通用 Trace Eval 处理。</span>}
                </div>
                {warnings.length > 0 && (
                  <div className="reference-normalization-warnings">
                    {warnings.map((warning: string, index: number) => <span key={index}>{warning}</span>)}
                  </div>
                )}
                <button className="reference-run-btn" disabled={busy} onClick={handleConfirm}>
                  {isRegression ? '确认用于本次回归检测' : '确认并绑定到当前 Trace'}
                </button>
              </>
            )}
            {preview && !pendingPreview && (
              <div className="reference-preview">
                <span>{preview.title || preview.case_id || '评测标准'}</span>
                <pre>{expected || '未解析到标准答案文本，将使用其他已识别的评测依据。'}</pre>
                {keywords.length > 0 && <em>{keywords.slice(0, 8).join(' / ')}</em>}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function AbCompareDialog({
  result,
  loading,
  baseline,
  candidate,
  compareMode,
  onRefresh,
  onClose,
}: {
  result: any | null;
  loading: boolean;
  baseline: { session_id: string; turn_index: number } | null;
  candidate: { session_id: string; turn_index: number } | null;
  compareMode: 'ab' | 'regression';
  onRefresh?: () => void;
  onClose: () => void;
}) {
  const dimensionLabels: Record<string, string> = {
    task_completion: '任务完成',
    tool_use: '工具使用',
    reasoning: '推理路径',
    instruction_following: '指令遵循',
    workflow_adherence: '流程遵循',
    faithfulness: '忠实度',
    efficiency: '效率',
    reliability: '可靠性',
  };
  const assertionGroupLabels: Record<string, string> = {
    task_outcome: '任务结果',
    execution_integrity: '执行完整性',
    instruction_following: '指令遵循',
    workflow_adherence: '流程遵循',
    tool_use: '工具使用',
    code_delivery: '代码交付',
    research_grounding: '研究与证据',
    planning_execution: '计划执行',
    computer_use: 'GUI/浏览器操作',
    gold_process: 'Gold 过程要求',
    optional_judge: 'LLM 评审',
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
    // Always shown in minutes for consistency with the LLM-authored review
    // text and evidence quotes, which are normalized to minutes too.
    return `${(value / 60000).toFixed(1)}min`;
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
  const isRegressionMode = (result?.compare_mode || compareMode) === 'regression';
  const regressionGate = result?.regression_gate || {};
  const regressionCompare = result?.regression_compare || {};
  const diffs = Array.isArray(result?.diffs) ? result.diffs : [];
  const groupMap = new Map<string, any>();
  for (const group of baselineData.groups || []) {
    groupMap.set(group.key, { key: group.key, label: assertionGroupLabels[group.key] || group.label, baseline: group.pass_rate, candidate: null });
  }
  for (const group of candidateData.groups || []) {
    const row = groupMap.get(group.key) || { key: group.key, label: group.label, baseline: null, candidate: null };
    row.candidate = group.pass_rate;
    row.label = assertionGroupLabels[group.key] || group.label || row.label;
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
  const dimensionDecisionText = (item: any) => {
    const winner = String(item?.winner || '').toLowerCase();
    if (winner === 'baseline') return 'Base 更强';
    if (winner === 'candidate') return '候选更强';
    if (winner === 'tie') return '基本持平';
    const verdict = String(item?.verdict || '').toLowerCase();
    if (verdict === 'stronger') return '候选更强';
    if (verdict === 'weaker') return 'Base 更强';
    if (verdict === 'comparable') return '基本持平';
    if (verdict === 'mixed') return '各有优劣';
    return '不明确';
  };
  const regressionDimensionLabels: Record<string, string> = {
    capability_preservation: '能力保持',
    user_goal_coverage: '用户目标覆盖',
    instruction_obligation_regression: '指令约束保持',
    workflow_adherence_regression: '流程遵循退化',
    behavioral_change_risk: '行为变化风险',
    evidence_faithfulness: '证据忠实度',
    workflow_integrity: '过程稳定性',
    efficiency_regression: '效率退化',
  };
  const regressionDimensions = Object.keys(regressionDimensionLabels);
  const listOf = (value: any): string[] => Array.isArray(value) ? value.filter(Boolean).map(String) : [];
  const gateText: Record<string, string> = {
    PASS: '通过',
    WARN: '需复核',
    FAIL: '阻断',
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
    if (isRegressionMode) return null;
    if (!llmCompare || Object.keys(llmCompare).length === 0) return null;
    const completed = llmCompare.status === 'completed';
    return (
      <div className={`ab-llm-report ${completed ? 'is-completed' : 'is-muted'}`}>
        <div className="ab-llm-head">
          <div>
            <span>LLM A/B 对比分析 · 盲审</span>
            <strong>{verdictText[llmCompare.comparison_verdict] || llmCompare.comparison_verdict || '对比报告'}</strong>
          </div>
          <em>{llmCompare.cache_hit ? '已复用' : (llmCompare.status || 'unknown')}{llmCompare.model ? ` · ${llmCompare.model}` : ''}</em>
        </div>
        <div className="ab-conclusion-card">
          <span>高亮结论</span>
          <strong>{verdictText[llmCompare.comparison_verdict] || llmCompare.comparison_verdict || '对比结论'}</strong>
          <p>{llmCompare.summary_conclusion || llmCompare.reason || '当前没有可展示的 LLM 对比结论。'}</p>
        </div>
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
                <span>{dimensionLabels[key]} · {dimensionDecisionText(item)}</span>
                <strong>{dimensionDecisionText(item)}</strong>
                <p>{item.review || `该维度缺少可展示内容，请结合总体结论和下方确定性指标复核。`}</p>
                <AbEvidenceBlock
                  baseline={item.baseline_evidence}
                  candidate={item.candidate_evidence}
                />
              </div>
            );
          })}
        </div>
      </div>
    );
  };
  const renderRegressionReport = () => {
    if (!isRegressionMode) return null;
    const gateVerdict = String(regressionGate.verdict || regressionCompare.gate_verdict || 'WARN').toUpperCase();
    const blockingReasons = listOf(regressionGate.blocking_reasons);
    const warningReasons = listOf(regressionGate.warning_reasons);
    const newFailures = Array.isArray(regressionGate.new_failed_assertions) ? regressionGate.new_failed_assertions : [];
    const completed = regressionCompare.status === 'completed';
    return (
      <div className={`ab-llm-report regression-report regression-${gateVerdict.toLowerCase()} ${completed ? 'is-completed' : 'is-muted'}`}>
        <div className="ab-llm-head">
          <div>
            <span>回归检测</span>
            <strong>{gateText[gateVerdict] || gateVerdict}</strong>
          </div>
          <em>{regressionCompare.cache_hit ? '已复用' : (regressionCompare.status || 'deterministic')}{regressionCompare.model ? ` · ${regressionCompare.model}` : ''}</em>
        </div>
        <div className={`ab-conclusion-card regression-conclusion-${gateVerdict.toLowerCase()}`}>
          <span>高亮结论</span>
          <strong>{gateText[gateVerdict] || gateVerdict}</strong>
          <p>{regressionCompare.summary_conclusion || `回归检测结论：${gateText[gateVerdict] || gateVerdict}。`}</p>
        </div>
        <div className="regression-gate-grid">
          <div className="regression-gate-card">
            <span>阻断原因</span>
            {blockingReasons.length ? blockingReasons.map((item, idx) => <p key={idx}>{item}</p>) : <p>没有检测到阻断级回归。</p>}
          </div>
          <div className="regression-gate-card">
            <span>复核提醒</span>
            {warningReasons.length ? warningReasons.map((item, idx) => <p key={idx}>{item}</p>) : <p>没有检测到需要复核的回归风险。</p>}
          </div>
          <div className="regression-gate-card">
            <span>新增失败断言</span>
            {newFailures.length ? newFailures.slice(0, 8).map((item: any) => (
              <p key={item.key || item.label_zh}>{item.label_zh || item.label_en || item.key} · {item.severity || 'medium'}</p>
            )) : <p>没有 baseline 已通过但 candidate 失败的断言。</p>}
          </div>
          <div className="regression-gate-card">
            <span>保持的能力</span>
            {listOf(regressionCompare.preserved_capabilities).length ? listOf(regressionCompare.preserved_capabilities).map((item, idx) => <p key={idx}>{item}</p>) : (
              <p>保留了 {regressionGate.preserved_passed_assertions || 0} / {regressionGate.baseline_passed_assertions || 0} 个 baseline 已通过断言。</p>
            )}
          </div>
        </div>
        <div className="ab-llm-dimensions">
          {regressionDimensions.map(key => {
            const item = regressionCompare[key] || {};
            return (
              <div className="ab-llm-dimension" key={key}>
                <span>{regressionDimensionLabels[key]}</span>
                <strong>{item.verdict || 'unclear'}</strong>
                <p>{item.review || '该维度没有 LLM 评审内容，请结合确定性 gate 和下方断言差异复核。'}</p>
                <AbEvidenceBlock
                  baseline={item.baseline_evidence}
                  candidate={item.candidate_evidence}
                />
              </div>
            );
          })}
          <div className="ab-llm-dimension is-wide">
            <span>人工复核备注</span>
            <p>{listOf(regressionCompare.manual_review_notes).join(' / ') || '处理 WARN 或 FAIL 前，请先复核下方断言差异和 trace 证据。'}</p>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="ab-modal-backdrop" onClick={onClose}>
      <div className="ab-modal" onClick={e => e.stopPropagation()}>
        <div className="ab-modal-head">
          <div>
            <strong>{isRegressionMode ? '回归检测' : 'A/B 评估对比'}</strong>
            <span>{isRegressionMode ? '检查 candidate 是否破坏 baseline 已具备的能力、断言、trace 证据和资源表现。' : '逐项比较 baseline 与 candidate 的断言、分组和资源消耗'}</span>
          </div>
          <button className="settings-close" onClick={onClose}>×</button>
        </div>

        {loading && (
          <div className="ab-modal-loading">
            <div className="loading-spinner" />
            <span>{isRegressionMode ? '正在生成回归检测报告...' : '正在生成 A/B 对比报告...'}</span>
          </div>
        )}

        {!loading && result?.error && (
          <div className="ab-modal-error">{result.error}</div>
        )}

        {!loading && !result?.error && (
          <>
            {result?.from_eval_log && (
              <div className="ab-saved-log-note">
                <span>Saved Eval Log conclusion</span>
                <p>当前只展示日志里的结论级元数据；需要完整报告可重新调用 compare API。</p>
                <button className="btn-refresh" onClick={onRefresh}>重新生成完整报告</button>
              </div>
            )}
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
            {renderRegressionReport()}

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
              {summary?.assertion_patterns && (
                <AssertionPatternStrip patterns={summary.assertion_patterns} />
              )}
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
                      {row.pattern && <PatternBadge pattern={row.pattern} />}
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

const PATTERN_META: Record<string, { label: string; tone: string; hint: string }> = {
  non_discriminating: {
    label: '双侧通过',
    tone: 'mute',
    hint: 'Base 与候选都通过；该断言无法区分版本差异，可能不该用作决策依据。',
  },
  always_failing: {
    label: '双侧失败',
    tone: 'warn',
    hint: 'Base 与候选都失败；可能是断言本身坏掉，或任务超出当前能力上限。',
  },
  candidate_helps: {
    label: '候选改善',
    tone: 'pass',
    hint: 'Base 失败但候选通过——候选确实带来收益。',
  },
  candidate_hurts: {
    label: '候选退化',
    tone: 'fail',
    hint: 'Base 通过但候选失败——候选在这里帮倒忙。',
  },
  mixed: {
    label: '单侧缺失',
    tone: 'mute',
    hint: '两侧断言集合不一致，无法直接对比。',
  },
};

function PatternBadge({ pattern }: { pattern: string }) {
  const meta = PATTERN_META[pattern];
  if (!meta || pattern === 'mixed') return null;
  return (
    <span className={`ab-pattern-badge is-${meta.tone}`} title={meta.hint}>{meta.label}</span>
  );
}

function AssertionPatternStrip({ patterns }: { patterns: Record<string, number> }) {
  const total = Object.values(patterns || {}).reduce((acc, val) => acc + Number(val || 0), 0);
  if (!total) return null;
  const order = ['candidate_helps', 'candidate_hurts', 'non_discriminating', 'always_failing', 'mixed'];
  const visible = order
    .map(key => ({ key, count: Number(patterns?.[key] || 0), meta: PATTERN_META[key] }))
    .filter(item => item.count > 0 && item.meta);
  if (!visible.length) return null;
  return (
    <div className="ab-pattern-strip" role="list">
      {visible.map(item => (
        <div className={`ab-pattern-cell is-${item.meta.tone}`} role="listitem" key={item.key} title={item.meta.hint}>
          <span>{item.meta.label}</span>
          <strong>{item.count}</strong>
        </div>
      ))}
    </div>
  );
}

type AbEvidenceItem = { ref?: string; quote?: string; source?: string };

function AbEvidenceBlock({ baseline, candidate }: { baseline: any; candidate: any }) {
  const clean = (list: any): AbEvidenceItem[] => {
    if (!Array.isArray(list)) return [];
    return list.filter((e: any) => e && (e.ref || e.quote)).slice(0, 4);
  };
  const b = clean(baseline);
  const c = clean(candidate);
  if (!b.length && !c.length) return null;
  const renderList = (label: string, items: AbEvidenceItem[]) => (
    <div className="ab-evidence-side">
      <div className="ab-evidence-head">{label}</div>
      {items.length ? items.map((e, idx) => (
        <div className="dp-critic-evidence-item" key={idx}>
          {e.ref && <code className="dp-critic-evidence-ref">{e.ref}</code>}
          {e.source && <span className="dp-critic-evidence-source">{e.source}</span>}
          {e.quote && <span className="dp-critic-evidence-quote">{e.quote}</span>}
        </div>
      )) : <div className="ab-evidence-empty">无</div>}
    </div>
  );
  return (
    <div className="ab-evidence-block">
      {renderList('Base 证据', b)}
      {renderList('候选 证据', c)}
    </div>
  );
}

function UploadTraceDialog({ onClose, onUploaded }: { onClose: () => void; onUploaded: (sessionId: string) => void }) {
  const [title, setTitle] = useState<string>('');
  const [traceText, setTraceText] = useState<string>('');
  const [transcriptText, setTranscriptText] = useState<string>('');
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const onTraceFile = (file?: File | null) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setTraceText(String(reader.result || ''));
    reader.onerror = () => setError('读取文件失败');
    reader.readAsText(file);
  };
  const onTranscriptFile = (file?: File | null) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setTranscriptText(String(reader.result || ''));
    reader.onerror = () => setError('读取 transcript 失败');
    reader.readAsText(file);
  };

  const submit = async () => {
    setError(null);
    if (!traceText.trim()) {
      setError('请粘贴或上传 trace 内容');
      return;
    }
    let traceValue: any = traceText;
    try {
      traceValue = JSON.parse(traceText);
    } catch {
      // 非 JSON 也允许：后端 normalizer 会把它当作纯文本兜底为单 turn。
    }
    setBusy(true);
    try {
      const res = await api.uploadTrace({
        source: 'user-upload',
        title: title.trim() || undefined,
        trace: traceValue,
        transcript: transcriptText.trim() || undefined,
      });
      const sid = res?.session_id;
      if (sid) onUploaded(sid);
      else onClose();
    } catch (err: any) {
      setError(String(err?.response?.data?.detail || err?.message || '上传失败'));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="settings-backdrop" onClick={onClose}>
      <div className="settings-dialog upload-trace-dialog" onClick={e => e.stopPropagation()}>
        <div className="settings-head">
          <div>
            <strong>上传 Trace</strong>
            <span className="hook-health-sub">从外部来源导入一条 trace</span>
          </div>
          <button className="settings-close" onClick={onClose}>×</button>
        </div>
        <div className="upload-trace-body">
          <label className="upload-trace-label">
            <span>标题（可选）</span>
            <input value={title} placeholder="便于识别这条 trace" onChange={e => setTitle(e.target.value)} />
          </label>
          <label className="upload-trace-label">
            <span>Trace（必填，JSON 或文本）</span>
            <input type="file" accept=".json,.jsonl,.txt,.log" onChange={e => onTraceFile(e.target.files?.[0])} />
            <textarea
              value={traceText}
              onChange={e => setTraceText(e.target.value)}
              placeholder="粘贴 trace JSON 或文本…"
              rows={10}
            />
          </label>
          <label className="upload-trace-label">
            <span>Transcript（可选）</span>
            <input type="file" accept=".txt,.md,.log,.json" onChange={e => onTranscriptFile(e.target.files?.[0])} />
            <textarea
              value={transcriptText}
              onChange={e => setTranscriptText(e.target.value)}
              placeholder="可选：上传或粘贴对应的 transcript 文本"
              rows={6}
            />
          </label>
          {error && <div className="upload-trace-error">{error}</div>}
        </div>
        <div className="upload-trace-foot">
          <button className="btn-refresh" onClick={onClose} disabled={busy}>取消</button>
          <button className="btn-refresh is-active" onClick={submit} disabled={busy}>
            {busy ? '上传中…' : '上传并保存'}
          </button>
        </div>
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
    timiai: ['gpt-4o-mini', 'gpt-5.4', 'gpt-4o', 'glm-5.1', 'glm-5.2'],
    deepseek: ['deepseek-v4-flash', 'deepseek-v4-pro'],
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
