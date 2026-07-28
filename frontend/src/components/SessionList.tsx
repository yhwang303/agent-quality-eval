import { useEffect, useMemo, useState } from 'react';
import type { SessionOverview, SessionOtelKpi } from '../types';
import { api } from '../hooks/api';

interface Props {
  sessions: SessionOverview[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

// v0.18.0：中央 dashboard 的特殊筛选值
//   '__local__' = 只看本机
//   '__all__'   = 所有用户
//   <user_id>   = 只看某个同事
type OwnerFilter = '__local__' | '__all__' | string;
const PROJECT_COLLAPSE_KEY = 'agent-observation.project-collapsed.v1';
const PROJECT_EXPANDED_KEY = 'agent-observation.project-expanded.v1';
const PROJECT_VISIBLE_LIMIT = 5;

interface ProjectGroup {
  id: string;
  name: string;
  path: string;
  latest: string;
  sessions: SessionOverview[];
}

// 给每个 owner 生成一个稳定的颜色（hash 化），让多用户场景下行内徽章好辨识
function ownerColor(name: string | null | undefined): string {
  if (!name) return '#888';
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) | 0;
  // 限制到一组好看的色盘，而不是整个 HSL 空间
  const palette = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];
  return palette[Math.abs(h) % palette.length];
}

function formatDate(iso: string): string {
  if (!iso) return '-';
  try {
    const d = new Date(iso);
    return d.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
  } catch {
    return iso.slice(0, 16);
  }
}

function ScoreBadge({ score }: { score: number | null }) {
  // v0.20.7: 「无报告」徽章已下线 —— response-verifier 功能已废弃，留空即可。
  if (score === null) return null;
  const cls = score >= 0.8 ? 'badge-green' : score >= 0.6 ? 'badge-yellow' : 'badge-red';
  return <span className={`badge ${cls}`}>{(score * 100).toFixed(0)}%</span>;
}

// v0.20.7: IDE 来源徽章 —— 替代废弃的「无报告」徽章占位，让用户在 SessionList
// 一眼分辨当前 session 来自 Cursor / Claude / Codex / CodeBuddy。
//
// 三家配色都走"低饱和度品牌色"——避免左侧列表满屏花花绿绿干扰阅读，
// 同时 hover / 选中态保持现有视觉规则不变。
//
// 颜色挑选规则：
//   * cursor    — 中性 slate 蓝（呼应 Cursor 黑白极简风）
//   * claude    — 浅 amber/terracotta（Anthropic 品牌琥珀的低饱和版）
//   * codebuddy — 柔和 teal（呼应 CodeBuddy 蓝绿主色，但调淡）
const IDE_META: Record<string, { label: string; cls: string }> = {
  cursor:    { label: 'Cursor',      cls: 'ide-badge-cursor' },
  claude:    { label: 'Claude Code', cls: 'ide-badge-claude' },
  codex:     { label: 'Codex',       cls: 'ide-badge-codex' },
  codebuddy: { label: 'CodeBuddy',   cls: 'ide-badge-codebuddy' },
};

function IdeBadge({ agentType }: { agentType?: string | null }) {
  const meta = IDE_META[(agentType || '').toLowerCase()];
  if (!meta) {
    return <span className="ide-badge ide-badge-unknown">{agentType || '未知'}</span>;
  }
  return <span className={`ide-badge ${meta.cls}`} title={`Session 来源: ${meta.label}`}>{meta.label}</span>;
}

// v0.11.2：把 model 名简化成行内可读的 mini-tag，例如 'claude-opus-4-7' → 'opus-4'
function shortModel(m?: string | null): string {
  if (!m || m === 'unknown') return 'unknown';
  // claude-opus-4-7 / claude-4.6-sonnet / gpt-5.5-medium 等
  return m
    .replace(/^claude-/, 'cla-')
    .replace(/^gpt-/, 'gpt-')
    .replace(/-medium-thinking|-thinking-xhigh|-medium$/, '')
    .slice(0, 18);
}

function readBoolMap(key: string): Record<string, boolean> {
  try {
    return JSON.parse(localStorage.getItem(key) || '{}') || {};
  } catch {
    return {};
  }
}

function writeBoolMap(key: string, value: Record<string, boolean>) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Ignore storage failures; the tree still works for the current render.
  }
}

function sessionProjectId(s: SessionOverview): string {
  // The backend forces project_id=__uploaded__ for user-uploaded traces. We
  // honor that exact id here so the sidebar can pin the group to the top and
  // style it differently without re-deriving from the project name.
  if (s.project_id === '__uploaded__') return '__uploaded__';
  const name = sessionProjectName(s);
  return `project:${name.trim().toLowerCase() || 'unknown-project'}`;
}

function sessionProjectName(s: SessionOverview): string {
  if (s.project_id === '__uploaded__') return s.project_name || 'Uploaded Traces';
  const raw = s.project_name || (s.project_path ? s.project_path.split(/[\\/]/).filter(Boolean).pop() : '') || 'Unknown Project';
  return normalizeProjectName(raw);
}

function normalizeProjectName(value: string): string {
  let text = String(value || '').trim();
  if (!text) return 'Unknown Project';
  text = text.replace(/\\/g, '/').split('/').filter(Boolean).pop() || text;
  text = text.replace(/^[A-Za-z]-+/, '').replace(/^-+/, '');
  if (text.includes('--')) {
    const parts = text.split('--').filter(Boolean);
    text = parts[parts.length - 1] || text;
  }
  return text || 'Unknown Project';
}

function OtelMiniRow({ otel }: { otel: SessionOtelKpi }) {
  const isUnknown = !otel.model || otel.model === 'unknown';
  return (
    <div className="session-otel-line">
      <span
        className={`session-otel-mini ${isUnknown ? 'session-otel-mini-warn' : ''}`}
        title={
          isUnknown
            ? `自动检测未拿到 model（未启用 cot-stream.js hook 或本次没有 LLM 调用）`
            : `model = ${otel.model}\nsource = ${otel.model_source || '-'}\nprovider = ${otel.provider || '-'}`
        }
      >
        🤖 {shortModel(otel.model)}
      </span>
    </div>
  );
}

export default function SessionList({ sessions, selectedId, onSelect, onDelete }: Props) {
  // ── v0.18.0：owner 筛选状态。默认只看本机（filter 模式）──
  const [ownerFilter, setOwnerFilter] = useState<OwnerFilter>('__local__');
  // 服务端有上行用户列表（分开拿，能看到『从未上传过的同事』也提示一下）
  const [, setRemoteOwners] = useState<string[]>([]);
  const [collapsedProjects, setCollapsedProjects] = useState<Record<string, boolean>>(() => readBoolMap(PROJECT_COLLAPSE_KEY));
  const [expandedProjects, setExpandedProjects] = useState<Record<string, boolean>>(() => readBoolMap(PROJECT_EXPANDED_KEY));

  // 从当前 sessions 列表里推断出现过的 owner，再 union 服务端返回的 user 列表
  const localOwners = useMemo(() => {
    const set = new Set<string>();
    for (const s of sessions) {
      if (s.owner) set.add(s.owner);
    }
    return Array.from(set).sort();
  }, [sessions]);

  // 后端 /api/uplink/users 拉一次，与本地 sessions 推断到的 owner 求并集
  useEffect(() => {
    let cancelled = false;
    api.listUplinkUsers().then(r => {
      if (cancelled) return;
      setRemoteOwners((r.users || []).map(u => u.user_id));
    }).catch(() => { /* 中央服务自检失败不影响本机使用 */ });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    writeBoolMap(PROJECT_COLLAPSE_KEY, collapsedProjects);
  }, [collapsedProjects]);

  useEffect(() => {
    writeBoolMap(PROJECT_EXPANDED_KEY, expandedProjects);
  }, [expandedProjects]);

  const allOwners = useMemo(() => {
    const set = new Set<string>(localOwners);
    return Array.from(set).sort();
  }, [localOwners]);

  // 根据筛选条件过滤 sessions
  const filtered = useMemo(() => {
    if (ownerFilter === '__all__') return sessions;
    if (ownerFilter === '__local__') return sessions.filter(s => !s.owner);
    return sessions.filter(s => s.owner === ownerFilter);
  }, [sessions, ownerFilter]);

  const projectGroups = useMemo<ProjectGroup[]>(() => {
    const map = new Map<string, ProjectGroup>();
    for (const session of filtered) {
      const id = sessionProjectId(session);
      let group = map.get(id);
      if (!group) {
        group = {
          id,
          name: sessionProjectName(session),
          path: session.project_path || '',
          latest: session.extracted_at,
          sessions: [],
        };
        map.set(id, group);
      }
      group.sessions.push(session);
      if (!group.path && session.project_path) {
        group.path = session.project_path;
      }
      if (Date.parse(session.extracted_at || '') > Date.parse(group.latest || '')) {
        group.latest = session.extracted_at;
      }
    }
    return Array.from(map.values()).sort((a, b) => Date.parse(b.latest || '') - Date.parse(a.latest || ''));
  }, [filtered]);

  useEffect(() => {
    if (!selectedId) return;
    const selected = sessions.find(s => s.session_id === selectedId);
    if (!selected) return;
    const projectId = sessionProjectId(selected);
    setCollapsedProjects(prev => prev[projectId] ? { ...prev, [projectId]: false } : prev);
  }, [selectedId, sessions]);

  // 中央上行总数（用来在筛选器旁显示徽章）
  const centralCount = useMemo(
    () => sessions.reduce((acc, s) => acc + (s.owner ? 1 : 0), 0),
    [sessions],
  );
  const localCount = sessions.length - centralCount;

  const toggleProject = (projectId: string) => {
    setCollapsedProjects(prev => ({ ...prev, [projectId]: !(prev[projectId] ?? true) }));
  };

  const toggleProjectExpanded = (projectId: string) => {
    setExpandedProjects(prev => ({ ...prev, [projectId]: !prev[projectId] }));
  };

  const renderSessionItem = (s: SessionOverview) => {
    const ideCls = s.agent_type ? `ide-card-${(s.agent_type || '').toLowerCase()}` : '';
    return (
      <div
        key={s.session_id}
        className={`session-item session-item-compact ${ideCls} ${selectedId === s.session_id ? 'selected' : ''} ${s.owner ? 'session-item-uplink' : ''}`}
        onClick={() => onSelect(s.session_id)}
      >
        <div className="session-item-top">
          {s.owner && (
            <span
              className="owner-badge"
              style={{ backgroundColor: ownerColor(s.owner) }}
              title={`来自同事: ${s.owner}${s.host ? ' @ ' + s.host : ''}${s.received_at ? '\n收到时间: ' + s.received_at : ''}`}
            >
              {s.owner}
            </span>
          )}
          <span className="session-topic" title={s.topic}>{s.topic || '未知主题'}</span>
          <button
            className="btn-delete"
            onClick={(e) => handleDelete(e, s.session_id)}
            title={s.owner ? '删除中央服务保存的这份副本' : '删除此 Session'}
          >×</button>
        </div>
        <div className="session-item-row2">
          <IdeBadge agentType={s.agent_type} />
          <ScoreBadge score={s.response_score} />
          <span className="session-date">🕐 {formatDate(s.extracted_at)}</span>
        </div>
        <div className="session-item-stats">
          <span title="Turns">🔁 {s.total_turns}</span>
          <span title="工具调用">🔧 {s.total_tool_calls}</span>
        </div>
        {s.otel && <OtelMiniRow otel={s.otel} />}
        <div className="session-item-tools">
          {Object.entries(s.tool_call_distribution).map(([tool, count]) => (
            <span key={tool} className="tool-tag">{tool}×{count}</span>
          ))}
        </div>
      </div>
    );
  };

  const handleDelete = (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation();
    if (!confirm(`确认删除该 Session 的所有数据？\n${sessionId}`)) return;
    api.deleteSession(sessionId)
      .then(() => onDelete(sessionId))
      .catch(() => alert('删除失败，请检查后端服务'));
  };

  return (
    <div className="session-list">
      <div className="session-list-header">
        Sessions <span className="session-count">{filtered.length}</span>
        {centralCount > 0 && (
          <span
            className="session-count session-count-uplink"
            title={`本机 ${localCount} · 来自小组同事 ${centralCount}`}
          >
            ↑{centralCount}
          </span>
        )}
      </div>

      {/* v0.18.0：owner 筛选下拉 —— 中央 dashboard 收到过同事数据时才显示 */}
      {(centralCount > 0 || allOwners.length > 0) && (
        <div className="session-owner-filter">
          <label className="session-owner-filter-label" htmlFor="owner-filter">视角:</label>
          <select
            id="owner-filter"
            className="session-owner-filter-select"
            value={ownerFilter}
            onChange={(e) => setOwnerFilter(e.target.value as OwnerFilter)}
          >
            <option value="__local__">🏠 本机 ({localCount})</option>
            <option value="__all__">👥 全部 ({sessions.length})</option>
            {allOwners.map(o => {
              const cnt = sessions.filter(s => s.owner === o).length;
              return <option key={o} value={o}>👤 {o} ({cnt})</option>;
            })}
          </select>
        </div>
      )}

      {filtered.length === 0 && (
        <div className="session-empty">
          {ownerFilter === '__local__' && '本机暂无 session，跟 Agent 聊几句就会出现'}
          {ownerFilter === '__all__' && '暂无数据'}
          {ownerFilter !== '__local__' && ownerFilter !== '__all__' && `${ownerFilter} 暂无上行数据`}
        </div>
      )}

      {projectGroups.map(group => {
        const collapsed = collapsedProjects[group.id] ?? true;
        const expanded = !!expandedProjects[group.id];
        let visible = expanded ? group.sessions : group.sessions.slice(0, PROJECT_VISIBLE_LIMIT);
        const selectedHidden = group.sessions.find(s => s.session_id === selectedId && !visible.some(v => v.session_id === selectedId));
        if (selectedHidden) visible = [...visible, selectedHidden];
        const hiddenCount = Math.max(0, group.sessions.length - visible.length);
        const isUploaded = group.id === '__uploaded__';
        return (
          <div key={group.id} className={`session-project ${collapsed ? 'is-collapsed' : ''} ${isUploaded ? 'is-uploaded' : ''}`}>
            <button
              className="session-project-head"
              onClick={() => toggleProject(group.id)}
              title={group.path || group.name}
            >
              <span className="session-project-caret">{collapsed ? '▸' : '▾'}</span>
              <span className="session-project-icon">{isUploaded ? '⬆' : '📁'}</span>
              <span className="session-project-name">{group.name}</span>
              <span className="session-project-meta">{group.sessions.length}</span>
              <span className="session-project-time">{formatDate(group.latest)}</span>
            </button>
            {!collapsed && (
              <>
                <div className="session-project-items">
                  {visible.map(renderSessionItem)}
                </div>
                {group.sessions.length > PROJECT_VISIBLE_LIMIT && (
                  <button
                    className="session-project-more"
                    onClick={() => toggleProjectExpanded(group.id)}
                  >
                    {expanded ? '收起显示' : `展开显示 ${hiddenCount} 条`}
                  </button>
                )}
              </>
            )}
          </div>
        );
      })}
    </div>
  );
}
