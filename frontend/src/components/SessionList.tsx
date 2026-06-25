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
// 一眼分辨当前 session 来自 Cursor / Claude / CodeBuddy / VSCode Copilot。
//
// 三家配色都走"低饱和度品牌色"——避免左侧列表满屏花花绿绿干扰阅读，
// 同时 hover / 选中态保持现有视觉规则不变。
//
// 颜色挑选规则：
//   * cursor    — 中性 slate 蓝（呼应 Cursor 黑白极简风）
//   * claude    — 浅 amber/terracotta（Anthropic 品牌琥珀的低饱和版）
//   * codebuddy — 柔和 teal（呼应 CodeBuddy 蓝绿主色，但调淡）
//   * vscode    — 浅 sky 蓝（VSCode 招牌色调暗）
const IDE_META: Record<string, { label: string; cls: string }> = {
  cursor:    { label: 'Cursor',      cls: 'ide-badge-cursor' },
  claude:    { label: 'Claude Code', cls: 'ide-badge-claude' },
  codex:     { label: 'Codex',       cls: 'ide-badge-codex' },
  codebuddy: { label: 'CodeBuddy',   cls: 'ide-badge-codebuddy' },
  vscode:    { label: 'VSCode',      cls: 'ide-badge-vscode' },
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

function fmtCostUsd(usd: number | null | undefined): string {
  if (usd == null) return '—';
  if (usd < 0.01) return `$${usd.toFixed(4)}`;
  if (usd < 1) return `$${usd.toFixed(3)}`;
  return `$${usd.toFixed(2)}`;
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
      {otel.cost_usd != null && otel.cost_usd > 0 && (
        <span
          className="session-otel-mini session-otel-mini-cost"
          title={
            otel.full_price_cost_usd && otel.full_price_cost_usd > otel.cost_usd
              ? `cache-aware 实际花费 ${fmtCostUsd(otel.cost_usd)}\n全价对比 ${fmtCostUsd(otel.full_price_cost_usd)}（cache 帮你省了 ${fmtCostUsd(otel.full_price_cost_usd - otel.cost_usd)}）`
              : `本会话累计花费（cache-aware）`
          }
        >
          💰 {fmtCostUsd(otel.cost_usd)}
        </span>
      )}
      {otel.cache_hit_rate != null && otel.cache_hit_rate > 0 && (
        <span
          className="session-otel-mini"
          title={`cache 命中率 = cache_read / (non_cache_in + cache_read + cache_write)\n输入 ${otel.input_tokens} tok · 输出 ${otel.output_tokens} tok\ncache_read ${otel.cache_read_tokens} · cache_write ${otel.cache_write_tokens}`}
        >
          ⚡ {Math.round(otel.cache_hit_rate * 100)}%
        </span>
      )}
    </div>
  );
}

export default function SessionList({ sessions, selectedId, onSelect, onDelete }: Props) {
  // ── v0.18.0：owner 筛选状态。默认只看本机（filter 模式）──
  const [ownerFilter, setOwnerFilter] = useState<OwnerFilter>('__local__');
  // 服务端有上行用户列表（分开拿，能看到『从未上传过的同事』也提示一下）
  const [, setRemoteOwners] = useState<string[]>([]);

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

  // 中央上行总数（用来在筛选器旁显示徽章）
  const centralCount = useMemo(
    () => sessions.reduce((acc, s) => acc + (s.owner ? 1 : 0), 0),
    [sessions],
  );
  const localCount = sessions.length - centralCount;

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
          {ownerFilter === '__local__' && '本机暂无 session，跟 Cursor 聊几句就会出现'}
          {ownerFilter === '__all__' && '暂无数据'}
          {ownerFilter !== '__local__' && ownerFilter !== '__all__' && `${ownerFilter} 暂无上行数据`}
        </div>
      )}

      {filtered.map(s => {
        const ideCls = s.agent_type ? `ide-card-${(s.agent_type || '').toLowerCase()}` : '';
        return (
        <div
          key={s.session_id}
          className={`session-item ${ideCls} ${selectedId === s.session_id ? 'selected' : ''} ${s.owner ? 'session-item-uplink' : ''}`}
          onClick={() => onSelect(s.session_id)}
        >
          {/* 主题 + owner 徽章 + 删除按钮 */}
          <div className="session-item-top">
            {s.owner && (
              <span
                className="owner-badge"
                style={{ backgroundColor: ownerColor(s.owner) }}
                title={`来自小组同事: ${s.owner}${s.host ? ' @ ' + s.host : ''}${s.received_at ? '\n收到时间: ' + s.received_at : ''}`}
              >
                {s.owner}
              </span>
            )}
            <span className="session-topic" title={s.topic}>{s.topic || '未知主题'}</span>
            <button
              className="btn-delete"
              onClick={(e) => handleDelete(e, s.session_id)}
              title={s.owner ? '删除中央服务上保存的这份副本（不影响同事的本地数据）' : '删除此 Session'}
            >✕</button>
          </div>
          {/* v0.20.7: IDE 来源徽章 + (可选)准确度 + 时间 */}
          <div className="session-item-row2">
            <IdeBadge agentType={s.agent_type} />
            <ScoreBadge score={s.response_score} />
            <span className="session-date">🕐 {formatDate(s.extracted_at)}</span>
          </div>
          {/* 统计数字 */}
          <div className="session-item-stats">
            <span title="Turns">🔄 {s.total_turns}</span>
            <span title="工具调用">🔧 {s.total_tool_calls}</span>
            <span title="复杂度">📊 {s.avg_complexity.toFixed(2)}</span>
          </div>
          {/* v0.11.2：OTel 自动检测的 model + cost + cache hit（来自 cot-stream events.jsonl） */}
          {s.otel && <OtelMiniRow otel={s.otel} />}
          {/* 工具分布 */}
          <div className="session-item-tools">
            {Object.entries(s.tool_call_distribution).map(([tool, count]) => (
              <span key={tool} className="tool-tag">{tool}×{count}</span>
            ))}
          </div>
        </div>
        );
      })}
    </div>
  );
}
