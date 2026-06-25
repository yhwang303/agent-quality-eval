#!/usr/bin/env node

// Cursor 细粒度事件 -> events.jsonl 流式写入
//
// 目的: 打破"只靠 stop 后从 transcript 恢复"的局限，实时捕捉每次
// shell / MCP / file_edit / read / agent_thought / agent_response 的
// 触发与结果，给 cot_extractor 提供 transcript 抓不到的真实 stdout/stderr
// 以及毫秒级时间轴。
//
// 每次挂到 hooks.json 任一细粒度事件上，都会往
//   <DATA_ROOT>/events/<conversation_id>/events.jsonl
// 追加一行 JSON:
//   { "t": ms, "event": "afterShellExecution", "cid": ..., "tool": "Shell",
//     "payload": { ... 截断后的原始 payload ... } }
//
// 约束:
//   - 同步路径必须 < 100ms（否则 Cursor UI 会卡）
//   - BOM strip（Windows 下 Cursor stdin 带 UTF-8 BOM，见 cot-bridge.js Bug 4）
//   - 单字段最多保留 ~8KB；单条事件上限 ~20KB，防止大 stdout 爆盘
//   - 任何异常都必须兜底返回 {"continue":true}，不能阻塞 Cursor
//
// v0.18.5 路径修正：events.jsonl 写到 ``~/.agent-cot/data/events/<cid>/`` 而不是
// ``<COT_ROOT>/output/events/``。在 wheel 安装态下 COT_ROOT = site-packages 子目录，
// 通常对当前用户只读 / 即使能写 backend 与 cot_extractor 也是去 ``~/.agent-cot/data/``
// 找 —— 写读两端不一致是 v0.17 ~ v0.18.4 一直存在的隐性 bug，导致同事拿到的 trace
// 只剩 thinking、没有任何 Tool Decision / Tool Execution。COT_ROOT 仍然保留作 hook
// 拼 extract_cot.py 路径用，跟 events.jsonl 解耦。
//
// 环境变量:
//   AGENT_COT_DATA_ROOT — events.jsonl 父目录（agent-cot start 注入；不设也兜底 ~/.agent-cot/data）
//   COT_EXTRACTOR_ROOT   — cot-extractor 安装根（保留：只用于 hook 拼别的脚本路径，不影响 events 写盘）
//   COT_STREAM_ENABLED   — "false" 禁用 (默认启用)
//   COT_STREAM_MAX_FIELD — 单字段最大字符数 (默认 8192)
//   COT_STREAM_MAX_TOTAL — 单条事件最大字符数 (默认 20480)
//   COT_STREAM_LOG       — 自定义日志路径 (默认 ~/.cursor/cot-stream.log)

import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs';
import { spawn } from 'child_process';
import { dirname, join } from 'path';
import { homedir } from 'os';

const RAW_COT_ROOT = process.env.COT_EXTRACTOR_ROOT || '__AGENT_COT_EXTRACTOR_ROOT_UNCONFIGURED__';

// v0.18.15: cot-stream.js doesn't actually invoke COT_ROOT/scripts/* —
// it only writes events.jsonl. But we still log COT_ROOT for diagnostics
// + accept ~/.agent-cot/runtime.json fallback so a future feature that
// needs it can rely on the same self-healing chain. DATA_ROOT is the
// only path that *actually* matters for this script's correctness.

function readRuntimeState() {
  try {
    const f = join(homedir(), '.agent-cot', 'runtime.json');
    if (!existsSync(f)) return null;
    const raw = readFileSync(f, 'utf-8');
    if (!raw) return null;
    return JSON.parse(raw.charCodeAt(0) === 0xFEFF ? raw.slice(1) : raw);
  } catch { return null; }
}

const _runtimeState = readRuntimeState();

// COT_ROOT resolution: env > runtime.json > patched literal (we don't probe
// Python here — cot-stream is hot-path, called for every Cursor hook event;
// we must never spawn anything synchronously beyond the file write).
//
// v0.20.6: order reversed (was env > literal > runtime.json). runtime.json
// is dynamic — written by `agent-cot start` with the *currently installed*
// extractor — so it must outrank the install-time literal which can go
// stale (e.g. dev/editable install detected a sibling git checkout as the
// "newest" source at init time, then the bundle moved on a later wheel
// install but the JS literal kept pointing at the stale path). See
// cot-bridge.js::resolveCotRoot for the matching change + rationale.
let COT_ROOT = process.env.COT_EXTRACTOR_ROOT;
if (!COT_ROOT && _runtimeState && _runtimeState.cot_extractor_root) {
  COT_ROOT = _runtimeState.cot_extractor_root;
}
if (!COT_ROOT) {
  COT_ROOT = RAW_COT_ROOT;
}

// v0.18.5: 用户可写的数据根，跟 backend / cot_extractor / transcript_watcher 完全对齐。
// 默认 ~/.agent-cot/data，可被 agent-cot start 注入的 AGENT_COT_DATA_ROOT 覆盖。
// v0.18.15: also honor runtime.json data_root so the stream goes to the
// dashboard's view even when the user moved it via env on a different shell.
//
// v0.19.3: data_root 自愈 —— 见 codebuddy cot-stream-codebuddy.js 同名注释。
// 修正历史上 init 把 runtime.json.data_root 写成 ~/.agent-cot 的损坏状态，
// 让 hook 写盘位置和 backend 默认对齐，避免 SessionList 缺会话。
function _normalizeDataRoot(p) {
  if (typeof p !== 'string' || !p.trim()) return p;
  const segs = p.replace(/\\/g, '/').split('/').filter(Boolean);
  const last = segs[segs.length - 1] || '';
  if (last === 'data') return p;
  if (last === '.agent-cot' || last === '.cursor-cot') {
    return join(p, 'data');
  }
  return p;
}

const DATA_ROOT = process.env.AGENT_COT_DATA_ROOT
  || (_runtimeState && _normalizeDataRoot(_runtimeState.data_root))
  || join(homedir(), '.agent-cot', 'data');
const PYTHON = process.env.COT_PYTHON
  || (_runtimeState && _runtimeState.python_executable)
  || (process.platform === 'win32' ? 'python.exe' : 'python3');
const ENABLED = (process.env.COT_STREAM_ENABLED || 'true').toLowerCase() !== 'false';
const MAX_FIELD = parseInt(process.env.COT_STREAM_MAX_FIELD || '8192', 10);
const MAX_TOTAL = parseInt(process.env.COT_STREAM_MAX_TOTAL || '20480', 10);
const STREAM_LOG = process.env.COT_STREAM_LOG
  || join(homedir(), '.cursor', 'cot-stream.log');
const CRITIC_TRIGGER_EVENTS = new Set(['stop', 'Stop', 'afterAgentResponse', 'SessionEnd', 'StopFailure']);
const CRITIC_DEBOUNCE_FILE = join(DATA_ROOT, 'critic_debounce_cursor_stream.json');
const CRITIC_DEBOUNCE_WINDOW_MS = 30 * 1000;

// v0.20.0：同时镜像一条到统一的 pipeline.log（跟 cot-bridge / codebuddy / claude /
// extractor / backend 共享），方便一站式定位"事件落在哪一步丢的"。
const PIPELINE_LOG = process.env.AGENT_COT_PIPELINE_LOG
  || join(homedir(), '.agent-cot', 'logs', 'pipeline.log');

function isAgentQualityEvalExe(py) {
  const name = String(py || '').split(/[\\/]/).pop().toLowerCase();
  return name.startsWith('agent-quality-eval') && name.endsWith('.exe');
}

function agentQualityEvalRunnerArgs(kind, args) {
  if (isAgentQualityEvalExe(PYTHON)) {
    return ['--agent-quality-eval-runner', kind, ...args];
  }
  const code = kind === 'live-critic'
    ? 'from agent_quality_eval.evaluation.live_critic import main; raise SystemExit(main())'
    : 'from agent_quality_eval.evaluation.critic import main; raise SystemExit(main())';
  return ['-c', code, ...args];
}

function log(msg) {
  try {
    mkdirSync(dirname(STREAM_LOG), { recursive: true });
    appendFileSync(STREAM_LOG, `[${new Date().toISOString()}] ${msg}\n`);
  } catch {}
}

function pipelineLog(event, fields = {}) {
  try {
    mkdirSync(dirname(PIPELINE_LOG), { recursive: true });
    const parts = [];
    for (const [k, v] of Object.entries(fields)) {
      if (v === undefined || v === null) continue;
      let s = String(v);
      if (s.includes(' ') || s.includes('"')) s = `"${s.replace(/"/g, '\\"')}"`;
      parts.push(`${k}=${s}`);
    }
    const line =
      `[${new Date().toISOString()}] [hook.cursor-stream] [cursor] ` +
      `[sid=${fields.sid || '-'}] event=${event} status=${fields.ok === false ? 'FAIL' : 'ok'}` +
      (parts.length ? ` ${parts.join(' ')}` : '') + '\n';
    appendFileSync(PIPELINE_LOG, line);
  } catch {}
}

function readStdin() {
  return new Promise((ok, fail) => {
    let buf = '';
    process.stdin.setEncoding('utf-8');
    process.stdin.on('data', (c) => (buf += c));
    process.stdin.on('end', () => {
      try {
        if (buf.charCodeAt(0) === 0xFEFF) buf = buf.slice(1);
        ok(buf ? JSON.parse(buf) : {});
      } catch (e) { fail(e); }
    });
    process.stdin.on('error', fail);
  });
}

// 深拷贝并对超长字符串字段做截断；保留原结构便于后续分析。
// 过深递归会被硬截断；原始类型、数组、对象都支持。
function truncateDeep(value, depth = 0) {
  if (depth > 8) return '[truncated-depth]';
  if (value == null) return value;
  if (typeof value === 'string') {
    if (value.length <= MAX_FIELD) return value;
    return value.slice(0, MAX_FIELD) + `…[+${value.length - MAX_FIELD}ch]`;
  }
  if (typeof value !== 'object') return value;
  if (Array.isArray(value)) {
    const arr = [];
    const cap = Math.min(value.length, 200);
    for (let i = 0; i < cap; i++) arr.push(truncateDeep(value[i], depth + 1));
    if (value.length > cap) arr.push(`…[+${value.length - cap} items]`);
    return arr;
  }
  const out = {};
  for (const [k, v] of Object.entries(value)) {
    out[k] = truncateDeep(v, depth + 1);
  }
  return out;
}

// 从 payload 启发式抽工具名，方便后续 join 到 cot_extractor 的 tool_decision。
function pickToolName(event, payload) {
  if (!payload || typeof payload !== 'object') return '';
  const direct = payload.tool_name || payload.toolName || payload.name;
  if (direct) return String(direct);
  switch (event) {
    case 'beforeShellExecution':
    case 'afterShellExecution':
      return 'Shell';
    case 'beforeMCPExecution':
    case 'afterMCPExecution':
      return payload.server_name || payload.mcp_server || 'MCP';
    case 'afterFileEdit':
    case 'afterTabFileEdit':
      return 'Edit';
    case 'beforeReadFile':
    case 'beforeTabFileRead':
      return 'Read';
    case 'afterAgentThought':
      return 'Thought';
    case 'afterAgentResponse':
      return 'Response';
    default:
      return '';
  }
}

function liveCriticPayload(entry) {
  const out = {};
  for (const key of ['event', 'tool', 'tool_name', 'toolName', 'tool_use_id', 'cwd']) {
    if (entry && entry[key] !== undefined && entry[key] !== null) out[key] = entry[key];
  }
  return out;
}

function spawnLiveCritic(event, cid, entry) {
  if (!cid || process.env.AGENT_QUALITY_EVAL_LIVE_CRITIC_DISABLE) return;
  const payload = liveCriticPayload(entry);
  const args = agentQualityEvalRunnerArgs('live-critic', [
    '--agent-type', 'cursor',
    '--source-event', event,
    '--session-id', cid,
  ]);
  if (Object.keys(payload).length) {
    args.push('--payload-json', JSON.stringify(payload));
  }
  try {
    const child = spawn(PYTHON, args, {
      cwd: homedir(),
      env: {
        ...process.env,
        AGENT_COT_DATA_ROOT: DATA_ROOT,
        AGENT_QUALITY_EVAL_LIVE_CRITIC_DISABLE: '1',
        PYTHONIOENCODING: 'utf-8',
      },
      detached: true,
      stdio: 'ignore',
      windowsHide: true,
    });
    child.on('error', (err) => {
      log(`live critic spawn error event=${event} cid=${cid}: ${err.message}`);
      pipelineLog('live_critic_pulse', { sid: cid, ok: false, error: err.message });
    });
    child.unref();
    pipelineLog('live_critic_pulse', { sid: cid, source_event: event });
  } catch (e) {
    log(`live critic spawn fatal event=${event} cid=${cid}: ${e.message}`);
    pipelineLog('live_critic_pulse', { sid: cid, ok: false, error: e.message });
  }
}

function criticDebounceShouldRun(event, cid) {
  const now = Date.now();
  let state = {};
  try {
    if (existsSync(CRITIC_DEBOUNCE_FILE)) {
      const raw = readFileSync(CRITIC_DEBOUNCE_FILE, 'utf-8');
      state = raw ? JSON.parse(raw) : {};
    }
  } catch {
    state = {};
  }
  const key = `${cid}:${event}`;
  const last = state[key] || 0;
  if (now - last < CRITIC_DEBOUNCE_WINDOW_MS) return false;
  state[key] = now;
  try {
    mkdirSync(dirname(CRITIC_DEBOUNCE_FILE), { recursive: true });
    writeFileSync(CRITIC_DEBOUNCE_FILE, JSON.stringify(state), 'utf-8');
  } catch {}
  return true;
}

function maybeTriggerCritic(event, cid) {
  if (!CRITIC_TRIGGER_EVENTS.has(event)) return;
  if (!cid || process.env.AGENT_QUALITY_EVAL_CRITIC_DISABLE) return;
  if (!criticDebounceShouldRun(event, cid)) return;
  const args = agentQualityEvalRunnerArgs('critic', [
    '--agent-type', 'cursor',
    '--source-event', `cursor-stream:${event}`,
    '--session-id', cid,
    '--wait-seconds', '75',
    '--no-persist-eval',
  ]);
  try {
    const child = spawn(PYTHON, args, {
      cwd: homedir(),
      env: {
        ...process.env,
        AGENT_COT_DATA_ROOT: DATA_ROOT,
        AGENT_COT_PIPELINE_LOG: PIPELINE_LOG,
        AGENT_QUALITY_EVAL_CRITIC_DISABLE: '1',
        PYTHONIOENCODING: 'utf-8',
      },
      detached: true,
      stdio: 'ignore',
      windowsHide: true,
    });
    child.on('error', (err) => {
      log(`critic spawn error event=${event} cid=${cid}: ${err.message}`);
      pipelineLog('critic_spawn', { sid: cid, ok: false, error: err.message });
    });
    child.unref();
    pipelineLog('critic_spawn', { sid: cid, python: PYTHON, source_event: event });
  } catch (e) {
    log(`critic spawn fatal event=${event} cid=${cid}: ${e.message}`);
    pipelineLog('critic_spawn', { sid: cid, ok: false, error: e.message });
  }
}

function clip(s) {
  if (typeof s !== 'string') return s;
  return s.length > MAX_FIELD ? s.slice(0, MAX_FIELD) + `…[+${s.length - MAX_FIELD}ch]` : s;
}

// 把 MCP payload.tool_input（实际是 JSON 字符串）解析回对象，方便下游直接读字段。
function parseMaybeJson(v) {
  if (typeof v !== 'string') return v;
  const s = v.trim();
  if (!s) return undefined;
  if (s[0] !== '{' && s[0] !== '[') return v;
  try { return JSON.parse(s); } catch { return v; }
}

// 把 MCP afterMCPExecution.payload.result_json 拆成纯文本，便于前端直接当作
// "RAG 召回 / LLM 返回" 显示。result_json 形如：
//   {"content":[{"type":"text","text":"..."}],"isError":false}
// 也可能是 {"text":"..."} / {"output":"..."} / 直接字符串。
function flattenMcpResult(raw) {
  const v = parseMaybeJson(raw);
  if (v == null) return { text: '' };
  if (typeof v === 'string') return { text: v, is_error: false };
  if (typeof v !== 'object') return { text: String(v), is_error: false };
  let text = '';
  if (Array.isArray(v.content)) {
    text = v.content
      .map(c => (typeof c === 'string' ? c : (c && (c.text || c.value || c.data)) || ''))
      .filter(Boolean).join('\n');
  } else if (typeof v.text === 'string') text = v.text;
  else if (typeof v.output === 'string') text = v.output;
  else if (typeof v.result === 'string') text = v.result;
  else if (typeof v.message === 'string') text = v.message;
  if (!text) {
    try { text = JSON.stringify(v); } catch { text = String(v); }
  }
  return { text, is_error: !!v.isError || !!v.is_error };
}

// 启发式挑选"值得单独展出"的 tool_input 摘要，用于 cot_extractor 快速配对。
function pickBriefInput(event, payload) {
  if (!payload || typeof payload !== 'object') return undefined;
  const brief = {};
  if (payload.command) brief.command = String(payload.command).slice(0, 400);
  if (payload.cwd || payload.working_directory) brief.cwd = String(payload.cwd || payload.working_directory).slice(0, 300);
  if (payload.file_path || payload.filepath || payload.path) {
    brief.file_path = String(payload.file_path || payload.filepath || payload.path).slice(0, 300);
  }
  if (payload.url) brief.url = String(payload.url).slice(0, 300);
  if (payload.query) brief.query = String(payload.query).slice(0, 300);
  // MCP 事件：tool_input 是 JSON 字符串，把它解析后整体保留方便前端展示
  if (event === 'beforeMCPExecution' || event === 'afterMCPExecution') {
    if (payload.tool_input != null) {
      const parsed = parseMaybeJson(payload.tool_input);
      if (parsed !== undefined) brief.tool_input = parsed;
    }
    if (payload.tool_name) brief.tool_name = String(payload.tool_name);
    if (payload.command) brief.mcp_server = String(payload.command);
  }
  // 文件编辑事件：把 edits[] 摘要成 edits_count / added_lines / removed_lines，
  // 让 cot_extractor / 前端不用再回去解析 payload.edits 也能知道改了多少行。
  if (event === 'afterFileEdit' || event === 'afterTabFileEdit') {
    if (Array.isArray(payload.edits)) {
      let added = 0;
      let removed = 0;
      for (const ed of payload.edits) {
        if (!ed || typeof ed !== 'object') continue;
        const olds = typeof ed.old_string === 'string' ? ed.old_string : '';
        const news = typeof ed.new_string === 'string' ? ed.new_string : '';
        if (olds) removed += (olds.match(/\n/g) || []).length + (olds.endsWith('\n') ? 0 : 1);
        if (news) added += (news.match(/\n/g) || []).length + (news.endsWith('\n') ? 0 : 1);
      }
      brief.edits_count = payload.edits.length;
      brief.added_lines = added;
      brief.removed_lines = removed;
    }
    if (payload.generation_id) brief.generation_id = String(payload.generation_id);
    if (payload.model) brief.model = String(payload.model);
  }
  return Object.keys(brief).length ? brief : undefined;
}

// 启发式挑选"结果关键字段"，给 cot_extractor 当真实 tool_execution content 用。
// 实测 Cursor afterShellExecution payload 字段: command / output / duration / cwd (无 exit_code)
// 其中 output 是 stdout+stderr 的合流，不分开。
// MCP afterMCPExecution payload 字段: tool_name / tool_input / result_json / duration —— 真正
// 的"返回内容"在 result_json 字符串里 (MCP 协议形如 {"content":[{"type":"text","text":...}]}).
function pickBriefOutput(event, payload) {
  if (!payload || typeof payload !== 'object') return undefined;
  const brief = {};
  if (typeof payload.exit_code === 'number') brief.exit_code = payload.exit_code;
  if (typeof payload.exitCode === 'number') brief.exit_code = payload.exitCode;
  if (typeof payload.success === 'boolean') brief.success = payload.success;
  if (typeof payload.duration_ms === 'number') brief.duration_ms = payload.duration_ms;
  else if (typeof payload.duration === 'number') brief.duration_ms = payload.duration;
  const so = payload.stdout || payload.output || payload.result;
  if (typeof so === 'string' && so.length) brief.stdout = clip(so);
  const se = payload.stderr || payload.error;
  if (typeof se === 'string' && se.length) brief.stderr = clip(se);
  // MCP 真实结果在 result_json：拆 {content:[{text}]} → 平铺成 result_text 给前端直接展示
  if (event === 'afterMCPExecution' && payload.result_json) {
    const r = flattenMcpResult(payload.result_json);
    if (r.text) brief.result_text = clip(r.text);
    brief.is_error = r.is_error;
    brief.result_text_chars = (r.text || '').length;
  }
  return Object.keys(brief).length ? brief : undefined;
}

async function main() {
  // 与 cot-bridge.js 同理：先落盘一行，避免「stdin 永不 end」时零日志、误判 hook 未生效。
  // v0.18.5：把 DATA_ROOT 也打出来，方便排查"events 写到哪去了"。
  log(
    `process start pid=${process.pid} ENABLED=${ENABLED} ` +
    `DATA_ROOT=${DATA_ROOT} COT_ROOT=${COT_ROOT} ` +
    `runtimeStateLoaded=${_runtimeState ? 'yes' : 'no'}`,
  );

  if (!ENABLED) {
    process.stdout.write('{"continue":true}');
    return;
  }

  let payload;
  try {
    payload = await readStdin();
  } catch (e) {
    log(`stdin parse error: ${e.message}`);
    process.stdout.write('{"continue":true}');
    return;
  }

  const event = payload.hook_event_name || payload.event_name || 'unknown';
  const cid = payload.conversation_id || payload.sessionId || payload.session_id;
  if (!cid) {
    log(`skip event=${event}, no conversation_id`);
    process.stdout.write('{"continue":true}');
    return;
  }

  const entry = {
    t: Date.now(),
    event,
    cid,
    tool: pickToolName(event, payload),
    brief_input: pickBriefInput(event, payload),
    brief_output: pickBriefOutput(event, payload),
    payload: truncateDeep(payload),
  };

  // 最后兜底硬限：整条 JSON 不超过 MAX_TOTAL
  let line = JSON.stringify(entry);
  if (line.length > MAX_TOTAL) {
    entry.payload = '[truncated-total]';
    line = JSON.stringify(entry);
  }
  line += '\n';

  try {
    // v0.18.5: events.jsonl 写到 ``<DATA_ROOT>/events/<cid>/`` —— 跟
    // cot_extractor._load_cursor_events / transcript_watcher / backend 完全对齐。
    const dir = join(DATA_ROOT, 'events', cid);
    mkdirSync(dir, { recursive: true });
    appendFileSync(join(dir, 'events.jsonl'), line);
    pipelineLog(event, { sid: cid, tool: entry.tool, bytes: line.length });
  } catch (e) {
    log(`append error event=${event} cid=${cid}: ${e.message}`);
    pipelineLog(event, { sid: cid, ok: false, error: e.message });
  }
  spawnLiveCritic(event, cid, entry);
  maybeTriggerCritic(event, cid);

  process.stdout.write('{"continue":true}');
}

main().catch((err) => {
  log(`FATAL ${err.message}\n${err.stack}`);
  process.stdout.write('{"continue":true}');
  process.exit(0);
});
