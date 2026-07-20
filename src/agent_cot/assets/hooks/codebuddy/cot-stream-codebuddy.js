#!/usr/bin/env node

// CodeBuddy IDE 事件 → events.jsonl 流式写入
//
// 配置位置：~/.codebuddy/settings.json
// 事件命名：PascalCase（PreToolUse / PostToolUse / Stop / SessionEnd / ...）
// payload 字段：session_id / message_id / model / hook_event_name /
//   agent_version / workspace / user_name / transcript_path / ...
//
// 落盘路径：<DATA_ROOT>/events/codebuddy-<sid>/events.jsonl
//   DATA_ROOT 默认 ~/.agent-cot/data；agent-cot start 会注入 AGENT_COT_DATA_ROOT
// 加 ``codebuddy-`` 前缀避免和 Cursor / Claude 同 sid 撞车。
//
// agent-cot v0.18.6：跟 cursor 的 cot-stream.js 同款修复。
//   v0.18.5 之前 cot-stream-codebuddy.js 把 events.jsonl 写到
//   ``<COT_EXTRACTOR_ROOT>/output/events/``，wheel 安装态下 COT_EXTRACTOR_ROOT
//   要么是 site-packages 子目录（普通用户只读），要么 fallback 到原作者本机
//   ``D:/ai-ide-langfuse/cot-extractor``（其他用户根本不存在），结果同事跑
//   codebuddy 整条事件链全部静默丢失，前端永远只能看到 thinking 没有 tool。
//   v0.18.6 跟 cursor 一样改写到用户可写的 ``~/.agent-cot/data/events/`` 下，
//   backend / cot_extractor 的读路径也对齐到这里。
//
// agent-cot v0.19.0：再叠一层 ``~/.agent-cot/runtime.json`` 兜底。
//   场景：服务式启动的 shell 没继承用户 env（HOME 未导出 / AGENT_COT_DATA_ROOT
//   被外层 unset），homedir() 会算到一个奇怪的目录，DATA_ROOT 跟 backend 读取的
//   不在同一个分支，重演 v0.18.5 的"前端缺 tool"问题。runtime.json 由
//   ``agent-cot init / upgrade / start`` 写入，里面记录了 backend 真正去读的
//   data_root，hook 拿到后跟 env / 默认值 *按可信度排序* 选用，从而避免
//   pip install -U 后忘了重新跑 init --apply 也能继续工作。
//
// agent-cot v0.19.2：在 Stop / SessionEnd / SubagentStop / StopFailure 触发时
//   后台 spawn ``extract_cot.py --session-id codebuddy-<sid> --no-upload``，把
//   events.jsonl 转成 cot.json 落到 ``<data_root>/cot/codebuddy-<sid>_cot.json``，
//   让 backend SessionList 立刻能看到 codebuddy 会话。之前 codebuddy 唯一能跑
//   extract 的途径是用户手动跑 CLI（或者后台有 watcher），所以新装环境一旦缺
//   watcher，前端就永远看不到 codebuddy session（事件都写了，但没人转换）。
//   镜像 cursor cot-bridge.js / claude_stream_hook.py 的 4 层路径解析 +
//   30 秒去抖；DETACH spawn 立即 return，不阻塞 IDE 热路径。

import { appendFileSync, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { homedir } from 'os';
import { createHash } from 'crypto';
import { spawn } from 'child_process';

// v0.19.0: secondary fallback chain for AGENT_COT_DATA_ROOT.
// Read order: env var > runtime.json (most authoritative current install) >
// homedir-default. We intentionally don't probe Python here — codebuddy hooks
// are hot-path (every PreToolUse fires this) and we must stay <100ms.
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

// v0.19.3: data_root 自愈逻辑 ——
// 历史上某次 `agent-cot init --apply` 把 runtime.json 的 ``data_root`` 写成了
// ``~/.agent-cot``（无 /data 子目录），导致：
//   - hook 写 events / cot 到 ``~/.agent-cot/...``
//   - 但 backend 默认扫 ``~/.agent-cot/data/...``
//   - SessionList 中 codebuddy / claude 永远不出现。
// 我们在 hook 端识别这种坏值并自动补上 /data，保证写盘目录与 backend 默认对齐。
// env 显式设置的值不做自愈（信任使用者），只对从 runtime.json 读出的值兜底。
function _normalizeDataRoot(p) {
  if (typeof p !== 'string' || !p.trim()) return p;
  const segs = p.replace(/\\/g, '/').split('/').filter(Boolean);
  const last = segs[segs.length - 1] || '';
  if (last === 'data') return p;
  // 只对 ``.agent-cot`` 这一类已知 root 做修正；其他用户自定义路径保持原样。
  if (last === '.agent-cot' || last === '.cursor-cot') {
    return join(p, 'data');
  }
  return p;
}

const DATA_ROOT = process.env.AGENT_COT_DATA_ROOT
  || (_runtimeState && _normalizeDataRoot(_runtimeState.data_root))
  || join(homedir(), '.agent-cot', 'data');
// COT_ROOT 仍然保留：只用于诊断日志 / future feature；不影响 events.jsonl 写盘。
const COT_ROOT = process.env.COT_EXTRACTOR_ROOT
  || (_runtimeState && _runtimeState.cot_extractor_root)
  || join(homedir(), '.agent-cot', 'cot-extractor');
const ENABLED = (process.env.COT_STREAM_ENABLED || 'true').toLowerCase() !== 'false';
const MAX_FIELD = parseInt(process.env.COT_STREAM_MAX_FIELD || '8192', 10);
const MAX_TOTAL = parseInt(process.env.COT_STREAM_MAX_TOTAL || '20480', 10);
const STREAM_LOG = process.env.COT_STREAM_LOG
  || join(homedir(), '.codebuddy', 'cot-stream-codebuddy.log');
const SID_PREFIX = process.env.COT_STREAM_SID_PREFIX || 'codebuddy-';

// v0.20.0: 统一 pipeline.log（跟 cursor / claude / extractor / backend 共享）
const PIPELINE_LOG = process.env.AGENT_COT_PIPELINE_LOG
  || join(homedir(), '.agent-cot', 'logs', 'pipeline.log');

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
      `[${new Date().toISOString()}] [hook.codebuddy] [codebuddy] ` +
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

function clip(s) {
  if (typeof s !== 'string') return s;
  return s.length > MAX_FIELD ? s.slice(0, MAX_FIELD) + '…[+' + (s.length - MAX_FIELD) + 'c]' : s;
}

function truncateDeep(obj, depth = 0) {
  if (depth > 4) return '[deep]';
  if (typeof obj === 'string') return clip(obj);
  if (Array.isArray(obj)) {
    if (obj.length > 50) return obj.slice(0, 50).map((x) => truncateDeep(x, depth + 1)).concat(['…[+' + (obj.length - 50) + ']']);
    return obj.map((x) => truncateDeep(x, depth + 1));
  }
  if (obj && typeof obj === 'object') {
    const out = {};
    for (const k of Object.keys(obj)) out[k] = truncateDeep(obj[k], depth + 1);
    return out;
  }
  return obj;
}

function pick(payload, keys) {
  if (!payload || typeof payload !== 'object') return undefined;
  for (const k of keys) {
    if (k.includes('.')) {
      let cur = payload;
      for (const seg of k.split('.')) {
        if (cur && typeof cur === 'object' && seg in cur) cur = cur[seg];
        else { cur = undefined; break; }
      }
      if (cur !== undefined && cur !== null && cur !== '') return cur;
    } else if (k in payload && payload[k] !== undefined && payload[k] !== '') {
      return payload[k];
    }
  }
  return undefined;
}

function shortHash(value) {
  return createHash('sha1').update(String(value || 'unknown')).digest('hex').slice(0, 12);
}

function resolveSessionId(payload) {
  const sid = pick(payload, [
    'session_id', 'sessionId', 'conversation_id', 'conversationId',
    'chat.session.id', 'chatSessionId',
    'codebuddy.session.id', 'codebuddySessionId',
    'run_id', 'runId', 'message.session_id',
  ]);
  if (sid) return String(sid);

  const transcript = pick(payload, ['transcript_path', 'transcriptPath', 'TRANSCRIPT_PATH']);
  if (transcript) return `transcript-${shortHash(transcript)}`;

  const workspace = pick(payload, ['workspace', 'workspace_dir', 'cwd', 'project_dir', 'CODEBUDDY_PROJECT_DIR']);
  const user = pick(payload, ['user_name', 'userName', 'user.email', 'user']);
  return `orphan-${shortHash(`${workspace || ''}|${user || ''}|${process.cwd()}`)}`;
}

function pickToolName(event, payload) {
  return pick(payload, [
    'tool_name', 'toolName', 'tool.name', 'tool',
    'name', 'tool_input.name', 'toolInput.name',
  ]) || event;
}

function pickBriefInput(event, payload) {
  if (!payload || typeof payload !== 'object') return undefined;
  const brief = {};
  const toolInput = pick(payload, ['tool_input', 'toolInput', 'input', 'args', 'params']);
  if (toolInput !== undefined) brief.tool_input = truncateDeep(toolInput);
  const prompt = pick(payload, ['prompt', 'user_prompt', 'message.content', 'text']);
  if (typeof prompt === 'string') brief.prompt = clip(prompt);
  for (const key of ['command', 'matcher', 'file_path', 'path', 'cwd', 'workspace']) {
    const v = pick(payload, [key]);
    if (v !== undefined) brief[key] = truncateDeep(v);
  }
  if (event === 'UserPromptSubmit' && !brief.prompt) {
    const raw = JSON.stringify(truncateDeep(payload));
    brief.prompt_probe = clip(raw);
  }
  return Object.keys(brief).length ? brief : undefined;
}

function pickBriefOutput(event, payload) {
  if (!payload || typeof payload !== 'object') return undefined;
  const brief = {};
  for (const key of ['success', 'exit_code', 'exitCode', 'duration_ms', 'durationMs']) {
    const v = pick(payload, [key]);
    if (v !== undefined) brief[key] = v;
  }
  const out = pick(payload, [
    'stdout', 'output', 'result', 'tool_result.output', 'toolResult.output',
    'response', 'assistant_response', 'message.content',
  ]);
  if (typeof out === 'string' && out.length) brief.stdout = clip(out);
  const err = pick(payload, ['stderr', 'error', 'tool_result.error', 'toolResult.error']);
  if (typeof err === 'string' && err.length) brief.stderr = clip(err);
  return Object.keys(brief).length ? brief : undefined;
}

function pickThinkingProbe(payload) {
  const thought = pick(payload, [
    'thinking', 'thought', 'reasoning', 'reasoning_content',
    'assistant_thought', 'assistantThought',
    'message.thinking', 'message.reasoning',
    'response.thinking', 'response.reasoning',
  ]);
  if (typeof thought === 'string' && thought.trim()) return clip(thought);
  return undefined;
}

async function main() {
  // v0.19.0: emit a startup line just like cursor's cot-stream.js so that
  // ``agent-cot doctor --deep`` can confirm the codebuddy hook actually
  // resolved a sane DATA_ROOT (matches what backend / cot_extractor read).
  log(
    `process start pid=${process.pid} ENABLED=${ENABLED} ` +
    `DATA_ROOT=${DATA_ROOT} runtimeStateLoaded=${_runtimeState ? 'yes' : 'no'}`,
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

  const event = (process.argv[2] || payload.hook_event_name || payload.event_name || payload.event || 'unknown').toString();
  const sid = resolveSessionId(payload);
  const cid = sid.startsWith(SID_PREFIX) ? sid : (SID_PREFIX + sid);

  const entry = {
    t: Date.now(),
    event,
    cid,
    provider: 'codebuddy',
    tool: pickToolName(event, payload),
    brief_input: pickBriefInput(event, payload),
    brief_output: pickBriefOutput(event, payload),
    thinking_probe: pickThinkingProbe(payload),
    payload: truncateDeep(payload),
  };

  let line = JSON.stringify(entry);
  if (line.length > MAX_TOTAL) {
    entry.payload = '[truncated-total]';
    line = JSON.stringify(entry);
  }
  line += '\n';

  try {
    // agent-cot v0.18.6: 写到 ``<DATA_ROOT>/events/<cid>/`` —— 跟 cursor 的
    // cot-stream.js / cot_extractor._load_cursor_events / transcript_watcher
    // 完全对齐。site-packages 只读 / 路径找不到的问题彻底解开。
    const dir = join(DATA_ROOT, 'events', cid);
    mkdirSync(dir, { recursive: true });
    appendFileSync(join(dir, 'events.jsonl'), line);
    pipelineLog(event, { sid: cid, tool: entry.tool, bytes: line.length });
  } catch (e) {
    log(`append error event=${event} cid=${cid}: ${e.message}`);
    pipelineLog(event, { sid: cid, ok: false, error: e.message });
  }

  // v0.19.2: turn 结束时后台 spawn extract_cot.py，让 cot.json 自动落地。
  // 失败一律吞掉、绝不阻塞 hook（IDE 看到非 0 exit 会红字）。
  try {
    maybeTriggerLiveCritic(event, cid, entry);
  } catch (e) {
    log(`maybeTriggerLiveCritic error event=${event} cid=${cid}: ${e.message}`);
  }

  try {
    maybeTriggerExtract(event, cid);
  } catch (e) {
    log(`maybeTriggerExtract error event=${event} cid=${cid}: ${e.message}`);
  }
  try {
    maybeTriggerCritic(event, cid);
  } catch (e) {
    log(`maybeTriggerCritic error event=${event} cid=${cid}: ${e.message}`);
  }
  process.stdout.write('{"continue":true}');
}

// ═══════════════════════════════════════════════════════════
//  v0.19.2: extract_cot.py 自动触发 —— 让 codebuddy session 立刻出现在
//  前端 SessionList。镜像 cot-bridge.js / claude_stream_hook.py 的 4 层
//  路径解析 + 30 秒去抖。
// ═══════════════════════════════════════════════════════════

const EXTRACT_TRIGGER_EVENTS = new Set(['Stop', 'SessionEnd', 'SubagentStop', 'StopFailure']);
const CRITIC_TRIGGER_EVENTS = new Set(['Stop', 'SessionEnd', 'SubagentStop', 'StopFailure']);
const DEBOUNCE_WINDOW_MS = 30 * 1000;
const DEBOUNCE_FILE = join(DATA_ROOT, 'extract_debounce_codebuddy.json');
const CRITIC_DEBOUNCE_FILE = join(DATA_ROOT, 'critic_debounce_codebuddy.json');

function looksLikeExtractorRoot(p) {
  try { return statSync(join(p, 'scripts', 'extract_cot.py')).isFile(); } catch { return false; }
}

function readNamespaceRuntime(name) {
  try {
    const f = join(homedir(), name, 'runtime.json');
    if (!existsSync(f)) return null;
    const raw = readFileSync(f, 'utf-8');
    if (!raw) return null;
    return JSON.parse(raw.charCodeAt(0) === 0xFEFF ? raw.slice(1) : raw);
  } catch { return null; }
}

function resolveExtractorRoot() {
  const envV = process.env.COT_EXTRACTOR_ROOT;
  if (envV && looksLikeExtractorRoot(envV)) return envV;
  for (const ns of ['.agent-cot', '.cursor-cot']) {
    const st = readNamespaceRuntime(ns);
    if (st && typeof st.cot_extractor_root === 'string' && looksLikeExtractorRoot(st.cot_extractor_root)) {
      return st.cot_extractor_root;
    }
  }
  for (const c of [
    join(homedir(), '.agent-cot', 'cot-extractor'),
    join(homedir(), '.cursor-cot', 'cot-extractor'),
  ]) {
    if (looksLikeExtractorRoot(c)) return c;
  }
  return null;
}

function resolvePython() {
  const envV = process.env.COT_PYTHON;
  if (envV) return envV;
  for (const ns of ['.agent-cot', '.cursor-cot']) {
    const st = readNamespaceRuntime(ns);
    if (st && typeof st.python_executable === 'string' && st.python_executable.trim()) {
      return st.python_executable;
    }
  }
  return process.platform === 'win32' ? 'python.exe' : 'python3';
}

function isAgentQualityEvalExe(py) {
  const name = String(py || '').split(/[\\/]/).pop().toLowerCase();
  return name.startsWith('agent-quality-eval') && name.endsWith('.exe');
}

function agentQualityEvalRunnerArgs(kind, args) {
  if (isAgentQualityEvalExe(resolvePython())) {
    return ['--agent-quality-eval-runner', kind, ...args];
  }
  const code = kind === 'live-critic'
    ? 'from agent_quality_eval.evaluation.live_critic import main; raise SystemExit(main())'
    : 'from agent_quality_eval.evaluation.critic import main; raise SystemExit(main())';
  return ['-c', code, ...args];
}

function debounceShouldRun(cid) {
  const now = Date.now();
  let state = {};
  try {
    if (existsSync(DEBOUNCE_FILE)) {
      const raw = readFileSync(DEBOUNCE_FILE, 'utf-8');
      if (raw && raw.trim()) {
        const loaded = JSON.parse(raw);
        if (loaded && typeof loaded === 'object') {
          for (const [k, v] of Object.entries(loaded)) {
            if (typeof v === 'number' && now - v < 60 * 60 * 1000) state[k] = v;
          }
        }
      }
    }
  } catch { state = {}; }
  const last = state[cid] || 0;
  if (now - last < DEBOUNCE_WINDOW_MS) return false;
  state[cid] = now;
  try {
    mkdirSync(dirname(DEBOUNCE_FILE), { recursive: true });
    writeFileSync(DEBOUNCE_FILE, JSON.stringify(state), 'utf-8');
  } catch {}
  return true;
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
  if (now - last < DEBOUNCE_WINDOW_MS) return false;
  state[key] = now;
  try {
    mkdirSync(dirname(CRITIC_DEBOUNCE_FILE), { recursive: true });
    writeFileSync(CRITIC_DEBOUNCE_FILE, JSON.stringify(state), 'utf-8');
  } catch {}
  return true;
}

function maybeTriggerExtract(event, cid) {
  if (!EXTRACT_TRIGGER_EVENTS.has(event)) return;
  if (!cid) return;
  if (!debounceShouldRun(cid)) {
    log(`extract debounced cid=${cid} event=${event}`);
    return;
  }
  const extractorRoot = resolveExtractorRoot();
  if (!extractorRoot) {
    log(`extract skip: no extractor root resolved (env / runtime.json / probes all empty)`);
    return;
  }
  const py = resolvePython();
  const env = {
    ...process.env,
    AGENT_COT_DATA_ROOT: DATA_ROOT,
    AGENT_COT_PIPELINE_LOG: PIPELINE_LOG,
    PYTHONIOENCODING: 'utf-8',
  };
  // codebuddy session 没有外部 transcript JSONL（CodeBuddy 把 transcript 存在
  // index.json 里），cot_extractor.run_extract 在 ``--session-id codebuddy-*``
  // 模式下会从 events.jsonl 自己 reconstruct turns —— 已在 0.19.x 验证通过。
  const args = [
    join(extractorRoot, 'scripts', 'extract_cot.py'),
    '--session-id', cid,
    '--no-upload',
  ];
  log(`extract spawn cid=${cid} event=${event} py=${py} root=${extractorRoot}`);
  pipelineLog('extract_spawn', { sid: cid, python: py, cot_root: extractorRoot });
  try {
    const child = spawn(py, args, {
      cwd: extractorRoot,
      env,
      detached: true,
      stdio: 'ignore',
      windowsHide: true,
    });
    child.on('error', (err) => {
      log(`extract spawn err cid=${cid}: ${err.message}`);
      pipelineLog('extract_spawn', { sid: cid, ok: false, error: err.message });
    });
    child.unref();
  } catch (e) {
    log(`extract spawn fatal cid=${cid}: ${e.message}`);
    pipelineLog('extract_spawn', { sid: cid, ok: false, error: e.message });
  }
}

function liveCriticPayload(entry) {
  const out = {};
  for (const key of ['event', 'tool', 'tool_name', 'toolName', 'tool_use_id', 'cwd']) {
    if (entry && entry[key] !== undefined && entry[key] !== null) out[key] = entry[key];
  }
  return out;
}

function maybeTriggerLiveCritic(event, cid, entry) {
  if (!cid || process.env.AGENT_QUALITY_EVAL_LIVE_CRITIC_DISABLE) return;
  const py = resolvePython();
  const payload = liveCriticPayload(entry);
  const args = agentQualityEvalRunnerArgs('live-critic', [
    '--agent-type', 'codebuddy',
    '--source-event', event,
    '--session-id', cid,
  ]);
  if (Object.keys(payload).length) {
    args.push('--payload-json', JSON.stringify(payload));
  }
  const env = {
    ...process.env,
    AGENT_COT_DATA_ROOT: DATA_ROOT,
    AGENT_COT_PIPELINE_LOG: PIPELINE_LOG,
    AGENT_QUALITY_EVAL_LIVE_CRITIC_DISABLE: '1',
    PYTHONIOENCODING: 'utf-8',
  };
  try {
    const child = spawn(py, args, {
      cwd: homedir(),
      env,
      detached: true,
      stdio: 'ignore',
      windowsHide: true,
    });
    child.on('error', (err) => {
      log(`live critic spawn err cid=${cid}: ${err.message}`);
      pipelineLog('live_critic_pulse', { sid: cid, ok: false, error: err.message });
    });
    child.unref();
    pipelineLog('live_critic_pulse', { sid: cid, python: py, source_event: event });
  } catch (e) {
    log(`live critic spawn fatal cid=${cid}: ${e.message}`);
    pipelineLog('live_critic_pulse', { sid: cid, ok: false, error: e.message });
  }
}

function maybeTriggerCritic(event, cid) {
  if (!CRITIC_TRIGGER_EVENTS.has(event)) return;
  if (!cid) return;
  if (process.env.AGENT_QUALITY_EVAL_CRITIC_DISABLE) return;
  if (!criticDebounceShouldRun(event, cid)) {
    log(`critic debounced cid=${cid} event=${event}`);
    return;
  }
  const py = resolvePython();
  const args = agentQualityEvalRunnerArgs('critic', [
    '--agent-type', 'codebuddy',
    '--source-event', `codebuddy-stream:${event}`,
    '--session-id', cid,
    '--wait-seconds', '75',
    '--no-persist-eval',
  ]);
  const env = {
    ...process.env,
    AGENT_COT_DATA_ROOT: DATA_ROOT,
    AGENT_COT_PIPELINE_LOG: PIPELINE_LOG,
    AGENT_QUALITY_EVAL_CRITIC_DISABLE: '1',
    PYTHONIOENCODING: 'utf-8',
  };
  log(`critic spawn cid=${cid} event=${event} py=${py}`);
  pipelineLog('critic_spawn', { sid: cid, python: py, source_event: event });
  try {
    const child = spawn(py, args, {
      cwd: homedir(),
      env,
      detached: true,
      stdio: 'ignore',
      windowsHide: true,
    });
    child.on('error', (err) => {
      log(`critic spawn err cid=${cid}: ${err.message}`);
      pipelineLog('critic_spawn', { sid: cid, ok: false, error: err.message });
    });
    child.unref();
  } catch (e) {
    log(`critic spawn fatal cid=${cid}: ${e.message}`);
    pipelineLog('critic_spawn', { sid: cid, ok: false, error: e.message });
  }
}

main().catch((err) => {
  log(`FATAL ${err.message}\n${err.stack}`);
  process.stdout.write('{"continue":true}');
  process.exit(0);
});
