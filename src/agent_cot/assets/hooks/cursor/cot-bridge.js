#!/usr/bin/env node

// Cursor -> 本地 CoT 提取器 桥接脚本
//
// 在 Cursor 的 stop（或 afterAgentResponse）事件里运行，
// 读取 stdin 拿到 conversation_id + workspace_roots，
// 定位 Cursor transcript 文件，然后异步 spawn 本地 Python 提取器
// （不阻塞 Cursor UI），把 CoT 结果写到 agent-dashboard 可见的目录。
//
// 数据流:
//   stdin(JSON: conversation_id / workspace_roots / hook_event_name)
//     -> ~/.cursor/projects/<slug>/agent-transcripts/<conv_id>/<conv_id>.jsonl
//     -> python extract_cot.py --transcript ... --session-id <conv_id> --no-upload [--history]
//     -> cot-extractor/output/cot/<conv_id>_cot.json            (主文件, dashboard 看)
//     -> cot-extractor/output/sessions/<conv_id>/<ts>_cot.json  (每次 stop 的历史快照, 需开 COT_KEEP_HISTORY)
//
// 环境变量:
//   COT_EXTRACTOR_ROOT   — cot-extractor 根目录；未设置时后备字面量仅作占位，
//                          ``agent-cot init --apply`` 会把它 patch 成当前安装机上的真实路径
//   COT_PYTHON           — Python 解释器；未设置时后备由 ``agent-cot init`` 写成
//                          运行 init 时的 ``sys.executable``（避免 Cursor 子进程 PATH 无 python）
//   COT_BRIDGE_ENABLED   — "false" 可禁用 (默认启用)
//   COT_RUN_VERIFIER     — "true" 同时跑 response-verifier (默认 false)
//   COT_KEEP_HISTORY     — "true" 保留每次 stop 的历史快照 (默认 true)
//   COT_BRIDGE_LOG       — 自定义日志路径 (默认 ~/.cursor/cot-bridge.log)

import { existsSync, appendFileSync, mkdirSync, readdirSync, readFileSync } from 'fs';
import { spawn, spawnSync } from 'child_process';
import { resolve, dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { homedir } from 'os';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// v0.18.15: self-healing path resolution.
//
// Previously COT_ROOT / PYTHON were ONLY one of:
//   1. an env var set by the caller
//   2. an install-time literal patched in by `agent-cot init/upgrade --apply`
//
// If neither resolved (e.g. user did `pip install -U agent-cot`
// without re-running `init/upgrade --apply`, or upgraded Python and the
// patched literal now points at a vanished site-packages), the hook would
// silently spawn a missing path and Cursor would never produce a cot.json.
//
// The new fallback chain is:
//   env COT_EXTRACTOR_ROOT  →  patched literal  →  ~/.agent-cot/runtime.json
//                                              →  python -c "..." probe
//
// Each step is tried only when the prior produced a non-existent path
// pointing at scripts/extract_cot.py. The runtime.json file is rewritten
// every `agent-cot start / init / upgrade` so a single invocation of
// any of those self-heals the hook.

const RAW_COT_ROOT = process.env.COT_EXTRACTOR_ROOT || '__AGENT_COT_EXTRACTOR_ROOT_UNCONFIGURED__';
const RAW_PYTHON = process.env.COT_PYTHON || '__AGENT_COT_PYTHON_UNCONFIGURED__';

function isExtractorRoot(p) {
  if (!p || typeof p !== 'string') return false;
  try {
    return existsSync(join(p, 'scripts', 'extract_cot.py'));
  } catch { return false; }
}

function isExecutable(p) {
  if (!p || typeof p !== 'string') return false;
  try {
    return existsSync(p);
  } catch { return false; }
}

function readRuntimeState() {
  try {
    const f = join(homedir(), '.agent-cot', 'runtime.json');
    if (!existsSync(f)) return null;
    const raw = readFileSync(f, 'utf-8');
    if (!raw) return null;
    return JSON.parse(raw.charCodeAt(0) === 0xFEFF ? raw.slice(1) : raw);
  } catch { return null; }
}

// v0.19.3: data_root 自愈，跟 codebuddy / claude / cursor cot-stream.js 完全等价。
// runtime.json 历史损坏（``data_root`` = ``~/.agent-cot`` 而非 ``~/.agent-cot/data``）
// 时自动补 ``/data``；env 显式值不动。详见 cot-stream-codebuddy.js 同名注释。
function normalizeDataRoot(p) {
  if (typeof p !== 'string' || !p.trim()) return p;
  const segs = p.replace(/\\/g, '/').split('/').filter(Boolean);
  const last = segs[segs.length - 1] || '';
  if (last === 'data') return p;
  if (last === '.agent-cot' || last === '.cursor-cot') {
    return join(p, 'data');
  }
  return p;
}

// v0.19.3: 解析 extract_cot.py 的 AGENT_COT_DATA_ROOT。
// 链路 = env > runtime.json (经 normalizeDataRoot 自愈) > 默认 ~/.agent-cot/data。
// 跟 backend/config.py:_user_data_root() / claude_stream_hook._resolve_data_root /
// codebuddy & cursor cot-stream.js 的 DATA_ROOT 解析逻辑完全一致 —— 这是
// 「single source of truth」机制对 cursor 桥接链路的最后一个补丁。
function resolveDataRoot() {
  const envV = process.env.AGENT_COT_DATA_ROOT;
  if (envV) return envV;
  const st = readRuntimeState();
  if (st && typeof st.data_root === 'string' && st.data_root.trim()) {
    return normalizeDataRoot(st.data_root);
  }
  return join(homedir(), '.agent-cot', 'data');
}

function probePythonForExtractor(pythonExe) {
  // Last-resort: ask Python where the bundled extractor lives. Costs
  // ~150 ms (one cold Python interpreter start). Only invoked when the
  // baked-in literal AND runtime.json BOTH failed — i.e. a single
  // user-visible "first time after pip upgrade" hit.
  if (!isExecutable(pythonExe)) return null;
  try {
    const out = spawnSync(pythonExe, [
      '-c',
      "import json,sys;\n" +
      "from pathlib import Path;\n" +
      "try:\n" +
      "  from agent_cot._assets import bundled_extractor_root, has_bundled_extractor\n" +
      "  print(json.dumps({'root': str(bundled_extractor_root().resolve()) if has_bundled_extractor() else None}))\n" +
      "except Exception as e:\n" +
      "  print(json.dumps({'root': None, 'err': str(e)}))",
    ], { encoding: 'utf-8', windowsHide: true, timeout: 4000 });
    if (out.status !== 0 || !out.stdout) return null;
    const parsed = JSON.parse(out.stdout.trim());
    return parsed.root || null;
  } catch { return null; }
}

function probeForPython() {
  // When the patched PYTHON literal is invalid (often after Python upgrade),
  // fall back to PATH lookup — Windows has `where`, POSIX has `which`.
  const candidates = process.platform === 'win32'
    ? ['python.exe', 'python3.exe', 'py.exe']
    : ['python3', 'python', 'python3.12', 'python3.11', 'python3.10'];
  for (const cand of candidates) {
    try {
      const out = spawnSync(cand, ['--version'], { encoding: 'utf-8', windowsHide: true, timeout: 3000 });
      if (out.status === 0 && /Python 3\.(1[0-9]|[2-9][0-9])/.test(out.stdout + out.stderr)) {
        return cand; // bare name; OS resolves via PATH
      }
    } catch { /* continue */ }
  }
  return null;
}

function resolveCotRoot(pythonExe) {
  // v0.20.6: source priority reversed for runtime.json vs patched literal.
  //
  // Pre-0.20.6 order was [env, patched_literal, runtime.json] but that meant:
  //   * If `init --apply` ran in a dev-mode editable install where the
  //     auto-detected ``cot-extractor`` candidate was a stale sibling git
  //     checkout, the WRONG path would be baked into the JS as the
  //     ``patched literal`` and would WIN over the correct value that
  //     ``agent-cot start`` later writes into runtime.json.
  //   * Result: every Cursor session got its CoT routed to a directory
  //     the backend never scans → dashboard silently stops showing new
  //     sessions, even though pipeline.log shows the extractor running.
  //
  // 0.20.6 makes runtime.json win over patched literal so that:
  //   * `agent-cot start` (which writes runtime.json with the *currently
  //     installed* extractor) can always heal a stale install-time bake.
  //   * patched literal is now a true fallback for the rare case where
  //     hooks fire BEFORE `agent-cot start` has ever run on this machine
  //     (e.g. fresh-install + first Cursor open).
  const sources = [
    ['env COT_EXTRACTOR_ROOT', process.env.COT_EXTRACTOR_ROOT],
  ];
  // Runtime state — fast, no subprocess. Updated by `agent-cot start`.
  const state = readRuntimeState();
  if (state && state.cot_extractor_root) {
    sources.push(['~/.agent-cot/runtime.json', state.cot_extractor_root]);
  }
  sources.push(['patched literal', RAW_COT_ROOT]);
  for (const [src, val] of sources) {
    if (isExtractorRoot(val)) {
      return { root: val, source: src };
    }
  }
  // Last resort: ask Python.
  const probed = probePythonForExtractor(pythonExe);
  if (isExtractorRoot(probed)) {
    return { root: probed, source: 'python -c probe' };
  }
  return { root: RAW_COT_ROOT, source: 'no valid source — using patched literal (will fail)' };
}

function resolvePython() {
  // v0.20.6: same priority reversal as resolveCotRoot — runtime.json wins
  // over the patched literal so a stale install-time bake can be healed by
  // re-running `agent-cot start`. See resolveCotRoot for the full rationale.
  const sources = [
    ['env COT_PYTHON', process.env.COT_PYTHON],
  ];
  const state = readRuntimeState();
  if (state && state.python_executable) {
    sources.push(['~/.agent-cot/runtime.json', state.python_executable]);
  }
  sources.push(['patched literal', RAW_PYTHON]);
  for (const [src, val] of sources) {
    if (isExecutable(val)) {
      return { python: val, source: src };
    }
  }
  const probed = probeForPython();
  if (probed) {
    return { python: probed, source: 'PATH probe' };
  }
  return { python: RAW_PYTHON, source: 'no valid source — using patched literal (will fail)' };
}

const _resolvedPython = resolvePython();
const PYTHON = _resolvedPython.python;
const _resolvedExtractor = resolveCotRoot(PYTHON);
const COT_ROOT = _resolvedExtractor.root;
const ENABLED = (process.env.COT_BRIDGE_ENABLED || 'true').toLowerCase() !== 'false';
const RUN_VERIFIER = (process.env.COT_RUN_VERIFIER || 'false').toLowerCase() === 'true';
const KEEP_HISTORY = (process.env.COT_KEEP_HISTORY || 'true').toLowerCase() === 'true';

// 日志固定放到 ~/.cursor/cot-bridge.log，项目级 / 用户级共用，避免污染家目录
const BRIDGE_LOG = process.env.COT_BRIDGE_LOG
  || join(homedir(), '.cursor', 'cot-bridge.log');

// v0.20.0: 同时写一份到 ~/.agent-cot/logs/pipeline.log，跟 codebuddy / claude /
// extractor / backend 共用一个文件。``tail -f`` 一个文件就能看到完整链路。
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
  const line = `[${new Date().toISOString()}] [cot-bridge] ${msg}\n`;
  try {
    mkdirSync(dirname(BRIDGE_LOG), { recursive: true });
    appendFileSync(BRIDGE_LOG, line);
  } catch {}
}

function pipelineLog(event, fields = {}) {
  // 统一 pipeline.log 行格式 —— 跟 diag.py 的 log() 完全一样的列。
  // 排错时颜色组（cursor）+ stage（hook.cursor）+ event（PostToolUse/Stop/…）
  // 跨语言对得上是关键。
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
      `[${new Date().toISOString()}] [hook.cursor] [cursor] ` +
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
        // Windows 下 Cursor 在 stdin 前会带 UTF-8 BOM (EF BB BF)，
        // JSON.parse 不吃 BOM 会直接抛 "Unexpected token" —— 先 strip。
        if (buf.charCodeAt(0) === 0xFEFF) buf = buf.slice(1);
        ok(buf ? JSON.parse(buf) : {});
      } catch (e) { fail(e); }
    });
    process.stdin.on('error', fail);
  });
}

// 把 workspace_roots 里的路径映射为 Cursor 的 project slug。
// 观察到的规则：
//   D:\SST                   -> d-sst
//   D:\ai-ide-langfuse       -> d-ai-ide-langfuse
//   C:\Users\x\AppData\Local -> c-users-x-appdata-local
function pathToSlug(p) {
  if (!p) return null;
  return p
    .replace(/[\\/:]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .toLowerCase();
}

// 在 ~/.cursor/projects/<slug>/agent-transcripts/<conv_id>/<conv_id>.jsonl 里定位 transcript。
// 优先用 workspace_roots 推断的 slug 匹配，失败则回退到全局扫描。
function locateTranscript(convId, workspaceRoots) {
  const cursorProjects = join(homedir(), '.cursor', 'projects');
  if (!existsSync(cursorProjects)) {
    log(`Cursor projects dir not found: ${cursorProjects}`);
    return null;
  }

  const candidates = [];
  if (Array.isArray(workspaceRoots)) {
    for (const r of workspaceRoots) {
      const slug = pathToSlug(r);
      if (slug) candidates.push(slug);
    }
  }

  for (const slug of candidates) {
    const p = join(cursorProjects, slug, 'agent-transcripts', convId, `${convId}.jsonl`);
    if (existsSync(p)) {
      log(`Located by slug=${slug}: ${p}`);
      return p;
    }
  }

  try {
    for (const entry of readdirSync(cursorProjects)) {
      const p = join(cursorProjects, entry, 'agent-transcripts', convId, `${convId}.jsonl`);
      if (existsSync(p)) {
        log(`Located by fallback scan: ${p}`);
        return p;
      }
    }
  } catch (e) {
    log(`Fallback scan error: ${e.message}`);
  }

  log(`Transcript not found for conv=${convId}, roots=${JSON.stringify(workspaceRoots)}`);
  return null;
}

// Fire-and-forget 启动 Python 提取器，不阻塞 Cursor。
// Windows 下用 detached + unref，让子进程独立存活。
//
// v0.18.13: 关键 env injection ——
//   AGENT_COT_DATA_ROOT  让 extract_cot.py 把 cot.json 写到
//                          ~/.agent-cot/data/cot/（前端 / backend 实际读的目录）
//                          而不是 cot-extractor/output/cot/（dev tree 里那个，
//                          backend 不扫，所以写了等于没写）。
//
// 历史 bug：dev tree 用户跑这个 hook 后，extract_cot.py 把 32MB 的 cot.json
// 写进 D:/ai-ide-langfuse/cot-extractor/output/cot/，然后纳闷"为什么 dashboard
// 不刷新" —— dashboard 一直在扫 ~/.agent-cot/data/cot/，根本就不看 dev 目录。
function spawnDetached(cmd, args, cwd, tag, extraEnv = {}) {
  try {
    const child = spawn(cmd, args, {
      cwd,
      detached: true,
      stdio: 'ignore',
      windowsHide: true,
      shell: false,
      env: { ...process.env, ...extraEnv },
    });
    child.on('error', (err) => log(`${tag} spawn error: ${err.message}`));
    child.unref();
    log(`${tag} spawned pid=${child.pid}: ${cmd} ${args.join(' ')}`);
  } catch (e) {
    log(`${tag} spawn exception: ${e.message}`);
  }
}

async function main() {
  // 必须在 readStdin() 之前写一行：否则若 Cursor 未关 stdin / 子进程卡住，
  // 日志永远不会出现，排障时误以为「hook 没装」。有这行即可区分
  // 「进程已起、卡在等 stdin」vs「根本没 spawn」。
  log(
    `process start pid=${process.pid} ENABLED=${ENABLED} ` +
    `PYTHON=${PYTHON} (via ${_resolvedPython.source}) ` +
    `COT_ROOT=${COT_ROOT} (via ${_resolvedExtractor.source})`,
  );
  pipelineLog('process_start', {
    pid: process.pid, enabled: ENABLED,
    python_src: _resolvedPython.source, cot_root_src: _resolvedExtractor.source,
  });

  if (!ENABLED) {
    process.stdout.write(JSON.stringify({ continue: true }));
    return;
  }

  let payload;
  try {
    payload = await readStdin();
  } catch (e) {
    log(`stdin parse error: ${e.message}`);
    process.stdout.write(JSON.stringify({ continue: true }));
    return;
  }

  const event = payload.hook_event_name || '?';
  const convId = payload.conversation_id || payload.sessionId || payload.session_id;
  const workspaceRoots = payload.workspace_roots || [];

  log(`triggered event=${event} conv=${convId} keepHistory=${KEEP_HISTORY}`);
  pipelineLog(event, { sid: convId, ws_roots: workspaceRoots.length });

  if (!convId) {
    log('no conversation_id, skip');
    pipelineLog(event, { sid: '-', ok: false, error: 'no conversation_id' });
    process.stdout.write(JSON.stringify({ continue: true }));
    return;
  }

  const transcript = locateTranscript(convId, workspaceRoots);
  if (!transcript) {
    pipelineLog(event, { sid: convId, ok: false, error: 'transcript not found' });
    process.stdout.write(JSON.stringify({ continue: true }));
    return;
  }
  pipelineLog('transcript_found', { sid: convId, path: transcript });

  const extractScript = resolve(COT_ROOT, 'scripts', 'extract_cot.py');
  if (!existsSync(extractScript)) {
    log(
      `extract_cot.py missing at ${extractScript}. ` +
      `Resolution chain failed (env=${process.env.COT_EXTRACTOR_ROOT || '(unset)'}, ` +
      `literal=${RAW_COT_ROOT}, runtime.json=${(readRuntimeState() || {}).cot_extractor_root || '(missing)'}). ` +
      `Fix: run \`agent-cot start\` (auto-heals runtime.json) or ` +
      `\`agent-cot upgrade --apply\` (re-patches this script).`,
    );
    pipelineLog('extract_skip', {
      sid: convId, ok: false,
      error: 'extract_cot.py missing', path: extractScript,
    });
    process.stdout.write(JSON.stringify({ continue: true }));
    return;
  }

  const extractArgs = [
    extractScript,
    '--transcript', transcript,
    '--session-id', convId,
    '--no-upload',
  ];
  if (KEEP_HISTORY) extractArgs.push('--history');

  // v0.18.13: pin output dir to ~/.agent-cot/data so the dashboard
  // sees the result. Otherwise extract_cot.py will write to
  // <COT_ROOT>/output/cot/ which the backend doesn't scan.
  //
  // v0.19.3: 跟 cot-stream.js / cot-stream-codebuddy.js / claude_stream_hook.py
  // / backend/config.py 一起走完全相同的「env > runtime.json (自愈) > 默认」
  // 解析链 —— 之前这里直接 fallback 到 ``~/.agent-cot/data``，凑巧和 backend 旧
  // 默认对齐所以一直没事，但只要用户在 ``agent-cot start`` 注入了自定义
  // ``AGENT_COT_DATA_ROOT`` 或者 runtime.json 改过 data_root 字段，cot-bridge
  // 就会跟其他三家产生分裂。统一到单一真相源（runtime.json）后，整套系统对
  // "用户改路径"是真正幂等的，不再有任何隐性硬编码差异。
  const dataRoot = resolveDataRoot();
  pipelineLog('extract_spawn', {
    sid: convId, python: PYTHON, cot_root: COT_ROOT, data_root: dataRoot,
  });
  spawnDetached(PYTHON, extractArgs, COT_ROOT, 'cot-extractor', {
    AGENT_COT_DATA_ROOT: dataRoot,
    AGENT_COT_PIPELINE_LOG: PIPELINE_LOG,
  });
  spawnDetached(
    PYTHON,
    agentQualityEvalRunnerArgs('live-critic', [
      '--agent-type', 'cursor',
      '--source-event', event,
      '--session-id', convId,
    ]),
    homedir(),
    'live-critic',
    {
      AGENT_COT_DATA_ROOT: dataRoot,
      AGENT_COT_PIPELINE_LOG: PIPELINE_LOG,
      AGENT_QUALITY_EVAL_LIVE_CRITIC_DISABLE: '1',
      PYTHONIOENCODING: 'utf-8',
    },
  );
  pipelineLog('live_critic_pulse', { sid: convId, python: PYTHON, source_event: event });
  spawnDetached(
    PYTHON,
    agentQualityEvalRunnerArgs('critic', [
      '--agent-type', 'cursor',
      '--source-event', `cursor-bridge:${event}`,
      '--session-id', convId,
      '--wait-seconds', '75',
      '--no-persist-eval',
    ]),
    homedir(),
    'agent-critic',
    {
      AGENT_COT_DATA_ROOT: dataRoot,
      AGENT_COT_PIPELINE_LOG: PIPELINE_LOG,
      AGENT_QUALITY_EVAL_CRITIC_DISABLE: '1',
      PYTHONIOENCODING: 'utf-8',
    },
  );
  pipelineLog('critic_spawn', { sid: convId, python: PYTHON, source_event: event });
  if (RUN_VERIFIER) {
    const verifierRoot = resolve(COT_ROOT, '..', 'response-verifier');
    const verifierScript = resolve(verifierRoot, 'scripts', 'run_check.py');
    if (existsSync(verifierScript)) {
      spawnDetached(
        PYTHON,
        [verifierScript, '--transcript', transcript, '--session-id', convId],
        verifierRoot,
        'response-verifier',
      );
    } else {
      log(`run_check.py missing at ${verifierScript}`);
    }
  }

  process.stdout.write(JSON.stringify({ continue: true }));
}

main().catch((err) => {
  log(`FATAL ${err.message}\n${err.stack}`);
  process.stdout.write(JSON.stringify({ continue: true }));
  process.exit(0);
});
