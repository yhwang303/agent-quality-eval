#!/usr/bin/env node

// VSCode (GitHub Copilot Chat Agent hooks, Preview) 事件 → events.jsonl 流
//
// VSCode 1.95+ 提供的 Agent hooks 机制和 Cursor / Claude 的 hook 模型一脉相承：
// IDE 在生命周期事件触发时，向我们注册的 ``"command"`` 进程喂一段 stdin JSON，
// 我们解析后追加到本地 jsonl，cot_extractor 后续把它跟 OTel events / transcript
// 一起组装成统一的 ThoughtStep 时间轴。
//
// 真实的 stdin schema 仍在 Preview，所以本脚本做了大量字段名 fallback——
// 任何一个常见命名（Cursor 风格 / Claude 风格 / GitHub 风格）都能匹配上：
//
//   session id : session_id / sessionId / chat.session.id / conversation_id
//   event name : hook_event_name / event_name / event / name / hookName
//   tool name  : tool_name / toolName / tool / name（在 tool_use 事件里）
//
// 每条事件最终落到：
//   <COT_ROOT>/output/events/<sid>/events.jsonl
//   { "t": ms, "event": "<EventName>", "cid": "<sid>", "tool": "<name?>",
//     "brief_input": {...?}, "brief_output": {...?}, "payload": {...截断} }
//
// 跟 cursor 版本的主要差异：
//   1. session 目录前缀加 ``vscode-``，避免和 Cursor / Claude 的同 sid 撞车
//   2. event 名做归一化（VSCode 用 PascalCase，统一小写 snake_case）
//   3. 没有 conversation_id 时用 process.env.COPILOT_SESSION_ID 兜底（hooks
//      Preview 阶段确实有些事件不带 sid，我们尽量不丢数据）

import { appendFileSync, mkdirSync } from 'fs';
import { dirname, join } from 'path';
import { homedir } from 'os';

// agent-cot v0.18.6：用户可写的数据根，跟 cursor 的 cot-stream.js 同款。events.jsonl
// 写到 ``<DATA_ROOT>/events/vscode-<sid>/`` 而不是 ``<COT_ROOT>/output/events/``，
// 解决"site-packages 只读 / 硬编码 D:/ai-ide-langfuse 在他人机器不存在"的双坑。
const DATA_ROOT = process.env.AGENT_COT_DATA_ROOT
  || join(homedir(), '.agent-cot', 'data');
const COT_ROOT = process.env.COT_EXTRACTOR_ROOT || join(homedir(), '.agent-cot', 'cot-extractor');
const ENABLED = (process.env.COT_STREAM_ENABLED || 'true').toLowerCase() !== 'false';
const MAX_FIELD = parseInt(process.env.COT_STREAM_MAX_FIELD || '8192', 10);
const MAX_TOTAL = parseInt(process.env.COT_STREAM_MAX_TOTAL || '20480', 10);
const STREAM_LOG = process.env.COT_STREAM_LOG
  || join(homedir(), '.copilot', 'cot-stream-vscode.log');
const SID_PREFIX = process.env.COT_STREAM_SID_PREFIX || 'vscode-';

function log(msg) {
  try {
    mkdirSync(dirname(STREAM_LOG), { recursive: true });
    appendFileSync(STREAM_LOG, `[${new Date().toISOString()}] ${msg}\n`);
  } catch {}
}

function readStdin() {
  return new Promise((ok, fail) => {
    let buf = '';
    process.stdin.setEncoding('utf-8');
    process.stdin.on('data', (c) => (buf += c));
    process.stdin.on('end', () => {
      try {
        // BOM strip（Windows shell 偶尔会带 UTF-8 BOM，跟 cursor 的 cot-stream 一样的兜底）
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

// 多路径取值：'a.b.c' 形式 + 多 key 列表
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

function resolveSessionId(payload) {
  return pick(payload, [
    'session_id', 'sessionId',
    'chat.session.id', 'chatSessionId',
    'conversation_id', 'conversationId',
    'gen_ai.conversation.id',
  ]) || process.env.COPILOT_SESSION_ID;
}

function resolveEventName(payload, argv) {
  // VSCode Agent hooks Preview：CLI 通常会在 argv[2] 里塞事件名，跟 claude_stream_hook.py 一致
  const fromArgs = (argv && argv[2]) ? String(argv[2]) : null;
  const fromBody = pick(payload, [
    'hook_event_name', 'event_name', 'event', 'name', 'hookName',
  ]);
  return (fromArgs || fromBody || 'unknown').toString();
}

function pickToolName(event, payload) {
  // tool_use / shell / mcp 类事件里的工具名
  return pick(payload, [
    'tool_name', 'toolName', 'tool', 'name',
    'tool_input.name', 'toolInput.name',
  ]) || event;
}

function pickBriefInput(event, payload) {
  if (!payload || typeof payload !== 'object') return undefined;
  // 通用 fallback：tool_input / input / args / params
  const cand = pick(payload, ['tool_input', 'toolInput', 'input', 'args', 'params']);
  if (cand !== undefined) {
    return truncateDeep(cand);
  }
  return undefined;
}

function pickBriefOutput(event, payload) {
  if (!payload || typeof payload !== 'object') return undefined;
  const brief = {};
  if (typeof payload.exit_code === 'number') brief.exit_code = payload.exit_code;
  if (typeof payload.exitCode === 'number') brief.exit_code = payload.exitCode;
  if (typeof payload.success === 'boolean') brief.success = payload.success;
  if (typeof payload.duration_ms === 'number') brief.duration_ms = payload.duration_ms;
  const so = pick(payload, ['stdout', 'output', 'result', 'tool_result.output', 'toolResult.output']);
  if (typeof so === 'string' && so.length) brief.stdout = clip(so);
  const se = pick(payload, ['stderr', 'error', 'tool_result.error']);
  if (typeof se === 'string' && se.length) brief.stderr = clip(se);
  return Object.keys(brief).length ? brief : undefined;
}

async function main() {
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

  const event = resolveEventName(payload, process.argv);
  const sid = resolveSessionId(payload);
  if (!sid) {
    log(`skip event=${event}, no session_id`);
    process.stdout.write('{"continue":true}');
    return;
  }
  const cid = sid.startsWith(SID_PREFIX) ? sid : (SID_PREFIX + sid);

  const entry = {
    t: Date.now(),
    event,
    cid,
    tool: pickToolName(event, payload),
    brief_input: pickBriefInput(event, payload),
    brief_output: pickBriefOutput(event, payload),
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
    // cot-stream.js / cot_extractor 完全对齐，避免硬编码 D:/ai-ide-langfuse 在
    // 他人机器找不到 + site-packages 只读 双坑。
    const dir = join(DATA_ROOT, 'events', cid);
    mkdirSync(dir, { recursive: true });
    appendFileSync(join(dir, 'events.jsonl'), line);
  } catch (e) {
    log(`append error event=${event} cid=${cid}: ${e.message}`);
  }

  process.stdout.write('{"continue":true}');
}

main().catch((err) => {
  log(`FATAL ${err.message}\n${err.stack}`);
  process.stdout.write('{"continue":true}');
  process.exit(0);
});
