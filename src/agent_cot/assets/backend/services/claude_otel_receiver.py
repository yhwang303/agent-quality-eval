"""Claude Code 原生 OTel 接收器（OTLP/HTTP/JSON 协议）。

v0.16.0 引入。Claude Code 原生支持 OpenTelemetry，启用方式是设置环境变量
``CLAUDE_CODE_ENABLE_TELEMETRY=1`` + ``OTEL_EXPORTER_OTLP_ENDPOINT=http://...``
后，进程会主动把 metrics / logs / traces 三种信号 POST 到 collector。

本模块就是那个 collector——但是「极简版」：

  * 只支持 OTLP/HTTP/JSON 协议（``OTEL_EXPORTER_OTLP_PROTOCOL=http/json``）
  * 不做采样 / 路由 / 重试，直接按 ``session.id`` attribute 分桶落盘
  * 落盘位置：``~/.claude/state/otel/<session_id>/{events,metrics,traces}.jsonl``
  * 任何错误都返回 200 + ``{"partialSuccess": ...}``，避免阻塞 Claude Code

Claude Code 默认每 60s 推 metrics / 每 5s 推 logs，所以 receiver 必须够快
（< 50ms 处理一批），但本地 IPC 这是天然达到的。

为什么不部署官方 OTel Collector？
  * 用户场景是『前端展示』而不是『生产 observability』
  * 多一个进程多一份故障面，本地直接落盘更稳
  * 后期想接 Phoenix / Langfuse 时再叠 collector 也兼容

OTLP/JSON 协议参考：
  * https://opentelemetry.io/docs/specs/otlp/#otlphttp-request
  * 顶层 schema：ExportLogsServiceRequest / ExportMetricsServiceRequest /
    ExportTraceServiceRequest（proto3 → JSON 直转）
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fastapi import Request, Response

# 落盘根目录：与 hook stream 共用 ~/.claude/state，不冲突
OTEL_ROOT = Path.home() / ".claude" / "state" / "otel"


# ════════════════════════════════════════════════════════════
#  OTLP attribute 解析
# ════════════════════════════════════════════════════════════

def _coerce_anyvalue(v: Any) -> Any:
    """OTLP AnyValue 解码：``{"stringValue": "x"}`` → ``"x"``。

    支持的字段：stringValue, intValue, doubleValue, boolValue, arrayValue,
    kvlistValue, bytesValue（base64 字符串）。任何无法识别的形态原样返回。
    """
    if not isinstance(v, dict):
        return v
    if "stringValue" in v:
        return v["stringValue"]
    if "intValue" in v:
        # OTLP 把 int64 写成字符串（"1234"），这里统一转 int
        try:
            return int(v["intValue"])
        except (TypeError, ValueError):
            return v["intValue"]
    if "doubleValue" in v:
        try:
            return float(v["doubleValue"])
        except (TypeError, ValueError):
            return v["doubleValue"]
    if "boolValue" in v:
        return bool(v["boolValue"])
    if "arrayValue" in v:
        arr = v["arrayValue"].get("values") or []
        return [_coerce_anyvalue(x) for x in arr]
    if "kvlistValue" in v:
        return _coerce_attributes(v["kvlistValue"].get("values") or [])
    if "bytesValue" in v:
        return v["bytesValue"]
    return v


def _coerce_attributes(attrs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """OTLP attributes 数组（``[{key, value:{stringValue:..}}, ...]``）→ dict。"""
    out: Dict[str, Any] = {}
    for kv in attrs or []:
        if not isinstance(kv, dict):
            continue
        k = kv.get("key")
        if not isinstance(k, str):
            continue
        out[k] = _coerce_anyvalue(kv.get("value") or {})
    return out


def _ns_to_iso(nanos_str: Any) -> Optional[str]:
    """OTel 时间戳是 nanos string → ISO8601 with Z。"""
    if not nanos_str:
        return None
    try:
        ns = int(nanos_str)
    except (TypeError, ValueError):
        return None
    if ns <= 0:
        return None
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat().replace("+00:00", "Z")


# ════════════════════════════════════════════════════════════
#  落盘工具
# ════════════════════════════════════════════════════════════

# v0.17.0: 多 IDE 支持。OTLP receiver 不再绑定 Claude Code 一家——
# VSCode Copilot / CodeBuddy 之类只要走 OTLP/HTTP/JSON 也能落到这里。
# session id 字段可能叫 session.id（Claude）/ chat.session.id（Copilot）/
# gen_ai.conversation.id（OTel GenAI 语义约定标准字段）/ session_uuid（自定义）等，
# 这里把 fallback 字段集合扩到所有已知形态。
_SESSION_ID_KEYS = (
    # Claude Code 风格
    "session.id",
    "session_id",
    "sessionId",
    # OTel GenAI 语义约定（OpenTelemetry semantic conventions for GenAI）
    "gen_ai.conversation.id",
    "gen_ai.session.id",
    # GitHub Copilot Chat 风格（agent_monitoring.md）
    "chat.session.id",
    "copilot.session.id",
    "copilot.chat.session.id",
    # CodeBuddy 推测
    "codebuddy.session.id",
    # Codex native OTel / rollout-adjacent shapes
    "codex.conversation.id",
    "codex.session.id",
    "codex.thread.id",
    "thread.id",
    "turn.id",
    # 兜底
    "session_uuid",
    "conversation.id",
    "conversation_id",
)


def _resolve_session_id(*attr_dicts: Dict[str, Any]) -> Optional[str]:
    """从多个 attribute 来源中找 session id；按 _SESSION_ID_KEYS 顺序匹配。"""
    for d in attr_dicts:
        if not isinstance(d, dict):
            continue
        for key in _SESSION_ID_KEYS:
            v = d.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def _resolve_provider_tag(*attr_dicts: Dict[str, Any]) -> Optional[str]:
    """根据 OTLP 上报方的 service.name / gen_ai.system / scope 等，识别它是
    哪个 IDE/agent。返回值会作为 session 目录前缀（vscode-<sid> / codebuddy-<sid>），
    避免不同 IDE 同 session id 撞车，也方便 list_otel_sessions 区分来源。

    返回 None 表示「Claude Code 或未识别」——保持 ~/.claude/state/otel/<sid>/
    的现状，向后兼容。
    """
    haystack: List[str] = []
    for d in attr_dicts:
        if not isinstance(d, dict):
            continue
        for key in (
            "service.name", "service.namespace",
            "gen_ai.system", "gen_ai.provider",
            "telemetry.sdk.name",
            "scope", "scope.name",
            "event.name",
            "codex.provider",
        ):
            v = d.get(key)
            if isinstance(v, str) and v:
                haystack.append(v.lower())
        for key in d.keys():
            if isinstance(key, str) and key.lower().startswith("codex."):
                haystack.append("codex")
    blob = " ".join(haystack)
    if not blob:
        return None
    if "claude" in blob:
        return None  # 向后兼容：Claude 不加前缀
    if "codex" in blob or "openai-codex" in blob:
        return "codex"
    if "copilot" in blob or "github-copilot" in blob:
        return "vscode"
    if "vscode" in blob and "copilot" not in blob:
        return "vscode"
    if "codebuddy" in blob or "code-buddy" in blob:
        return "codebuddy"
    return None


def _resolve_session_key(*attr_dicts: Dict[str, Any]) -> Optional[str]:
    """组合 session_id + provider 前缀，作为最终落盘目录名。"""
    sid = _resolve_session_id(*attr_dicts)
    if not sid:
        return None
    tag = _resolve_provider_tag(*attr_dicts)
    if tag and not sid.startswith(f"{tag}-"):
        return f"{tag}-{sid}"
    return sid


def _append_jsonl(target: Path, records: List[Dict[str, Any]]) -> int:
    if not records:
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as f:
        for r in records:
            try:
                f.write(json.dumps(r, ensure_ascii=False, default=str))
            except Exception:
                # 序列化兜底：替换不可序列化字段
                f.write(json.dumps({"_serialize_error": True, "preview": str(r)[:300]}))
            f.write("\n")
    return len(records)


def _bucket_by_session(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        sid = rec.get("_session_id") or "_orphan"
        out.setdefault(sid, []).append(rec)
    return out


# ════════════════════════════════════════════════════════════
#  Logs / Events
# ════════════════════════════════════════════════════════════

def _flatten_log_records(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """ExportLogsServiceRequest → 扁平 log_record 列表。"""
    out: List[Dict[str, Any]] = []
    for rl in payload.get("resourceLogs") or []:
        res_attrs = _coerce_attributes((rl.get("resource") or {}).get("attributes") or [])
        for sl in rl.get("scopeLogs") or []:
            scope = sl.get("scope") or {}
            scope_name = scope.get("name") or ""
            scope_version = scope.get("version") or ""
            for lr in sl.get("logRecords") or []:
                rec_attrs = _coerce_attributes(lr.get("attributes") or [])
                body = lr.get("body") or {}
                rec = {
                    "ts": _ns_to_iso(lr.get("timeUnixNano")),
                    "observed_ts": _ns_to_iso(lr.get("observedTimeUnixNano")),
                    "severity": lr.get("severityText"),
                    "scope": {"name": scope_name, "version": scope_version},
                    "event_name": rec_attrs.get("event.name") or scope_name,
                    "body": _coerce_anyvalue(body) if body else None,
                    "attributes": rec_attrs,
                    "resource": res_attrs,
                    "_session_id": _resolve_session_key(rec_attrs, res_attrs),
                    "_provider": _resolve_provider_tag(rec_attrs, res_attrs),
                }
                out.append(rec)
    return out


# ════════════════════════════════════════════════════════════
#  Metrics
# ════════════════════════════════════════════════════════════

def _flatten_metric_records(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """ExportMetricsServiceRequest → 扁平 metric_data_point 列表。

    把 sum / gauge / histogram 三种类型都展开成单条 data_point 记录，
    带上 metric.name / metric.unit / 各种 attributes，方便后端按时间序列展示。
    """
    out: List[Dict[str, Any]] = []
    for rm in payload.get("resourceMetrics") or []:
        res_attrs = _coerce_attributes((rm.get("resource") or {}).get("attributes") or [])
        for sm in rm.get("scopeMetrics") or []:
            scope = sm.get("scope") or {}
            scope_name = scope.get("name") or ""
            for m in sm.get("metrics") or []:
                name = m.get("name") or ""
                unit = m.get("unit") or ""
                desc = m.get("description") or ""
                # 三种 metric 数据形态都有 dataPoints
                for type_key in ("sum", "gauge", "histogram", "exponentialHistogram", "summary"):
                    block = m.get(type_key)
                    if not isinstance(block, dict):
                        continue
                    for dp in block.get("dataPoints") or []:
                        dp_attrs = _coerce_attributes(dp.get("attributes") or [])
                        # 拿数值：sum/gauge 用 asInt/asDouble，histogram 用 sum/count
                        val: Any = None
                        if "asInt" in dp:
                            try:
                                val = int(dp["asInt"])
                            except (TypeError, ValueError):
                                val = dp["asInt"]
                        elif "asDouble" in dp:
                            try:
                                val = float(dp["asDouble"])
                            except (TypeError, ValueError):
                                val = dp["asDouble"]
                        elif type_key in ("histogram", "exponentialHistogram", "summary"):
                            val = {
                                "sum": dp.get("sum"),
                                "count": dp.get("count"),
                                "min": dp.get("min"),
                                "max": dp.get("max"),
                            }
                        rec = {
                            "ts": _ns_to_iso(dp.get("timeUnixNano")),
                            "start_ts": _ns_to_iso(dp.get("startTimeUnixNano")),
                            "metric": name,
                            "unit": unit,
                            "description": desc,
                            "type": type_key,
                            "value": val,
                            "attributes": dp_attrs,
                            "scope": scope_name,
                            "resource": res_attrs,
                            "_session_id": _resolve_session_key(dp_attrs, res_attrs),
                            "_provider": _resolve_provider_tag(dp_attrs, res_attrs),
                        }
                        out.append(rec)
    return out


# ════════════════════════════════════════════════════════════
#  Traces
# ════════════════════════════════════════════════════════════

def _flatten_span_records(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """ExportTraceServiceRequest → 扁平 span 列表。"""
    out: List[Dict[str, Any]] = []
    for rs in payload.get("resourceSpans") or []:
        res_attrs = _coerce_attributes((rs.get("resource") or {}).get("attributes") or [])
        for ss in rs.get("scopeSpans") or []:
            scope = ss.get("scope") or {}
            scope_name = scope.get("name") or ""
            for sp in ss.get("spans") or []:
                sp_attrs = _coerce_attributes(sp.get("attributes") or [])
                events = []
                for ev in sp.get("events") or []:
                    events.append({
                        "name": ev.get("name"),
                        "ts": _ns_to_iso(ev.get("timeUnixNano")),
                        "attributes": _coerce_attributes(ev.get("attributes") or []),
                    })
                rec = {
                    "trace_id": sp.get("traceId"),
                    "span_id": sp.get("spanId"),
                    "parent_span_id": sp.get("parentSpanId"),
                    "name": sp.get("name") or "",
                    "kind": sp.get("kind"),
                    "start_ts": _ns_to_iso(sp.get("startTimeUnixNano")),
                    "end_ts": _ns_to_iso(sp.get("endTimeUnixNano")),
                    "status": (sp.get("status") or {}).get("code"),
                    "status_message": (sp.get("status") or {}).get("message"),
                    "attributes": sp_attrs,
                    "events": events,
                    "scope": scope_name,
                    "resource": res_attrs,
                    "_session_id": _resolve_session_key(sp_attrs, res_attrs),
                    "_provider": _resolve_provider_tag(sp_attrs, res_attrs),
                }
                out.append(rec)
    return out


# ════════════════════════════════════════════════════════════
#  FastAPI handlers
# ════════════════════════════════════════════════════════════

async def _read_otlp_json(request: Request) -> Optional[Dict[str, Any]]:
    """读取 OTLP/JSON body。Claude Code 在 ``OTEL_EXPORTER_OTLP_PROTOCOL=http/json``
    下发的就是 ``application/json`` 单条 ExportXxxServiceRequest。

    解析失败或空 body 都返回 None，调用方走兜底路径。
    """
    try:
        raw = await request.body()
    except Exception:
        return None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


async def receive_otlp_logs(request: Request) -> Response:
    payload = await _read_otlp_json(request)
    if not payload:
        return Response(content='{"partialSuccess":{"rejectedLogRecords":0}}',
                        media_type="application/json")
    records = _flatten_log_records(payload)
    written_total = 0
    for sid, recs in _bucket_by_session(records).items():
        target = OTEL_ROOT / sid / "events.jsonl"
        written_total += _append_jsonl(target, recs)
    return Response(
        content=json.dumps({"partialSuccess": {"rejectedLogRecords": 0}, "_written": written_total}),
        media_type="application/json",
    )


async def receive_otlp_metrics(request: Request) -> Response:
    payload = await _read_otlp_json(request)
    if not payload:
        return Response(content='{"partialSuccess":{"rejectedDataPoints":0}}',
                        media_type="application/json")
    records = _flatten_metric_records(payload)
    written_total = 0
    for sid, recs in _bucket_by_session(records).items():
        target = OTEL_ROOT / sid / "metrics.jsonl"
        written_total += _append_jsonl(target, recs)
    return Response(
        content=json.dumps({"partialSuccess": {"rejectedDataPoints": 0}, "_written": written_total}),
        media_type="application/json",
    )


async def receive_otlp_traces(request: Request) -> Response:
    payload = await _read_otlp_json(request)
    if not payload:
        return Response(content='{"partialSuccess":{"rejectedSpans":0}}',
                        media_type="application/json")
    records = _flatten_span_records(payload)
    written_total = 0
    for sid, recs in _bucket_by_session(records).items():
        target = OTEL_ROOT / sid / "traces.jsonl"
        written_total += _append_jsonl(target, recs)
    return Response(
        content=json.dumps({"partialSuccess": {"rejectedSpans": 0}, "_written": written_total}),
        media_type="application/json",
    )


# ════════════════════════════════════════════════════════════
#  GET 接口：把已经落盘的 OTel 数据吐给前端
# ════════════════════════════════════════════════════════════

def _read_jsonl(target: Path, max_lines: int = 50000) -> List[Dict[str, Any]]:
    if not target.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(target, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                # 容忍一行解析失败，不影响整体
                continue
    return out


def _first_number(*values: Any) -> int:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _extract_usage_from_event(event: Dict[str, Any]) -> tuple[str | None, Dict[str, int]]:
    """Best-effort token extraction from Codex/OpenAI OTel log bodies."""
    candidates: List[Dict[str, Any]] = []
    for raw in (event.get("body"), event.get("attributes")):
        if isinstance(raw, dict):
            candidates.append(raw)
            response = raw.get("response")
            if isinstance(response, dict):
                candidates.append(response)
            usage = raw.get("usage")
            if isinstance(usage, dict):
                candidates.append(usage)
    model: str | None = None
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
    }
    for c in candidates:
        if model is None:
            raw_model = c.get("model") or c.get("gen_ai.response.model") or c.get("gen_ai.request.model")
            if isinstance(raw_model, str) and raw_model:
                model = raw_model
        totals["input_tokens"] += _first_number(
            c.get("input_tokens"),
            c.get("prompt_tokens"),
            c.get("inputTokenCount"),
            c.get("promptTokenCount"),
        )
        totals["output_tokens"] += _first_number(
            c.get("output_tokens"),
            c.get("completion_tokens"),
            c.get("outputTokenCount"),
            c.get("completionTokenCount"),
        )
        details = c.get("input_token_details") or c.get("prompt_token_details")
        if isinstance(details, dict):
            totals["cache_read_tokens"] += _first_number(
                details.get("cache_read_tokens"),
                details.get("cached_tokens"),
            )
            totals["cache_creation_tokens"] += _first_number(
                details.get("cache_creation_tokens"),
            )
    return model, totals


def load_session_otel(session_id: str) -> Dict[str, Any]:
    """读 ``~/.claude/state/otel/<sid>/`` 下三个 jsonl，组装成前端友好的结构。

    返回值：

        {
          "session_id": ...,
          "events": [...],     # 最多 5000 条，按时间正序
          "metrics": [...],    # 最多 20000 条
          "spans": [...],      # 最多 5000 条
          "summary": {
              "events_total": N,
              "metrics_total": N,
              "spans_total": N,
              "first_ts": iso,
              "last_ts": iso,
              "metrics_by_name": {name: count},
              "events_by_name": {name: count},
              "spans_by_name":  {name: count},
              "models":  {model: total_tokens},
              "totals":  {input_tokens, output_tokens, cache_read, cache_write, cost_usd},
          }
        }
    """
    sid = (session_id or "").strip()
    if not sid:
        return {"session_id": "", "events": [], "metrics": [], "spans": [], "summary": {}, "provider": "claude"}

    # v0.17.0: 兼容前端传裸 session_id（不带 vscode-/codebuddy- 前缀）。
    # 优先精确匹配；找不到时按裸 id 回退到 vscode-/codebuddy- 前缀目录。
    sess_dir = OTEL_ROOT / sid
    if not sess_dir.exists():
        for p in _KNOWN_PROVIDER_PREFIXES:
            cand = OTEL_ROOT / f"{p}{sid}"
            if cand.exists():
                sess_dir = cand
                sid = cand.name  # 回填带前缀的真名
                break
    provider, _bare = _split_provider_from_sid(sid)

    events = _read_jsonl(sess_dir / "events.jsonl", max_lines=5000)
    metrics = _read_jsonl(sess_dir / "metrics.jsonl", max_lines=20000)
    spans = _read_jsonl(sess_dir / "traces.jsonl", max_lines=5000)

    # 时间排序（缺 ts 的丢到末尾，相对稳定）
    def _ts_key(r: Dict[str, Any]) -> str:
        return (r.get("ts") or r.get("start_ts") or "9999")
    events.sort(key=_ts_key)
    metrics.sort(key=_ts_key)
    spans.sort(key=_ts_key)

    # ── summary ──
    metrics_by_name: Dict[str, int] = {}
    events_by_name: Dict[str, int] = {}
    spans_by_name: Dict[str, int] = {}
    models: Dict[str, int] = {}
    totals = {
        "input_tokens": 0, "output_tokens": 0,
        "cache_read_tokens": 0, "cache_creation_tokens": 0,
        "cost_usd": 0.0,
    }

    for m in metrics:
        nm = m.get("metric") or "?"
        metrics_by_name[nm] = metrics_by_name.get(nm, 0) + 1
        attrs = m.get("attributes") or {}
        mdl = (
            attrs.get("model")
            or attrs.get("gen_ai.response.model")
            or attrs.get("gen_ai.request.model")
        )
        v = m.get("value")
        if isinstance(mdl, str) and isinstance(v, (int, float)):
            token_metric = (
                nm == "claude_code.token.usage"
                or ("token" in nm and "usage" in nm)
            )
            if token_metric:
                models[mdl] = models.get(mdl, 0) + int(v)
                t = attrs.get("type") or attrs.get("token.type") or attrs.get("gen_ai.token.type")
                if t in ("input", "prompt"):
                    totals["input_tokens"] += int(v)
                elif t in ("output", "completion"):
                    totals["output_tokens"] += int(v)
                elif t in ("cacheRead", "cache_read"):
                    totals["cache_read_tokens"] += int(v)
                elif t in ("cacheCreation", "cache_creation"):
                    totals["cache_creation_tokens"] += int(v)
            elif nm == "claude_code.cost.usage" or ("cost" in nm and "usage" in nm):
                try:
                    totals["cost_usd"] += float(v)
                except (TypeError, ValueError):
                    pass

    for e in events:
        nm = e.get("event_name") or (e.get("attributes") or {}).get("event.name") or "?"
        events_by_name[nm] = events_by_name.get(nm, 0) + 1
        mdl, usage = _extract_usage_from_event(e)
        if mdl:
            total = usage["input_tokens"] + usage["output_tokens"]
            if total:
                models[mdl] = models.get(mdl, 0) + total
        for k, v in usage.items():
            totals[k] += v

    for sp in spans:
        nm = sp.get("name") or "?"
        spans_by_name[nm] = spans_by_name.get(nm, 0) + 1

    first_ts = None
    last_ts = None
    for col in (events, metrics, spans):
        for r in col:
            t = r.get("ts") or r.get("start_ts")
            if isinstance(t, str):
                if first_ts is None or t < first_ts:
                    first_ts = t
                if last_ts is None or t > last_ts:
                    last_ts = t

    return {
        "session_id": sid,
        "provider": provider or "claude",
        "events": events,
        "metrics": metrics,
        "spans": spans,
        "summary": {
            "events_total": len(events),
            "metrics_total": len(metrics),
            "spans_total": len(spans),
            "first_ts": first_ts,
            "last_ts": last_ts,
            "metrics_by_name": metrics_by_name,
            "events_by_name": events_by_name,
            "spans_by_name": spans_by_name,
            "models": models,
            "totals": totals,
        },
    }


_KNOWN_PROVIDER_PREFIXES = ("vscode-", "codebuddy-", "codex-")


def _split_provider_from_sid(sid: str) -> tuple[Optional[str], str]:
    """sid 形如 ``vscode-<uuid>`` / ``codebuddy-<uuid>`` 时拆出 provider 与裸 id；
    否则当作 Claude（向后兼容）。
    """
    for p in _KNOWN_PROVIDER_PREFIXES:
        if sid.startswith(p):
            return p[:-1], sid[len(p):]
    return None, sid


def list_otel_sessions() -> List[Dict[str, Any]]:
    """扫 OTEL_ROOT 下所有 session 目录，返回每个的 summary 概览。

    用于 SessionList 标记『此 session 已有 OTel 数据』徽章，以及让用户排查
    『为什么前端没显示 OTel』时一眼看到落盘有没有发生。

    v0.17.0：解析 vscode-/codebuddy- 前缀，暴露 ``provider`` 字段；前端按
    provider 切换徽章颜色 / 默认面板视图。
    """
    if not OTEL_ROOT.exists():
        return []
    out: List[Dict[str, Any]] = []
    for sub in sorted(OTEL_ROOT.iterdir()):
        if not sub.is_dir():
            continue
        sid = sub.name
        provider, bare_id = _split_provider_from_sid(sid)
        ev = sub / "events.jsonl"
        mt = sub / "metrics.jsonl"
        tr = sub / "traces.jsonl"
        out.append({
            "session_id": sid,             # 原始 dir name，保留前缀供前端定位
            "bare_session_id": bare_id,    # 去掉前缀的真实 session id
            "provider": provider or "claude",
            "events_bytes": ev.stat().st_size if ev.exists() else 0,
            "metrics_bytes": mt.stat().st_size if mt.exists() else 0,
            "spans_bytes": tr.stat().st_size if tr.exists() else 0,
            "last_modified": max(
                (p.stat().st_mtime for p in (ev, mt, tr) if p.exists()),
                default=0,
            ),
        })
    return out
