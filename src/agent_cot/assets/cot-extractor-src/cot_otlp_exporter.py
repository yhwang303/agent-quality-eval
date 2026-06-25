"""cot_otlp_exporter — 把 cot.json 重放为 OTLP/HTTP traces，推送到任意 OTel 兼容后端。

设计目标
========
1. **多后端支持**：通过标准 OTLP/HTTP 端点（默认 ``:4318/v1/traces``）即可
   推到 Phoenix / Langfuse / SigNoz / Jaeger / Datadog / Grafana Tempo /
   Honeycomb / 任意 OTel collector。
2. **复用本地分析**：``cot-extractor`` 仍生成自家 ``_cot.json``（本地前端的核心数据），
   OTLP 导出是 *追加* 的便利通道，不是替换。
3. **不修改原始数据**：本模块只读 ``cot.json``，把它重放为 OTel SDK Tracer span 树。
4. **真实时间还原**：尽量从 ``step.timestamp`` / ``step.duration_ms`` 还原 wall-clock，
   保证导出的 trace 在 UI 上时间线正确。

调用方式
========
- Python API：``export_session_to_otlp(cot_data, endpoint=..., headers=...)``
- CLI：``python -m scripts.export_otlp --session-id <sid> --endpoint http://...``
- HTTP API：``POST /api/sessions/{sid}/export/otlp``（agent-dashboard backend）

返回结构
========
::

    {
        "ok": true,
        "trace_id": "01abcdef...32 hex",
        "endpoint": "http://localhost:4318/v1/traces",
        "service_name": "cot-extractor",
        "span_count": 42,
        "dry_run": false,
        "sample_spans": [...]   # 仅 dry_run=True 时返回
    }
"""

from __future__ import annotations

import os
import time
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ─── 软依赖：opentelemetry SDK ─────────────────────────────
# 不安装 SDK 时只允许 dry_run=True（本地序列化预览），真实导出会抛 ImportError。
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SimpleSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.trace import SpanKind, Status, StatusCode

    _HAS_OTEL_SDK = True
except ImportError:  # pragma: no cover
    _HAS_OTEL_SDK = False

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    _HAS_OTLP_HTTP = True
except ImportError:  # pragma: no cover
    _HAS_OTLP_HTTP = False


# ─── 内部工具 ─────────────────────────────────────────────


def _iso_to_ns(ts: Optional[str]) -> Optional[int]:
    """ISO 时间字符串 → epoch nanoseconds。失败返回 None。"""
    if not ts or not isinstance(ts, str):
        return None
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def _now_ns() -> int:
    return int(time.time() * 1_000_000_000)


def _coerce_attr_value(v: Any) -> Any:
    """OTel SDK 只接受 str/bool/int/float/list[同型]，其它类型转 JSON 字符串。"""
    if v is None:
        return ""  # None 会被 SDK 丢弃，索性转空串保留 key
    if isinstance(v, (str, bool, int, float)):
        return v
    if isinstance(v, (list, tuple)):
        if not v:
            return []
        # 全是基本类型 → 直接传
        if all(isinstance(x, (str, bool, int, float)) for x in v):
            return list(v)
        # 否则压成 JSON 字符串数组
        return [json.dumps(x, ensure_ascii=False, default=str) for x in v]
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False, default=str)
    return str(v)


def _flatten_attrs(d: Dict[str, Any]) -> Dict[str, Any]:
    """把 attribute dict 整体洗成 OTel 合法值。"""
    if not isinstance(d, dict):
        return {}
    return {str(k): _coerce_attr_value(v) for k, v in d.items()}


def _step_span_kind(step_otel: Dict[str, Any]) -> "SpanKind":
    """根据 cot 的 step.otel.kind 映射到 OTel SpanKind。"""
    k = (step_otel or {}).get("kind") or "internal"
    mapping = {
        "internal": SpanKind.INTERNAL,
        "client": SpanKind.CLIENT,
        "server": SpanKind.SERVER,
        "producer": SpanKind.PRODUCER,
        "consumer": SpanKind.CONSUMER,
    }
    return mapping.get(k.lower(), SpanKind.INTERNAL)


def _add_genai_message_events(
    span: "trace.Span",
    step_otel: Dict[str, Any],
    base_ts_ns: int,
) -> None:
    """把 input_messages / output_messages 拍成 OTel GenAI events。

    依据 OTel GenAI semantic conventions（v1.27+）：
      - gen_ai.user.message
      - gen_ai.assistant.message
      - gen_ai.tool.message
      - gen_ai.choice
    """
    # input messages
    for i, m in enumerate(step_otel.get("input_messages") or []):
        role = (m.get("role") or "user").lower()
        ev_name = {
            "user": "gen_ai.user.message",
            "system": "gen_ai.system.message",
            "tool": "gen_ai.tool.message",
        }.get(role, "gen_ai.user.message")
        attrs = {
            "gen_ai.message.role": role,
            "gen_ai.message.index": i,
            "gen_ai.message.content": json.dumps(
                m.get("parts") or m.get("content") or "", ensure_ascii=False, default=str
            ),
        }
        if m.get("tool_call_id"):
            attrs["gen_ai.tool.call.id"] = str(m["tool_call_id"])
        try:
            span.add_event(ev_name, attributes=_flatten_attrs(attrs), timestamp=base_ts_ns)
        except Exception:
            pass

    # output messages → gen_ai.choice
    for i, m in enumerate(step_otel.get("output_messages") or []):
        role = (m.get("role") or "assistant").lower()
        attrs = {
            "gen_ai.message.role": role,
            "gen_ai.choice.index": i,
            "gen_ai.choice.finish_reason": m.get("finish_reason") or "stop",
            "gen_ai.choice.content": json.dumps(
                m.get("parts") or m.get("content") or "", ensure_ascii=False, default=str
            ),
        }
        if m.get("tool_call_id"):
            attrs["gen_ai.tool.call.id"] = str(m["tool_call_id"])
        if m.get("tool_calls"):
            attrs["gen_ai.choice.tool_calls"] = json.dumps(
                m["tool_calls"], ensure_ascii=False, default=str
            )
        try:
            span.add_event(
                "gen_ai.choice", attributes=_flatten_attrs(attrs), timestamp=base_ts_ns
            )
        except Exception:
            pass

    # retrieval documents → gen_ai.retrieval.document events
    for i, doc in enumerate(step_otel.get("retrieval_documents") or []):
        attrs = {
            "retrieval.document.index": i,
            "retrieval.document.content": json.dumps(
                doc, ensure_ascii=False, default=str
            ),
        }
        try:
            span.add_event(
                "gen_ai.retrieval.document",
                attributes=_flatten_attrs(attrs),
                timestamp=base_ts_ns,
            )
        except Exception:
            pass


# ─── 主入口 ───────────────────────────────────────────────


def export_session_to_otlp(
    cot_data: Dict[str, Any],
    *,
    endpoint: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    service_name: str = "cot-extractor",
    service_version: str = "v0.12.0",
    deployment_environment: Optional[str] = None,
    dry_run: bool = False,
    timeout: float = 10.0,
    extra_resource_attrs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """读 cot.json (dict)，重放为 OTLP/HTTP traces 推送出去。

    - ``cot_data``: ``json.load(open(_cot.json))`` 的 dict。
    - ``endpoint``: OTLP/HTTP traces endpoint，默认 ``http://localhost:4318/v1/traces``。
      若环境变量 ``OTEL_EXPORTER_OTLP_ENDPOINT`` 已设置且未传 endpoint，会自动用它。
    - ``headers``: 给 OTLP 请求加 header（如 Langfuse/Honeycomb 鉴权）。
    - ``dry_run=True``: 不连后端，把 span 收到内存 exporter 里，序列化预览。
    """
    if not _HAS_OTEL_SDK:
        raise RuntimeError(
            "opentelemetry-sdk 未安装。请：\n"
            "  pip install -r cot-extractor/requirements.txt\n"
            "或单独安装：\n"
            "  pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http"
        )

    if not isinstance(cot_data, dict):
        raise TypeError("cot_data 必须是 dict（cot.json 的解析结果）")

    session_id = cot_data.get("session_id") or "unknown"
    otel_view = cot_data.get("otel_view") or {}
    resource_attrs_raw = (
        cot_data.get("resource_attributes")
        or otel_view.get("service")  # fallback
        or {}
    )

    # ─ Resource ─
    resource_attrs: Dict[str, Any] = {
        "service.name": service_name,
        "service.version": service_version,
    }
    if isinstance(resource_attrs_raw, dict):
        resource_attrs.update(resource_attrs_raw)
        resource_attrs["service.name"] = service_name
        resource_attrs["service.version"] = service_version
    if deployment_environment:
        resource_attrs["deployment.environment"] = deployment_environment
    if extra_resource_attrs:
        resource_attrs.update(extra_resource_attrs)
    resource = Resource.create(_flatten_attrs(resource_attrs))

    # ─ Provider + Exporter ─
    provider = TracerProvider(resource=resource)
    in_memory: Optional["InMemorySpanExporter"] = None
    real_endpoint: Optional[str] = None

    if dry_run:
        in_memory = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(in_memory))
    else:
        if not _HAS_OTLP_HTTP:
            raise RuntimeError(
                "opentelemetry-exporter-otlp-proto-http 未安装。\n"
                "请：pip install opentelemetry-exporter-otlp-proto-http"
            )
        real_endpoint = (
            endpoint
            or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
            or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
            or "http://localhost:4318/v1/traces"
        )
        # 用户经常给的是 base url（http://host:4318），帮他自动补 /v1/traces
        if real_endpoint and not real_endpoint.rstrip("/").endswith("/v1/traces"):
            real_endpoint = real_endpoint.rstrip("/") + "/v1/traces"

        otlp_exporter = OTLPSpanExporter(
            endpoint=real_endpoint,
            headers=headers or None,
            timeout=int(timeout),
        )
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # 不污染全局 TracerProvider —— 用一个 provider 局部 tracer
    tracer = provider.get_tracer("cot_otlp_exporter", service_version)

    # ─ Span 树重建 ─
    turns = cot_data.get("turns") or []
    session_attrs = _flatten_attrs(otel_view.get("attributes") or {})
    session_attrs["cot.session.id"] = session_id
    session_attrs["cot.totals.turns"] = len(turns)
    session_attrs["cot.totals.steps"] = sum(len(t.get("steps") or []) for t in turns)
    if isinstance(otel_view.get("token_usage"), dict):
        tu = otel_view["token_usage"]
        if tu.get("cost_usd") is not None:
            session_attrs["gen_ai.usage.cost.usd"] = float(tu["cost_usd"])

    # session 起止时间
    sess_start_ns: Optional[int] = None
    sess_end_ns: Optional[int] = None
    for turn in turns:
        for step in turn.get("steps") or []:
            ts_ns = _iso_to_ns(step.get("timestamp"))
            if ts_ns:
                sess_start_ns = ts_ns if sess_start_ns is None else min(sess_start_ns, ts_ns)
                # 含 duration 的取 step end
                end = ts_ns + int((step.get("duration_ms") or 0) * 1_000_000)
                sess_end_ns = end if sess_end_ns is None else max(sess_end_ns, end)
    if sess_start_ns is None:
        sess_start_ns = _now_ns() - 1_000_000  # 1 ms
    if sess_end_ns is None or sess_end_ns <= sess_start_ns:
        sess_end_ns = sess_start_ns + 1_000_000

    span_count = 0

    # ── root span：session ──
    root_span = tracer.start_span(
        name=f"agent.session {session_id[:8]}",
        kind=SpanKind.SERVER,
        attributes=session_attrs,
        start_time=sess_start_ns,
    )
    span_count += 1
    trace_id_int = root_span.get_span_context().trace_id
    trace_id_hex = format(trace_id_int, "032x")

    root_ctx = trace.set_span_in_context(root_span)

    try:
        for turn in turns:
            turn_otel = turn.get("otel") or {}
            turn_attrs = _flatten_attrs(turn_otel.get("attributes") or {})
            turn_attrs.update(_flatten_attrs({
                "cot.turn.index": turn.get("turn_index"),
                "cot.turn.steps": len(turn.get("steps") or []),
            }))
            tu = (turn_otel.get("token_usage") or {})
            if tu.get("cost_usd") is not None:
                turn_attrs["gen_ai.usage.cost.usd"] = float(tu["cost_usd"])

            # turn 时间窗
            turn_start_ns = _iso_to_ns(turn.get("turn_start_time"))
            steps = turn.get("steps") or []
            if turn_start_ns is None and steps:
                turn_start_ns = _iso_to_ns(steps[0].get("timestamp"))
            if turn_start_ns is None:
                turn_start_ns = sess_start_ns

            turn_end_ns: Optional[int] = None
            if turn.get("turn_duration_ms"):
                turn_end_ns = turn_start_ns + int(turn["turn_duration_ms"] * 1_000_000)
            elif steps:
                last = steps[-1]
                last_ts = _iso_to_ns(last.get("timestamp")) or turn_start_ns
                turn_end_ns = last_ts + int((last.get("duration_ms") or 1) * 1_000_000)
            else:
                turn_end_ns = turn_start_ns + 1_000_000

            turn_span = tracer.start_span(
                name=f"agent.turn[{turn.get('turn_index')}]",
                kind=SpanKind.SERVER,
                context=root_ctx,
                attributes=turn_attrs,
                start_time=turn_start_ns,
            )
            span_count += 1
            turn_ctx = trace.set_span_in_context(turn_span)

            # 每个 step 一个 span
            prev_end_ns = turn_start_ns
            for step in steps:
                step_otel = step.get("otel") or {}
                step_attrs = _flatten_attrs(step_otel.get("attributes") or {})
                # 把一些常用 step 元信息也塞进去（cot 自有命名空间，不会和 gen_ai.* 冲突）
                step_attrs.update(_flatten_attrs({
                    "cot.step.index": step.get("step_index"),
                    "cot.step.type": step.get("step_type"),
                    "cot.step.tool_name": step.get("tool_name"),
                    "cot.step.kind": step_otel.get("step_kind"),
                    "cot.step.model_source": step_otel.get("model_source"),
                }))
                step_tu = step_otel.get("token_usage") or {}
                if step_tu.get("cost_usd") is not None:
                    step_attrs["gen_ai.usage.cost.usd"] = float(step_tu["cost_usd"])

                # 时间还原
                start_ns = _iso_to_ns(step.get("timestamp")) or prev_end_ns
                duration_ms = step.get("duration_ms") or 1
                end_ns = start_ns + int(max(1, duration_ms) * 1_000_000)
                if end_ns <= start_ns:
                    end_ns = start_ns + 1_000_000

                op_name = step_otel.get("operation_name") or "chat"
                step_name = f"{op_name}: {step.get('step_type') or 'step'}"
                if step.get("tool_name"):
                    step_name = f"{op_name}: {step['tool_name']}"

                step_span = tracer.start_span(
                    name=step_name,
                    kind=_step_span_kind(step_otel),
                    context=turn_ctx,
                    attributes=step_attrs,
                    start_time=start_ns,
                )
                span_count += 1

                # GenAI events（messages / retrieval）
                _add_genai_message_events(step_span, step_otel, start_ns)

                # error status
                md = step.get("metadata") or {}
                if md.get("is_error"):
                    try:
                        step_span.set_status(
                            Status(StatusCode.ERROR, description=str(md.get("error_type") or "error"))
                        )
                    except Exception:
                        pass

                step_span.end(end_time=end_ns)
                prev_end_ns = end_ns

            turn_span.end(end_time=turn_end_ns or sess_end_ns)
    finally:
        root_span.end(end_time=sess_end_ns)

    # 强制 flush，确保 BatchSpanProcessor 在函数返回前把数据真的发出去
    provider.shutdown()

    result: Dict[str, Any] = {
        "ok": True,
        "trace_id": trace_id_hex,
        "endpoint": real_endpoint,
        "service_name": service_name,
        "span_count": span_count,
        "dry_run": dry_run,
    }

    if dry_run and in_memory is not None:
        # 把 in-memory exporter 的 span 序列化预览（前 10 条 + 总数）
        spans = in_memory.get_finished_spans()
        sample: List[Dict[str, Any]] = []
        for sp in spans[:10]:
            try:
                sample.append({
                    "name": sp.name,
                    "kind": str(sp.kind),
                    "trace_id": format(sp.context.trace_id, "032x"),
                    "span_id": format(sp.context.span_id, "016x"),
                    "parent_span_id": (
                        format(sp.parent.span_id, "016x") if sp.parent else None
                    ),
                    "start_time_ns": sp.start_time,
                    "end_time_ns": sp.end_time,
                    "duration_ms": (
                        (sp.end_time - sp.start_time) / 1_000_000 if sp.end_time and sp.start_time else None
                    ),
                    "attributes": dict(sp.attributes or {}),
                    "events": [
                        {"name": e.name, "attributes": dict(e.attributes or {})}
                        for e in (sp.events or [])
                    ],
                })
            except Exception:
                pass
        result["sample_spans"] = sample
        result["sample_total"] = len(spans)

    return result


def export_session_file_to_otlp(
    cot_path: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """便捷入口：直接传 cot.json 路径。"""
    with open(cot_path, "r", encoding="utf-8") as f:
        cot_data = json.load(f)
    return export_session_to_otlp(cot_data, **kwargs)


# ─── v0.14.8: 离线 OTLP/JSON 协议导出 ──────────────────────
#
# `export_session_to_otlp` 走的是 OTLP/HTTP 实时推送（dry_run 时只返回前 10 条
# sample 预览，不是协议文件）。但用户需求是"我点一下按钮就拿到一份完整的 OTLP/JSON
# 协议文件"——可以离线交给 Phoenix / SigNoz / Jaeger CLI / otel-cli / 任何二次
# 处理脚本。这一段就是实现：
#
#   1. 把 cot.json 重放为 OTel SDK Span 树（复用上面的 InMemorySpanExporter 路径）
#   2. 把 ReadableSpan 列表序列化成 OTLP/JSON proto 协议格式
#      （resourceSpans → scopeSpans → spans，AnyValue 用 stringValue/intValue/...）
#   3. 整个 payload dump 出来即可 .json 落盘
#
# Why not 直接复用 export_session_to_otlp(dry_run=True)：
#   - 现有的 dry_run 把 sample_spans 截到前 10 条（UI 预览语义），不是完整协议
#   - 导出协议文件时没必要带 ok / span_count / endpoint 这些壳字段
#   - 协议序列化逻辑独立放在这里，与 OTLP/HTTP exporter 解耦，便于以后
#     单独替换（比如改输出 protobuf 二进制）
#
# 协议字段对照参考：
#   https://github.com/open-telemetry/opentelemetry-proto/blob/main/opentelemetry/proto/trace/v1/trace.proto


# OTLP/JSON span.kind 枚举 —— 必须与 proto 定义一致（int 数字，不是字符串），
# 否则 Phoenix / Jaeger 等下游会报 "unknown span kind"
_OTLP_SPAN_KIND = {
    # SpanKind 枚举值 → OTLP/JSON int
    # (OTel SDK 里 SpanKind.INTERNAL == 1，但有些版本是 0；以名字为准)
    "INTERNAL": 1,
    "SERVER": 2,
    "CLIENT": 3,
    "PRODUCER": 4,
    "CONSUMER": 5,
}


def _otlp_kind_int(kind: Any) -> int:
    """SDK SpanKind → OTLP/JSON int（容错：拿不到归 INTERNAL=1）。"""
    if kind is None:
        return 1
    name = getattr(kind, "name", None) or str(kind).rsplit(".", 1)[-1]
    return _OTLP_SPAN_KIND.get(name.upper(), 1)


def _otlp_any_value(v: Any) -> Dict[str, Any]:
    """Python 值 → OTLP/JSON AnyValue。

    需要按 proto 定义的 union 字段名输出（stringValue / intValue / ...），不能
    直接 json.dumps 原生 dict。注意 bool 必须在 int 之前判断（True isinstance int）。
    Int64 用字符串避免 JS 这一侧 53-bit 精度损失（OTel proto JSON 习惯）。
    """
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    if isinstance(v, str):
        return {"stringValue": v}
    if isinstance(v, (list, tuple)):
        return {"arrayValue": {"values": [_otlp_any_value(x) for x in v]}}
    # 兜底：不认识的类型转 JSON 字符串
    return {"stringValue": json.dumps(v, ensure_ascii=False, default=str)}


def _otlp_kv_list(attrs: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """attrs dict → OTLP/JSON KeyValue[]。"""
    if not attrs:
        return []
    return [{"key": str(k), "value": _otlp_any_value(v)} for k, v in attrs.items()]


def _otlp_status(status: Any) -> Optional[Dict[str, Any]]:
    """SDK Status → OTLP/JSON Status。Unset 直接省略以减小体积。"""
    if status is None:
        return None
    code = getattr(status, "status_code", None)
    code_name = getattr(code, "name", str(code) if code is not None else "")
    # OTLP proto: 0=UNSET, 1=OK, 2=ERROR
    code_int = {"UNSET": 0, "OK": 1, "ERROR": 2}.get(code_name.upper(), 0)
    if code_int == 0:
        return None  # UNSET 协议允许省略
    out: Dict[str, Any] = {"code": code_int}
    desc = getattr(status, "description", None)
    if desc:
        out["message"] = str(desc)
    return out


def _readable_span_to_otlp(sp: Any) -> Dict[str, Any]:
    """OTel SDK ReadableSpan → OTLP/JSON spans[i] 结构。"""
    ctx = sp.context
    span: Dict[str, Any] = {
        "traceId": format(ctx.trace_id, "032x"),
        "spanId": format(ctx.span_id, "016x"),
        "name": sp.name or "",
        "kind": _otlp_kind_int(sp.kind),
        "startTimeUnixNano": str(sp.start_time) if sp.start_time else "0",
        "endTimeUnixNano": str(sp.end_time) if sp.end_time else "0",
        "attributes": _otlp_kv_list(dict(sp.attributes or {})),
        # 协议字段：dropped_*_count 不写也合法，省略保持紧凑
    }
    if sp.parent is not None:
        span["parentSpanId"] = format(sp.parent.span_id, "016x")
    if sp.events:
        span["events"] = [
            {
                "timeUnixNano": str(e.timestamp) if getattr(e, "timestamp", None) else "0",
                "name": e.name,
                "attributes": _otlp_kv_list(dict(e.attributes or {})),
            }
            for e in sp.events
        ]
    if sp.links:
        span["links"] = [
            {
                "traceId": format(lk.context.trace_id, "032x"),
                "spanId": format(lk.context.span_id, "016x"),
                "attributes": _otlp_kv_list(dict(lk.attributes or {})),
            }
            for lk in sp.links
        ]
    st = _otlp_status(getattr(sp, "status", None))
    if st is not None:
        span["status"] = st
    return span


def build_session_otlp_json(
    cot_data: Dict[str, Any],
    *,
    service_name: str = "cot-extractor",
    service_version: str = "v0.12.0",
    deployment_environment: Optional[str] = None,
    extra_resource_attrs: Optional[Dict[str, Any]] = None,
    scope_name: str = "cot_otlp_exporter",
) -> Dict[str, Any]:
    """重放 cot.json 为 OTel Span 树并序列化成 OTLP/JSON 协议 payload。

    返回值就是一份合法的 ``{"resourceSpans": [...]}`` —— 可以直接 ``json.dump``
    成文件，喂给 Phoenix / Jaeger / SigNoz / otel-cli 等任意 OTLP/JSON 兼容工具。

    复用 ``export_session_to_otlp(dry_run=True)`` 内部的 InMemorySpanExporter 把
    span 收下来，然后用本文件的 ``_readable_span_to_otlp`` 转为协议格式。
    """
    if not _HAS_OTEL_SDK:
        raise RuntimeError(
            "opentelemetry-sdk 未安装。请：\n"
            "  pip install -r cot-extractor/requirements.txt"
        )
    if not isinstance(cot_data, dict):
        raise TypeError("cot_data 必须是 dict（cot.json 的解析结果）")

    # 借 dry_run=True 路径搭出 span 树，但不限于 sample 前 10
    # 实现：复制 export_session_to_otlp 的 span 构造逻辑，把所有 finished_spans 都收下
    # （为了不在 export_session_to_otlp 上加新参数破坏现有契约，这里独立跑一遍 SDK）
    session_id = cot_data.get("session_id") or "unknown"
    otel_view = cot_data.get("otel_view") or {}
    resource_attrs_raw = (
        cot_data.get("resource_attributes")
        or otel_view.get("service")
        or {}
    )
    resource_attrs: Dict[str, Any] = {
        "service.name": service_name,
        "service.version": service_version,
    }
    if isinstance(resource_attrs_raw, dict):
        resource_attrs.update(resource_attrs_raw)
        resource_attrs["service.name"] = service_name
        resource_attrs["service.version"] = service_version
    if deployment_environment:
        resource_attrs["deployment.environment"] = deployment_environment
    if extra_resource_attrs:
        resource_attrs.update(extra_resource_attrs)
    resource = Resource.create(_flatten_attrs(resource_attrs))

    provider = TracerProvider(resource=resource)
    in_memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(in_memory))
    tracer = provider.get_tracer(scope_name, service_version)

    turns = cot_data.get("turns") or []
    session_attrs = _flatten_attrs(otel_view.get("attributes") or {})
    session_attrs["cot.session.id"] = session_id
    session_attrs["cot.totals.turns"] = len(turns)
    session_attrs["cot.totals.steps"] = sum(len(t.get("steps") or []) for t in turns)
    if isinstance(otel_view.get("token_usage"), dict):
        tu = otel_view["token_usage"]
        if tu.get("cost_usd") is not None:
            session_attrs["gen_ai.usage.cost.usd"] = float(tu["cost_usd"])

    sess_start_ns: Optional[int] = None
    sess_end_ns: Optional[int] = None
    for turn in turns:
        for step in turn.get("steps") or []:
            ts_ns = _iso_to_ns(step.get("timestamp"))
            if ts_ns:
                sess_start_ns = ts_ns if sess_start_ns is None else min(sess_start_ns, ts_ns)
                end = ts_ns + int((step.get("duration_ms") or 0) * 1_000_000)
                sess_end_ns = end if sess_end_ns is None else max(sess_end_ns, end)
    if sess_start_ns is None:
        sess_start_ns = _now_ns() - 1_000_000
    if sess_end_ns is None or sess_end_ns <= sess_start_ns:
        sess_end_ns = sess_start_ns + 1_000_000

    root_span = tracer.start_span(
        name=f"agent.session {session_id[:8]}",
        kind=SpanKind.SERVER,
        attributes=session_attrs,
        start_time=sess_start_ns,
    )
    root_ctx = trace.set_span_in_context(root_span)

    try:
        for turn in turns:
            turn_otel = turn.get("otel") or {}
            turn_attrs = _flatten_attrs(turn_otel.get("attributes") or {})
            turn_attrs.update(_flatten_attrs({
                "cot.turn.index": turn.get("turn_index"),
                "cot.turn.steps": len(turn.get("steps") or []),
            }))
            tu = (turn_otel.get("token_usage") or {})
            if tu.get("cost_usd") is not None:
                turn_attrs["gen_ai.usage.cost.usd"] = float(tu["cost_usd"])

            turn_start_ns = _iso_to_ns(turn.get("turn_start_time"))
            steps = turn.get("steps") or []
            if turn_start_ns is None and steps:
                turn_start_ns = _iso_to_ns(steps[0].get("timestamp"))
            if turn_start_ns is None:
                turn_start_ns = sess_start_ns

            turn_end_ns: Optional[int] = None
            if turn.get("turn_duration_ms"):
                turn_end_ns = turn_start_ns + int(turn["turn_duration_ms"] * 1_000_000)
            elif steps:
                last = steps[-1]
                last_ts = _iso_to_ns(last.get("timestamp")) or turn_start_ns
                turn_end_ns = last_ts + int((last.get("duration_ms") or 1) * 1_000_000)
            else:
                turn_end_ns = turn_start_ns + 1_000_000

            turn_span = tracer.start_span(
                name=f"agent.turn[{turn.get('turn_index')}]",
                kind=SpanKind.SERVER,
                context=root_ctx,
                attributes=turn_attrs,
                start_time=turn_start_ns,
            )
            turn_ctx = trace.set_span_in_context(turn_span)

            prev_end_ns = turn_start_ns
            for step in steps:
                step_otel = step.get("otel") or {}
                step_attrs = _flatten_attrs(step_otel.get("attributes") or {})
                step_attrs.update(_flatten_attrs({
                    "cot.step.index": step.get("step_index"),
                    "cot.step.type": step.get("step_type"),
                    "cot.step.tool_name": step.get("tool_name"),
                    "cot.step.kind": step_otel.get("step_kind"),
                    "cot.step.model_source": step_otel.get("model_source"),
                }))
                step_tu = step_otel.get("token_usage") or {}
                if step_tu.get("cost_usd") is not None:
                    step_attrs["gen_ai.usage.cost.usd"] = float(step_tu["cost_usd"])

                start_ns = _iso_to_ns(step.get("timestamp")) or prev_end_ns
                duration_ms = step.get("duration_ms") or 1
                end_ns = start_ns + int(max(1, duration_ms) * 1_000_000)
                if end_ns <= start_ns:
                    end_ns = start_ns + 1_000_000

                op_name = step_otel.get("operation_name") or "chat"
                step_name = f"{op_name}: {step.get('step_type') or 'step'}"
                if step.get("tool_name"):
                    step_name = f"{op_name}: {step['tool_name']}"

                step_span = tracer.start_span(
                    name=step_name,
                    kind=_step_span_kind(step_otel),
                    context=turn_ctx,
                    attributes=step_attrs,
                    start_time=start_ns,
                )
                _add_genai_message_events(step_span, step_otel, start_ns)

                md = step.get("metadata") or {}
                if md.get("is_error"):
                    try:
                        step_span.set_status(
                            Status(StatusCode.ERROR, description=str(md.get("error_type") or "error"))
                        )
                    except Exception:
                        pass
                step_span.end(end_time=end_ns)
                prev_end_ns = end_ns
            turn_span.end(end_time=turn_end_ns or sess_end_ns)
    finally:
        root_span.end(end_time=sess_end_ns)

    provider.shutdown()
    spans = in_memory.get_finished_spans()

    # spans 序列化成 OTLP/JSON 协议格式
    otlp_spans = [_readable_span_to_otlp(sp) for sp in spans]

    # 顶层 trace_id（root span 的）方便 UI / 后端搜索
    root_trace_id_hex: Optional[str] = None
    for sp in spans:
        if sp.parent is None:
            root_trace_id_hex = format(sp.context.trace_id, "032x")
            break

    payload: Dict[str, Any] = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": _otlp_kv_list(_flatten_attrs(resource_attrs)),
                },
                "scopeSpans": [
                    {
                        "scope": {
                            "name": scope_name,
                            "version": service_version,
                        },
                        "spans": otlp_spans,
                    }
                ],
            }
        ],
        # 非协议字段，但对调试 / UI 提示有用 —— OTLP receiver 会忽略未知顶层字段
        "_meta": {
            "session_id": session_id,
            "trace_id": root_trace_id_hex,
            "span_count": len(otlp_spans),
            "schema": "opentelemetry.proto.collector.trace.v1.ExportTraceServiceRequest (JSON)",
            "schema_url": (
                "https://github.com/open-telemetry/opentelemetry-proto/blob/main/"
                "opentelemetry/proto/trace/v1/trace.proto"
            ),
            "exporter": "cot_otlp_exporter.build_session_otlp_json",
            "exporter_version": service_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    return payload


# ─── 后端 preset（给 UI 用）─────────────────────────────────


BACKEND_PRESETS: List[Dict[str, Any]] = [
    {
        "id": "otel-collector",
        "label": "OTel Collector (本地)",
        "endpoint": "http://localhost:4318/v1/traces",
        "headers_hint": {},
        "doc": "通用 OpenTelemetry collector，4318 是 OTLP/HTTP 默认端口。",
    },
    {
        "id": "phoenix",
        "label": "Arize Phoenix (本地)",
        "endpoint": "http://localhost:6006/v1/traces",
        "headers_hint": {},
        "doc": "Phoenix 默认在 6006 暴露 OTLP/HTTP，启动：``pip install arize-phoenix && phoenix serve``。",
    },
    {
        "id": "langfuse-cloud",
        "label": "Langfuse Cloud",
        "endpoint": "https://cloud.langfuse.com/api/public/otel/v1/traces",
        "headers_hint": {
            "Authorization": "Basic <base64(public_key:secret_key)>",
        },
        "doc": "Langfuse 把 OTLP 包成自家 ingestion，Authorization 用 Basic auth。",
    },
    {
        "id": "signoz",
        "label": "SigNoz (Self-hosted)",
        "endpoint": "http://localhost:4318/v1/traces",
        "headers_hint": {"signoz-access-token": "<token>"},
        "doc": "SigNoz 自托管默认走 4318，云版本端点替换成 ingest.{region}.signoz.cloud。",
    },
    {
        "id": "jaeger",
        "label": "Jaeger (OTLP)",
        "endpoint": "http://localhost:4318/v1/traces",
        "headers_hint": {},
        "doc": "Jaeger 1.49+ 原生支持 OTLP/HTTP，端口 4318。",
    },
    {
        "id": "honeycomb",
        "label": "Honeycomb",
        "endpoint": "https://api.honeycomb.io/v1/traces",
        "headers_hint": {"x-honeycomb-team": "<api_key>"},
        "doc": "Honeycomb 直接接收 OTLP/HTTP，header 放 API key。",
    },
    {
        "id": "datadog",
        "label": "Datadog (OTLP via Agent)",
        "endpoint": "http://localhost:4318/v1/traces",
        "headers_hint": {},
        "doc": "Datadog Agent 7.42+ 自带 OTLP receiver；云端则推 collector。",
    },
]


__all__ = [
    "export_session_to_otlp",
    "export_session_file_to_otlp",
    "build_session_otlp_json",
    "BACKEND_PRESETS",
]
