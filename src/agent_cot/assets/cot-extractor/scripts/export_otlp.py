#!/usr/bin/env python
"""export_otlp.py — 把本地 cot.json 重放为 OTLP/HTTP traces 推到任意 OTel 后端。

用法
====

直接传 session id（自动找最新 cot.json）::

    python -m scripts.export_otlp --session-id <sid>
    python -m scripts.export_otlp --session-id <sid> --endpoint http://localhost:4318

直接传 cot.json 路径::

    python -m scripts.export_otlp --cot-path output/cot/<sid>_cot.json

dry-run（不真发，本地预览 span 结构）::

    python -m scripts.export_otlp --session-id <sid> --dry-run

带鉴权 header（Honeycomb / Langfuse 等）::

    python -m scripts.export_otlp --session-id <sid> \
        --endpoint https://api.honeycomb.io/v1/traces \
        --header "x-honeycomb-team=YOUR_KEY"

环境变量
========
- ``OTEL_EXPORTER_OTLP_ENDPOINT``：默认 endpoint（traces 路径会自动补 ``/v1/traces``）
- ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``：仅 traces endpoint
- ``OTEL_SERVICE_NAME``：覆盖 service.name
- ``COT_DIR``：cot.json 所在目录（默认 ``cot-extractor/output/cot``）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional


# 让脚本既能 ``python -m scripts.export_otlp`` 跑，也能 ``python scripts/export_otlp.py`` 直跑
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent  # cot-extractor/
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src.cot_otlp_exporter import (  # noqa: E402
    export_session_to_otlp,
    BACKEND_PRESETS,
)


def _default_cot_dir() -> Path:
    env = os.environ.get("COT_DIR")
    if env:
        return Path(env)
    return _ROOT / "output" / "cot"


def _resolve_cot_path(session_id: Optional[str], cot_path: Optional[str]) -> Path:
    """根据 --session-id / --cot-path 推断 cot.json 实际路径。

    支持 layout：
      1. ``<COT_DIR>/<sid>_cot.json``（扁平）
      2. ``<COT_DIR>/../sessions/<sid>/<timestamp>_cot.json``（增量目录最新一份）
    """
    if cot_path:
        p = Path(cot_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"cot.json 不存在：{p}")
        return p

    if not session_id:
        raise ValueError("必须提供 --session-id 或 --cot-path 之一")

    cot_dir = _default_cot_dir()
    flat = cot_dir / f"{session_id}_cot.json"
    if flat.exists():
        return flat

    sessions_dir = cot_dir.parent / "sessions" / session_id
    if sessions_dir.exists():
        candidates = sorted(sessions_dir.glob("*_cot.json"))
        if candidates:
            return candidates[-1]

    raise FileNotFoundError(
        f"找不到 session={session_id} 的 cot.json。已尝试：\n"
        f"  - {flat}\n"
        f"  - {sessions_dir}/*_cot.json\n"
        "可手动用 --cot-path 指定。"
    )


def _parse_headers(items: List[str]) -> Dict[str, str]:
    """把 ['k1=v1', 'k2=v2'] 解析成 dict；也支持 ``k1: v1`` 形式。"""
    out: Dict[str, str] = {}
    for it in items or []:
        if "=" in it:
            k, v = it.split("=", 1)
        elif ":" in it:
            k, v = it.split(":", 1)
        else:
            raise ValueError(f"--header 格式必须是 ``k=v`` 或 ``k: v``：{it}")
        out[k.strip()] = v.strip()
    return out


def _print_presets() -> None:
    print("可选后端 preset（--preset <id> 自动填 endpoint）：")
    for p in BACKEND_PRESETS:
        print(f"  • [{p['id']:<16}] {p['label']:<28} {p['endpoint']}")
        if p.get("doc"):
            print(f"      ↳ {p['doc']}")
        if p.get("headers_hint"):
            print(f"      ↳ headers: {p['headers_hint']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="export_otlp",
        description="把 cot.json 重放为 OTLP/HTTP traces，导出到任意 OTel 后端",
    )
    parser.add_argument("--session-id", help="cot 的 session_id（自动定位 cot.json）")
    parser.add_argument("--cot-path", help="直接传 cot.json 路径（优先于 --session-id）")
    parser.add_argument(
        "--endpoint",
        help="OTLP/HTTP traces 端点；缺省读 OTEL_EXPORTER_OTLP_ENDPOINT 或 http://localhost:4318/v1/traces",
    )
    parser.add_argument(
        "--preset",
        help="后端预设 id（如 phoenix / langfuse-cloud / signoz / jaeger）",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="HTTP header（可多次指定），形如 ``k=v``",
    )
    parser.add_argument(
        "--service-name",
        default=os.environ.get("OTEL_SERVICE_NAME", "cot-extractor"),
        help="service.name（resource attribute）",
    )
    parser.add_argument(
        "--service-version",
        default="v0.12.0",
    )
    parser.add_argument(
        "--env",
        dest="environment",
        default=os.environ.get("COT_ENV", "local-dev"),
        help="deployment.environment",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="OTLP HTTP 超时（秒）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="不连后端，只本地把 span 树序列化预览（用于调试）",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="列出所有内置后端 preset 后退出",
    )
    args = parser.parse_args()

    if args.list_presets:
        _print_presets()
        return 0

    # preset → endpoint
    endpoint = args.endpoint
    if args.preset:
        preset = next((p for p in BACKEND_PRESETS if p["id"] == args.preset), None)
        if not preset:
            print(f"未知 preset: {args.preset}。可用：", file=sys.stderr)
            _print_presets()
            return 2
        if not endpoint:
            endpoint = preset["endpoint"]

    headers = _parse_headers(args.header)

    cot_path = _resolve_cot_path(args.session_id, args.cot_path)
    print(f"[export_otlp] reading cot.json: {cot_path}")
    with open(cot_path, "r", encoding="utf-8") as f:
        cot_data = json.load(f)

    if not cot_data.get("otel_view"):
        print(
            "[export_otlp] WARN：这份 cot.json 没有 otel_view 字段（v0.11+ 才有）；"
            "导出会缺 gen_ai.* attribute。建议先用最新 cot_extractor 重新生成。",
            file=sys.stderr,
        )

    print(f"[export_otlp] session_id  : {cot_data.get('session_id')}")
    print(f"[export_otlp] turns       : {len(cot_data.get('turns') or [])}")
    print(f"[export_otlp] endpoint    : {endpoint or '(env / default)'}")
    print(f"[export_otlp] dry_run     : {args.dry_run}")

    try:
        result = export_session_to_otlp(
            cot_data,
            endpoint=endpoint,
            headers=headers or None,
            service_name=args.service_name,
            service_version=args.service_version,
            deployment_environment=args.environment,
            dry_run=args.dry_run,
            timeout=args.timeout,
        )
    except Exception as e:
        print(f"[export_otlp] FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print()
    print("=" * 60)
    print(f"  ok          : {result['ok']}")
    print(f"  trace_id    : {result['trace_id']}")
    print(f"  spans       : {result['span_count']}")
    print(f"  endpoint    : {result.get('endpoint') or '(dry-run)'}")
    print(f"  service     : {result['service_name']}")
    print(f"  dry_run     : {result['dry_run']}")
    if args.dry_run and result.get("sample_spans"):
        print(f"  sample      : {len(result['sample_spans'])} / {result.get('sample_total')} spans")
        print()
        print(json.dumps(result["sample_spans"], indent=2, ensure_ascii=False, default=str))
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
