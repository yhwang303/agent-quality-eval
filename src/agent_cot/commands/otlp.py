"""``agent-cot otlp`` — list-presets / send.

Both subcommands are thin wrappers over
:mod:`agent_cot.commands.otlp_bridge`, which does all the locating
and import work. Keeping this file presentation-only makes the unit
tests for the bridge much smaller.
"""

from __future__ import annotations

import json

import click

from .otlp_bridge import (
    OtlpBridgeError,
    find_preset,
    get_presets,
    import_exporter,
    parse_headers,
    resolve_cot_json,
)

# ---------------------------------------------------------------------------
# list-presets
# ---------------------------------------------------------------------------


def render_presets_human(presets: list[dict]) -> str:
    """Format the preset list for terminals."""
    lines = [click.style("OTLP backend presets:", bold=True), ""]
    label_w = max((len(p.get("label", "")) for p in presets), default=0)
    for p in presets:
        pid = p.get("id", "?")
        label = p.get("label", "")
        endpoint = p.get("endpoint", "")
        lines.append(
            "  "
            + click.style(f"[{pid}]", fg="cyan")
            + f"  {label:<{label_w}}  "
            + click.style(endpoint, dim=True)
        )
        if p.get("doc"):
            lines.append(click.style(f"      ↳ {p['doc']}", dim=True))
        if p.get("headers_hint"):
            for k, v in p["headers_hint"].items():
                lines.append(
                    click.style(f"      header: {k} = {v}", fg="yellow", dim=True)
                )
    return "\n".join(lines)


def run_list_presets(*, as_json: bool) -> int:
    try:
        presets = get_presets()
    except OtlpBridgeError as exc:
        click.secho(f"error: {exc}", fg="red", err=True)
        return 1

    if as_json:
        click.echo(json.dumps(presets, indent=2, ensure_ascii=False))
    else:
        click.echo(render_presets_human(presets))
    return 0


# ---------------------------------------------------------------------------
# send
# ---------------------------------------------------------------------------


def run_send(
    *,
    session_id: str | None,
    cot_path: str | None,
    preset: str | None,
    endpoint: str | None,
    headers: list[str],
    service_name: str,
    service_version: str | None,
    environment: str | None,
    timeout: float,
    dry_run: bool,
    as_json: bool,
) -> int:
    """Forward a single session to an OTLP/HTTP backend.

    The argument list mirrors ``cot-extractor/scripts/export_otlp.py``
    one-for-one so power users can copy invocations between the two.
    """

    try:
        cot = resolve_cot_json(session_id=session_id, cot_path=cot_path)
    except OtlpBridgeError as exc:
        click.secho(f"error: {exc}", fg="red", err=True)
        return 1

    real_endpoint = endpoint
    preset_obj: dict | None = None
    if preset:
        try:
            preset_obj = find_preset(preset)
        except OtlpBridgeError as exc:
            click.secho(f"error: {exc}", fg="red", err=True)
            return 1
        if not real_endpoint:
            real_endpoint = preset_obj.get("endpoint")

    try:
        parsed_headers = parse_headers(headers)
    except OtlpBridgeError as exc:
        click.secho(f"error: {exc}", fg="red", err=True)
        return 1

    try:
        mod = import_exporter()
    except OtlpBridgeError as exc:
        click.secho(f"error: {exc}", fg="red", err=True)
        return 1

    if not as_json:
        click.echo(click.style("agent-cot otlp send", bold=True))
        click.echo(f"  session_id : {cot.session_id}")
        click.echo(f"  cot.json   : {cot.path}")
        click.echo(f"  preset     : {preset or '-'}")
        click.echo(f"  endpoint   : {real_endpoint or '(env / default)'}")
        click.echo(f"  service    : {service_name}")
        click.echo(f"  dry_run    : {dry_run}")
        if preset_obj and preset_obj.get("headers_hint") and not parsed_headers:
            click.echo(
                click.style(
                    "  ! preset wants auth headers; see `otlp list-presets`",
                    fg="yellow",
                )
            )

    try:
        result = mod.export_session_to_otlp(
            cot.raw,
            endpoint=real_endpoint,
            headers=parsed_headers or None,
            service_name=service_name,
            service_version=service_version or "v0.13.0a0",
            deployment_environment=environment,
            dry_run=dry_run,
            timeout=timeout,
        )
    except RuntimeError as exc:
        # The exporter raises RuntimeError for "OTel SDK not installed";
        # surface that as a friendly error rather than a stack trace.
        click.secho(f"error: {exc}", fg="red", err=True)
        return 1
    except Exception as exc:
        click.secho(
            f"error: export_session_to_otlp raised {type(exc).__name__}: {exc}",
            fg="red",
            err=True,
        )
        return 1

    if as_json:
        click.echo(
            json.dumps(
                _result_to_jsonable(result),
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
        return 0 if result.get("ok") else 2

    click.echo("")
    click.echo("=" * 60)
    click.echo(f"  ok          : {result.get('ok')}")
    click.echo(f"  trace_id    : {result.get('trace_id')}")
    click.echo(f"  spans       : {result.get('span_count')}")
    click.echo(f"  endpoint    : {result.get('endpoint') or '(dry-run)'}")
    click.echo(f"  service     : {result.get('service_name')}")
    click.echo(f"  dry_run     : {result.get('dry_run')}")
    if dry_run and result.get("sample_spans"):
        click.echo(
            f"  sample      : {len(result['sample_spans'])} / "
            f"{result.get('sample_total')} spans"
        )
        click.echo("")
        click.echo(
            json.dumps(
                _result_to_jsonable(result["sample_spans"]),
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
    click.echo("=" * 60)
    return 0 if result.get("ok") else 2


def _result_to_jsonable(obj):
    """Best-effort coercion: drop bytes / objects we can't easily print."""
    if isinstance(obj, dict):
        return {k: _result_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_result_to_jsonable(x) for x in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return repr(obj)


__all__ = ["render_presets_human", "run_list_presets", "run_send"]
