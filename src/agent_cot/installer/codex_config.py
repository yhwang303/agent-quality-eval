"""Codex user config helpers.

Codex enables native telemetry through ``~/.codex/config.toml`` rather than
through environment variables.  The helpers here append or replace only the
``[otel]`` family of tables and preserve the rest of the user's config text.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .platform_paths import backup_path


@dataclass(frozen=True)
class CodexOtelConfigResult:
    config_path: Path
    backup_path: Path | None
    status: str
    endpoint_logs: str
    endpoint_traces: str
    endpoint_metrics: str


_OTEL_TABLE_RE = re.compile(r"^\[(otel(?:\.|]).*)$", re.IGNORECASE)
_ANY_TABLE_RE = re.compile(r"^\[[^\]]+\]\s*$")


def codex_home() -> Path:
    import os

    raw = os.environ.get("CODEX_HOME")
    return Path(raw).expanduser() if raw else Path.home() / ".codex"


def codex_config_path(home: Path | None = None) -> Path:
    return (home or codex_home()) / "config.toml"


def render_codex_otel_block(*, backend_port: int) -> str:
    logs = f"http://127.0.0.1:{int(backend_port)}/v1/logs"
    traces = f"http://127.0.0.1:{int(backend_port)}/v1/traces"
    metrics = f"http://127.0.0.1:{int(backend_port)}/v1/metrics"
    return (
        "[otel]\n"
        'environment = "local"\n'
        'exporter = "otlp-http"\n'
        'trace_exporter = "otlp-http"\n'
        'metrics_exporter = "otlp-http"\n'
        "log_user_prompt = true\n"
        "\n"
        '[otel.exporter."otlp-http"]\n'
        f'endpoint = "{logs}"\n'
        'protocol = "json"\n'
        "\n"
        '[otel.trace_exporter."otlp-http"]\n'
        f'endpoint = "{traces}"\n'
        'protocol = "json"\n'
        "\n"
        '[otel.metrics_exporter."otlp-http"]\n'
        f'endpoint = "{metrics}"\n'
        'protocol = "json"\n'
    )


def _remove_otel_tables(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    skipping = False
    for line in lines:
        stripped = line.strip()
        if _OTEL_TABLE_RE.match(stripped):
            skipping = True
            continue
        if skipping and _ANY_TABLE_RE.match(stripped):
            skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out).rstrip()


def apply_codex_otel_config(
    *,
    backend_port: int,
    home: Path | None = None,
) -> CodexOtelConfigResult:
    home_dir = home or codex_home()
    path = codex_config_path(home_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    old = ""
    backup: Path | None = None
    if path.is_file():
        old = path.read_text(encoding="utf-8", errors="replace")
        backup = backup_path(path)
        shutil.copy2(path, backup)

    block = render_codex_otel_block(backend_port=backend_port)
    new = (_remove_otel_tables(old) + "\n\n" + block).lstrip()
    if not new.endswith("\n"):
        new += "\n"

    if old == new:
        return CodexOtelConfigResult(
            config_path=path,
            backup_path=backup,
            status="ok",
            endpoint_logs=f"http://127.0.0.1:{int(backend_port)}/v1/logs",
            endpoint_traces=f"http://127.0.0.1:{int(backend_port)}/v1/traces",
            endpoint_metrics=f"http://127.0.0.1:{int(backend_port)}/v1/metrics",
        )

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(new, encoding="utf-8")
    tmp.replace(path)
    return CodexOtelConfigResult(
        config_path=path,
        backup_path=backup,
        status="updated",
        endpoint_logs=f"http://127.0.0.1:{int(backend_port)}/v1/logs",
        endpoint_traces=f"http://127.0.0.1:{int(backend_port)}/v1/traces",
        endpoint_metrics=f"http://127.0.0.1:{int(backend_port)}/v1/metrics",
    )


__all__ = [
    "CodexOtelConfigResult",
    "apply_codex_otel_config",
    "codex_config_path",
    "codex_home",
    "render_codex_otel_block",
]
