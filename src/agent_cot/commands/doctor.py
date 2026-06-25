"""``agent-cot doctor`` — render the diagnostic report.

The actual checks live in :mod:`agent_cot.doctor`. This module is
just the CLI-facing presentation layer: text vs. ``--json``, color
choices, and exit-code mapping.
"""

from __future__ import annotations

import json

import click

from ..doctor import CheckStatus, run_all
from ..doctor.runner import DoctorReport

_STATUS_GLYPH = {
    CheckStatus.OK: ("✓", "green"),
    CheckStatus.WARN: ("⚠", "yellow"),
    CheckStatus.FAIL: ("✗", "red"),
    CheckStatus.SKIP: ("·", "cyan"),
}


def _exit_code_for(status: CheckStatus) -> int:
    """FAIL → 2 (CI failure), WARN → 1 (advisory), OK → 0.

    SKIP is informational and always 0; it never affects the exit code.
    """
    if status is CheckStatus.FAIL:
        return 2
    if status is CheckStatus.WARN:
        return 1
    return 0


def render_human(report: DoctorReport, *, verbose: bool) -> str:
    """Emit a friendly, color-coded summary."""
    lines: list[str] = []
    lines.append(click.style("agent-cot doctor", bold=True))
    lines.append("")

    for c in report.checks:
        glyph, color = _STATUS_GLYPH[c.status]
        if not verbose and c.status is CheckStatus.OK:
            continue
        head = f"  {click.style(glyph, fg=color)} {click.style(c.name, bold=True):<28} {c.message}"
        lines.append(head)
        if c.hint and c.status is not CheckStatus.OK:
            lines.append(click.style(f"      ↳ {c.hint}", fg="white", dim=True))

    if not verbose:
        ok_count = report.counts.get(CheckStatus.OK.value, 0)
        if ok_count:
            lines.append("")
            lines.append(
                click.style(
                    f"  ({ok_count} healthy checks hidden — use --verbose to see them.)",
                    dim=True,
                )
            )

    lines.append("")
    lines.append(_render_summary(report))
    return "\n".join(lines)


def _render_summary(report: DoctorReport) -> str:
    counts = report.counts
    overall = report.overall_status
    glyph, color = _STATUS_GLYPH[overall]
    parts = [
        click.style(glyph + " overall", fg=color, bold=True),
        click.style(
            f"ok={counts[CheckStatus.OK.value]}",
            fg="green",
        ),
        click.style(
            f"warn={counts[CheckStatus.WARN.value]}",
            fg="yellow" if counts[CheckStatus.WARN.value] else None,
        ),
        click.style(
            f"fail={counts[CheckStatus.FAIL.value]}",
            fg="red" if counts[CheckStatus.FAIL.value] else None,
        ),
        click.style(
            f"skip={counts[CheckStatus.SKIP.value]}",
            fg="cyan" if counts[CheckStatus.SKIP.value] else None,
            dim=True,
        ),
    ]
    return "  " + "  ".join(parts)


def run_doctor(*, verbose: bool, as_json: bool, deep: bool = False) -> int:
    """End-to-end entry: run checks, print, return exit code.

    ``deep=True`` adds the v0.19.0 self-heal diagnostics: runtime.json
    freshness, on-disk Cursor + CodeBuddy hook script staleness, recent
    cot.json richness. These are slower (extra disk reads) but answer
    the colleague-debugging case directly.
    """
    report = run_all(deep=deep)

    if as_json:
        click.echo(json.dumps(report.to_dict(), indent=2))
    else:
        click.echo(render_human(report, verbose=verbose))

    return _exit_code_for(report.overall_status)


__all__ = ["render_human", "run_doctor"]
