"""Drive the full check suite and bundle results into a report."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .checks import Check, CheckStatus, all_checks


@dataclass
class DoctorReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def overall_status(self) -> CheckStatus:
        """Worst status across all checks (FAIL > WARN > OK).

        SKIP is informational and never affects the overall verdict —
        e.g. an unimplemented Claude adapter shouldn't make doctor
        red on a Cursor-only machine.
        """
        if any(c.status is CheckStatus.FAIL for c in self.checks):
            return CheckStatus.FAIL
        if any(c.status is CheckStatus.WARN for c in self.checks):
            return CheckStatus.WARN
        return CheckStatus.OK

    @property
    def counts(self) -> dict[str, int]:
        out = {s.value: 0 for s in CheckStatus}
        for c in self.checks:
            out[c.status.value] += 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "counts": self.counts,
            "checks": [c.to_dict() for c in self.checks],
        }


def run_all(*, deep: bool = False) -> DoctorReport:
    """Execute every check in canonical order. Always succeeds (no
    individual check is allowed to raise, see ``checks._safe``).

    When ``deep`` is True, the v0.18.15+ deep checks (runtime.json,
    on-disk hook script staleness, recent cot.json richness) are also
    included.
    """
    return DoctorReport(checks=list(all_checks(deep=deep)))


# ``asdict`` re-export so consumers can dump without importing dataclasses.
__all__ = ["DoctorReport", "asdict", "run_all"]
