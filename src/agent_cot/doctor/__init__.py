"""doctor/ — read-only self-diagnosis for the local install.

Doctor never touches the file system; it only **reads** state and
emits actionable hints. Anything that should be auto-repaired belongs
in ``commands.init`` / ``commands.upgrade`` (P5).

Public surface:

* :class:`Check` / :class:`CheckStatus` — the unit of report.
* :func:`run_all` — runs every check group in a stable order and
  returns a flat list. CLI rendering lives in :mod:`commands.doctor`.
"""

from __future__ import annotations

from .checks import Check, CheckStatus
from .runner import DoctorReport, run_all

__all__ = ["Check", "CheckStatus", "DoctorReport", "run_all"]
