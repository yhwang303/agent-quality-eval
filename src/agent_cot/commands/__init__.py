"""High-level command implementations wired up by ``cli.py``.

Each module here contains the orchestration for one subcommand
(``init``, ``start``, ``doctor``, …). Keep them focused on *what to do
in what order*, not on low-level primitives — those live in
``installer/`` and ``runtime/``.
"""

from __future__ import annotations

from . import init as _init  # noqa: F401  (eager import surfaces import-time errors early)

__all__: list[str] = []
