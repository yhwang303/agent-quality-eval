"""Pick a free TCP port for the local dashboard backend.

Why bother:

* Cursor users frequently already have something on 8765 / 8000 / 5000.
* Hard-coding a single port forces collisions; randomising every run
  breaks bookmarks. The middle ground we want is "prefer the configured
  port, fall back to the next free one in a tight band, persist the
  choice".

Persistence happens in :mod:`agent_cot.installer.config`; this module
only handles the *picking* part.
"""

from __future__ import annotations

import socket
from contextlib import closing

from ..agents.base import CursorCotError

# Default scan band when ``prefer`` is occupied. 8800-8899 keeps us
# clearly above the well-known port range while staying memorable.
DEFAULT_FALLBACK_RANGE: tuple[int, int] = (8800, 8900)


class PortNotFoundError(CursorCotError):
    """Raised when no free port could be found in the requested band."""

    def __init__(self, prefer: int, fallback_range: tuple[int, int]):
        self.prefer = prefer
        self.fallback_range = fallback_range
        super().__init__(
            f"No free TCP port found. Tried {prefer} and "
            f"[{fallback_range[0]}, {fallback_range[1]}). "
            "Pass --port-backend to specify one manually."
        )


def is_port_free(port: int, *, host: str = "127.0.0.1") -> bool:
    """Return ``True`` iff a TCP server can bind to ``host:port``.

    We rely on ``SO_REUSEADDR`` not being set so the test mirrors what
    uvicorn / vite would actually see at startup.
    """
    if port < 1 or port > 65535:
        return False
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
        return True


def pick_port(
    prefer: int = 8765,
    fallback_range: tuple[int, int] = DEFAULT_FALLBACK_RANGE,
    *,
    host: str = "127.0.0.1",
) -> int:
    """Return a free TCP port, preferring ``prefer``.

    Algorithm:

    1. If ``prefer`` is free, return it.
    2. Else scan ``[fallback_range[0], fallback_range[1])`` ascending,
       return the first free port.
    3. Else raise :class:`PortNotFoundError`.

    The function never holds a socket open across its return — callers
    are responsible for handling the (small) TOCTOU window between
    ``pick_port`` returning and the actual server binding.
    """
    if is_port_free(prefer, host=host):
        return prefer
    lo, hi = fallback_range
    if lo > hi:
        lo, hi = hi, lo
    for port in range(lo, hi):
        if is_port_free(port, host=host):
            return port
    raise PortNotFoundError(prefer=prefer, fallback_range=fallback_range)


__all__ = [
    "DEFAULT_FALLBACK_RANGE",
    "PortNotFoundError",
    "is_port_free",
    "pick_port",
]
