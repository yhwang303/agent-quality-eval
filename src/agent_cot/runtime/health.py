"""Backend health probing.

We use ``urllib`` from the stdlib instead of ``requests`` so the
runtime layer has zero dependencies beyond Python itself — important
for ``status`` to keep working even if the user's venv is half-broken.

The probed endpoint is intentionally generic (``/api/sessions``) so
this module doesn't need to be updated when the dashboard adds new
routes.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request

from ..agents.base import CursorCotError

DEFAULT_HEALTH_PATH = "/api/health"
DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_POLL_INTERVAL = 0.25


class HealthCheckError(CursorCotError):
    """Raised when ``wait_for_backend`` exceeds its deadline."""

    def __init__(self, port: int, timeout: float, last_error: str | None):
        self.port = port
        self.timeout = timeout
        self.last_error = last_error
        msg = (
            f"backend on 127.0.0.1:{port} did not become healthy within "
            f"{timeout:.1f}s"
        )
        if last_error:
            msg += f" (last error: {last_error})"
        super().__init__(msg)


def probe_once(
    port: int,
    *,
    path: str = DEFAULT_HEALTH_PATH,
    host: str = "127.0.0.1",
    timeout: float = 1.0,
) -> tuple[bool, str | None]:
    """Single non-blocking health probe.

    Returns ``(ok, last_error)``. ``ok`` is true on any 2xx/3xx — the
    dashboard's ``/api/sessions`` returns 200 even when no sessions
    exist yet, but we accept 3xx so the API is forwards-compatible
    with future redirects.
    """
    url = f"http://{host}:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return (200 <= resp.status < 400, None)
    except urllib.error.HTTPError as exc:
        # 4xx / 5xx still means the server is up enough to respond,
        # which is more than ``status`` cares about.
        return (False, f"HTTP {exc.code}")
    except urllib.error.URLError as exc:
        return (False, str(exc.reason))
    except (TimeoutError, ConnectionError) as exc:
        return (False, str(exc) or exc.__class__.__name__)
    except OSError as exc:
        return (False, str(exc) or exc.__class__.__name__)


def wait_for_backend(
    port: int,
    *,
    path: str = DEFAULT_HEALTH_PATH,
    host: str = "127.0.0.1",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    clock=time.monotonic,
    sleep=time.sleep,
) -> None:
    """Block until the backend answers a health probe, or raise.

    ``clock`` / ``sleep`` are injectable for tests so we don't burn 15
    seconds of wall clock per failed-startup test.
    """
    deadline = clock() + timeout
    last_error: str | None = None

    while True:
        ok, last_error = probe_once(port, path=path, host=host, timeout=1.0)
        if ok:
            return
        if clock() >= deadline:
            raise HealthCheckError(port=port, timeout=timeout, last_error=last_error)
        sleep(poll_interval)


__all__ = [
    "DEFAULT_HEALTH_PATH",
    "DEFAULT_POLL_INTERVAL",
    "DEFAULT_TIMEOUT_SECONDS",
    "HealthCheckError",
    "probe_once",
    "wait_for_backend",
]
