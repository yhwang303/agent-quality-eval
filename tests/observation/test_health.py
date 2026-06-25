"""Unit tests for runtime/health.py."""

from __future__ import annotations

import http.server
import socket
import threading
from contextlib import closing, contextmanager

import pytest

from agent_cot.runtime.health import (
    HealthCheckError,
    probe_once,
    wait_for_backend,
)

# ---------------------------------------------------------------------------
# Tiny ad-hoc HTTP server fixture
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _OkHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"[]")

    def log_message(self, *_args, **_kw) -> None:  # silence stderr noise
        pass


@contextmanager
def _serving(handler=_OkHandler):
    port = _free_port()
    httpd = http.server.HTTPServer(("127.0.0.1", port), handler)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    try:
        yield port
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------------
# probe_once
# ---------------------------------------------------------------------------


def test_probe_returns_ok_on_live_server() -> None:
    with _serving() as port:
        ok, err = probe_once(port, path="/")
        assert ok is True
        assert err is None


def test_probe_returns_error_on_dead_port() -> None:
    # Port 1 is reserved on Linux/macOS and almost always closed.
    ok, err = probe_once(_free_port(), path="/", timeout=0.5)
    assert ok is False
    assert err is not None


def test_probe_returns_false_on_4xx() -> None:
    class NotFound(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(404)
            self.end_headers()

        def log_message(self, *_a, **_k) -> None:
            pass

    with _serving(NotFound) as port:
        ok, err = probe_once(port, path="/")
        assert ok is False
        assert err is not None


# ---------------------------------------------------------------------------
# wait_for_backend
# ---------------------------------------------------------------------------


def test_wait_for_backend_succeeds_against_live_server() -> None:
    with _serving() as port:
        # Should return ~immediately; high timeout just for safety.
        wait_for_backend(port, timeout=2.0, poll_interval=0.05)


def test_wait_for_backend_times_out_with_injected_clock() -> None:
    """Don't actually sleep 15 seconds — use injected clock."""
    fake_now = [0.0]

    def fake_clock() -> float:
        return fake_now[0]

    def fake_sleep(s: float) -> None:
        fake_now[0] += s

    port = _free_port()  # nothing listening
    with pytest.raises(HealthCheckError) as exc:
        wait_for_backend(
            port,
            timeout=0.5,
            poll_interval=0.1,
            clock=fake_clock,
            sleep=fake_sleep,
        )
    assert exc.value.port == port
    assert exc.value.timeout == 0.5
    assert exc.value.last_error is not None
