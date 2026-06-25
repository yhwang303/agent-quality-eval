"""Unit tests for installer/port_picker.py."""

from __future__ import annotations

import socket
from contextlib import closing

import pytest

from agent_cot.installer.port_picker import (
    PortNotFoundError,
    is_port_free,
    pick_port,
)


def _occupy(port: int) -> socket.socket:
    """Bind a real socket to ``port`` so ``is_port_free`` sees it taken."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", port))
    sock.listen(1)
    return sock


def test_invalid_port_returns_false() -> None:
    assert is_port_free(0) is False
    assert is_port_free(70000) is False
    assert is_port_free(-1) is False


def test_pick_port_returns_prefer_when_free() -> None:
    # Port 0 means "let the OS assign"; we use it to find a guaranteed-free
    # high port, then close it before passing to pick_port.
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert pick_port(prefer=port) == port


def test_pick_port_falls_back_when_prefer_busy() -> None:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        busy_port = s.getsockname()[1]
        chosen = pick_port(prefer=busy_port, fallback_range=(50000, 50100))
    assert chosen != busy_port
    assert 50000 <= chosen < 50100


def test_pick_port_raises_when_band_exhausted() -> None:
    sockets: list[socket.socket] = []
    try:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            busy_prefer = s.getsockname()[1]

            # Monkey-patch is_port_free so we don't have to actually
            # exhaust 100 OS ports; we just say "everything in the band
            # is taken" deterministically.
            from agent_cot.installer import port_picker as pp

            saved = pp.is_port_free
            pp.is_port_free = lambda port, host="127.0.0.1": False  # type: ignore[assignment]
            try:
                with pytest.raises(PortNotFoundError) as exc:
                    pick_port(prefer=busy_prefer, fallback_range=(50000, 50050))
            finally:
                pp.is_port_free = saved  # type: ignore[assignment]

        assert "No free TCP port" in str(exc.value)
        assert "[50000, 50050)" in str(exc.value)
    finally:
        for s in sockets:
            s.close()
