from __future__ import annotations

from agent_quality_eval import desktop


class _FakeDwm:
    def __init__(self, *, fail_dark: bool = False) -> None:
        self.fail_dark = fail_dark
        self.attributes: list[int] = []

    def DwmSetWindowAttribute(self, hwnd, attribute, value, size):  # noqa: N802
        self.attributes.append(int(attribute))
        if self.fail_dark and int(attribute) == desktop._DWMWA_USE_IMMERSIVE_DARK_MODE:
            return -1
        return 0


def test_apply_window_dwm_theme_sets_modern_dark_chrome(monkeypatch) -> None:
    monkeypatch.setattr(desktop.sys, "platform", "win32")
    dwm = _FakeDwm()

    desktop._apply_window_dwm_theme(123, _dwmapi=dwm)

    assert desktop._DWMWA_USE_IMMERSIVE_DARK_MODE in dwm.attributes
    assert desktop._DWMWA_BORDER_COLOR in dwm.attributes
    assert desktop._DWMWA_TEXT_COLOR in dwm.attributes
    assert desktop._DWMWA_SYSTEMBACKDROP_TYPE in dwm.attributes


def test_apply_window_dwm_theme_falls_back_to_legacy_dark_mode(monkeypatch) -> None:
    monkeypatch.setattr(desktop.sys, "platform", "win32")
    dwm = _FakeDwm(fail_dark=True)

    desktop._apply_window_dwm_theme(123, _dwmapi=dwm)

    assert desktop._DWMWA_USE_IMMERSIVE_DARK_MODE_LEGACY in dwm.attributes


def test_apply_window_dwm_theme_is_noop_off_windows(monkeypatch) -> None:
    monkeypatch.setattr(desktop.sys, "platform", "linux")
    dwm = _FakeDwm()

    desktop._apply_window_dwm_theme(123, _dwmapi=dwm)

    assert dwm.attributes == []
