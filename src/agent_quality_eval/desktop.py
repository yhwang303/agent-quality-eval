"""Native desktop shell for the packaged app.

This turns the local dashboard from "a website that suddenly opens in your
browser" into a **real desktop application window** with an immediate startup
splash, a proper title bar, and a system-tray presence.

Design goals / guarantees:

* **No product behavior changes.** Same backend, same bootstrap, same APIs,
  same on-disk data. This module only changes *presentation*: instead of
  ``webbrowser.open`` we host the exact same ``http://127.0.0.1:<port>/`` UI
  inside a native WebView2 window (via ``pywebview``).
* **Instant feedback.** A splash screen is shown the moment the window is
  created, so double-clicking the app no longer looks "dead" for a few
  seconds before a browser tab appears.
* **Background capture preserved.** Closing the window does NOT stop the
  backend — it minimizes to the tray, exactly like the previous tray-based
  model kept observing IDE traces in the background. Only *Quit* (from the
  tray or the app menu) stops the backend.
* **Graceful fallback.** If the native window cannot be created for any
  reason (missing WebView2, import failure, etc.), :func:`run_app` returns
  ``False`` and the caller falls back to the historical browser + tray path,
  so the app always comes up.
"""

from __future__ import annotations

import base64
import os
import sys
import threading
import webbrowser
from pathlib import Path

APP_TITLE = "Agent Observation"


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------


def _logo_path() -> Path | None:
    """Locate the bundled logo.png (used for splash + tray)."""
    candidates: list[Path] = []
    assets_root = os.environ.get("AGENT_COT_ASSETS_ROOT")
    if assets_root:
        candidates.append(Path(assets_root) / "frontend-dist" / "logo.png")
    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        candidates.append(Path(meipass) / "agent_cot" / "assets" / "frontend-dist" / "logo.png")
    here = Path(__file__).resolve()
    candidates.append(
        here.parents[1] / "agent_cot" / "assets" / "frontend-dist" / "logo.png"
    )
    candidates.append(here.parents[2] / "assets" / "logo.png")
    for cand in candidates:
        try:
            if cand.is_file():
                return cand
        except OSError:
            continue
    return None


def _ico_path() -> Path | None:
    """Locate the bundled .ico file for the native window title bar / taskbar."""
    candidates: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        candidates.append(
            Path(meipass) / "agent_quality_eval" / "assets" / "agent-quality-eval.ico"
        )
        candidates.append(
            Path(meipass) / "agent_cot" / "assets" / "frontend-dist" / "agent-quality-eval.ico"
        )
    here = Path(__file__).resolve()
    candidates.append(here.parents[0] / "assets" / "agent-quality-eval.ico")
    candidates.append(here.parents[2] / "assets" / "agent-quality-eval.ico")
    for cand in candidates:
        try:
            if cand.is_file():
                return cand
        except OSError:
            continue
    return None


def _apply_window_icon(hwnd: int, ico: Path) -> None:
    """Set the native window's small+big icons via SendMessage(WM_SETICON).

    Called after the window is shown so the title bar and taskbar reflect the
    branded ico immediately, even when the exe hasn't been re-signed with the
    new icon (Windows sometimes caches shell icons). No-op on failure.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        LR_LOADFROMFILE = 0x00000010
        LR_DEFAULTSIZE = 0x00000040
        IMAGE_ICON = 1
        WM_SETICON = 0x0080
        ICON_SMALL = 0
        ICON_BIG = 1

        user32 = ctypes.windll.user32
        user32.LoadImageW.restype = wintypes.HANDLE
        user32.LoadImageW.argtypes = [
            wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT,
            ctypes.c_int, ctypes.c_int, wintypes.UINT,
        ]
        ico_str = str(ico)
        big = user32.LoadImageW(None, ico_str, IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE)
        small = user32.LoadImageW(None, ico_str, IMAGE_ICON, 16, 16, LR_LOADFROMFILE)
        if big:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, big)
        if small:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, small)
    except Exception:
        pass


def _logo_data_uri() -> str:
    path = _logo_path()
    if path is None:
        return ""
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _splash_html(message: str = "正在启动本地服务…") -> str:
    logo = _logo_data_uri()
    logo_markup = (
        f'<img class="logo" src="{logo}" alt="logo" />' if logo else '<div class="logo fallback"></div>'
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>{APP_TITLE}</title>
<style>
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; height: 100%; }}
  body {{
    font-family: -apple-system, "Segoe UI", "Microsoft YaHei", Roboto, sans-serif;
    background: radial-gradient(1200px 600px at 20% 0%, #1e293b 0%, #0b1120 55%, #070b16 100%);
    color: #e2e8f0;
    display: flex; align-items: center; justify-content: center;
    height: 100vh; user-select: none; -webkit-user-select: none;
  }}
  .wrap {{ text-align: center; transform: translateY(-6px); }}
  .logo {{
    width: 96px; height: 96px; border-radius: 22px;
    box-shadow: 0 12px 40px rgba(6, 182, 212, 0.28), 0 2px 8px rgba(0,0,0,0.4);
    animation: float 2.6s ease-in-out infinite;
  }}
  .logo.fallback {{ background: linear-gradient(135deg, #4f46e5, #06b6d4); }}
  .title {{ margin-top: 22px; font-size: 22px; font-weight: 700; letter-spacing: 0.2px; }}
  .subtitle {{ margin-top: 6px; font-size: 13px; color: #94a3b8; }}
  .msg {{ margin-top: 4px; font-size: 12.5px; color: #64748b; }}
  .ring {{
    margin: 26px auto 0; width: 34px; height: 34px; border-radius: 50%;
    border: 3px solid rgba(148, 163, 184, 0.25);
    border-top-color: #22d3ee; animation: spin 0.9s linear infinite;
  }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  @keyframes float {{ 0%,100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-8px); }} }}
</style>
</head>
<body>
  <div class="wrap">
    {logo_markup}
    <div class="title">{APP_TITLE}</div>
    <div class="subtitle">本地 Agent 观测与评测平台</div>
    <div class="ring"></div>
    <div class="msg">{message}</div>
  </div>
</body>
</html>"""


def _error_html(detail: str) -> str:
    safe = (detail or "").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{APP_TITLE}</title>
<style>
  html,body{{margin:0;height:100%;font-family:"Segoe UI","Microsoft YaHei",sans-serif;
    background:#0b1120;color:#e2e8f0;display:flex;align-items:center;justify-content:center;}}
  .box{{max-width:520px;text-align:center;padding:24px;}}
  h1{{font-size:18px;margin:0 0 10px;}} p{{color:#94a3b8;font-size:13px;line-height:1.6;}}
  code{{color:#f87171;font-size:12px;word-break:break-all;}}
</style></head>
<body><div class="box">
  <h1>启动失败</h1>
  <p>本地服务未能启动。请重试，或从系统托盘退出后重新打开。</p>
  <p><code>{safe}</code></p>
</div></div></body></html>"""


# ---------------------------------------------------------------------------
# Backend bootstrap (identical steps to the historical launcher path)
# ---------------------------------------------------------------------------


def _bootstrap_and_start() -> int:
    """Bootstrap workspace + observation runtime, start backend, return port.

    Mirrors the exact sequence the launcher used before, so nothing about the
    product's startup behavior changes — only what we do with the port
    afterwards (open a native window instead of a browser tab).
    """
    from .cli import bootstrap_observation_runtime, bootstrap_workspace
    from agent_cot.commands import start as start_cmd
    from agent_cot.commands.stop import stop_backend

    try:
        stop_backend(force=False)
    except Exception:
        pass
    bootstrap_workspace(overwrite_config=False)
    bootstrap_observation_runtime()
    result = start_cmd.start_backend(open_browser=False)
    return result.port


def _stop_backend_quiet() -> None:
    try:
        from agent_cot.commands.stop import stop_backend

        stop_backend(force=False)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tray (runs in a background thread alongside the native window)
# ---------------------------------------------------------------------------


def _make_tray_image():
    from PIL import Image

    logo = _logo_path()
    if logo is not None:
        try:
            return Image.open(logo).convert("RGBA")
        except Exception:
            pass
    from PIL import ImageDraw

    img = Image.new("RGBA", (64, 64), (11, 17, 32, 255))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((6, 6, 58, 58), radius=14, fill=(37, 99, 235, 255))
    draw.text((18, 22), "AE", fill=(255, 255, 255, 255))
    return img


def _start_tray(window, port: int, on_quit) -> "object | None":
    try:
        import pystray
    except Exception:
        return None

    def show(_icon=None, _item=None) -> None:
        try:
            window.show()
        except Exception:
            webbrowser.open(f"http://127.0.0.1:{port}/")

    def open_in_browser(_icon=None, _item=None) -> None:
        webbrowser.open(f"http://127.0.0.1:{port}/")

    def do_quit(icon, _item=None) -> None:
        try:
            icon.stop()
        except Exception:
            pass
        on_quit()

    try:
        image = _make_tray_image()
    except Exception:
        return None

    icon = pystray.Icon(
        "agent-quality-eval",
        image,
        APP_TITLE,
        pystray.Menu(
            pystray.MenuItem("打开 / Open", show, default=True),
            pystray.MenuItem("在浏览器中打开", open_in_browser),
            pystray.MenuItem("退出 / Quit", do_quit),
        ),
    )
    thread = threading.Thread(target=icon.run, name="aqe-tray", daemon=True)
    thread.start()
    return icon


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_app() -> bool:
    """Run the app as a native desktop window. Returns False if unavailable.

    On success this blocks until the user chooses *Quit* (window close only
    hides to tray), then stops the backend and returns True. On any failure
    to create the native window it returns False so the caller can fall back
    to the browser + tray path.
    """
    try:
        import webview  # noqa: F401  (pywebview)
    except Exception:
        return False

    import webview

    state = {"quitting": False, "icon": None, "port": None}

    try:
        window = webview.create_window(
            APP_TITLE,
            html=_splash_html(),
            width=1360,
            height=880,
            min_size=(1024, 680),
            background_color="#0b1120",
            text_select=True,
        )
    except Exception:
        return False

    def _on_closing():
        # Close = minimize to tray (keep observing in background), never quit.
        if state["quitting"]:
            return True
        try:
            window.hide()
        except Exception:
            return True
        return False

    def _quit_now() -> None:
        state["quitting"] = True
        try:
            window.destroy()
        except Exception:
            pass

    try:
        window.events.closing += _on_closing
    except Exception:
        pass

    def _worker() -> None:
        try:
            port = _bootstrap_and_start()
            state["port"] = port
            try:
                from .frozen_entry import _refresh_frozen_runtime_state

                _refresh_frozen_runtime_state()
            except Exception:
                pass
            state["icon"] = _start_tray(window, port, _quit_now)
            window.load_url(f"http://127.0.0.1:{port}/")
            # Best-effort: apply the branded window icon after the window is up.
            try:
                ico = _ico_path()
                if ico is not None and sys.platform == "win32":
                    import ctypes
                    # Look up hwnd by unique title (pywebview doesn't expose it).
                    hwnd = ctypes.windll.user32.FindWindowW(None, APP_TITLE)
                    if hwnd:
                        _apply_window_icon(hwnd, ico)
            except Exception:
                pass
        except Exception as exc:  # pragma: no cover - defensive
            try:
                window.load_html(_error_html(str(exc)))
            except Exception:
                pass

    try:
        webview.start(_worker)
    except Exception:
        # GUI loop could not start — let caller fall back.
        return False

    # Reached only after the window is destroyed (Quit).
    icon = state.get("icon")
    if icon is not None:
        try:
            icon.stop()
        except Exception:
            pass
    _stop_backend_quiet()
    return True
