"""Generate the app logo asset files from a single source PNG.

Inputs:
    assets/logo-source.png   (square, RGBA, transparent corners)

Outputs:
    frontend/public/logo.png                 256x256 web/header/favicon
    src/agent_cot/assets/frontend-dist/logo.png  (copy for the bundled SPA)
    assets/agent-quality-eval.ico            multi-size Windows icon
    assets/logo.png                          256x256 (used by tray/window)

Run with the build Python that has Pillow installed:
    python scripts/make_logo_assets.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "assets" / "logo-source.png"


def main() -> int:
    if not SOURCE.is_file():
        raise SystemExit(f"missing source logo: {SOURCE}")

    src = Image.open(SOURCE).convert("RGBA")
    # Square-crop defensively (generator returns 1:1 already).
    side = min(src.size)
    left = (src.width - side) // 2
    top = (src.height - side) // 2
    src = src.crop((left, top, left + side, top + side))

    # 1) Web logo (header + favicon).
    web = src.resize((256, 256), Image.LANCZOS)
    web_targets = [
        ROOT / "frontend" / "public" / "logo.png",
        ROOT / "src" / "agent_cot" / "assets" / "frontend-dist" / "logo.png",
        ROOT / "assets" / "logo.png",
    ]
    for target in web_targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        web.save(target, format="PNG")
        print(f"wrote {target}")

    # 2) Windows .ico (multi-resolution).
    ico_target = ROOT / "assets" / "agent-quality-eval.ico"
    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (24, 24), (16, 16)]
    src.save(ico_target, format="ICO", sizes=sizes)
    print(f"wrote {ico_target} ({len(sizes)} sizes)")

    # 3) Also drop a copy of the old favicon.svg out of the way is not needed;
    #    index.html now points at logo.png.
    # Remove stale bundled favicon.svg reference cleanup is handled elsewhere.
    _ = shutil  # keep import if future copy steps are added
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
