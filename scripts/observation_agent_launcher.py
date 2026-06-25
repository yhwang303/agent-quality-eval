from __future__ import annotations

import os
import tempfile
import traceback
from pathlib import Path


def _log_failure() -> None:
    try:
        log_path = Path(tempfile.gettempdir()) / "agent-quality-eval-launcher.log"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write("\n=== launcher failure ===\n")
            fh.write(traceback.format_exc())
            fh.write("\n")
    except Exception:
        pass


try:
    from agent_quality_eval.frozen_entry import main
except Exception:
    _log_failure()
    if os.environ.get("AGENT_QUALITY_EVAL_RAISE_LAUNCH_ERRORS"):
        raise

    def main() -> None:
        return None


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _log_failure()
        if os.environ.get("AGENT_QUALITY_EVAL_RAISE_LAUNCH_ERRORS"):
            raise
