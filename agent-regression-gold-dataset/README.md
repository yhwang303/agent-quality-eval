# AQE Regression Gold Dataset

This suite contains six one-to-one Trace/Gold cases selected from real local
observations: four simple turns and two multi-step tool turns across Cursor,
Claude, Codex, and CodeBuddy.

Each case contains:

- `trace.json`: minimized observed trace.
- `raw/`: simulated incomplete Gold written naturally by a new user.
- `canonical/gold.canonical.json`: deterministic normalized Gold.
- `normalization-report.json`: source-to-canonical field mapping and warnings.
- `negative-control/trace.json`: explicitly synthetic degraded candidate used
  only to prove that the regression gate detects a loss.

The six raw uploads are ordered from missingness level 6 (only a final answer)
to level 1 (question, answer, and a rough natural-language acceptance note).
Even level 1 has no internal IDs, keyword list, assertions, field mapping, or
structured process requirements. The manifest binds every trace to one Gold
and records its missingness level.

Run `scripts/build_regression_gold_dataset.py` with `PYTHONPATH=src` to rebuild
the suite from the local source traces.
