# Agent Quality Eval

Local-first agent **observability** + automated **evaluation** and **A/B testing**
for LLM / agent workflows — packaged as a single Windows executable that boots a
local dashboard and eval workbench in seconds.

No Python install, no Node install, no manual setup — download the exe,
double-click, and start observing and grading your agent sessions.

---

## Quick Start (recommended)

1. Grab the latest release exe:

   ```
   dist/agent-quality-eval-1.0.7.exe
   ```

2. Double-click it. On first run it will:
   - create the local workspace under `%USERPROFILE%\.agent-quality-eval\`
   - start a local FastAPI dashboard on a free port
   - open the Eval workbench in your default browser (path: `/eval`)

That's it. All observation traces, evaluation runs, baselines and A/B compare
results are stored locally in SQLite. Nothing is uploaded anywhere.

To close it, right-click the tray icon and choose *Exit*.

---

## What you get out of the box

**Observability**

- Auto-captured agent traces from Cursor, Claude Code, VS Code, CodeBuddy and
  Codex hooks (drop-in — the exe wires up the hooks for you).
- Per-session view: token / input / output / cache tokens, cost, latency, tool
  calls, LLM calls, error counts, plus the full call tree.
- Session list is filterable by project, agent, time range, model, tag.

**Evaluation**

- Declarative YAML eval pipelines with a rich catalog of scorers:
  content match, regex, JSON schema, length, no-error, response time, Python
  expressions, LLM rubric judge, PII, token/cost budgets, tool-called,
  max-tool-calls, step efficiency.
- LLM Judge (v3) with dedicated dimensions for correctness, completeness,
  plan quality and task completion — the rubric templates live in
  `docs/llm-judge-template.md` and `docs/llm-judge-template-v3.md`.
- Reference-answer evaluation: upload a gold reference, get an automatic
  quality score against it.
- Uploaded-trace evaluation: paste or drop a raw trace to grade a run that
  happened outside the local observer.
- Session eval reports are persisted, so every observed session keeps its
  scoring history for later review.

**A/B, regression and baselines**

- Paired A/B compare between two experiments with per-case win / tie / loss.
- Regression gate that fails CI when a candidate drops below the baseline
  according to your policy.
- Baseline promotion — pin a known-good experiment as the reference for
  future runs.

**Bundled goodies**

- Example eval suite in `examples/gold-standard-eval-suite/`
  (individual cases + tool-use cases with fixtures).
- Reference-answer testset in `examples/reference-answer-testset.yaml`.
- See `docs/AB_TESTING_SKILL_FLOW.md` for the recommended no-skill vs
  with-skill A/B workflow.

---

## Data location

By default all state (SQLite DB, reports, uploaded traces) lives under:

```text
%USERPROFILE%\.agent-quality-eval\
    data\eval.db
    reports\...
```

Override the location with either:

- environment variable `AGENT_QUALITY_EVAL_HOME`
- CLI flag `--db <path>` on any command

---

## Advanced: using the exe as a CLI

The same exe is also a full CLI. Passing arguments switches from "launch
dashboard" mode to command mode:

```powershell
# health check
.\dist\agent-quality-eval-1.0.7.exe doctor

# initialize workspace manually (usually not needed — first run does it)
.\dist\agent-quality-eval-1.0.7.exe init

# run an eval pipeline
.\dist\agent-quality-eval-1.0.7.exe eval run .\examples\gold-standard-eval-suite\gold-standard-suite.yaml

# compare two experiments
.\dist\agent-quality-eval-1.0.7.exe eval compare --baseline <experiment_id> --candidate <experiment_id>

# regression gate (non-zero exit on regression)
.\dist\agent-quality-eval-1.0.7.exe eval regression --baseline <experiment_id> --candidate <experiment_id>

# promote a baseline
.\dist\agent-quality-eval-1.0.7.exe eval promote-baseline <experiment_id>

# observation passthrough (starts the observation runtime)
.\dist\agent-quality-eval-1.0.7.exe observe start
```

`--help` on any subcommand lists all options.

---

## Advanced: run / build from source

Only needed if you want to modify the tool. The packaged exe is
self-contained and does not require any of the following.

```powershell
# Python side (evaluation engine + CLI + FastAPI backend)
cd D:\agent-quality-eval
python -m pip install -e ".[dev]"
python -m pytest -q

# Frontend (Vite + React dashboard / eval workbench)
cd frontend
npm install
npm run build   # emits into src/agent_cot/assets/frontend-dist

# Build the Windows exe
cd D:\agent-quality-eval
pyinstaller agent-quality-eval.spec
# → dist\agent-quality-eval-<version>.exe
```

The current on-disk version comes from `src/agent_quality_eval/__init__.py`
and `pyproject.toml` (both `1.0.7`).

---

## License

MIT — see `pyproject.toml`. This is a personal / internal tool; use at your
own risk on your own data.
