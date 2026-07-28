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
   dist/agent-quality-eval-1.0.14.exe
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

**Export (for closing the harness loop)**

An agent that is supposed to improve your harness needs to read what the last
run actually did. Two things are exportable straight from the dashboard:

- **Per-turn trace** — the 导出 button on every interaction card, next to Eval.
  Gives you that one turn's full event stream as JSONL: thinking, tool calls and
  results, plan snapshots, permission prompts, subagent activity. One event per
  line, in execution order, so a script can read the skeleton first and expand
  only the events it cares about.
- **Per-turn eval result** — the 导出 button after the 面板 title in the eval
  panel. Gives you the whole report as JSON: every assertion result, the
  dimension panel, the safety gate, and the hook-stage Agent Critic provenance.

Both are scoped to a single turn of a single session and never mix in data from
anywhere else. Whole-session traces are available too, via HTTP or CLI:

```text
GET /api/sessions/{sid}/turns/{n}/export/trace.jsonl     # one turn
GET /api/sessions/{sid}/export/trace.{jsonl,json,md}     # whole session
GET /api/evals/session/{sid}/turn/{n}/export.json        # one turn's eval
```

```powershell
# offline export for harness scripts
.\dist\agent-quality-eval-1.0.14.exe observe export-trace --session-id <sid>
```

Tool results are capped at 2000 characters at capture time; events that were
truncated carry the original length so you can tell a short result from a
clipped one.

**Duplicate thinking in older Cursor sessions**

Cursor pushes the same block of reasoning through the `afterAgentThought` hook
twice — once with a bare `generation_id`, once with a suffixed one — and the
transcript's text block produces yet another copy. Sessions captured before
1.0.12 therefore show every thought two or three times. New captures are clean;
to clean up what is already on disk:

```powershell
# dry run first — it rewrites captured data
.\dist\agent-quality-eval-1.0.14.exe observe dedupe-thinking
.\dist\agent-quality-eval-1.0.14.exe observe dedupe-thinking --apply

# or just one session (an id prefix is enough)
.\dist\agent-quality-eval-1.0.14.exe observe dedupe-thinking 3206a7e7 --apply
```

Redacted thoughts (`[REDACTED]`) all look identical but are genuinely different
steps, so they are deliberately left alone.

What this does *not* fix: in some Cursor sessions the transcript channel and the
hook channel each captured a partial view of the same turn, and the extractor
concatenated them instead of merging them by time. The turn then reads as two
halves — transcript events first, then the hook's recording of the *same*
minutes appended after it. Removing the duplicated thoughts makes the turn
shorter but leaves that ordering intact. This is a known open issue; the
evidence, the affected sessions and the shape of a fix are written up in
[docs/known-issue-channel-concatenation.md](docs/known-issue-channel-concatenation.md).

**Turn token counts**

A turn's input/output tokens come from, in order: the per-turn hook truth
(`turn.usage`), the sum of the turn's LLM steps, and finally the enricher's
turn-level estimate. Steps marked `non_llm_step` — tool executions and user
input, whose "tokens" are character estimates of arguments and results — never
count, because the model did not produce them. The trace tree, the detail panel
and the eval report all use this one chain, so the same turn reports the same
number everywhere.

Eval reports store the numbers computed when they were written. Reports created
before 1.0.12 keep the old values until that turn is evaluated again.

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
.\dist\agent-quality-eval-1.0.14.exe doctor

# initialize workspace manually (usually not needed — first run does it)
.\dist\agent-quality-eval-1.0.14.exe init

# run an eval pipeline
.\dist\agent-quality-eval-1.0.14.exe eval run .\examples\gold-standard-eval-suite\gold-standard-suite.yaml

# compare two experiments
.\dist\agent-quality-eval-1.0.14.exe eval compare --baseline <experiment_id> --candidate <experiment_id>

# regression gate (non-zero exit on regression)
.\dist\agent-quality-eval-1.0.14.exe eval regression --baseline <experiment_id> --candidate <experiment_id>

# promote a baseline
.\dist\agent-quality-eval-1.0.14.exe eval promote-baseline <experiment_id>

# observation passthrough (starts the observation runtime)
.\dist\agent-quality-eval-1.0.14.exe observe start
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
npm run build   # emits into frontend/dist

# Copy the build into the package — the spec bundles src/agent_cot/assets,
# so a build that stays in frontend/dist will NOT reach the exe.
cd D:\agent-quality-eval
Remove-Item -Recurse -Force src\agent_cot\assets\frontend-dist\assets
Copy-Item -Recurse -Force frontend\dist\* src\agent_cot\assets\frontend-dist

# Build the Windows exe
pyinstaller agent-quality-eval.spec
# → dist\agent-quality-eval-<version>.exe
```

The current on-disk version comes from `src/agent_quality_eval/__init__.py`
and `pyproject.toml` (both `1.0.14`).

---

## License

MIT — see `pyproject.toml`. This is a personal / internal tool; use at your
own risk on your own data.
