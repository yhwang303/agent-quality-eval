# Agent Quality Eval

Local-first Agent observability plus evaluation.

This project starts from the existing `observation-agent==1.0.15` runtime and adds:

- declarative eval pipelines
- dataset and experiment storage in SQLite
- A/B comparison
- baseline promotion and regression detection
- JSON and HTML reports
- CLI and FastAPI surfaces for local dashboard integration

## Quick Start

```powershell
cd D:\agent-quality-eval
python -m pip install -e ".[dev]"
agent-eval init
agent-eval eval run .\src\agent_quality_eval\templates\sample_eval.yaml
agent-eval eval compare --baseline <experiment_id> --candidate <experiment_id>
```

The packaged executable is built at:

```powershell
D:\agent-quality-eval\dist\agent-quality-eval-0.1.0.exe
```

Double-clicking the executable performs first-run setup automatically, then
starts the local dashboard and opens the Eval workbench at `/eval`. Passing
arguments uses the same CLI surface:

```powershell
.\dist\agent-quality-eval-0.1.0.exe doctor
.\dist\agent-quality-eval-0.1.0.exe eval run .\src\agent_quality_eval\templates\sample_eval.yaml
.\dist\agent-quality-eval-0.1.0.exe eval regression --baseline <id> --candidate <id>
```

Manual initialization is still available, but not required for the double-click
path:

```powershell
.\dist\agent-quality-eval-0.1.0.exe init
```

Observation compatibility remains available:

```powershell
agent-eval observe start
agent-cot start
```

## Data

By default, eval state is stored under:

```text
%USERPROFILE%\.agent-quality-eval\data\eval.db
```

Use `AGENT_QUALITY_EVAL_HOME` or `--db` to override it.

## Current Product Scope

Implemented: local dataset/experiment store, declarative eval pipeline,
trace-aware deterministic scorers, LLM rubric hooks, A/B comparison, regression
gate, JSON/HTML reports, `/api/evals/*`, browser-based `/eval` workbench,
observed-session eval with persisted reports, token/cost/latency/tool/error
metrics, observation-to-eval entry point, observation passthrough, and Windows
exe packaging.

Remaining product work: replace the injected observation Eval entry with a
source-built React navigation item once the original frontend source is copied,
DB-backed human review UI, encrypted secret storage, richer task-completion
scorers, release installer scripts, and statistical confidence intervals for
paired A/B runs.

See `docs\AB_TESTING_SKILL_FLOW.md` for the recommended no-skill vs skill A/B
testing workflow.
