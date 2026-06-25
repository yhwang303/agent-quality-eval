# Eval Config Template

Use `sample_eval.yaml` as the minimal local pipeline:

```powershell
agent-eval eval run sample_eval.yaml
```

Provider types supported in v0.1:

- `mock`
- `openai` / `openai-compatible`
- `knot`
- `python:path/to/provider.py:call_api`

Assertions include deterministic checks, LLM rubric checks, token/cost budgets,
PII checks, and trace/tool checks.
