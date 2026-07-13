# Gold Standard Eval Test Suite

This folder contains a small but non-trivial Gold Standard test set for checking
trace evaluation with an uploaded reference answer.

Use it like this:

1. Pick one case from `prompts.md`.
2. Copy that case's prompt into any agent and let it answer.
3. In Agent Quality Eval, select that trace turn.
4. Click `Gold` next to `Eval`.
5. Upload the matching single-case file from `individual/`.
6. Save, return to the main view, then click the normal `Eval` button.

Important product detail:

- `gold-standard-suite.yaml` contains all cases for review.
- The current trace-level `Gold` upload binds one reference answer to one trace.
- For trace-level validation, upload files from `individual/`, not the whole suite.

Suggested cases:

- `case-01_emergency-routing.yaml`: logistics, constraints, numerical allocation.
- `case-02_subscription-sql.yaml`: SQL, business rules, edge cases.
- `case-03_privacy-incident.yaml`: incident classification, privacy-safe response.
- `case-04_contract-risk.yaml`: legal-style risk memo without inventing law.
- `case-05_experiment-analysis.yaml`: experiment interpretation and decision logic.
