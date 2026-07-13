# Tool-Use Gold Standard Cases

These cases are designed to force real file inspection, calculation, and multi-step reasoning.
They are better suited for validating Gold Standard trace evaluation than simple writing prompts.

How to test one case:

1. Copy the matching prompt from `prompts.md`.
2. Run it in an agent from the repository root `D:\agent-quality-eval`.
3. The agent should inspect files under `examples/gold-standard-eval-suite/tool-use-cases/data/`.
4. In Agent Quality Eval, select that trace turn.
5. Click `Gold` next to `Eval`.
6. Upload the matching file from `individual/`.
7. Save, then click the normal `Eval`.

Cases:

- Case 01: marketplace reconciliation across orders, payments, refunds, and chargebacks.
- Case 02: API incident root-cause analysis using deploy timeline, metrics, logs, and regional failure counts.
- Case 03: access-policy audit using users, resource policy, and access logs.

