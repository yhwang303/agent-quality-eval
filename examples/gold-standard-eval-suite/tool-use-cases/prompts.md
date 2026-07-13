# Tool-Use Prompts

## Case 01: Marketplace Reconciliation

You are an operations analyst. Inspect these local files:

- `D:\agent-quality-eval\examples\gold-standard-eval-suite\tool-use-cases\data\case-01_marketplace_reconciliation\orders.csv`
- `D:\agent-quality-eval\examples\gold-standard-eval-suite\tool-use-cases\data\case-01_marketplace_reconciliation\payments.csv`
- `D:\agent-quality-eval\examples\gold-standard-eval-suite\tool-use-cases\data\case-01_marketplace_reconciliation\refunds.csv`
- `D:\agent-quality-eval\examples\gold-standard-eval-suite\tool-use-cases\data\case-01_marketplace_reconciliation\chargebacks.csv`

Reconcile the marketplace ledger for June 1-5, 2026.

Return:

1. A table of every discrepancy with `order_id`, issue type, amount, evidence, severity, and next action.
2. Total transport/payment exposure by category: missing or underpaid revenue, refund/customer liability, orphan payment liability, and confirmed chargeback loss.
3. A short executive summary.

Do not guess. Use the files as source of truth and show the arithmetic.

## Case 02: Checkout API Incident

You are the incident commander for a checkout outage. Inspect these local files:

- `D:\agent-quality-eval\examples\gold-standard-eval-suite\tool-use-cases\data\case-02_checkout_incident\deploys.csv`
- `D:\agent-quality-eval\examples\gold-standard-eval-suite\tool-use-cases\data\case-02_checkout_incident\metrics.csv`
- `D:\agent-quality-eval\examples\gold-standard-eval-suite\tool-use-cases\data\case-02_checkout_incident\logs.jsonl`
- `D:\agent-quality-eval\examples\gold-standard-eval-suite\tool-use-cases\data\case-02_checkout_incident\regional_failures.csv`

Write a post-incident analysis.

Return:

1. Timeline.
2. Root cause.
3. Blast radius with calculations.
4. Why rollback helped.
5. Immediate remediation and long-term prevention.
6. Any false leads you ruled out from the evidence.

Do not produce a generic incident template. The answer must cite concrete rows or facts from the files.

## Case 03: Access Policy Audit

You are a security reviewer. Inspect these local files:

- `D:\agent-quality-eval\examples\gold-standard-eval-suite\tool-use-cases\data\case-03_access_policy_audit\users.csv`
- `D:\agent-quality-eval\examples\gold-standard-eval-suite\tool-use-cases\data\case-03_access_policy_audit\resources.csv`
- `D:\agent-quality-eval\examples\gold-standard-eval-suite\tool-use-cases\data\case-03_access_policy_audit\access_log.csv`
- `D:\agent-quality-eval\examples\gold-standard-eval-suite\tool-use-cases\data\case-03_access_policy_audit\policy.md`

Audit the current access grants.

Return:

1. A violations table with `user_id`, user name, resource, permission, violated rule, severity, and required action.
2. A list of grants that are compliant and why.
3. Counts by severity.
4. The top three remediation priorities.

Do not rely on intuition. Join the files and apply the policy exactly.

