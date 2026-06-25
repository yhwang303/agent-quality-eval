# Skill A/B Testing Flow

This flow answers one question: does enabling a skill improve the same agent on
the same dataset under the same scoring policy?

## 1. Define The Variant

Keep everything identical except the skill switch.

- Baseline: agent without the skill.
- Candidate: agent with the skill.
- Same dataset, model family, temperature, test cases, assertions, and trial count.

Examples of the variant field:

- different `system_prompt`
- different `agent_client_uuid`
- different API endpoint
- different custom Python provider config
- different environment variable read by your agent runtime

## 2. Create One Dataset

Use stable cases that represent the workflow the skill should improve.

Good skill-eval cases usually include:

- tasks the skill should solve better
- common happy paths
- hard edge cases
- safety or PII cases
- cases where the skill should not over-trigger

Each case should have an `id`, `question`, optional `expected_answer`,
`priority`, and assertions.

## 3. Run Baseline

Create a config with only the no-skill provider and run it.

```powershell
.\dist\agent-quality-eval-0.1.0.exe eval run .\configs\skill_without.yaml --promote-baseline
```

Save the printed `experiment` id.

## 4. Run Candidate

Create a config that uses the same dataset but points to the skill-enabled
provider.

```powershell
.\dist\agent-quality-eval-0.1.0.exe eval run .\configs\skill_with.yaml
```

Save the printed `experiment` id.

## 5. Compare

```powershell
.\dist\agent-quality-eval-0.1.0.exe eval compare --baseline <baseline_exp_id> --candidate <candidate_exp_id>
```

Read:

- win/tie/loss
- average score delta
- pass-rate delta
- latency delta
- regressed case list

## 6. Gate Regression

```powershell
.\dist\agent-quality-eval-0.1.0.exe eval regression --baseline <baseline_exp_id> --candidate <candidate_exp_id>
```

Exit code:

- `0`: candidate passes the gate
- `1`: candidate regressed and should not be promoted

Default blocking rules:

- overall pass rate cannot drop by more than 3 percentage points
- high-priority cases cannot newly fail
- p95 latency cannot increase by more than 20%
- safety or PII failures block release

## 7. Promote

If the skill version passes and is the new preferred behavior:

```powershell
.\dist\agent-quality-eval-0.1.0.exe eval promote-baseline <candidate_exp_id>
```

## Notes

For a quick local example, start from:

```powershell
.\src\agent_quality_eval\templates\skill_ab_eval.yaml
```

The template includes two mock variants in one file for demonstration. For a
real release gate, prefer two explicit configs, one no-skill baseline and one
skill-enabled candidate, so each experiment has a clear identity.
