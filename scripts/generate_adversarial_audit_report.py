from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


FINDINGS = [
    {
        "id": "SEC-01",
        "severity": "critical",
        "status": "confirmed",
        "title": "Session ID path traversal permits reads outside the COT directory",
        "area": "Observation / local API",
        "evidence": "services/session_scanner.py · get_session_cot / delete_session",
        "detail": (
            "The session ID is interpolated into '<session>_cot.json' without a containment check. "
            "An isolated dynamic probe using '../outside' successfully read a file outside COT_DIR. "
            "FastAPI path decoding and route matching still need endpoint-level confirmation, but the storage boundary itself is broken."
        ),
        "remediation": "Accept only canonical session IDs, resolve the path, and require resolved_path.is_relative_to(COT_DIR.resolve()) before every read/delete.",
    },
    {
        "id": "SEC-02",
        "severity": "critical",
        "status": "static-confirmed",
        "title": "OTLP and hook writers do not consistently sanitize session IDs",
        "area": "OTel / hook ingestion",
        "evidence": "claude_otel_receiver.py · _bucket_by_session; codex_stream_hook.py · target path",
        "detail": "Locally supplied telemetry attributes can influence directory paths. Local-first reduces remote exposure, but any process running as the user can submit to the localhost receiver.",
        "remediation": "Use one shared safe_session_id function for OTLP, hook events, reads, deletes, and uplink paths; add traversal contract tests.",
    },
    {
        "id": "REL-01",
        "severity": "critical",
        "status": "dynamic-confirmed",
        "title": "Concurrent Eval writes lose results under SQLite locking",
        "area": "Eval persistence",
        "evidence": "evaluation/store.py · connect/save_turn_eval",
        "detail": "32 writers attempted 320 writes; only 140 persisted. 18 writer tasks failed with 'database is locked'. Journal mode was DELETE.",
        "remediation": "Enable WAL, set busy_timeout, use retry with bounded jitter, and serialize high-contention writes through one process-level queue.",
    },
    {
        "id": "EVAL-01",
        "severity": "high",
        "status": "fixed-in-1.0.3",
        "title": "Regression could reuse Turn Eval results created with stale Gold or stale trace content",
        "area": "Gold / Regression",
        "evidence": "evaluation/api.py · _build_or_load_turn_eval",
        "detail": "The 1.0.2 cache keyed only on session/turn and report version. The targeted fix adds Gold and turn-source fingerprints and reruns both baseline and candidate against the same current Gold.",
        "remediation": "Retain fingerprint tests and include evaluator/prompt schema versions in the fingerprint when those templates change.",
    },
    {
        "id": "EVAL-02",
        "severity": "high",
        "status": "fixed-in-1.0.3",
        "title": "Lexical Gold scoring produced systematic false negatives for semantic paraphrases",
        "area": "Reference Eval",
        "evidence": "evaluation/reference_eval.py · _deterministic_reference_eval",
        "detail": "Initial isolated execution classified all six expected real traces as fail when the LLM judge was unconfigured. Structured evidence could not outweigh character/token similarity. Dynamic evidence weighting now yields 3 pass and 3 partial while all six negative controls fail.",
        "remediation": "Calibrate thresholds on a human-labelled set; do not convert partial results into pass/fail gates without task-specific acceptance criteria.",
    },
    {
        "id": "EVAL-03",
        "severity": "high",
        "status": "confirmed",
        "title": "Judge parse failures are converted into a numeric 0.5",
        "area": "Batch Eval",
        "evidence": "evaluation/runner.py · _parse_judge_response",
        "detail": "Non-JSON or malformed judge output returns score 0.5. Provider failure is therefore mixed into quality statistics rather than represented as missing/invalid.",
        "remediation": "Return an invalid trial state, exclude it from quality aggregation, expose failure rate, and optionally retry with strict JSON repair.",
    },
    {
        "id": "EVAL-04",
        "severity": "high",
        "status": "confirmed",
        "title": "Experiment metric names do not match their implemented semantics",
        "area": "Aggregation",
        "evidence": "evaluation/models.py · pass_at_k/pass_power_k/overall_pass_rate",
        "detail": "pass_at_k is implemented as any-trial-pass, pass_power_k as all-trials-pass, and overall_pass_rate averages fractional scorer pass rates rather than case-level binary outcomes.",
        "remediation": "Rename metrics or implement the documented estimators; publish numerator, denominator, sample count, variance, and confidence intervals.",
    },
    {
        "id": "EVAL-05",
        "severity": "high",
        "status": "confirmed",
        "title": "A/B is an offline paired comparison, not an online controlled experiment",
        "area": "A/B testing",
        "evidence": "evaluation/compare.py and evaluation/api.py · turn comparison",
        "detail": "There is no random assignment, sample-ratio mismatch check, power calculation, confidence interval, or stopping rule. Calling this A/B can overstate statistical meaning.",
        "remediation": "Label it 'offline paired comparison' or add experimental units, stable randomization, SRM checks, effect sizes, confidence intervals, and predeclared stopping.",
    },
    {
        "id": "OBS-01",
        "severity": "high",
        "status": "confirmed",
        "title": "Configured COT_SCAN_DIRS are overwritten during backend config initialization",
        "area": "Observation discovery",
        "evidence": "assets/backend/config.py · COT_SCAN_DIRS assigned at lines 82 and 127",
        "detail": "The environment-aware list is replaced by COT_DIR plus legacy roots, silently hiding explicitly configured scan directories.",
        "remediation": "Build the final list once, preserve environment entries, deduplicate resolved paths, and test multi-root deployments.",
    },
    {
        "id": "OBS-02",
        "severity": "high",
        "status": "confirmed",
        "title": "Codex events without a session ID can attach to the newest rollout",
        "area": "Codex ingestion",
        "evidence": "assets/hooks/codex/codex_stream_hook.py · _latest_rollout_id",
        "detail": "Concurrent Codex sessions can be cross-attributed when hook payloads omit session_id.",
        "remediation": "Quarantine orphan events, correlate using process/thread metadata, and never attach to another active session solely by mtime.",
    },
    {
        "id": "OBS-03",
        "severity": "medium",
        "status": "confirmed",
        "title": "Cross-IDE session aliases can select the wrong COT file",
        "area": "Critic binding",
        "evidence": "evaluation/critic.py · find_cot_file",
        "detail": "The lookup tries bare, codex-prefixed, and codebuddy-prefixed IDs without requiring the hook's agent type, creating collisions across providers.",
        "remediation": "Use (agent_type, session_id) as the storage key and reject ambiguous matches.",
    },
    {
        "id": "PERF-01",
        "severity": "high",
        "status": "dynamic-confirmed",
        "title": "Session listing remains a synchronous full directory walk",
        "area": "Trace scale",
        "evidence": "assets/backend/services/session_scanner.py · scan_sessions",
        "detail": "Isolated measurements: 100 traces 35.77 ms cold/4.72 ms warm; 1,000 traces 349.61/40.70 ms; 5,000 traces 2,092.01/252.39 ms with +26.79 MB RSS on cold parse. This runs in the single backend worker.",
        "remediation": "Maintain an incremental index, paginate the API, move scans off the request thread, and expose scan age/partial state.",
    },
    {
        "id": "PERF-02",
        "severity": "high",
        "status": "confirmed",
        "title": "Critic sidecars have no process-wide concurrency limit or wall-clock kill",
        "area": "Process management",
        "evidence": "agent_critic_hook.py · Popen; evaluation/critic.py · 45s wait + provider timeout",
        "detail": "Bursting turns can create many long-lived child processes, increasing memory, handles, and model traffic.",
        "remediation": "Use a bounded queue/semaphore, process groups, hard deadlines, cancellation, and orphan cleanup at startup.",
    },
    {
        "id": "PERF-03",
        "severity": "medium",
        "status": "dynamic-confirmed",
        "title": "Gold upload accepts oversized payloads without a product limit",
        "area": "Upload friction / resource control",
        "evidence": "reference-answer normalize endpoint",
        "detail": "A 5 MB expected_answer was accepted with no warning in 88.4 ms. Larger payloads are retained in preview memory and can be written twice as raw and canonical data.",
        "remediation": "Set request and per-field limits, truncate only display excerpts, and return 413 with actionable guidance.",
    },
    {
        "id": "PKG-01",
        "severity": "high",
        "status": "confirmed",
        "title": "UPX-compressed unsigned-style bundle increases antivirus and startup friction",
        "area": "Windows packaging",
        "evidence": "agent-quality-eval.spec · upx=True",
        "detail": "UPX plus fire-and-forget child processes is a common enterprise antivirus friction pattern. The windowed executable also suppresses stderr.",
        "remediation": "Ship a non-UPX signed build, publish SHA-256 and SBOM, and retain a diagnostic console/log collection mode.",
    },
    {
        "id": "PKG-02",
        "severity": "medium",
        "status": "confirmed",
        "title": "Frozen asset completion marker does not prove asset integrity",
        "area": "Upgrade / recovery",
        "evidence": "agent_cot/frozen_entry.py · frozen asset extraction",
        "detail": "A partial or externally modified extraction can retain the .complete marker and will not self-heal.",
        "remediation": "Use a manifest of file hashes and atomically promote a fully verified temporary extraction.",
    },
    {
        "id": "UX-01",
        "severity": "medium",
        "status": "fixed-in-1.0.3",
        "title": "Native Windows title bar did not match the dark application shell",
        "area": "Desktop UI",
        "evidence": "agent_quality_eval/desktop.py · pywebview create_window",
        "detail": "The app used default light Win32 chrome above a dark in-app header. 1.0.3 applies best-effort DWM dark mode, border/text colors, and system backdrop while preserving native controls.",
        "remediation": "Keep a Windows 10 fallback and visually smoke-test DPI, resize, tray-close, and WebView2 fallback.",
    },
    {
        "id": "SCOPE-01",
        "severity": "info",
        "status": "resolved-in-1.0.3",
        "title": "VSCode/Copilot code was residual, not a supported product integration",
        "area": "Product scope",
        "evidence": "agent registry, hook assets, installer, extractor and frontend enums",
        "detail": "The residual adapter, hook, installer merger, provider detection, UI badges, and tests were removed. Current product coverage is Cursor, Claude, Codex, and CodeBuddy.",
        "remediation": "Keep product claims and registry tests aligned with those four integrations.",
    },
]


SOURCES = [
    ("OpenTelemetry GenAI semantic conventions", "https://opentelemetry.io/docs/specs/semconv/gen-ai/"),
    ("W3C Trace Context", "https://www.w3.org/TR/trace-context-1/"),
    ("OWASP Top 10 for LLM Applications 2025", "https://owasp.org/www-project-top-10-for-large-language-model-applications/"),
    ("OWASP Agentic Top 10 2026", "https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/"),
    ("NIST AI Risk Management Framework 1.0", "https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf"),
    ("NIST AI TEVV", "https://www.nist.gov/ai-test-evaluation-validation-and-verification-tevv"),
    ("NIST Secure Software Development Framework", "https://csrc.nist.gov/pubs/sp/800/218/final"),
    ("OpenAI Evaluation Best Practices", "https://developers.openai.com/api/docs/guides/evaluation-best-practices"),
]


def esc(value: Any) -> str:
    return html.escape(str(value))


def render_finding(item: dict[str, Any]) -> str:
    return f"""
    <article class="finding {esc(item['severity'])}">
      <div class="finding-id"><span>{esc(item['id'])}</span><b>{esc(item['severity'])}</b></div>
      <div class="finding-body">
        <div class="finding-title"><h3>{esc(item['title'])}</h3><em>{esc(item['status'])}</em></div>
        <p>{esc(item['detail'])}</p>
        <dl><div><dt>Area</dt><dd>{esc(item['area'])}</dd></div>
        <div><dt>Evidence</dt><dd><code>{esc(item['evidence'])}</code></dd></div>
        <div><dt>Remediation</dt><dd>{esc(item['remediation'])}</dd></div></dl>
      </div>
    </article>"""


def main() -> None:
    stress = json.loads((ROOT / "product-stress-results.json").read_text(encoding="utf-8"))
    gold = json.loads(
        (ROOT / "agent-regression-gold-dataset" / "eval-results.json").read_text(encoding="utf-8")
    )
    counts = {
        severity: sum(1 for item in FINDINGS if item["severity"] == severity)
        for severity in ("critical", "high", "medium", "info")
    }
    scan_rows = "".join(
        f"<tr><td>{row['trace_count']:,}</td><td>{row['cold_ms']:.2f} ms</td>"
        f"<td>{row['warm_ms']:.2f} ms</td><td>{row['rss_delta_mb']:.2f} MB</td></tr>"
        for row in stress["session_scanner"]
    )
    source_links = "".join(
        f'<li><a href="{esc(url)}">{esc(label)}</a></li>' for label, url in SOURCES
    )
    findings = "".join(render_finding(item) for item in FINDINGS)
    sqlite = stress["sqlite_concurrency"]
    gold_summary = gold["summary"]
    output = ROOT / "agent-quality-audit-1.0.2.html"
    output.write_text(
        f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent Quality Eval 1.0.2 Adversarial Audit</title>
<style>
:root{{color-scheme:dark;font-family:Inter,"Segoe UI",sans-serif;background:#080b12;color:#e5e7eb}}
*{{box-sizing:border-box}}body{{margin:0;background:#080b12}}a{{color:#7dd3fc}}
main{{width:min(1240px,calc(100% - 40px));margin:auto;padding:46px 0 80px}}
.hero{{display:grid;grid-template-columns:1.4fr .6fr;gap:36px;padding-bottom:28px;border-bottom:1px solid #263244}}
.kicker{{color:#7dd3fc;font-size:12px;letter-spacing:.12em;text-transform:uppercase}}h1{{font-size:34px;margin:8px 0 12px}}
.hero p,.muted{{color:#94a3b8;line-height:1.65}}.verdict{{border:1px solid #334155;padding:18px;background:#0d1420}}
.verdict b{{display:block;color:#fbbf24;font-size:24px;margin:5px 0}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);margin:24px 0}}
.stats div{{padding:18px;border:1px solid #263244;border-right:0}}.stats div:last-child{{border-right:1px solid #263244}}
.stats strong{{display:block;font-size:24px}}section{{margin-top:34px}}h2{{font-size:21px;margin-bottom:14px}}
.finding{{display:grid;grid-template-columns:120px 1fr;border:1px solid #263244;border-bottom:0;background:#0d1420}}
.finding:last-child{{border-bottom:1px solid #263244}}.finding-id{{padding:18px;border-right:1px solid #263244}}
.finding-id span,.finding-id b{{display:block}}.finding-id span{{color:#94a3b8;font-size:12px}}.finding-id b{{margin-top:8px;text-transform:uppercase;font-size:11px}}
.critical .finding-id b{{color:#fca5a5}}.high .finding-id b{{color:#fdba74}}.medium .finding-id b{{color:#fde68a}}.info .finding-id b{{color:#7dd3fc}}
.finding-body{{padding:17px 20px}}.finding-title{{display:flex;justify-content:space-between;gap:20px}}h3{{margin:0;font-size:16px}}
.finding-title em{{color:#94a3b8;font-size:11px;font-style:normal;text-transform:uppercase}}.finding-body p{{color:#cbd5e1;line-height:1.55}}
dl{{display:grid;gap:7px;margin:0}}dl div{{display:grid;grid-template-columns:100px 1fr;gap:10px}}dt{{color:#64748b;font-size:12px}}dd{{margin:0;color:#94a3b8;font-size:12px}}code{{color:#bae6fd}}
.evidence-grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.panel{{border:1px solid #263244;background:#0d1420;padding:18px}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #263244;text-align:left;font-size:12px}}th{{color:#7dd3fc}}
.sources li{{margin:8px 0;color:#94a3b8}}footer{{margin-top:40px;padding-top:20px;border-top:1px solid #263244;color:#64748b;font-size:12px}}
@media(max-width:800px){{.hero,.evidence-grid{{grid-template-columns:1fr}}.stats{{grid-template-columns:1fr 1fr}}.finding{{grid-template-columns:1fr}}.finding-id{{border-right:0;border-bottom:1px solid #263244}}}}
</style></head><body><main>
<header class="hero"><div><span class="kicker">Adversarial product review · 2026-07-17</span>
<h1>Agent Quality Eval 1.0.2</h1><p>Static code review, isolated dynamic probes, Gold/reference evaluation,
scale measurements, and current official standards. This is an engineering risk assessment, not a legal compliance certification.</p></div>
<div class="verdict"><small>Release readiness</small><b>Conditional</b><span class="muted">1.0.3 fixes the desktop shell, Gold workflow, stale regression inputs, semantic weighting, and removes residual VSCode scope. Security and concurrency blockers remain.</span></div></header>
<div class="stats"><div><small>Critical</small><strong>{counts['critical']}</strong></div><div><small>High</small><strong>{counts['high']}</strong></div>
<div><small>Medium</small><strong>{counts['medium']}</strong></div><div><small>Automated tests</small><strong>301</strong></div></div>
<section><h2>Scope and data boundaries</h2><div class="panel"><p>Current product integrations: Cursor, Claude, Codex, and CodeBuddy.
VSCode/Copilot was residual code and is removed in 1.0.3. The product is assessed as a local, single-user app with no active team-sharing feature.
Local prompt storage is not classified as a leak. Data-boundary review applies only to path escape, localhost process trust, and explicitly configured external judge calls.</p></div></section>
<section><h2>Dynamic evidence</h2><div class="evidence-grid"><div class="panel"><h3>Trace scan scaling</h3><table><thead><tr><th>Traces</th><th>Cold</th><th>Warm</th><th>RSS delta</th></tr></thead><tbody>{scan_rows}</tbody></table></div>
<div class="panel"><h3>Eval persistence</h3><p><strong>{sqlite['writes_persisted']}/{sqlite['writes_expected']}</strong> writes persisted with {sqlite['error_count']} writer failures.</p>
<p class="muted">32 concurrent writers · {sqlite['elapsed_ms']:.2f} ms · journal mode {esc(sqlite['journal_mode'])}</p>
<h3>Gold suite</h3><p><strong>{gold_summary['controls_detected']}/6</strong> synthetic regressions detected; observed traces: {gold_summary['observed_pass']} pass, {gold_summary['observed_partial']} partial, {gold_summary['observed_fail']} fail.</p></div></div></section>
<section><h2>Findings</h2>{findings}</section>
<section><h2>Recommended release gates</h2><div class="panel"><ol>
<li>Block untrusted session IDs from escaping every storage root.</li><li>Enable WAL/busy timeout/retry and prove 100% persistence under the concurrency test.</li>
<li>Move session scans and long Eval jobs off the single request worker.</li><li>Bound critic sidecars and terminate overdue process groups.</li>
<li>Calibrate Gold thresholds and LLM-judge agreement on a human-labelled dataset before treating scores as release gates.</li>
<li>Ship a signed non-UPX build with SBOM, hash, offline VM smoke test, and asset integrity manifest.</li></ol></div></section>
<section><h2>Primary sources</h2><ul class="sources">{source_links}</ul></section>
<footer>Baseline: dist/agent-quality-eval-1.0.2.exe · Targeted implementation: 1.0.3 source · Dynamic probes ran in an isolated workspace that was deleted after results were captured.</footer>
</main></body></html>""",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
