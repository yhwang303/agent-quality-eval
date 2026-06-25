"""JSON and HTML reporting."""

from __future__ import annotations

import html
import json
from pathlib import Path

from .config import EvalConfig
from .models import ExperimentResult


class ReportService:
    def write_outputs(self, result: ExperimentResult, config: EvalConfig) -> dict[str, str]:
        out_dir = Path(config.output_dir)
        if not out_dir.is_absolute():
            out_dir = config.base_dir / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        json_name = config.output_file or f"{result.experiment_id}.json"
        html_name = config.report_file or f"{result.experiment_id}.html"
        json_path = out_dir / json_name
        html_path = out_dir / html_name
        json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        html_path.write_text(self.render_html(result), encoding="utf-8")
        return {"json": str(json_path), "html": str(html_path)}

    def render_html(self, result: ExperimentResult) -> str:
        rows = []
        for case in result.case_results:
            for trial in case.trials:
                score_rows = "".join(
                    f"<tr><td>{html.escape(s.name)}</td><td>{s.score:.2f}</td>"
                    f"<td>{'PASS' if s.passed else 'FAIL'}</td>"
                    f"<td>{html.escape(s.reason)}</td></tr>"
                    for s in trial.scores
                )
                rows.append(
                    f"""
                    <section class="case {'pass' if trial.passed else 'fail'}">
                      <header>
                        <strong>{html.escape(case.test_case.id)}</strong>
                        <span>{html.escape(case.provider_name)}</span>
                        <span>trial {trial.trial_number}</span>
                        <span>{trial.pass_rate * 100:.0f}%</span>
                        <span>{trial.response_time:.2f}s</span>
                      </header>
                      <p class="question">{html.escape(case.test_case.question)}</p>
                      <pre>{html.escape(trial.answer)}</pre>
                      <table>
                        <thead><tr><th>Score</th><th>Value</th><th>Status</th><th>Reason</th></tr></thead>
                        <tbody>{score_rows}</tbody>
                      </table>
                    </section>
                    """
                )
        body = "\n".join(rows)
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Agent Eval Report - {html.escape(result.experiment_id)}</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f6f7f9;color:#1f2937}}
.wrap{{max-width:1180px;margin:0 auto;padding:24px}}
.hero{{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:20px;margin-bottom:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-top:16px}}
.metric{{background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:12px}}
.metric b{{display:block;font-size:24px;margin-top:6px}}
.case{{background:#fff;border-left:4px solid #ef4444;border-radius:8px;margin:14px 0;padding:16px;border-top:1px solid #e5e7eb;border-right:1px solid #e5e7eb;border-bottom:1px solid #e5e7eb}}
.case.pass{{border-left-color:#16a34a}}
header{{display:flex;gap:12px;align-items:center;flex-wrap:wrap;color:#4b5563}}
.question{{font-weight:600}}
pre{{background:#111827;color:#f9fafb;border-radius:6px;padding:12px;white-space:pre-wrap;max-height:280px;overflow:auto}}
table{{width:100%;border-collapse:collapse;margin-top:10px}}
td,th{{border-bottom:1px solid #e5e7eb;padding:8px;text-align:left;vertical-align:top}}
th{{background:#f9fafb}}
</style>
</head>
<body>
<main class="wrap">
  <section class="hero">
    <h1>{html.escape(result.name)}</h1>
    <p>{html.escape(result.experiment_id)} · dataset {html.escape(result.dataset_name)}@{html.escape(result.dataset_version)}</p>
    <div class="grid">
      <div class="metric">Overall Pass Rate<b>{result.overall_pass_rate * 100:.1f}%</b></div>
      <div class="metric">Average Score<b>{result.average_score:.2f}</b></div>
      <div class="metric">Average Response<b>{result.average_response_time:.2f}s</b></div>
      <div class="metric">Trials<b>{result.total_trials}</b></div>
    </div>
  </section>
  {body}
</main>
</body>
</html>"""
