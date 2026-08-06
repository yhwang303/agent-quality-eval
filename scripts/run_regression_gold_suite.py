from __future__ import annotations

import html
import json
import os
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "agent-regression-gold-dataset"
WORKSPACE = DATASET / ".eval-workspace"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def turn_eval_from_trace(trace: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    turn = trace["turns"][0]
    steps = turn.get("steps") or []
    tools = [str(step.get("tool_name") or "") for step in steps if step.get("tool_name")]
    report = {
        "session_id": trace["session_id"],
        "turn_index": int(turn["turn_index"]),
        "metrics": {
            "user_query": turn.get("user_query") or "",
            "final_response": turn.get("final_response") or "",
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "tool_count": len(tools),
            "tool_name_counts": {
                name: tools.count(name) for name in sorted(set(tools))
            },
        },
    }
    context = {"cot": trace, "turn": turn, "transcript": {}, "otel": {}, "overview": {}}
    return report, context


def verdict_rank(value: str) -> int:
    return {"pass": 2, "partial": 1, "fail": 0}.get(value, -1)


def render_html(results: list[dict[str, Any]]) -> str:
    rows = []
    for item in results:
        status = "PASS" if item["control_detected"] else "FAIL"
        rows.append(
            f"""
            <article class="case">
              <header>
                <div><span>{html.escape(item['agent_type'])} · {html.escape(item['complexity'])}</span>
                <h2>{html.escape(item['case_id'])}</h2></div>
                <b class="{status.lower()}">{status}</b>
              </header>
              <div class="metrics">
                <section><small>Observed score</small><strong>{item['observed_score']:.3f}</strong>
                <span>{html.escape(item['observed_verdict'])}</span></section>
                <section><small>Negative-control score</small><strong>{item['negative_score']:.3f}</strong>
                <span>{html.escape(item['negative_verdict'])}</span></section>
                <section><small>Regression delta</small><strong>{item['score_delta']:+.3f}</strong>
                <span>candidate − baseline</span></section>
                <section><small>Gold normalization</small><strong>{item['mapping_count']}</strong>
                <span>mapped fields · {item['warning_count']} warnings</span></section>
              </div>
              <details><summary>Evaluation evidence</summary>
                <pre>{html.escape(json.dumps(item['evidence'], ensure_ascii=False, indent=2))}</pre>
              </details>
            </article>
            """
        )
    passed = sum(1 for item in results if item["control_detected"])
    observed_accepted = sum(
        1 for item in results if item["observed_verdict"] in {"pass", "partial"}
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AQE Gold Regression Evaluation</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, "Segoe UI", sans-serif; background:#080b12; color:#e5e7eb; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:#080b12; }}
main {{ width:min(1180px,calc(100% - 40px)); margin:0 auto; padding:42px 0 70px; }}
.hero {{ display:grid; grid-template-columns:1fr auto; gap:30px; align-items:end; border-bottom:1px solid #283244; padding-bottom:24px; }}
.eyebrow, small, header span {{ color:#7dd3fc; font-size:12px; letter-spacing:.08em; text-transform:uppercase; }}
h1 {{ margin:8px 0; font-size:30px; }} .hero p {{ color:#94a3b8; max-width:760px; line-height:1.65; }}
.summary {{ border:1px solid #334155; padding:16px 20px; min-width:180px; }} .summary strong {{ display:block; font-size:28px; color:#5eead4; }}
.case {{ margin-top:20px; border:1px solid #263244; background:#0d1420; }}
.case header {{ display:flex; justify-content:space-between; align-items:center; gap:16px; padding:18px 20px; border-bottom:1px solid #263244; }}
h2 {{ margin:4px 0 0; font-size:17px; }} header b {{ padding:6px 10px; border:1px solid; font-size:12px; }}
.pass {{ color:#5eead4; border-color:#2dd4bf66; }} .fail {{ color:#fca5a5; border-color:#f8717166; }}
.metrics {{ display:grid; grid-template-columns:repeat(4,1fr); }}
.metrics section {{ padding:18px 20px; border-right:1px solid #263244; }} .metrics section:last-child {{ border-right:0; }}
.metrics strong {{ display:block; margin:6px 0 3px; font-size:22px; }} .metrics span {{ color:#94a3b8; font-size:12px; }}
details {{ border-top:1px solid #263244; padding:13px 20px; }} summary {{ cursor:pointer; color:#cbd5e1; }}
pre {{ overflow:auto; padding:14px; background:#080b12; color:#cbd5e1; font-size:12px; line-height:1.5; }}
footer {{ margin-top:28px; color:#64748b; font-size:12px; }}
@media(max-width:800px) {{ .hero {{ grid-template-columns:1fr; }} .metrics {{ grid-template-columns:1fr 1fr; }} }}
</style>
</head>
<body><main>
  <section class="hero">
    <div><span class="eyebrow">Agent Quality Eval · isolated verification</span>
    <h1>Gold normalization and regression evaluation</h1>
    <p>Six minimized real traces were evaluated against independently authored Gold contracts.
    Each raw user upload was normalized through the production parser, then its reviewed canonical
    result was replayed in this isolated evaluator. Explicitly synthetic negative controls verify
    that the evaluator detects degraded behavior; they are not presented as real agent outputs.</p></div>
    <div class="summary"><small>Controls detected</small><strong>{passed}/{len(results)}</strong><span>{observed_accepted}/{len(results)} observed traces pass or partial</span></div>
  </section>
  {''.join(rows)}
  <footer>Generated from agent-regression-gold-dataset · isolated home: .eval-workspace · no production data was modified.</footer>
</main></body></html>"""


def main() -> None:
    local_settings = Path.home() / ".agent-quality-eval" / "config" / "settings.json"
    isolated_settings = WORKSPACE / "config" / "settings.json"
    judge_settings_reused = local_settings.exists()
    if judge_settings_reused:
        isolated_settings.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_settings, isolated_settings)

    os.environ["AGENT_QUALITY_EVAL_HOME"] = str(WORKSPACE)
    os.environ["AGENT_COT_DATA_ROOT"] = str(WORKSPACE / "cot-data")

    from agent_quality_eval.evaluation.reference_eval import (
        evaluate_turn_against_reference,
        get_reference_dataset,
        upload_reference_dataset,
    )

    manifest = load_json(DATASET / "manifest.json")
    # eval-pipeline-regression 类用例（如 case-07）没有 canonical_gold /
    # negative_control——它们守的是提取器+评估管线回归，不参与本套件。
    runnable = [item for item in manifest["cases"] if item.get("canonical_gold")]
    skipped_cases = len(manifest["cases"]) - len(runnable)
    results = []
    for item in runnable:
        canonical_path = DATASET / item["canonical_gold"]
        canonical = load_json(canonical_path)
        summary = upload_reference_dataset(
            canonical_path.name,
            json.dumps(canonical["dataset"], ensure_ascii=False),
        )
        dataset = get_reference_dataset(summary["dataset_id"])
        case_id = dataset["cases"][0]["id"]

        observed_trace = load_json(DATASET / item["trace"])
        observed_eval, observed_context = turn_eval_from_trace(observed_trace)
        observed = evaluate_turn_against_reference(
            session_id=observed_trace["session_id"],
            turn_index=observed_eval["turn_index"],
            dataset_id=summary["dataset_id"],
            case_id=case_id,
            turn_eval=observed_eval,
            turn_context=observed_context,
        )

        negative_trace = load_json(DATASET / item["negative_control"])
        negative_trace["session_id"] = f"{observed_trace['session_id']}-negative"
        negative_eval, negative_context = turn_eval_from_trace(negative_trace)
        negative = evaluate_turn_against_reference(
            session_id=negative_trace["session_id"],
            turn_index=negative_eval["turn_index"],
            dataset_id=summary["dataset_id"],
            case_id=case_id,
            turn_eval=negative_eval,
            turn_context=negative_context,
        )

        normalization = load_json(DATASET / item["normalization_report"])
        observed_score = float(observed.get("final_score") or 0)
        negative_score = float(negative.get("final_score") or 0)
        control_detected = (
            negative_score < observed_score
            and verdict_rank(str(negative.get("verdict"))) <= verdict_rank(str(observed.get("verdict")))
        )
        results.append(
            {
                **item,
                "dataset_id": summary["dataset_id"],
                "observed_score": observed_score,
                "observed_verdict": observed.get("verdict"),
                "negative_score": negative_score,
                "negative_verdict": negative.get("verdict"),
                "score_delta": negative_score - observed_score,
                "control_detected": control_detected,
                "mapping_count": len(normalization.get("mapping") or []),
                "warning_count": len(normalization.get("warnings") or []),
                "evidence": {
                    "observed_deterministic": observed.get("deterministic"),
                    "negative_deterministic": negative.get("deterministic"),
                    "observed_llm_status": (observed.get("llm_compare") or {}).get("status"),
                    "negative_llm_status": (negative.get("llm_compare") or {}).get("status"),
                    "normalization_hash": normalization.get("canonical_hash"),
                },
            }
        )

    payload = {
        "schema_version": "aqe-gold-suite-results-v1",
        "execution": {
            "mode": "isolated reference eval",
            "case_count": len(results),
            "skipped_non_gold_cases": skipped_cases,
            "negative_controls_are_synthetic": True,
            "production_data_modified": False,
            "local_judge_settings_reused": judge_settings_reused,
        },
        "summary": {
            "controls_detected": sum(1 for item in results if item["control_detected"]),
            "all_controls_detected": all(item["control_detected"] for item in results),
            "observed_pass": sum(1 for item in results if item["observed_verdict"] == "pass"),
            "observed_partial": sum(1 for item in results if item["observed_verdict"] == "partial"),
            "observed_fail": sum(1 for item in results if item["observed_verdict"] == "fail"),
        },
        "results": results,
    }
    (DATASET / "eval-results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (DATASET / "eval-results.html").write_text(render_html(results), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
