"""Unified CLI for observation and evaluation."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import click

from . import __version__
from .evaluation.compare import compare_experiments
from .evaluation.config import load_eval_config, write_default_config
from .evaluation.regression import RegressionPolicy, detect_regression
from .evaluation.runner import ExperimentRunner
from .evaluation.store import DatasetStore, default_db_path, default_home


def _die(message: str, code: int = 1) -> NoReturn:
    click.secho(f"error: {message}", fg="red", err=True)
    raise SystemExit(code)


@dataclass
class BootstrapResult:
    home: Path
    db_path: Path
    sample_config: Path
    sample_created: bool


def bootstrap_workspace(
    *,
    home: str | Path | None = None,
    config_path: str | Path | None = None,
    overwrite_config: bool = False,
) -> BootstrapResult:
    """Create the local product workspace without clobbering user config by default."""
    root = Path(home) if home else default_home()
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    config_target = Path(config_path) if config_path else root / "configs" / "sample_eval.yaml"
    sample_created = False
    if overwrite_config or not config_target.exists():
        write_default_config(config_target)
        sample_created = True
    store = DatasetStore(root / "data" / "eval.db")
    return BootstrapResult(
        home=root,
        db_path=store.db_path,
        sample_config=config_target,
        sample_created=sample_created,
    )


def bootstrap_observation_runtime() -> None:
    """Make product launches use the bundled observation assets, not stale config paths."""
    try:
        from agent_cot.installer.config import load_config, save_config

        cfg = load_config()
        changed = False
        if cfg.dashboard_repo:
            cfg.dashboard_repo = None
            changed = True
        if cfg.cot_extractor_repo and "_MEI" in str(cfg.cot_extractor_repo):
            cfg.cot_extractor_repo = None
            changed = True
        if changed:
            save_config(cfg)
    except Exception:
        # Observation startup can still fall back to bundled assets; this is best-effort.
        return


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version", prog_name="agent-eval")
def main() -> None:
    """agent-eval: local agent observability plus eval pipeline."""


@main.command("init", help="Initialize local eval workspace and sample config.")
@click.option("--home", type=click.Path(file_okay=False), default=None, help="Workspace home. Defaults to ~/.agent-quality-eval.")
@click.option("--config", "config_path", type=click.Path(dir_okay=False), default=None, help="Sample config path to write.")
@click.option("--force", is_flag=True, help="Overwrite the sample config if it already exists.")
def cmd_init(home: str | None, config_path: str | None, force: bool) -> None:
    result = bootstrap_workspace(home=home, config_path=config_path, overwrite_config=force)
    bootstrap_observation_runtime()
    click.secho("agent-eval init - done", fg="green", bold=True)
    click.echo(f"  home   : {result.home}")
    click.echo(f"  db     : {result.db_path}")
    click.echo(f"  sample : {result.sample_config}")
    click.echo(f"  sample {'created' if result.sample_created else 'exists'}")


@main.group("eval", help="Run evals, compare experiments, and manage baselines.")
def cmd_eval() -> None:
    pass


@cmd_eval.command("run", help="Run an eval config.")
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--db", "db_path", type=click.Path(dir_okay=False), default=None, help="SQLite DB path.")
@click.option("--json", "as_json", is_flag=True, help="Print machine-readable result.")
@click.option("--promote-baseline", is_flag=True, help="Promote this experiment as dataset baseline if it passes.")
def cmd_eval_run(config: str, db_path: str | None, as_json: bool, promote_baseline: bool) -> None:
    cfg = load_eval_config(config)
    if db_path:
        cfg.store_path = db_path
    runner = ExperimentRunner(cfg)
    result = runner.run()
    if promote_baseline and result.overall_pass_rate >= cfg.pass_threshold:
        runner.store.promote_baseline(result.experiment_id)
    if as_json:
        click.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        click.secho("eval run - completed", fg="green", bold=True)
        click.echo(f"  experiment : {result.experiment_id}")
        click.echo(f"  dataset    : {result.dataset_name}@{result.dataset_version}")
        click.echo(f"  providers  : {', '.join(result.providers)}")
        click.echo(f"  pass rate  : {result.overall_pass_rate * 100:.1f}%")
        click.echo(f"  avg score  : {result.average_score:.2f}")
        click.echo(f"  avg latency: {result.average_response_time:.2f}s")
        click.echo(f"  db         : {runner.store.db_path}")
    raise SystemExit(0 if result.overall_pass_rate >= cfg.pass_threshold else 1)


@cmd_eval.command("compare", help="Compare two experiments with paired A/B stats.")
@click.option("--baseline", "baseline_id", required=True, help="Baseline experiment id.")
@click.option("--candidate", "candidate_id", required=True, help="Candidate experiment id.")
@click.option("--db", "db_path", type=click.Path(dir_okay=False), default=None, help="SQLite DB path.")
@click.option("--json", "as_json", is_flag=True)
def cmd_eval_compare(baseline_id: str, candidate_id: str, db_path: str | None, as_json: bool) -> None:
    try:
        result = compare_experiments(baseline_id, candidate_id, store=DatasetStore(db_path))
    except KeyError as exc:
        _die(str(exc))
    if as_json:
        click.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return
    click.secho("eval compare", bold=True)
    click.echo(f"  baseline : {baseline_id}")
    click.echo(f"  candidate: {candidate_id}")
    click.echo(f"  win/tie/loss: {result.win}/{result.tie}/{result.loss}")
    click.echo(f"  score delta : {result.average_score_delta:+.3f}")
    click.echo(f"  pass delta  : {result.average_pass_rate_delta:+.3f}")
    click.echo(f"  latency delta: {result.average_latency_delta:+.3f}s")
    if result.regressions:
        click.secho("  regressions:", fg="yellow")
        for item in result.regressions[:20]:
            click.echo(f"    - {item['case_id']}: score {item['score_delta']:+.3f}, pass {item['pass_rate_delta']:+.3f}")


@cmd_eval.command("regression", help="Run baseline regression gate.")
@click.option("--baseline", "baseline_id", required=True, help="Baseline experiment id.")
@click.option("--candidate", "candidate_id", required=True, help="Candidate experiment id.")
@click.option("--db", "db_path", type=click.Path(dir_okay=False), default=None)
@click.option("--policy", "policy_path", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--json", "as_json", is_flag=True)
def cmd_eval_regression(
    baseline_id: str,
    candidate_id: str,
    db_path: str | None,
    policy_path: str | None,
    as_json: bool,
) -> None:
    policy_data = {}
    if policy_path:
        policy_data = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    try:
        result = detect_regression(
            baseline_id,
            candidate_id,
            store=DatasetStore(db_path),
            policy=RegressionPolicy.from_dict(policy_data),
        )
    except KeyError as exc:
        _die(str(exc))
    if as_json:
        click.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        color = "green" if result.passed else "red"
        click.secho("regression gate - " + ("PASS" if result.passed else "FAIL"), fg=color, bold=True)
        for reason in result.reasons:
            click.echo(f"  - {reason}")
    raise SystemExit(0 if result.passed else 1)


@cmd_eval.command("promote-baseline", help="Promote an experiment as the dataset baseline.")
@click.argument("experiment_id")
@click.option("--db", "db_path", type=click.Path(dir_okay=False), default=None)
def cmd_eval_promote_baseline(experiment_id: str, db_path: str | None) -> None:
    store = DatasetStore(db_path)
    try:
        store.promote_baseline(experiment_id)
    except KeyError as exc:
        _die(str(exc))
    click.secho(f"promoted baseline: {experiment_id}", fg="green")


@cmd_eval.command("export", help="Export an experiment JSON from the local store.")
@click.argument("experiment_id")
@click.option("--db", "db_path", type=click.Path(dir_okay=False), default=None)
@click.option("--out", "out_path", type=click.Path(dir_okay=False), default=None)
def cmd_eval_export(experiment_id: str, db_path: str | None, out_path: str | None) -> None:
    store = DatasetStore(db_path)
    try:
        data = store.get_experiment_dict(experiment_id)
    except KeyError as exc:
        _die(str(exc))
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
        click.echo(out_path)
    else:
        click.echo(text)


@main.group(
    "observe",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    help="Delegate to the copied observation-agent command tree.",
)
def cmd_observe() -> None:
    pass


def _run_agent_cot(args: list[str]) -> None:
    from agent_cot.cli import main as agent_cot_main

    if args and args[0] in {"start", "init"}:
        bootstrap_observation_runtime()
    try:
        agent_cot_main(args=args, prog_name="agent-cot", standalone_mode=False)
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from exc


def _passthrough_command(name: str):
    @cmd_observe.command(
        name,
        context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
        help=f"Run `agent-cot {name}`.",
    )
    @click.pass_context
    def _cmd(ctx: click.Context) -> None:
        _run_agent_cot([name, *ctx.args])

    return _cmd


for _name in ["agents", "init", "start", "stop", "status", "doctor", "upgrade", "uninstall"]:
    _passthrough_command(_name)


@cmd_observe.group("otlp", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def cmd_observe_otlp(ctx: click.Context) -> None:
    if ctx.invoked_subcommand is None and ctx.args:
        _run_agent_cot(["otlp", *ctx.args])


@cmd_observe_otlp.command("list-presets", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def cmd_observe_otlp_list(ctx: click.Context) -> None:
    _run_agent_cot(["otlp", "list-presets", *ctx.args])


@cmd_observe_otlp.command("send", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def cmd_observe_otlp_send(ctx: click.Context) -> None:
    _run_agent_cot(["otlp", "send", *ctx.args])


@main.command("doctor", help="Show eval workspace health.")
def cmd_doctor() -> None:
    db = default_db_path()
    click.secho("agent-eval doctor", bold=True)
    click.echo(f"  version : {__version__}")
    click.echo(f"  home    : {default_home()}")
    click.echo(f"  db      : {db} ({'exists' if db.exists() else 'missing'})")
    try:
        import agent_cot

        click.echo(f"  observe : agent_cot {agent_cot.__version__}")
    except Exception as exc:
        click.echo(f"  observe : unavailable ({exc})")


def _entrypoint() -> None:
    try:
        main()
    except KeyboardInterrupt:
        _die("interrupted", 130)


if __name__ == "__main__":
    _entrypoint()
