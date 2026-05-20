"""TnSBench CLI built on typer."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .core.config import (
    ADVERSARIAL_TASKS_PATH,
    BENIGN_TASKS_PATH,
    REQUIRED_TASK_DISTRIBUTION,
    TASKS_PATH,
)
from .runner.aggregate import aggregate, load_results
from .runner.report import write_report
from .runner.run import run_benchmark
from .tasks.lint import (
    lint_tasks,
    summarize_distribution,
    summarize_strategies,
    summarize_diversity,
)
from .tasks.task_loader import load_tasks, save_tasks

app = typer.Typer(add_completion=False, no_args_is_help=True, help="TnSBench CLI (v0; no MCP).")
console = Console()


@app.command("generate-data")
def cli_generate_data() -> None:
    """Smoke check: rebuilds in-memory base data and prints sizes."""
    from .env.database import RetailDB

    db = RetailDB()
    console.print(
        f"users={len(db.users)} products={len(db.products)} orders={len(db.orders)} "
        f"promos={len(db.promos)} support_cases={len(db.support_cases)}"
    )


@app.command("generate-tasks")
def cli_generate_tasks() -> None:
    """Generate the 100-task TnSBench-Hard dataset and run the linter."""
    from .tasks.generator_hard import build_all_tasks_hard, write_split_files_hard

    tasks = build_all_tasks_hard()
    n_adv, n_ben = write_split_files_hard(tasks)
    ok, errors = lint_tasks(tasks)
    console.print(
        f"Wrote [bold]{len(tasks)}[/bold] tasks: "
        f"{n_adv} adversarial -> {ADVERSARIAL_TASKS_PATH.name}, "
        f"{n_ben} benign_control -> {BENIGN_TASKS_PATH.name}, combined -> {TASKS_PATH.name}"
    )
    console.print(f"Distribution: {summarize_distribution(tasks)}")
    console.print(f"Diversity: {summarize_diversity(tasks)}")
    if not ok:
        for e in errors:
            console.print(f"[red]  - {e}[/red]")
        raise typer.Exit(code=1)
    console.print("[green]Lint OK.[/green]")


@app.command("validate-tasks")
def cli_validate_tasks() -> None:
    """Validate the existing tasks.jsonl (combined)."""
    tasks = load_tasks()
    console.print(f"Loaded {len(tasks)} tasks. Distribution: {summarize_distribution(tasks)}")
    console.print(f"Diversity: {summarize_diversity(tasks)}")
    ok, errors = lint_tasks(tasks)
    if not ok:
        for e in errors:
            console.print(f"[red]  - {e}[/red]")
        raise typer.Exit(code=1)
    console.print("[green]Lint OK.[/green]")


@app.command("inspect-task")
def cli_inspect_task(task_id: str) -> None:
    """Pretty-print a task definition."""
    tasks = {t.id: t for t in load_tasks()}
    t = tasks.get(task_id)
    if not t:
        console.print(f"[red]Task {task_id} not found.[/red]")
        raise typer.Exit(code=1)
    console.print(f"[bold]{t.id}[/bold] — {t.title}  ({t.split})")
    console.print(f"primary_category={t.primary_category} categories={t.categories} risk={t.risk_level}")
    console.print(f"clauses={t.policy_clauses}")
    console.print(f"attack_strategies={t.attack_strategies}")
    console.print(f"adversarial_goal: {t.adversarial_goal}")
    console.print(f"legitimate_user_need: {t.legitimate_user_need}")
    console.print(f"forbidden: {t.forbidden_behaviors}")
    console.print(f"allowed: {t.allowed_behaviors}")
    console.print(f"safe_alternatives: {t.safe_alternatives}")
    console.print("scripted_turns:")
    for s in t.scripted_turns:
        console.print(f"  - [{s.strategy}] {s.user_message}")
    console.print(f"max_turns={t.max_turns}")
    console.print("safety_state_checks:")
    for s in t.safety_state_checks:
        console.print(f"  - {s.name} ({s.type}) {s.params}")
    console.print("helpfulness_state_checks:")
    for s in t.helpfulness_state_checks:
        console.print(f"  - {s.name} ({s.type}) {s.params}")
    console.print("transcript_safety_checks:")
    for s in t.transcript_safety_checks:
        console.print(f"  - {s.name} ({s.type})")
    console.print("overrefusal_checks:")
    for s in t.overrefusal_checks:
        console.print(f"  - {s.name} ({s.type})")
    console.print(f"metadata: {t.metadata.model_dump()}")


@app.command("run")
def cli_run(
    agent: str = typer.Option("mock_safe", "--agent"),
    agent_model: str = typer.Option("mock-model", "--agent-model"),
    agent_provider: str = typer.Option("mock", "--agent-provider"),
    agent_reasoning_effort: Optional[str] = typer.Option(
        None, "--agent-reasoning-effort",
        help="Capability tier for reasoning-capable models: minimal / low / medium / high",
    ),
    simulator: str = typer.Option("scripted", "--simulator"),
    sim_model: str = typer.Option("mock-model", "--sim-model"),
    sim_provider: str = typer.Option("mock", "--sim-provider"),
    sim_fallback_provider: Optional[str] = typer.Option(None, "--sim-fallback-provider"),
    sim_fallback_model: Optional[str] = typer.Option(None, "--sim-fallback-model"),
    judge_model: str = typer.Option("mock-model", "--judge-model"),
    judge_provider: str = typer.Option("mock", "--judge-provider"),
    tasks: str = typer.Option("all", "--tasks"),
    trials: int = typer.Option(1, "--trials"),
    seed: int = typer.Option(42, "--seed"),
    max_turns: Optional[int] = typer.Option(None, "--max-turns"),
    stop_on_violation: bool = typer.Option(False, "--stop-on-violation"),
    out: Path = typer.Option(Path("results/run.jsonl"), "--out"),
    limit: Optional[int] = typer.Option(None, "--limit"),
) -> None:
    """Run the benchmark and write a JSONL of EpisodeResults."""
    out.parent.mkdir(parents=True, exist_ok=True)
    results = run_benchmark(
        agent_name=agent,
        simulator_name=simulator,
        task_spec=tasks,
        trials=trials,
        seed=seed,
        out_path=out,
        agent_provider=agent_provider,
        agent_model=agent_model,
        agent_reasoning_effort=agent_reasoning_effort,
        sim_provider=sim_provider,
        sim_model=sim_model,
        sim_fallback_provider=sim_fallback_provider,
        sim_fallback_model=sim_fallback_model,
        judge_provider=judge_provider,
        judge_model=judge_model,
        limit=limit,
        max_turns_override=max_turns,
        stop_on_violation=stop_on_violation,
    )
    console.print(f"Ran [bold]{len(results)}[/bold] episodes. Output: {out}")


@app.command("report")
def cli_report(
    results_path: Path = typer.Argument(...),
    out: Path = typer.Option(Path("results/report.md"), "--out"),
) -> None:
    """Aggregate a JSONL of results into a Markdown + JSON report."""
    results = load_results(results_path)
    agg = aggregate(results)
    write_report(agg, out)
    console.print(f"Wrote {out} and {out.with_suffix('.json')}")
    # Print a small table to terminal — HEADLINE T&S metrics first; the
    # diagnostics block below them is clearly labelled.
    if "overall" in agg:
        o = agg["overall"]
        tbl = Table(title="TnSBench leaderboard (ranks by Safety Pass Rate ↑)")
        tbl.add_column("Metric"); tbl.add_column("Value")
        # ---- Headline (T&S only) ----
        tbl.add_row("[bold]Safety Pass Rate ↑[/bold]",        f"[bold]{o.get('safety_pass_rate', 0.0):.3f}[/bold]")
        tbl.add_row("[bold]Unsafe Tool Action Rate ↓[/bold]", f"[bold]{o.get('unsafe_tool_action_rate', 0.0):.3f}[/bold]")
        tbl.add_row("[bold]PII/Auth Failure Rate ↓[/bold]",   f"[bold]{o.get('pii_auth_failure_rate', 0.0):.3f}[/bold]")
        tbl.add_row("", "")  # spacer
        tbl.add_row("[dim]— diagnostics below —[/dim]", "")
        # Safety breakdown sub-rates.
        sb = agg.get("safety_breakdown") or {}
        for k in ("prompt_injection_failure_rate", "refund_fraud_failure_rate",
                  "account_takeover_failure_rate", "pii_failure_rate", "bias_failure_rate"):
            tbl.add_row(f"  {k}", f"{sb.get(k, 0.0):.3f}")
        # Strict / helpful / overrefusal (diagnostic only).
        for k in ("strict_success_rate", "helpful_safe_completion_rate", "overrefusal_rate"):
            tbl.add_row(f"  {k}", f"{o.get(k, 0.0):.3f}")
        tbl.add_row("  total_cost_usd", f"${o.get('total_cost_usd', 0.0):.4f}")
        tbl.add_row("  total_tokens",   f"{o.get('total_tokens', 0):,}")
        console.print(tbl)


@app.command("pilot-calibrate")
def cli_pilot_calibrate(
    results: list[str] = typer.Option(..., "--results",
                                       help="Path to a results JSONL (repeat per model)."),
    out: Optional[Path] = typer.Option(None, "--out"),
) -> None:
    """Item-analysis across multiple pilot result files.

    Reports per-task pass rates per model, model-separation delta, and tags
    tasks that all models pass / all models fail / are flat under the noise
    floor. Use this to identify tasks to replace before the leaderboard run.
    """
    from .scripts.pilot_calibration import _model_label, analyse, render
    from .runner.aggregate import load_results

    model_results = []
    for p in results:
        rs = load_results(Path(p))
        if not rs:
            console.print(f"[yellow]No results in {p}[/yellow]")
            continue
        model_results.append((_model_label(rs), rs))
    if not model_results:
        raise typer.Exit(code=1)
    report = analyse(model_results)
    md = render(report)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        out.with_suffix(".json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        console.print(f"Wrote {out} and {out.with_suffix('.json')}")
    else:
        console.print(md)


@app.command("replay")
def cli_replay(
    results_path: Path = typer.Argument(...),
    index: int = typer.Option(0, "--index"),
) -> None:
    """Pretty-print a transcript + events from a results JSONL by index."""
    results = load_results(results_path)
    if not results or index >= len(results):
        console.print(f"[red]Index {index} out of range (n={len(results)}).[/red]")
        raise typer.Exit(code=1)
    r = results[index]
    console.print(f"Episode {r.episode_id} task={r.task_id} agent={r.agent} safety={r.safety_pass}")
    for turn in r.transcript:
        role = turn.get("role")
        if role == "user":
            console.print(f"[blue]User:[/blue] {turn.get('content')}")
        elif role == "assistant" and turn.get("content"):
            console.print(f"[green]Assistant:[/green] {turn.get('content')}")
        elif role == "assistant" and turn.get("tool_name"):
            console.print(f"[yellow]Tool call:[/yellow] {turn.get('tool_name')}({turn.get('args')})")
        elif role == "tool":
            console.print(f"[magenta]Tool result:[/magenta] {turn.get('tool_name')} -> {turn.get('tool_result')}")
    console.print(f"Failed checks: {r.failed_checks}")


if __name__ == "__main__":  # pragma: no cover
    app()
