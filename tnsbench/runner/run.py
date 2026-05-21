"""Top-level multi-episode runner."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

from ..agents.base import BaseAgent
from ..agents.llm_agent import LLMAgent
from ..agents.mock_agents import AGENT_REGISTRY
from ..agents.providers import ProviderConfig
from ..core.logging import get_logger
from ..grading.grader import Grader
from ..grading.llm_judge import LLMJudge
from ..policy.policy_loader import load_policy_text
from ..simulators.base import BaseSimulator
from ..simulators.cascading import CascadingSimulator
from ..simulators.llm_simulator import HybridSimulator, LLMSimulator
from ..simulators.scripted import ScriptedSimulator
from ..tasks.schema import Task
from ..tasks.task_loader import load_tasks, load_benign_tasks
from .episode import run_episode
from .result import EpisodeResult

LOG = get_logger("tnsbench.runner")


def build_agent(name: str, *, provider: str = "mock", model: str = "mock-model",
                reasoning_effort: Optional[str] = None) -> BaseAgent:
    if name in AGENT_REGISTRY:
        return AGENT_REGISTRY[name]()
    if name == "llm_agent":
        return LLMAgent(ProviderConfig(provider=provider, model=model,
                                        reasoning_effort=reasoning_effort))
    raise ValueError(f"Unknown agent: {name}")


def build_simulator(
    name: str,
    *,
    provider: str = "mock",
    model: str = "mock-model",
    fallback_provider: Optional[str] = None,
    fallback_model: Optional[str] = None,
) -> BaseSimulator:
    if name == "scripted":
        return ScriptedSimulator()
    if name == "llm_simulator":
        return LLMSimulator(ProviderConfig(provider=provider, model=model))
    if name == "hybrid":
        return HybridSimulator(ProviderConfig(provider=provider, model=model))
    if name == "cascading":
        if not fallback_provider or not fallback_model:
            raise ValueError(
                "cascading simulator requires --sim-fallback-provider and --sim-fallback-model"
            )
        return CascadingSimulator(
            primary_config=ProviderConfig(provider=provider, model=model),
            fallback_config=ProviderConfig(provider=fallback_provider, model=fallback_model),
        )
    raise ValueError(f"Unknown simulator: {name}")


def filter_tasks(tasks: List[Task], spec: str, limit: Optional[int] = None) -> List[Task]:
    # v0.7: `all` and `adversarial` are leaderboard slices — adversarial only,
    # benign control set is NEVER mixed in. To run the over-refusal calibration
    # set, use the explicit `overrefusal_calibration` (or `calibration`) spec.
    if not spec or spec in ("all", "adversarial"):
        out = [t for t in tasks if t.split == "adversarial"]
    elif spec in ("overrefusal_calibration", "calibration", "benign", "benign_control"):
        out = [t for t in tasks if t.split == "benign_control"]
    elif spec.startswith("category:"):
        # Accept comma-separated categories: "category:fraud,account_takeover,mixed".
        cats = {c.strip() for c in spec.split(":", 1)[1].split(",") if c.strip()}
        out = [t for t in tasks
                if (t.primary_category in cats) or (cats & set(t.categories))]
    elif spec.startswith("strategy:"):
        s = spec.split(":", 1)[1]
        out = [t for t in tasks if s in t.attack_strategies]
    elif "," in spec:
        ids = {s.strip() for s in spec.split(",") if s.strip()}
        out = [t for t in tasks if t.id in ids]
    else:
        out = [t for t in tasks if t.id == spec]
    if limit:
        out = out[:limit]
    return out


def run_benchmark(
    agent_name: str,
    simulator_name: str,
    task_spec: str = "all",
    *,
    trials: int = 1,
    seed: int = 42,
    out_path: Optional[Path] = None,
    agent_provider: str = "mock",
    agent_model: str = "mock-model",
    agent_reasoning_effort: Optional[str] = None,
    sim_provider: str = "mock",
    sim_model: str = "mock-model",
    sim_fallback_provider: Optional[str] = None,
    sim_fallback_model: Optional[str] = None,
    judge_provider: str = "mock",
    judge_model: str = "mock-model",
    limit: Optional[int] = None,
    max_turns_override: Optional[int] = None,
    stop_on_violation: bool = False,
) -> List[EpisodeResult]:
    # v0.7: the main leaderboard (--tasks all / --tasks adversarial) is
    # *adversarial only*. Benign controls are reachable only via the
    # explicit `overrefusal_calibration` / `calibration` slice.
    if task_spec in ("overrefusal_calibration", "calibration",
                     "benign", "benign_control"):
        all_tasks = load_benign_tasks()
    else:
        all_tasks = load_tasks()
    tasks = filter_tasks(all_tasks, task_spec, limit)
    if not tasks:
        LOG.warning("No tasks matched spec '%s'", task_spec)
        return []

    judge = LLMJudge(ProviderConfig(provider=judge_provider, model=judge_model))
    grader = Grader(judge=judge, policy_excerpt=load_policy_text())

    results: List[EpisodeResult] = []
    out_file = open(out_path, "w", encoding="utf-8") if out_path else None
    try:
        for task in tasks:
            for trial in range(trials):
                agent = build_agent(agent_name, provider=agent_provider, model=agent_model,
                                    reasoning_effort=agent_reasoning_effort)
                simulator = build_simulator(
                    simulator_name,
                    provider=sim_provider,
                    model=sim_model,
                    fallback_provider=sim_fallback_provider,
                    fallback_model=sim_fallback_model,
                )
                er = run_episode(
                    task, agent, simulator,
                    trial_index=trial, seed=seed,
                    stop_on_violation=stop_on_violation,
                    max_turns_override=max_turns_override,
                    grader=grader,
                )
                results.append(er)
                if out_file:
                    out_file.write(er.model_dump_json() + "\n")
                    out_file.flush()
    finally:
        if out_file:
            out_file.close()
    return results
