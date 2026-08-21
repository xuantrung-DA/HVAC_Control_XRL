"""Multi-scenario V2 DQN training with constraint-first checkpoint selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import psutil
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.dqn import DQNAgent  # noqa: E402
from src.envs.v2 import V2HVACEnv, V2ScenarioSamplerEnv  # noqa: E402
from src.evaluation import aggregate_v2_results, evaluate_v2_controller  # noqa: E402
from src.utils.config import deep_merge, load_agent_config, load_yaml  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def agent_config(experiment: dict, seed: int) -> dict:
    settings = experiment["dqn_overrides"]
    return deep_merge(
        deepcopy(load_agent_config("dqn")),
        {
            "agent": {"seed": seed, "device": "auto"},
            "model": {"hidden_sizes": settings["model_hidden_sizes"]},
            "replay_buffer": {
                "capacity": settings["replay_capacity"],
                "batch_size": settings["batch_size"],
                "warmup_steps": settings["warmup_steps"],
            },
            "optimization": {
                "learning_rate": settings["learning_rate"],
                "reward_scale": settings["reward_scale"],
                "target_update_frequency": settings["target_update_frequency"],
            },
            "exploration": {"epsilon_decay_steps": settings["epsilon_decay_steps"]},
        },
    )


def selection_score(summary: dict, selection: dict) -> tuple:
    metrics = summary["metrics"]
    comfort = metrics["comfort_violation_percent"]["mean"]
    co2 = metrics["co2_violation_percent"]["mean"]
    critical = metrics["critical_safety_violations"]["mean"]
    comfort_excess = max(
        comfort - float(selection["comfort_violation_percent_max_exclusive"]), 0.0
    )
    co2_excess = max(
        co2 - float(selection["co2_violation_percent_max_exclusive"]), 0.0
    )
    constraints_pass = (
        comfort < float(selection["comfort_violation_percent_max_exclusive"])
        and co2 < float(selection["co2_violation_percent_max_exclusive"])
        and critical <= int(selection["critical_safety_violations_max"])
    )
    return (
        0 if constraints_pass else 1,
        critical,
        comfort_excess + 2.0 * co2_excess,
        metrics["whole_building_kwh"]["mean"],
        metrics["electricity_cost"]["mean"],
        -metrics["reward"]["mean"],
    )


def evaluate(agent, config: dict, *, shield_enabled: bool, smoke: bool):
    experiment = config["experiment"]
    seeds = experiment["validation_seeds"][:1] if smoke else experiment["validation_seeds"]
    results = evaluate_v2_controller(
        agent,
        controller_name="v2_dqn",
        scenarios=experiment["validation_scenarios"],
        seeds=seeds,
        shield_enabled=shield_enabled,
        reward_mode=experiment["reward_mode"],
    )
    return aggregate_v2_results(results), results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    config_path = PROJECT_ROOT / "configs/v2/training.yaml"
    config = load_yaml(config_path)
    experiment = config["experiment"]
    if args.seed not in experiment["training_seeds"]:
        raise ValueError("Seed is outside the locked V2 training seed list")
    scenarios_config = load_yaml(PROJECT_ROOT / "configs/v2/scenarios.yaml")
    held_out = {
        item
        for split, names in scenarios_config["splits"].items()
        if split.startswith("held_out")
        for item in names
    }
    used = set(experiment["training_scenarios"]) | set(experiment["validation_scenarios"])
    if used & held_out:
        raise RuntimeError("Held-out leakage in V2 DQN development pipeline")
    smoke = args.mode == "smoke"
    total_steps = int(
        config["budgets"]["smoke_total_steps" if smoke else "full_total_steps"]
    )
    chunk_steps = int(config["budgets"]["chunk_steps"])
    env = V2ScenarioSamplerEnv(
        experiment["training_scenarios"],
        seed=args.seed,
        normal_only_episodes=int(config["curriculum"]["normal_only_episodes"]),
        shield_enabled=bool(experiment["train_with_shield"]),
        reward_mode=experiment["reward_mode"],
    )
    configured_agent = agent_config(config, args.seed)
    agent = DQNAgent(env.observation_space, env.action_space, config=configured_agent)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    rss_before = process.memory_info().rss
    curves = []
    best_score = None
    best_summary = None
    best_step = 0
    checkpoint = (
        PROJECT_ROOT
        / "models/v2/dqn"
        / experiment["id"]
        / f"seed_{args.seed}_{args.mode}_best.pt"
    )
    started = time.perf_counter()
    consumed = 0
    while consumed < total_steps:
        requested = min(chunk_steps, total_steps - consumed)
        training = agent.learn(env, total_steps=requested, seed=args.seed + consumed)
        consumed += requested
        validation, _ = evaluate(agent, config, shield_enabled=False, smoke=smoke)
        score = selection_score(validation, config["selection"])
        selected = best_score is None or score < best_score
        if selected:
            best_score = score
            best_summary = validation
            best_step = consumed
            agent.save(checkpoint)
        curve = {
            "step": consumed,
            "selected": selected,
            "epsilon": agent.epsilon,
            "training_reward": training.mean_episode_reward,
            "training_loss": training.mean_loss,
            "steps_per_second": training.steps_per_second,
            "validation_reward": validation["metrics"]["reward"]["mean"],
            "validation_energy_kwh": validation["metrics"]["whole_building_kwh"]["mean"],
            "validation_comfort_percent": validation["metrics"]["comfort_violation_percent"]["mean"],
            "validation_co2_percent": validation["metrics"]["co2_violation_percent"]["mean"],
            "validation_constraint_pass": score[0] == 0,
        }
        curves.append(curve)
        print(json.dumps(curve), flush=True)
    duration = time.perf_counter() - started
    reloaded = DQNAgent(env.observation_space, env.action_space, config=configured_agent)
    reloaded.load(checkpoint)
    without_shield, raw_results = evaluate(
        reloaded, config, shield_enabled=False, smoke=smoke
    )
    with_shield, shield_results = evaluate(
        reloaded, config, shield_enabled=True, smoke=smoke
    )
    first_env = V2HVACEnv(
        experiment["validation_scenarios"][0], shield_enabled=False
    )
    observation, _ = first_env.reset(seed=experiment["validation_seeds"][0])
    actions_first = [reloaded.predict(observation, deterministic=True) for _ in range(5)]
    second = DQNAgent(env.observation_space, env.action_space, config=configured_agent)
    second.load(checkpoint)
    actions_second = [second.predict(observation, deterministic=True) for _ in range(5)]
    reproducible = actions_first == actions_second
    rss_after = process.memory_info().rss
    output = PROJECT_ROOT / "outputs/v2/training" / experiment["id"] / args.mode
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "experiment_id": experiment["id"],
        "mode": args.mode,
        "seed": args.seed,
        "held_out_used": False,
        "training_scenarios": experiment["training_scenarios"],
        "validation_scenarios": experiment["validation_scenarios"],
        "steps": total_steps,
        "best_step": best_step,
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_reproducible": reproducible,
        "parameters": agent.parameter_count(),
        "device": str(agent.device),
        "duration_seconds": duration,
        "mean_training_steps_per_second": total_steps / duration,
        "rss_delta_mb": (rss_after - rss_before) / (1024 * 1024),
        "cuda_peak_allocated_mb": (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available()
            else None
        ),
        "best_validation_during_training": best_summary,
        "validation_without_shield": without_shield,
        "validation_with_shield": with_shield,
        "curves": curves,
        "raw_episode_count_without_shield": len(raw_results),
        "raw_episode_count_with_shield": len(shield_results),
    }
    report_path = output / f"dqn_seed_{args.seed}_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / f"dqn_seed_{args.seed}_curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    print(json.dumps({key: report[key] for key in (
        "seed", "steps", "best_step", "checkpoint_reproducible", "parameters",
        "device", "duration_seconds", "mean_training_steps_per_second",
        "rss_delta_mb", "cuda_peak_allocated_mb",
    )}, indent=2))


if __name__ == "__main__":
    main()
