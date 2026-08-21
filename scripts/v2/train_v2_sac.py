"""Train evidence-justified SAC with decoupled cooling and ventilation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import psutil
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents import SACV2Agent  # noqa: E402
from src.envs.v2 import V2ContinuousHVACEnv, V2ContinuousScenarioSamplerEnv  # noqa: E402
from src.evaluation import aggregate_continuous_results, evaluate_continuous_controller  # noqa: E402
from src.utils.config import deep_merge, load_yaml  # noqa: E402


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_score(summary: dict) -> tuple:
    metrics = summary["metrics"]
    comfort = metrics["comfort_violation_percent"]["mean"]
    co2 = metrics["co2_violation_percent"]["mean"]
    critical = metrics["critical_safety_violations"]["mean"]
    passed = comfort < 5.0 and co2 < 1.0 and critical == 0.0
    return (
        0 if passed else 1,
        critical,
        max(comfort - 5.0, 0.0) + 2.0 * max(co2 - 1.0, 0.0),
        metrics["whole_building_kwh"]["mean"],
        metrics["electricity_cost"]["mean"],
        -metrics["reward"]["mean"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    training = load_yaml(PROJECT_ROOT / "configs/v2/training.yaml")
    experiment = training["continuous_experiment"]
    sac_config = load_yaml(PROJECT_ROOT / "configs/v2/sac.yaml")
    if args.seed not in experiment["training_seeds"]:
        raise ValueError("Seed is outside locked SAC training seeds")
    scenario_config = load_yaml(PROJECT_ROOT / "configs/v2/scenarios.yaml")
    held_out = {
        scenario
        for split, scenarios in scenario_config["splits"].items()
        if split.startswith("held_out")
        for scenario in scenarios
    }
    if (set(experiment["training_scenarios"]) | set(experiment["validation_scenarios"])) & held_out:
        raise RuntimeError("Held-out leakage in SAC development")
    smoke = args.mode == "smoke"
    total_steps = int(
        sac_config["training"]["smoke_total_steps" if smoke else "full_total_steps"]
    )
    chunk_steps = int(sac_config["training"]["chunk_steps"])
    config = deep_merge(deepcopy(sac_config), {"agent": {"seed": args.seed}})
    env = V2ContinuousScenarioSamplerEnv(
        experiment["training_scenarios"], seed=args.seed, normal_only_episodes=10
    )
    agent = SACV2Agent(env, config)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    rss_before = process.memory_info().rss
    checkpoint = (
        PROJECT_ROOT / "models/v2/sac" / experiment["id"] / f"seed_{args.seed}_{args.mode}_best.zip"
    )
    best_score = None
    best_step = 0
    curves = []
    stopped_early = False
    consumed = 0
    started = time.perf_counter()
    validation_seeds = experiment["validation_seeds"][:1] if smoke else experiment["validation_seeds"]
    while consumed < total_steps:
        chunk = min(chunk_steps, total_steps - consumed)
        summary = agent.learn(chunk)
        consumed += summary.total_steps
        results = evaluate_continuous_controller(
            agent,
            controller_name="sac_v2",
            scenarios=experiment["validation_scenarios"],
            seeds=validation_seeds,
        )
        validation = aggregate_continuous_results(results)
        score = selection_score(validation)
        selected = best_score is None or score < best_score
        if selected:
            best_score = score
            best_step = consumed
            agent.save(checkpoint)
        curve = {
            "step": consumed,
            "selected": selected,
            "training_reward": summary.mean_episode_reward,
            "steps_per_second": summary.steps_per_second,
            "validation_reward": validation["metrics"]["reward"]["mean"],
            "validation_energy_kwh": validation["metrics"]["whole_building_kwh"]["mean"],
            "validation_comfort_percent": validation["metrics"]["comfort_violation_percent"]["mean"],
            "validation_co2_percent": validation["metrics"]["co2_violation_percent"]["mean"],
            "validation_cooling_mean": validation["metrics"]["cooling_fraction_mean"]["mean"],
            "validation_ventilation_mean": validation["metrics"]["ventilation_fraction_mean"]["mean"],
            "constraint_pass": score[0] == 0,
        }
        curves.append(curve)
        print(json.dumps(curve), flush=True)
        if (
            not smoke
            and len(curves) >= int(sac_config["training"]["go_no_go_evaluations"])
            and curve["validation_comfort_percent"]
            > float(sac_config["training"]["stop_if_comfort_above_percent"])
        ):
            stopped_early = True
            break
    duration = time.perf_counter() - started
    reload_env = V2ContinuousScenarioSamplerEnv(
        experiment["training_scenarios"], seed=args.seed, normal_only_episodes=10
    )
    reloaded = SACV2Agent(reload_env, config)
    reloaded.load(checkpoint)
    final_results = evaluate_continuous_controller(
        reloaded,
        controller_name="sac_v2",
        scenarios=experiment["validation_scenarios"],
        seeds=validation_seeds,
    )
    final = aggregate_continuous_results(final_results)
    check_env = V2ContinuousHVACEnv(experiment["validation_scenarios"][0])
    observation, _ = check_env.reset(seed=validation_seeds[0])
    first_action = reloaded.predict(observation, deterministic=True)
    second_agent = SACV2Agent(reload_env, config)
    second_agent.load(checkpoint)
    second_action = second_agent.predict(observation, deterministic=True)
    reproducible = bool(np.allclose(first_action, second_action, atol=1e-7))
    resolved_checkpoint = checkpoint
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "experiment_id": experiment["id"],
        "mode": args.mode,
        "seed": args.seed,
        "held_out_used": False,
        "steps_requested": total_steps,
        "steps_used": consumed,
        "stopped_early": stopped_early,
        "best_step": best_step,
        "checkpoint": str(resolved_checkpoint.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": file_hash(resolved_checkpoint),
        "checkpoint_reproducible": reproducible,
        "parameters": agent.parameter_count(),
        "device": str(agent.model.device),
        "duration_seconds": duration,
        "steps_per_second_including_validation": consumed / duration,
        "rss_delta_mb": (process.memory_info().rss - rss_before) / (1024 * 1024),
        "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else None,
        "validation": final,
        "curves": curves,
    }
    output = PROJECT_ROOT / "outputs/v2/training" / experiment["id"] / args.mode
    output.mkdir(parents=True, exist_ok=True)
    (output / f"sac_seed_{args.seed}_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in (
        "seed", "steps_requested", "steps_used", "stopped_early", "best_step", "checkpoint_reproducible", "parameters",
        "device", "duration_seconds", "steps_per_second_including_validation",
        "rss_delta_mb", "cuda_peak_allocated_mb",
    )}, indent=2))


if __name__ == "__main__":
    main()
