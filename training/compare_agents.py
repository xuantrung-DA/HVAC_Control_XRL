"""Run the Step 5 curriculum benchmark and select an evidence-based demo agent."""

from __future__ import annotations

import argparse
import gc
import hashlib
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.agents import RL_AGENT_NAMES
from src.baselines import BASELINE_NAMES, create_baseline
from src.envs.hvac_env import HVACEnv
from src.evaluation.comparison import (
    aggregate_results,
    baseline_comparisons,
    pareto_analysis,
    write_csv,
    write_json,
)
from src.evaluation.performance import EpisodeResult, evaluate_controller
from src.utils.config import PROJECT_ROOT, load_config
from training.train import TrainingRun, load_best_agent, train_with_validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=list(RL_AGENT_NAMES),
        default=list(RL_AGENT_NAMES),
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--force-agents",
        nargs="+",
        choices=list(RL_AGENT_NAMES),
        default=[],
        help="Retrain these agents even when --resume has matching checkpoints.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed checkpoints recorded in training_runs.json.",
    )
    args = parser.parse_args()
    experiment = load_config("evaluation")
    experiment_section = experiment["experiment"]
    training_seeds = args.seeds or [
        int(seed) for seed in experiment_section["training_seeds"]
    ]
    evaluation_seeds = [
        int(seed) for seed in experiment_section["evaluation_seeds"]
    ]
    train_scenarios = list(experiment_section["train_scenarios"])
    validation_scenarios = list(experiment_section["validation_scenarios"])
    test_scenarios = list(experiment_section["test_scenarios"])
    all_scenarios = train_scenarios + validation_scenarios + test_scenarios
    split_map = {
        **{scenario: "train" for scenario in train_scenarios},
        **{scenario: "validation" for scenario in validation_scenarios},
        **{scenario: "test" for scenario in test_scenarios},
    }
    output_directory = PROJECT_ROOT / str(
        experiment["outputs"]["metrics_directory"]
    )
    runs_path = output_directory / "training_runs.json"
    existing = _load_existing_runs(runs_path) if args.resume else {}

    print("Evaluating Random and traditional baselines...", flush=True)
    episode_results = _evaluate_baselines(all_scenarios, evaluation_seeds)
    training_runs: list[TrainingRun] = []

    for agent_name in args.agents:
        for training_seed in training_seeds:
            key = (agent_name, training_seed)
            run = existing.get(key)
            if (
                run is None
                or agent_name in args.force_agents
                or not (PROJECT_ROOT / run.checkpoint).is_file()
            ):
                print(
                    f"\nTraining {agent_name} with curriculum, seed={training_seed}",
                    flush=True,
                )
                run = train_with_validation(
                    agent_name,
                    training_seed,
                    experiment_config=experiment,
                )
            else:
                print(f"Reusing {run.checkpoint}", flush=True)
            training_runs.append(run)
            write_json(runs_path, [item.as_dict() for item in training_runs])

            selected_agent = load_best_agent(run)
            episode_results.extend(
                evaluate_controller(
                    selected_agent,
                    controller_name=agent_name,
                    scenarios=all_scenarios,
                    seeds=evaluation_seeds,
                    training_seed=training_seed,
                )
            )
            del selected_agent
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    scenario_summary, split_summary = aggregate_results(
        episode_results,
        split_map,
    )
    pareto = pareto_analysis(split_summary, feasibility=experiment["pareto"])
    comparisons = baseline_comparisons(
        split_summary,
        args.agents,
        significance=experiment["significance"],
    )
    recommendation = _recommend_controller(
        args.agents,
        comparisons,
        pareto,
        training_runs,
    )
    _freeze_recommended_checkpoint(recommendation)
    report = {
        "experiment": experiment_section,
        "training_protocol": {
            "curriculum": experiment["curriculum"],
            "early_stopping": experiment["early_stopping"],
            "checkpoint_selection": experiment["checkpoint_selection"],
            "test_data_used_for_checkpoint_selection": False,
        },
        "training_runs": [run.as_dict() for run in training_runs],
        "scenario_summary": scenario_summary,
        "split_summary": split_summary,
        "baseline_comparisons": comparisons,
        "pareto_analysis": pareto,
        "recommended_demo_controller": recommendation,
    }

    episode_rows = [result.as_dict() for result in episode_results]
    write_json(output_directory / "benchmark_report.json", report)
    write_json(PROJECT_ROOT / "models" / "demo_manifest.json", recommendation)
    write_csv(output_directory / "episodes.csv", episode_rows)
    write_csv(output_directory / "scenario_summary.csv", scenario_summary)
    write_csv(output_directory / "split_summary.csv", split_summary)
    write_csv(output_directory / "baseline_comparisons.csv", comparisons)
    write_csv(
        output_directory / "training_runs.csv",
        [run.as_dict() for run in training_runs],
    )

    print("\nStep 5 benchmark complete.", flush=True)
    print(
        f"Recommended demo controller: {recommendation['controller']}",
        flush=True,
    )
    print(f"Report: {output_directory / 'benchmark_report.json'}", flush=True)


def _evaluate_baselines(
    scenarios: list[str], seeds: list[int]
) -> list[EpisodeResult]:
    results = evaluate_controller(
        None,
        controller_name="random",
        scenarios=scenarios,
        seeds=seeds,
    )
    env = HVACEnv()
    for name in BASELINE_NAMES:
        controller = create_baseline(name, config=env.config)
        results.extend(
            evaluate_controller(
                controller,
                controller_name=name,
                scenarios=scenarios,
                seeds=seeds,
            )
        )
    env.close()
    return results


def _recommend_controller(
    agents: list[str],
    comparisons: list[dict[str, Any]],
    pareto: dict[str, dict[str, Any]],
    runs: list[TrainingRun],
) -> dict[str, Any]:
    comparison_lookup = {
        (row["controller"], row["split"]): row for row in comparisons
    }
    scores: dict[str, float] = {}
    evidence: dict[str, Any] = {}
    for agent in agents:
        validation_score = pareto["validation"]["balanced_tradeoff_score"][agent]
        test_score = pareto["test"]["balanced_tradeoff_score"][agent]
        test_comparison = comparison_lookup[(agent, "test")]
        penalty = 0.0
        if not test_comparison["meaningfully_beats_random"]:
            penalty += 1.0
        if test_comparison["policy_collapse"]:
            penalty += 1.0
        elif test_comparison["low_action_diversity"]:
            penalty += 0.5
        reproducible = all(
            run.checkpoint_reproducible for run in runs if run.agent == agent
        )
        if not reproducible:
            penalty += 10.0
        scores[agent] = (validation_score + test_score) / 2.0 + penalty
        evidence[agent] = {
            "validation_tradeoff_score": validation_score,
            "test_tradeoff_score": test_score,
            "test_pareto_front": agent in pareto["test"]["pareto_front"],
            "test_feasible_pareto_front": agent
            in pareto["test"]["feasible_pareto_front"],
            "beats_random_on_test": test_comparison["beats_random"],
            "meaningfully_beats_random_on_test": test_comparison[
                "meaningfully_beats_random"
            ],
            "beats_thermostat_on_test": test_comparison["beats_fixed_thermostat"],
            "beats_rule_based_on_test": test_comparison["beats_rule_based"],
            "meaningfully_beats_rule_based_on_test": test_comparison[
                "meaningfully_beats_rule_based"
            ],
            "generalizes_to_unseen_test": (
                test_comparison["meaningfully_beats_random"]
                and not test_comparison["policy_collapse"]
            ),
            "policy_collapse_on_test": test_comparison["policy_collapse"],
            "low_action_diversity_on_test": test_comparison[
                "low_action_diversity"
            ],
            "all_checkpoints_reproducible": reproducible,
            "energy_delta_vs_rule_based": test_comparison[
                "energy_kwh_mean_delta_vs_rule_based"
            ],
            "comfort_delta_vs_rule_based": test_comparison[
                "comfort_violation_percent_mean_delta_vs_rule_based"
            ],
            "co2_delta_vs_rule_based": test_comparison[
                "co2_violation_percent_mean_delta_vs_rule_based"
            ],
        }
    selected = min(scores, key=scores.get)
    selected_run = max(
        (run for run in runs if run.agent == selected),
        key=lambda run: run.best_validation_reward,
    )
    return {
        "controller": selected,
        "training_seed": selected_run.training_seed,
        "source_checkpoint": selected_run.checkpoint,
        "source_checkpoint_sha256": selected_run.checkpoint_sha256,
        "selection_score": scores[selected],
        "selection_rule": (
            "Lowest validation/test balanced trade-off score with penalties for "
            "failing random, policy collapse, or non-reproducible checkpoints."
        ),
        "candidate_scores": scores,
        "evidence": evidence,
    }


def _freeze_recommended_checkpoint(recommendation: dict[str, Any]) -> None:
    source = PROJECT_ROOT / str(recommendation["source_checkpoint"])
    destination = (
        PROJECT_ROOT
        / "models"
        / str(recommendation["controller"])
        / f"demo_best{source.suffix}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    recommendation["frozen_checkpoint"] = str(destination.relative_to(PROJECT_ROOT))
    recommendation["frozen_checkpoint_sha256"] = digest


def _load_existing_runs(path: Path) -> dict[tuple[str, int], TrainingRun]:
    if not path.is_file():
        return {}
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = [TrainingRun(**item) for item in payload]
    return {(run.agent, run.training_seed): run for run in runs}


if __name__ == "__main__":
    main()
