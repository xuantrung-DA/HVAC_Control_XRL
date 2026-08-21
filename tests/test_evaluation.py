"""Tests for robust evaluation, aggregation, and training evidence."""

from __future__ import annotations

from pathlib import Path

from src.baselines import create_baseline
from src.envs.hvac_env import HVACEnv
from src.evaluation.comparison import aggregate_results, pareto_analysis
from src.evaluation.performance import evaluate_controller, run_episode
from src.utils.config import deep_merge, load_config
from training.train import train_with_validation


def test_episode_metrics_cover_all_objectives() -> None:
    env = HVACEnv()
    controller = create_baseline("rule_based", config=env.config)
    result = run_episode(
        controller,
        controller_name="rule_based",
        scenario="normal",
        evaluation_seed=42,
    )

    assert result.energy_kwh > 0
    assert result.electricity_cost > 0
    assert 0 <= result.comfort_violation_percent <= 100
    assert 0 <= result.co2_violation_percent <= 100
    assert result.hvac_switches >= 0
    assert abs(
        result.action_0_fraction
        + result.action_1_fraction
        + result.action_2_fraction
        + result.action_3_fraction
        - 1.0
    ) < 1e-9
    env.close()


def test_seeded_random_evaluation_is_reproducible() -> None:
    first = run_episode(
        None,
        controller_name="random",
        scenario="hot_day",
        evaluation_seed=77,
    )
    second = run_episode(
        None,
        controller_name="random",
        scenario="hot_day",
        evaluation_seed=77,
    )
    assert first == second


def test_aggregation_reports_mean_std_and_pareto_front() -> None:
    env = HVACEnv()
    thermostat = create_baseline("fixed_thermostat", config=env.config)
    rule = create_baseline("rule_based", config=env.config)
    results = evaluate_controller(
        thermostat,
        controller_name="fixed_thermostat",
        scenarios=["normal"],
        seeds=[1, 2],
    ) + evaluate_controller(
        rule,
        controller_name="rule_based",
        scenarios=["normal"],
        seeds=[1, 2],
    )
    scenario_summary, split_summary = aggregate_results(
        results, {"normal": "train"}
    )
    pareto = pareto_analysis(split_summary)

    assert len(scenario_summary) == 2
    assert all(row["samples"] == 2 for row in split_summary)
    assert pareto["train"]["pareto_front"]
    assert set(pareto["train"]["balanced_tradeoff_score"]) == {
        "fixed_thermostat",
        "rule_based",
    }
    env.close()


def test_tiny_q_learning_training_selects_reproducible_checkpoint(
    tmp_path: Path,
) -> None:
    experiment = deep_merge(
        load_config("evaluation"),
        {
            "experiment": {
                "train_scenarios": ["normal"],
                "validation_scenarios": ["normal"],
                "validation_seeds": [81],
            },
            "curriculum": {
                "normal_only_episodes": 0,
                "expansion_episodes": 0,
            },
            "budgets": {
                "q_learning": {
                    "total_episodes": 2,
                    "chunk_episodes": 1,
                }
            },
            "early_stopping": {"enabled": False},
            "outputs": {
                "metrics_directory": str(tmp_path / "metrics"),
                "curves_directory": str(tmp_path / "curves"),
                "checkpoint_directory": str(tmp_path / "models"),
            },
        },
    )
    run = train_with_validation(
        "q_learning",
        training_seed=17,
        experiment_config=experiment,
    )

    assert run.checkpoint_reproducible
    assert run.selected_at_budget in {1, 2}
    assert Path(run.checkpoint).suffix == ".npz"
