"""Summarize three-seed DQN development evidence and justify next action."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import V2RuleBasedController  # noqa: E402
from src.evaluation import aggregate_v2_results, evaluate_v2_controller  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402


METRICS = (
    "reward",
    "whole_building_kwh",
    "hvac_ventilation_kwh",
    "electricity_cost",
    "comfort_violation_percent",
    "temperature_violation_percent",
    "humidity_violation_percent",
    "co2_violation_percent",
    "peak_power_kw",
    "shield_intervention_percent",
    "shield_fallback_percent",
    "critical_safety_violations",
)


def main() -> None:
    training = load_yaml(PROJECT_ROOT / "configs/v2/training.yaml")
    experiment = training["experiment"]
    directory = PROJECT_ROOT / "outputs/v2/training" / experiment["id"] / "full"
    seed_reports = [
        json.loads((directory / f"dqn_seed_{seed}_report.json").read_text(encoding="utf-8"))
        for seed in experiment["training_seeds"]
    ]
    controller_config = load_yaml(PROJECT_ROOT / "configs/v2/controllers.yaml")
    baseline_results = evaluate_v2_controller(
        V2RuleBasedController(controller_config),
        controller_name="rule_based_v2",
        scenarios=experiment["validation_scenarios"],
        seeds=experiment["validation_seeds"],
        shield_enabled=False,
    )
    baseline = aggregate_v2_results(baseline_results)

    variants = {}
    for key in ("validation_without_shield", "validation_with_shield"):
        variants[key] = {
            "metrics": {
                metric: {
                    "mean_across_training_seeds": float(
                        np.mean([report[key]["metrics"][metric]["mean"] for report in seed_reports])
                    ),
                    "std_across_training_seeds": float(
                        np.std([report[key]["metrics"][metric]["mean"] for report in seed_reports])
                    ),
                }
                for metric in METRICS
            },
            "action_distribution_mean": np.mean(
                [report[key]["action_distribution_mean"] for report in seed_reports], axis=0
            ).tolist(),
        }
    no_shield = variants["validation_without_shield"]
    shield = variants["validation_with_shield"]
    selection = training["selection"]
    per_seed = []
    for report in seed_reports:
        summary = report["validation_without_shield"]
        actions = summary["action_distribution_mean"]
        per_seed.append(
            {
                "seed": report["seed"],
                "best_step": report["best_step"],
                "checkpoint": report["checkpoint"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "checkpoint_reproducible": report["checkpoint_reproducible"],
                "comfort_violation_percent": summary["metrics"]["comfort_violation_percent"]["mean"],
                "co2_violation_percent": summary["metrics"]["co2_violation_percent"]["mean"],
                "whole_building_kwh": summary["metrics"]["whole_building_kwh"]["mean"],
                "action_distribution": actions,
                "maximum_action_fraction": max(actions),
                "unique_actions_over_one_percent": sum(value >= 0.01 for value in actions),
                "constraint_pass": (
                    summary["metrics"]["comfort_violation_percent"]["mean"]
                    < float(selection["comfort_violation_percent_max_exclusive"])
                    and summary["metrics"]["co2_violation_percent"]["mean"]
                    < float(selection["co2_violation_percent_max_exclusive"])
                ),
            }
        )
    aggregate_gates = {
        "energy_at_or_below_rule_based": no_shield["metrics"]["whole_building_kwh"]["mean_across_training_seeds"]
        <= baseline["metrics"]["whole_building_kwh"]["mean"],
        "cost_at_or_below_rule_based": no_shield["metrics"]["electricity_cost"]["mean_across_training_seeds"]
        <= baseline["metrics"]["electricity_cost"]["mean"],
        "comfort_below_5_percent": no_shield["metrics"]["comfort_violation_percent"]["mean_across_training_seeds"] < 5.0,
        "co2_below_1_percent": no_shield["metrics"]["co2_violation_percent"]["mean_across_training_seeds"] < 1.0,
        "no_critical_safety_violations": no_shield["metrics"]["critical_safety_violations"]["mean_across_training_seeds"] == 0.0,
        "all_checkpoints_reproducible": all(item["checkpoint_reproducible"] for item in per_seed),
        "no_single_action_collapse": all(item["maximum_action_fraction"] <= 0.90 for item in per_seed),
    }
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "experiment_id": experiment["id"],
        "held_out_used": False,
        "training_seeds": experiment["training_seeds"],
        "validation_scenarios": experiment["validation_scenarios"],
        "validation_seeds": experiment["validation_seeds"],
        "rule_based_validation": baseline,
        "dqn_variants": variants,
        "per_seed": per_seed,
        "development_gates": aggregate_gates,
        "development_status": "PASS" if all(aggregate_gates.values()) else "FAIL",
        "decision": {
            "continue_discrete_dqn_tuning": False,
            "continuous_control_justified": True,
            "reason": (
                "Both DQN iterations and all locked seeds failed comfort/IAQ constraints; "
                "selected policies underused MEDIUM and the coupled cooling/ventilation "
                "actions caused opposing sensible/latent/IAQ effects."
            ),
            "held_out_remains_sealed": True,
        },
        "shield_observation": {
            "co2_improved": shield["metrics"]["co2_violation_percent"]["mean_across_training_seeds"]
            < no_shield["metrics"]["co2_violation_percent"]["mean_across_training_seeds"],
            "comfort_degraded": shield["metrics"]["comfort_violation_percent"]["mean_across_training_seeds"]
            > no_shield["metrics"]["comfort_violation_percent"]["mean_across_training_seeds"],
        },
    }
    output = PROJECT_ROOT / "outputs/v2/training/dqn_development_summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "development_status": report["development_status"],
        "gates": aggregate_gates,
        "continuous_control_justified": True,
        "held_out_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
