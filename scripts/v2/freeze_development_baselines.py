"""Freeze V2 development baselines without opening any held-out scenario."""

from __future__ import annotations

import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.dqn import DQNAgent  # noqa: E402
from src.baselines import V2RandomController, V2RuleBasedController  # noqa: E402
from src.envs.hvac_env import HVACEnv  # noqa: E402
from src.envs.v2 import V1AgentOnV2Adapter, V1ObservationAdapter  # noqa: E402
from src.evaluation import aggregate_v2_results, evaluate_v2_controller  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402
from src.utils.v2_manifest import file_sha256  # noqa: E402


def main() -> None:
    scenarios_config = load_yaml(PROJECT_ROOT / "configs/v2/scenarios.yaml")
    controller_path = PROJECT_ROOT / "configs/v2/controllers.yaml"
    controller_config = load_yaml(controller_path)
    scenarios = scenarios_config["splits"]["train"] + scenarios_config["splits"]["validation"]
    seeds = [901, 902, 903, 904, 905]
    v1_env = HVACEnv()
    v1_dqn = DQNAgent(v1_env.observation_space, v1_env.action_space)
    v1_checkpoint = PROJECT_ROOT / "models/dqn/demo_best.pt"
    v1_dqn.load(v1_checkpoint)
    controllers = [
        ("random_v2", V2RandomController(42)),
        ("rule_based_v2", V2RuleBasedController(controller_config)),
        (
            "v1_dqn_on_v2",
            V1AgentOnV2Adapter(v1_dqn, V1ObservationAdapter(v1_env.observation_space)),
        ),
    ]
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "held_out_used": False,
        "development_scenarios": scenarios,
        "evaluation_seeds": seeds,
        "rule_based_version": controller_config["rule_based_v2"]["version"],
        "rule_based_config_sha256": file_sha256(controller_path),
        "v1_checkpoint_sha256": file_sha256(v1_checkpoint),
        "controllers": {},
    }
    rows = []
    for name, controller in controllers:
        results = evaluate_v2_controller(
            controller,
            controller_name=name,
            scenarios=scenarios,
            seeds=seeds,
            shield_enabled=False,
        )
        report["controllers"][name] = aggregate_v2_results(results)
        rows.extend(item.as_dict() for item in results)
    output = PROJECT_ROOT / "outputs/v2/baselines"
    output.mkdir(parents=True, exist_ok=True)
    (output / "development_baseline_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "development_baseline_episodes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "controller": "rule_based_v2",
        "version": controller_config["rule_based_v2"]["version"],
        "config_sha256": file_sha256(controller_path),
        "frozen_before_held_out": True,
        "held_out_used": False,
        "development_report": "outputs/v2/baselines/development_baseline_report.json",
    }
    manifest_path = PROJECT_ROOT / "models/v2/rule_based_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    compact = {
        name: {
            metric: summary["metrics"][metric]["mean"]
            for metric in (
                "whole_building_kwh", "hvac_ventilation_kwh", "electricity_cost",
                "comfort_violation_percent", "co2_violation_percent",
            )
        }
        for name, summary in report["controllers"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
