"""Measure shield selectivity on development scenarios before RL training."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.v2 import V2HVACEnv  # noqa: E402
from src.utils.v2_manifest import file_sha256  # noqa: E402


def conservative_proposal(observation, names) -> int:
    state = dict(zip(names, observation, strict=True))
    if state["co2_ppm"] >= 900 or state["thermal_risk"] >= 0.75 or state["co2_risk"] >= 0.70:
        return 3
    if state["indoor_temperature_c"] >= 24.5 or state["co2_ppm"] >= 800:
        return 2
    if state["occupancy"] > 0:
        return 1
    return 0


def rollout(scenario: str, seed: int) -> tuple[dict, list[dict]]:
    env = V2HVACEnv(scenario, shield_enabled=True)
    observation, _ = env.reset(seed=seed)
    records = []
    terminated = False
    while not terminated:
        proposed = conservative_proposal(observation, env.observation_names)
        observation, _, terminated, _, info = env.step(proposed)
        shield = info["control"]["shield"]
        records.append(
            {
                "scenario": scenario,
                "seed": seed,
                "step": info["step"],
                "proposed_action": shield["proposed_action"],
                "executed_action": shield["executed_action"],
                "decision": shield["decision"],
                "intervention": shield["intervention"],
                "constraint": shield["constraint"],
                "reason": shield["reason"],
            }
        )
    counts = Counter(record["decision"] for record in records)
    metrics = {
        "scenario": scenario,
        "seed": seed,
        "steps": len(records),
        "decision_counts": dict(counts),
        "intervention_rate": sum(record["intervention"] for record in records) / len(records),
        "fallback_rate": counts["FALLBACK"] / len(records),
        "proposed_executed_mismatch_rate": sum(
            record["proposed_action"] != record["executed_action"] for record in records
        )
        / len(records),
    }
    return metrics, records


def main() -> None:
    all_records = []
    metrics = []
    for scenario in ("normal_v2", "meeting_surge_v2"):
        result, records = rollout(scenario, seed=901)
        metrics.append(result)
        all_records.extend(records)
    normal = next(item for item in metrics if item["scenario"] == "normal_v2")
    checks = {
        "normal_intervention_rate_at_most_20_percent": normal["intervention_rate"] <= 0.20,
        "normal_fallback_rate_at_most_5_percent": normal["fallback_rate"] <= 0.05,
        "all_decisions_logged": len(all_records) == 192,
        "executed_actions_valid": all(record["executed_action"] in range(4) for record in all_records),
        "action_mismatch_implies_intervention": all(
            record["intervention"]
            or record["proposed_action"] == record["executed_action"]
            for record in all_records
        ),
    }
    config_path = PROJECT_ROOT / "configs/v2/shield.yaml"
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "shield_config_sha256": file_sha256(config_path),
        "held_out_used": False,
        "controller": "conservative_development_proposal",
        "scenario_metrics": metrics,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "limitations": [
            "Selectivity is validated with a deterministic development controller; RL-specific intervention rates are measured after training.",
            "No held-out safety or resilience scenario is opened in this phase.",
        ],
    }
    output = PROJECT_ROOT / "outputs/v2/shield"
    output.mkdir(parents=True, exist_ok=True)
    (output / "shield_validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "shield_decisions_development.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_records[0]))
        writer.writeheader()
        writer.writerows(all_records)
    print(json.dumps({"all_checks_passed": report["all_checks_passed"], "scenarios": metrics}, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
