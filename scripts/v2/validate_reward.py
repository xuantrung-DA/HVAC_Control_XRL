"""Generate reward audit evidence without selecting a controller/profile."""

from __future__ import annotations

import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.v2 import V2HVACEnv  # noqa: E402
from src.utils.v2_manifest import file_sha256  # noqa: E402


def action_for(observation, names) -> int:
    values = dict(zip(names, observation, strict=True))
    if values["co2_ppm"] > 900 or values["indoor_temperature_c"] > 25.0:
        return 3
    if values["occupancy"] > 0:
        return 1
    return 0


def rollout(mode: str) -> tuple[dict, list[dict]]:
    env = V2HVACEnv("normal_v2", reward_mode=mode)
    observation, _ = env.reset(seed=42)
    records = []
    terminated = False
    while not terminated:
        action = action_for(observation, env.observation_names)
        observation, _, terminated, _, info = env.step(action)
        audit = info["reward_audit"]
        records.append(
            {
                "step": info["step"],
                "action": action,
                "reward": audit["reward"],
                "energy_component": audit["normalized_components"]["energy"],
                "comfort_component": audit["normalized_components"]["comfort"],
                "co2_component": audit["normalized_components"]["co2"],
                "switch_component": audit["normalized_components"]["switching"],
                "peak_component": audit["normalized_components"]["peak_power"],
                "overcooling_component": audit["normalized_components"]["overcooling"],
                "energy_weight": audit["effective_weights"]["energy"],
                "comfort_weight": audit["effective_weights"]["comfort"],
                "co2_weight": audit["effective_weights"]["co2"],
                "energy_priority_percent": audit["priority_percent"]["energy"],
                "comfort_priority_percent": audit["priority_percent"]["comfort"],
                "co2_priority_percent": audit["priority_percent"]["co2"],
                "comfort_violation": audit["comfort_violation"],
                "co2_violation": audit["co2_violation"],
            }
        )
    return info["episode_metrics"], records


def main() -> None:
    profile_path = PROJECT_ROOT / "configs/reward_profiles/reward_profile_v2_001.json"
    results = {}
    dynamic_records = []
    for mode in ("fixed", "dynamic", "cmdp_lagrangian"):
        metrics, records = rollout(mode)
        results[mode] = metrics
        if mode == "dynamic":
            dynamic_records = records
    reconstructed = all(
        abs(
            record["reward"]
            + sum(
                record[component] * record[weight]
                for component, weight in (
                    ("energy_component", "energy_weight"),
                    ("comfort_component", "comfort_weight"),
                    ("co2_component", "co2_weight"),
                )
            )
            + record["switch_component"] * 0.1
            + record["peak_component"] * 0.2
            + record["overcooling_component"] * 0.5
        )
        < 1e-9
        for record in dynamic_records
    )
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "profile_id": "reward_profile_v2_001",
        "profile_sha256": file_sha256(profile_path),
        "scenario": "normal_v2",
        "seed": 42,
        "profile_selection_performed": False,
        "held_out_used": False,
        "mode_rollout_metrics": results,
        "checks": {
            "all_modes_complete_96_steps": len(dynamic_records) == 96,
            "dynamic_audit_reconstructs_reward": reconstructed,
            "dynamic_weights_changed": len(
                {round(record["energy_weight"], 8) for record in dynamic_records}
            )
            > 1,
            "priorities_sum_to_100": all(
                abs(
                    record["energy_priority_percent"]
                    + record["comfort_priority_percent"]
                    + record["co2_priority_percent"]
                    - 100.0
                )
                < 1e-9
                for record in dynamic_records
            ),
        },
        "limitations": [
            "This is a reward-accounting validation with one deterministic controller, not model selection.",
            "The profile remains unevaluated until multi-seed train/validation benchmarking.",
        ],
    }
    report["all_checks_passed"] = all(report["checks"].values())
    output = PROJECT_ROOT / "outputs/v2/reward"
    output.mkdir(parents=True, exist_ok=True)
    (output / "reward_validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output / "reward_audit_normal_dynamic.json").write_text(
        json.dumps(dynamic_records, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "reward_audit_normal_dynamic.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dynamic_records[0]))
        writer.writeheader()
        writer.writerows(dynamic_records)
    print(json.dumps({"all_checks_passed": report["all_checks_passed"], **report["checks"]}, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
