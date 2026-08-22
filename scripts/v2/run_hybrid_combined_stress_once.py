"""Irreversible one-shot Combined Stress evaluation for the frozen hybrid candidate."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.dqn import DQNAgent  # noqa: E402
from src.baselines.v2_controllers import HybridRuleBasedController  # noqa: E402
from src.envs.v2 import V2HybridHVACEnv  # noqa: E402
from src.evaluation.v2_hybrid import (  # noqa: E402
    aggregate_hybrid_results,
    evaluate_hybrid_controller,
)
from src.services.v2_agent_service import V2AgentService  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402
from src.utils.v2_manifest import file_sha256, write_json  # noqa: E402


def main() -> None:
    manifest_path = PROJECT_ROOT / "outputs/v2/hybrid/frozen_candidate_manifest.json"
    receipt_path = PROJECT_ROOT / "outputs/v2/hybrid/combined_stress_one_shot.json"
    if receipt_path.exists():
        raise RuntimeError(
            "Combined Stress has already been opened; the one-shot protocol forbids reruns"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for relative, expected in manifest["component_hashes"].items():
        actual = file_sha256(PROJECT_ROOT / relative)
        if actual != expected:
            mismatches.append({"path": relative, "expected": expected, "actual": actual})
    if mismatches:
        raise RuntimeError(f"Frozen component hash mismatch: {mismatches}")

    hybrid = load_yaml(PROJECT_ROOT / "configs/v2/hybrid_control.yaml")
    protocol = hybrid["development_protocol"]
    scenario = str(protocol["held_out_scenario"])
    seeds = [int(value) for value in protocol["held_out_seeds"]]
    if scenario != manifest["held_out_scenario"] or seeds != manifest["held_out_seeds"]:
        raise RuntimeError("Frozen held-out protocol mismatch")
    selected = manifest["candidate"]
    checkpoint = PROJECT_ROOT / selected["checkpoint"]
    reference_config = V2AgentService().agent.config
    config = deepcopy(reference_config)
    config["agent"]["seed"] = int(selected["training_seed"])
    env = V2HybridHVACEnv()
    agent = DQNAgent(env.observation_space, env.action_space, config=config)
    agent.load(checkpoint)
    env.close()
    baseline = HybridRuleBasedController(hybrid)

    access_started = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "ACCESS_STARTED",
        "final_test_opened": True,
        "rerun_permitted": False,
        "scenario": scenario,
        "seeds": seeds,
        "frozen_bundle_sha256": manifest["frozen_bundle_sha256"],
        "candidate": selected,
        "local_only": True,
    }
    write_json(receipt_path, access_started)

    baseline_episodes = evaluate_hybrid_controller(
        baseline,
        controller_name=baseline.name,
        scenarios=[scenario],
        seeds=seeds,
    )
    candidate_episodes = evaluate_hybrid_controller(
        agent,
        controller_name=f"dqn_seed_{selected['training_seed']}",
        scenarios=[scenario],
        seeds=seeds,
    )
    baseline_aggregate = aggregate_hybrid_results(baseline_episodes)
    candidate_aggregate = aggregate_hybrid_results(candidate_episodes)
    baseline_metrics = baseline_aggregate["metrics"]
    candidate_metrics = candidate_aggregate["metrics"]
    gates = {
        "energy_at_or_below_matched_rule_based": (
            candidate_metrics["whole_building_kwh"]["mean"]
            <= baseline_metrics["whole_building_kwh"]["mean"]
        ),
        "cost_at_or_below_matched_rule_based": (
            candidate_metrics["electricity_cost"]["mean"]
            <= baseline_metrics["electricity_cost"]["mean"]
        ),
        "comfort_below_5_percent": (
            candidate_metrics["comfort_violation_percent"]["mean"] < 5.0
        ),
        "co2_below_1_percent": candidate_metrics["co2_violation_percent"]["mean"] < 1.0,
        "no_critical_safety_violations": (
            candidate_metrics["critical_safety_violations"]["mean"] == 0.0
        ),
    }
    receipt = {
        **access_started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETED_PASS" if all(gates.values()) else "COMPLETED_FAIL",
        "matched_rule_based": {
            "episodes": [item.as_dict() for item in baseline_episodes],
            "aggregate": baseline_aggregate,
        },
        "candidate_result": {
            "episodes": [item.as_dict() for item in candidate_episodes],
            "aggregate": candidate_aggregate,
        },
        "acceptance_gates": gates,
        "acceptance_pass": all(gates.values()),
    }
    write_json(receipt_path, receipt)

    global_status_path = PROJECT_ROOT / "outputs/v2/protocol/held_out_status.json"
    global_status = json.loads(global_status_path.read_text(encoding="utf-8"))
    global_status.update({
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "status": "PARTIALLY_OPENED_HYBRID_COMBINED_STRESS",
        "final_test_opened": True,
        "candidate_checkpoint": selected["checkpoint"],
        "reason": "Frozen hybrid candidate was eligible and Combined Stress was opened once.",
        "acceptance_result": "PASS" if receipt["acceptance_pass"] else "FAIL",
        "hybrid_receipt": receipt_path.relative_to(PROJECT_ROOT).as_posix(),
        "remaining_sealed_scenarios": [
            "unexpected_occupancy_surge_v2",
            "forecast_failure_v2",
            "heatwave_v2",
            "door_left_open_v2",
        ],
    })
    write_json(global_status_path, global_status)
    print(json.dumps({
        "status": receipt["status"],
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "gates": gates,
        "receipt": receipt_path.relative_to(PROJECT_ROOT).as_posix(),
    }, indent=2))


if __name__ == "__main__":
    main()
