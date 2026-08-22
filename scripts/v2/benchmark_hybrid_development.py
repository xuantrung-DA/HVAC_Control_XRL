"""Clean development benchmark for the learning-augmented V2 candidate."""

from __future__ import annotations

import hashlib
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gates(aggregate: dict, baseline: dict) -> dict[str, bool]:
    metrics = aggregate["metrics"]
    baseline_metrics = baseline["metrics"]
    return {
        "energy_at_or_below_matched_rule_based": (
            metrics["whole_building_kwh"]["mean"]
            <= baseline_metrics["whole_building_kwh"]["mean"]
        ),
        "cost_at_or_below_matched_rule_based": (
            metrics["electricity_cost"]["mean"]
            <= baseline_metrics["electricity_cost"]["mean"]
        ),
        "comfort_below_5_percent": (
            metrics["comfort_violation_percent"]["mean"] < 5.0
        ),
        "co2_below_1_percent": metrics["co2_violation_percent"]["mean"] < 1.0,
        "no_critical_safety_violations": (
            metrics["critical_safety_violations"]["mean"] == 0.0
        ),
    }


def main() -> None:
    hybrid = load_yaml(PROJECT_ROOT / "configs/v2/hybrid_control.yaml")
    protocol = hybrid["development_protocol"]
    scenarios = tuple(protocol["validation_scenarios"])
    seeds = tuple(int(value) for value in protocol["validation_seeds"])
    if "combined_stress_v2" in scenarios:
        raise RuntimeError("Sealed Combined Stress cannot be used in development")

    baseline_controller = HybridRuleBasedController(hybrid)
    baseline_episodes = evaluate_hybrid_controller(
        baseline_controller,
        controller_name=baseline_controller.name,
        scenarios=scenarios,
        seeds=seeds,
    )
    baseline = aggregate_hybrid_results(baseline_episodes)

    training_summary = json.loads(
        (PROJECT_ROOT / "outputs/v2/training/dqn_development_summary.json").read_text(
            encoding="utf-8"
        )
    )
    by_seed = {int(item["seed"]): item for item in training_summary["per_seed"]}
    reference_config = V2AgentService().agent.config
    reference_env = V2HybridHVACEnv()
    candidates = []
    for training_seed in protocol["training_checkpoints"]:
        selected = by_seed[int(training_seed)]
        checkpoint = PROJECT_ROOT / str(selected["checkpoint"]).replace("\\", "/")
        checkpoint_hash = sha256(checkpoint)
        if checkpoint_hash != selected["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint integrity failure: {checkpoint}")
        config = deepcopy(reference_config)
        config["agent"]["seed"] = int(training_seed)
        agent = DQNAgent(
            reference_env.observation_space,
            reference_env.action_space,
            config=config,
        )
        agent.load(checkpoint)
        episodes = evaluate_hybrid_controller(
            agent,
            controller_name=f"dqn_seed_{training_seed}",
            scenarios=scenarios,
            seeds=seeds,
        )
        aggregate = aggregate_hybrid_results(episodes)
        candidate_gates = gates(aggregate, baseline)
        candidates.append(
            {
                "training_seed": int(training_seed),
                "checkpoint": checkpoint.relative_to(PROJECT_ROOT).as_posix(),
                "checkpoint_sha256": checkpoint_hash,
                "episodes": [item.as_dict() for item in episodes],
                "aggregate": aggregate,
                "gates": candidate_gates,
                "pass": all(candidate_gates.values()),
            }
        )
    reference_env.close()
    eligible = [item for item in candidates if item["pass"]]
    eligible.sort(
        key=lambda item: (
            item["aggregate"]["metrics"]["whole_building_kwh"]["mean"],
            item["aggregate"]["metrics"]["electricity_cost"]["mean"],
            item["aggregate"]["metrics"]["comfort_violation_percent"]["mean"],
        )
    )
    selected = eligible[0] if eligible else None
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "experiment": "xrl_hvac_v2_hybrid_clean_development_benchmark",
        "simulator_version": V2HybridHVACEnv.simulator_version,
        "held_out_used": False,
        "local_only": True,
        "validation_scenarios": list(scenarios),
        "validation_seeds": list(seeds),
        "matched_rule_based": {
            "controller_version": baseline_controller.version,
            "episodes": [item.as_dict() for item in baseline_episodes],
            "aggregate": baseline,
        },
        "candidates": candidates,
        "all_training_seed_candidates_pass": all(item["pass"] for item in candidates),
        "selected_candidate": (
            {
                "training_seed": selected["training_seed"],
                "checkpoint": selected["checkpoint"],
                "checkpoint_sha256": selected["checkpoint_sha256"],
                "selection_rule": "constraints_then_energy_then_cost_then_comfort",
            }
            if selected
            else None
        ),
        "development_pass": selected is not None,
    }
    output = PROJECT_ROOT / "outputs/v2/hybrid/development_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "matched_rule_based": baseline["metrics"],
        "candidates": [
            {
                "training_seed": item["training_seed"],
                "metrics": item["aggregate"]["metrics"],
                "pass": item["pass"],
            }
            for item in candidates
        ],
        "selected_candidate": report["selected_candidate"],
        "development_pass": report["development_pass"],
        "output": output.relative_to(PROJECT_ROOT).as_posix(),
    }, indent=2))


if __name__ == "__main__":
    main()
