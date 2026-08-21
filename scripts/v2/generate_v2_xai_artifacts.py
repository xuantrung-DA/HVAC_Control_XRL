"""Generate representative development-only V2 XAI artifacts.

Held-out scenarios are intentionally unavailable through the service used here.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.v2_agent_service import V2AgentService  # noqa: E402
from src.services.v2_simulation_service import V2SimulationService  # noqa: E402


SCENARIOS = ("normal_v2", "hot_day_v2", "high_occupancy_v2")
REPRESENTATIVE_STEPS = (32, 48, 60, 72)


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs/v2/xai"
    output_dir.mkdir(parents=True, exist_ok=True)
    agent_service = V2AgentService()
    simulation_service = V2SimulationService(agent_service)
    samples: list[dict] = []
    csv_rows: list[dict] = []
    for scenario in SCENARIOS:
        result = simulation_service.run(
            scenario=scenario, seed=901, include_explanations=True
        )
        for index in REPRESENTATIVE_STEPS:
            record = result["trajectory"][index]
            samples.append(record)
            explanation = record["policy_explanation"]
            top = sorted(
                explanation["contributions"],
                key=lambda item: item["absolute_importance_pct"],
                reverse=True,
            )[0]
            csv_rows.append({
                "scenario": scenario,
                "step": index,
                "timestamp": record["timestamp"],
                "proposed_action": record["proposed_action_name"],
                "shield_decision": record["shield_explanation"]["decision"],
                "executed_action": record["executed_action_name"],
                "top_feature": top["feature"],
                "top_feature_importance_pct": top["absolute_importance_pct"],
                "energy_priority_pct": record["reward_audit"]["priority_percent"]["energy"],
                "comfort_priority_pct": record["reward_audit"]["priority_percent"]["comfort"],
                "co2_priority_pct": record["reward_audit"]["priority_percent"]["co2"],
                "counterfactual_found": explanation["counterfactual"]["found"],
            })

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "simulator_version": "XRL-HVAC-v2",
        "scope": "development_only",
        "held_out_used": False,
        "controller_status": "DEVELOPMENT_FAIL",
        "method": "local_q_margin_feature_ablation",
        "shield_method": "deterministic_predictive_constraint_check",
        "causal_claim": False,
        "scenarios": list(SCENARIOS),
        "representative_steps": list(REPRESENTATIVE_STEPS),
        "checkpoint": agent_service.metadata(),
        "samples": samples,
    }
    json_path = output_dir / "v2_development_xai_samples.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = output_dir / "v2_development_xai_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({
        "json": str(json_path.relative_to(PROJECT_ROOT)),
        "csv": str(csv_path.relative_to(PROJECT_ROOT)),
        "samples": len(samples),
        "held_out_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
