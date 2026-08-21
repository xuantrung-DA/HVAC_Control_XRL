"""Record the evidence-based SAC go/no-go decision."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    experiment_id = "xrl_hvac_v2_sac_002_cpu_decoupled"
    report_path = (
        PROJECT_ROOT
        / "outputs/v2/training"
        / experiment_id
        / "full/sac_seed_42_report.json"
    )
    training_report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = training_report["validation"]["metrics"]
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "experiment_id": experiment_id,
        "held_out_used": False,
        "continuous_control_physically_decoupled": True,
        "go_no_go_seed": 42,
        "steps_requested": training_report["steps_requested"],
        "steps_used": training_report["steps_used"],
        "stopped_early": training_report["stopped_early"],
        "checkpoint_reproducible": training_report["checkpoint_reproducible"],
        "validation": {
            "whole_building_kwh": metrics["whole_building_kwh"]["mean"],
            "electricity_cost": metrics["electricity_cost"]["mean"],
            "comfort_violation_percent": metrics["comfort_violation_percent"]["mean"],
            "co2_violation_percent": metrics["co2_violation_percent"]["mean"],
            "cooling_fraction_mean": metrics["cooling_fraction_mean"]["mean"],
            "ventilation_fraction_mean": metrics["ventilation_fraction_mean"]["mean"],
        },
        "development_status": "FAIL",
        "decision": {
            "run_remaining_training_seeds": False,
            "reason": (
                "After replay warmup and 10k transitions, comfort remained above 70% "
                "and worsened between evaluations. The predeclared go/no-go threshold "
                "stopped an unpromising, resource-expensive candidate."
            ),
            "claim_continuous_control_improves_v2": False,
            "retain_continuous_interface_for_future_work": True,
        },
        "resource_usage": {
            "parameters": training_report["parameters"],
            "device": training_report["device"],
            "steps_per_second_including_validation": training_report[
                "steps_per_second_including_validation"
            ],
            "rss_delta_mb": training_report["rss_delta_mb"],
            "cuda_peak_allocated_mb": training_report["cuda_peak_allocated_mb"],
        },
        "dqn_justification_artifact": "outputs/v2/training/dqn_development_summary.json",
    }
    output = PROJECT_ROOT / "outputs/v2/training/sac_go_no_go_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "development_status": summary["development_status"],
        "stopped_early": summary["stopped_early"],
        "held_out_used": False,
        "run_remaining_training_seeds": False,
    }, indent=2))


if __name__ == "__main__":
    main()
