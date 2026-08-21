"""Generate auditable summaries for all V2 exogenous scenario profiles."""

from __future__ import annotations

import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.v2 import V2ScenarioGenerator  # noqa: E402
from src.utils.v2_manifest import file_sha256  # noqa: E402


def main() -> None:
    generator = V2ScenarioGenerator.from_project()
    summaries: list[dict[str, object]] = []
    all_passed = True
    for scenario in generator.definitions:
        timeline = generator.generate(scenario, seed=42)
        temperature = np.array([item.outdoor_temperature_c for item in timeline.inputs])
        humidity = np.array([item.outdoor_relative_humidity_pct for item in timeline.inputs])
        solar = np.array([item.solar_radiation_w_per_m2 for item in timeline.inputs])
        occupancy = np.array([item.occupancy for item in timeline.inputs])
        night_solar = [
            item.solar_radiation_w_per_m2
            for item in timeline.inputs
            if item.hour < 6.0 or item.hour > 18.0
        ]
        checks = {
            "episode_has_96_steps": len(timeline.inputs) == 96,
            "occupancy_within_capacity": bool(
                occupancy.min() >= 0 and occupancy.max() <= generator.capacity
            ),
            "weather_bounds_valid": bool(
                humidity.min() >= 5.0 and humidity.max() <= 100.0 and solar.min() >= 0.0
            ),
            "night_solar_zero": max(night_solar, default=0.0) < 1e-9,
            "temperature_humidity_correlated": bool(
                np.corrcoef(temperature, humidity)[0, 1] < -0.75
            ),
        }
        all_passed = all_passed and all(checks.values())
        summaries.append(
            {
                "scenario": scenario,
                "split": generator.scenario_split(scenario),
                "description": generator.definitions[scenario]["description"],
                "temperature_min_c": float(temperature.min()),
                "temperature_max_c": float(temperature.max()),
                "humidity_min_pct": float(humidity.min()),
                "humidity_max_pct": float(humidity.max()),
                "solar_peak_w_per_m2": float(solar.max()),
                "occupancy_peak": int(occupancy.max()),
                "price_peak": max(item.electricity_price_per_kwh for item in timeline.inputs),
                "door_open_steps": sum(int(item.door_state != 0) for item in timeline.inputs),
                "cleaning_steps": sum(int(item.cleaning_equipment_on) for item in timeline.inputs),
                "events": list(timeline.event_metadata),
                "forecast_failure": timeline.forecast_failure,
                "checks": checks,
            }
        )

    output_directory = PROJECT_ROOT / "outputs/v2/validation"
    output_directory.mkdir(parents=True, exist_ok=True)
    config_path = PROJECT_ROOT / "configs/v2/scenarios.yaml"
    report = {
        "schema_version": 1,
        "simulator_version": "XRL-HVAC-v2",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "seed": 42,
        "scenario_config_sha256": file_sha256(config_path),
        "scenario_count": len(summaries),
        "all_checks_passed": all_passed,
        "profile_design": {
            "weather": "daily curves plus seeded AR(1) disturbances",
            "humidity": "anti-correlated with temperature plus seeded AR(1) residual",
            "occupancy": "arrival, office, lunch dip, departure, bounded events",
            "event_forecast_visibility": "explicit per event; unexpected events hidden",
        },
        "scenarios": summaries,
    }
    report_path = output_directory / "scenario_profile_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    csv_path = output_directory / "scenario_profile_summary.csv"
    scalar_fields = [
        "scenario",
        "split",
        "temperature_min_c",
        "temperature_max_c",
        "humidity_min_pct",
        "humidity_max_pct",
        "solar_peak_w_per_m2",
        "occupancy_peak",
        "price_peak",
        "door_open_steps",
        "cleaning_steps",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary[field] for field in scalar_fields})
    print(
        json.dumps(
            {
                "all_checks_passed": all_passed,
                "scenarios": len(summaries),
                "report": str(report_path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
    )
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
