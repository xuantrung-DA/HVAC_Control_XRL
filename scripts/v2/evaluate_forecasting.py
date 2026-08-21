"""Fit on V2 train scenarios and evaluate forecasts on validation only."""

from __future__ import annotations

import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.v2 import V2ScenarioGenerator  # noqa: E402
from src.forecasting import SeasonalProfileForecaster, evaluate_forecaster  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402
from src.utils.v2_manifest import file_sha256  # noqa: E402


def main() -> None:
    generator = V2ScenarioGenerator.from_project()
    config_path = PROJECT_ROOT / "configs/v2/forecasting.yaml"
    config = load_yaml(config_path)
    evaluation_config = config["evaluation"]
    training_scenarios = set(generator.config["splits"][evaluation_config["fit_split"]])
    validation_scenarios = set(
        generator.config["splits"][evaluation_config["evaluation_split"]]
    )
    held_out = {
        scenario
        for split, scenarios in generator.config["splits"].items()
        if split.startswith("held_out")
        for scenario in scenarios
    }
    if (training_scenarios | validation_scenarios) & held_out:
        raise RuntimeError("Held-out scenario leaked into forecast development")
    training = [
        generator.generate(scenario, seed)
        for scenario in sorted(training_scenarios)
        for seed in evaluation_config["training_seeds"]
    ]
    validation = [
        generator.generate(scenario, seed)
        for scenario in sorted(validation_scenarios)
        for seed in evaluation_config["evaluation_seeds"]
    ]
    forecaster = SeasonalProfileForecaster(config)
    forecaster.fit(training, allowed_scenarios=training_scenarios)
    evaluation = evaluate_forecaster(forecaster, validation)
    evaluation["schema_version"] = 1
    evaluation["generated_at_utc"] = datetime.now(UTC).isoformat()
    evaluation["forecast_config_sha256"] = file_sha256(config_path)
    evaluation["fit_metadata"] = forecaster.fit_metadata
    evaluation["held_out_scenarios_excluded"] = sorted(held_out)

    output_directory = PROJECT_ROOT / "outputs/v2/forecasting"
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "forecast_model.json").write_text(
        json.dumps(forecaster.model_state(), indent=2) + "\n", encoding="utf-8"
    )
    report_path = output_directory / "forecast_validation_report.json"
    report_path.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
    with (output_directory / "forecast_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evaluation["metrics"][0]))
        writer.writeheader()
        writer.writerows(evaluation["metrics"])
    temperature = [
        item for item in evaluation["metrics"] if item["feature"] == "outdoor_temperature_c"
    ]
    print(
        json.dumps(
            {
                "fit_timelines": len(training),
                "validation_timelines": len(validation),
                "held_out_used": False,
                "temperature_mae_by_horizon": {
                    str(item["horizon_hours"]): item["mae"] for item in temperature
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
