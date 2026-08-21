"""Validate monitoring/reliability/risk on development scenarios only."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.v2 import TwoR1CBuildingModel, V2ScenarioGenerator  # noqa: E402
from src.forecasting import FORECAST_FEATURES, SeasonalProfileForecaster  # noqa: E402
from src.risk import (  # noqa: E402
    ForecastReliabilityTracker,
    ObservableRiskAnalyzer,
    OnlineSignalMonitor,
)
from src.utils.config import load_yaml  # noqa: E402
from src.utils.v2_manifest import file_sha256  # noqa: E402


def main() -> None:
    generator = V2ScenarioGenerator.from_project()
    risk_path = PROJECT_ROOT / "configs/v2/risk.yaml"
    forecast_config = load_yaml(PROJECT_ROOT / "configs/v2/forecasting.yaml")
    risk_config = load_yaml(risk_path)
    environment = load_yaml(PROJECT_ROOT / "configs/v2/environment.yaml")
    action_mapping = load_yaml(PROJECT_ROOT / "configs/v2/action_mapping.yaml")
    training_scenarios = set(generator.config["splits"]["train"])
    forecaster = SeasonalProfileForecaster(forecast_config)
    forecaster.fit(
        [generator.generate(name, seed) for name in training_scenarios for seed in (42, 123, 2026)],
        allowed_scenarios=training_scenarios,
    )

    scenario = "meeting_surge_v2"
    if scenario not in generator.config["splits"]["validation"]:
        raise RuntimeError("Monitoring validation scenario must remain in validation split")
    timeline = generator.generate(scenario, 901)
    model = TwoR1CBuildingModel(environment, action_mapping)
    state = model.initial_state()
    monitor = OnlineSignalMonitor(risk_config)
    tracker = ForecastReliabilityTracker(risk_config)
    analyzer = ObservableRiskAnalyzer(risk_config, environment)
    forecasts_by_target = {}
    records = []
    detected_step = None
    pre_event_alert_steps = []
    event_start_step = 56
    for step, inputs in enumerate(timeline.inputs):
        residuals = {}
        issued = forecasts_by_target.get(step)
        if issued is not None:
            for feature in FORECAST_FEATURES:
                value = issued.values[feature]
                residuals[feature] = (
                    float(getattr(inputs, feature)) - value.point,
                    value.standard_deviation,
                )
        monitoring = monitor.update(
            {
                "occupancy": inputs.occupancy,
                "indoor_temperature_c": state.indoor_temperature_c,
                "co2_ppm": state.co2_ppm,
            },
            residuals,
        )
        reliability = tracker.update(
            residuals, anomaly_score=monitoring.forecast_error_score
        )
        bundle = forecaster.predict(
            inputs, step, planned_events=timeline.event_metadata
        )
        forecasts_by_target[
            min(step + bundle.forecasts[0].horizon_steps, 95)
        ] = bundle.forecasts[0]
        risk = analyzer.analyze(state, inputs, monitoring, bundle, reliability)
        if monitoring.occupancy_surge_score >= 0.5:
            if 52 <= step < event_start_step:
                pre_event_alert_steps.append(step)
            if detected_step is None and step >= event_start_step:
                detected_step = step
        records.append(
            {
                "step": step,
                "hour": inputs.hour,
                "occupancy": inputs.occupancy,
                "occupancy_surge_score": monitoring.occupancy_surge_score,
                "forecast_error_score": monitoring.forecast_error_score,
                "anomaly_detected": monitoring.anomaly_detected,
                "reliability": reliability,
                "risk": risk.as_dict(),
            }
        )
        action = 3 if state.co2_ppm > 900 or state.indoor_temperature_c > 25.0 else 1
        state, _ = model.step(state, action, inputs)

    detection_delay = None if detected_step is None else detected_step - event_start_step
    normal_tracker = ForecastReliabilityTracker(risk_config)
    initial = normal_tracker.update({"occupancy": (0.5, 5.0)})["occupancy"]
    degraded = initial
    for _ in range(8):
        degraded = normal_tracker.update(
            {"occupancy": (35.0, 3.0)}, anomaly_score=1.0
        )["occupancy"]
    unavailable_weather = normal_tracker.update(
        {}, unavailable_features={"outdoor_temperature_c"}
    )["weather"]
    checks = {
        "development_scenario_only": scenario in generator.config["splits"]["validation"],
        "held_out_not_used": True,
        "meeting_surge_detected": detected_step is not None,
        "detection_delay_within_two_steps": detection_delay is not None and detection_delay <= 2,
        "bad_forecast_lowers_reliability": degraded < initial,
        "unavailable_weather_reliability_zero": unavailable_weather == 0.0,
        "all_risk_values_bounded": all(
            0.0 <= value <= 1.0
            for record in records
            for value in record["risk"].values()
        ),
    }
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "risk_config_sha256": file_sha256(risk_path),
        "validation_scenario": scenario,
        "held_out_used": False,
        "meeting_event_start_step": event_start_step,
        "detection_step": detected_step,
        "detection_delay_steps": detection_delay,
        "pre_event_alert_steps": pre_event_alert_steps,
        "occupancy_reliability_before_fault": initial,
        "occupancy_reliability_after_fault": degraded,
        "weather_reliability_when_unavailable": unavailable_weather,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "risk_formula_note": (
            "All values use current state, past residuals, declared forecasts, and "
            "configured thresholds; no future truth or simulator latent state is used."
        ),
        "limitations": [
            "Raw change detection may alert on a legitimate post-lunch return; policy context must combine the surge score with forecast residual and reliability.",
            "This phase validates signals on a development meeting scenario, not final controller resilience on the sealed unexpected-surge test.",
        ],
        "trajectory": records,
    }
    output = PROJECT_ROOT / "outputs/v2/risk"
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "risk_pipeline_validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "all_checks_passed": report["all_checks_passed"],
                "meeting_detection_delay_steps": detection_delay,
                "reliability_before_fault": initial,
                "reliability_after_fault": degraded,
                "held_out_used": False,
            },
            indent=2,
        )
    )
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
