"""Tests for online trends, reliability, and observable bounded risk."""

from __future__ import annotations

from src.envs.v2 import V2ScenarioGenerator
from src.envs.v2.models import V2BuildingState
from src.forecasting import SeasonalProfileForecaster
from src.risk import ForecastReliabilityTracker, ObservableRiskAnalyzer, OnlineSignalMonitor
from src.utils.config import PROJECT_ROOT, load_yaml


def configs():
    return (
        load_yaml(PROJECT_ROOT / "configs/v2/risk.yaml"),
        load_yaml(PROJECT_ROOT / "configs/v2/environment.yaml"),
        load_yaml(PROJECT_ROOT / "configs/v2/forecasting.yaml"),
    )


def test_monitor_detects_sudden_occupancy_and_co2_growth() -> None:
    risk_config, _, _ = configs()
    monitor = OnlineSignalMonitor(risk_config)
    for occupancy, co2 in ((10, 500), (11, 510), (12, 520)):
        quiet = monitor.update(
            {"occupancy": occupancy, "indoor_temperature_c": 24.0, "co2_ppm": co2}
        )
    surge = monitor.update(
        {"occupancy": 55, "indoor_temperature_c": 24.4, "co2_ppm": 610}
    )
    assert quiet.occupancy_surge_score < 0.2
    assert surge.occupancy_surge_score == 1.0
    assert surge.co2_growth_score > quiet.co2_growth_score
    assert surge.signals["occupancy"].delta == 43.0


def test_cusum_detects_repeated_forecast_bias() -> None:
    risk_config, _, _ = configs()
    monitor = OnlineSignalMonitor(risk_config)
    detected = False
    for _ in range(8):
        snapshot = monitor.update(
            {"occupancy": 40, "indoor_temperature_c": 24.0, "co2_ppm": 700},
            {"occupancy": (20.0, 3.0)},
        )
        detected = detected or snapshot.anomaly_detected
    assert detected
    assert snapshot.forecast_error_score > 0.9


def test_reliability_decreases_with_error_and_unavailability() -> None:
    risk_config, _, _ = configs()
    tracker = ForecastReliabilityTracker(risk_config)
    good = tracker.update(
        {
            "outdoor_temperature_c": (0.1, 1.0),
            "outdoor_relative_humidity_pct": (0.2, 3.0),
            "solar_radiation_w_per_m2": (3.0, 30.0),
            "occupancy": (1.0, 5.0),
            "electricity_price_per_kwh": (0.0, 0.02),
        }
    )
    bad = good
    for _ in range(6):
        bad = tracker.update({"occupancy": (35.0, 3.0)}, anomaly_score=1.0)
    unavailable = tracker.update({}, unavailable_features={"outdoor_temperature_c"})
    assert bad["occupancy"] < good["occupancy"]
    assert unavailable["weather"] == 0.0


def test_risk_vector_is_bounded_observable_and_changes_with_context() -> None:
    risk_config, environment, forecast_config = configs()
    generator = V2ScenarioGenerator.from_project()
    training = set(generator.config["splits"]["train"])
    forecaster = SeasonalProfileForecaster(forecast_config)
    forecaster.fit(
        [generator.generate(name, 42) for name in training],
        allowed_scenarios=training,
    )
    timeline = generator.generate("normal_v2", 42)
    monitor = OnlineSignalMonitor(risk_config)
    normal_snapshot = monitor.update(
        {"occupancy": 30, "indoor_temperature_c": 24.0, "co2_ppm": 650}
    )
    analyzer = ObservableRiskAnalyzer(risk_config, environment)
    forecast = forecaster.predict(timeline.inputs[40], 40)
    normal = analyzer.analyze(
        V2BuildingState(24.0, 50.0, 650.0, 0.0, 0, 40),
        timeline.inputs[40],
        normal_snapshot,
        forecast,
        {"weather": 1.0, "occupancy": 1.0, "price": 1.0},
    )
    danger_snapshot = monitor.update(
        {"occupancy": 90, "indoor_temperature_c": 25.8, "co2_ppm": 970}
    )
    danger = analyzer.analyze(
        V2BuildingState(25.8, 70.0, 970.0, 0.0, 0, 41),
        timeline.inputs[41],
        danger_snapshot,
        forecaster.predict(timeline.inputs[41], 41),
        {"weather": 0.3, "occupancy": 0.2, "price": 0.9},
    )
    assert all(0.0 <= value <= 1.0 for value in danger.as_dict().values())
    assert danger.thermal_risk > normal.thermal_risk
    assert danger.co2_risk > normal.co2_risk
    assert danger.occupancy_surge > normal.occupancy_surge
