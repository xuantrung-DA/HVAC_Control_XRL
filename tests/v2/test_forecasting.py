"""Forecast correctness, isolation, uncertainty, and fault tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.envs.v2 import V2ScenarioGenerator
from src.forecasting import ForecastFaultInjector, SeasonalProfileForecaster
from src.utils.config import PROJECT_ROOT, load_yaml


@pytest.fixture(scope="module")
def fitted() -> tuple[V2ScenarioGenerator, SeasonalProfileForecaster, dict]:
    generator = V2ScenarioGenerator.from_project()
    config = load_yaml(PROJECT_ROOT / "configs/v2/forecasting.yaml")
    training = set(generator.config["splits"]["train"])
    timelines = [generator.generate(name, seed) for name in sorted(training) for seed in (42, 123, 2026)]
    forecaster = SeasonalProfileForecaster(config)
    forecaster.fit(timelines, allowed_scenarios=training)
    return generator, forecaster, config


def test_fit_uses_only_allowed_training_scenarios(fitted) -> None:
    generator, forecaster, _ = fitted
    assert set(forecaster.fit_metadata["scenarios"]) == set(generator.config["splits"]["train"])
    assert forecaster.fit_metadata["held_out_used"] is False
    held_out = generator.generate("combined_stress_v2", 1701)
    with pytest.raises(ValueError, match="forbidden"):
        forecaster.fit([held_out], allowed_scenarios=set(generator.config["splits"]["train"]))


def test_forecast_is_deterministic_bounded_and_has_four_horizons(fitted) -> None:
    generator, forecaster, _ = fitted
    timeline = generator.generate("expensive_electricity_v2", 901)
    first = forecaster.predict(timeline.inputs[32], 32)
    second = forecaster.predict(timeline.inputs[32], 32)
    assert first.as_dict() == second.as_dict()
    assert [item.horizon_hours for item in first.forecasts] == [1.0, 2.0, 3.0, 4.0]
    for forecast in first.forecasts:
        for value in forecast.values.values():
            assert value.lower <= value.point <= value.upper
            assert value.standard_deviation > 0.0


def test_hidden_surge_does_not_leak_into_pre_event_forecast(fitted) -> None:
    generator, forecaster, _ = fitted
    normal = generator.generate("normal_v2", 42)
    surge = generator.generate("unexpected_occupancy_surge_v2", 42)
    step_before_event = 52
    assert normal.inputs[step_before_event] == surge.inputs[step_before_event]
    normal_forecast = forecaster.predict(normal.inputs[step_before_event], step_before_event)
    surge_forecast = forecaster.predict(surge.inputs[step_before_event], step_before_event)
    assert normal_forecast.as_dict() == surge_forecast.as_dict()
    assert surge.inputs[56].occupancy > normal.inputs[56].occupancy


def test_visible_meeting_can_be_included_but_hidden_surge_cannot(fitted) -> None:
    generator, forecaster, _ = fitted
    normal = generator.generate("normal_v2", 42)
    meeting = generator.generate("meeting_surge_v2", 42)
    surge = generator.generate("unexpected_occupancy_surge_v2", 42)
    current_step = 52
    baseline = forecaster.predict(normal.inputs[current_step], current_step)
    meeting_forecast = forecaster.predict(
        meeting.inputs[current_step], current_step, planned_events=meeting.event_metadata
    )
    surge_forecast = forecaster.predict(
        surge.inputs[current_step], current_step, planned_events=surge.event_metadata
    )
    baseline_occupancy = baseline.forecasts[0].values["occupancy"].point
    assert meeting_forecast.forecasts[0].values["occupancy"].point > baseline_occupancy
    assert surge_forecast.forecasts[0].values["occupancy"].point == baseline_occupancy


def test_validation_errors_are_finite(fitted) -> None:
    generator, forecaster, _ = fitted
    errors: list[float] = []
    for scenario in generator.config["splits"]["validation"]:
        timeline = generator.generate(scenario, 901)
        for step in range(0, 80, 8):
            bundle = forecaster.predict(timeline.inputs[step], step)
            for forecast in bundle.forecasts:
                actual = timeline.inputs[forecast.target_step].outdoor_temperature_c
                errors.append(actual - forecast.values["outdoor_temperature_c"].point)
    assert np.isfinite(errors).all()
    assert float(np.mean(np.abs(errors))) < 8.0


def test_fault_injection_is_explicit(fitted) -> None:
    generator, forecaster, config = fitted
    timeline = generator.generate("forecast_failure_v2", 1701)
    current = forecaster.predict(timeline.inputs[48], 48)
    stale = forecaster.predict(timeline.inputs[40], 40)
    injector = ForecastFaultInjector(config)
    unavailable = injector.apply(current, "unavailable")
    stale_result = injector.apply(current, "stale", stale_bundle=stale)
    biased = injector.apply(current, "biased")
    assert unavailable.forecasts == tuple()
    assert stale_result.forecasts == stale.forecasts
    assert biased.forecasts[0].values["occupancy"].point < current.forecasts[0].values["occupancy"].point
