"""Tests for correlated and reproducible V2 scenario profiles."""

from __future__ import annotations

import numpy as np
import pytest

from src.envs.v2 import DoorState, TwoR1CBuildingModel, V2ScenarioGenerator
from src.utils.config import PROJECT_ROOT, load_yaml


@pytest.fixture(scope="module")
def generator() -> V2ScenarioGenerator:
    return V2ScenarioGenerator.from_project()


def test_all_locked_scenarios_are_defined_once(generator: V2ScenarioGenerator) -> None:
    assigned = [name for names in generator.config["splits"].values() for name in names]
    assert len(assigned) == len(set(assigned)) == 13
    assert set(assigned) == set(generator.definitions)


def test_generation_is_seed_reproducible(generator: V2ScenarioGenerator) -> None:
    first = generator.generate("normal_v2", 42)
    second = generator.generate("normal_v2", 42)
    third = generator.generate("normal_v2", 43)
    assert first.as_records() == second.as_records()
    assert first.as_records() != third.as_records()


def test_weather_is_correlated_and_solar_is_zero_at_night(generator: V2ScenarioGenerator) -> None:
    timeline = generator.generate("normal_v2", 42)
    temperature = np.array([item.outdoor_temperature_c for item in timeline.inputs])
    humidity = np.array([item.outdoor_relative_humidity_pct for item in timeline.inputs])
    solar = np.array([item.solar_radiation_w_per_m2 for item in timeline.inputs])
    assert np.corrcoef(temperature, humidity)[0, 1] < -0.85
    assert all(item.solar_radiation_w_per_m2 == 0.0 for item in timeline.inputs if item.hour < 6.0 or item.hour > 18.0)
    assert solar.max() > 500.0


def test_occupancy_has_arrival_lunch_dip_and_departure(generator: V2ScenarioGenerator) -> None:
    timeline = generator.generate("normal_v2", 42)
    at = {item.hour: item.occupancy for item in timeline.inputs}
    assert at[6.0] == 0
    assert at[10.0] > at[8.0]
    assert at[12.5] < at[11.0]
    assert at[20.0] == 0
    assert max(at.values()) <= generator.capacity


def test_scenario_events_are_explicit_and_forecast_visibility_is_honest(generator: V2ScenarioGenerator) -> None:
    meeting = generator.generate("meeting_surge_v2", 42)
    surge = generator.generate("unexpected_occupancy_surge_v2", 42)
    door = generator.generate("door_left_open_v2", 42)
    assert next(event for event in meeting.event_metadata if event["event"] == "meeting")["forecast_visible"] is True
    assert next(event for event in surge.event_metadata if event["event"] == "occupancy_surge")["forecast_visible"] is False
    assert next(event for event in door.event_metadata if event["event"] == "door_left_open")["forecast_visible"] is False
    assert all(item.door_state is DoorState.LEFT_OPEN for item in door.inputs if 11.0 <= item.hour < 13.0)


def test_tou_and_equipment_profiles_change_by_scenario(generator: V2ScenarioGenerator) -> None:
    normal = generator.generate("normal_v2", 42)
    expensive = generator.generate("expensive_electricity_v2", 42)
    electronics = generator.generate("high_electronics_load_v2", 42)
    assert max(item.electricity_price_per_kwh for item in expensive.inputs) > max(item.electricity_price_per_kwh for item in normal.inputs)
    normal_active = normal.inputs[40]
    electronics_active = electronics.inputs[40]
    assert electronics_active.desktop_count == normal_active.desktop_count
    assert electronics_active.electronics_load_multiplier > normal_active.electronics_load_multiplier


def test_every_scenario_completes_a_physics_episode(generator: V2ScenarioGenerator) -> None:
    model = TwoR1CBuildingModel(
        load_yaml(PROJECT_ROOT / "configs/v2/environment.yaml"),
        load_yaml(PROJECT_ROOT / "configs/v2/action_mapping.yaml"),
    )
    for scenario in generator.definitions:
        state = model.initial_state()
        timeline = generator.generate(scenario, 42)
        assert len(timeline.inputs) == 96
        for index, inputs in enumerate(timeline.inputs):
            state, transition = model.step(state, index % 4, inputs)
            assert transition.energy.whole_building_kwh >= 0.0
        assert state.step == 96


def test_unknown_scenario_is_rejected(generator: V2ScenarioGenerator) -> None:
    with pytest.raises(ValueError, match="Unknown V2 scenario"):
        generator.generate("not_a_scenario", 42)
