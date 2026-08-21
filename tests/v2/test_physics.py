"""Unit tests for the V2 2R1C building physics core."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.envs.v2 import (
    DoorState,
    TwoR1CBuildingModel,
    V2BuildingState,
    V2ExogenousInputs,
)
from src.envs.v2.psychrometrics import humidity_ratio, relative_humidity_pct
from src.utils.config import PROJECT_ROOT
from src.utils.v2_manifest import load_yaml


@pytest.fixture
def model() -> TwoR1CBuildingModel:
    return TwoR1CBuildingModel(
        load_yaml(PROJECT_ROOT / "configs/v2/environment.yaml"),
        load_yaml(PROJECT_ROOT / "configs/v2/action_mapping.yaml"),
    )


@pytest.fixture
def mild_inputs() -> V2ExogenousInputs:
    return V2ExogenousInputs(
        outdoor_temperature_c=30.0,
        outdoor_relative_humidity_pct=60.0,
        solar_radiation_w_per_m2=0.0,
        occupancy=0,
        electricity_price_per_kwh=0.10,
        hour=0.0,
    )


def test_config_is_genuine_lumped_2r1c() -> None:
    config = load_yaml(PROJECT_ROOT / "configs/v2/environment.yaml")
    assert config["simulation"]["thermal_network"] == "2R1C"
    assert "opaque_resistance_c_per_kw" in config["envelope"]
    assert "window_resistance_c_per_kw" in config["envelope"]
    assert "effective_thermal_capacity_kwh_per_c" in config["envelope"]
    assert "thermal_mass_temperature" not in str(config)


def test_psychrometric_round_trip() -> None:
    ratio = humidity_ratio(25.0, 55.0)
    assert ratio > 0.0
    assert relative_humidity_pct(25.0, ratio) == pytest.approx(55.0, abs=1e-6)


def test_same_state_and_inputs_are_deterministic(
    model: TwoR1CBuildingModel, mild_inputs: V2ExogenousInputs
) -> None:
    state = model.initial_state()
    first = model.step(state, 2, mild_inputs)
    second = model.step(state, 2, mild_inputs)
    assert first == second


def test_hvac_has_actuator_inertia(
    model: TwoR1CBuildingModel, mild_inputs: V2ExogenousInputs
) -> None:
    state = model.initial_state()
    next_state, transition = model.step(state, 3, mild_inputs)
    assert 0.0 < transition.delivered_cooling_kw < transition.commanded_cooling_kw
    assert transition.effective_cooling_kw < transition.delivered_cooling_kw
    assert next_state.delivered_cooling_kw == transition.delivered_cooling_kw


def test_hvac_energy_is_ordered_by_action(
    model: TwoR1CBuildingModel, mild_inputs: V2ExogenousInputs
) -> None:
    state = model.initial_state()
    energy = [
        model.step(state, action, mild_inputs)[1].energy.controllable_hvac_ventilation_kwh
        for action in range(4)
    ]
    assert energy[0] == 0.0
    assert energy[3] >= energy[2] >= energy[1] >= energy[0]


def test_occupancy_increases_heat_co2_and_moisture(
    model: TwoR1CBuildingModel, mild_inputs: V2ExogenousInputs
) -> None:
    state = model.initial_state()
    low_state, low = model.step(state, 0, replace(mild_inputs, occupancy=5))
    high_state, high = model.step(state, 0, replace(mild_inputs, occupancy=50))
    assert high.heat_flows.occupants_kw > low.heat_flows.occupants_kw
    assert high_state.indoor_temperature_c > low_state.indoor_temperature_c
    assert high_state.co2_ppm > low_state.co2_ppm
    assert high.air_quality.occupant_moisture_generation_kg > low.air_quality.occupant_moisture_generation_kg


def test_electronics_increase_heat_and_whole_building_energy(
    model: TwoR1CBuildingModel, mild_inputs: V2ExogenousInputs
) -> None:
    state = model.initial_state()
    _, low = model.step(state, 0, mild_inputs)
    loaded = replace(
        mild_inputs,
        desktop_count=40,
        laptop_count=20,
        monitor_count=50,
    )
    high_state, high = model.step(state, 0, loaded)
    assert high.heat_flows.electronics_kw > low.heat_flows.electronics_kw
    assert high.energy.electronics_kwh > low.energy.electronics_kwh
    assert high.energy.whole_building_kwh > low.energy.whole_building_kwh
    assert high_state.indoor_temperature_c > model.step(state, 0, mild_inputs)[0].indoor_temperature_c


def test_door_open_duration_category_increases_hot_air_heat_gain(
    model: TwoR1CBuildingModel, mild_inputs: V2ExogenousInputs
) -> None:
    state = model.initial_state()
    hot = replace(mild_inputs, outdoor_temperature_c=40.0)
    transitions = [
        model.step(state, 0, replace(hot, door_state=door_state))[1]
        for door_state in DoorState
    ]
    assert transitions[2].heat_flows.infiltration_kw > transitions[1].heat_flows.infiltration_kw
    assert transitions[1].heat_flows.infiltration_kw > transitions[0].heat_flows.infiltration_kw


def test_ventilation_reduces_co2_but_adds_hot_weather_load(
    model: TwoR1CBuildingModel, mild_inputs: V2ExogenousInputs
) -> None:
    state = replace(model.initial_state(), co2_ppm=1400.0)
    hot = replace(mild_inputs, outdoor_temperature_c=40.0)
    off_state, off = model.step(state, 0, hot)
    high_state, high = model.step(state, 3, hot)
    assert high_state.co2_ppm < off_state.co2_ppm
    assert high.heat_flows.ventilation_kw > off.heat_flows.ventilation_kw


def test_energy_accounting_sums_and_tou_cost_changes(
    model: TwoR1CBuildingModel, mild_inputs: V2ExogenousInputs
) -> None:
    state = model.initial_state()
    occupied = replace(
        mild_inputs,
        occupancy=40,
        desktop_count=30,
        monitor_count=30,
        lighting_fraction=1.0,
        cleaning_equipment_on=True,
    )
    _, off_peak = model.step(state, 2, occupied)
    _, peak = model.step(
        state,
        2,
        replace(occupied, electricity_price_per_kwh=0.60),
    )
    energy = off_peak.energy
    component_sum = (
        energy.hvac_cooling_kwh
        + energy.ventilation_fan_kwh
        + energy.lighting_kwh
        + energy.electronics_kwh
        + energy.base_building_kwh
        + energy.cleaning_equipment_kwh
    )
    assert energy.whole_building_kwh == pytest.approx(component_sum)
    assert peak.energy.whole_building_kwh == pytest.approx(energy.whole_building_kwh)
    assert peak.energy.electricity_cost > energy.electricity_cost


def test_temperature_humidity_and_co2_remain_bounded_under_stress(
    model: TwoR1CBuildingModel, mild_inputs: V2ExogenousInputs
) -> None:
    state = V2BuildingState(
        indoor_temperature_c=44.0,
        indoor_relative_humidity_pct=98.0,
        co2_ppm=4900.0,
        delivered_cooling_kw=0.0,
        hvac_action=0,
        step=0,
    )
    stress = replace(
        mild_inputs,
        outdoor_temperature_c=45.0,
        outdoor_relative_humidity_pct=100.0,
        solar_radiation_w_per_m2=1000.0,
        occupancy=100,
        door_state=DoorState.LEFT_OPEN,
        desktop_count=100,
        monitor_count=100,
        lighting_fraction=1.0,
    )
    for _ in range(96):
        previous_temperature = state.indoor_temperature_c
        state, transition = model.step(state, 0, stress)
        assert 5.0 <= state.indoor_temperature_c <= 45.0
        assert 5.0 <= state.indoor_relative_humidity_pct <= 100.0
        assert 350.0 <= state.co2_ppm <= 5000.0
        assert abs(state.indoor_temperature_c - previous_temperature) <= 2.0
        assert transition.energy.whole_building_kwh >= 0.0


def test_invalid_inputs_are_rejected(
    model: TwoR1CBuildingModel, mild_inputs: V2ExogenousInputs
) -> None:
    with pytest.raises(ValueError, match="capacity"):
        model.step(model.initial_state(), 0, replace(mild_inputs, occupancy=101))
    with pytest.raises(ValueError, match="solar"):
        model.step(
            model.initial_state(),
            0,
            replace(mild_inputs, solar_radiation_w_per_m2=-1.0),
        )
    with pytest.raises(ValueError, match="action"):
        model.step(model.initial_state(), 4, mild_inputs)
