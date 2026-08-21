"""Physical sanity tests for thermal and CO₂ dynamics."""

from __future__ import annotations

from src.envs.thermal_model import CO2Model, ThermalModel
from src.utils.config import load_environment_config


def test_hot_outdoor_air_and_occupants_add_heat() -> None:
    config = load_environment_config()
    model = ThermalModel(config["building"], config["hvac"])

    empty = model.step(
        indoor_temperature_c=24.0,
        outdoor_temperature_c=35.0,
        occupancy=0,
        action=0,
        solar_heat_kw=0.0,
        timestep_hours=0.25,
    )
    occupied = model.step(
        indoor_temperature_c=24.0,
        outdoor_temperature_c=35.0,
        occupancy=40,
        action=0,
        solar_heat_kw=0.0,
        timestep_hours=0.25,
    )

    assert empty.indoor_temperature_c > 24.0
    assert occupied.indoor_temperature_c > empty.indoor_temperature_c


def test_high_hvac_produces_lower_temperature_than_off() -> None:
    config = load_environment_config()
    model = ThermalModel(config["building"], config["hvac"])
    common = {
        "indoor_temperature_c": 27.0,
        "outdoor_temperature_c": 36.0,
        "occupancy": 30,
        "solar_heat_kw": 2.0,
        "timestep_hours": 0.25,
    }

    off = model.step(action=0, **common)
    high = model.step(action=3, **common)

    assert high.indoor_temperature_c < off.indoor_temperature_c
    assert high.hvac_cooling_kw > off.hvac_cooling_kw


def test_occupancy_accumulates_co2_and_ventilation_removes_it() -> None:
    config = load_environment_config()
    model = CO2Model(config["iaq"])

    occupied_off = model.step(
        current_co2_ppm=900.0,
        occupancy=40,
        action=0,
        timestep_hours=0.25,
    )
    occupied_high = model.step(
        current_co2_ppm=900.0,
        occupancy=40,
        action=3,
        timestep_hours=0.25,
    )

    assert occupied_off > 900.0
    assert occupied_high < occupied_off


def test_unoccupied_ventilation_moves_co2_toward_outdoor_level() -> None:
    config = load_environment_config()
    model = CO2Model(config["iaq"])
    next_co2 = model.step(
        current_co2_ppm=1500.0,
        occupancy=0,
        action=3,
        timestep_hours=0.25,
    )
    assert config["iaq"]["outdoor_co2_ppm"] < next_co2 < 1500.0
