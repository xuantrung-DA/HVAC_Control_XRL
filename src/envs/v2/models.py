"""Typed states and transition records for the V2 building simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any


class DoorState(IntEnum):
    CLOSED = 0
    OPEN = 1
    LEFT_OPEN = 2


@dataclass(frozen=True)
class V2BuildingState:
    indoor_temperature_c: float
    indoor_relative_humidity_pct: float
    co2_ppm: float
    delivered_cooling_kw: float
    hvac_action: int
    step: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class V2ExogenousInputs:
    outdoor_temperature_c: float
    outdoor_relative_humidity_pct: float
    solar_radiation_w_per_m2: float
    occupancy: int
    electricity_price_per_kwh: float
    hour: float
    door_state: DoorState = DoorState.CLOSED
    desktop_count: int = 0
    laptop_count: int = 0
    monitor_count: int = 0
    lighting_fraction: float = 0.0
    other_electronics_fraction: float = 1.0
    electronics_load_multiplier: float = 1.0
    cleaning_equipment_on: bool = False


@dataclass(frozen=True)
class HeatFlowBreakdown:
    opaque_envelope_kw: float
    windows_kw: float
    infiltration_kw: float
    ventilation_kw: float
    solar_kw: float
    occupants_kw: float
    electronics_kw: float
    lighting_kw: float
    base_load_kw: float
    cleaning_equipment_kw: float
    hvac_cooling_kw: float
    net_heat_kw: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class EnergyBreakdown:
    hvac_cooling_kwh: float
    ventilation_fan_kwh: float
    lighting_kwh: float
    electronics_kwh: float
    base_building_kwh: float
    cleaning_equipment_kwh: float
    whole_building_kwh: float
    controllable_hvac_ventilation_kwh: float
    interval_peak_power_kw: float
    electricity_cost: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class AirQualityBreakdown:
    total_air_changes_per_hour: float
    infiltration_air_changes_per_hour: float
    ventilation_air_changes_per_hour: float
    occupant_co2_generation_ppm: float
    occupant_moisture_generation_kg: float
    hvac_dehumidification_kg: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class V2Transition:
    action: int
    action_name: str
    commanded_cooling_kw: float
    previous_delivered_cooling_kw: float
    delivered_cooling_kw: float
    effective_cooling_kw: float
    coefficient_of_performance: float
    temperature_change_c: float
    heat_flows: HeatFlowBreakdown
    energy: EnergyBreakdown
    air_quality: AirQualityBreakdown

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["heat_flows"] = self.heat_flows.as_dict()
        result["energy"] = self.energy.as_dict()
        result["air_quality"] = self.air_quality.as_dict()
        return result
