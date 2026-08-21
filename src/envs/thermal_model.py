"""Lightweight thermal and indoor-air-quality dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ThermalStep:
    """Heat-flow details for one simulation interval."""

    indoor_temperature_c: float
    envelope_heat_kw: float
    occupant_heat_kw: float
    solar_heat_kw: float
    hvac_cooling_kw: float
    net_heat_kw: float


class ThermalModel:
    """Single-zone resistance/capacitance thermal approximation."""

    def __init__(self, building_config: Mapping[str, float], hvac_config: Mapping) -> None:
        self.thermal_mass_kwh_per_c = float(
            building_config["thermal_mass_kwh_per_c"]
        )
        self.envelope_conductance_kw_per_c = float(
            building_config["envelope_conductance_kw_per_c"]
        )
        self.occupant_heat_gain_kw_per_person = float(
            building_config["occupant_heat_gain_kw_per_person"]
        )
        self.cooling_power_kw = {
            int(action): float(power)
            for action, power in hvac_config["cooling_power_kw"].items()
        }

        if self.thermal_mass_kwh_per_c <= 0:
            raise ValueError("thermal_mass_kwh_per_c must be positive")
        if self.envelope_conductance_kw_per_c < 0:
            raise ValueError("envelope_conductance_kw_per_c cannot be negative")

    def step(
        self,
        *,
        indoor_temperature_c: float,
        outdoor_temperature_c: float,
        occupancy: int,
        action: int,
        solar_heat_kw: float,
        timestep_hours: float,
    ) -> ThermalStep:
        """Advance indoor temperature by one Euler integration step."""

        if action not in self.cooling_power_kw:
            raise ValueError(f"Unsupported HVAC action: {action}")
        if timestep_hours <= 0:
            raise ValueError("timestep_hours must be positive")

        envelope_heat_kw = self.envelope_conductance_kw_per_c * (
            outdoor_temperature_c - indoor_temperature_c
        )
        occupant_heat_kw = max(0, occupancy) * self.occupant_heat_gain_kw_per_person
        hvac_cooling_kw = self.cooling_power_kw[action]
        net_heat_kw = (
            envelope_heat_kw
            + occupant_heat_kw
            + max(0.0, solar_heat_kw)
            - hvac_cooling_kw
        )
        temperature_delta_c = (
            net_heat_kw * timestep_hours / self.thermal_mass_kwh_per_c
        )
        next_temperature = float(
            np.clip(indoor_temperature_c + temperature_delta_c, -10.0, 55.0)
        )

        return ThermalStep(
            indoor_temperature_c=next_temperature,
            envelope_heat_kw=float(envelope_heat_kw),
            occupant_heat_kw=float(occupant_heat_kw),
            solar_heat_kw=float(max(0.0, solar_heat_kw)),
            hvac_cooling_kw=float(hvac_cooling_kw),
            net_heat_kw=float(net_heat_kw),
        )


class CO2Model:
    """Well-mixed single-zone CO₂ mass-balance approximation."""

    def __init__(self, iaq_config: Mapping) -> None:
        self.outdoor_co2_ppm = float(iaq_config["outdoor_co2_ppm"])
        self.generation_ppm_per_person_hour = float(
            iaq_config["co2_generation_ppm_per_person_hour"]
        )
        self.natural_air_changes_per_hour = float(
            iaq_config["natural_air_changes_per_hour"]
        )
        self.hvac_air_changes_per_hour = {
            int(action): float(rate)
            for action, rate in iaq_config["hvac_air_changes_per_hour"].items()
        }

    def step(
        self,
        *,
        current_co2_ppm: float,
        occupancy: int,
        action: int,
        timestep_hours: float,
    ) -> float:
        """Advance CO₂ using generation and first-order ventilation removal."""

        if action not in self.hvac_air_changes_per_hour:
            raise ValueError(f"Unsupported HVAC action: {action}")
        if timestep_hours <= 0:
            raise ValueError("timestep_hours must be positive")

        generation = (
            max(0, occupancy)
            * self.generation_ppm_per_person_hour
            * timestep_hours
        )
        ventilation_rate = (
            self.natural_air_changes_per_hour
            + self.hvac_air_changes_per_hour[action]
        )
        removal = (
            ventilation_rate
            * max(0.0, current_co2_ppm - self.outdoor_co2_ppm)
            * timestep_hours
        )
        return float(np.clip(current_co2_ppm + generation - removal, 350.0, 5000.0))
