"""Explainable 2R1C thermal, humidity, CO2, and energy model for V2."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from src.envs.building import HVACAction
from src.envs.v2.models import (
    AirQualityBreakdown,
    DoorState,
    EnergyBreakdown,
    HeatFlowBreakdown,
    V2BuildingState,
    V2ExogenousInputs,
    V2Transition,
)
from src.envs.v2.psychrometrics import humidity_ratio, relative_humidity_pct


class TwoR1CBuildingModel:
    """One lumped thermal capacitance with opaque and window resistances."""

    def __init__(
        self,
        environment_config: Mapping[str, Any],
        action_config: Mapping[str, Any],
    ) -> None:
        self.config = environment_config
        self.action_config = action_config["action_space"]["actions"]
        self.dt_hours = (
            float(environment_config["simulation"]["timestep_minutes"]) / 60.0
        )
        self.zone = environment_config["zone"]
        self.envelope = environment_config["envelope"]
        self.internal = environment_config["internal_gains"]
        self.airflow = environment_config["airflow"]
        self.hvac = environment_config["hvac"]
        self.loads = environment_config["building_loads"]
        self.bounds = environment_config["physical_bounds"]
        self._validate_config()

    def initial_state(self) -> V2BuildingState:
        return V2BuildingState(
            indoor_temperature_c=float(
                self.zone["initial_indoor_temperature_c"]
            ),
            indoor_relative_humidity_pct=float(
                self.zone["initial_relative_humidity_pct"]
            ),
            co2_ppm=float(self.zone["initial_co2_ppm"]),
            delivered_cooling_kw=0.0,
            hvac_action=int(HVACAction.OFF),
            step=0,
        )

    def step(
        self,
        state: V2BuildingState,
        action: int,
        inputs: V2ExogenousInputs,
    ) -> tuple[V2BuildingState, V2Transition]:
        if action not in range(len(HVACAction)):
            raise ValueError("V2 HVAC action must be one of 0, 1, 2, 3")
        self._validate_inputs(inputs)
        action_settings = self._action(action)
        command_kw = (
            float(action_settings["cooling_command_fraction"])
            * float(self.hvac["maximum_delivered_cooling_kw"])
        )
        delivered_kw = self._delivered_cooling(
            state.delivered_cooling_kw, command_kw
        )
        effective_cooling_kw = (state.delivered_cooling_kw + delivered_kw) / 2.0

        infiltration_ach = self._infiltration_ach(inputs.door_state)
        ventilation_ach = float(
            action_settings["mechanical_ventilation_ach"]
        )
        heat_flows, electrical_powers = self._heat_flows(
            state,
            inputs,
            effective_cooling_kw,
            infiltration_ach,
            ventilation_ach,
        )
        raw_temperature_delta = (
            self.dt_hours
            * heat_flows.net_heat_kw
            / float(self.envelope["effective_thermal_capacity_kwh_per_c"])
        )
        maximum_delta = float(
            self.bounds["maximum_temperature_change_c_per_step"]
        )
        temperature_delta = float(
            np.clip(raw_temperature_delta, -maximum_delta, maximum_delta)
        )
        temperature_bounds = self.bounds["indoor_temperature_c"]
        next_temperature = float(
            np.clip(
                state.indoor_temperature_c + temperature_delta,
                float(temperature_bounds[0]),
                float(temperature_bounds[1]),
            )
        )

        next_co2, co2_generation_ppm = self._next_co2(
            state.co2_ppm,
            inputs.occupancy,
            infiltration_ach + ventilation_ach,
        )
        next_humidity, moisture_kg, dehumidification_kg = self._next_humidity(
            state,
            next_temperature,
            inputs,
            infiltration_ach + ventilation_ach,
            effective_cooling_kw,
        )
        cop = self._coefficient_of_performance(
            action, inputs.outdoor_temperature_c
        )
        energy = self._energy_breakdown(
            inputs,
            effective_cooling_kw,
            cop,
            ventilation_ach,
            electrical_powers,
        )
        next_state = V2BuildingState(
            indoor_temperature_c=next_temperature,
            indoor_relative_humidity_pct=next_humidity,
            co2_ppm=next_co2,
            delivered_cooling_kw=delivered_kw,
            hvac_action=action,
            step=state.step + 1,
        )
        transition = V2Transition(
            action=action,
            action_name=HVACAction(action).name,
            commanded_cooling_kw=command_kw,
            previous_delivered_cooling_kw=state.delivered_cooling_kw,
            delivered_cooling_kw=delivered_kw,
            effective_cooling_kw=effective_cooling_kw,
            coefficient_of_performance=cop,
            temperature_change_c=next_temperature
            - state.indoor_temperature_c,
            heat_flows=heat_flows,
            energy=energy,
            air_quality=AirQualityBreakdown(
                total_air_changes_per_hour=infiltration_ach + ventilation_ach,
                infiltration_air_changes_per_hour=infiltration_ach,
                ventilation_air_changes_per_hour=ventilation_ach,
                occupant_co2_generation_ppm=co2_generation_ppm,
                occupant_moisture_generation_kg=moisture_kg,
                hvac_dehumidification_kg=dehumidification_kg,
            ),
        )
        return next_state, transition

    def _heat_flows(
        self,
        state: V2BuildingState,
        inputs: V2ExogenousInputs,
        effective_cooling_kw: float,
        infiltration_ach: float,
        ventilation_ach: float,
    ) -> tuple[HeatFlowBreakdown, dict[str, float]]:
        delta_temperature = (
            inputs.outdoor_temperature_c - state.indoor_temperature_c
        )
        opaque_kw = delta_temperature / float(
            self.envelope["opaque_resistance_c_per_kw"]
        )
        windows_kw = delta_temperature / float(
            self.envelope["window_resistance_c_per_kw"]
        )
        infiltration_kw = self._air_exchange_heat_kw(
            infiltration_ach, delta_temperature
        )
        ventilation_kw = self._air_exchange_heat_kw(
            ventilation_ach, delta_temperature
        )
        solar_kw = (
            inputs.solar_radiation_w_per_m2
            * float(self.envelope["window_area_m2"])
            * float(self.envelope["solar_heat_gain_coefficient"])
            / 1000.0
        )
        occupants_kw = inputs.occupancy * float(
            self.internal["occupant_sensible_kw_per_person"]
        )
        electronics_power_kw = self._electronics_power_kw(inputs)
        lighting_power_kw = self._lighting_power_kw(inputs)
        heat_fraction = float(self.internal["heat_fraction_of_electric_load"])
        electronics_heat_kw = electronics_power_kw * heat_fraction
        lighting_heat_kw = lighting_power_kw * heat_fraction
        base_load_heat_kw = (
            float(self.loads["network_and_server_power_kw"]) * heat_fraction
        )
        cleaning_heat_kw = (
            float(self.loads["cleaning_equipment_power_kw"]) * heat_fraction
            if inputs.cleaning_equipment_on
            else 0.0
        )
        net_heat_kw = (
            opaque_kw
            + windows_kw
            + infiltration_kw
            + ventilation_kw
            + solar_kw
            + occupants_kw
            + electronics_heat_kw
            + lighting_heat_kw
            + base_load_heat_kw
            + cleaning_heat_kw
            - effective_cooling_kw
        )
        return (
            HeatFlowBreakdown(
                opaque_envelope_kw=float(opaque_kw),
                windows_kw=float(windows_kw),
                infiltration_kw=float(infiltration_kw),
                ventilation_kw=float(ventilation_kw),
                solar_kw=float(solar_kw),
                occupants_kw=float(occupants_kw),
                electronics_kw=float(electronics_heat_kw),
                lighting_kw=float(lighting_heat_kw),
                base_load_kw=float(base_load_heat_kw),
                cleaning_equipment_kw=float(cleaning_heat_kw),
                hvac_cooling_kw=float(effective_cooling_kw),
                net_heat_kw=float(net_heat_kw),
            ),
            {
                "electronics_kw": electronics_power_kw,
                "lighting_kw": lighting_power_kw,
            },
        )

    def _next_co2(
        self, current_ppm: float, occupancy: int, total_ach: float
    ) -> tuple[float, float]:
        outdoor_ppm = float(self.airflow["outdoor_co2_ppm"])
        exchanged_ppm = outdoor_ppm + (current_ppm - outdoor_ppm) * math.exp(
            -total_ach * self.dt_hours
        )
        generated_liters = (
            occupancy
            * float(self.airflow["co2_generation_liters_per_hour_person"])
            * self.dt_hours
        )
        generation_ppm = (
            generated_liters
            / (float(self.zone["air_volume_m3"]) * 1000.0)
            * 1_000_000.0
        )
        bounds = self.bounds["co2_ppm"]
        return (
            float(
                np.clip(
                    exchanged_ppm + generation_ppm,
                    float(bounds[0]),
                    float(bounds[1]),
                )
            ),
            float(generation_ppm),
        )

    def _next_humidity(
        self,
        state: V2BuildingState,
        next_temperature_c: float,
        inputs: V2ExogenousInputs,
        total_ach: float,
        effective_cooling_kw: float,
    ) -> tuple[float, float, float]:
        indoor_ratio = humidity_ratio(
            state.indoor_temperature_c,
            state.indoor_relative_humidity_pct,
        )
        outdoor_ratio = humidity_ratio(
            inputs.outdoor_temperature_c,
            inputs.outdoor_relative_humidity_pct,
        )
        dry_air_mass_kg = (
            float(self.airflow["air_density_kg_per_m3"])
            * float(self.zone["air_volume_m3"])
        )
        exchanged_ratio = outdoor_ratio + (indoor_ratio - outdoor_ratio) * math.exp(
            -total_ach * self.dt_hours
        )
        occupant_moisture_kg = (
            inputs.occupancy
            * float(self.internal["occupant_latent_kg_per_hour_person"])
            * self.dt_hours
        )
        dehumidification_kg = min(
            effective_cooling_kw
            * self.dt_hours
            * float(self.hvac["dehumidification_kg_per_kwh_cooling"]),
            max(0.0, exchanged_ratio * dry_air_mass_kg + occupant_moisture_kg),
        )
        next_ratio = exchanged_ratio + (
            occupant_moisture_kg - dehumidification_kg
        ) / dry_air_mass_kg
        next_relative_humidity = relative_humidity_pct(
            next_temperature_c, next_ratio
        )
        bounds = self.bounds["relative_humidity_pct"]
        return (
            float(
                np.clip(
                    next_relative_humidity,
                    float(bounds[0]),
                    float(bounds[1]),
                )
            ),
            float(occupant_moisture_kg),
            float(dehumidification_kg),
        )

    def _energy_breakdown(
        self,
        inputs: V2ExogenousInputs,
        effective_cooling_kw: float,
        cop: float,
        ventilation_ach: float,
        electrical_powers: Mapping[str, float],
    ) -> EnergyBreakdown:
        compressor_power_kw = effective_cooling_kw / cop
        maximum_ventilation_ach = max(
            float(self._action(index)["mechanical_ventilation_ach"])
            for index in range(len(HVACAction))
        )
        fan_fraction = (
            ventilation_ach / maximum_ventilation_ach
            if maximum_ventilation_ach > 0
            else 0.0
        )
        fan_power_kw = float(self.hvac["fan_rated_power_kw"]) * fan_fraction**3
        lighting_power_kw = float(electrical_powers["lighting_kw"])
        electronics_power_kw = float(electrical_powers["electronics_kw"])
        base_power_kw = float(self.loads["base_power_kw"]) + float(
            self.loads["network_and_server_power_kw"]
        )
        cleaning_power_kw = (
            float(self.loads["cleaning_equipment_power_kw"])
            if inputs.cleaning_equipment_on
            else 0.0
        )
        total_power_kw = (
            compressor_power_kw
            + fan_power_kw
            + lighting_power_kw
            + electronics_power_kw
            + base_power_kw
            + cleaning_power_kw
        )
        dt = self.dt_hours
        hvac_energy = compressor_power_kw * dt
        fan_energy = fan_power_kw * dt
        lighting_energy = lighting_power_kw * dt
        electronics_energy = electronics_power_kw * dt
        base_energy = base_power_kw * dt
        cleaning_energy = cleaning_power_kw * dt
        whole_energy = total_power_kw * dt
        return EnergyBreakdown(
            hvac_cooling_kwh=float(hvac_energy),
            ventilation_fan_kwh=float(fan_energy),
            lighting_kwh=float(lighting_energy),
            electronics_kwh=float(electronics_energy),
            base_building_kwh=float(base_energy),
            cleaning_equipment_kwh=float(cleaning_energy),
            whole_building_kwh=float(whole_energy),
            controllable_hvac_ventilation_kwh=float(hvac_energy + fan_energy),
            interval_peak_power_kw=float(total_power_kw),
            electricity_cost=float(
                whole_energy * inputs.electricity_price_per_kwh
            ),
        )

    def _delivered_cooling(self, previous_kw: float, command_kw: float) -> float:
        time_constant_hours = (
            float(self.hvac["cooling_time_constant_minutes"]) / 60.0
        )
        alpha = 1.0 - math.exp(-self.dt_hours / time_constant_hours)
        return float(previous_kw + alpha * (command_kw - previous_kw))

    def _coefficient_of_performance(
        self, action: int, outdoor_temperature_c: float
    ) -> float:
        outdoor_multiplier = float(
            np.clip(1.0 - 0.015 * max(outdoor_temperature_c - 30.0, 0.0), 0.65, 1.05)
        )
        part_load_multiplier = {
            int(HVACAction.OFF): 1.0,
            int(HVACAction.LOW): 1.05,
            int(HVACAction.MEDIUM): 1.0,
            int(HVACAction.HIGH): float(
                self.hvac["high_action_cop_multiplier"]
            ),
        }[action]
        return max(
            1.0,
            float(self.hvac["rated_cop"])
            * outdoor_multiplier
            * part_load_multiplier,
        )

    def _air_exchange_heat_kw(self, ach: float, delta_temperature_c: float) -> float:
        mass_flow_kg_per_second = (
            float(self.airflow["air_density_kg_per_m3"])
            * float(self.zone["air_volume_m3"])
            * ach
            / 3600.0
        )
        return (
            mass_flow_kg_per_second
            * float(self.airflow["air_specific_heat_kj_per_kg_c"])
            * delta_temperature_c
        )

    def _electronics_power_kw(self, inputs: V2ExogenousInputs) -> float:
        connected_power_kw = (
            inputs.desktop_count * float(self.internal["desktop_kw_each"])
            + inputs.laptop_count * float(self.internal["laptop_kw_each"])
            + inputs.monitor_count * float(self.internal["monitor_kw_each"])
            + float(self.internal["other_electronics_kw"])
            * inputs.other_electronics_fraction
        )
        return connected_power_kw * inputs.electronics_load_multiplier

    def _lighting_power_kw(self, inputs: V2ExogenousInputs) -> float:
        rated_kw = (
            float(self.internal["lighting_w_per_m2"])
            * float(self.zone["floor_area_m2"])
            / 1000.0
        )
        return rated_kw * inputs.lighting_fraction

    def _infiltration_ach(self, door_state: DoorState) -> float:
        key = {
            DoorState.CLOSED: "closed_door_infiltration_ach",
            DoorState.OPEN: "open_door_infiltration_ach",
            DoorState.LEFT_OPEN: "left_open_door_infiltration_ach",
        }[DoorState(door_state)]
        return float(self.airflow[key])

    def _action(self, action: int) -> Mapping[str, Any]:
        return self.action_config.get(action, self.action_config.get(str(action)))

    def _validate_inputs(self, inputs: V2ExogenousInputs) -> None:
        if inputs.occupancy < 0 or inputs.occupancy > int(
            self.zone["maximum_occupancy"]
        ):
            raise ValueError("occupancy is outside the configured zone capacity")
        if not 0.0 <= inputs.outdoor_relative_humidity_pct <= 100.0:
            raise ValueError("outdoor relative humidity must be in [0, 100]")
        if inputs.solar_radiation_w_per_m2 < 0.0:
            raise ValueError("solar radiation cannot be negative")
        for value, label in (
            (inputs.lighting_fraction, "lighting_fraction"),
            (inputs.other_electronics_fraction, "other_electronics_fraction"),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be in [0, 1]")
        if not 0.0 <= inputs.electronics_load_multiplier <= 3.0:
            raise ValueError("electronics_load_multiplier must be in [0, 3]")

    def _validate_config(self) -> None:
        if self.dt_hours <= 0:
            raise ValueError("V2 timestep must be positive")
        for key in (
            "opaque_resistance_c_per_kw",
            "window_resistance_c_per_kw",
            "effective_thermal_capacity_kwh_per_c",
        ):
            if float(self.envelope[key]) <= 0:
                raise ValueError(f"envelope.{key} must be positive")
        if float(self.hvac["cooling_time_constant_minutes"]) <= 0:
            raise ValueError("HVAC cooling time constant must be positive")
