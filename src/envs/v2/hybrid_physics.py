"""Three-actuator V2 physics with explicit latent-energy accounting."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from src.envs.building import HVACAction
from src.envs.v2.models import V2BuildingState, V2ExogenousInputs, V2Transition
from src.envs.v2.physics import TwoR1CBuildingModel
from src.envs.v2.psychrometrics import humidity_ratio, relative_humidity_pct


class HybridBuildingModel(TwoR1CBuildingModel):
    """Decouple dehumidification from sensible cooling and meter it separately."""

    def __init__(
        self,
        environment_config: Mapping[str, Any],
        action_config: Mapping[str, Any],
        hybrid_config: Mapping[str, Any],
    ) -> None:
        super().__init__(environment_config, action_config)
        settings = hybrid_config["hybrid_control"]["dehumidifier"]
        self.dehumidifier_capacity_kg_per_hour = float(
            settings["rated_capacity_kg_per_hour"]
        )
        self.dehumidifier_power_kw = float(settings["rated_power_kw"])
        self.dehumidifier_target_rh_pct = float(
            settings["target_relative_humidity_pct"]
        )
        self._dehumidification_fraction = 0.0
        self._independent_removed_kg = 0.0

    def step_hybrid(
        self,
        state: V2BuildingState,
        *,
        cooling_action: int,
        cooling_fraction: float,
        ventilation_fraction: float,
        dehumidification_fraction: float,
        inputs: V2ExogenousInputs,
    ) -> tuple[V2BuildingState, V2Transition]:
        if cooling_action not in range(len(HVACAction)):
            raise ValueError("Hybrid cooling action must be one of 0, 1, 2, 3")
        for name, value in (
            ("cooling", cooling_fraction),
            ("ventilation", ventilation_fraction),
            ("dehumidification", dehumidification_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Hybrid {name} fraction must be in [0, 1]")
        self._dehumidification_fraction = float(dehumidification_fraction)
        self._independent_removed_kg = 0.0
        next_state, transition = self._step_commands(
            state=state,
            action_code=cooling_action,
            action_name=f"HYBRID_{HVACAction(cooling_action).name}",
            cooling_fraction=float(cooling_fraction),
            ventilation_fraction=float(ventilation_fraction),
            continuous_control=True,
            inputs=inputs,
        )
        transition = replace(
            transition,
            air_quality=replace(
                transition.air_quality,
                independent_dehumidification_kg=self._independent_removed_kg,
            ),
        )
        return next_state, transition

    def _next_humidity(
        self,
        state,
        next_temperature_c,
        inputs,
        total_ach,
        effective_cooling_kw,
        sensible_heat_ratio,
    ):
        humidity, moisture, cooling_removed = super()._next_humidity(
            state,
            next_temperature_c,
            inputs,
            total_ach,
            effective_cooling_kw,
            sensible_heat_ratio,
        )
        current_ratio = humidity_ratio(next_temperature_c, humidity)
        target_ratio = humidity_ratio(
            next_temperature_c, self.dehumidifier_target_rh_pct
        )
        dry_air_mass_kg = float(self.airflow["air_density_kg_per_m3"]) * float(
            self.zone["air_volume_m3"]
        )
        removable_kg = max((current_ratio - target_ratio) * dry_air_mass_kg, 0.0)
        independent_removed_kg = min(
            removable_kg,
            self.dehumidifier_capacity_kg_per_hour
            * self._dehumidification_fraction
            * self.dt_hours,
        )
        next_ratio = current_ratio - independent_removed_kg / dry_air_mass_kg
        bounded_humidity = float(
            np.clip(
                relative_humidity_pct(next_temperature_c, next_ratio),
                float(self.bounds["relative_humidity_pct"][0]),
                float(self.bounds["relative_humidity_pct"][1]),
            )
        )
        self._independent_removed_kg = float(independent_removed_kg)
        return bounded_humidity, moisture, cooling_removed + independent_removed_kg

    def _energy_breakdown(
        self,
        inputs,
        effective_cooling_kw,
        cop,
        ventilation_ach,
        electrical_powers,
    ):
        energy = super()._energy_breakdown(
            inputs,
            effective_cooling_kw,
            cop,
            ventilation_ach,
            electrical_powers,
        )
        dehumidifier_power_kw = (
            self.dehumidifier_power_kw * self._dehumidification_fraction
        )
        dehumidifier_kwh = dehumidifier_power_kw * self.dt_hours
        return replace(
            energy,
            dehumidification_kwh=float(dehumidifier_kwh),
            whole_building_kwh=energy.whole_building_kwh + dehumidifier_kwh,
            controllable_hvac_ventilation_kwh=(
                energy.controllable_hvac_ventilation_kwh + dehumidifier_kwh
            ),
            interval_peak_power_kw=(
                energy.interval_peak_power_kw + dehumidifier_power_kw
            ),
            electricity_cost=(
                energy.electricity_cost
                + dehumidifier_kwh * inputs.electricity_price_per_kwh
            ),
        )
