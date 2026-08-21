"""Intervention-based sanity validation for the V2 physics core."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import psutil

from src.envs.v2 import (
    DoorState,
    TwoR1CBuildingModel,
    V2BuildingState,
    V2ExogenousInputs,
)
from src.utils.config import PROJECT_ROOT
from src.utils.v2_manifest import file_sha256, load_yaml


@dataclass(frozen=True)
class ValidationCase:
    name: str
    passed: bool
    requirement: str
    measurements: dict[str, float | int | bool]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_validation_report(
    *,
    project_root: Path = PROJECT_ROOT,
    performance_steps: int = 100_000,
) -> dict[str, Any]:
    environment_path = project_root / "configs/v2/environment.yaml"
    action_path = project_root / "configs/v2/action_mapping.yaml"
    model = TwoR1CBuildingModel(
        load_yaml(environment_path), load_yaml(action_path)
    )
    state = model.initial_state()
    mild = V2ExogenousInputs(
        outdoor_temperature_c=30.0,
        outdoor_relative_humidity_pct=60.0,
        solar_radiation_w_per_m2=0.0,
        occupancy=0,
        electricity_price_per_kwh=0.10,
        hour=0.0,
    )
    cases: list[ValidationCase] = []

    low_occupancy_state, low_occupancy = model.step(
        state, 0, replace(mild, occupancy=5)
    )
    high_occupancy_state, high_occupancy = model.step(
        state, 0, replace(mild, occupancy=50)
    )
    cases.append(
        ValidationCase(
            name="occupancy_heat_co2_moisture",
            passed=(
                high_occupancy.heat_flows.occupants_kw
                > low_occupancy.heat_flows.occupants_kw
                and high_occupancy_state.indoor_temperature_c
                > low_occupancy_state.indoor_temperature_c
                and high_occupancy_state.co2_ppm > low_occupancy_state.co2_ppm
                and high_occupancy_state.indoor_relative_humidity_pct
                > low_occupancy_state.indoor_relative_humidity_pct
            ),
            requirement="More occupants increase sensible heat, CO2, and moisture.",
            measurements={
                "heat_kw_at_5": low_occupancy.heat_flows.occupants_kw,
                "heat_kw_at_50": high_occupancy.heat_flows.occupants_kw,
                "temperature_delta_c": high_occupancy_state.indoor_temperature_c
                - low_occupancy_state.indoor_temperature_c,
                "co2_delta_ppm": high_occupancy_state.co2_ppm
                - low_occupancy_state.co2_ppm,
                "humidity_delta_pct": high_occupancy_state.indoor_relative_humidity_pct
                - low_occupancy_state.indoor_relative_humidity_pct,
            },
        )
    )

    low_electronics_state, low_electronics = model.step(state, 0, mild)
    high_load_inputs = replace(
        mild,
        desktop_count=40,
        laptop_count=20,
        monitor_count=50,
        lighting_fraction=1.0,
    )
    high_electronics_state, high_electronics = model.step(
        state, 0, high_load_inputs
    )
    cases.append(
        ValidationCase(
            name="internal_electrical_gains",
            passed=(
                high_electronics.heat_flows.electronics_kw
                > low_electronics.heat_flows.electronics_kw
                and high_electronics_state.indoor_temperature_c
                > low_electronics_state.indoor_temperature_c
                and high_electronics.energy.whole_building_kwh
                > low_electronics.energy.whole_building_kwh
            ),
            requirement="Electronics and lighting increase heat and electricity use.",
            measurements={
                "additional_internal_heat_kw": (
                    high_electronics.heat_flows.electronics_kw
                    + high_electronics.heat_flows.lighting_kw
                    - low_electronics.heat_flows.electronics_kw
                    - low_electronics.heat_flows.lighting_kw
                ),
                "temperature_delta_c": high_electronics_state.indoor_temperature_c
                - low_electronics_state.indoor_temperature_c,
                "whole_building_energy_delta_kwh": high_electronics.energy.whole_building_kwh
                - low_electronics.energy.whole_building_kwh,
            },
        )
    )

    _, temperate_outdoor = model.step(
        state, 0, replace(mild, outdoor_temperature_c=24.0)
    )
    _, hot_outdoor = model.step(
        state, 0, replace(mild, outdoor_temperature_c=40.0)
    )
    temperate_envelope = (
        temperate_outdoor.heat_flows.opaque_envelope_kw
        + temperate_outdoor.heat_flows.windows_kw
    )
    hot_envelope = (
        hot_outdoor.heat_flows.opaque_envelope_kw
        + hot_outdoor.heat_flows.windows_kw
    )
    cases.append(
        ValidationCase(
            name="outdoor_envelope_direction",
            passed=hot_envelope > temperate_envelope,
            requirement="Hotter outdoor air increases envelope heat gain.",
            measurements={
                "envelope_kw_at_24c": temperate_envelope,
                "envelope_kw_at_40c": hot_envelope,
            },
        )
    )

    hot = replace(mild, outdoor_temperature_c=40.0)
    door_transitions = {
        door.name: model.step(state, 0, replace(hot, door_state=door))[1]
        for door in DoorState
    }
    closed_infiltration = door_transitions["CLOSED"].heat_flows.infiltration_kw
    open_infiltration = door_transitions["OPEN"].heat_flows.infiltration_kw
    left_open_infiltration = door_transitions[
        "LEFT_OPEN"
    ].heat_flows.infiltration_kw
    cases.append(
        ValidationCase(
            name="door_infiltration_duration",
            passed=left_open_infiltration
            > open_infiltration
            > closed_infiltration,
            requirement="Longer/higher door opening category increases hot-air gain.",
            measurements={
                "closed_infiltration_kw": closed_infiltration,
                "open_infiltration_kw": open_infiltration,
                "left_open_infiltration_kw": left_open_infiltration,
            },
        )
    )

    high_co2_state = replace(state, co2_ppm=1400.0)
    off_state, off_transition = model.step(high_co2_state, 0, hot)
    ventilated_state, ventilated_transition = model.step(
        high_co2_state, 3, hot
    )
    cases.append(
        ValidationCase(
            name="ventilation_co2_thermal_tradeoff",
            passed=(
                ventilated_state.co2_ppm < off_state.co2_ppm
                and ventilated_transition.heat_flows.ventilation_kw
                > off_transition.heat_flows.ventilation_kw
            ),
            requirement="Ventilation lowers CO2 but imports heat in hot weather.",
            measurements={
                "off_co2_ppm": off_state.co2_ppm,
                "ventilated_co2_ppm": ventilated_state.co2_ppm,
                "ventilation_heat_kw": ventilated_transition.heat_flows.ventilation_kw,
            },
        )
    )

    _, night = model.step(state, 0, replace(mild, solar_radiation_w_per_m2=0.0))
    _, noon = model.step(state, 0, replace(mild, solar_radiation_w_per_m2=800.0))
    cases.append(
        ValidationCase(
            name="solar_day_night",
            passed=night.heat_flows.solar_kw == 0.0
            and noon.heat_flows.solar_kw > 0.0,
            requirement="Night solar is zero and daytime solar is positive.",
            measurements={
                "night_solar_kw": night.heat_flows.solar_kw,
                "noon_solar_kw": noon.heat_flows.solar_kw,
            },
        )
    )

    action_transitions = [model.step(state, action, mild)[1] for action in range(4)]
    action_energy = [
        item.energy.controllable_hvac_ventilation_kwh
        for item in action_transitions
    ]
    cases.append(
        ValidationCase(
            name="hvac_action_energy_ordering",
            passed=action_energy[3]
            >= action_energy[2]
            >= action_energy[1]
            >= action_energy[0]
            == 0.0,
            requirement="HIGH consumes at least MEDIUM, LOW, and OFF.",
            measurements={
                f"action_{index}_controllable_kwh": value
                for index, value in enumerate(action_energy)
            },
        )
    )

    inertia_state = state
    delivered_curve: list[float] = []
    command_kw = 0.0
    temperature_changes: list[float] = []
    for _ in range(6):
        inertia_state, transition = model.step(inertia_state, 3, mild)
        command_kw = transition.commanded_cooling_kw
        delivered_curve.append(transition.delivered_cooling_kw)
        temperature_changes.append(abs(transition.temperature_change_c))
    monotonic = all(
        later > earlier
        for earlier, later in zip(delivered_curve, delivered_curve[1:])
    )
    cases.append(
        ValidationCase(
            name="hvac_inertia_and_gradual_temperature",
            passed=(
                0.0 < delivered_curve[0] < command_kw
                and monotonic
                and max(temperature_changes) < 1.0
            ),
            requirement="HVAC ramps toward command and temperature changes gradually.",
            measurements={
                "command_kw": command_kw,
                "first_delivered_kw": delivered_curve[0],
                "sixth_delivered_kw": delivered_curve[-1],
                "maximum_temperature_change_c": max(temperature_changes),
            },
        )
    )

    stress_state = V2BuildingState(
        indoor_temperature_c=44.0,
        indoor_relative_humidity_pct=98.0,
        co2_ppm=4900.0,
        delivered_cooling_kw=0.0,
        hvac_action=0,
        step=0,
    )
    stress_inputs = replace(
        mild,
        outdoor_temperature_c=45.0,
        outdoor_relative_humidity_pct=100.0,
        solar_radiation_w_per_m2=1000.0,
        occupancy=100,
        door_state=DoorState.LEFT_OPEN,
        desktop_count=100,
        monitor_count=100,
        lighting_fraction=1.0,
    )
    bounds_valid = True
    maximum_observed_delta = 0.0
    minimum_energy = float("inf")
    for _ in range(96):
        previous_temperature = stress_state.indoor_temperature_c
        stress_state, stress_transition = model.step(stress_state, 0, stress_inputs)
        observed_delta = abs(
            stress_state.indoor_temperature_c - previous_temperature
        )
        maximum_observed_delta = max(maximum_observed_delta, observed_delta)
        minimum_energy = min(
            minimum_energy, stress_transition.energy.whole_building_kwh
        )
        bounds_valid = bounds_valid and (
            5.0 <= stress_state.indoor_temperature_c <= 45.0
            and 5.0 <= stress_state.indoor_relative_humidity_pct <= 100.0
            and 350.0 <= stress_state.co2_ppm <= 5000.0
            and observed_delta <= 2.0
            and stress_transition.energy.whole_building_kwh >= 0.0
        )
    cases.append(
        ValidationCase(
            name="physical_bounds_stress_rollout",
            passed=bounds_valid,
            requirement="State bounds and non-negative energy hold under 24h stress.",
            measurements={
                "steps": 96,
                "maximum_temperature_change_c": maximum_observed_delta,
                "minimum_whole_building_energy_kwh": minimum_energy,
                "final_temperature_c": stress_state.indoor_temperature_c,
                "final_humidity_pct": stress_state.indoor_relative_humidity_pct,
                "final_co2_ppm": stress_state.co2_ppm,
            },
        )
    )

    performance = _profile_throughput(model, mild, performance_steps)
    return {
        "report_version": 1,
        "simulator_version": "XRL-HVAC-v2",
        "thermal_network": "2R1C",
        "configuration_sha256": file_sha256(environment_path),
        "action_mapping_sha256": file_sha256(action_path),
        "physics_source_sha256": file_sha256(
            project_root / "src/envs/v2/physics.py"
        ),
        "all_checks_passed": all(case.passed for case in cases),
        "checks_passed": sum(case.passed for case in cases),
        "checks_total": len(cases),
        "cases": [case.as_dict() for case in cases],
        "performance": performance,
        "training_authorized": all(case.passed for case in cases),
        "limitations": [
            "2R1C uses one lumped zone capacitance and does not resolve individual wall layers.",
            "Synthetic weather/events are validated in the next milestone.",
            "Parameters are engineering approximations and are not calibrated to a real building.",
            "The simulator is suitable for controlled RL experiments, not equipment sizing.",
        ],
    }


def _profile_throughput(
    model: TwoR1CBuildingModel,
    inputs: V2ExogenousInputs,
    steps: int,
) -> dict[str, float | int]:
    process = psutil.Process()
    memory_before = process.memory_info().rss
    state = model.initial_state()
    started = time.perf_counter()
    for index in range(steps):
        state, _ = model.step(state, index % 4, inputs)
    duration = time.perf_counter() - started
    memory_after = process.memory_info().rss
    return {
        "steps": steps,
        "duration_seconds": duration,
        "steps_per_second": steps / max(duration, 1e-9),
        "rss_before_mb": memory_before / (1024**2),
        "rss_after_mb": memory_after / (1024**2),
        "rss_delta_mb": (memory_after - memory_before) / (1024**2),
    }
