"""Seeded, correlated exogenous profiles for every locked V2 scenario."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.envs.v2.models import DoorState, V2ExogenousInputs
from src.utils.config import PROJECT_ROOT, deep_merge, load_yaml


@dataclass(frozen=True)
class ScenarioTimeline:
    """A complete physical episode plus metadata safe for forecasting."""

    scenario: str
    seed: int
    inputs: tuple[V2ExogenousInputs, ...]
    event_metadata: tuple[dict[str, Any], ...]
    forecast_failure: Mapping[str, Any] | None = None

    def as_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for step, item in enumerate(self.inputs):
            record = asdict(item)
            record["step"] = step
            record["door_state"] = DoorState(item.door_state).name
            records.append(record)
        return records


class V2ScenarioGenerator:
    """Generate realistic lightweight daily profiles without independent noise."""

    def __init__(
        self,
        scenario_config: Mapping[str, Any],
        environment_config: Mapping[str, Any],
    ) -> None:
        self.config = scenario_config
        self.environment = environment_config
        self.defaults = scenario_config["defaults"]
        self.definitions = scenario_config["definitions"]
        self.steps = int(environment_config["simulation"]["steps_per_episode"])
        self.dt_hours = (
            float(environment_config["simulation"]["timestep_minutes"]) / 60.0
        )
        self.capacity = int(environment_config["zone"]["maximum_occupancy"])
        self._validate()

    @classmethod
    def from_project(cls, project_root: Path = PROJECT_ROOT) -> "V2ScenarioGenerator":
        return cls(
            load_yaml(project_root / "configs/v2/scenarios.yaml"),
            load_yaml(project_root / "configs/v2/environment.yaml"),
        )

    def generate(self, scenario: str, seed: int) -> ScenarioTimeline:
        if scenario not in self.definitions:
            available = ", ".join(sorted(self.definitions))
            raise ValueError(f"Unknown V2 scenario {scenario!r}; choose from {available}")
        profile = deep_merge(self.defaults, self.definitions[scenario])
        rng = np.random.default_rng(seed)
        hours = np.arange(self.steps, dtype=np.float64) * self.dt_hours
        temperatures, humidities, solar = self._weather(hours, profile["weather"], rng)
        occupancy = self._occupancy(hours, profile["occupancy"], profile["events"], rng)
        prices = self._prices(hours, profile["electricity_price"])
        doors, event_metadata = self._events(hours, profile["events"], rng)
        inputs: list[V2ExogenousInputs] = []
        equipment = profile["equipment"]
        multiplier = float(equipment["electronics_multiplier"])
        for index, hour in enumerate(hours):
            people = int(occupancy[index])
            occupied = people > 0
            daylight = float(solar[index]) / max(float(profile["weather"]["solar_peak_w_per_m2"]), 1.0)
            lighting_fraction = float(
                np.clip((0.22 + 0.78 * (1.0 - daylight)) if occupied else 0.06, 0.0, 1.0)
            )
            inputs.append(
                V2ExogenousInputs(
                    outdoor_temperature_c=float(temperatures[index]),
                    outdoor_relative_humidity_pct=float(humidities[index]),
                    solar_radiation_w_per_m2=float(solar[index]),
                    occupancy=people,
                    electricity_price_per_kwh=float(prices[index]),
                    hour=float(hour),
                    door_state=doors[index],
                    desktop_count=int(round(people * float(equipment["desktop_fraction_per_occupant"]))),
                    laptop_count=int(round(people * float(equipment["laptop_fraction_per_occupant"]))),
                    monitor_count=int(round(people * float(equipment["monitor_fraction_per_occupant"]))),
                    lighting_fraction=lighting_fraction,
                    other_electronics_fraction=1.0 if occupied else 0.35,
                    electronics_load_multiplier=multiplier,
                    cleaning_equipment_on=self._in_any_window(float(hour), profile["events"]["cleaning"]),
                )
            )
        forecast_failure = profile.get("forecast")
        return ScenarioTimeline(
            scenario=scenario,
            seed=seed,
            inputs=tuple(inputs),
            event_metadata=tuple(event_metadata),
            forecast_failure=forecast_failure,
        )

    def scenario_split(self, scenario: str) -> str:
        for split, names in self.config["splits"].items():
            if scenario in names:
                return split
        raise ValueError(f"Scenario {scenario!r} is not assigned to a split")

    def _weather(
        self, hours: np.ndarray, config: Mapping[str, Any], rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        minimum = float(config["temperature_min_c"])
        maximum = float(config["temperature_max_c"])
        peak_hour = float(config["temperature_peak_hour"])
        mean = (minimum + maximum) / 2.0
        amplitude = (maximum - minimum) / 2.0
        base_temperature = mean + amplitude * np.cos(2.0 * np.pi * (hours - peak_hour) / 24.0)
        ar_noise = self._ar1(rng, self.steps, float(config["autocorrelation"]), 0.18)
        temperature = base_temperature + ar_noise

        sunrise = float(config["sunrise_hour"])
        sunset = float(config["sunset_hour"])
        daylight_phase = np.clip((hours - sunrise) / (sunset - sunrise), 0.0, 1.0)
        clear_sky = np.where(
            (hours >= sunrise) & (hours <= sunset),
            np.sin(np.pi * daylight_phase),
            0.0,
        )
        cloud_noise = self._ar1(rng, self.steps, 0.93, 0.04)
        transmittance = np.clip(1.0 - float(config["cloud_fraction"]) + cloud_noise, 0.35, 1.0)
        solar = np.maximum(0.0, float(config["solar_peak_w_per_m2"]) * clear_sky * transmittance)

        humidity_min = float(config["relative_humidity_min_pct"])
        humidity_max = float(config["relative_humidity_max_pct"])
        normalized_temperature = (temperature - temperature.min()) / max(float(np.ptp(temperature)), 1e-6)
        humidity_noise = self._ar1(rng, self.steps, float(config["autocorrelation"]), 0.7)
        humidity = humidity_max - (humidity_max - humidity_min) * normalized_temperature + humidity_noise
        return temperature, np.clip(humidity, 5.0, 100.0), solar

    def _occupancy(
        self,
        hours: np.ndarray,
        config: Mapping[str, Any],
        events: Mapping[str, Any],
        rng: np.random.Generator,
    ) -> np.ndarray:
        peak = self.capacity * float(config["peak_fraction_of_capacity"])
        fraction = np.zeros(self.steps, dtype=np.float64)
        arrival_start = float(config["arrival_start_hour"])
        arrival_end = float(config["arrival_end_hour"])
        departure_start = float(config["departure_start_hour"])
        departure_end = float(config["departure_end_hour"])
        arriving = (hours >= arrival_start) & (hours < arrival_end)
        fraction[arriving] = (hours[arriving] - arrival_start) / (arrival_end - arrival_start)
        fraction[(hours >= arrival_end) & (hours < departure_start)] = 1.0
        departing = (hours >= departure_start) & (hours < departure_end)
        fraction[departing] = 1.0 - (hours[departing] - departure_start) / (departure_end - departure_start)
        lunch = (hours >= float(config["lunch_start_hour"])) & (hours < float(config["lunch_end_hour"]))
        fraction[lunch] *= float(config["lunch_fraction"])
        noise = self._ar1(rng, self.steps, 0.78, float(config["stochastic_fraction"]))
        people = peak * np.clip(fraction + noise * (fraction > 0), 0.0, 1.0)
        for name in ("meeting", "occupancy_surge"):
            event = events.get(name)
            if event:
                active = (hours >= float(event["start_hour"])) & (hours < float(event["end_hour"]))
                people[active] += self.capacity * float(event["additional_fraction_of_capacity"])
        return np.rint(np.clip(people, 0, self.capacity)).astype(np.int64)

    def _prices(self, hours: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
        prices = np.full(self.steps, float(config["off_peak"]), dtype=np.float64)
        for start, end in config["shoulder_windows"]:
            prices[(hours >= float(start)) & (hours < float(end))] = float(config["shoulder"])
        for start, end in config["peak_windows"]:
            prices[(hours >= float(start)) & (hours < float(end))] = float(config["peak"])
        return prices

    def _events(
        self, hours: np.ndarray, config: Mapping[str, Any], rng: np.random.Generator
    ) -> tuple[list[DoorState], list[dict[str, Any]]]:
        doors = [DoorState.CLOSED for _ in range(self.steps)]
        metadata: list[dict[str, Any]] = []
        probability = float(config["stochastic_door_probability_per_step"])
        for index, hour in enumerate(hours):
            if 7.0 <= hour < 19.0 and rng.random() < probability:
                doors[index] = DoorState.OPEN
        for event_name in ("meeting", "occupancy_surge", "door_left_open"):
            event = config.get(event_name)
            if not event:
                continue
            start = float(event["start_hour"])
            end = float(event["end_hour"])
            if event_name == "door_left_open":
                for index, hour in enumerate(hours):
                    if start <= hour < end:
                        doors[index] = DoorState.LEFT_OPEN
            metadata.append(
                {
                    "event": event_name,
                    "start_hour": start,
                    "end_hour": end,
                    "forecast_visible": bool(event.get("forecast_visible", True)),
                    "additional_fraction_of_capacity": float(
                        event.get("additional_fraction_of_capacity", 0.0)
                    ),
                }
            )
        for start, end in config["cleaning"]:
            metadata.append(
                {
                    "event": "cleaning",
                    "start_hour": float(start),
                    "end_hour": float(end),
                    "forecast_visible": True,
                }
            )
        return doors, metadata

    @staticmethod
    def _ar1(
        rng: np.random.Generator, length: int, coefficient: float, scale: float
    ) -> np.ndarray:
        values = np.zeros(length, dtype=np.float64)
        innovations = rng.normal(0.0, scale, size=length)
        for index in range(1, length):
            values[index] = coefficient * values[index - 1] + innovations[index]
        return values

    @staticmethod
    def _in_any_window(hour: float, windows: list[list[float]]) -> bool:
        return any(float(start) <= hour < float(end) for start, end in windows)

    def _validate(self) -> None:
        assigned = [name for names in self.config["splits"].values() for name in names]
        if len(assigned) != len(set(assigned)):
            raise ValueError("V2 scenario split assignments must be disjoint")
        if set(assigned) != set(self.definitions):
            missing = set(assigned).symmetric_difference(self.definitions)
            raise ValueError(f"Scenario definitions and splits differ: {sorted(missing)}")
        if self.steps <= 0 or self.dt_hours <= 0.0 or self.capacity <= 0:
            raise ValueError("Invalid V2 episode duration or zone capacity")
