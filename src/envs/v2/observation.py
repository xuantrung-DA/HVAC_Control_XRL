"""Stable, interpretable V2 observation schema and frozen-V1 adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from src.agents.base_agent import BaseAgent
from src.envs.v2.models import V2BuildingState, V2ExogenousInputs
from src.forecasting import ForecastBundle
from src.risk import MonitoringSnapshot, RiskVector


V2_OBSERVATION_NAMES = (
    "indoor_temperature_c", "outdoor_temperature_c", "indoor_relative_humidity_pct",
    "occupancy", "co2_ppm", "electricity_price_per_kwh", "time_sin", "time_cos",
    "hvac_action", "forecast_1h_outdoor_temperature_c",
    "forecast_1h_outdoor_relative_humidity_pct", "forecast_1h_solar_radiation_w_per_m2",
    "forecast_1h_occupancy", "forecast_1h_electricity_price_per_kwh",
    "forecast_4h_outdoor_temperature_c", "forecast_4h_outdoor_relative_humidity_pct",
    "forecast_4h_solar_radiation_w_per_m2", "forecast_4h_occupancy",
    "forecast_4h_electricity_price_per_kwh", "uncertainty_1h_outdoor_temperature_c",
    "uncertainty_1h_outdoor_relative_humidity_pct", "uncertainty_1h_occupancy",
    "indoor_temperature_slope", "co2_slope", "occupancy_delta",
    "thermal_risk", "humidity_risk", "co2_risk", "occupancy_surge", "forecast_uncertainty",
    "forecast_error", "energy_peak_risk", "weather_reliability",
    "occupancy_reliability", "price_reliability",
)


def v2_observation_space() -> gym.spaces.Box:
    low = np.array([
        5, -20, 5, 0, 350, 0, -1, -1, 0,
        -20, 0, 0, 0, 0, -20, 0, 0, 0, 0,
        0, 0, 0, -2, -500, -100,
        *([0] * 10),
    ], dtype=np.float32)
    high = np.array([
        45, 55, 100, 100, 5000, 2, 1, 1, 3,
        55, 100, 1200, 100, 2, 55, 100, 1200, 100, 2,
        20, 50, 100, 2, 500, 100,
        *([1] * 10),
    ], dtype=np.float32)
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)


def build_v2_observation(
    state: V2BuildingState,
    inputs: V2ExogenousInputs,
    forecast: ForecastBundle,
    monitoring: MonitoringSnapshot,
    risk: RiskVector,
) -> np.ndarray:
    if len(forecast.forecasts) < 4:
        raise ValueError("V2 observation requires 1h through 4h forecasts")
    one_hour = next(item for item in forecast.forecasts if item.horizon_hours == 1.0)
    four_hour = next(item for item in forecast.forecasts if item.horizon_hours == 4.0)
    angle = 2.0 * np.pi * inputs.hour / 24.0
    forecast_features = (
        "outdoor_temperature_c", "outdoor_relative_humidity_pct",
        "solar_radiation_w_per_m2", "occupancy", "electricity_price_per_kwh",
    )
    values = [
        state.indoor_temperature_c, inputs.outdoor_temperature_c,
        state.indoor_relative_humidity_pct, inputs.occupancy, state.co2_ppm,
        inputs.electricity_price_per_kwh, np.sin(angle), np.cos(angle), state.hvac_action,
        *(one_hour.values[name].point for name in forecast_features),
        *(four_hour.values[name].point for name in forecast_features),
        one_hour.values["outdoor_temperature_c"].standard_deviation,
        one_hour.values["outdoor_relative_humidity_pct"].standard_deviation,
        one_hour.values["occupancy"].standard_deviation,
        monitoring.signals["indoor_temperature_c"].rolling_slope,
        monitoring.signals["co2_ppm"].rolling_slope,
        monitoring.signals["occupancy"].delta,
        *risk.as_dict().values(),
    ]
    observation = np.asarray(values, dtype=np.float32)
    if observation.shape != (len(V2_OBSERVATION_NAMES),) or not np.isfinite(observation).all():
        raise RuntimeError("Invalid V2 observation was constructed")
    return np.clip(observation, v2_observation_space().low, v2_observation_space().high)


class V1ObservationAdapter:
    """Map physical V2 observations to the frozen nine-feature V1 contract."""

    V2_TO_V1 = (0, 1, 2, 3, 4, 5, 6, 7, 8)

    def __init__(self, v1_observation_space: gym.spaces.Box) -> None:
        self.v1_space = v1_observation_space

    def transform(self, observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (len(V2_OBSERVATION_NAMES),):
            raise ValueError("V1 adapter received an invalid V2 observation shape")
        adapted = values[np.asarray(self.V2_TO_V1)]
        return np.clip(adapted, self.v1_space.low, self.v1_space.high).astype(np.float32)


@dataclass
class V1AgentOnV2Adapter:
    """Run an unchanged V1 BaseAgent against a V2 environment."""

    agent: BaseAgent
    observation_adapter: V1ObservationAdapter

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        return self.agent.predict(
            self.observation_adapter.transform(observation),
            deterministic=deterministic,
        )

    def reset(self) -> None:
        self.agent.reset()

    def metadata(self) -> dict[str, Any]:
        return {
            **self.agent.metadata(),
            "adapter": "v1_agent_on_v2_observation",
            "checkpoint_modified": False,
        }
