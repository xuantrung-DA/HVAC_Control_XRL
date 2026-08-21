"""Shared inference interface for traditional and learned controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np


class ObservationIndex(IntEnum):
    """Stable positions in the environment observation vector."""

    INDOOR_TEMPERATURE = 0
    OUTDOOR_TEMPERATURE = 1
    RELATIVE_HUMIDITY = 2
    OCCUPANCY = 3
    CO2 = 4
    ELECTRICITY_PRICE = 5
    TIME_SIN = 6
    TIME_COS = 7
    HVAC_ACTION = 8


@dataclass(frozen=True)
class ObservationView:
    """Named, typed access to the environment observation vector."""

    indoor_temperature_c: float
    outdoor_temperature_c: float
    relative_humidity_pct: float
    occupancy: int
    co2_ppm: float
    electricity_price_per_kwh: float
    time_sin: float
    time_cos: float
    hvac_action: int

    @classmethod
    def from_array(cls, observation: np.ndarray) -> "ObservationView":
        array = np.asarray(observation, dtype=np.float32)
        if array.shape != (len(ObservationIndex),):
            raise ValueError(
                f"Expected observation shape ({len(ObservationIndex)},), "
                f"received {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("Observation contains a non-finite value")
        return cls(
            indoor_temperature_c=float(array[ObservationIndex.INDOOR_TEMPERATURE]),
            outdoor_temperature_c=float(array[ObservationIndex.OUTDOOR_TEMPERATURE]),
            relative_humidity_pct=float(array[ObservationIndex.RELATIVE_HUMIDITY]),
            occupancy=int(round(float(array[ObservationIndex.OCCUPANCY]))),
            co2_ppm=float(array[ObservationIndex.CO2]),
            electricity_price_per_kwh=float(
                array[ObservationIndex.ELECTRICITY_PRICE]
            ),
            time_sin=float(array[ObservationIndex.TIME_SIN]),
            time_cos=float(array[ObservationIndex.TIME_COS]),
            hvac_action=int(round(float(array[ObservationIndex.HVAC_ACTION]))),
        )

    def to_array(self) -> np.ndarray:
        """Reconstruct the canonical float32 observation vector."""

        return np.array(
            [
                self.indoor_temperature_c,
                self.outdoor_temperature_c,
                self.relative_humidity_pct,
                self.occupancy,
                self.co2_ppm,
                self.electricity_price_per_kwh,
                self.time_sin,
                self.time_cos,
                self.hvac_action,
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class TrainingSummary:
    """Compact metrics returned by short or full agent training calls."""

    algorithm: str
    total_steps: int
    episodes: int
    mean_episode_reward: float
    final_episode_reward: float
    mean_loss: float | None
    duration_seconds: float

    @property
    def steps_per_second(self) -> float:
        return self.total_steps / max(self.duration_seconds, 1e-9)

    def as_dict(self) -> dict[str, float | int | str | None]:
        return {
            "algorithm": self.algorithm,
            "total_steps": self.total_steps,
            "episodes": self.episodes,
            "mean_episode_reward": self.mean_episode_reward,
            "final_episode_reward": self.final_episode_reward,
            "mean_loss": self.mean_loss,
            "duration_seconds": self.duration_seconds,
            "steps_per_second": self.steps_per_second,
        }

class ObservationScaler:
    """Fixed min/max scaling shared by DQN and PPO checkpoints."""

    def __init__(self, observation_space: gym.spaces.Box) -> None:
        self.low = np.asarray(observation_space.low, dtype=np.float32)
        self.high = np.asarray(observation_space.high, dtype=np.float32)
        self.span = np.where(self.high > self.low, self.high - self.low, 1.0)

    def transform(self, observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float32)
        scaled = 2.0 * (values - self.low) / self.span - 1.0
        return np.clip(scaled, -1.0, 1.0).astype(np.float32)


class ScaledObservationEnv(gym.ObservationWrapper):
    """Expose fixed [-1, 1] observations to third-party RL libraries."""

    def __init__(self, env: gym.Env, scaler: ObservationScaler) -> None:
        super().__init__(env)
        self.scaler = scaler
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return self.scaler.transform(observation)


class BaseAgent(ABC):
    """Minimal controller contract shared by baselines and RL policies."""

    action_count = 4
    trainable = False
    name = "base"

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """Validate an observation and return one of the four HVAC actions."""

        view = ObservationView.from_array(observation)
        action = int(self._predict(view, deterministic=deterministic))
        if action not in range(self.action_count):
            raise RuntimeError(
                f"Controller '{self.name}' returned invalid action {action}"
            )
        return action

    @abstractmethod
    def _predict(
        self, observation: ObservationView, *, deterministic: bool
    ) -> int:
        """Implement controller-specific inference on a validated observation."""

    def reset(self) -> None:
        """Reset optional controller state between episodes."""

    def metadata(self) -> dict[str, Any]:
        """Return serializable controller capabilities."""

        return {
            "name": self.name,
            "trainable": self.trainable,
            "action_count": self.action_count,
        }

    def parameter_count(self) -> int:
        """Return trainable parameter count, or zero for tabular/rule policies."""

        return 0

    def save(self, path: str) -> None:
        """Save a trainable policy checkpoint."""

        raise NotImplementedError(f"Controller '{self.name}' does not use checkpoints")

    def load(self, path: str) -> None:
        """Load a trainable policy checkpoint into this instance."""

        raise NotImplementedError(f"Controller '{self.name}' does not use checkpoints")
