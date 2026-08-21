"""Observation-group masking used only for controlled V2 ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import gymnasium as gym
import numpy as np


class MaskedObservationEnv(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, indices: Sequence[int]) -> None:
        super().__init__(env)
        self.indices = np.asarray(sorted(set(indices)), dtype=np.int64)
        self.midpoint = (
            np.asarray(env.observation_space.low, dtype=np.float32)
            + np.asarray(env.observation_space.high, dtype=np.float32)
        ) / 2.0

    def observation(self, observation: np.ndarray) -> np.ndarray:
        masked = np.asarray(observation, dtype=np.float32).copy()
        masked[self.indices] = self.midpoint[self.indices]
        return masked


@dataclass
class MaskedController:
    controller: object
    indices: Sequence[int]
    observation_space: gym.spaces.Box
    name: str = "masked_controller"

    def __post_init__(self) -> None:
        self.indices = tuple(sorted(set(int(index) for index in self.indices)))
        self.midpoint = (
            np.asarray(self.observation_space.low, dtype=np.float32)
            + np.asarray(self.observation_space.high, dtype=np.float32)
        ) / 2.0

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        masked = np.asarray(observation, dtype=np.float32).copy()
        masked[list(self.indices)] = self.midpoint[list(self.indices)]
        return int(self.controller.predict(masked, deterministic=deterministic))

    def reset(self) -> None:
        self.controller.reset()
