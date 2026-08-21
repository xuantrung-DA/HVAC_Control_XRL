"""Seeded curriculum sampler for continuous V2 control."""

from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np

from src.envs.v2.continuous_env import V2ContinuousHVACEnv


class V2ContinuousScenarioSamplerEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(self, scenarios: Sequence[str], *, seed: int, normal_only_episodes: int = 10) -> None:
        super().__init__()
        if "normal_v2" not in scenarios:
            raise ValueError("Continuous curriculum requires normal_v2")
        self.scenarios = tuple(scenarios)
        self.rng = np.random.default_rng(seed)
        self.normal_only_episodes = normal_only_episodes
        self.episodes = 0
        self.env = V2ContinuousHVACEnv()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        scenario = (
            "normal_v2"
            if self.episodes < self.normal_only_episodes
            else str(self.rng.choice(self.scenarios))
        )
        if options and "scenario" in options:
            scenario = str(options["scenario"])
        if scenario not in self.scenarios:
            raise ValueError("Scenario is outside continuous training pool")
        self.episodes += 1
        return self.env.reset(
            seed=int(self.rng.integers(0, 2**31 - 1)),
            options={"scenario": scenario},
        )

    def step(self, action):
        return self.env.step(action)

    def close(self) -> None:
        self.env.close()
