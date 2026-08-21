"""Seeded curriculum wrapper for V2 DQN training scenarios."""

from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np

from src.envs.v2.hvac_env import V2HVACEnv


class V2ScenarioSamplerEnv(gym.Env[np.ndarray, int]):
    """Sample development scenarios per episode without touching held-out splits."""

    def __init__(
        self,
        scenarios: Sequence[str],
        *,
        seed: int,
        normal_only_episodes: int = 10,
        shield_enabled: bool = False,
        reward_mode: str = "dynamic",
    ) -> None:
        super().__init__()
        if not scenarios or "normal_v2" not in scenarios:
            raise ValueError("V2 curriculum requires normal_v2 and at least one scenario")
        self.scenarios = tuple(scenarios)
        self.normal_only_episodes = int(normal_only_episodes)
        self.rng = np.random.default_rng(seed)
        self.episodes = 0
        self.env = V2HVACEnv(
            "normal_v2", shield_enabled=shield_enabled, reward_mode=reward_mode
        )
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.current_scenario = "normal_v2"

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if options and "scenario" in options:
            scenario = str(options["scenario"])
        elif self.episodes < self.normal_only_episodes:
            scenario = "normal_v2"
        else:
            scenario = str(self.rng.choice(self.scenarios))
        if scenario not in self.scenarios:
            raise ValueError(f"Scenario {scenario!r} is outside the training pool")
        self.current_scenario = scenario
        self.episodes += 1
        episode_seed = int(self.rng.integers(0, 2**31 - 1))
        observation, info = self.env.reset(seed=episode_seed, options={"scenario": scenario})
        info["curriculum_episode"] = self.episodes
        return observation, info

    def step(self, action: int):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()
