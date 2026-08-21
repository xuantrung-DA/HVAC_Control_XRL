"""Multi-scenario environment used for curriculum training."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from src.envs.hvac_env import HVACEnv


class ScenarioSamplerEnv(gym.Env):
    """Deterministically sample configured scenarios across training episodes."""

    metadata = HVACEnv.metadata

    def __init__(
        self,
        scenarios: list[str] | tuple[str, ...],
        *,
        strategy: str = "curriculum",
        normal_only_episodes: int = 12,
        expansion_episodes: int = 36,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not scenarios:
            raise ValueError("ScenarioSamplerEnv requires at least one scenario")
        if strategy not in {"curriculum", "round_robin", "random"}:
            raise ValueError(
                "strategy must be 'curriculum', 'round_robin', or 'random'"
            )
        if normal_only_episodes < 0 or expansion_episodes < normal_only_episodes:
            raise ValueError("Invalid curriculum episode boundaries")

        self.scenarios = tuple(scenarios)
        self.strategy = strategy
        self.normal_only_episodes = normal_only_episodes
        self.expansion_episodes = expansion_episodes
        self.rng = np.random.default_rng(seed)
        self.environments = {
            scenario: HVACEnv(scenario=scenario) for scenario in self.scenarios
        }
        reference = self.environments[self.scenarios[0]]
        self.action_space = reference.action_space
        self.observation_space = reference.observation_space
        self.max_steps = reference.max_steps
        self.episode_index = 0
        self.current_scenario = self.scenarios[0]
        self.current_environment = reference

        for environment in self.environments.values():
            if environment.observation_space != self.observation_space:
                raise ValueError("All sampled scenarios must share an observation space")
            if environment.action_space != self.action_space:
                raise ValueError("All sampled scenarios must share an action space")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_scenario = self._select_scenario()
        self.current_environment = self.environments[self.current_scenario]
        observation, info = self.current_environment.reset(seed=seed, options=options)
        info = {
            **info,
            "training_scenario": self.current_scenario,
            "curriculum_episode": self.episode_index,
        }
        self.episode_index += 1
        return observation, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return self.current_environment.step(action)

    def render(self) -> str | None:
        return self.current_environment.render()

    def close(self) -> None:
        for environment in self.environments.values():
            environment.close()

    def _select_scenario(self) -> str:
        if self.strategy == "random":
            return str(self.rng.choice(self.scenarios))
        active = self.scenarios
        if self.strategy == "curriculum":
            if self.episode_index < self.normal_only_episodes:
                active = self.scenarios[:1]
            elif self.episode_index < self.expansion_episodes:
                active = self.scenarios[: min(2, len(self.scenarios))]
        return active[self.episode_index % len(active)]
