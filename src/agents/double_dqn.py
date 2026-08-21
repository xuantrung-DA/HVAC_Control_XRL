"""Double DQN variant reusing the shared DQN training infrastructure."""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import torch

from src.agents.dqn import DQNAgent


class DoubleDQNAgent(DQNAgent):
    """Reduce maximization bias by separating action selection/evaluation."""

    name = "double_dqn"

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            config,
            algorithm_name=self.name,
        )

    def _next_state_values(self, next_observations: torch.Tensor) -> torch.Tensor:
        selected_actions = self.online_network(next_observations).argmax(
            dim=1, keepdim=True
        )
        return self.target_network(next_observations).gather(
            1, selected_actions
        ).squeeze(1)
