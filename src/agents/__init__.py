"""Reinforcement-learning agents and construction helpers."""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym

from src.agents.base_agent import BaseAgent
from src.agents.double_dqn import DoubleDQNAgent
from src.agents.dqn import DQNAgent
from src.agents.ppo import PPOAgent
from src.agents.q_learning import QLearningAgent


RL_AGENT_NAMES = ("q_learning", "dqn", "double_dqn", "ppo")


def create_agent(
    name: str,
    env: gym.Env,
    config: Mapping[str, Any] | None = None,
) -> BaseAgent:
    """Create an RL agent using the environment's public spaces."""

    if name == "q_learning":
        return QLearningAgent(config=config)
    if name == "dqn":
        return DQNAgent(env.observation_space, env.action_space, config=config)
    if name == "double_dqn":
        return DoubleDQNAgent(env.observation_space, env.action_space, config=config)
    if name == "ppo":
        return PPOAgent(env, config=config)
    available = ", ".join(RL_AGENT_NAMES)
    raise ValueError(f"Unknown RL agent '{name}'. Available agents: {available}")


__all__ = [
    "DQNAgent",
    "DoubleDQNAgent",
    "PPOAgent",
    "QLearningAgent",
    "RL_AGENT_NAMES",
    "create_agent",
]
