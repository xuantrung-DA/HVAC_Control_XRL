"""Tabular Q-learning with a compact, interpretable state discretization."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import numpy as np

from src.agents.base_agent import BaseAgent, ObservationView, TrainingSummary
from src.utils.config import load_agent_config
from src.utils.seed import seed_everything


DiscreteState = tuple[int, ...]


class QLearningAgent(BaseAgent):
    """Dictionary-backed Q-learning agent for the continuous HVAC state."""

    name = "q_learning"
    trainable = True

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = (
            dict(config) if config is not None else load_agent_config("q_learning")
        )
        learning = self.config["learning"]
        self.learning_rate = float(learning["learning_rate"])
        self.gamma = float(learning["gamma"])
        self.epsilon_start = float(learning["epsilon_start"])
        self.epsilon_end = float(learning["epsilon_end"])
        self.epsilon_decay_episodes = int(learning["epsilon_decay_episodes"])
        self.seed = int(self.config["agent"]["seed"])
        self.rng = np.random.default_rng(self.seed)
        self.q_table: dict[DiscreteState, np.ndarray] = {}
        self.episodes_trained = 0
        self.total_steps = 0
        seed_everything(self.seed)

        discretization = self.config["discretization"]
        self.bins = {
            key: np.asarray(values, dtype=np.float32)
            for key, values in discretization.items()
        }

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self.episodes_trained / self.epsilon_decay_episodes)
        return self.epsilon_start + progress * (
            self.epsilon_end - self.epsilon_start
        )

    def _predict(
        self, observation: ObservationView, *, deterministic: bool
    ) -> int:
        state = self.discretize(observation)
        if not deterministic and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_count))

        values = self.q_table.get(state)
        if values is None:
            values = np.zeros(self.action_count, dtype=np.float64)
        best_actions = np.flatnonzero(values == np.max(values))
        if deterministic:
            return int(best_actions[0])
        return int(self.rng.choice(best_actions))

    def discretize(self, observation: ObservationView | np.ndarray) -> DiscreteState:
        """Convert physical state values into a finite tabular key."""

        view = (
            observation
            if isinstance(observation, ObservationView)
            else ObservationView.from_array(observation)
        )
        angle = np.arctan2(view.time_sin, view.time_cos)
        hour = float((angle % (2.0 * np.pi)) * 24.0 / (2.0 * np.pi))
        return (
            self._bin(view.indoor_temperature_c, "indoor_temperature_c"),
            self._bin(view.outdoor_temperature_c, "outdoor_temperature_c"),
            self._bin(view.occupancy, "occupancy"),
            self._bin(view.co2_ppm, "co2_ppm"),
            self._bin(
                view.electricity_price_per_kwh,
                "electricity_price_per_kwh",
            ),
            self._bin(hour, "hour"),
            view.hvac_action,
        )

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> float:
        """Apply one temporal-difference update and return the TD error."""

        state = self.discretize(observation)
        next_state = self.discretize(next_observation)
        values = self._values(state)
        next_value = 0.0 if done else float(np.max(self._values(next_state)))
        target = reward + self.gamma * next_value
        td_error = target - float(values[action])
        values[action] += self.learning_rate * td_error
        self.total_steps += 1
        return float(td_error)

    def learn(
        self,
        env: gym.Env,
        *,
        episodes: int,
        seed: int | None = None,
    ) -> TrainingSummary:
        """Train for a small or full number of complete daily episodes."""

        if episodes <= 0:
            raise ValueError("episodes must be positive")
        base_seed = self.seed if seed is None else seed
        started = time.perf_counter()
        episode_rewards: list[float] = []
        td_errors: list[float] = []

        for episode in range(episodes):
            observation, _ = env.reset(seed=base_seed + self.episodes_trained)
            total_reward = 0.0
            done = False
            while not done:
                action = self.predict(observation, deterministic=False)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                td_errors.append(
                    abs(
                        self.update(
                            observation,
                            action,
                            reward,
                            next_observation,
                            done,
                        )
                    )
                )
                total_reward += reward
                observation = next_observation
            self.episodes_trained += 1
            episode_rewards.append(total_reward)

        duration = time.perf_counter() - started
        return TrainingSummary(
            algorithm=self.name,
            total_steps=episodes * int(env.unwrapped.max_steps),
            episodes=episodes,
            mean_episode_reward=float(np.mean(episode_rewards)),
            final_episode_reward=float(episode_rewards[-1]),
            mean_loss=float(np.mean(td_errors)) if td_errors else None,
            duration_seconds=duration,
        )

    def save(self, path: str | Path) -> None:
        """Save Q-values and training state without using pickle."""

        checkpoint = Path(path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        states = np.asarray(list(self.q_table), dtype=np.int16)
        if self.q_table:
            values = np.stack([self.q_table[state] for state in self.q_table])
        else:
            states = np.empty((0, 7), dtype=np.int16)
            values = np.empty((0, self.action_count), dtype=np.float32)
        metadata = json.dumps(
            {
                "name": self.name,
                "episodes_trained": self.episodes_trained,
                "total_steps": self.total_steps,
            }
        )
        np.savez_compressed(
            checkpoint,
            states=states,
            q_values=values.astype(np.float32),
            metadata=np.asarray(metadata),
        )

    def load(self, path: str | Path) -> None:
        checkpoint = Path(path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Q-learning checkpoint not found: {checkpoint}")
        with np.load(checkpoint, allow_pickle=False) as data:
            states = data["states"]
            values = data["q_values"]
            metadata = json.loads(str(data["metadata"].item()))
        if states.ndim != 2 or states.shape[1] != 7:
            raise ValueError("Invalid Q-learning checkpoint state shape")
        if values.shape != (states.shape[0], self.action_count):
            raise ValueError("Invalid Q-learning checkpoint value shape")
        self.q_table = {
            tuple(int(item) for item in state): value.astype(np.float64)
            for state, value in zip(states, values, strict=True)
        }
        self.episodes_trained = int(metadata["episodes_trained"])
        self.total_steps = int(metadata["total_steps"])

    def parameter_count(self) -> int:
        return len(self.q_table) * self.action_count

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "states_visited": len(self.q_table),
            "episodes_trained": self.episodes_trained,
            "epsilon": self.epsilon,
        }

    def _values(self, state: DiscreteState) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_count, dtype=np.float64)
        return self.q_table[state]

    def _bin(self, value: float, key: str) -> int:
        return int(np.digitize(value, self.bins[key], right=False))
