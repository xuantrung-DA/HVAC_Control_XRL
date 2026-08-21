"""Compact PyTorch DQN implementation for the XRL-HVAC environment."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.agents.base_agent import (
    BaseAgent,
    ObservationScaler,
    ObservationView,
    TrainingSummary,
)
from src.utils.config import load_agent_config
from src.utils.seed import seed_everything


class QNetwork(nn.Module):
    """Small MLP mapping normalized building states to action values."""

    def __init__(
        self,
        observation_size: int,
        action_count: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_size = observation_size
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(input_size, hidden_size), nn.ReLU()))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, action_count))
        self.layers = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.layers(observation)


class ReplayBuffer:
    """Fixed-size NumPy replay memory with deterministic sampling."""

    def __init__(self, capacity: int, observation_size: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive")
        self.capacity = capacity
        self.observations = np.empty(
            (capacity, observation_size), dtype=np.float32
        )
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if batch_size > self.size:
            raise ValueError("Cannot sample more transitions than the buffer contains")
        indices = self.rng.integers(0, self.size, size=batch_size)
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size


class DQNAgent(BaseAgent):
    """Deep Q-Network agent with replay memory and a target network."""

    name = "dqn"
    trainable = True

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        config: Mapping[str, Any] | None = None,
        *,
        algorithm_name: str | None = None,
    ) -> None:
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("DQN requires a Box observation space")
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError("DQN requires a Discrete action space")
        if int(action_space.n) != self.action_count:
            raise ValueError("DQN expects exactly four HVAC actions")

        self.config = dict(config) if config is not None else load_agent_config("dqn")
        self.name = algorithm_name or self.name
        self.seed = int(self.config["agent"]["seed"])
        seed_everything(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.scaler = ObservationScaler(observation_space)
        self.observation_size = int(np.prod(observation_space.shape))

        requested_device = str(self.config["agent"].get("device", "auto"))
        self.device = self._resolve_device(requested_device)
        hidden_sizes = [int(size) for size in self.config["model"]["hidden_sizes"]]
        self.online_network = QNetwork(
            self.observation_size, self.action_count, hidden_sizes
        ).to(self.device)
        self.target_network = QNetwork(
            self.observation_size, self.action_count, hidden_sizes
        ).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        optimization = self.config["optimization"]
        replay = self.config["replay_buffer"]
        self.gamma = float(optimization["gamma"])
        self.reward_scale = float(optimization.get("reward_scale", 1.0))
        self.gradient_clip_norm = float(optimization["gradient_clip_norm"])
        self.train_frequency = int(optimization["train_frequency"])
        self.target_update_frequency = int(
            optimization["target_update_frequency"]
        )
        self.batch_size = int(replay["batch_size"])
        self.warmup_steps = int(replay["warmup_steps"])
        self.replay_buffer = ReplayBuffer(
            int(replay["capacity"]), self.observation_size, self.seed
        )
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=float(optimization["learning_rate"]),
        )

        exploration = self.config["exploration"]
        self.epsilon_start = float(exploration["epsilon_start"])
        self.epsilon_end = float(exploration["epsilon_end"])
        self.epsilon_decay_steps = int(exploration["epsilon_decay_steps"])
        self.total_steps = 0
        self.gradient_updates = 0
        self.episodes_trained = 0
        self.recent_losses: deque[float] = deque(maxlen=1000)

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self.total_steps / self.epsilon_decay_steps)
        return self.epsilon_start + progress * (
            self.epsilon_end - self.epsilon_start
        )

    def _predict(
        self, observation: ObservationView, *, deterministic: bool
    ) -> int:
        if not deterministic and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_count))
        scores = self.action_scores(observation.to_array())
        return int(np.argmax(scores))

    def action_scores(self, observation: np.ndarray) -> np.ndarray:
        """Return Q-values for XAI and deterministic inference."""

        scaled = self.scaler.transform(observation)
        tensor = torch.as_tensor(scaled, device=self.device).unsqueeze(0)
        with torch.no_grad():
            values = self.online_network(tensor).squeeze(0)
        return values.detach().cpu().numpy().astype(np.float32)

    def learn(
        self,
        env: gym.Env,
        *,
        total_steps: int,
        seed: int | None = None,
    ) -> TrainingSummary:
        """Collect experience and train for the requested transition budget."""

        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        base_seed = self.seed if seed is None else seed
        started = time.perf_counter()
        episode_rewards: list[float] = []
        current_episode_reward = 0.0
        observation, _ = env.reset(seed=base_seed + self.episodes_trained)

        for _ in range(total_steps):
            action = self.predict(observation, deterministic=False)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            self.replay_buffer.add(
                self.scaler.transform(observation),
                action,
                reward * self.reward_scale,
                self.scaler.transform(next_observation),
                done,
            )
            self.total_steps += 1
            current_episode_reward += reward

            if (
                len(self.replay_buffer) >= max(self.warmup_steps, self.batch_size)
                and self.total_steps % self.train_frequency == 0
            ):
                self.recent_losses.append(self._train_batch())
            if self.total_steps % self.target_update_frequency == 0:
                self.target_network.load_state_dict(self.online_network.state_dict())

            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                self.episodes_trained += 1
                observation, _ = env.reset(
                    seed=base_seed + self.episodes_trained
                )
            else:
                observation = next_observation

        duration = time.perf_counter() - started
        completed_rewards = episode_rewards or [current_episode_reward]
        return TrainingSummary(
            algorithm=self.name,
            total_steps=total_steps,
            episodes=len(episode_rewards),
            mean_episode_reward=float(np.mean(completed_rewards)),
            final_episode_reward=float(completed_rewards[-1]),
            mean_loss=(
                float(np.mean(self.recent_losses)) if self.recent_losses else None
            ),
            duration_seconds=duration,
        )

    def save(self, path: str | Path) -> None:
        checkpoint = Path(path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": self.name,
                "online_network": self.online_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "gradient_updates": self.gradient_updates,
                "episodes_trained": self.episodes_trained,
            },
            checkpoint,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = Path(path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"DQN checkpoint not found: {checkpoint}")
        payload = torch.load(
            checkpoint,
            map_location=self.device,
            weights_only=False,
        )
        if payload.get("algorithm") != self.name:
            raise ValueError(
                f"Checkpoint algorithm {payload.get('algorithm')!r} "
                f"does not match agent {self.name!r}"
            )
        self.online_network.load_state_dict(payload["online_network"])
        self.target_network.load_state_dict(payload["target_network"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.total_steps = int(payload["total_steps"])
        self.gradient_updates = int(payload["gradient_updates"])
        self.episodes_trained = int(payload["episodes_trained"])

    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.online_network.parameters()
            if parameter.requires_grad
        )

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "device": str(self.device),
            "parameters": self.parameter_count(),
            "total_steps": self.total_steps,
            "gradient_updates": self.gradient_updates,
            "epsilon": self.epsilon,
        }

    def _train_batch(self) -> float:
        observations, actions, rewards, next_observations, dones = (
            self.replay_buffer.sample(self.batch_size)
        )
        observation_tensor = torch.as_tensor(observations, device=self.device)
        action_tensor = torch.as_tensor(actions, device=self.device).unsqueeze(1)
        reward_tensor = torch.as_tensor(rewards, device=self.device)
        next_observation_tensor = torch.as_tensor(
            next_observations, device=self.device
        )
        done_tensor = torch.as_tensor(dones, device=self.device)

        predicted = self.online_network(observation_tensor).gather(
            1, action_tensor
        ).squeeze(1)
        with torch.no_grad():
            next_values = self._next_state_values(next_observation_tensor)
            target = reward_tensor + self.gamma * (1.0 - done_tensor) * next_values

        loss = F.smooth_l1_loss(predicted, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.online_network.parameters(), self.gradient_clip_norm
        )
        self.optimizer.step()
        self.gradient_updates += 1
        return float(loss.detach().cpu().item())

    def _next_state_values(self, next_observations: torch.Tensor) -> torch.Tensor:
        return self.target_network(next_observations).max(dim=1).values

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device(requested)
