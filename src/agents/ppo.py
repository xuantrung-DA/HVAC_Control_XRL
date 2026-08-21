"""Stable-Baselines3 PPO adapter implementing the project BaseAgent API."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.agents.base_agent import (
    BaseAgent,
    ObservationScaler,
    ObservationView,
    ScaledObservationEnv,
    TrainingSummary,
)
from src.utils.config import load_agent_config
from src.utils.seed import seed_everything


class _EpisodeRewardCallback(BaseCallback):
    """Collect raw episode returns without requiring a Monitor wrapper."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.current_reward = 0.0
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals["rewards"]).reshape(-1)
        dones = np.asarray(self.locals["dones"]).reshape(-1)
        self.current_reward += float(rewards[0])
        if bool(dones[0]):
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0.0
        return True


class PPOAgent(BaseAgent):
    """Reliable PPO policy backed by Stable-Baselines3."""

    name = "ppo"
    trainable = True

    def __init__(
        self,
        env: gym.Env,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = dict(config) if config is not None else load_agent_config("ppo")
        self.seed = int(self.config["agent"]["seed"])
        seed_everything(self.seed)
        self.scaler = ObservationScaler(env.observation_space)
        self.training_env = ScaledObservationEnv(env, self.scaler)

        requested_device = str(self.config["agent"].get("device", "auto"))
        # SB3 recommends CPU for small MLP PPO policies; CUDA remains used by DQN.
        self.device = "cpu" if requested_device == "auto" else requested_device
        model_config = self.config["model"]
        optimization = self.config["optimization"]
        training = self.config["training"]
        activation = {
            "tanh": torch.nn.Tanh,
            "relu": torch.nn.ReLU,
        }.get(str(model_config["activation"]).lower())
        if activation is None:
            raise ValueError("PPO activation must be 'tanh' or 'relu'")

        self.model = PPO(
            policy=str(model_config["policy"]),
            env=self.training_env,
            learning_rate=float(optimization["learning_rate"]),
            n_steps=int(training["n_steps"]),
            batch_size=int(training["batch_size"]),
            n_epochs=int(training["n_epochs"]),
            gamma=float(optimization["gamma"]),
            gae_lambda=float(optimization["gae_lambda"]),
            clip_range=float(optimization["clip_range"]),
            ent_coef=float(optimization["ent_coef"]),
            vf_coef=float(optimization["vf_coef"]),
            max_grad_norm=float(optimization["max_grad_norm"]),
            policy_kwargs={
                "net_arch": [int(size) for size in model_config["hidden_sizes"]],
                "activation_fn": activation,
            },
            seed=self.seed,
            device=self.device,
            verbose=0,
        )

    def _predict(
        self, observation: ObservationView, *, deterministic: bool
    ) -> int:
        scaled = self.scaler.transform(observation.to_array())
        action, _ = self.model.predict(scaled, deterministic=deterministic)
        return int(np.asarray(action).item())

    def action_scores(self, observation: np.ndarray) -> np.ndarray:
        """Return categorical action logits for inference and future XAI."""

        scaled = self.scaler.transform(observation)
        observation_tensor, _ = self.model.policy.obs_to_tensor(scaled)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(observation_tensor)
            logits = distribution.distribution.logits.squeeze(0)
        return logits.detach().cpu().numpy().astype(np.float32)

    def learn(self, *, total_timesteps: int) -> TrainingSummary:
        if total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        callback = _EpisodeRewardCallback()
        started = time.perf_counter()
        previous_steps = int(self.model.num_timesteps)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        duration = time.perf_counter() - started
        actual_steps = int(self.model.num_timesteps) - previous_steps
        rewards = callback.episode_rewards or [callback.current_reward]
        return TrainingSummary(
            algorithm=self.name,
            total_steps=actual_steps,
            episodes=len(callback.episode_rewards),
            mean_episode_reward=float(np.mean(rewards)),
            final_episode_reward=float(rewards[-1]),
            mean_loss=None,
            duration_seconds=duration,
        )

    def save(self, path: str | Path) -> None:
        checkpoint = Path(path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(checkpoint)

    def load(self, path: str | Path) -> None:
        checkpoint = Path(path)
        resolved = checkpoint if checkpoint.suffix == ".zip" else checkpoint.with_suffix(".zip")
        if not resolved.is_file():
            raise FileNotFoundError(f"PPO checkpoint not found: {resolved}")
        self.model = PPO.load(
            resolved,
            env=self.training_env,
            device=self.device,
        )

    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.model.policy.parameters()
            if parameter.requires_grad
        )

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "device": str(self.model.device),
            "parameters": self.parameter_count(),
            "total_steps": int(self.model.num_timesteps),
            "library": "stable-baselines3",
        }
