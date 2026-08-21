"""Stable-Baselines3 SAC adapter for independent cooling/ventilation control."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from src.agents.base_agent import ObservationScaler, ScaledObservationEnv, TrainingSummary
from src.utils.seed import seed_everything


class _SACRewardCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.current_reward = 0.0
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        self.current_reward += float(np.asarray(self.locals["rewards"]).reshape(-1)[0])
        if bool(np.asarray(self.locals["dones"]).reshape(-1)[0]):
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0.0
        return True


class SACV2Agent:
    name = "sac_v2"
    trainable = True

    def __init__(self, env: gym.Env, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.seed = int(config["agent"]["seed"])
        seed_everything(self.seed)
        self.scaler = ObservationScaler(env.observation_space)
        self.training_env = ScaledObservationEnv(env, self.scaler)
        requested = str(config["agent"].get("device", "auto"))
        self.device = "cuda" if requested == "auto" and torch.cuda.is_available() else (
            "cpu" if requested == "auto" else requested
        )
        activation = {
            "relu": torch.nn.ReLU,
            "tanh": torch.nn.Tanh,
        }[str(config["model"]["activation"]).lower()]
        opt = config["optimization"]
        self.model = SAC(
            policy=str(config["model"]["policy"]),
            env=self.training_env,
            learning_rate=float(opt["learning_rate"]),
            buffer_size=int(config["replay_buffer"]["capacity"]),
            learning_starts=int(opt["learning_starts"]),
            batch_size=int(opt["batch_size"]),
            tau=float(opt["tau"]),
            gamma=float(opt["gamma"]),
            train_freq=int(opt["train_frequency"]),
            gradient_steps=int(opt["gradient_steps"]),
            ent_coef=opt["ent_coef"],
            policy_kwargs={
                "net_arch": [int(value) for value in config["model"]["hidden_sizes"]],
                "activation_fn": activation,
            },
            seed=self.seed,
            device=self.device,
            verbose=0,
        )

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        scaled = self.scaler.transform(observation)
        action, _ = self.model.predict(scaled, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32).reshape(2)

    def learn(self, total_timesteps: int) -> TrainingSummary:
        callback = _SACRewardCallback()
        previous = int(self.model.num_timesteps)
        started = time.perf_counter()
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        duration = time.perf_counter() - started
        steps = int(self.model.num_timesteps) - previous
        rewards = callback.episode_rewards or [callback.current_reward]
        return TrainingSummary(
            algorithm=self.name,
            total_steps=steps,
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
        self.model = SAC.load(resolved, env=self.training_env, device=self.device)

    def reset(self) -> None:
        return None

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.policy.parameters())

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameters": self.parameter_count(),
            "device": str(self.model.device),
            "timesteps": int(self.model.num_timesteps),
            "action_semantics": ["cooling_fraction", "ventilation_fraction"],
        }
