"""Lifecycle and inference access for the frozen demo controller."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from src.agents.dqn import DQNAgent
from src.envs.building import HVACAction
from src.envs.hvac_env import HVACEnv
from src.utils.config import PROJECT_ROOT, deep_merge, load_agent_config


class AgentService:
    """Lazily load and integrity-check the immutable DQN demo checkpoint."""

    def __init__(self, manifest_path: Path | None = None) -> None:
        self.manifest_path = manifest_path or PROJECT_ROOT / "models" / "demo_manifest.json"
        self._agent: DQNAgent | None = None
        self._manifest: dict[str, Any] | None = None
        self._lock = threading.Lock()

    @property
    def agent(self) -> DQNAgent:
        self._ensure_loaded()
        assert self._agent is not None
        return self._agent

    @property
    def manifest(self) -> dict[str, Any]:
        self._ensure_loaded()
        assert self._manifest is not None
        return self._manifest

    def predict(self, observation: np.ndarray) -> dict[str, Any]:
        scores = self.agent.action_scores(observation)
        action = int(np.argmax(scores))
        return {
            "action": action,
            "action_name": HVACAction(action).name,
            "q_values": [float(value) for value in scores],
            "deterministic": True,
        }

    def metadata(self) -> dict[str, Any]:
        manifest = self.manifest
        return {
            **self.agent.metadata(),
            "role": "frozen_demo_controller",
            "training_seed": manifest["training_seed"],
            "checkpoint": manifest["frozen_checkpoint"].replace("\\", "/"),
            "checkpoint_sha256": manifest["frozen_checkpoint_sha256"],
            "selection_rule": manifest["selection_rule"],
            "selection_score": manifest["selection_score"],
        }

    def _ensure_loaded(self) -> None:
        if self._agent is not None:
            return
        with self._lock:
            if self._agent is not None:
                return
            with self.manifest_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
            relative = str(manifest["frozen_checkpoint"]).replace("\\", "/")
            checkpoint = PROJECT_ROOT / relative
            actual_hash = _sha256(checkpoint)
            if actual_hash != manifest["frozen_checkpoint_sha256"]:
                raise RuntimeError("Frozen DQN checkpoint failed its SHA-256 integrity check")
            env = HVACEnv()
            config = deep_merge(
                load_agent_config("dqn"),
                {"agent": {"seed": int(manifest["training_seed"])}},
            )
            agent = DQNAgent(env.observation_space, env.action_space, config=config)
            agent.load(checkpoint)
            env.close()
            self._manifest = manifest
            self._agent = agent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
