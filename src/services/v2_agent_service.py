"""Integrity-checked access to the best *experimental* V2 DQN checkpoint."""

from __future__ import annotations

import hashlib
import json
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.agents.dqn import DQNAgent
from src.envs.v2 import V2HVACEnv
from src.utils.config import PROJECT_ROOT, deep_merge, load_agent_config, load_yaml
from src.xai.v2_explainer import V2PolicyExplainer


class V2AgentService:
    def __init__(self) -> None:
        self._agent: DQNAgent | None = None
        self._explainer: V2PolicyExplainer | None = None
        self._metadata: dict[str, Any] | None = None
        self._lock = threading.Lock()

    @property
    def agent(self) -> DQNAgent:
        self._ensure_loaded()
        assert self._agent is not None
        return self._agent

    @property
    def explainer(self) -> V2PolicyExplainer:
        self._ensure_loaded()
        assert self._explainer is not None
        return self._explainer

    def metadata(self) -> dict[str, Any]:
        self._ensure_loaded()
        assert self._metadata is not None
        return dict(self._metadata)

    def _ensure_loaded(self) -> None:
        if self._agent is not None:
            return
        with self._lock:
            if self._agent is not None:
                return
            summary_path = PROJECT_ROOT / "outputs/v2/training/dqn_development_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            selected = next(item for item in summary["per_seed"] if item["seed"] == 2026)
            checkpoint = PROJECT_ROOT / str(selected["checkpoint"]).replace("\\", "/")
            actual_hash = _sha256(checkpoint)
            if actual_hash != selected["checkpoint_sha256"]:
                raise RuntimeError("Experimental V2 DQN checkpoint failed SHA-256 integrity check")

            settings = load_yaml(PROJECT_ROOT / "configs/v2/training.yaml")["dqn_overrides"]
            config = deep_merge(deepcopy(load_agent_config("dqn")), {
                "agent": {"seed": 2026, "device": "auto"},
                "model": {"hidden_sizes": settings["model_hidden_sizes"]},
                "replay_buffer": {
                    "capacity": settings["replay_capacity"],
                    "batch_size": settings["batch_size"],
                    "warmup_steps": settings["warmup_steps"],
                },
                "optimization": {
                    "learning_rate": settings["learning_rate"],
                    "reward_scale": settings["reward_scale"],
                    "target_update_frequency": settings["target_update_frequency"],
                },
                "exploration": {"epsilon_decay_steps": settings["epsilon_decay_steps"]},
            })
            env = V2HVACEnv(shield_enabled=True)
            agent = DQNAgent(env.observation_space, env.action_space, config=config)
            agent.load(checkpoint)
            env.close()
            self._agent = agent
            self._explainer = V2PolicyExplainer.from_agent(agent)
            self._metadata = {
                **agent.metadata(),
                "role": "experimental_development_controller",
                "development_status": summary["development_status"],
                "eligible_for_demo_replacement": False,
                "held_out_used": False,
                "checkpoint": str(selected["checkpoint"]).replace("\\", "/"),
                "checkpoint_sha256": actual_hash,
                "training_seed": 2026,
                "development_gates": summary["development_gates"],
            }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
