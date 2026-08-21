"""Application-level access to DQN attribution and counterfactuals."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.envs.hvac_env import OBSERVATION_NAMES
from src.services.agent_service import AgentService
from src.utils.config import load_config
from src.xai.counterfactual import DQNCounterfactualExplainer
from src.xai.feature_attribution import DQNFeatureAttributor


class ExplanationService:
    def __init__(self, agent_service: AgentService) -> None:
        config = load_config("xai")
        reference = np.array(
            [config["attribution"]["reference_state"][name] for name in OBSERVATION_NAMES],
            dtype=np.float32,
        )
        self.attributor = DQNFeatureAttributor(
            agent_service.agent,
            reference,
            integration_steps=int(config["attribution"]["integration_steps"]),
        )
        self.counterfactual = DQNCounterfactualExplainer(
            agent_service.agent,
            config["counterfactual"]["searchable_features"],
            two_feature_fallback=config["counterfactual"]["two_feature_fallback"],
        )

    def explain(
        self, observation: np.ndarray, *, include_counterfactual: bool = True
    ) -> dict[str, Any]:
        attribution = self.attributor.explain(observation)
        result: dict[str, Any] = {"attribution": attribution.as_dict()}
        if include_counterfactual:
            preferred = [
                item.feature
                for item in sorted(
                    attribution.contributions,
                    key=lambda item: item.absolute_importance,
                    reverse=True,
                )
            ]
            result["counterfactual"] = self.counterfactual.explain(
                observation, preferred_features=preferred
            ).as_dict()
        else:
            result["counterfactual"] = None
        return result
