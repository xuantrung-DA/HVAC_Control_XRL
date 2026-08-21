"""Faithful, local explanations for the experimental V2 DQN policy.

The explainer deliberately separates the learned policy's proposal from the
deterministic safety shield's intervention.  Feature ablation is associational:
it measures a local Q-margin response and is never presented as a causal claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from src.agents.dqn import DQNAgent
from src.envs.building import HVACAction
from src.envs.v2.observation import V2_OBSERVATION_NAMES


_FRONTEND_FEATURES = (
    "indoor_temperature_c",
    "indoor_relative_humidity_pct",
    "occupancy",
    "co2_ppm",
    "electricity_price_per_kwh",
)


@dataclass(frozen=True)
class V2PolicyExplainer:
    agent: DQNAgent
    low: np.ndarray
    high: np.ndarray

    @classmethod
    def from_agent(cls, agent: DQNAgent) -> "V2PolicyExplainer":
        return cls(agent=agent, low=agent.scaler.low.copy(), high=agent.scaler.high.copy())

    def explain(self, observation: np.ndarray, *, counterfactual: bool = True) -> dict[str, Any]:
        values = self._validate(observation)
        q_values = self.agent.action_scores(values)
        action = int(np.argmax(q_values))
        contrast = int(np.argsort(q_values)[-2])
        margin = float(q_values[action] - q_values[contrast])
        reference = (self.low + self.high) / 2.0
        contributions: list[dict[str, Any]] = []

        for index, name in enumerate(V2_OBSERVATION_NAMES):
            ablated = values.copy()
            ablated[index] = reference[index]
            scores = self.agent.action_scores(ablated)
            ablated_margin = float(scores[action] - scores[contrast])
            signed = margin - ablated_margin
            contributions.append(
                {
                    "feature": name,
                    "value": float(values[index]),
                    "reference_value": float(reference[index]),
                    "signed_contribution": signed,
                    "absolute_importance": abs(signed),
                    "direction": (
                        "supports_selected_action" if signed > 1e-8
                        else "opposes_selected_action" if signed < -1e-8
                        else "neutral"
                    ),
                    "ablated_decision_margin": ablated_margin,
                }
            )

        total = sum(item["absolute_importance"] for item in contributions)
        for item in contributions:
            item["absolute_importance_pct"] = (
                100.0 * item["absolute_importance"] / total if total > 1e-12 else 0.0
            )
        ranked = sorted(contributions, key=lambda item: item["absolute_importance"], reverse=True)
        top = ", ".join(item["feature"].replace("_", " ") for item in ranked[:3])
        result: dict[str, Any] = {
            "method": "local_q_margin_feature_ablation",
            "action": action,
            "action_name": HVACAction(action).name,
            "contrast_action": contrast,
            "contrast_action_name": HVACAction(contrast).name,
            "q_values": [float(value) for value in q_values],
            "decision_margin": margin,
            "contributions": contributions,
            "human_readable": (
                f"The experimental DQN proposed {HVACAction(action).name}. Its local Q-margin "
                f"was most sensitive to {top}."
            ),
            "causal_claim": False,
            "limitations": "Local ablation sensitivity; correlated features and off-manifold edits can distort importance.",
        }
        result["counterfactual"] = self._counterfactual(values, action) if counterfactual else None
        return result

    def _counterfactual(self, observation: np.ndarray, original_action: int) -> dict[str, Any]:
        best: tuple[float, int, float, int] | None = None
        for feature in _FRONTEND_FEATURES:
            index = V2_OBSERVATION_NAMES.index(feature)
            span = float(self.high[index] - self.low[index])
            for candidate in np.linspace(self.low[index], self.high[index], 41, dtype=np.float32):
                if abs(float(candidate) - float(observation[index])) < 1e-9:
                    continue
                edited = observation.copy()
                edited[index] = candidate
                action = int(np.argmax(self.agent.action_scores(edited)))
                if action == original_action:
                    continue
                distance = abs(float(candidate) - float(observation[index])) / max(span, 1e-9)
                proposal = (distance, index, float(candidate), action)
                if best is None or proposal < best:
                    best = proposal
        if best is None:
            return {
                "found": False,
                "action_changed": False,
                "within_bounds": True,
                "changes": [],
                "counterfactual_action": None,
                "counterfactual_action_name": None,
                "normalized_l1_distance": None,
                "human_readable": "No single-feature counterfactual was found within the declared physical bounds.",
            }
        distance, index, candidate, action = best
        feature = V2_OBSERVATION_NAMES[index]
        original = float(observation[index])
        verified = observation.copy()
        verified[index] = candidate
        verified_action = int(np.argmax(self.agent.action_scores(verified)))
        return {
            "found": True,
            "action_changed": verified_action != original_action,
            "within_bounds": bool(self.low[index] <= candidate <= self.high[index]),
            "changes": [{
                "feature": feature,
                "original_value": original,
                "counterfactual_value": candidate,
                "delta": candidate - original,
            }],
            "counterfactual_action": action,
            "counterfactual_action_name": HVACAction(action).name,
            "normalized_l1_distance": distance,
            "human_readable": (
                f"If {feature.replace('_', ' ')} changed from {original:.2f} to {candidate:.2f}, "
                f"the experimental policy would propose {HVACAction(action).name}."
            ),
        }

    def _validate(self, observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (len(V2_OBSERVATION_NAMES),):
            raise ValueError(f"Expected {len(V2_OBSERVATION_NAMES)} V2 features, received {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("V2 explanation input contains non-finite values")
        return np.clip(values, self.low, self.high)


def explain_shield(control: dict[str, Any]) -> dict[str, Any]:
    """Return a distinct, rule-grounded explanation of shield behavior."""

    shield = control.get("shield") or {}
    proposed = control.get("proposed_action")
    executed = control.get("executed_action")
    intervention = bool(shield.get("intervention", proposed != executed))
    decision = str(shield.get("decision", "ALLOW"))
    reason = str(shield.get("reason", "No safety-shield intervention was required."))
    return {
        "method": "deterministic_predictive_constraint_check",
        "decision": decision,
        "intervention": intervention,
        "proposed_action": proposed,
        "proposed_action_name": HVACAction(proposed).name if proposed is not None else None,
        "executed_action": executed,
        "executed_action_name": HVACAction(executed).name if executed is not None else None,
        "constraint": shield.get("constraint"),
        "reason": reason,
        "projection": shield.get("projection"),
        "human_readable": (
            f"Safety shield {decision}: policy proposed "
            f"{HVACAction(proposed).name if proposed is not None else 'N/A'}; "
            f"executed {HVACAction(executed).name if executed is not None else 'N/A'}. {reason}"
        ),
        "causal_claim": False,
    }
