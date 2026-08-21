"""Local feature attribution for discrete DQN HVAC decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol

import numpy as np
import torch

from src.envs.building import HVACAction
from src.envs.hvac_env import OBSERVATION_NAMES


FEATURE_LABELS = {
    "indoor_temperature_c": "indoor temperature",
    "outdoor_temperature_c": "outdoor temperature",
    "relative_humidity_pct": "relative humidity",
    "occupancy": "occupancy",
    "co2_ppm": "CO2 concentration",
    "electricity_price_per_kwh": "electricity price",
    "time_sin": "time of day",
    "time_cos": "time of day",
    "hvac_action": "previous HVAC state",
}


class DQNLikePolicy(Protocol):
    """The small policy surface required by the XAI implementation."""

    device: torch.device
    online_network: torch.nn.Module
    scaler: Any

    def action_scores(self, observation: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    value: float
    reference_value: float
    signed_contribution: float
    signed_percentage: float
    absolute_importance: float
    absolute_importance_pct: float
    direction: str
    ablation_margin_change: float

    def as_dict(self) -> dict[str, str | float]:
        return asdict(self)


@dataclass(frozen=True)
class AttributionResult:
    method: str
    action: int
    action_name: str
    contrast_action: int
    contrast_action_name: str
    q_values: tuple[float, ...]
    decision_margin: float
    contributions: tuple[FeatureContribution, ...]
    faithfulness: Mapping[str, float | bool | None]
    human_readable: str
    causal_claim: bool = False

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["contributions"] = [item.as_dict() for item in self.contributions]
        return result


def batch_action_scores(
    policy: DQNLikePolicy, observations: np.ndarray
) -> np.ndarray:
    """Evaluate raw observations in one DQN forward pass."""

    values = np.asarray(observations, dtype=np.float32)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    scaled = policy.scaler.transform(values)
    tensor = torch.as_tensor(scaled, dtype=torch.float32, device=policy.device)
    with torch.no_grad():
        scores = policy.online_network(tensor)
    return scores.detach().cpu().numpy().astype(np.float32)


class DQNFeatureAttributor:
    """Explain a DQN action with Integrated Gradients and ablation checks.

    Attribution targets the selected action's Q-value margin over the current
    runner-up. This is a local explanation of the model, not a causal statement
    about the physical building.
    """

    def __init__(
        self,
        policy: DQNLikePolicy,
        reference_observation: np.ndarray,
        *,
        integration_steps: int = 32,
        feature_names: tuple[str, ...] = OBSERVATION_NAMES,
    ) -> None:
        if integration_steps < 2:
            raise ValueError("integration_steps must be at least 2")
        reference = np.asarray(reference_observation, dtype=np.float32)
        if reference.shape != (len(feature_names),):
            raise ValueError("reference_observation has an invalid shape")
        self.policy = policy
        self.reference = reference
        self.integration_steps = integration_steps
        self.feature_names = feature_names

    def explain(self, observation: np.ndarray) -> AttributionResult:
        raw = np.asarray(observation, dtype=np.float32)
        if raw.shape != self.reference.shape or not np.all(np.isfinite(raw)):
            raise ValueError("observation must be a finite canonical state vector")

        q_values = self.policy.action_scores(raw)
        action = int(np.argmax(q_values))
        ranked = np.argsort(q_values)
        contrast_action = int(ranked[-2])
        decision_margin = float(q_values[action] - q_values[contrast_action])

        scaled_input = self.policy.scaler.transform(raw)
        scaled_reference = self.policy.scaler.transform(self.reference)
        delta = scaled_input - scaled_reference
        alphas = torch.linspace(
            0.0,
            1.0,
            self.integration_steps + 1,
            dtype=torch.float32,
            device=self.policy.device,
        )
        path = torch.as_tensor(
            scaled_reference, dtype=torch.float32, device=self.policy.device
        ).unsqueeze(0) + alphas.unsqueeze(1) * torch.as_tensor(
            delta, dtype=torch.float32, device=self.policy.device
        ).unsqueeze(0)
        path.requires_grad_(True)
        scores = self.policy.online_network(path)
        margins = scores[:, action] - scores[:, contrast_action]
        gradients = torch.autograd.grad(margins.sum(), path)[0]
        # Trapezoidal integration is materially more accurate than a right-hand
        # Riemann sum when the path crosses ReLU boundaries.
        average_gradient_tensor = (
            gradients[0]
            + gradients[-1]
            + 2.0 * gradients[1:-1].sum(dim=0)
        ) / (2.0 * self.integration_steps)
        average_gradient = average_gradient_tensor.detach().cpu().numpy()
        signed = delta * average_gradient

        reference_scores = self.policy.action_scores(self.reference)
        reference_margin = float(
            reference_scores[action] - reference_scores[contrast_action]
        )
        expected_delta = decision_margin - reference_margin
        completeness_delta = float(np.sum(signed))
        completeness_absolute_error = abs(completeness_delta - expected_delta)
        completeness_relative_error = completeness_absolute_error / max(
            abs(expected_delta), 1e-8
        )

        ablated = np.repeat(raw[np.newaxis, :], len(self.feature_names), axis=0)
        for index in range(len(self.feature_names)):
            ablated[index, index] = self.reference[index]
        ablated_scores = batch_action_scores(self.policy, ablated)
        ablated_margins = (
            ablated_scores[:, action] - ablated_scores[:, contrast_action]
        )
        ablation_changes = decision_margin - ablated_margins

        absolute = np.abs(signed)
        denominator = float(np.sum(absolute))
        if denominator <= 1e-12:
            absolute_pct = np.zeros_like(absolute)
            signed_pct = np.zeros_like(signed)
        else:
            absolute_pct = 100.0 * absolute / denominator
            signed_pct = 100.0 * signed / denominator

        correlation = _safe_correlation(absolute, np.abs(ablation_changes))
        contributions = tuple(
            FeatureContribution(
                feature=name,
                value=float(raw[index]),
                reference_value=float(self.reference[index]),
                signed_contribution=float(signed[index]),
                signed_percentage=float(signed_pct[index]),
                absolute_importance=float(absolute[index]),
                absolute_importance_pct=float(absolute_pct[index]),
                direction=(
                    "supports_selected_action"
                    if signed[index] > 1e-10
                    else "opposes_selected_action"
                    if signed[index] < -1e-10
                    else "neutral"
                ),
                ablation_margin_change=float(ablation_changes[index]),
            )
            for index, name in enumerate(self.feature_names)
        )
        ordered = sorted(
            contributions, key=lambda item: item.absolute_importance, reverse=True
        )
        influential = [item for item in ordered if item.absolute_importance_pct >= 1.0]
        leading = influential[:2] or ordered[:1]
        feature_phrase = " and ".join(_describe_feature(item) for item in leading)
        human = (
            f"DQN selected {HVACAction(action).name} mainly because its local "
            f"decision margin was most associated with {feature_phrase}. "
            "These are model attributions, not causal effects."
        )

        top_index = int(np.argmax(absolute)) if len(absolute) else 0
        faithfulness: dict[str, float | bool | None] = {
            "integrated_gradients_sum": completeness_delta,
            "expected_margin_change": expected_delta,
            "completeness_absolute_error": float(completeness_absolute_error),
            "completeness_relative_error": float(completeness_relative_error),
            "absolute_attribution_ablation_correlation": correlation,
            "top_feature_changes_margin_when_ablated": bool(
                abs(float(ablation_changes[top_index])) > 1e-8
            ),
        }
        return AttributionResult(
            method="integrated_gradients_decision_margin",
            action=action,
            action_name=HVACAction(action).name,
            contrast_action=contrast_action,
            contrast_action_name=HVACAction(contrast_action).name,
            q_values=tuple(float(value) for value in q_values),
            decision_margin=decision_margin,
            contributions=contributions,
            faithfulness=faithfulness,
            human_readable=human,
        )


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _describe_feature(contribution: FeatureContribution) -> str:
    name = contribution.feature
    value = contribution.value
    if name in {"indoor_temperature_c", "outdoor_temperature_c"}:
        formatted = f"{value:.1f}°C"
    elif name == "relative_humidity_pct":
        formatted = f"{value:.0f}%"
    elif name == "occupancy":
        formatted = f"{value:.0f} people"
    elif name == "co2_ppm":
        formatted = f"{value:.0f} ppm"
    elif name == "electricity_price_per_kwh":
        formatted = f"{value:.3f}/kWh"
    elif name == "time_sin":
        formatted = f"sin={value:.2f}"
    elif name == "time_cos":
        formatted = f"cos={value:.2f}"
    elif name == "hvac_action":
        formatted = HVACAction(int(round(value))).name
    else:
        formatted = f"{value:.2f}"
    return f"{FEATURE_LABELS.get(name, name)} ({formatted})"
