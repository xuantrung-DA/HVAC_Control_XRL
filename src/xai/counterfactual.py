"""Bounded counterfactual search for DQN HVAC actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Mapping

import numpy as np

from src.envs.building import HVACAction
from src.envs.hvac_env import OBSERVATION_NAMES
from src.xai.feature_attribution import (
    DQNLikePolicy,
    FEATURE_LABELS,
    batch_action_scores,
)


@dataclass(frozen=True)
class FeatureChange:
    feature: str
    original_value: float
    counterfactual_value: float
    delta: float

    def as_dict(self) -> dict[str, str | float]:
        return asdict(self)


@dataclass(frozen=True)
class CounterfactualResult:
    found: bool
    original_action: int
    original_action_name: str
    counterfactual_action: int | None
    counterfactual_action_name: str | None
    original_q_values: tuple[float, ...]
    counterfactual_q_values: tuple[float, ...] | None
    changes: tuple[FeatureChange, ...]
    normalized_l1_distance: float | None
    within_bounds: bool
    action_changed: bool
    search_strategy: str
    human_readable: str
    causal_claim: bool = False

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["changes"] = [change.as_dict() for change in self.changes]
        return result


class DQNCounterfactualExplainer:
    """Find sparse, bounded state perturbations that alter a DQN action."""

    def __init__(
        self,
        policy: DQNLikePolicy,
        feature_config: Mapping[str, Mapping[str, float | int]],
        *,
        two_feature_fallback: Mapping[str, Any] | None = None,
        feature_names: tuple[str, ...] = OBSERVATION_NAMES,
    ) -> None:
        self.policy = policy
        self.feature_names = feature_names
        self.feature_indices = {name: index for index, name in enumerate(feature_names)}
        unknown = set(feature_config) - set(self.feature_indices)
        if unknown:
            raise ValueError(f"Unknown counterfactual features: {sorted(unknown)}")
        self.feature_config = {
            name: self._validate_spec(name, spec)
            for name, spec in feature_config.items()
        }
        self.fallback = dict(two_feature_fallback or {})

    def explain(
        self,
        observation: np.ndarray,
        *,
        preferred_features: list[str] | None = None,
    ) -> CounterfactualResult:
        raw = np.asarray(observation, dtype=np.float32)
        if raw.shape != (len(self.feature_names),) or not np.all(np.isfinite(raw)):
            raise ValueError("observation must be a finite canonical state vector")
        original_scores = self.policy.action_scores(raw)
        original_action = int(np.argmax(original_scores))

        one_feature_candidates = self._single_feature_candidates(raw)
        result = self._select_best(
            raw,
            original_action,
            original_scores,
            one_feature_candidates,
            strategy="one_feature_grid",
        )
        if result is not None:
            return result

        if bool(self.fallback.get("enabled", True)):
            candidates = self._two_feature_candidates(raw, preferred_features)
            result = self._select_best(
                raw,
                original_action,
                original_scores,
                candidates,
                strategy="two_feature_coarse_grid",
            )
            if result is not None:
                return result

        return CounterfactualResult(
            found=False,
            original_action=original_action,
            original_action_name=HVACAction(original_action).name,
            counterfactual_action=None,
            counterfactual_action_name=None,
            original_q_values=tuple(float(value) for value in original_scores),
            counterfactual_q_values=None,
            changes=(),
            normalized_l1_distance=None,
            within_bounds=True,
            action_changed=False,
            search_strategy="bounded_grid_exhausted",
            human_readable=(
                f"No sparse bounded perturbation changed the DQN action from "
                f"{HVACAction(original_action).name}. This does not prove the action "
                "is invariant outside the searched grid."
            ),
        )

    def _single_feature_candidates(self, raw: np.ndarray) -> np.ndarray:
        candidates: list[np.ndarray] = []
        for name, spec in self.feature_config.items():
            index = self.feature_indices[name]
            for value in self._grid(spec):
                if np.isclose(value, raw[index], atol=float(spec["step"]) / 10.0):
                    continue
                candidate = raw.copy()
                candidate[index] = value
                candidates.append(candidate)
        return np.asarray(candidates, dtype=np.float32)

    def _two_feature_candidates(
        self, raw: np.ndarray, preferred_features: list[str] | None
    ) -> np.ndarray:
        ordered = [
            name
            for name in (preferred_features or list(self.feature_config))
            if name in self.feature_config
        ]
        ordered.extend(name for name in self.feature_config if name not in ordered)
        maximum_pairs = int(self.fallback.get("maximum_feature_pairs", 3))
        count = int(self.fallback.get("candidates_per_feature", 9))
        candidates: list[np.ndarray] = []
        for first, second in list(combinations(ordered, 2))[:maximum_pairs]:
            first_values = self._coarse_values(self.feature_config[first], count)
            second_values = self._coarse_values(self.feature_config[second], count)
            first_index = self.feature_indices[first]
            second_index = self.feature_indices[second]
            for first_value in first_values:
                for second_value in second_values:
                    if np.isclose(first_value, raw[first_index]) or np.isclose(
                        second_value, raw[second_index]
                    ):
                        continue
                    candidate = raw.copy()
                    candidate[first_index] = first_value
                    candidate[second_index] = second_value
                    candidates.append(candidate)
        return np.asarray(candidates, dtype=np.float32)

    def _select_best(
        self,
        raw: np.ndarray,
        original_action: int,
        original_scores: np.ndarray,
        candidates: np.ndarray,
        *,
        strategy: str,
    ) -> CounterfactualResult | None:
        if candidates.size == 0:
            return None
        scores = batch_action_scores(self.policy, candidates)
        actions = np.argmax(scores, axis=1)
        valid_indices = np.flatnonzero(actions != original_action)
        if not len(valid_indices):
            return None
        distances = np.array(
            [self._normalized_distance(raw, candidates[index]) for index in valid_indices]
        )
        best_index = int(valid_indices[int(np.argmin(distances))])
        counterfactual = candidates[best_index]
        counterfactual_scores = scores[best_index]
        counterfactual_action = int(actions[best_index])
        changes = tuple(
            FeatureChange(
                feature=name,
                original_value=float(raw[index]),
                counterfactual_value=float(counterfactual[index]),
                delta=float(counterfactual[index] - raw[index]),
            )
            for name, index in self.feature_indices.items()
            if name in self.feature_config
            and not np.isclose(raw[index], counterfactual[index])
        )
        within_bounds = all(
            float(self.feature_config[change.feature]["minimum"])
            <= change.counterfactual_value
            <= float(self.feature_config[change.feature]["maximum"])
            for change in changes
        )
        human_changes = " and ".join(self._format_change(change) for change in changes)
        human = (
            f"DQN selected {HVACAction(original_action).name}. Under the model, "
            f"changing {human_changes} switches the predicted action to "
            f"{HVACAction(counterfactual_action).name}. This is a local model "
            "counterfactual, not a causal claim."
        )
        return CounterfactualResult(
            found=True,
            original_action=original_action,
            original_action_name=HVACAction(original_action).name,
            counterfactual_action=counterfactual_action,
            counterfactual_action_name=HVACAction(counterfactual_action).name,
            original_q_values=tuple(float(value) for value in original_scores),
            counterfactual_q_values=tuple(float(value) for value in counterfactual_scores),
            changes=changes,
            normalized_l1_distance=float(self._normalized_distance(raw, counterfactual)),
            within_bounds=within_bounds,
            action_changed=counterfactual_action != original_action,
            search_strategy=strategy,
            human_readable=human,
        )

    def _normalized_distance(self, original: np.ndarray, candidate: np.ndarray) -> float:
        distance = 0.0
        for name, spec in self.feature_config.items():
            index = self.feature_indices[name]
            span = float(spec["maximum"]) - float(spec["minimum"])
            distance += abs(float(candidate[index] - original[index])) / span
        return distance

    @staticmethod
    def _validate_spec(
        name: str, spec: Mapping[str, float | int]
    ) -> dict[str, float | int]:
        required = {"minimum", "maximum", "step"}
        if not required.issubset(spec):
            raise ValueError(f"Counterfactual feature {name} lacks {required - set(spec)}")
        validated = dict(spec)
        if float(validated["minimum"]) >= float(validated["maximum"]):
            raise ValueError(f"Counterfactual feature {name} has invalid bounds")
        if float(validated["step"]) <= 0:
            raise ValueError(f"Counterfactual feature {name} has invalid step")
        return validated

    @staticmethod
    def _grid(spec: Mapping[str, float | int]) -> np.ndarray:
        minimum = float(spec["minimum"])
        maximum = float(spec["maximum"])
        step = float(spec["step"])
        decimals = int(spec.get("decimals", 4))
        values = np.arange(minimum, maximum + step / 2.0, step)
        return np.round(np.clip(values, minimum, maximum), decimals=decimals)

    @staticmethod
    def _coarse_values(spec: Mapping[str, float | int], count: int) -> np.ndarray:
        decimals = int(spec.get("decimals", 4))
        return np.round(
            np.linspace(float(spec["minimum"]), float(spec["maximum"]), count),
            decimals=decimals,
        )

    @staticmethod
    def _format_change(change: FeatureChange) -> str:
        label = FEATURE_LABELS.get(change.feature, change.feature)
        if change.feature == "indoor_temperature_c":
            original = f"{change.original_value:.1f}°C"
            counterfactual = f"{change.counterfactual_value:.1f}°C"
        elif change.feature == "occupancy":
            original = f"{change.original_value:.0f} people"
            counterfactual = f"{change.counterfactual_value:.0f} people"
        elif change.feature == "co2_ppm":
            original = f"{change.original_value:.0f} ppm"
            counterfactual = f"{change.counterfactual_value:.0f} ppm"
        elif change.feature == "electricity_price_per_kwh":
            original = f"{change.original_value:.3f}/kWh"
            counterfactual = f"{change.counterfactual_value:.3f}/kWh"
        else:
            original = f"{change.original_value:g}"
            counterfactual = f"{change.counterfactual_value:g}"
        return f"{label} from {original} to {counterfactual}"
