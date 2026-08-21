"""Auditable fixed, adaptive, and constrained reward formulations for V2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from src.envs.v2.models import V2BuildingState, V2ExogenousInputs, V2Transition
from src.risk import RiskVector


@dataclass(frozen=True)
class V2RewardBreakdown:
    reward: float
    mode: str
    profile_id: str
    raw_components: Mapping[str, float]
    normalized_components: Mapping[str, float]
    effective_weights: Mapping[str, float]
    weighted_penalties: Mapping[str, float]
    priority_percent: Mapping[str, float]
    occupied: bool
    comfort_violation: bool
    co2_violation: bool
    comfort_margin_c: float
    humidity_margin_pct: float
    co2_margin_ppm: float
    context: Mapping[str, float]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LagrangeConstraintController:
    """Episode-level dual updates for comfort and IAQ constraint budgets."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.comfort_multiplier = float(config["initial_comfort_multiplier"])
        self.co2_multiplier = float(config["initial_co2_multiplier"])

    def update(self, comfort_fraction: float, co2_fraction: float) -> dict[str, float]:
        rate = float(self.config["learning_rate"])
        lower = float(self.config["multiplier_min"])
        upper = float(self.config["multiplier_max"])
        self.comfort_multiplier = float(
            np.clip(
                self.comfort_multiplier
                + rate
                * (
                    comfort_fraction
                    - float(self.config["comfort_violation_target_fraction"])
                ),
                lower,
                upper,
            )
        )
        self.co2_multiplier = float(
            np.clip(
                self.co2_multiplier
                + rate
                * (co2_fraction - float(self.config["co2_violation_target_fraction"])),
                lower,
                upper,
            )
        )
        return self.multipliers()

    def multipliers(self) -> dict[str, float]:
        return {
            "comfort": self.comfort_multiplier,
            "co2": self.co2_multiplier,
        }


class V2RewardModel:
    """Compute exact reward audits from physical metrics and observable context."""

    def __init__(
        self,
        profile: Mapping[str, Any],
        environment_config: Mapping[str, Any],
        mode: str | None = None,
    ) -> None:
        self.profile = profile
        self.comfort = environment_config["comfort"]
        self.mode = mode or str(profile["default_mode"])
        if self.mode not in profile["supported_modes"]:
            raise ValueError(f"Unsupported V2 reward mode: {self.mode}")
        self.lagrange = LagrangeConstraintController(profile["cmdp"])
        self.reset_episode()

    def reset_episode(self) -> None:
        self.episode_steps = 0
        self.episode_comfort_violations = 0
        self.episode_co2_violations = 0

    def calculate(
        self,
        *,
        state: V2BuildingState,
        inputs: V2ExogenousInputs,
        transition: V2Transition,
        previous_action: int,
        action: float,
        decision_risk: RiskVector,
        control_change_magnitude: float | None = None,
    ) -> V2RewardBreakdown:
        occupied = inputs.occupancy > 0
        bounds = self._comfort_bounds(occupied)
        temperature_violation = max(
            bounds[0] - state.indoor_temperature_c,
            state.indoor_temperature_c - bounds[1],
            0.0,
        )
        humidity_violation = max(
            bounds[2] - state.indoor_relative_humidity_pct,
            state.indoor_relative_humidity_pct - bounds[3],
            0.0,
        )
        co2_violation_ppm = max(
            state.co2_ppm - float(self.comfort["co2_limit_ppm"]), 0.0
        )
        comfort_violation = temperature_violation > 0.0 or humidity_violation > 0.0
        co2_violation = co2_violation_ppm > 0.0
        temperature_margin = min(
            state.indoor_temperature_c - bounds[0],
            bounds[1] - state.indoor_temperature_c,
        )
        humidity_margin = min(
            state.indoor_relative_humidity_pct - bounds[2],
            bounds[3] - state.indoor_relative_humidity_pct,
        )
        co2_margin = float(self.comfort["co2_limit_ppm"]) - state.co2_ppm
        overcooling = (
            max(action - 1, 0)
            if temperature_margin
            >= float(
                self.profile["adaptive_rules"][
                    "anti_overcooling_temperature_margin_c"
                ]
            )
            and action >= 2
            else 0.0
        )
        raw = {
            "energy": transition.energy.electricity_cost,
            "comfort": temperature_violation + humidity_violation / 10.0,
            "co2": co2_violation_ppm,
            "temperature_violation_c": temperature_violation,
            "humidity_violation_pct": humidity_violation,
            "co2_violation_ppm": co2_violation_ppm,
            "switching": (
                abs(action - previous_action)
                if control_change_magnitude is None
                else float(control_change_magnitude)
            ),
            "peak_power": transition.energy.interval_peak_power_kw,
            "overcooling": overcooling,
        }
        normalization = self.profile["normalization"]
        normalized = {
            "energy": raw["energy"] / float(normalization["electricity_cost_per_step"]),
            "comfort": temperature_violation
            / float(normalization["temperature_violation_c"])
            + humidity_violation / float(normalization["humidity_violation_pct"]),
            "co2": raw["co2"] / float(normalization["co2_violation_ppm"]),
            "switching": raw["switching"] / float(normalization["switch_magnitude"]),
            "peak_power": max(
                raw["peak_power"] - float(normalization["peak_power_kw"]), 0.0
            )
            / float(normalization["peak_power_kw"]),
            "overcooling": raw["overcooling"]
            / float(normalization["overcooling_action_magnitude"]),
        }
        weights = self._weights(decision_risk)
        weighted = {name: normalized[name] * weights[name] for name in normalized}
        reward = -float(sum(weighted.values()))
        priorities = self._priorities(weights)
        self.episode_steps += 1
        self.episode_comfort_violations += int(comfort_violation)
        self.episode_co2_violations += int(co2_violation)
        return V2RewardBreakdown(
            reward=reward,
            mode=self.mode,
            profile_id=str(self.profile["profile_id"]),
            raw_components=raw,
            normalized_components=normalized,
            effective_weights=weights,
            weighted_penalties=weighted,
            priority_percent=priorities,
            occupied=occupied,
            comfort_violation=comfort_violation,
            co2_violation=co2_violation,
            comfort_margin_c=float(temperature_margin),
            humidity_margin_pct=float(humidity_margin),
            co2_margin_ppm=float(co2_margin),
            context=decision_risk.as_dict(),
        )

    def end_episode(self) -> dict[str, float] | None:
        if self.episode_steps == 0 or self.mode != "cmdp_lagrangian":
            return None
        return self.lagrange.update(
            self.episode_comfort_violations / self.episode_steps,
            self.episode_co2_violations / self.episode_steps,
        )

    def _weights(self, risk: RiskVector) -> dict[str, float]:
        base = {name: float(value) for name, value in self.profile["base_weights"].items()}
        if self.mode == "fixed":
            return base
        if self.mode == "cmdp_lagrangian":
            base["comfort"] = self.lagrange.comfort_multiplier
            base["co2"] = self.lagrange.co2_multiplier
            return base
        rules = self.profile["adaptive_rules"]
        comfort_risk = max(risk.thermal_risk, risk.humidity_risk)
        maximum_risk = max(comfort_risk, risk.co2_risk)
        safe_threshold = float(rules["safe_risk_threshold"])
        safe_fraction = (
            1.0 - maximum_risk / safe_threshold
            if maximum_risk <= safe_threshold
            else 0.0
        )
        base["energy"] *= 1.0 + float(rules["energy_safe_bonus"]) * safe_fraction
        base["comfort"] *= 1.0 + float(rules["comfort_risk_multiplier"]) * comfort_risk
        base["co2"] *= 1.0 + float(rules["co2_risk_multiplier"]) * risk.co2_risk
        return base

    @staticmethod
    def _priorities(weights: Mapping[str, float]) -> dict[str, float]:
        core = {name: weights[name] for name in ("energy", "comfort", "co2")}
        total = sum(core.values())
        return {name: 100.0 * value / total for name, value in core.items()}

    def _comfort_bounds(self, occupied: bool) -> tuple[float, float, float, float]:
        prefix = "occupied" if occupied else "unoccupied"
        return (
            float(self.comfort[f"{prefix}_temperature_min_c"]),
            float(self.comfort[f"{prefix}_temperature_max_c"]),
            float(self.comfort[f"{prefix}_humidity_min_pct"]),
            float(self.comfort[f"{prefix}_humidity_max_pct"]),
        )
