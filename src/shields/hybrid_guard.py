"""Deterministic actuator coordination for learning-augmented V2 control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from src.envs.v2.models import V2BuildingState, V2ExogenousInputs
from src.risk import RiskVector


@dataclass(frozen=True)
class HybridControlDecision:
    proposed_cooling_action: int
    executed_cooling_action: int
    cooling_fraction: float
    ventilation_fraction: float
    ventilation_ach: float
    dehumidification_fraction: float
    cooling_intervention: bool
    ventilation_intervention: bool
    reasons: tuple[str, ...]

    @property
    def intervention(self) -> bool:
        return self.cooling_intervention or self.ventilation_intervention

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["intervention"] = self.intervention
        result["reasons"] = list(self.reasons)
        return result


class HybridControlGuard:
    """Separate sensible cooling, IAQ ventilation, and latent control."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        settings = config["hybrid_control"]
        self.settings = settings
        self.cooling = tuple(float(v) for v in settings["cooling"]["command_fractions"])
        self.thermal = settings["thermal_guard"]
        self.ventilation = settings["ventilation"]
        self.ventilation_ach = tuple(float(v) for v in self.ventilation["command_ach"])
        self.dehumidifier = settings["dehumidifier"]
        self.legacy_ventilation = tuple(
            float(v) for v in settings["intervention_accounting"]["legacy_action_ventilation_ach"]
        )
        if len(self.cooling) != 4 or len(self.ventilation_ach) != 4:
            raise ValueError("Hybrid V2 requires four cooling and ventilation levels")

    def decide(
        self,
        *,
        state: V2BuildingState,
        inputs: V2ExogenousInputs,
        proposed_cooling_action: int,
        risk: RiskVector,
    ) -> HybridControlDecision:
        if proposed_cooling_action not in range(4):
            raise ValueError("Cooling proposal must be one of 0, 1, 2, 3")
        action = proposed_cooling_action
        reasons: list[str] = []
        occupied = inputs.occupancy > 0
        temperature = state.indoor_temperature_c
        if not occupied:
            limited = min(action, int(self.thermal["maximum_unoccupied_action"]))
            if limited != action:
                reasons.append("unoccupied_cooling_limit")
            action = limited
        elif temperature >= float(self.thermal["occupied_high_c"]):
            if action < 3:
                reasons.append("occupied_high_temperature")
            action = max(action, 3)
        elif temperature >= float(self.thermal["occupied_medium_c"]):
            if action < 2:
                reasons.append("occupied_warm_temperature")
            action = max(action, 2)
        elif temperature <= float(self.thermal["occupied_cold_stop_c"]):
            if action != 0:
                reasons.append("overcooling_stop")
            action = 0
        elif temperature <= float(self.thermal["occupied_cold_limit_c"]):
            limited = min(action, 1)
            if limited != action:
                reasons.append("overcooling_limit")
            action = limited

        if not occupied:
            ventilation_ach = 0.0
        elif (
            state.co2_ppm >= float(self.ventilation["co2_high_ppm"])
            or risk.co2_risk >= float(self.ventilation["co2_risk_high"])
        ):
            ventilation_ach = self.ventilation_ach[3]
            reasons.append("high_iaq_risk")
        elif (
            state.co2_ppm >= float(self.ventilation["co2_medium_ppm"])
            or risk.co2_risk >= float(self.ventilation["co2_risk_medium"])
        ):
            ventilation_ach = self.ventilation_ach[2]
            reasons.append("moderate_iaq_risk")
        else:
            ventilation_ach = self.ventilation_ach[1]
        maximum_ach = max(self.ventilation_ach)
        ventilation_fraction = ventilation_ach / maximum_ach
        dehumidification = float(
            (occupied or not bool(self.dehumidifier["occupied_only"]))
            and state.indoor_relative_humidity_pct
            >= float(self.dehumidifier["activation_relative_humidity_pct"])
        )
        if dehumidification:
            reasons.append("high_relative_humidity")
        return HybridControlDecision(
            proposed_cooling_action=proposed_cooling_action,
            executed_cooling_action=action,
            cooling_fraction=self.cooling[action],
            ventilation_fraction=ventilation_fraction,
            ventilation_ach=ventilation_ach,
            dehumidification_fraction=dehumidification,
            cooling_intervention=action != proposed_cooling_action,
            ventilation_intervention=abs(
                ventilation_ach - self.legacy_ventilation[proposed_cooling_action]
            ) > 1e-9,
            reasons=tuple(reasons),
        )
