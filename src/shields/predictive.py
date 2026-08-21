"""Short-horizon physical safety shield with explicit intervention evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Any, Mapping

import numpy as np

from src.envs.v2.models import V2BuildingState, V2ExogenousInputs
from src.envs.v2.physics import TwoR1CBuildingModel
from src.forecasting import ForecastBundle
from src.risk import RiskVector


class ShieldDecisionType(StrEnum):
    ALLOW = "ALLOW"
    CLAMP = "CLAMP"
    REJECT = "REJECT"
    FALLBACK = "FALLBACK"


@dataclass(frozen=True)
class ActionProjection:
    action: int
    final_temperature_c: float
    final_co2_ppm: float
    final_humidity_pct: float
    projected_energy_kwh: float
    projected_cost: float


@dataclass(frozen=True)
class ShieldDecision:
    decision: ShieldDecisionType
    proposed_action: int
    executed_action: int
    intervention: bool
    constraint: str | None
    reason: str
    risk: Mapping[str, float]
    projection: ActionProjection | None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["decision"] = self.decision.value
        return result


class PredictiveSafetyShield:
    """Validate actions with observable risk and conservative 2R1C projections."""

    def __init__(
        self,
        config: Mapping[str, Any],
        environment_config: Mapping[str, Any],
        action_config: Mapping[str, Any],
    ) -> None:
        self.config = config["shield"]
        self.environment = environment_config
        self.comfort = environment_config["comfort"]
        self.model = TwoR1CBuildingModel(environment_config, action_config)

    def decide(
        self,
        *,
        state: V2BuildingState,
        inputs: V2ExogenousInputs,
        proposed_action: int,
        forecast: ForecastBundle,
        risk: RiskVector,
    ) -> ShieldDecision:
        if proposed_action not in range(4):
            raise ValueError("Proposed HVAC action must be one of 0, 1, 2, 3")
        unreliable = min(
            risk.weather_reliability,
            risk.occupancy_reliability,
        ) < float(self.config["reliability_fallback_threshold"])
        high_context_risk = max(
            risk.thermal_risk, risk.humidity_risk, risk.co2_risk, risk.forecast_error
        ) >= float(self.config["high_context_risk_threshold"])
        if (not forecast.forecasts or unreliable) and high_context_risk:
            fallback = self._fallback_action(state, inputs, risk)
            return self._decision(
                ShieldDecisionType.FALLBACK,
                proposed_action,
                fallback,
                "FORECAST_RELIABILITY",
                "Forecast evidence is unavailable or unreliable under high current risk; conservative fallback selected.",
                risk,
                None,
            )

        lower, upper = self._temperature_bounds(inputs.occupancy > 0)
        if (
            state.indoor_temperature_c
            <= lower + float(self.config["overcool_reject_margin_c"])
            and proposed_action >= 2
        ):
            executed = 0 if inputs.occupancy == 0 else 1
            return self._decision(
                ShieldDecisionType.REJECT,
                proposed_action,
                executed,
                "OVERCOOLING",
                "Proposed cooling is unsafe near the lower comfort bound.",
                risk,
                self._project(state, inputs, executed, forecast),
            )

        required = 0
        constraint = None
        if risk.thermal_risk >= float(self.config["thermal_clamp_high_risk"]):
            required, constraint = 3, "THERMAL_RISK"
        elif risk.thermal_risk >= float(self.config["thermal_clamp_medium_risk"]):
            required, constraint = 2, "THERMAL_RISK"
        if risk.co2_risk >= float(self.config["co2_clamp_high_risk"]):
            required, constraint = max(required, 3), "CO2_RISK"
        elif risk.co2_risk >= float(self.config["co2_clamp_medium_risk"]):
            required, constraint = max(required, 2), "CO2_RISK"
        if risk.humidity_risk >= float(
            self.config["humidity_clamp_medium_risk"]
        ):
            required, constraint = max(required, 2), "HUMIDITY_RISK"

        proposed_projection = self._project(state, inputs, proposed_action, forecast)
        if proposed_projection.final_temperature_c > upper + float(
            self.config["projected_temperature_tolerance_c"]
        ):
            required, constraint = max(required, 2), "PROJECTED_THERMAL_LIMIT"
        if proposed_projection.final_co2_ppm > float(self.comfort["co2_limit_ppm"]) + float(
            self.config["projected_co2_tolerance_ppm"]
        ):
            required, constraint = max(required, 2), "PROJECTED_CO2_LIMIT"
        if proposed_action < required:
            executed = required
            return self._decision(
                ShieldDecisionType.CLAMP,
                proposed_action,
                executed,
                constraint,
                f"Action raised from {proposed_action} to {executed} to protect the predicted constraint margin.",
                risk,
                self._project(state, inputs, executed, forecast),
            )
        return self._decision(
            ShieldDecisionType.ALLOW,
            proposed_action,
            proposed_action,
            None,
            "Proposed action remains within configured short-horizon margins.",
            risk,
            proposed_projection,
        )

    def _project(
        self,
        state: V2BuildingState,
        inputs: V2ExogenousInputs,
        action: int,
        forecast: ForecastBundle,
    ) -> ActionProjection:
        projected = state
        energy = 0.0
        cost = 0.0
        for step in range(1, int(self.config["projection_steps"]) + 1):
            projected_inputs = self._projected_inputs(inputs, forecast, step)
            projected, transition = self.model.step(projected, action, projected_inputs)
            energy += transition.energy.controllable_hvac_ventilation_kwh
            cost += transition.energy.electricity_cost
        return ActionProjection(
            action=action,
            final_temperature_c=projected.indoor_temperature_c,
            final_co2_ppm=projected.co2_ppm,
            final_humidity_pct=projected.indoor_relative_humidity_pct,
            projected_energy_kwh=energy,
            projected_cost=cost,
        )

    def _projected_inputs(
        self, inputs: V2ExogenousInputs, forecast: ForecastBundle, step: int
    ) -> V2ExogenousInputs:
        if not forecast.forecasts:
            return replace(inputs, hour=inputs.hour + 0.25 * step)
        target = forecast.forecasts[0]
        fraction = min(step / max(target.horizon_steps, 1), 1.0)
        multiplier = float(self.config["uncertainty_multiplier"])
        temperature = self._interpolate(
            inputs.outdoor_temperature_c,
            target.values["outdoor_temperature_c"].point
            + multiplier * target.values["outdoor_temperature_c"].standard_deviation,
            fraction,
        )
        humidity = self._interpolate(
            inputs.outdoor_relative_humidity_pct,
            target.values["outdoor_relative_humidity_pct"].point,
            fraction,
        )
        occupancy = self._interpolate(
            inputs.occupancy,
            target.values["occupancy"].point
            + multiplier * target.values["occupancy"].standard_deviation,
            fraction,
        )
        return replace(
            inputs,
            outdoor_temperature_c=float(temperature),
            outdoor_relative_humidity_pct=float(np.clip(humidity, 0.0, 100.0)),
            solar_radiation_w_per_m2=max(
                0.0,
                self._interpolate(
                    inputs.solar_radiation_w_per_m2,
                    target.values["solar_radiation_w_per_m2"].point,
                    fraction,
                ),
            ),
            occupancy=int(np.clip(round(occupancy), 0, self.environment["zone"]["maximum_occupancy"])),
            electricity_price_per_kwh=max(
                0.0,
                self._interpolate(
                    inputs.electricity_price_per_kwh,
                    target.values["electricity_price_per_kwh"].point,
                    fraction,
                ),
            ),
            hour=(inputs.hour + 0.25 * step) % 24.0,
        )

    def _fallback_action(
        self, state: V2BuildingState, inputs: V2ExogenousInputs, risk: RiskVector
    ) -> int:
        _, upper = self._temperature_bounds(inputs.occupancy > 0)
        if state.co2_ppm >= 900.0 or state.indoor_temperature_c >= upper or max(
            risk.thermal_risk, risk.humidity_risk, risk.co2_risk
        ) >= 0.75:
            return 3
        if inputs.occupancy > 0:
            return 2
        return 1

    def _temperature_bounds(self, occupied: bool) -> tuple[float, float]:
        prefix = "occupied" if occupied else "unoccupied"
        return (
            float(self.comfort[f"{prefix}_temperature_min_c"]),
            float(self.comfort[f"{prefix}_temperature_max_c"]),
        )

    @staticmethod
    def _interpolate(current: float, target: float, fraction: float) -> float:
        return float(current + fraction * (target - current))

    @staticmethod
    def _decision(
        decision: ShieldDecisionType,
        proposed: int,
        executed: int,
        constraint: str | None,
        reason: str,
        risk: RiskVector,
        projection: ActionProjection | None,
    ) -> ShieldDecision:
        return ShieldDecision(
            decision=decision,
            proposed_action=proposed,
            executed_action=executed,
            intervention=decision is not ShieldDecisionType.ALLOW,
            constraint=constraint,
            reason=reason,
            risk=risk.as_dict(),
            projection=projection,
        )
