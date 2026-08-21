"""Observable, bounded context/risk representation for V2 policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from src.envs.v2.models import V2BuildingState, V2ExogenousInputs
from src.forecasting import ForecastBundle
from src.risk.monitoring import MonitoringSnapshot


@dataclass(frozen=True)
class RiskVector:
    thermal_risk: float
    humidity_risk: float
    co2_risk: float
    occupancy_surge: float
    forecast_uncertainty: float
    forecast_error: float
    energy_peak_risk: float
    weather_reliability: float
    occupancy_reliability: float
    price_reliability: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class ObservableRiskAnalyzer:
    """Calculate risk from state, trends, forecasts, and declared reliability."""

    def __init__(
        self,
        risk_config: Mapping[str, Any],
        environment_config: Mapping[str, Any],
    ) -> None:
        self.config = risk_config["risk"]
        self.comfort = environment_config["comfort"]

    def analyze(
        self,
        state: V2BuildingState,
        inputs: V2ExogenousInputs,
        monitoring: MonitoringSnapshot,
        forecast: ForecastBundle,
        reliability: Mapping[str, float],
    ) -> RiskVector:
        occupied = inputs.occupancy > 0
        lower = float(
            self.comfort[
                "occupied_temperature_min_c" if occupied else "unoccupied_temperature_min_c"
            ]
        )
        upper = float(
            self.comfort[
                "occupied_temperature_max_c" if occupied else "unoccupied_temperature_max_c"
            ]
        )
        warning = float(self.config["comfort_warning_margin_c"])
        temperature = state.indoor_temperature_c
        hot_risk = (temperature - (upper - warning)) / (2.0 * warning)
        cold_risk = ((lower + warning) - temperature) / (2.0 * warning)
        current_thermal = float(np.clip(max(hot_risk, cold_risk, 0.0), 0.0, 1.0))
        blend = self.config["blend"]
        thermal_risk = float(
            np.clip(
                float(blend["thermal_current"]) * current_thermal
                + float(blend["thermal_trend"]) * monitoring.thermal_change_score,
                0.0,
                1.0,
            )
        )
        humidity_lower = float(
            self.comfort[
                "occupied_humidity_min_pct" if occupied else "unoccupied_humidity_min_pct"
            ]
        )
        humidity_upper = float(
            self.comfort[
                "occupied_humidity_max_pct" if occupied else "unoccupied_humidity_max_pct"
            ]
        )
        humidity_warning = float(self.config["humidity_warning_margin_pct"])
        high_humidity_risk = (
            state.indoor_relative_humidity_pct
            - (humidity_upper - humidity_warning)
        ) / (2.0 * humidity_warning)
        low_humidity_risk = (
            (humidity_lower + humidity_warning)
            - state.indoor_relative_humidity_pct
        ) / (2.0 * humidity_warning)
        humidity_risk = float(
            np.clip(max(high_humidity_risk, low_humidity_risk, 0.0), 0.0, 1.0)
        )

        warning_co2 = float(self.config["co2_warning_ppm"])
        limit_co2 = float(self.comfort["co2_limit_ppm"])
        current_co2 = float(
            np.clip((state.co2_ppm - warning_co2) / (limit_co2 - warning_co2), 0.0, 1.0)
        )
        co2_risk = float(
            np.clip(
                float(blend["co2_current"]) * current_co2
                + float(blend["co2_trend"]) * monitoring.co2_growth_score,
                0.0,
                1.0,
            )
        )
        uncertainty = self._forecast_uncertainty(forecast)
        price_low = float(self.config["price_low"])
        price_high = float(self.config["price_high"])
        price_risk = float(
            np.clip(
                (inputs.electricity_price_per_kwh - price_low) / (price_high - price_low),
                0.0,
                1.0,
            )
        )
        return RiskVector(
            thermal_risk=thermal_risk,
            humidity_risk=humidity_risk,
            co2_risk=co2_risk,
            occupancy_surge=monitoring.occupancy_surge_score,
            forecast_uncertainty=uncertainty,
            forecast_error=monitoring.forecast_error_score,
            energy_peak_risk=price_risk,
            weather_reliability=float(np.clip(reliability.get("weather", 0.0), 0.0, 1.0)),
            occupancy_reliability=float(np.clip(reliability.get("occupancy", 0.0), 0.0, 1.0)),
            price_reliability=float(np.clip(reliability.get("price", 0.0), 0.0, 1.0)),
        )

    def _forecast_uncertainty(self, forecast: ForecastBundle) -> float:
        if not forecast.forecasts:
            return 1.0
        first = forecast.forecasts[0]
        normalized = [
            value.standard_deviation
            / float(self.config["forecast_uncertainty_scales"][feature])
            for feature, value in first.values.items()
        ]
        return float(np.clip(np.mean(normalized), 0.0, 1.0))
