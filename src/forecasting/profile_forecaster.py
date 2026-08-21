"""Leakage-resistant seasonal forecasts with online bias correction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.envs.v2.models import V2ExogenousInputs
from src.envs.v2.profiles import ScenarioTimeline


FORECAST_FEATURES = (
    "outdoor_temperature_c",
    "outdoor_relative_humidity_pct",
    "solar_radiation_w_per_m2",
    "occupancy",
    "electricity_price_per_kwh",
)


@dataclass(frozen=True)
class ForecastValue:
    point: float
    lower: float
    upper: float
    standard_deviation: float


@dataclass(frozen=True)
class HorizonForecast:
    horizon_steps: int
    horizon_hours: float
    target_step: int
    values: Mapping[str, ForecastValue]

    def as_dict(self) -> dict[str, Any]:
        return {
            "horizon_steps": self.horizon_steps,
            "horizon_hours": self.horizon_hours,
            "target_step": self.target_step,
            "values": {name: asdict(value) for name, value in self.values.items()},
        }


@dataclass(frozen=True)
class ForecastBundle:
    issued_at_step: int
    source: str
    forecasts: tuple[HorizonForecast, ...]
    fault_mode: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "issued_at_step": self.issued_at_step,
            "source": self.source,
            "fault_mode": self.fault_mode,
            "forecasts": [item.as_dict() for item in self.forecasts],
        }


class SeasonalProfileForecaster:
    """Fit time-of-day means on development data and anchor to observations."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config["forecasting"]
        self.horizons = tuple(int(value) for value in self.config["horizons_steps"])
        self.steps_per_day = int(24 * 60 / int(self.config["timestep_minutes"]))
        self._means: dict[str, np.ndarray] = {}
        self._std: dict[str, np.ndarray] = {}
        self.fit_metadata: dict[str, Any] = {}

    def fit(self, timelines: Sequence[ScenarioTimeline], *, allowed_scenarios: set[str]) -> None:
        if not timelines:
            raise ValueError("At least one development timeline is required")
        observed_scenarios = {timeline.scenario for timeline in timelines}
        if not observed_scenarios.issubset(allowed_scenarios):
            leaked = sorted(observed_scenarios - allowed_scenarios)
            raise ValueError(f"Forecast fit contains forbidden scenarios: {leaked}")
        if any(len(timeline.inputs) != self.steps_per_day for timeline in timelines):
            raise ValueError("All forecast timelines must span exactly one day")
        for feature in FORECAST_FEATURES:
            samples = np.stack(
                [np.array([float(getattr(item, feature)) for item in timeline.inputs]) for timeline in timelines]
            )
            self._means[feature] = samples.mean(axis=0)
            self._std[feature] = samples.std(axis=0, ddof=0)
        self.fit_metadata = {
            "scenarios": sorted(observed_scenarios),
            "seeds": sorted({timeline.seed for timeline in timelines}),
            "timeline_count": len(timelines),
            "held_out_used": False,
        }

    def predict(
        self,
        current: V2ExogenousInputs,
        current_step: int,
        *,
        planned_events: Sequence[Mapping[str, Any]] = (),
    ) -> ForecastBundle:
        if not self._means:
            raise RuntimeError("Forecaster must be fitted before predict")
        if not 0 <= current_step < self.steps_per_day:
            raise ValueError("current_step is outside the daily timeline")
        predictions: list[HorizonForecast] = []
        quantile = float(self.config["normal_quantile"])
        decay_steps = float(self.config["bias_decay_steps"])
        for horizon in self.horizons:
            target_step = min(current_step + horizon, self.steps_per_day - 1)
            decay = float(np.exp(-horizon / decay_steps))
            values: dict[str, ForecastValue] = {}
            for feature in FORECAST_FEATURES:
                current_value = float(getattr(current, feature))
                current_mean = float(self._means[feature][current_step])
                target_mean = float(self._means[feature][target_step])
                point = target_mean + decay * (current_value - current_mean)
                if feature == "occupancy":
                    target_hour = target_step * int(self.config["timestep_minutes"]) / 60.0
                    point += self._planned_occupancy_adjustment(target_hour, planned_events)
                minimum_std = float(self.config["minimum_standard_deviation"][feature])
                standard_deviation = max(
                    minimum_std,
                    float(self._std[feature][target_step]) * np.sqrt(max(horizon, 1) / 4.0),
                )
                lower_bound, upper_bound = self.config["physical_bounds"][feature]
                point = float(np.clip(point, float(lower_bound), float(upper_bound)))
                lower = float(np.clip(point - quantile * standard_deviation, lower_bound, upper_bound))
                upper = float(np.clip(point + quantile * standard_deviation, lower_bound, upper_bound))
                values[feature] = ForecastValue(point, lower, upper, standard_deviation)
            predictions.append(
                HorizonForecast(
                    horizon_steps=horizon,
                    horizon_hours=horizon * int(self.config["timestep_minutes"]) / 60.0,
                    target_step=target_step,
                    values=values,
                )
            )
        return ForecastBundle(
            issued_at_step=current_step,
            source="seasonal_profile_with_online_bias_correction",
            forecasts=tuple(predictions),
        )

    @staticmethod
    def _planned_occupancy_adjustment(
        target_hour: float, events: Sequence[Mapping[str, Any]]
    ) -> float:
        adjustment = 0.0
        for event in events:
            if not event.get("forecast_visible", False):
                continue
            if event.get("event") not in {"meeting", "occupancy_surge"}:
                continue
            if float(event["start_hour"]) <= target_hour < float(event["end_hour"]):
                # Scenario capacity is locked at 100 in the forecast config bounds.
                adjustment += 100.0 * float(event.get("additional_fraction_of_capacity", 0.0))
        return adjustment

    def model_state(self) -> dict[str, Any]:
        if not self._means:
            raise RuntimeError("Forecaster has not been fitted")
        return {
            "method": self.config["method"],
            "features": list(FORECAST_FEATURES),
            "horizons_steps": list(self.horizons),
            "fit_metadata": self.fit_metadata,
            "seasonal_mean": {key: value.tolist() for key, value in self._means.items()},
            "seasonal_standard_deviation": {key: value.tolist() for key, value in self._std.items()},
        }

    def load_model_state(self, state: Mapping[str, Any]) -> None:
        if tuple(int(value) for value in state["horizons_steps"]) != self.horizons:
            raise ValueError("Forecast artifact horizons do not match configuration")
        if tuple(state["features"]) != FORECAST_FEATURES:
            raise ValueError("Forecast artifact features do not match implementation")
        self._means = {
            feature: np.asarray(state["seasonal_mean"][feature], dtype=np.float64)
            for feature in FORECAST_FEATURES
        }
        self._std = {
            feature: np.asarray(
                state["seasonal_standard_deviation"][feature], dtype=np.float64
            )
            for feature in FORECAST_FEATURES
        }
        if any(values.shape != (self.steps_per_day,) for values in self._means.values()):
            raise ValueError("Forecast artifact has invalid seasonal array shape")
        self.fit_metadata = dict(state["fit_metadata"])


class ForecastFaultInjector:
    """Apply explicit fault modes without changing the physical timeline."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config["failure_injection"]

    def apply(
        self,
        bundle: ForecastBundle,
        mode: str,
        *,
        stale_bundle: ForecastBundle | None = None,
    ) -> ForecastBundle:
        if mode not in self.config["supported_modes"]:
            raise ValueError(f"Unsupported forecast fault mode: {mode}")
        if mode == "unavailable":
            return ForecastBundle(bundle.issued_at_step, "fault_injector", tuple(), mode)
        if mode == "stale":
            if stale_bundle is None:
                raise ValueError("stale fault requires a prior forecast bundle")
            return ForecastBundle(
                bundle.issued_at_step,
                "fault_injector",
                stale_bundle.forecasts,
                mode,
            )
        offsets = self.config["biased_offsets"]
        forecasts: list[HorizonForecast] = []
        for forecast in bundle.forecasts:
            values: dict[str, ForecastValue] = {}
            for feature, value in forecast.values.items():
                offset = float(offsets.get(feature, 0.0))
                values[feature] = ForecastValue(
                    point=value.point + offset,
                    lower=value.lower + offset,
                    upper=value.upper + offset,
                    standard_deviation=value.standard_deviation,
                )
            forecasts.append(
                HorizonForecast(
                    forecast.horizon_steps,
                    forecast.horizon_hours,
                    forecast.target_step,
                    values,
                )
            )
        return ForecastBundle(bundle.issued_at_step, "fault_injector", tuple(forecasts), mode)
