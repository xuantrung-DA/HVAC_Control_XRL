"""Online trend, change, anomaly, and forecast-reliability signals."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class SignalTrend:
    value: float
    delta: float
    second_delta: float
    rolling_mean: float
    rolling_standard_deviation: float
    rolling_slope: float


@dataclass(frozen=True)
class MonitoringSnapshot:
    signals: Mapping[str, SignalTrend]
    occupancy_surge_score: float
    thermal_change_score: float
    co2_growth_score: float
    forecast_error_score: float
    anomaly_detected: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "signals": {name: asdict(value) for name, value in self.signals.items()},
            "occupancy_surge_score": self.occupancy_surge_score,
            "thermal_change_score": self.thermal_change_score,
            "co2_growth_score": self.co2_growth_score,
            "forecast_error_score": self.forecast_error_score,
            "anomaly_detected": self.anomaly_detected,
        }


class OnlineSignalMonitor:
    """Deterministic engineered temporal features with EWMA/CUSUM residuals."""

    SIGNALS = ("occupancy", "indoor_temperature_c", "co2_ppm")

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config["monitoring"]
        window = int(self.config["rolling_window_steps"])
        self._history = {name: deque(maxlen=window) for name in self.SIGNALS}
        self._residual_ewma: dict[str, float] = defaultdict(float)
        self._cusum_positive: dict[str, float] = defaultdict(float)
        self._cusum_negative: dict[str, float] = defaultdict(float)

    def update(
        self,
        observations: Mapping[str, float],
        forecast_residuals: Mapping[str, tuple[float, float]] | None = None,
    ) -> MonitoringSnapshot:
        signals: dict[str, SignalTrend] = {}
        for name in self.SIGNALS:
            value = float(observations[name])
            history = self._history[name]
            previous = history[-1] if history else value
            previous_previous = history[-2] if len(history) >= 2 else previous
            delta = value - previous
            second_delta = value - 2.0 * previous + previous_previous
            history.append(value)
            values = np.asarray(history, dtype=np.float64)
            slope = (
                float(np.polyfit(np.arange(values.size), values, 1)[0])
                if values.size >= 2
                else 0.0
            )
            signals[name] = SignalTrend(
                value=value,
                delta=float(delta),
                second_delta=float(second_delta),
                rolling_mean=float(values.mean()),
                rolling_standard_deviation=float(values.std(ddof=0)),
                rolling_slope=slope,
            )
        forecast_error_score, anomaly_detected = self._update_residuals(
            forecast_residuals or {}
        )
        scales = self.config["score_scales"]
        return MonitoringSnapshot(
            signals=signals,
            occupancy_surge_score=self._positive_score(
                signals["occupancy"].delta,
                float(scales["occupancy_surge_per_step"]),
            ),
            thermal_change_score=self._absolute_score(
                signals["indoor_temperature_c"].rolling_slope,
                float(scales["thermal_change_c_per_step"]),
            ),
            co2_growth_score=self._positive_score(
                signals["co2_ppm"].rolling_slope,
                float(scales["co2_growth_ppm_per_step"]),
            ),
            forecast_error_score=forecast_error_score,
            anomaly_detected=anomaly_detected,
        )

    def _update_residuals(
        self, residuals: Mapping[str, tuple[float, float]]
    ) -> tuple[float, bool]:
        alpha = float(self.config["ewma_alpha"])
        drift = float(self.config["cusum_drift"])
        threshold = float(self.config["cusum_threshold"])
        maximum = 0.0
        detected = False
        for feature, (residual, uncertainty) in residuals.items():
            minimum = float(self.config["minimum_scale"].get(feature, 1.0))
            standardized = float(residual) / max(float(uncertainty), minimum)
            self._residual_ewma[feature] = (
                alpha * abs(standardized)
                + (1.0 - alpha) * self._residual_ewma[feature]
            )
            self._cusum_positive[feature] = max(
                0.0, self._cusum_positive[feature] + standardized - drift
            )
            self._cusum_negative[feature] = min(
                0.0, self._cusum_negative[feature] + standardized + drift
            )
            maximum = max(maximum, self._residual_ewma[feature])
            detected = detected or self._cusum_positive[feature] >= threshold
            detected = detected or -self._cusum_negative[feature] >= threshold
        scale = float(self.config["score_scales"]["forecast_standardized_error"])
        return float(np.clip(maximum / scale, 0.0, 1.0)), detected

    @staticmethod
    def _positive_score(value: float, scale: float) -> float:
        return float(np.clip(max(value, 0.0) / scale, 0.0, 1.0))

    @staticmethod
    def _absolute_score(value: float, scale: float) -> float:
        return float(np.clip(abs(value) / scale, 0.0, 1.0))


class ForecastReliabilityTracker:
    """Convert recent normalized residuals and anomaly evidence to [0, 1]."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config["reliability"]
        self._feature_error: dict[str, float] = defaultdict(float)
        self._available: dict[str, bool] = defaultdict(lambda: True)

    def update(
        self,
        residuals: Mapping[str, tuple[float, float]],
        *,
        unavailable_features: set[str] | None = None,
        anomaly_score: float = 0.0,
    ) -> dict[str, float]:
        alpha = float(self.config["ewma_alpha"])
        unavailable_features = unavailable_features or set()
        for feature, (residual, uncertainty) in residuals.items():
            normalized = abs(float(residual)) / max(float(uncertainty), 1e-6)
            self._feature_error[feature] = (
                alpha * normalized + (1.0 - alpha) * self._feature_error[feature]
            )
            self._available[feature] = feature not in unavailable_features
        for feature in unavailable_features:
            self._available[feature] = False
        reliability: dict[str, float] = {}
        anomaly_penalty = float(self.config["anomaly_penalty"]) * float(
            np.clip(anomaly_score, 0.0, 1.0)
        )
        for domain, features in self.config["domains"].items():
            if any(not self._available[feature] for feature in features):
                reliability[domain] = 0.0
                continue
            errors = [self._feature_error[feature] for feature in features]
            score = float(np.exp(-np.mean(errors))) if errors else 1.0
            reliability[domain] = float(np.clip(score - anomaly_penalty, 0.0, 1.0))
        return reliability
