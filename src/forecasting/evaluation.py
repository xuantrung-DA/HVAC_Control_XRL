"""Evaluation utilities for horizon-specific forecast evidence."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from src.envs.v2.profiles import ScenarioTimeline
from src.forecasting.profile_forecaster import FORECAST_FEATURES, SeasonalProfileForecaster


def evaluate_forecaster(
    forecaster: SeasonalProfileForecaster,
    timelines: Iterable[ScenarioTimeline],
) -> dict[str, Any]:
    samples: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: {"actual": [], "predicted": [], "lower": [], "upper": []}
    )
    scenario_names: set[str] = set()
    timeline_count = 0
    for timeline in timelines:
        timeline_count += 1
        scenario_names.add(timeline.scenario)
        for step, current in enumerate(timeline.inputs):
            bundle = forecaster.predict(current, step)
            for forecast in bundle.forecasts:
                if step + forecast.horizon_steps >= len(timeline.inputs):
                    continue
                target = timeline.inputs[step + forecast.horizon_steps]
                for feature in FORECAST_FEATURES:
                    value = forecast.values[feature]
                    bucket = samples[(feature, forecast.horizon_steps)]
                    bucket["actual"].append(float(getattr(target, feature)))
                    bucket["predicted"].append(value.point)
                    bucket["lower"].append(value.lower)
                    bucket["upper"].append(value.upper)
    metrics: list[dict[str, Any]] = []
    for (feature, horizon), values in sorted(samples.items()):
        actual = np.asarray(values["actual"])
        predicted = np.asarray(values["predicted"])
        lower = np.asarray(values["lower"])
        upper = np.asarray(values["upper"])
        errors = predicted - actual
        metrics.append(
            {
                "feature": feature,
                "horizon_steps": horizon,
                "horizon_hours": horizon * 0.25,
                "samples": int(actual.size),
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(math.sqrt(np.mean(np.square(errors)))),
                "bias": float(np.mean(errors)),
                "interval_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
                "mean_interval_width": float(np.mean(upper - lower)),
            }
        )
    return {
        "timeline_count": timeline_count,
        "scenarios": sorted(scenario_names),
        "held_out_used": False,
        "metrics": metrics,
    }
