"""Small psychrometric helpers for bounded V2 humidity mass balance."""

from __future__ import annotations

import math


STANDARD_PRESSURE_KPA = 101.325


def saturation_vapor_pressure_kpa(temperature_c: float) -> float:
    """Tetens approximation over the simulator's normal operating range."""

    return 0.61078 * math.exp(
        17.2694 * temperature_c / (temperature_c + 237.29)
    )


def humidity_ratio(
    temperature_c: float,
    relative_humidity_pct: float,
    pressure_kpa: float = STANDARD_PRESSURE_KPA,
) -> float:
    relative_humidity = min(max(relative_humidity_pct / 100.0, 0.0), 1.0)
    vapor_pressure = relative_humidity * saturation_vapor_pressure_kpa(
        temperature_c
    )
    return 0.62198 * vapor_pressure / max(pressure_kpa - vapor_pressure, 1e-6)


def relative_humidity_pct(
    temperature_c: float,
    humidity_ratio_kg_per_kg: float,
    pressure_kpa: float = STANDARD_PRESSURE_KPA,
) -> float:
    ratio = max(humidity_ratio_kg_per_kg, 0.0)
    vapor_pressure = pressure_kpa * ratio / (0.62198 + ratio)
    relative = vapor_pressure / max(
        saturation_vapor_pressure_kpa(temperature_c), 1e-6
    )
    return float(min(max(relative * 100.0, 0.0), 100.0))
