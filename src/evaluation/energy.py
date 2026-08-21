"""Energy and electricity-cost metric helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnergyMetrics:
    """Episode-level HVAC energy outcomes."""

    total_kwh: float
    electricity_cost: float

    @property
    def average_cost_per_kwh(self) -> float:
        return self.electricity_cost / self.total_kwh if self.total_kwh > 0 else 0.0
