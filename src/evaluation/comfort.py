"""Comfort and indoor-air-quality metric helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ViolationMetrics:
    """Frequency, duration, and magnitude of one constraint violation."""

    steps: int
    total_steps: int
    timestep_hours: float
    cumulative_magnitude: float

    @property
    def percentage(self) -> float:
        return 100.0 * self.steps / self.total_steps if self.total_steps else 0.0

    @property
    def hours(self) -> float:
        return self.steps * self.timestep_hours

    @property
    def average_magnitude(self) -> float:
        return self.cumulative_magnitude / self.total_steps if self.total_steps else 0.0
