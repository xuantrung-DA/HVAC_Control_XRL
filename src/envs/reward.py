"""Reward decomposition for HVAC control decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RewardBreakdown:
    """Reward and raw constraint violations for one timestep."""

    reward: float
    energy_cost_penalty: float
    comfort_penalty: float
    co2_penalty: float
    switching_penalty: float
    temperature_violation_c: float
    co2_violation_ppm: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class HVACReward:
    """Weighted cost function balancing energy, comfort, IAQ, and stability."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.comfort_config = config["comfort"]
        self.iaq_config = config["iaq"]
        self.reward_config = config["reward"]

    def calculate(
        self,
        *,
        indoor_temperature_c: float,
        occupancy: int,
        co2_ppm: float,
        electricity_cost: float,
        previous_action: int,
        action: int,
    ) -> RewardBreakdown:
        """Calculate a transparent, fully decomposed scalar reward."""

        minimum, maximum = self._comfort_bounds(occupancy)
        temperature_violation = max(
            minimum - indoor_temperature_c,
            indoor_temperature_c - maximum,
            0.0,
        )
        co2_limit = float(self.iaq_config["co2_limit_ppm"])
        co2_violation = max(0.0, co2_ppm - co2_limit)
        switch_magnitude = abs(action - previous_action)

        energy_penalty = (
            float(self.reward_config["energy_cost_weight"])
            * max(0.0, electricity_cost)
        )
        comfort_penalty = (
            float(self.reward_config["comfort_weight"])
            * temperature_violation
            / float(self.reward_config["temperature_scale_c"])
        )
        co2_penalty = (
            float(self.reward_config["co2_weight"])
            * co2_violation
            / float(self.reward_config["co2_scale_ppm"])
        )
        switching_penalty = (
            float(self.reward_config["switching_weight"]) * switch_magnitude
        )
        reward = -(
            energy_penalty + comfort_penalty + co2_penalty + switching_penalty
        )

        return RewardBreakdown(
            reward=float(reward),
            energy_cost_penalty=float(energy_penalty),
            comfort_penalty=float(comfort_penalty),
            co2_penalty=float(co2_penalty),
            switching_penalty=float(switching_penalty),
            temperature_violation_c=float(temperature_violation),
            co2_violation_ppm=float(co2_violation),
        )

    def _comfort_bounds(self, occupancy: int) -> tuple[float, float]:
        occupied = occupancy >= int(self.comfort_config["occupancy_threshold"])
        prefix = "occupied" if occupied else "unoccupied"
        return (
            float(self.comfort_config[f"{prefix}_temperature_min_c"]),
            float(self.comfort_config[f"{prefix}_temperature_max_c"]),
        )
