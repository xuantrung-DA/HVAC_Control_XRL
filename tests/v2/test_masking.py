"""Ablation masking preserves spaces and determinism."""

from __future__ import annotations

import numpy as np

from src.envs.v2 import MaskedObservationEnv, V2HVACEnv


def test_masking_replaces_only_selected_features_with_space_midpoint() -> None:
    base = V2HVACEnv(shield_enabled=False)
    masked = MaskedObservationEnv(base, [9, 10, 11])
    observation, _ = masked.reset(seed=42)
    original, _ = V2HVACEnv(shield_enabled=False).reset(seed=42)
    midpoint = (base.observation_space.low + base.observation_space.high) / 2.0
    assert np.array_equal(observation[[9, 10, 11]], midpoint[[9, 10, 11]])
    assert np.array_equal(observation[:9], original[:9])
    assert masked.observation_space.contains(observation)
