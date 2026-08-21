"""SAC V2 interface and checkpoint smoke tests."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from src.agents import SACV2Agent
from src.envs.v2 import V2ContinuousHVACEnv
from src.utils.config import PROJECT_ROOT, load_yaml


def test_sac_predicts_bounded_independent_controls(tmp_path) -> None:
    env = V2ContinuousHVACEnv()
    observation, _ = env.reset(seed=42)
    config = deepcopy(load_yaml(PROJECT_ROOT / "configs/v2/sac.yaml"))
    config["agent"]["device"] = "cpu"
    agent = SACV2Agent(env, config)
    action = agent.predict(observation, deterministic=True)
    assert action.shape == (2,)
    assert np.all((0.0 <= action) & (action <= 1.0))
    checkpoint = tmp_path / "sac_test.zip"
    agent.save(checkpoint)
    loaded = SACV2Agent(env, config)
    loaded.load(checkpoint)
    assert np.allclose(action, loaded.predict(observation, deterministic=True))
    assert agent.parameter_count() > 0
