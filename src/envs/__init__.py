"""Smart-building simulation environments."""

from gymnasium.envs.registration import register, registry


ENVIRONMENT_ID = "XRL-HVAC-v0"

if ENVIRONMENT_ID not in registry:
    register(
        id=ENVIRONMENT_ID,
        entry_point="src.envs.hvac_env:HVACEnv",
    )

__all__ = ["ENVIRONMENT_ID"]
