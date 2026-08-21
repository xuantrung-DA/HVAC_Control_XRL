"""Configuration loading utilities for XRL-HVAC."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRECTORY = PROJECT_ROOT / "configs"


class ConfigError(ValueError):
    """Raised when a project configuration file is missing or invalid."""


def deep_merge(
    base: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a recursive merge without mutating either input mapping."""

    merged: dict[str, Any] = deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and raise a descriptive error on invalid input."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Unable to read configuration {config_path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ConfigError(f"Configuration root must be a mapping: {config_path}")
    return loaded


def load_config(
    name: str,
    *,
    config_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a named project config and optionally apply recursive overrides."""

    path = Path(config_path) if config_path else CONFIG_DIRECTORY / f"{name}.yaml"
    config = load_yaml(path)
    return deep_merge(config, overrides) if overrides else config


def load_environment_config(
    config_path: str | Path | None = None,
    *,
    scenario: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load environment settings and apply a configured scenario profile."""

    config = load_config("environment", config_path=config_path)
    if scenario is not None:
        scenarios = config.get("scenarios", {})
        if scenario not in scenarios:
            available = ", ".join(sorted(scenarios))
            raise ConfigError(
                f"Unknown scenario '{scenario}'. Available scenarios: {available}"
            )
        config = deep_merge(config, scenarios[scenario])
    if overrides:
        config = deep_merge(config, overrides)
    return config


def load_agent_config(
    agent: str,
    config_path: str | Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load an agent configuration by name."""

    return load_config(agent, config_path=config_path, overrides=overrides)
