"""Retrain equal-budget V2 DQN ablations on development data only."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.dqn import DQNAgent  # noqa: E402
from src.envs.v2 import (  # noqa: E402
    MaskedController,
    MaskedObservationEnv,
    V2HVACEnv,
    V2ScenarioSamplerEnv,
)
from src.evaluation import aggregate_v2_results, evaluate_v2_controller  # noqa: E402
from src.utils.config import deep_merge, load_agent_config, load_yaml  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dqn_config(seed: int) -> dict:
    return deep_merge(
        deepcopy(load_agent_config("dqn")),
        {
            "agent": {"seed": seed, "device": "auto"},
            "replay_buffer": {"capacity": 20000, "batch_size": 64, "warmup_steps": 1000},
            "exploration": {"epsilon_decay_steps": 9000},
            "optimization": {"reward_scale": 0.10},
        },
    )


def main() -> None:
    seed = 2026
    budget = 10000
    training = load_yaml(PROJECT_ROOT / "configs/v2/training.yaml")["experiment"]
    probe = V2HVACEnv(shield_enabled=False)
    names = probe.observation_names
    groups = {
        "full_dynamic": (),
        "fixed_reward": (),
        "no_forecast": tuple(
            index
            for index, name in enumerate(names)
            if name.startswith("forecast_") or name.startswith("uncertainty_")
        ),
        "no_trend": tuple(
            index
            for index, name in enumerate(names)
            if name.endswith("_slope") or name == "occupancy_delta"
        ),
        "no_risk": tuple(range(25, len(names))),
    }
    results = []
    for variant, indices in groups.items():
        reward_mode = "fixed" if variant == "fixed_reward" else "dynamic"
        base_env = V2ScenarioSamplerEnv(
            training["training_scenarios"],
            seed=seed,
            normal_only_episodes=10,
            shield_enabled=False,
            reward_mode=reward_mode,
        )
        env = MaskedObservationEnv(base_env, indices) if indices else base_env
        config = dqn_config(seed)
        agent = DQNAgent(env.observation_space, env.action_space, config=config)
        started = time.perf_counter()
        summary = agent.learn(env, total_steps=budget, seed=seed)
        checkpoint = PROJECT_ROOT / "models/v2/ablations" / f"{variant}_seed_{seed}.pt"
        agent.save(checkpoint)
        evaluation_controller = (
            MaskedController(agent, indices, probe.observation_space)
            if indices
            else agent
        )
        episodes = evaluate_v2_controller(
            evaluation_controller,
            controller_name=variant,
            scenarios=training["validation_scenarios"],
            seeds=[901],
            shield_enabled=False,
            reward_mode=reward_mode,
        )
        aggregate = aggregate_v2_results(episodes)
        metrics = aggregate["metrics"]
        result = {
            "variant": variant,
            "masked_features": [names[index] for index in indices],
            "reward_mode": reward_mode,
            "seed": seed,
            "training_steps": budget,
            "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": sha256(checkpoint),
            "training_steps_per_second": summary.steps_per_second,
            "duration_seconds": time.perf_counter() - started,
            "validation": aggregate,
            "constraint_pass": (
                metrics["comfort_violation_percent"]["mean"] < 5.0
                and metrics["co2_violation_percent"]["mean"] < 1.0
            ),
        }
        results.append(result)
        print(json.dumps({
            "variant": variant,
            "energy": metrics["whole_building_kwh"]["mean"],
            "comfort": metrics["comfort_violation_percent"]["mean"],
            "co2": metrics["co2_violation_percent"]["mean"],
            "constraint_pass": result["constraint_pass"],
        }), flush=True)
        env.close()
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "held_out_used": False,
        "design": "equal 10k transition budget, seed 2026, validation seed 901",
        "results": results,
        "all_constraints_passed": any(item["constraint_pass"] for item in results),
        "limitations": [
            "Single-seed budgeted ablations identify directional effects, not final uncertainty.",
            "Perfect/noisy forecast resilience remains sealed with the held-out protocol because no candidate passed development gates.",
        ],
    }
    output = PROJECT_ROOT / "outputs/v2/ablations/development_ablation_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
