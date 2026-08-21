"""Smoke-train every RL agent and emit Step 4 learning evidence.

This is deliberately smaller than portfolio-quality training. It verifies that
the implementations update, outperform a seeded random policy, survive a
checkpoint round-trip, and run every configured scenario.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.agents import create_agent
from src.agents.base_agent import BaseAgent, TrainingSummary
from src.envs.hvac_env import HVACEnv
from src.utils.config import PROJECT_ROOT, deep_merge, load_agent_config


SCENARIOS = (
    "normal",
    "hot_day",
    "high_occupancy",
    "expensive_electricity",
    "combined_stress",
)
EVALUATION_SEEDS = (701, 702, 703, 704, 705)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-episodes", type=int, default=None)
    parser.add_argument("--dqn-steps", type=int, default=None)
    parser.add_argument("--ppo-steps", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics" / "step4_smoke_report.json",
    )
    return parser.parse_args()


def evaluate_agent(
    agent: BaseAgent,
    *,
    scenarios: tuple[str, ...] = SCENARIOS,
    seeds: tuple[int, ...] = EVALUATION_SEEDS,
) -> dict[str, Any]:
    scenario_results: dict[str, Any] = {}
    all_rewards: list[float] = []
    all_actions: Counter[int] = Counter()

    for scenario in scenarios:
        rewards: list[float] = []
        energies: list[float] = []
        costs: list[float] = []
        comfort_steps: list[int] = []
        co2_steps: list[int] = []
        switches: list[int] = []
        for seed in seeds:
            env = HVACEnv(scenario=scenario)
            observation, _ = env.reset(seed=seed)
            agent.reset()
            done = False
            while not done:
                action = agent.predict(observation, deterministic=True)
                all_actions[action] += 1
                observation, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            metrics = info["episode_metrics"]
            rewards.append(float(metrics["reward"]))
            energies.append(float(metrics["energy_kwh"]))
            costs.append(float(metrics["electricity_cost"]))
            comfort_steps.append(int(metrics["comfort_violation_steps"]))
            co2_steps.append(int(metrics["co2_violation_steps"]))
            switches.append(int(metrics["switch_count"]))
            env.close()

        all_rewards.extend(rewards)
        scenario_results[scenario] = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_energy_kwh": float(np.mean(energies)),
            "mean_cost": float(np.mean(costs)),
            "mean_comfort_violation_steps": float(np.mean(comfort_steps)),
            "mean_co2_violation_steps": float(np.mean(co2_steps)),
            "mean_switch_count": float(np.mean(switches)),
            "episodes_completed": len(rewards),
        }

    total_actions = sum(all_actions.values())
    distribution = {
        str(action): all_actions[action] / total_actions
        for action in range(4)
    }
    return {
        "mean_reward_all_scenarios": float(np.mean(all_rewards)),
        "scenario_results": scenario_results,
        "action_distribution": distribution,
        "unique_actions": sum(count > 0 for count in all_actions.values()),
        "episodes_completed": len(all_rewards),
    }


def evaluate_random() -> dict[str, float]:
    rewards: list[float] = []
    for seed in EVALUATION_SEEDS:
        env = HVACEnv(scenario="normal")
        observation, _ = env.reset(seed=seed)
        del observation
        rng = np.random.default_rng(seed + 10_000)
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(int(rng.integers(4)))
            done = terminated or truncated
        rewards.append(float(info["episode_metrics"]["reward"]))
        env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "episodes": len(rewards),
    }


def build_agent(name: str) -> tuple[BaseAgent, dict[str, Any], HVACEnv]:
    env = HVACEnv(scenario="normal")
    if name == "q_learning":
        config = load_agent_config("q_learning")
    elif name in {"dqn", "double_dqn"}:
        base = load_agent_config("dqn")
        smoke = base["smoke_test"]
        config = deep_merge(
            base,
            {
                "replay_buffer": {
                    "capacity": int(smoke["replay_buffer_capacity"]),
                    "warmup_steps": int(smoke["warmup_steps"]),
                }
            },
        )
    else:
        base = load_agent_config("ppo")
        smoke = base["smoke_test"]
        config = deep_merge(
            base,
            {
                "training": {
                    "n_steps": int(smoke["n_steps"]),
                    "batch_size": int(smoke["batch_size"]),
                }
            },
        )
    return create_agent(name, env, config=config), config, env


def train_agent(
    name: str,
    agent: BaseAgent,
    env: HVACEnv,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> TrainingSummary:
    if name == "q_learning":
        episodes = args.q_episodes or int(config["smoke_test"]["episodes"])
        return agent.learn(env, episodes=episodes, seed=42)  # type: ignore[attr-defined]
    if name in {"dqn", "double_dqn"}:
        steps = args.dqn_steps or int(config["smoke_test"]["total_steps"])
        return agent.learn(env, total_steps=steps, seed=42)  # type: ignore[attr-defined]
    steps = args.ppo_steps or int(config["smoke_test"]["total_timesteps"])
    return agent.learn(total_timesteps=steps)  # type: ignore[attr-defined]


def checkpoint_path(name: str) -> Path:
    suffix = ".npz" if name == "q_learning" else (".zip" if name == "ppo" else ".pt")
    return PROJECT_ROOT / "models" / name / f"smoke{name}{suffix}"


def resource_snapshot() -> dict[str, float]:
    memory = psutil.Process().memory_info()
    return {
        "rss_mb": memory.rss / (1024**2),
        "peak_rss_mb": getattr(memory, "peak_wset", memory.rss) / (1024**2),
    }


def run() -> int:
    args = parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    random_result = evaluate_random()
    report: dict[str, Any] = {
        "purpose": "Step 4 smoke-learning evidence; not final training",
        "random_baseline_normal": random_result,
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "agents": {},
    }

    for name in ("q_learning", "dqn", "double_dqn", "ppo"):
        print(f"\n[{name}] building agent and starting smoke training", flush=True)
        agent, config, training_env = build_agent(name)
        untrained_evaluation = evaluate_agent(
            agent,
            scenarios=("normal",),
            seeds=EVALUATION_SEEDS,
        )
        untrained_reward = untrained_evaluation["scenario_results"]["normal"][
            "mean_reward"
        ]
        before_memory = resource_snapshot()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        summary = train_agent(name, agent, training_env, config, args)
        wall_time = time.perf_counter() - started
        after_memory = resource_snapshot()
        agent_device = str(agent.metadata().get("device", "cpu"))
        gpu_peak_mb: float | None = (
            torch.cuda.max_memory_allocated() / (1024**2)
            if torch.cuda.is_available() and agent_device.startswith("cuda")
            else None
        )

        evaluation = evaluate_agent(agent)
        normal_reward = evaluation["scenario_results"]["normal"]["mean_reward"]
        required_improvement = max(5.0, abs(random_result["mean_reward"]) * 0.05)
        improvement = normal_reward - random_result["mean_reward"]
        improvement_over_untrained = normal_reward - untrained_reward
        required_untrained_improvement = max(5.0, abs(untrained_reward) * 0.05)

        checkpoint = checkpoint_path(name)
        agent.save(checkpoint)
        loaded, _, loaded_env = build_agent(name)
        loaded.load(checkpoint)
        loaded_evaluation = evaluate_agent(
            loaded,
            scenarios=("normal",),
            seeds=EVALUATION_SEEDS,
        )
        loaded_reward = loaded_evaluation["scenario_results"]["normal"][
            "mean_reward"
        ]
        probe_env = HVACEnv()
        probe, _ = probe_env.reset(seed=909)
        repeated_actions = [loaded.predict(probe, deterministic=True) for _ in range(5)]

        finite_training = all(
            math.isfinite(float(value))
            for value in (
                summary.mean_episode_reward,
                summary.final_episode_reward,
                summary.duration_seconds,
            )
        )
        checkpoint_match = math.isclose(
            normal_reward, loaded_reward, rel_tol=0.0, abs_tol=1e-6
        )
        learning_gate = improvement >= required_improvement
        untrained_gate = improvement_over_untrained >= required_untrained_improvement
        scenario_gate = evaluation["episodes_completed"] == (
            len(SCENARIOS) * len(EVALUATION_SEEDS)
        )
        deterministic_gate = len(set(repeated_actions)) == 1

        report["agents"][name] = {
            "training": summary.as_dict(),
            "wall_time_seconds": wall_time,
            "parameters": agent.parameter_count(),
            "device": agent_device,
            "resources": {
                "rss_before_mb": before_memory["rss_mb"],
                "rss_after_mb": after_memory["rss_mb"],
                "rss_delta_mb": after_memory["rss_mb"] - before_memory["rss_mb"],
                "process_peak_rss_mb": after_memory["peak_rss_mb"],
                "cuda_peak_allocated_mb": gpu_peak_mb,
            },
            "evaluation": evaluation,
            "untrained_normal_reward": untrained_reward,
            "random_improvement_normal": improvement,
            "untrained_improvement_normal": improvement_over_untrained,
            "required_improvement": required_improvement,
            "required_untrained_improvement": required_untrained_improvement,
            "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
            "checkpoint_reward_match": checkpoint_match,
            "deterministic_repeated_action": repeated_actions[0],
            "gates": {
                "finite_training": finite_training,
                "beats_random": learning_gate,
                "beats_untrained_policy": untrained_gate,
                "checkpoint_round_trip": checkpoint_match,
                "deterministic_inference": deterministic_gate,
                "all_scenarios_complete": scenario_gate,
            },
        }
        print(
            f"[{name}] reward={normal_reward:.2f}, random={random_result['mean_reward']:.2f}, "
            f"untrained={untrained_reward:.2f}, improvement={improvement:.2f}, "
            f"steps/s={summary.steps_per_second:.1f}, "
            f"actions={evaluation['action_distribution']}",
            flush=True,
        )
        training_env.close()
        loaded_env.close()
        probe_env.close()
        del agent, loaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_gates_pass = all(
        all(agent_report["gates"].values())
        for agent_report in report["agents"].values()
    )
    report["all_gates_pass"] = all_gates_pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {args.output}")
    print(f"All Step 4 smoke gates pass: {all_gates_pass}")
    return 0 if all_gates_pass else 1


if __name__ == "__main__":
    sys.exit(run())
