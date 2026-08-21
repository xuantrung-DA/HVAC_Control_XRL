"""Robust single-agent training with curriculum and validation checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.agents import create_agent
from src.agents.base_agent import BaseAgent, TrainingSummary
from src.envs.hvac_env import HVACEnv
from src.envs.scenario_sampler import ScenarioSamplerEnv
from src.evaluation.comparison import write_csv, write_json
from src.evaluation.performance import evaluate_controller
from src.utils.config import PROJECT_ROOT, deep_merge, load_agent_config, load_config


@dataclass(frozen=True)
class TrainingRun:
    """Evidence and artifact locations for one algorithm/training seed."""

    agent: str
    training_seed: int
    checkpoint: str
    checkpoint_sha256: str
    best_validation_reward: float
    selected_at_budget: int
    total_budget_used: int
    stopped_early: bool
    checkpoint_reproducible: bool
    curve_json: str
    curve_csv: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def train_with_validation(
    agent_name: str,
    training_seed: int,
    experiment_config: Mapping[str, Any] | None = None,
) -> TrainingRun:
    """Train one seed, select on validation only, and verify its checkpoint."""

    experiment = (
        dict(experiment_config)
        if experiment_config is not None
        else load_config("evaluation")
    )
    train_scenarios = list(experiment["experiment"]["train_scenarios"])
    validation_scenarios = list(
        experiment["experiment"]["validation_scenarios"]
    )
    validation_seeds = [
        int(seed) for seed in experiment["experiment"]["validation_seeds"]
    ]
    curriculum = experiment["curriculum"]
    training_env = ScenarioSamplerEnv(
        train_scenarios,
        strategy=str(curriculum["strategy"]),
        normal_only_episodes=int(curriculum["normal_only_episodes"]),
        expansion_episodes=int(curriculum["expansion_episodes"]),
        seed=training_seed,
    )
    agent_config = _agent_config(agent_name, training_seed)
    agent = create_agent(agent_name, training_env, config=agent_config)
    budget = experiment["budgets"][agent_name]
    total_budget, chunk_budget = _budgets(agent_name, budget)
    early_stopping = experiment["early_stopping"]
    output_config = experiment["outputs"]
    checkpoint = _checkpoint_path(
        agent_name,
        training_seed,
        directory=PROJECT_ROOT / str(output_config["checkpoint_directory"]),
    )

    best_validation_reward = -np.inf
    best_budget = 0
    consumed_budget = 0
    evaluations_without_improvement = 0
    curves: list[dict[str, Any]] = []
    stopped_early = False

    while consumed_budget < total_budget:
        requested = min(chunk_budget, total_budget - consumed_budget)
        summary = _train_chunk(
            agent_name,
            agent,
            training_env,
            requested,
            training_seed,
        )
        consumed_budget += requested
        validation_results = evaluate_controller(
            agent,
            controller_name=agent_name,
            scenarios=validation_scenarios,
            seeds=validation_seeds,
            training_seed=training_seed,
        )
        validation_reward = float(
            np.mean([result.reward for result in validation_results])
        )
        improvement = validation_reward - best_validation_reward
        checkpoint_selected = validation_reward > best_validation_reward
        if checkpoint_selected:
            best_validation_reward = validation_reward
            best_budget = consumed_budget
            agent.save(checkpoint)
        if improvement >= float(early_stopping["minimum_improvement"]):
            evaluations_without_improvement = 0
        else:
            evaluations_without_improvement += 1

        curves.append(
            {
                "agent": agent_name,
                "training_seed": training_seed,
                "budget": consumed_budget,
                "training_mean_reward": summary.mean_episode_reward,
                "training_final_reward": summary.final_episode_reward,
                "training_mean_loss": summary.mean_loss,
                "chunk_steps": summary.total_steps,
                "chunk_episodes": summary.episodes,
                "chunk_duration_seconds": summary.duration_seconds,
                "steps_per_second": summary.steps_per_second,
                "validation_reward_mean": validation_reward,
                "validation_reward_std": float(
                    np.std([result.reward for result in validation_results])
                ),
                "best_validation_reward": best_validation_reward,
                "selected_checkpoint": checkpoint_selected,
            }
        )
        print(
            f"[{agent_name} seed={training_seed}] budget={consumed_budget}/{total_budget} "
            f"train_reward={summary.mean_episode_reward:.2f} "
            f"validation_reward={validation_reward:.2f} "
            f"best={best_validation_reward:.2f}",
            flush=True,
        )

        if (
            bool(early_stopping["enabled"])
            and len(curves) >= int(early_stopping["minimum_evaluations"])
            and evaluations_without_improvement >= int(early_stopping["patience"])
        ):
            stopped_early = True
            break

    curve_directory = PROJECT_ROOT / str(output_config["curves_directory"])
    curve_json = curve_directory / f"{agent_name}_seed_{training_seed}.json"
    curve_csv = curve_directory / f"{agent_name}_seed_{training_seed}.csv"
    write_json(curve_json, curves)
    write_csv(curve_csv, curves)

    reproducible = _checkpoint_reproducible(
        agent_name,
        checkpoint,
        agent_config,
        validation_scenarios[0],
        validation_seeds[0],
    )
    training_env.close()
    return TrainingRun(
        agent=agent_name,
        training_seed=training_seed,
        checkpoint=_portable_path(checkpoint),
        checkpoint_sha256=_sha256(checkpoint),
        best_validation_reward=best_validation_reward,
        selected_at_budget=best_budget,
        total_budget_used=consumed_budget,
        stopped_early=stopped_early,
        checkpoint_reproducible=reproducible,
        curve_json=_portable_path(curve_json),
        curve_csv=_portable_path(curve_csv),
    )


def load_best_agent(run: TrainingRun) -> BaseAgent:
    """Reconstruct and load a selected checkpoint for final evaluation."""

    env = HVACEnv(scenario="normal")
    config = _agent_config(run.agent, run.training_seed)
    agent = create_agent(run.agent, env, config=config)
    agent.load(PROJECT_ROOT / run.checkpoint)
    return agent


def _agent_config(agent_name: str, seed: int) -> dict[str, Any]:
    config_name = "q_learning" if agent_name == "q_learning" else (
        "ppo" if agent_name == "ppo" else "dqn"
    )
    return deep_merge(
        load_agent_config(config_name),
        {"agent": {"seed": seed}},
    )


def _budgets(agent_name: str, budget: Mapping[str, Any]) -> tuple[int, int]:
    if agent_name == "q_learning":
        return int(budget["total_episodes"]), int(budget["chunk_episodes"])
    if agent_name == "ppo":
        return int(budget["total_timesteps"]), int(budget["chunk_timesteps"])
    return int(budget["total_steps"]), int(budget["chunk_steps"])


def _train_chunk(
    agent_name: str,
    agent: BaseAgent,
    env: ScenarioSamplerEnv,
    budget: int,
    seed: int,
) -> TrainingSummary:
    if agent_name == "q_learning":
        return agent.learn(env, episodes=budget, seed=seed)  # type: ignore[attr-defined]
    if agent_name in {"dqn", "double_dqn"}:
        return agent.learn(env, total_steps=budget, seed=seed)  # type: ignore[attr-defined]
    return agent.learn(total_timesteps=budget)  # type: ignore[attr-defined]


def _checkpoint_path(
    agent_name: str, seed: int, *, directory: Path
) -> Path:
    suffix = ".npz" if agent_name == "q_learning" else (
        ".zip" if agent_name == "ppo" else ".pt"
    )
    return directory / agent_name / f"step5_seed_{seed}_best{suffix}"


def _checkpoint_reproducible(
    agent_name: str,
    checkpoint: Path,
    config: Mapping[str, Any],
    scenario: str,
    seed: int,
) -> bool:
    first_env = HVACEnv()
    second_env = HVACEnv()
    first = create_agent(agent_name, first_env, config=config)
    second = create_agent(agent_name, second_env, config=config)
    first.load(checkpoint)
    second.load(checkpoint)
    first_result = evaluate_controller(
        first,
        controller_name=agent_name,
        scenarios=[scenario],
        seeds=[seed],
    )[0]
    second_result = evaluate_controller(
        second,
        controller_name=agent_name,
        scenarios=[scenario],
        seeds=[seed],
    )[0]
    first_env.close()
    second_env.close()
    return first_result == second_result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent",
        required=True,
        choices=["q_learning", "dqn", "double_dqn", "ppo"],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run = train_with_validation(args.agent, args.seed)
    print(run.as_dict())


if __name__ == "__main__":
    main()
