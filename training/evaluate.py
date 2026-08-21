"""Evaluate one controller/checkpoint over configured scenarios and seeds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.agents import RL_AGENT_NAMES, create_agent
from src.baselines import BASELINE_NAMES, create_baseline
from src.envs.hvac_env import HVACEnv
from src.evaluation.comparison import aggregate_results, write_csv, write_json
from src.evaluation.performance import evaluate_controller
from src.utils.config import PROJECT_ROOT


ALL_SCENARIOS = (
    "normal",
    "hot_day",
    "high_occupancy",
    "expensive_electricity",
    "combined_stress",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
        required=True,
        choices=["random", *BASELINE_NAMES, *RL_AGENT_NAMES],
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--scenario", choices=["all", *ALL_SCENARIOS], default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[701, 702, 703])
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics" / "evaluation",
    )
    args = parser.parse_args()

    env = HVACEnv()
    if args.controller == "random":
        controller = None
    elif args.controller in BASELINE_NAMES:
        controller = create_baseline(args.controller, config=env.config)
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for RL controllers")
        controller = create_agent(args.controller, env)
        controller.load(args.checkpoint)

    scenarios = ALL_SCENARIOS if args.scenario == "all" else (args.scenario,)
    results = evaluate_controller(
        controller,
        controller_name=args.controller,
        scenarios=scenarios,
        seeds=args.seeds,
    )
    split_map = {scenario: "evaluation" for scenario in scenarios}
    scenario_summary, split_summary = aggregate_results(results, split_map)
    records = [result.as_dict() for result in results]
    write_json(
        args.output_prefix.with_suffix(".json"),
        {
            "episodes": records,
            "scenario_summary": scenario_summary,
            "summary": split_summary,
        },
    )
    write_csv(args.output_prefix.with_suffix(".csv"), records)
    env.close()
    print(split_summary)


if __name__ == "__main__":
    main()
