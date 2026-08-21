"""Convenience launcher for curriculum DQN or Double DQN training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.train import train_with_validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--double", action="store_true")
    args = parser.parse_args()
    name = "double_dqn" if args.double else "dqn"
    print(train_with_validation(name, args.seed).as_dict())


if __name__ == "__main__":
    main()
