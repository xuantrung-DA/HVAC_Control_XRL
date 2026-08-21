"""Reproducibility helpers shared by environments and agents."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def seed_everything(
    seed: int,
    *,
    env: Any | None = None,
    deterministic_torch: bool = True,
) -> int:
    """Seed Python, NumPy, PyTorch, and optional Gymnasium spaces."""

    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

    if env is not None:
        for space_name in ("action_space", "observation_space"):
            space = getattr(env, space_name, None)
            if space is not None and hasattr(space, "seed"):
                space.seed(seed)

    return seed
