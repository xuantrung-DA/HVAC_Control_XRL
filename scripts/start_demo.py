"""Start the local XRL-HVAC FastAPI demo backend."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=os.getenv("XRL_HVAC_API_HOST", "127.0.0.1"),
        port=int(os.getenv("XRL_HVAC_API_PORT", "8000")),
        reload=False,
    )
