"""Generate immutable V1 evidence and versioned V2 protocol manifests."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.utils.v2_manifest import (
    build_v1_baseline_manifest,
    build_v2_protocol_manifest,
    write_json,
)


def main() -> None:
    output = REPOSITORY_ROOT / "outputs/v2/protocol"
    v1_path = output / "v1_baseline_manifest.json"
    v2_path = output / "v2_protocol_manifest.json"
    write_json(v1_path, build_v1_baseline_manifest(REPOSITORY_ROOT))
    write_json(v2_path, build_v2_protocol_manifest(REPOSITORY_ROOT))
    print(f"Wrote {v1_path.relative_to(REPOSITORY_ROOT)}")
    print(f"Wrote {v2_path.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
