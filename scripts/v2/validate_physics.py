"""Generate the pre-training V2 simulator physics validation artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.envs.v2.validation import build_validation_report
from src.utils.v2_manifest import write_json


def main() -> None:
    report = build_validation_report(project_root=REPOSITORY_ROOT)
    output = REPOSITORY_ROOT / "outputs/v2/validation"
    json_path = output / "simulator_validation_report.json"
    csv_path = output / "physics_sanity.csv"
    write_json(json_path, report)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["name", "passed", "requirement", "measurements_json"],
        )
        writer.writeheader()
        for case in report["cases"]:
            writer.writerow(
                {
                    "name": case["name"],
                    "passed": case["passed"],
                    "requirement": case["requirement"],
                    "measurements_json": json.dumps(case["measurements"]),
                }
            )
    print(json.dumps({
        "all_checks_passed": report["all_checks_passed"],
        "checks": f"{report['checks_passed']}/{report['checks_total']}",
        "steps_per_second": report["performance"]["steps_per_second"],
        "rss_delta_mb": report["performance"]["rss_delta_mb"],
        "training_authorized": report["training_authorized"],
    }, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
