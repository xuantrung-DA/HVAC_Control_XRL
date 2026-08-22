"""Generate README diagrams and charts from repository evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"
ASSETS.mkdir(parents=True, exist_ok=True)

BG = "#030417"
SURFACE = "#11102b"
TEXT = "#f7f4ff"
MUTED = "#aaa5bb"
VIOLET = "#875cff"
MAGENTA = "#e943a4"
CORAL = "#ff596d"
AMBER = "#ffad42"
CYAN = "#50d6e8"
GREEN = "#62e2a2"
RED = "#ff657a"


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def finish(fig: plt.Figure, name: str) -> None:
    fig.savefig(ASSETS / name, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="white", alpha=0.08, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def box(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str,
        subtitle: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=SURFACE, edgecolor=color, linewidth=1.6,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.62, title, color=TEXT, ha="center",
            va="center", fontsize=13, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.31, subtitle, color=MUTED, ha="center",
            va="center", fontsize=8.5, linespacing=1.35)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
          color: str = MUTED) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.35, color=color))


def architecture_overview() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.6), facecolor=BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.04, 0.94, "XRL-HVAC · LEARNING-AUGMENTED CONTROL", color=AMBER,
            fontsize=10, fontweight="bold")
    ax.text(0.04, 0.885, "Forecast-aware decisions with explicit physical guardrails",
            color=TEXT, fontsize=22, fontweight="bold")

    box(ax, .04, .58, .16, .17, "Building", "2R1C thermal model\nmoisture · CO₂ · energy", VIOLET)
    box(ax, .25, .58, .16, .17, "State", "sensors · schedule\nprice · current action", CYAN)
    box(ax, .46, .58, .19, .17, "Forecast + Risk", "1h/4h forecast\ntrend · reliability", MAGENTA)
    box(ax, .70, .58, .13, .17, "DQN", "cooling\nproposal", CORAL)
    box(ax, .86, .58, .10, .17, "Guard", "thermal\nlimits", AMBER)
    for a, b in [((.20, .665), (.25, .665)), ((.41, .665), (.46, .665)),
                 ((.65, .665), (.70, .665)), ((.83, .665), (.86, .665))]:
        arrow(ax, a, b)

    box(ax, .38, .25, .17, .16, "Cooling", "DQN proposal\n+ thermal clamp", CORAL)
    box(ax, .59, .25, .17, .16, "Ventilation", "deterministic\nIAQ control", CYAN)
    box(ax, .80, .25, .17, .16, "Dehumidifier", "deterministic\nRH control", VIOLET)
    arrow(ax, (.91, .58), (.465, .41), AMBER)
    arrow(ax, (.91, .58), (.675, .41), AMBER)
    arrow(ax, (.91, .58), (.885, .41), AMBER)
    arrow(ax, (.38, .33), (.20, .63), CORAL)
    arrow(ax, (.59, .33), (.20, .63), CYAN)
    arrow(ax, (.80, .33), (.20, .63), VIOLET)

    box(ax, .04, .12, .25, .16, "Evidence Surface", "reward audit · energy ledger\nXAI · API · dashboard", GREEN)
    arrow(ax, (.12, .58), (.165, .28), GREEN)
    ax.text(.04, .035, "DQN optimizes sensible cooling; deterministic layers retain IAQ, humidity and safety authority.",
            color=MUTED, fontsize=9)
    finish(fig, "xrl-hvac-overview.png")


def control_problem() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 3.4), facecolor=BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    points = np.array([[.50, .83], [.20, .22], [.80, .22], [.50, .83]])
    ax.plot(points[:, 0], points[:, 1], color=VIOLET, linewidth=2)
    ax.fill(points[:, 0], points[:, 1], color=VIOLET, alpha=.08)
    ax.scatter([.50, .20, .80], [.83, .22, .22], s=170,
               color=[AMBER, MAGENTA, CYAN], edgecolors="white", linewidths=.7)
    ax.text(.50, .94, "ENERGY + COST ↓", color=AMBER, ha="center", fontsize=13, fontweight="bold")
    ax.text(.09, .12, "COMFORT ↑", color=MAGENTA, ha="center", fontsize=13, fontweight="bold")
    ax.text(.91, .12, "INDOOR AIR QUALITY ↑", color=CYAN, ha="center", fontsize=13, fontweight="bold")
    ax.text(.50, .42, "A useful controller must satisfy constraints\nbefore optimizing cumulative reward.",
            color=TEXT, ha="center", va="center", fontsize=15, fontweight="bold", linespacing=1.5)
    finish(fig, "control-objectives.png")


def system_flow() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.4), facecolor=BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(.04, .91, "PROPOSAL ≠ EXECUTION", color=AMBER, fontsize=10, fontweight="bold")
    ax.text(.04, .83, "The policy suggests; the control layer decides what reaches physics.",
            color=TEXT, fontsize=20, fontweight="bold")
    items = [
        (.04, "Current State", "temperature · RH · CO₂", VIOLET),
        (.22, "Forecast + Risk", "future load · uncertainty", MAGENTA),
        (.42, "RL Policy", "DQN", CORAL),
        (.57, "Proposed", "cooling action", CORAL),
        (.72, "Hybrid Guard", "thermal · IAQ · RH", AMBER),
        (.87, "Executed", "3 actuators", GREEN),
    ]
    for x, title, subtitle, color in items:
        width = .12 if x not in (.22, .72) else .14
        box(ax, x, .35, width, .25, title, subtitle, color)
    for i in range(len(items) - 1):
        x, *_ = items[i]; nx, *_ = items[i + 1]
        w = .12 if x not in (.22, .72) else .14
        arrow(ax, (x + w, .475), (nx, .475), AMBER if i >= 3 else MUTED)
    ax.text(.635, .24, "model output", color=CORAL, ha="center", fontsize=9)
    ax.text(.925, .24, "physical command", color=GREEN, ha="center", fontsize=9)
    finish(fig, "proposed-vs-executed.png")


def grouped_metrics(name: str, title: str, labels: list[str], values: list[list[float]],
                    metrics: list[str], source: str, colors: list[str]) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4.1), facecolor=BG)
    fig.suptitle(title, color=TEXT, fontsize=17, fontweight="bold", y=1.02)
    for index, ax in enumerate(np.atleast_1d(axes)):
        style_ax(ax)
        data = [row[index] for row in values]
        bars = ax.bar(labels, data, color=colors, width=.62)
        ax.set_title(metrics[index], color=MUTED, fontsize=10, pad=12)
        ax.tick_params(axis="x", rotation=18)
        top = max(data) if max(data) else 1
        ax.set_ylim(0, top * 1.22)
        for bar, value in zip(bars, data, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * .035,
                    f"{value:.2f}", ha="center", color=TEXT, fontsize=8, fontweight="bold")
    fig.text(.04, -.02, source, color=MUTED, fontsize=7)
    fig.tight_layout()
    finish(fig, name)


def benchmark_charts() -> None:
    v1 = load_json("outputs/metrics/step5/benchmark_report.json")
    wanted = ["rule_based", "dqn", "double_dqn", "ppo"]
    rows = {
        item["controller"]: item for item in v1["scenario_summary"]
        if item["scenario"] == "combined_stress" and item["controller"] in wanted
    }
    grouped_metrics(
        "v1-controller-comparison.png",
        "V1 · Sealed Combined Stress (mean across training/evaluation seeds)",
        ["Rule", "DQN", "Double DQN", "PPO"],
        [[rows[key]["energy_kwh_mean"], rows[key]["comfort_violation_percent_mean"],
          rows[key]["co2_violation_percent_mean"]] for key in wanted],
        ["Energy (kWh)", "Comfort violation (%)", "CO₂ violation (%)"],
        "Source: outputs/metrics/step5/benchmark_report.json",
        [AMBER, MAGENTA, VIOLET, CYAN],
    )

    development = load_json("outputs/v2/hybrid/development_benchmark.json")
    baseline = development["matched_rule_based"]["aggregate"]["metrics"]
    candidate = next(item for item in development["candidates"] if item["training_seed"] == 42)["aggregate"]["metrics"]
    grouped_metrics(
        "v2-development-comparison.png",
        "V2 Hybrid · Development Validation",
        ["Matched Rule", "DQN seed 42"],
        [[baseline["whole_building_kwh"]["mean"], baseline["comfort_violation_percent"]["mean"], baseline["co2_violation_percent"]["mean"]],
         [candidate["whole_building_kwh"]["mean"], candidate["comfort_violation_percent"]["mean"], candidate["co2_violation_percent"]["mean"]]],
        ["Whole-building energy (kWh)", "Comfort violation (%)", "CO₂ violation (%)"],
        "Source: outputs/v2/hybrid/development_benchmark.json",
        [AMBER, MAGENTA],
    )

    final = load_json("outputs/v2/hybrid/combined_stress_one_shot.json")
    baseline_final = final["matched_rule_based"]["aggregate"]["metrics"]
    candidate_final = final["candidate_result"]["aggregate"]["metrics"]
    grouped_metrics(
        "v2-final-heldout.png",
        "V2 Hybrid · One-shot Combined Stress · FINAL FAIL",
        ["Matched Rule", "Frozen DQN"],
        [[baseline_final["whole_building_kwh"]["mean"], baseline_final["comfort_violation_percent"]["mean"], baseline_final["co2_violation_percent"]["mean"]],
         [candidate_final["whole_building_kwh"]["mean"], candidate_final["comfort_violation_percent"]["mean"], candidate_final["co2_violation_percent"]["mean"]]],
        ["Whole-building energy (kWh)", "Comfort violation (%)", "CO₂ violation (%)"],
        "Source: outputs/v2/hybrid/combined_stress_one_shot.json · status COMPLETED_FAIL",
        [AMBER, RED],
    )

    fig, ax = plt.subplots(figsize=(8.7, 5.2), facecolor=BG)
    style_ax(ax)
    for key, label, color in zip(wanted, ["Rule", "DQN", "Double DQN", "PPO"],
                                 [AMBER, MAGENTA, VIOLET, CYAN], strict=True):
        ax.scatter(rows[key]["energy_kwh_mean"], rows[key]["comfort_violation_percent_mean"],
                   s=150, color=color, edgecolor="white", linewidth=.7, label=label)
    ax.scatter(baseline_final["whole_building_kwh"]["mean"], baseline_final["comfort_violation_percent"]["mean"],
               s=170, marker="X", color=AMBER, label="V2 matched rule")
    ax.scatter(candidate_final["whole_building_kwh"]["mean"], candidate_final["comfort_violation_percent"]["mean"],
               s=170, marker="X", color=RED, label="V2 frozen DQN · FAIL")
    ax.set_xlabel("Energy (kWh) → lower is better", color=MUTED)
    ax.set_ylabel("Comfort violation (%) → lower is better", color=MUTED)
    ax.set_title("Energy–comfort trade-off across simulator generations", color=TEXT,
                 fontsize=16, fontweight="bold", pad=16)
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=8, ncol=2)
    fig.text(.01, .01, "V1 and V2 use different simulator scales; compare within each generation, not absolute kWh across versions.",
             color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, .04, 1, 1))
    finish(fig, "comfort-energy-tradeoff.png")


def main() -> None:
    architecture_overview()
    control_problem()
    system_flow()
    benchmark_charts()
    print(f"Generated README assets in {ASSETS}")


if __name__ == "__main__":
    main()
