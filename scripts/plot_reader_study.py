"""Draw the figure of a frozen-side study (``scripts/reader_study.py``).

Usage::

    python scripts/plot_reader_study.py results/reader_study/qwen2.5-vl-3b \
        [--panels contrast] [--output assets/results/reader_study_qwen2.5-vl-3b.png]

Left: paired-token contrast of every held-out pair at L and H (layout of Fig. 2a).
Middle: median contrast ratio by projected edit scale.  Right: retention and share of
retained pairs assigned to H for the thresholds of the grid (layout of Fig. 5a, 5b).
``--panels contrast`` draws the first two panels only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
FAMILY_COLOURS = {"chart": "#2a78d6", "table": "#eb6834", "card": "#1baf7a"}
FAMILY_NAMES = {"chart": "Chart", "table": "Table", "card": "Text card"}


def style(axis) -> None:
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(AXIS)
    axis.tick_params(colors=MUTED, labelsize=8, length=0)
    axis.grid(color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)


def draw_routing(axis, summary: dict, high: str) -> None:
    style(axis)
    axis.grid(axis="x", visible=False)
    labels = [f"log {round(r['exp_threshold'])}" for r in summary["routing"]]
    retained = [r["retained_percent"] for r in summary["routing"]]
    share = [r["top_mode_percent"] for r in summary["routing"]]
    positions = np.arange(len(labels))
    for values, colour, name in (
        (retained, "#2a78d6", "Retained pairs"),
        (share, "#eb6834", f"Share at {high} among retained"),
    ):
        axis.plot(
            positions,
            values,
            color=colour,
            linewidth=2,
            marker="o",
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=name,
            zorder=3,
        )
        for position, value in zip(positions, values, strict=True):
            axis.annotate(
                f"{value:.1f}",
                (position, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=INK,
            )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 112)
    axis.set_xlim(-0.35, len(labels) - 0.65)
    axis.set_xlabel("Threshold γ", fontsize=9, color=SECONDARY)
    axis.set_ylabel("Percent", fontsize=9, color=SECONDARY)
    axis.set_title(
        "Routing on held-out pairs", fontsize=10.5, color=INK, loc="left", fontweight="bold"
    )
    axis.legend(
        frameon=False, fontsize=8.5, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2,
        columnspacing=1.2, handletextpad=0.4,
    )  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--panels", choices=("all", "contrast"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.directory / "summary.json").read_text())
    with open(args.directory / "pairs.csv", newline="") as stream:
        rows = list(csv.DictReader(stream))
    low, high = summary["modes"][0], summary["modes"][-1]
    routing = args.panels == "all"
    figure, axes = plt.subplots(
        1,
        3 if routing else 2,
        figsize=(12.5 if routing else 9.0, 4.0),
        facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.15, 1, 1] if routing else [1.15, 1]},
    )

    # (a) contrast scatter
    axis = axes[0]
    style(axis)
    for family, colour in FAMILY_COLOURS.items():
        x = [
            math.log10(max(float(r[f"contrast_{low}"]), 1e-6))
            for r in rows
            if r["family"] == family
        ]
        y = [
            math.log10(max(float(r[f"contrast_{high}"]), 1e-6))
            for r in rows
            if r["family"] == family
        ]
        axis.scatter(
            x,
            y,
            s=16,
            color=colour,
            edgecolors=SURFACE,
            linewidths=0.5,
            label=FAMILY_NAMES[family],
            zorder=3,
        )
    lo = min(axis.get_xlim()[0], axis.get_ylim()[0])
    hi = max(axis.get_xlim()[1], axis.get_ylim()[1])
    axis.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    axis.set_xlim(lo, hi)
    axis.set_ylim(lo, hi)
    axis.set_xlabel(
        f"Contrast of paired vision tokens, log10 D({low})", fontsize=9, color=SECONDARY
    )
    axis.set_ylabel(f"log10 D({high})", fontsize=9, color=SECONDARY)
    axis.set_title(
        "Visual representation contrast", fontsize=10.5, color=INK, loc="left", fontweight="bold"
    )
    axis.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right", handletextpad=0.2)
    axis.text(
        0.03,
        0.95,
        f"above the line: {high} retains more contrast",
        transform=axis.transAxes,
        fontsize=8,
        color=SECONDARY,
        va="top",
    )

    # (b) median ratio by edit scale
    axis = axes[1]
    style(axis)
    axis.grid(axis="x", visible=False)
    scales = np.array([float(r["edit_scale"]) for r in rows])
    ratios = np.array(
        [float(r[f"contrast_{high}"]) / max(float(r[f"contrast_{low}"]), 1e-12) for r in rows]
    )
    edges = np.quantile(scales, np.linspace(0, 1, 6))
    centres, medians = [], []
    for k in range(5):
        member = (scales >= edges[k]) & (
            scales <= edges[k + 1] if k == 4 else scales < edges[k + 1]
        )
        centres.append(math.sqrt(edges[k] * edges[k + 1]))
        medians.append(float(np.median(ratios[member])))
    positions = np.arange(5)
    axis.bar(positions, medians, width=0.62, color=FAMILY_COLOURS["chart"], zorder=3)
    for position, value in zip(positions, medians, strict=True):
        axis.text(
            position,
            value + 0.02 * max(medians),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=INK,
        )
    axis.axhline(1.0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    axis.set_ylim(0, 1.16 * max(max(medians), 1.0))
    axis.set_xticks(positions)
    axis.set_xticklabels([f"{c:.1f}" for c in centres])
    axis.set_xlabel(
        f"Edit scale at {low} (px; centres of quintile bins)", fontsize=9, color=SECONDARY
    )
    axis.set_ylabel(f"Median D({high}) / D({low})", fontsize=9, color=SECONDARY)
    axis.set_title(
        "Contrast ratio by edit scale", fontsize=10.5, color=INK, loc="left", fontweight="bold"
    )

    if routing:
        draw_routing(axes[2], summary, high)

    name = summary["backbone"].split("/")[-1]
    title = f"{name}: {summary['heldout_pairs']} held-out pairs"
    if routing:
        title += f", reader fitted on {summary['reader_fit_pairs']} pairs"
    figure.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    stem = "reader_study" if routing else "contrast"
    output = args.output or Path("assets/results") / f"{stem}_{args.directory.name}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, facecolor=SURFACE)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
