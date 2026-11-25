"""Draw the result figures of the README from the CSV files in ``results/``.

Usage::

    python scripts/plot_results.py [--output assets/results]

``main_results.png`` and ``accuracy_vs_tokens.png`` show the numbers reported in the paper
(``results/paper``).  Figures of the runs made with this repository are drawn by
``scripts/report_study.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rat.reported import BACKBONES, BUDGETS, load

SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
BLUE, ORANGE = "#2a78d6", "#eb6834"

LABELS = {
    "unadapted": "Unadapted", "full_pool_L": "Full pool (L)", "full_pool_H": "Full pool (H)",
    "random_L": "Random (L)", "random_H": "Random (H)", "clipscore": "CLIPScore", "d2_pruning": "D2 Pruning",
    "coincide": "COINCIDE", "xmas": "XMAS", "dose": "DOSE", "dataprophet": "DataProphet", "rat": "RAT",
}  # fmt: skip


def style(axis) -> None:
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(AXIS)
    axis.tick_params(colors=MUTED, labelsize=8, length=0)
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)


def main_results(output: Path) -> None:
    rows = load("T1")
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 4.3), sharey=True, facecolor=SURFACE)
    order = list(LABELS)[::-1]
    for axis, backbone in zip(axes, BACKBONES, strict=True):
        style(axis)
        table = {row["method"]: row for row in rows if row["backbone"] == backbone}
        for position, method in enumerate(order):
            row = table[method]
            mean, sd = float(row["pair_mean"]), float(row["pair_sd"] or 0)
            reference = row["group"] == "reference"
            color = BLUE if method == "rat" else (MUTED if reference else SECONDARY)
            axis.errorbar(mean, position, xerr=sd, color=color, linewidth=1.2, capsize=0, zorder=2)
            axis.scatter(
                mean, position, s=46 if method == "rat" else 30, zorder=3, linewidths=1.4,
                facecolors=SURFACE if reference else color, edgecolors=color,
            )  # fmt: skip
            if method == "rat":
                axis.annotate(
                    f"{mean:.2f}",
                    (mean, position),
                    xytext=(-9, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=9,
                    color=INK,
                    fontweight="bold",
                )
        axis.set_yticks(range(len(order)))
        axis.set_yticklabels([LABELS[m] for m in order], color=INK, fontsize=9)
        if axis is axes[0]:
            axis.get_yticklabels()[order.index("rat")].set_fontweight("bold")
        axis.set_title(backbone, fontsize=10.5, color=INK, loc="left", fontweight="bold")
        axis.set_xlabel("Paired accuracy (%)", fontsize=9, color=SECONDARY)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=7, label="RAT"),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=SECONDARY,
            markersize=6,
            label="Within the budget",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor=SURFACE,
            markeredgecolor=MUTED,
            markeredgewidth=1.4,
            markersize=6,
            label="Reference configuration",
        ),
    ]
    figure.legend(
        handles=handles, loc="lower center", ncols=3, frameon=False, fontsize=9, labelcolor=INK
    )
    figure.suptitle(
        "Paired accuracy on 4,800 held-out pairs (Table 1; mean and standard deviation over three seeds)",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    figure.savefig(output / "main_results.png", dpi=200, facecolor=SURFACE)
    plt.close(figure)


def accuracy_vs_tokens(output: Path) -> None:
    rows = load("T1")
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), facecolor=SURFACE)
    # Direct labels for RAT, the uniform baselines and the strongest selector per backbone.
    named = {
        "Qwen2.5-VL-3B": {
            "rat": (9, 0, "left"),
            "random_H": (0, -13, "center"),
            "random_L": (9, 0, "left"),
            "xmas": (0, 12, "center"),
        },
        "Qwen2.5-VL-7B": {
            "rat": (9, 0, "left"),
            "random_H": (0, -13, "center"),
            "random_L": (9, 0, "left"),
            "xmas": (0, 12, "center"),
        },
        "LLaVA-1.5-7B": {
            "rat": (9, 0, "left"),
            "random_H": (-9, 0, "right"),
            "random_L": (-9, 0, "right"),
            "dataprophet": (0, 12, "center"),
        },
    }
    for axis, backbone in zip(axes, BACKBONES, strict=True):
        style(axis)
        axis.grid(axis="y", color=GRID, linewidth=0.6)
        for row in rows:
            if row["backbone"] != backbone or row["group"] != "budget":
                continue
            method = row["method"]
            x, y = float(row["vtok_millions_mean"]), float(row["pair_mean"])
            color = BLUE if method == "rat" else SECONDARY
            axis.scatter(
                x,
                y,
                s=62 if method == "rat" else 34,
                color=color,
                zorder=3,
                edgecolors=SURFACE,
                linewidths=1.2,
            )
            if method in named[backbone]:
                dx, dy, align = named[backbone][method]
                axis.annotate(
                    LABELS[method],
                    (x, y),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha=align,
                    va="center",
                    fontsize=8.5,
                    color=INK,
                    fontweight="bold" if method == "rat" else "normal",
                )
        axis.set_title(
            f"{backbone} (budget {BUDGETS[backbone]:.0f} GPU-h)",
            fontsize=10.5,
            color=INK,
            loc="left",
            fontweight="bold",
        )
        axis.set_xlabel("Visual tokens in student training (millions)", fontsize=9, color=SECONDARY)
        axis.margins(x=0.16, y=0.12)
    axes[0].set_ylabel("Paired accuracy (%)", fontsize=9, color=SECONDARY)
    figure.suptitle(
        "Accuracy against training visual tokens, methods within the budget (Table 1)",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output / "accuracy_vs_tokens.png", dpi=200, facecolor=SURFACE)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=Path("assets/results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    main_results(args.output)
    accuracy_vs_tokens(args.output)
    print(f"wrote {args.output}/main_results.png and accuracy_vs_tokens.png")


if __name__ == "__main__":
    main()
