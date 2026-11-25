"""Summarise the runs of ``scripts/run_study.py``.

Usage::

    python scripts/report_study.py --runs runs/laptop --config configs/laptop/smolvlm_256m.yaml \
        --output results/laptop_study/smolvlm-256m [--figures assets/results]

Reads ``<runs>/seed*/<name>.json`` and writes

* ``runs/`` - a copy of the run files (metrics, costs, assignments, predictions);
* ``main.csv`` - one row per method with mean and standard deviation over seeds, in the
  layout of Tables 1 and 4;
* ``preliminary.csv`` - the conditions of Section 3 by projected edit scale (layout of
  Table S1);
* ``comparisons.csv`` - paired bootstrap intervals of paired-accuracy differences;
* ``report.md`` - the tables as Markdown.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from rat.analysis.contrast import group_masks
from rat.config import load_config
from rat.data.pairs import edit_scale
from rat.data.splits import load_pool
from rat.evaluation.paired import mean_sd, pair_correct
from rat.evaluation.stats import mcnemar_exact, paired_bootstrap
from rat.interface.geometry import tile_grid

MAIN = [
    ("unadapted", "Unadapted"),
    ("full_L", "Full pool (L)"),
    ("full_H", "Full pool (H)"),
    ("random_L", "Random (L)"),
    ("random_H", "Random (H)"),
    ("rat", "RAT"),
    ("filter_only", "Filtering-only: S_R, all H"),
    ("alloc_only", "Allocation-only: full pool"),
    ("random_alloc", "Random allocation: S_R"),
    ("mean_margin", "Mean branch margin"),
    ("student_margin", "Initial-student candidate margin"),
    ("equal_random", "Equal-count random, all H"),
    ("equal_loss", "Equal-count answer loss, all H"),
]
PRELIM = [
    ("prelim_unadapted", "Unadapted"),
    ("prelim_L1", "L1x"),
    ("prelim_L4", "L4x"),
    ("prelim_U1", "U1x"),
    ("prelim_H1", "H1x"),
    ("prelim_H4", "H4x"),
]
COMPARISONS = [
    ("rat", "random_alloc"), ("rat", "filter_only"), ("filter_only", "equal_random"), ("rat", "random_H"),
    ("random_H", "random_L"), ("prelim_H1", "prelim_L4"), ("prelim_H1", "prelim_U1"), ("prelim_H1", "prelim_L1"),
]  # fmt: skip


def fmt(values: list[float], digits: int = 2) -> str:
    mean, sd = mean_sd(values)
    return f"{mean:.{digits}f} ± {sd:.{digits}f}" if len(values) > 1 else f"{mean:.{digits}f}"


def load_runs(directory: Path) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = {}
    for path in sorted(directory.glob("seed*/*.json")):
        data = json.loads(path.read_text())
        runs.setdefault(path.stem, {})[int(path.parent.name[4:])] = data
    return runs


def correctness(run: dict, answers: list) -> np.ndarray:
    predictions = [tuple(p) for p in run["extra"]["predictions"]]
    return pair_correct(predictions, answers)


def markdown(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    runs = load_runs(args.runs)
    args.output.mkdir(parents=True, exist_ok=True)
    copies = args.output / "runs"
    if copies.exists():
        shutil.rmtree(copies)
    for path in sorted(args.runs.glob("seed*/*.json")):
        (copies / path.parent.name).mkdir(parents=True, exist_ok=True)
        shutil.copy(path, copies / path.parent.name / path.name)
    if (args.runs / "cost_model.json").exists():
        shutil.copy(args.runs / "cost_model.json", args.output / "cost_model.json")
    report = [f"# Study report: {config['backbone']['model_id']}", ""]
    data = config["data"]

    # ---- main comparison and controls -------------------------------------------------
    header = [
        "Method",
        "Seeds",
        "Pairs",
        "H (%)",
        "Pair",
        "Single",
        "Collapse",
        "Familiar",
        "Held-out",
        "Worst",
        "VTok (k)",
        "Prep. (s)",
        "Routing (s)",
        "Training (s)",
        "Total (s)",
    ]
    rows, table = [], []
    for name, label in MAIN:
        if name not in runs:
            continue
        seeds = sorted(runs[name])
        entries = [runs[name][s] for s in seeds]

        def column(getter, entries=entries) -> list[float]:
            return [float(getter(e)) for e in entries]

        metrics = {
            key: column(lambda e, key=key: e["metrics"][key])
            for key in ("pair", "single", "collapse", "familiar", "heldout", "worst")
        }
        pairs = column(lambda e: e["train"].get("pairs", 0))
        share = column(lambda e: 100 * e["train"].get("top_mode_share", 0.0))
        tokens = column(lambda e: e["train"].get("visual_tokens", 0) / 1e3)
        costs = {
            key: column(lambda e, key=key: e["costs"].get(key, 0.0))
            for key in ("preparation", "routing", "training", "total")
        }
        rows.append(
            [
                label,
                " ".join(map(str, seeds)),
                fmt(pairs, 0),
                fmt(share, 1),
                *(fmt(metrics[k]) for k in metrics),
                fmt(tokens, 1),
                *(fmt(costs[k], 0) for k in costs),
            ]
        )
        record = {"method": name, "label": label, "seeds": " ".join(map(str, seeds))}
        for key, values in {
            "pairs": pairs,
            "h_percent": share,
            **metrics,
            "vtok_thousands": tokens,
            **{f"{k}_seconds": v for k, v in costs.items()},
        }.items():
            record[f"{key}_mean"], record[f"{key}_sd"] = mean_sd(values)
        table.append(record)
    if table:
        with open(args.output / "main.csv", "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(table)
        budget = next(iter(runs[table[0]["method"]].values()))["costs"].get("budget")
        report += [
            "## Paired test set at H",
            "",
            f"{data['test']} test pairs; pools: reader fitting {data['reader_fit']}, tuning {data['tune']}, development {data['dev']}, student candidates {data['train']}.  Budget B_ref = {budget:.0f} device-seconds.  Accuracies in percent, mean ± standard deviation over the listed seeds.",
            "",
            markdown(header, rows),
            "",
        ]
    if "rat" in runs:
        trial_rows = []
        for seed, run in sorted(runs["rat"].items()):
            for trial in run["extra"].get("trials", []):
                trial_rows.append(
                    [
                        str(seed),
                        f"{trial['threshold']:.4f}",
                        f"{trial['retained_percent']:.2f}",
                        f"{trial['top_mode_percent']:.1f}",
                        f"{trial['dev_pair']:.2f}",
                        f"{trial['cost']:.1f}",
                        "selected" if abs(trial["threshold"] - run["threshold"]) < 1e-9 else "",
                    ]
                )
        report += [
            "## Threshold trials (layout of Table S5)",
            "",
            markdown(
                ["Seed", "gamma", "Retained (%)", "H (%)", "Dev Pair", "Trial cost (s)", ""],
                trial_rows,
            ),
            "",
        ]

    # ---- preliminary study --------------------------------------------------------------
    if any(name in runs for name, _ in PRELIM):
        records = load_pool("prelim_test", data["prelim_test"], data["manifests"])
        answers = [tuple(r["answers"]) for r in records]
        # Edit scale in the pixels of the low mode (512-pixel view of the tiled interface).
        tile = 512
        scales = np.array(
            [
                edit_scale(
                    r["edit_area"],
                    (r["width"], r["height"]),
                    tile_grid((r["width"], r["height"]), tile, tile, 8).resized,
                )
                for r in records
            ]
        )
        thirds = np.quantile(scales, [1 / 3, 2 / 3])
        groups = {
            "all": np.ones(len(records), bool),
            "s < 2": group_masks(scales)["small"],
            "s > 7": group_masks(scales)["large"],
            "lower third": scales <= thirds[0],
            "middle third": (scales > thirds[0]) & (scales <= thirds[1]),
            "upper third": scales > thirds[1],
        }
        present = [(name, label) for name, label in PRELIM if name in runs]
        correct = {
            name: {seed: correctness(run, answers) for seed, run in runs[name].items()}
            for name, _ in present
        }
        header = ["Statistic", *groups]
        rows = [["Pairs", *(str(int(mask.sum())) for mask in groups.values())]]
        prelim_table = [{"statistic": "pairs", **{g: int(m.sum()) for g, m in groups.items()}}]
        for name, label in present:
            values = {
                g: [100 * float(c[mask].mean()) for c in correct[name].values()]
                for g, mask in groups.items()
            }
            rows.append(
                [
                    f"{label} accuracy ({len(correct[name])} runs)",
                    *(fmt(v) for v in values.values()),
                ]
            )
            prelim_table.append(
                {"statistic": f"{label}_accuracy", **{g: mean_sd(v)[0] for g, v in values.items()}}
            )
        for first, second in (
            ("prelim_H1", "prelim_L4"),
            ("prelim_H1", "prelim_U1"),
            ("prelim_H1", "prelim_L1"),
            ("prelim_H4", "prelim_L4"),
        ):
            if first in correct and second in correct:
                seeds = sorted(set(correct[first]) & set(correct[second]))
                values = {
                    g: [
                        100
                        * (
                            float(correct[first][s][mask].mean())
                            - float(correct[second][s][mask].mean())
                        )
                        for s in seeds
                    ]
                    for g, mask in groups.items()
                }
                label = f"{first[7:]}x − {second[7:]}x"
                rows.append([f"{label} (pp)", *(fmt(v) for v in values.values())])
                prelim_table.append(
                    {
                        "statistic": label.replace(" − ", "_minus_"),
                        **{g: mean_sd(v)[0] for g, v in values.items()},
                    }
                )
        with open(args.output / "preliminary.csv", "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(prelim_table[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(prelim_table)
        report += [
            "## Preliminary study (layout of Table S1)",
            "",
            f"{data['prelim_train']} training pairs, {len(records)} test pairs at H; paired accuracy in percent.  Edit scale s in pixels of the low mode; thirds are terciles of s (cut points {thirds[0]:.2f} and {thirds[1]:.2f}).",
            "",
            markdown(header, rows),
            "",
        ]

    # ---- paired comparisons ---------------------------------------------------------------
    test_answers = None
    comparison_rows, comparison_table = [], []
    for first, second in COMPARISONS:
        if first not in runs or second not in runs:
            continue
        seeds = sorted(set(runs[first]) & set(runs[second]))
        if not seeds:
            continue
        pool = "prelim_test" if first.startswith("prelim") else "test"
        if test_answers is None or test_answers[0] != pool:
            test_answers = (
                pool,
                [tuple(r["answers"]) for r in load_pool(pool, data[pool], data["manifests"])],
            )
        a = [correctness(runs[first][s], test_answers[1]) for s in seeds]
        b = [correctness(runs[second][s], test_answers[1]) for s in seeds]
        interval = paired_bootstrap(a, b)
        test = mcnemar_exact(a[0], b[0])
        comparison_rows.append(
            [
                f"{first} − {second}",
                str(len(seeds)),
                f"{interval['difference']:+.2f}",
                f"[{interval['low']:+.2f}, {interval['high']:+.2f}]",
                f"{test['p_value']:.3g}",
            ]
        )
        comparison_table.append(
            {
                "first": first,
                "second": second,
                "seeds": len(seeds),
                **interval,
                "mcnemar_p_first_seed": test["p_value"],
            }
        )
    if comparison_table:
        with open(args.output / "comparisons.csv", "w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(comparison_table[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(comparison_table)
        report += [
            "## Paired differences",
            "",
            "Difference of paired accuracy in percentage points, averaged over the common seeds, with a 95% paired bootstrap interval over test pairs; the McNemar p-value is for the first common seed.",
            "",
            markdown(
                ["Comparison", "Seeds", "Difference", "95% interval", "McNemar p"], comparison_rows
            ),
            "",
        ]

    (args.output / "report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
