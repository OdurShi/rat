"""Results reported in the paper, stored as CSV files in ``results/paper``.

:func:`load` reads one table; :func:`check_consistency` verifies every value that the paper
states more than once (text, tables and figure annotations) against the stored tables and
returns the list of disagreements.  The numbers are the paper's; this module does not
produce them.
"""

from __future__ import annotations

import csv
from pathlib import Path

TABLES: dict[str, str] = {
    "T1": "table1_paired_accuracy_and_cost.csv",
    "T4": "table4_core_controls.csv",
    "T5": "table5_cost_decomposition.csv",
    "T6": "table6_token_savings.csv",
    "S1": "tableS1_small_and_large_edits.csv",
    "S3": "tableS3_paired_diagnostics_and_general_capabilities.csv",
    "S4": "tableS4_transfer.csv",
    "S5": "tableS5_threshold_trials.csv",
}

BACKBONES = ("Qwen2.5-VL-3B", "Qwen2.5-VL-7B", "LLaVA-1.5-7B")
BUDGETS = {"Qwen2.5-VL-3B": 106.0, "Qwen2.5-VL-7B": 176.0, "LLaVA-1.5-7B": 128.0}
#: Visual tokens of one full-pool pass (millions): Qwen at L and H, LLaVA in either mode.
FULL_POOL_TOKENS = {"qwen_L": 18.51, "qwen_H": 68.44, "llava": 57.60}


def default_directory() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "paper"


def load(table: str, directory: str | Path | None = None) -> list[dict[str, str]]:
    """Rows of a reported table as dictionaries of strings (empty string: not reported)."""
    path = Path(directory) if directory else default_directory()
    with open(path / TABLES[table], newline="") as stream:
        return list(csv.DictReader(stream))


def value(rows: list[dict[str, str]], column: str, **where: str) -> float:
    """The number in ``column`` of the unique row that matches ``where``."""
    matches = [row for row in rows if all(row[key] == wanted for key, wanted in where.items())]
    if len(matches) != 1:
        raise KeyError(f"{len(matches)} rows match {where}")
    return float(matches[0][column])


def best_baseline(
    rows: list[dict[str, str]], backbone: str, column: str = "pair_mean"
) -> tuple[str, float]:
    """Strongest budget-constrained method other than RAT on a backbone."""
    candidates = [
        (float(row[column]), row["method"])
        for row in rows
        if row["backbone"] == backbone
        and row.get("group", "budget") == "budget"
        and row["method"] != "rat"
    ]
    score, method = max(candidates)
    return method, score


def check_consistency(directory: str | Path | None = None) -> list[str]:
    """Compare repeated statements of the paper with the tables; returns the mismatches."""
    t1, t4, t5, t6 = (load(name, directory) for name in ("T1", "T4", "T5", "T6"))
    s1, s3, s4, s5 = (load(name, directory) for name in ("S1", "S3", "S4", "S5"))
    problems: list[str] = []

    def expect(label: str, actual: float, stated: float, tolerance: float = 0.0051) -> None:
        if abs(actual - stated) > tolerance:
            problems.append(f"{label}: tables give {actual:.3f}, the paper states {stated}")

    # Abstract and Section 5.2: gains over the strongest budget-constrained baseline.
    for backbone, gain in zip(BACKBONES, (3.11, 2.72, 2.70), strict=True):
        rat = value(t1, "pair_mean", backbone=backbone, method="rat")
        expect(f"Pair gain on {backbone}", rat - best_baseline(t1, backbone)[1], gain)
        if value(t1, "gpu_hours_mean", backbone=backbone, method="rat") > BUDGETS[backbone]:
            problems.append(f"RAT exceeds the budget on {backbone}")
    # Section 5.2: token reduction relative to full-pool H training.
    for backbone, stated in (("Qwen2.5-VL-3B", 53.3), ("Qwen2.5-VL-7B", 49.3)):
        rat = value(t1, "vtok_millions_mean", backbone=backbone, method="rat")
        expect(
            f"token reduction on {backbone}",
            100 * (1 - rat / FULL_POOL_TOKENS["qwen_H"]),
            stated,
            0.051,
        )
    # Section 5.3: ChartQA human questions.
    for backbone, gain in zip(BACKBONES, (1.17, 0.80, 1.74), strict=True):
        rat = value(s4, "chartqa_human_mean", backbone=backbone, method="rat")
        others = [
            float(row["chartqa_human_mean"])
            for row in s4
            if row["backbone"] == backbone
            and row["method"] not in ("rat", "unadapted", "full_pool_L", "full_pool_H")
        ]
        expect(f"ChartQA human gain on {backbone}", rat - max(others), gain)
    # RAT has the best mean on all nine transfer metrics of every backbone.
    metrics = [name for name in s4[0] if name.endswith("_mean")]
    for backbone in BACKBONES:
        for metric in metrics:
            best = max(float(row[metric]) for row in s4 if row["backbone"] == backbone)
            if value(s4, metric, backbone=backbone, method="rat") < best:
                problems.append(f"RAT is not best on {metric} for {backbone}")
    # Appendix F: held-out paired accuracy and its lead.
    for backbone, heldout, lead in zip(
        BACKBONES, (79.35, 83.68, 60.68), (2.89, 1.90, 3.28), strict=True
    ):
        rat = value(s3, "heldout_pair_mean", backbone=backbone, method="rat")
        expect(f"held-out Pair on {backbone}", rat, heldout)
        others = [
            float(row["heldout_pair_mean"])
            for row in s3
            if row["backbone"] == backbone and row["method"] != "rat"
        ]
        expect(f"held-out lead on {backbone}", rat - max(others), lead)
    # Table 4 repeats Table 1, Table S3 and Table 6 for RAT on Qwen2.5-VL-3B.
    qwen = "Qwen2.5-VL-3B"
    for column, source, name in (
        ("pair_mean", t1, "pair_mean"), ("gpu_hours_mean", t1, "gpu_hours_mean"),
        ("vtok_millions_mean", t1, "vtok_millions_mean"), ("collapse_mean", s3, "collapse_mean"),
        ("heldout_mean", s3, "heldout_pair_mean"), ("worst_mean", s3, "worst_pair_mean"),
    ):  # fmt: skip
        expect(
            f"Table 4 {column}",
            value(t4, column, key="rat"),
            value(source, name, backbone=qwen, method="rat"),
        )
    expect(
        "Table 4 pairs",
        value(t4, "pairs_k_mean", key="rat"),
        value(t6, "trained_pairs_k_mean", backbone=qwen),
    )
    # Section 5.4 / Figure 4: matched comparisons.
    rat = value(t4, "pair_mean", key="rat")
    filter_only = value(t4, "pair_mean", key="filter_only")
    expect(
        "filtering vs random selection",
        filter_only - value(t4, "pair_mean", key="equal_random"),
        2.42,
    )
    expect(
        "filtering vs loss selection", filter_only - value(t4, "pair_mean", key="equal_loss"), 3.03
    )
    expect(
        "allocation vs random allocation", rat - value(t4, "pair_mean", key="random_alloc"), 1.73
    )
    expect(
        "allocation, held-out",
        value(t4, "heldout_mean", key="rat") - value(t4, "heldout_mean", key="random_alloc"),
        2.29,
    )
    expect("min vs mean margin", rat - value(t4, "pair_mean", key="mean_margin"), 2.46)
    expect("min vs student margin", rat - value(t4, "pair_mean", key="student_margin"), 4.86)
    # Tables 5 and 6.
    for row in t5:
        parts = sum(float(row[f"{name}_mean"]) for name in ("preparation", "routing", "training"))
        expect(f"cost decomposition of {row['backbone']}", parts, float(row["total_mean"]), 0.11)
        expect(
            f"total cost of {row['backbone']}",
            float(row["total_mean"]),
            value(t1, "gpu_hours_mean", backbone=row["backbone"], method="rat"),
        )
    shares = [
        100
        * (float(row["preparation_mean"]) + float(row["routing_mean"]))
        / float(row["total_mean"])
        for row in t5
    ]
    expect("smallest preparation and routing share", min(shares), 39.2, 0.051)
    expect("largest preparation and routing share", max(shares), 49.3, 0.051)
    for row in t6:
        actual, same = (
            float(row["actual_vtok_millions_mean"]),
            float(row["same_pair_h_millions_mean"]),
        )
        expect(
            f"token saving of {row['backbone']}",
            100 * (1 - actual / same),
            float(row["saved_percent_mean"]),
            0.06,
        )
    llava = next(row for row in t6 if row["backbone"] == "LLaVA-1.5-7B")
    expect(
        "LLaVA tokens from pairs",
        float(llava["trained_pairs_k_mean"]) * 2 * 576 / 1000,
        float(llava["actual_vtok_millions_mean"]),
        0.006,
    )
    filter_hours = value(t4, "gpu_hours_mean", key="filter_only")
    expect(
        "GPU-hour saving",
        100 * (1 - value(t4, "gpu_hours_mean", key="rat") / filter_hours),
        16.1,
        0.06,
    )
    # Table S1: differences of the listed accuracies.
    table = {
        row["statistic"]: (float(row["small_edits_s_lt_2"]), float(row["large_edits_s_gt_7"]))
        for row in s1
    }
    for name, (first, second) in (
        ("H1x_minus_L4x", ("H1x_accuracy", "L4x_accuracy")), ("H1x_minus_U1x", ("H1x_accuracy", "U1x_accuracy")),
        ("H1x_minus_L1x", ("H1x_accuracy", "L1x_accuracy")), ("H4x_minus_L4x", ("H4x_accuracy", "L4x_accuracy")),
    ):  # fmt: skip
        for column in (0, 1):
            # Differences are rounded after averaging over runs, hence the wider tolerance.
            expect(
                f"Table S1 {name}",
                table[first][column] - table[second][column],
                table[name][column],
                0.0151,
            )
    if table["pairs"][0] + table["pairs"][1] + 1770 != 4800:
        problems.append("Table S1 group sizes do not add up to 4,800 pairs")
    # Table S5 and Section 5.5.
    best = max(s5, key=lambda row: float(row["dev_pair_mean"]))
    if best["gamma"] != "log 4":
        problems.append("the selected threshold is not log 4")
    for row, share in zip((s5[0], s5[-1]), (42.0, 39.0), strict=True):
        overall = float(row["retained_percent_mean"]) * float(row["h_percent_mean"]) / 100
        expect(f"share of all candidates retained at H for {row['gamma']}", overall, share, 0.6)
    return problems
