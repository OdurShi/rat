"""Render the reported tables (``results/paper``) as Markdown and update the README.

Usage::

    python scripts/render_tables.py --show T1          # print one table
    python scripts/render_tables.py --update README.md  # refresh every <!-- table:ID --> block

Blocks in a Markdown file are delimited by ``<!-- table:ID -->`` and ``<!-- /table:ID -->``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from rat.reported import BACKBONES, load

METHODS = {
    "unadapted": "Unadapted", "full_pool_L": "Full pool (L)", "full_pool_H": "Full pool (H)",
    "random_L": "Random (L)", "random_H": "Random (H)", "clipscore": "CLIPScore", "d2_pruning": "D2 Pruning",
    "coincide": "COINCIDE", "xmas": "XMAS", "dose": "DOSE", "dataprophet": "DataProphet", "rat": "**RAT**",
}  # fmt: skip


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|---|" + "---:|" * (len(header) - 1)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def pm(row: dict[str, str], name: str) -> str:
    mean, sd = row[f"{name}_mean"], row.get(f"{name}_sd", "")
    return f"{mean} ± {sd}" if sd else mean


def best(rows: list[dict[str, str]], column: str, lowest: bool = False) -> str:
    values = [float(row[column]) for row in rows]
    return f"{min(values) if lowest else max(values)}"


def emphasise(
    text: str, row: dict[str, str], column: str, rows: list[dict[str, str]], lowest: bool = False
) -> str:
    values = [float(r[column]) for r in rows]
    target = min(values) if lowest else max(values)
    return f"**{text}**" if float(row[column]) == target else text


def t1() -> str:
    rows = load("T1")
    header = ["Method"] + [f"{b}: {c}" for b in BACKBONES for c in ("Pair", "GPU-h", "VTok (M)")]
    body = []
    for method, label in METHODS.items():
        line = [label]
        for backbone in BACKBONES:
            subset = [r for r in rows if r["backbone"] == backbone]
            row = next(r for r in subset if r["method"] == method)
            line += [
                emphasise(pm(row, "pair"), row, "pair_mean", subset),
                pm(row, "gpu_hours"),
                pm(row, "vtok_millions"),
            ]
        body.append(line)
    return table(header, body)


TRANSFER = [
    ("chartqa_human", "ChartQA human"), ("chartqa_aug", "ChartQA aug."), ("chartqapro", "ChartQAPro"),
    ("textvqa", "TextVQA"), ("docvqa_anls", "DocVQA"), ("infovqa_anls", "InfoVQA"), ("ocrbench_v1", "OCRBench v1"),
    ("ocrbench_v2_en", "OCRBench v2 EN"), ("ocrbench_v2_zh", "OCRBench v2 ZH"),
]  # fmt: skip


def transfer(backbone: str) -> str:
    rows = [r for r in load("S4") if r["backbone"] == backbone]
    body = []
    for method, label in METHODS.items():
        row = next(r for r in rows if r["method"] == method)
        body.append(
            [label]
            + [emphasise(row[f"{key}_mean"], row, f"{key}_mean", rows) for key, _ in TRANSFER]
        )
    return table(["Method"] + [name for _, name in TRANSFER], body)


def s3(backbone: str) -> str:
    rows = [r for r in load("S3") if r["backbone"] == backbone]
    columns = [
        ("single", "Single", False), ("collapse", "Collapse ↓", True), ("familiar_pair", "Familiar pair", False),
        ("heldout_pair", "Held-out pair", False), ("worst_pair", "Worst pair", False), ("mathvista", "MathVista", False),
        ("mmbench", "MMBench", False), ("pope_random", "POPE random", False), ("pope_popular", "POPE popular", False),
        ("pope_adversarial", "POPE adv.", False),
    ]  # fmt: skip
    body = []
    for method, label in METHODS.items():
        row = next(r for r in rows if r["method"] == method)
        body.append(
            [label]
            + [emphasise(pm(row, key), row, f"{key}_mean", rows, low) for key, _, low in columns]
        )
    return table(["Method"] + [name for _, name, _ in columns], body)


def t4() -> str:
    rows = load("T4")
    header = [
        "Control",
        "Pairs (k)",
        "H (%)",
        "Pair",
        "Collapse ↓",
        "Held-out",
        "Worst",
        "GPU-h",
        "VTok (M)",
    ]
    body = []
    for row in rows:
        label = f"**{row['control']}**" if row["key"] == "rat" else row["control"]
        body.append([
            label, pm(row, "pairs_k"), pm(row, "h_percent"), emphasise(pm(row, "pair"), row, "pair_mean", rows),
            emphasise(pm(row, "collapse"), row, "collapse_mean", rows, True), emphasise(pm(row, "heldout"), row, "heldout_mean", rows),
            emphasise(pm(row, "worst"), row, "worst_mean", rows), pm(row, "gpu_hours"), pm(row, "vtok_millions"),
        ])  # fmt: skip
    return table(header, body)


def t5() -> str:
    rows = load("T5")
    body = [
        [r["backbone"], pm(r, "preparation"), pm(r, "routing"), pm(r, "training"), pm(r, "total")]
        for r in rows
    ]
    return table(["Backbone", "Preparation", "Routing", "Training", "Total"], body)


def t6() -> str:
    rows = load("T6")
    body = [
        [
            r["backbone"],
            pm(r, "trained_pairs_k"),
            pm(r, "h_percent"),
            pm(r, "actual_vtok_millions"),
            pm(r, "same_pair_h_millions"),
            pm(r, "saved_percent"),
        ]
        for r in rows
    ]
    return table(
        [
            "Backbone",
            "Trained pairs (k)",
            "H (%)",
            "Actual VTok (M)",
            "Same-pair H (M)",
            "Saved (%)",
        ],
        body,
    )


def s1() -> str:
    names = {
        "pairs": "Pairs", "median_DH_over_DL": "Median D_H / D_L", "L1x_accuracy": "L1× accuracy", "L4x_accuracy": "L4× accuracy",
        "U1x_accuracy": "U1× accuracy", "H1x_accuracy": "H1× accuracy", "H4x_accuracy": "H4× accuracy", "H1x_minus_L4x": "H1× − L4×",
        "H1x_minus_U1x": "H1× − U1×", "H1x_minus_L1x": "H1× − L1×", "H4x_minus_L4x": "H4× − L4×",
    }  # fmt: skip
    body = [
        [names[r["statistic"]], r["small_edits_s_lt_2"], r["large_edits_s_gt_7"]]
        for r in load("S1")
    ]
    return table(["Statistic", "Small edits (s < 2)", "Large edits (s > 7)"], body)


def s5() -> str:
    rows = load("S5")
    body = [
        [
            f"{r['gamma']}",
            pm(r, "retained_percent"),
            pm(r, "h_percent"),
            emphasise(pm(r, "dev_pair"), r, "dev_pair_mean", rows),
            pm(r, "trial_gpu_hours"),
        ]
        for r in rows
    ]
    return table(["γ", "Retained (%)", "H (%)", "Dev Pair", "Trial GPU-h"], body)


TABLES = {
    "T1": t1,
    "T2a": lambda: transfer("Qwen2.5-VL-3B"),
    "T2b": lambda: transfer("Qwen2.5-VL-7B"),
    "T3": lambda: transfer("LLaVA-1.5-7B"),
    "T4": t4,
    "T5": t5,
    "T6": t6,
    "S1": s1,
    "S3a": lambda: s3("Qwen2.5-VL-3B"),
    "S3b": lambda: s3("Qwen2.5-VL-7B"),
    "S3c": lambda: s3("LLaVA-1.5-7B"),
    "S5": s5,
}


def update(path: Path) -> int:
    text = path.read_text()
    count = 0
    for name, render in TABLES.items():
        pattern = re.compile(
            rf"(<!-- table:{name} -->\n)(?:.*?\n)?(<!-- /table:{name} -->)", re.DOTALL
        )
        text, replaced = pattern.subn(
            lambda m, render=render: m.group(1) + render() + "\n" + m.group(2), text
        )
        count += replaced
    path.write_text(text)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--show", choices=sorted(TABLES))
    parser.add_argument("--update", type=Path)
    args = parser.parse_args()
    if args.show:
        print(TABLES[args.show]())
    if args.update:
        print(f"updated {update(args.update)} tables in {args.update}")


if __name__ == "__main__":
    main()
