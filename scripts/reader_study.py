"""Frozen-side study: visual contrast, reader fitting and routing statistics.

Usage::

    python scripts/reader_study.py --config configs/laptop/qwen2.5-vl-3b_reader.yaml --output results/reader_study/qwen2.5-vl-3b

The script needs no student training.  It

1. encodes the reader-fitting pool and a held-out pool in both input modes;
2. measures the paired-token contrast ``D(L)``, ``D(H)`` and its ratio by projected edit
   scale (Section 3, Fig. 2a, Appendix A.2);
3. fits the shared reader (Section 4.2) and reports its held-out margins;
4. applies the routing rule of Eq. (4) for every threshold of the grid (Fig. 5a, 5b);
5. with ``--student`` scores the unadapted student on the held-out pool at ``H``.

Outputs: ``summary.json`` and ``pairs.csv`` (one row per held-out pair).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from rat.analysis.contrast import group_masks, median_ratio, pool_contrast, projected_edit_scales
from rat.config import build_backbone, load_config
from rat.data.splits import load_pool
from rat.evaluation.paired import paired_metrics
from rat.features import open_pool
from rat.pipeline import pair_scores, reader_margin_table
from rat.reader import CandidateReader, ReaderFitConfig, fit_reader
from rat.routing import route
from rat.student import predict


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout", default="tune", help="pool used for held-out statistics")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--student", action="store_true", help="also score the unadapted student at H"
    )
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    args.output.mkdir(parents=True, exist_ok=True)
    backbone = build_backbone(config)
    data = config["data"]

    def progress(label: str, done: int, total: int) -> None:
        if done % 40 == 0 or done == total:
            log(f"encode {label}: {done}/{total} pairs")

    fit = open_pool(
        backbone,
        load_pool("reader_fit", data["reader_fit"], data["manifests"]),
        config["cache_dir"],
        "reader_fit",
        progress=progress,
    )
    held = open_pool(
        backbone,
        load_pool(args.heldout, data[args.heldout], data["manifests"]),
        config["cache_dir"],
        args.heldout,
        progress=progress,
    )
    top = len(backbone.modes) - 1
    summary: dict = {
        "backbone": backbone.name,
        "modes": [m.name for m in backbone.modes],
        "reader_fit_pairs": len(fit),
        "heldout_pool": args.heldout,
        "heldout_pairs": len(held),
        "seed": args.seed,
    }

    # Visual tokens per image and encoding time, from the stores.
    for name, pool in (("reader_fit", fit), (args.heldout, held)):
        summary[f"{name}_tokens_per_image"] = {
            m.name: float(np.mean([pool.costs(i)[k] / 2 for i in range(len(pool))]))
            for k, m in enumerate(backbone.modes)
        }
        summary[f"{name}_encode_seconds_per_image"] = {
            m.name: float(pool.stores[k].meta["seconds"] / (2 * len(pool)))
            for k, m in enumerate(backbone.modes)
        }

    # Contrast and edit scale (Appendix A.2).
    contrast = pool_contrast(held)
    scales = projected_edit_scales(backbone, held.records)
    groups = group_masks(scales)
    summary["contrast"] = {
        name: {"pairs": int(mask.sum()), "median_ratio_H_over_L": median_ratio(contrast, mask)}
        for name, mask in groups.items()
    }
    summary["contrast"]["all"] = {
        "pairs": len(held),
        "median_ratio_H_over_L": median_ratio(contrast, np.ones(len(held), bool)),
    }
    thirds = np.quantile(scales, [1 / 3, 2 / 3])
    for label, mask in (("lower_third", scales <= thirds[0]), ("upper_third", scales > thirds[1])):
        summary["contrast"][label] = {
            "pairs": int(mask.sum()),
            "median_ratio_H_over_L": median_ratio(contrast, mask),
        }
    summary["edit_scale_quantiles"] = {
        str(q): float(np.quantile(scales, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
    }
    log(f"contrast ratios {summary['contrast']}")

    # Reader (Section 4.2).
    for pool in (fit, held):
        pool.embed_candidates()
    reader_config = ReaderFitConfig(**config["reader"])
    reader = CandidateReader(
        backbone.visual_dim, backbone.text_dim, reader_config.init_gain, seed=args.seed
    ).to(backbone.device)
    tic = time.perf_counter()
    fit_log = fit_reader(
        reader, fit.fetch, fit.candidates, len(fit.modes), reader_config, args.seed
    )
    summary["reader"] = {
        "config": config["reader"],
        "fit_seconds": time.perf_counter() - tic,
        "steps": fit_log.steps,
        "first_loss": fit_log.losses[0],
        "final_loss": float(np.mean(fit_log.losses[-5:])),
    }
    tables = {"fit": reader_margin_table(reader, fit), "heldout": reader_margin_table(reader, held)}
    for name, table in tables.items():
        summary["reader"][name] = {
            m.name: {
                "branch_accuracy": float((table[:, k] > 0).float().mean()),
                "pair_accuracy": float((table[:, k].min(-1).values > 0).float().mean()),
                "median_min_margin": float(table[:, k].min(-1).values.median()),
            }
            for k, m in enumerate(backbone.modes)
        }
    log(f"reader {summary['reader']}")

    # Routing statistics for the threshold grid (Eq. (4), Fig. 5a/5b).
    scores = pair_scores(tables["heldout"], "min")
    decisions = {}
    summary["routing"] = []
    for threshold in config["routing"]["thresholds"]:
        chosen = [route(scores[i].tolist(), held.costs(i), threshold) for i in range(len(held))]
        decisions[threshold] = chosen
        retained = [c for c in chosen if c is not None]
        tokens = sum(held.costs(i)[c] for i, c in enumerate(chosen) if c is not None)
        same_pair = sum(held.costs(i)[top] for i, c in enumerate(chosen) if c is not None)
        row = {
            "threshold": threshold,
            "exp_threshold": math.exp(threshold),
            "retained_percent": 100.0 * len(retained) / len(held),
            "top_mode_percent": 100.0 * sum(c == top for c in retained) / max(1, len(retained)),
            "token_saving_percent": 100.0 * (1 - tokens / same_pair) if same_pair else 0.0,
        }
        for label, mask in groups.items():
            members = [c for c, inside in zip(chosen, mask) if inside]
            row[f"{label}_retained_percent"] = (
                100.0 * sum(c is not None for c in members) / max(1, len(members))
            )
            row[f"{label}_low_mode_percent"] = (
                100.0 * sum(c == 0 for c in members) / max(1, len(members))
            )
        summary["routing"].append(row)
    log(f"routing {summary['routing']}")

    predictions = None
    if args.student:
        tic = time.perf_counter()
        predictions, _seconds = predict(
            backbone,
            held,
            top,
            None,
            config["evaluation"]["batch_examples"],
            config["evaluation"]["max_new_tokens"],
        )
        summary["unadapted"] = paired_metrics(predictions, held.answers, held.records)
        summary["unadapted"]["seconds"] = time.perf_counter() - tic
        log(f"unadapted {summary['unadapted']}")

    with open(args.output / "pairs.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        header = [
            "pair_id",
            "family",
            "task",
            "width",
            "height",
            "font_px",
            "edit_area",
            "edit_scale",
        ]
        for mode in backbone.modes:
            header += [
                f"tokens_{mode.name}",
                f"contrast_{mode.name}",
                f"margin0_{mode.name}",
                f"margin1_{mode.name}",
            ]
        header += [f"route_gamma{k}" for k in range(len(decisions))]
        if predictions is not None:
            header += ["prediction0", "prediction1", "answer0", "answer1"]
        writer.writerow(header)
        for i, record in enumerate(held.records):
            row = [
                record["pair_id"],
                record["family"],
                record["task"],
                record["width"],
                record["height"],
                record["font_px"],
                record["edit_area"],
                f"{scales[i]:.4f}",
            ]
            for k, _mode in enumerate(backbone.modes):
                row += [
                    held.costs(i)[k] // 2,
                    f"{contrast[i, k]:.6f}",
                    f"{float(tables['heldout'][i, k, 0]):.4f}",
                    f"{float(tables['heldout'][i, k, 1]):.4f}",
                ]
            for threshold in decisions:
                choice = decisions[threshold][i]
                row.append("skip" if choice is None else backbone.modes[choice].name)
            if predictions is not None:
                row += [*predictions[i], *held.answers[i]]
            writer.writerow(row)
    torch.save(
        {"reader": reader.state_dict(), "fit": tables["fit"], "heldout": tables["heldout"]},
        args.output / "reader.pt",
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote {args.output}/summary.json and pairs.csv")


if __name__ == "__main__":
    main()
