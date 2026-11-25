"""Run the main comparison, the core controls and the preliminary study.

Usage::

    python scripts/run_study.py --config configs/laptop/smolvlm_256m.yaml --output runs/laptop \
        --stages main controls --seeds 17 29 43

Stages
    ``lr-pilot``      learning-rate check with uniformly selected high-resolution pairs
    ``main``          RAT, uniform selection at L and H within the budget, unadapted model
    ``controls``      filtering-only, random allocation, equal-count random selection
    ``references``    full-pool passes at L and H
    ``variants``      mean branch margin, initial-student margin, equal-count answer loss,
                      allocation-only routing of the full pool
    ``preliminary``   L1x, L4x, U1x, H1x (and H4x with ``--prelim-h4``) of Section 3

``--save-adapters`` also stores the adapted LoRA weights as ``<name>.adapter.pt`` for
``scripts/eval_benchmark.py``.

Every run is stored as ``<output>/seed<seed>/<name>.json``; existing files are skipped, so
the command can be repeated to resume.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from rat.config import load_config
from rat.experiments import Study, save_result
from rat.pipeline import RunResult, fixed_permutation
from rat.routing import RoutingRecord


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def describe(result: RunResult) -> str:
    m, t, c = result.metrics, result.train, result.costs
    parts = [f"{result.name}: Pair {m.get('pair', float('nan')):.2f}"]
    if "heldout" in m:
        parts.append(f"held-out {m['heldout']:.2f} worst {m['worst']:.2f}")
    parts.append(f"collapse {m.get('collapse', float('nan')):.2f}")
    if t:
        parts.append(
            f"pairs {t['pairs']} H {100 * t['top_mode_share']:.1f}% vtok {t['visual_tokens']}"
        )
    if c:
        parts.append(f"cost {c.get('total', 0):.0f}/{c.get('budget', 0):.0f}s")
    return " | ".join(parts)


def load_result(path: Path) -> RunResult:
    data = json.loads(path.read_text())
    return RunResult(
        name=data["name"], seed=data["seed"], threshold=data["threshold"], routing=data["routing"],
        assignments=[tuple(a) for a in data["assignments"]], train=data["train"], costs=data["costs"],
        metrics=data["metrics"], extra=data["extra"],
    )  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", default=["main", "controls"])
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--prelim-h4", action="store_true")
    parser.add_argument(
        "--save-adapters", action="store_true", help="store the adapted LoRA weights of every run"
    )
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    seeds = args.seeds or config["seeds"]
    study = Study(config, args.output, log)
    study.prepare_costs()

    def run(seed: int, name: str, make) -> RunResult:
        path = args.output / f"seed{seed}" / f"{name}.json"
        if path.exists():
            return load_result(path)
        start = time.time()
        result = make()
        result.name = name
        save_result(result, path.parent)
        if args.save_adapters:
            torch.save(study.backbone.adapter_state(), path.with_suffix(".adapter.pt"))
        log(f"[seed {seed}] {describe(result)} ({time.time() - start:.0f}s)")
        return result

    if "lr-pilot" in args.stages:
        train, top = study.pool("train"), len(study.backbone.modes) - 1
        seed = seeds[0]
        pairs = fixed_permutation(len(train), seed, stream=7)[: config.get("lr_pilot_pairs", 300)]
        for rate in config.get("lr_pilot_rates", [2e-5, 1e-4, 2e-4, 5e-4]):
            study.train_config.learning_rate = rate
            result = run(
                seed, f"lr_pilot_{rate:g}",
                lambda: study.run_assigned(seed, "lr_pilot", [(p, top) for p in pairs], test="dev"),
            )  # fmt: skip
            log(
                f"lr {rate:g}: dev pair {result.metrics['pair']:.2f} single {result.metrics['single']:.2f}"
            )
        study.train_config.learning_rate = config["student"]["learning_rate"]

    for seed in seeds:
        rat = records = None
        if any(stage in args.stages for stage in ("main", "controls", "variants")):
            path = args.output / f"seed{seed}" / "rat.json"
            if path.exists():
                rat = load_result(path)
            else:
                rat = run(seed, "rat", lambda: study.run_gated(seed)[0])
            records = [
                RoutingRecord(
                    r["index"], r["status"], r["mode"], {int(k): v for k, v in r["margins"].items()}
                )
                for r in rat.extra["routing_records"]
            ]
        if "main" in args.stages:
            for mode, label in enumerate(m.name for m in study.backbone.modes):
                run(seed, f"random_{label}", lambda mode=mode: study.run_random(seed, mode))
            run(seed, "unadapted", lambda: study.run_unadapted(seed))
        if "controls" in args.stages:
            for name in ("random_alloc", "filter_only", "equal_random"):
                run(
                    seed,
                    name,
                    lambda name=name: study.controls_on_retained(seed, rat, records, [name])[0],
                )
        if "references" in args.stages:
            for mode, label in enumerate(m.name for m in study.backbone.modes):
                run(
                    seed,
                    f"full_{label}",
                    lambda mode=mode: study.run_random(seed, mode, budgeted=False),
                )
        if "variants" in args.stages:
            run(seed, "mean_margin", lambda: study.run_gated(seed, "mean_margin", rule="mean")[0])
            run(
                seed,
                "equal_loss",
                lambda: study.controls_on_retained(seed, rat, records, ["equal_loss"])[0],
            )
            run(seed, "alloc_only", lambda: study.run_allocation_only(seed, rat.threshold))
            run(
                seed,
                "student_margin",
                lambda: study.run_gated(seed, "student_margin", source="student")[0],
            )
        if "preliminary" in args.stages:
            conditions = ["H1", "L4", "L1", "U1"] + (["H4"] if args.prelim_h4 else [])
            for condition in conditions:
                run(seed, f"prelim_{condition}", lambda c=condition: study.run_preliminary(seed, c))
            run(seed, "prelim_unadapted", lambda: study.run_unadapted(seed, "prelim_test"))
    log("done")


if __name__ == "__main__":
    main()
