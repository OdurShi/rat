"""Experiment drivers: the main comparison, the core controls and the preliminary study.

:class:`Study` holds the backbone, the cached pools, the cost model and the per-seed reader
and exposes one method per row of Tables 1 and 4.  Every method returns a
:class:`rat.pipeline.RunResult`; ``scripts/run_study.py`` stores them as JSON and
``scripts/report_study.py`` turns them into tables.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .backbones.base import LoraSettings
from .budget import CostLedger, CostModel
from .config import build_backbone
from .data.splits import load_pool
from .features import FeatureStore, PoolData, open_pool
from .pipeline import (
    Accounting,
    RunResult,
    evaluate,
    extraction_cost,
    fixed_permutation,
    pair_scores,
    prepare_reader,
    reader_margin_table,
    reference_budget,
    route_and_train,
    route_pool,
    run_pilot,
    select_threshold,
    student_margin_table,
    token_saving_percent,
    train_summary,
    training_assignments,
)
from .reader import ReaderFitConfig
from .routing import RETAINED, permute_assignments_within_strata, routing_summary
from .student import TrainConfig, answer_losses, train_pass

#: Retention fractions tried by selectors other than RAT (Appendix E.2).
RETENTION_FRACTIONS = (0.2, 0.5, 1.0)


class Study:
    """One backbone with its pools; runs are independent given the seed."""

    def __init__(self, config: dict[str, Any], run_dir: str | Path, log=print) -> None:
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.backbone = build_backbone(config)
        lora = dict(config["lora"])
        lora["targets"] = tuple(lora["targets"])
        self.lora = LoraSettings(**lora)
        self.train_config = TrainConfig.from_dict(config["student"])
        self.reader_config = ReaderFitConfig(**config["reader"])
        self.thresholds = list(config["routing"]["thresholds"])
        self.eval_options = {
            "batch_examples": config["evaluation"]["batch_examples"],
            "max_new_tokens": config["evaluation"]["max_new_tokens"],
        }
        self.pools: dict[str, PoolData] = {}
        self._adapter_seed: int | None = None
        self._readers: dict[int, dict[str, Any]] = {}
        self.accounting: Accounting | None = None
        self.budget: float | None = None
        self.pilot_seconds = 0.0

    # ---- data ---------------------------------------------------------------------

    def pool(self, name: str) -> PoolData:
        if name not in self.pools:
            data = self.config["data"]
            records = load_pool(name, data[name], data["manifests"])
            both = name in ("reader_fit", "tune", "train", "prelim_train")
            modes = self.backbone.modes if both else (self.backbone.eval_mode,)
            self.pools[name] = open_pool(
                self.backbone, records, self.config["cache_dir"], name, modes
            )
        return self.pools[name]

    def degraded_pool(self, name: str) -> PoolData:
        """Pool whose top mode carries low-mode image detail on the high-mode grid (``U``)."""
        key = f"{name}/U"
        if key not in self.pools:
            base = self.pool(name)
            low, high = self.backbone.modes[0], self.backbone.eval_mode
            directory = Path(self.config["cache_dir"]) / f"{name}-{len(base)}-U"
            if FeatureStore.exists(directory):
                store = FeatureStore(directory)
            else:
                store = FeatureStore.build(directory, self.backbone, base.records, high, detail=low)
            self.pools[key] = PoolData(base.records, self.backbone, [base.stores[0], store])
        return self.pools[key]

    # ---- student initialisation ---------------------------------------------------

    def start_seed(self, seed: int) -> None:
        """Attach or re-initialise LoRA so that every run of ``seed`` starts from the same
        ``theta_0``."""
        if self._adapter_seed is None:
            self.backbone.add_adapter(self.lora, seed)
        if self._adapter_seed != seed:
            reinitialise_adapter(self.backbone, seed)
        self._adapter_seed = seed
        self.backbone.reset_adapter()

    # ---- costs --------------------------------------------------------------------

    def prepare_costs(self) -> None:
        """Pilot (cached in the run directory) and reference budget ``B_ref``."""
        if self.accounting is not None:
            return
        budget = self.config["budget"]
        path = self.run_dir / "cost_model.json"
        train = self.pool("train")
        if path.exists():
            stored = json.loads(path.read_text())
            model, self.pilot_seconds = CostModel(**stored["model"]), stored["pilot_seconds"]
        else:
            if self._adapter_seed is None:
                self.start_seed(self.config["seeds"][0])
            pairs = fixed_permutation(len(train), 0, stream=9)[
                : 6 * self.train_config.pairs_per_forward
            ]
            model, self.pilot_seconds = run_pilot(
                self.backbone,
                train,
                self.train_config,
                pairs,
                budget["devices"],
                self.eval_options["max_new_tokens"],
            )
            path.write_text(
                json.dumps(
                    {"model": model.to_dict(), "pilot_seconds": self.pilot_seconds}, indent=2
                )
            )
        self.accounting = Accounting(model, budget["accounting"], budget["devices"])
        if budget["reference_hours"] is not None:
            self.budget = 3600.0 * budget["reference_hours"]
        else:
            self.budget = reference_budget(train, model, self.train_config)
        self.log(f"cost model {model.to_dict()}")
        self.log(f"B_ref = {self.budget:.1f} device-seconds; pilot {self.pilot_seconds:.1f}s")

    @property
    def trial_budget(self) -> float:
        return self.config["routing"]["trial_fraction"] * self.budget

    # ---- reader -------------------------------------------------------------------

    def reader(self, seed: int) -> dict[str, Any]:
        """Fitted reader of ``seed`` with its margin tables and preparation costs."""
        if seed in self._readers:
            return self._readers[seed]
        self.prepare_costs()
        fit, tune, train = self.pool("reader_fit"), self.pool("tune"), self.pool("train")
        ledger = CostLedger(float("inf"))
        reader, info = prepare_reader(
            self.backbone, fit, self.reader_config, seed, ledger, self.accounting
        )
        entry = {
            "reader": reader,
            "info": info,
            "fit_cost": ledger.preparation,
            "tune": reader_margin_table(reader, tune),
            "train": reader_margin_table(reader, train),
            "fit": reader_margin_table(reader, fit),
        }
        torch.save(
            {
                "state": reader.state_dict(),
                "tune": entry["tune"],
                "train": entry["train"],
                "fit": entry["fit"],
                "info": info,
            },
            self.run_dir / f"reader_seed{seed}.pt",
        )
        self._readers[seed] = entry
        self.log(
            f"[seed {seed}] reader fitted: loss {info['first_loss']:.4f} -> {info['final_loss']:.4f} in {info['fit_seconds']:.1f}s"
        )
        return entry

    def shared_preparation(self, modes: Sequence[int] | None = None) -> float:
        """Preparation shared by every budgeted method: pilot, tuning features (cached before
        the trials) and development features."""
        tune, dev = self.pool("tune"), self.pool("dev")
        top = len(dev.modes) - 1
        return (
            self.pilot_seconds
            + extraction_cost(tune, self.accounting, modes)
            + extraction_cost(dev, self.accounting, [top])
        )

    # ---- evaluation ---------------------------------------------------------------

    def finish(self, result: RunResult, test: str = "test") -> RunResult:
        metrics, predictions, seconds = evaluate(
            self.backbone, self.pool(test), **self.eval_options
        )
        result.metrics = metrics
        result.extra["predictions"] = [list(p) for p in predictions]
        result.extra["eval_seconds"] = seconds
        return result

    # ---- RAT and score variants ---------------------------------------------------

    def run_gated(
        self,
        seed: int,
        name: str = "rat",
        rule: str = "min",
        source: str = "reader",
        evaluate_run: bool = True,
    ) -> tuple[RunResult, list]:
        """RAT (``rule="min"``, ``source="reader"``) or one of the score variants of
        Appendix E.3 (mean branch margin; initial-student candidate margin)."""
        self.prepare_costs()
        self.start_seed(seed)
        tune, dev, train = self.pool("tune"), self.pool("dev"), self.pool("train")
        ledger = CostLedger(self.budget)
        ledger.charge("preparation", self.shared_preparation(), "shared")
        options: dict[str, Any] = {}
        tune_options: dict[str, Any] = {}
        if source == "reader":
            entry = self.reader(seed)
            ledger.charge("preparation", entry["fit_cost"], "reader")
            tune_scores, train_scores = (
                pair_scores(entry["tune"], rule),
                pair_scores(entry["train"], rule),
            )
        else:
            tune_margins, tune_seconds = student_margin_table(self.backbone, tune)
            train_margins, train_seconds = student_margin_table(self.backbone, train)
            tune_scores, train_scores = (
                pair_scores(tune_margins, rule),
                pair_scores(train_margins, rule),
            )
            options["score_seconds"] = train_seconds
            tune_options["score_seconds"] = tune_seconds
        threshold, trials = select_threshold(
            self.backbone, tune, tune_scores, dev, self.thresholds, self.trial_budget,
            self.accounting, self.train_config, seed, **self.eval_options, **tune_options,
        )  # fmt: skip
        ledger.charge("preparation", sum(t["cost"] for t in trials), "threshold_trials")
        self.log(
            f"[seed {seed}] {name}: trials {[(round(t['threshold'], 3), round(t['dev_pair'], 2), round(t['retained_percent'], 1)) for t in trials]} -> gamma {threshold:.3f}"
        )
        result, records = route_and_train(
            self.backbone,
            train,
            train_scores,
            threshold,
            ledger,
            self.accounting,
            self.train_config,
            seed,
            name,
            **options,
        )
        result.extra["trials"] = trials
        result.extra["token_saving_percent"] = token_saving_percent(result.assignments, train)
        result.extra["routing_records"] = [
            {"index": r.index, "status": r.status, "mode": r.mode, "margins": r.margins}
            for r in records
        ]
        return (self.finish(result) if evaluate_run else result), records

    # ---- fixed assignments --------------------------------------------------------

    def run_assigned(
        self,
        seed: int,
        name: str,
        assignments: Sequence[tuple[int, int]],
        base_costs: dict[str, float] | None = None,
        extra_routing: float = 0.0,
        pool_name: str = "train",
        passes: int = 1,
        test: str = "test",
        pool: PoolData | None = None,
    ) -> RunResult:
        """One complete pass over given ``(pair, mode)`` assignments; the realised cost is
        recorded without a budget stop (shared control protocol of Appendix E.3)."""
        self.prepare_costs()
        self.start_seed(seed)
        pool = pool or self.pool(pool_name)
        ledger = CostLedger(float("inf"))
        if base_costs:
            ledger.charge("preparation", base_costs.get("preparation", 0.0))
            ledger.charge("routing", base_costs.get("routing", 0.0))
        ledger.charge("routing", extra_routing)
        log = train_pass(
            self.backbone, pool, list(assignments), self.train_config, ledger,
            self.accounting.cost_model, self.accounting.mode, self.accounting.devices, passes=passes,
        )  # fmt: skip
        result = RunResult(
            name=name, seed=seed, assignments=log.trained, train=train_summary(log, pool)
        )
        result.costs = ledger.summary(unit=1.0)
        result.costs["budget"] = self.budget
        result.extra["token_saving_percent"] = token_saving_percent(result.assignments, pool)
        return self.finish(result, test)

    def _mode_extraction(self, pool: PoolData, pairs_modes: Sequence[tuple[int, int]]) -> float:
        model = self.accounting.cost_model
        return sum(
            self.accounting.cost(
                model.readout(pool.costs(pair)[mode]), pool.extraction_seconds(pair, mode)
            )
            for pair, mode in pairs_modes
        )

    def controls_on_retained(
        self, seed: int, rat: RunResult, records: list, which: Sequence[str]
    ) -> list[RunResult]:
        """Controls that share RAT's retained set ``S_R`` or its size (Appendix E.3)."""
        train = self.pool("train")
        top = len(train.modes) - 1
        assignments = [tuple(a) for a in rat.assignments]
        evaluated = {r.index: set(r.margins) for r in records}
        base = {"preparation": rat.costs["preparation"], "routing": rat.costs["routing"]}
        out = []
        if "filter_only" in which:
            needed = [(pair, top) for pair, _mode in assignments if top not in evaluated[pair]]
            out.append(
                self.run_assigned(
                    seed,
                    "filter_only",
                    [(pair, top) for pair, _ in assignments],
                    base,
                    self._mode_extraction(train, needed),
                )
            )
        if "random_alloc" in which:
            modes = permute_assignments_within_strata(
                [mode for _pair, mode in assignments],
                [train.costs(pair) for pair, _ in assignments],
                seed,
            )
            shuffled = [(pair, mode) for (pair, _old), mode in zip(assignments, modes, strict=True)]
            needed = [(pair, mode) for pair, mode in shuffled if mode not in evaluated[pair]]
            result = self.run_assigned(
                seed, "random_alloc", shuffled, base, self._mode_extraction(train, needed)
            )
            result.extra["changed_assignments"] = sum(
                a != b for a, b in zip(assignments, shuffled, strict=True)
            )
            out.append(result)
        count = len(assignments)
        train_order = fixed_permutation(len(train), seed, stream=2)
        position = np.argsort(train_order)
        pilot = {"preparation": self.pilot_seconds, "routing": 0.0}
        if "equal_random" in which:
            chosen = sorted(
                fixed_permutation(len(train), seed, stream=5)[:count], key=lambda i: position[i]
            )
            picks = [(pair, top) for pair in chosen]
            out.append(
                self.run_assigned(
                    seed, "equal_random", picks, pilot, self._mode_extraction(train, picks)
                )
            )
        if "equal_loss" in which:
            self.start_seed(seed)
            tic = time.perf_counter()
            losses = answer_losses(
                self.backbone,
                train,
                top,
                list(range(len(train))),
                self.eval_options["batch_examples"],
            )
            scoring = (time.perf_counter() - tic) * self.accounting.devices
            # Descending initial answer loss; ties follow the fixed candidate order.
            candidate_order = fixed_permutation(len(train), seed, stream=1)
            rank = {pair: k for k, pair in enumerate(candidate_order)}
            chosen = sorted(range(len(train)), key=lambda i: (-float(losses[i]), rank[i]))[:count]
            chosen.sort(key=lambda i: position[i])
            everything = [(pair, top) for pair in range(len(train))]
            out.append(
                self.run_assigned(
                    seed,
                    "equal_loss",
                    [(pair, top) for pair in chosen],
                    pilot,
                    self._mode_extraction(train, everything) + scoring,
                )  # fmt: skip
            )
        return out

    def run_allocation_only(self, seed: int, threshold: float) -> RunResult:
        """Full pool, first passing mode, fallback to ``H`` after failing every mode."""
        self.prepare_costs()
        self.start_seed(seed)
        train = self.pool("train")
        entry = self.reader(seed)
        ledger = CostLedger(float("inf"))
        ledger.charge("preparation", self.shared_preparation() + entry["fit_cost"])
        result, _records = route_and_train(
            self.backbone, train, pair_scores(entry["train"], "min"), threshold, ledger, self.accounting,
            self.train_config, seed, "alloc_only", fallback_mode=len(train.modes) - 1, enforce_budget=False,
        )  # fmt: skip
        result.costs["budget"] = self.budget
        result.extra["token_saving_percent"] = token_saving_percent(result.assignments, train)
        return self.finish(result)

    # ---- uniform selection --------------------------------------------------------

    def run_random(self, seed: int, mode: int, budgeted: bool = True) -> RunResult:
        """Uniform random selection at a fixed mode.

        With ``budgeted`` the method receives the same development-search allowance as RAT:
        three trials over the retention fractions, then a final run that stops before the
        budget is exceeded.  Without it the full pool is trained once (reference row).
        """
        self.prepare_costs()
        self.start_seed(seed)
        train, tune, dev = self.pool("train"), self.pool("tune"), self.pool("dev")
        label = train.modes[mode].name
        scores = torch.full((len(train), len(train.modes)), float("-inf"))
        scores[:, mode] = float("inf")
        if not budgeted:
            ledger = CostLedger(float("inf"))
            result, _ = route_and_train(
                self.backbone, train, scores, 1.0, ledger, self.accounting, self.train_config, seed,
                f"full_{label}", allowed_modes=[mode], enforce_budget=False,
            )  # fmt: skip
            result.costs["budget"] = self.budget
            return self.finish(result)
        ledger = CostLedger(self.budget)
        ledger.charge("preparation", self.shared_preparation([mode]), "shared")
        tune_scores = torch.full((len(tune), len(tune.modes)), float("-inf"))
        tune_scores[:, mode] = float("inf")
        trials = []
        order = fixed_permutation(len(tune), seed, stream=3)
        position = np.argsort(fixed_permutation(len(tune), seed, stream=4)).tolist()
        dev_estimate = self.accounting.cost_model.generation(
            [n for pair in range(len(dev)) for n in dev.lengths(pair, len(dev.modes) - 1)]
        )
        for fraction in RETENTION_FRACTIONS:
            trial = CostLedger(self.trial_budget)
            trial.charge("preparation", dev_estimate)
            subset = order[: max(1, round(fraction * len(tune)))]
            records = route_pool(
                tune, tune_scores, 1.0, subset, position, trial, self.accounting, self.train_config,
                allowed_modes=[mode], charge_extraction=False,
            )  # fmt: skip
            self.backbone.reset_adapter()
            log = train_pass(
                self.backbone, tune, training_assignments(records, position), self.train_config, trial,
                self.accounting.cost_model, self.accounting.mode, self.accounting.devices,
            )  # fmt: skip
            metrics, _p, seconds = evaluate(self.backbone, dev, **self.eval_options)
            if self.accounting.mode == "measured":
                trial.preparation += seconds * self.accounting.devices - dev_estimate
            trials.append(
                {
                    "fraction": fraction,
                    "dev_pair": metrics["pair"],
                    "cost": trial.total,
                    "trained_pairs": log.pairs,
                }
            )
        ledger.charge("preparation", sum(t["cost"] for t in trials), "retention_trials")
        best = min(trials, key=lambda t: (-t["dev_pair"], t["cost"], t["fraction"]))
        self.backbone.reset_adapter()
        full_order = fixed_permutation(len(train), seed, stream=1)
        keep = full_order[: max(1, round(best["fraction"] * len(train)))]
        result, _ = route_and_train(
            self.backbone,
            train,
            scores,
            1.0,
            ledger,
            self.accounting,
            self.train_config,
            seed,
            f"random_{label}",
            candidates=keep,
            allowed_modes=[mode],
        )
        result.extra["trials"] = trials
        result.extra["retention_fraction"] = best["fraction"]
        return self.finish(result)

    def run_unadapted(self, seed: int, test: str = "test") -> RunResult:
        self.prepare_costs()
        self.start_seed(seed)
        self.backbone.set_adapter_enabled(False)
        try:
            result = self.finish(RunResult(name="unadapted", seed=seed), test)
        finally:
            self.backbone.set_adapter_enabled(True)
        result.costs = {
            "preparation": 0.0,
            "routing": 0.0,
            "training": 0.0,
            "total": 0.0,
            "budget": self.budget,
        }
        return result

    # ---- preliminary study --------------------------------------------------------

    def run_preliminary(self, seed: int, condition: str) -> RunResult:
        """One condition of Section 3: ``L1``, ``L4``, ``U1``, ``H1`` or ``H4``."""
        kind, passes = condition[0], int(condition[1:])
        pool = self.degraded_pool("prelim_train") if kind == "U" else self.pool("prelim_train")
        mode = 0 if kind == "L" else len(pool.modes) - 1
        order = fixed_permutation(len(pool), seed, stream=6)
        return self.run_assigned(
            seed,
            f"prelim_{condition}",
            [(pair, mode) for pair in order],
            passes=passes,
            test="prelim_test",
            pool=pool,
        )


def reinitialise_adapter(backbone, seed: int) -> None:
    """Draw a new ``theta_0`` for another seed: LoRA ``A`` Kaiming-uniform, ``B`` zero."""
    generator = torch.Generator().manual_seed(seed)
    state = {}
    for name, value in backbone.adapter_state().items():
        if "lora_A" in name:
            bound = 1.0 / (value.shape[1] ** 0.5)
            state[name] = (torch.rand(value.shape, generator=generator) * 2 - 1) * bound
        else:
            state[name] = torch.zeros_like(value)
    backbone.load_adapter_state(state)
    backbone.remember_initial_adapter()


def save_result(result: RunResult, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result.name}.json"
    path.write_text(json.dumps(result.to_dict(), indent=1))
    return path


def summarise_routing(records: list, top: int) -> dict[str, float]:
    return routing_summary(records, top)


__all__ = ["RETAINED", "RETENTION_FRACTIONS", "Study", "reinitialise_adapter", "save_result"]
