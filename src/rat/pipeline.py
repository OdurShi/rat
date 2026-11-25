"""Representation-Aware Training: threshold selection, routing and student adaptation
(Section 4.3-4.4, Appendix D, Algorithm S1).

The stages are

1. a pilot that calibrates operation-cost estimates (:func:`run_pilot`);
2. reader fitting on the pool ``F`` with both input modes (:func:`prepare_reader`);
3. one equal-budget student trial per threshold on the tuning pool, scored by paired
   accuracy on the development pool at mode ``H`` (:func:`select_threshold`);
4. lazy minimum-margin routing of the student pool with budget reservation, followed by one
   pass of answer-only cross-entropy (:func:`route_pool`, :func:`route_and_train`).

All stages charge one :class:`rat.budget.CostLedger`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .backbones.base import Backbone
from .budget import CostLedger, CostModel, TrainingCostEstimator
from .evaluation.paired import paired_metrics
from .features import PoolData
from .margins import branch_margins
from .reader import CandidateReader, ReaderFitConfig, fit_reader, score_pairs
from .routing import (
    EXCLUDED,
    RETAINED,
    UNPROCESSED,
    UNRETAINED,
    RoutingRecord,
    mode_order,
    routing_summary,
)
from .student import TrainConfig, TrainLog, predict, student_margins, train_pass


@dataclass
class Accounting:
    """How operations are charged: realised device time or the pilot's estimates."""

    cost_model: CostModel
    mode: str = "measured"
    devices: int = 1

    def cost(self, estimate: float, measured: float) -> float:
        return measured * self.devices if self.mode == "measured" else estimate


def fixed_permutation(size: int, seed: int, stream: int) -> list[int]:
    """Random order that is independent of labels and reader scores."""
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, stream])))
    return generator.permutation(size).tolist()


# --------------------------------------------------------------------------------------
# Pilot and reference budget
# --------------------------------------------------------------------------------------


def _fit_line(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Least-squares ``y = a + b x`` with non-negative coefficients."""
    x_array, y_array = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(set(x_array.tolist())) < 2:
        return 0.0, float(y_array.sum() / x_array.sum())
    slope, intercept = np.polyfit(x_array, y_array, 1)
    if intercept < 0 or slope < 0:
        return 0.0, float((x_array * y_array).sum() / (x_array * x_array).sum())
    return float(intercept), float(slope)


def run_pilot(
    backbone: Backbone,
    pool: PoolData,
    train_config: TrainConfig,
    pairs: Sequence[int],
    devices: int = 1,
    max_new_tokens: int = 24,
) -> tuple[CostModel, float]:
    """Calibrate the cost model on a few pairs of ``pool`` (Appendix D.2).

    Extraction costs are regressed on the token counts recorded when the pilot pairs were
    encoded; training costs on timed forward/backward passes in every mode; generation on a
    timed greedy pass at the evaluation mode.  The adapter is reset afterwards.  Returns the
    model and the device time spent, which belongs to the preparation cost.
    """
    start = time.perf_counter()
    modes = range(len(pool.modes))
    tokens, seconds = [], []
    for mode in modes:
        for pair in pairs:
            tokens.append(pool.costs(pair)[mode] / 2)
            seconds.append(pool.extraction_seconds(pair, mode) / 2 * devices)
    extract_fixed, extract_per_token = _fit_line(tokens, seconds)

    parameters = backbone.trainable_parameters()
    size = train_config.pairs_per_forward
    work, elapsed = [], []
    backbone.set_training(True)
    for mode in modes:
        for index, begin in enumerate(range(0, len(pairs) - size + 1, size)):
            chunk = pairs[begin : begin + size]
            examples = [pool.example(pair, branch, mode) for pair in chunk for branch in (0, 1)]
            lengths = [n for pair in chunk for n in pool.lengths(pair, mode)]
            for parameter in parameters:
                parameter.grad = None
            tic = time.perf_counter()
            backbone.answer_nll(examples).mean().backward()
            if backbone.device.type == "mps":
                torch.mps.synchronize()
            if index:  # the first pass of a mode warms up kernels and caches
                work.append(len(lengths) * max(lengths))
                elapsed.append((time.perf_counter() - tic) * devices)
    for parameter in parameters:
        parameter.grad = None
    backbone.set_training(False)
    train_fixed, train_per_token = _fit_line(work, elapsed)

    top = len(pool.modes) - 1
    probe = list(pairs[: max(2, size)])
    predict(backbone, pool, top, probe[:1], max_new_tokens=max_new_tokens)
    _answers, generation = predict(backbone, pool, top, probe, max_new_tokens=max_new_tokens)
    prompt_tokens = sum(n for pair in probe for n in pool.lengths(pair, top))
    backbone.reset_adapter()
    model = CostModel(
        extract_fixed=extract_fixed,
        extract_per_token=extract_per_token,
        train_fixed=train_fixed,
        train_per_token=train_per_token,
        update_fixed=0.0,
        generate_fixed=0.0,
        generate_per_token=generation * devices / prompt_tokens,
    )
    return model, (time.perf_counter() - start) * devices


def reference_budget(pool: PoolData, cost_model: CostModel, train_config: TrainConfig) -> float:
    """``B_ref``: one fixed-``H`` pass over the complete candidate pool, including the
    feature extraction it requires (Appendix E.2)."""
    top = len(pool.modes) - 1
    estimator = TrainingCostEstimator(
        cost_model, train_config.batch_pairs, train_config.pairs_per_forward
    )
    extraction = sum(cost_model.readout(pool.costs(pair)[top]) for pair in range(len(pool)))
    return extraction + estimator.pass_cost([pool.lengths(pair, top) for pair in range(len(pool))])


def extraction_cost(
    pool: PoolData, accounting: Accounting, modes: Sequence[int] | None = None
) -> float:
    """Cost of encoding a whole pool in the given modes (all modes by default)."""
    modes = range(len(pool.modes)) if modes is None else modes
    total = 0.0
    for mode in modes:
        for pair in range(len(pool)):
            total += accounting.cost(
                accounting.cost_model.readout(pool.costs(pair)[mode]),
                pool.extraction_seconds(pair, mode),
            )
    return total


# --------------------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------------------


def prepare_reader(
    backbone: Backbone,
    fit_pool: PoolData,
    config: ReaderFitConfig,
    seed: int,
    ledger: CostLedger | None = None,
    accounting: Accounting | None = None,
) -> tuple[CandidateReader, dict[str, float]]:
    """Fit the shared reader on ``F`` with every input mode and freeze it.

    Charges the extraction of the reader features, the candidate embeddings and the fit to
    the preparation cost.
    """
    embed_seconds = 0.0 if fit_pool.candidates is not None else fit_pool.embed_candidates()
    reader = CandidateReader(backbone.visual_dim, backbone.text_dim, config.init_gain, seed=seed)
    reader.to(backbone.device)
    tic = time.perf_counter()
    log = fit_reader(reader, fit_pool.fetch, fit_pool.candidates, len(fit_pool.modes), config, seed)
    fit_seconds = time.perf_counter() - tic
    for parameter in reader.parameters():
        parameter.requires_grad_(False)
    info = {
        "fit_seconds": fit_seconds,
        "embed_seconds": embed_seconds,
        "first_loss": log.losses[0],
        "final_loss": log.losses[-1],
        "steps": log.steps,
    }
    if ledger is not None and accounting is not None:
        ledger.charge("preparation", extraction_cost(fit_pool, accounting), "reader_features")
        ledger.charge(
            "preparation", (embed_seconds + fit_seconds) * accounting.devices, "reader_fit"
        )
    return reader, info


def reader_margin_table(
    reader: CandidateReader, pool: PoolData, batch_pairs: int = 64
) -> torch.Tensor:
    """Branch margins ``N x M x 2`` of a pool in every mode with a fixed reader."""
    if pool.candidates is None:
        pool.embed_candidates()
    pairs = range(len(pool))
    tables = [
        branch_margins(score_pairs(reader, pool.fetch, pool.candidates, pairs, mode, batch_pairs))
        for mode in range(len(pool.modes))
    ]
    return torch.stack(tables, dim=1)


def pair_scores(margins: torch.Tensor, rule: str = "min") -> torch.Tensor:
    """Pair score per mode from branch margins: ``min`` (Eq. (3)) or ``mean`` (control)."""
    if rule == "min":
        return margins.min(dim=-1).values
    if rule == "mean":
        return margins.mean(dim=-1)
    raise ValueError("rule must be 'min' or 'mean'")


# --------------------------------------------------------------------------------------
# Algorithm S1
# --------------------------------------------------------------------------------------


def route_pool(
    pool: PoolData,
    scores: torch.Tensor,
    threshold: float,
    order: Sequence[int],
    train_position: Sequence[int],
    ledger: CostLedger,
    accounting: Accounting,
    train_config: TrainConfig,
    allowed_modes: Sequence[int] | None = None,
    fallback_mode: int | None = None,
    charge_extraction: bool = True,
    score_seconds: torch.Tensor | None = None,
    enforce_budget: bool = True,
) -> list[RoutingRecord]:
    """Lines 1-14 of Algorithm S1: lazy routing with budget reservation.

    ``scores[i, a]`` is the pair score in mode ``a``; a mode passes when the score reaches
    ``threshold``.  ``order`` is the fixed traversal order and ``train_position[i]`` the
    position of pair ``i`` in the separate training order used for the reservation.
    ``fallback_mode`` assigns that mode to pairs that fail everywhere instead of excluding
    them (allocation-only control).  ``score_seconds[i, a]`` adds a measured scoring cost
    to every readout (student-margin control).
    """
    model = accounting.cost_model
    estimator = TrainingCostEstimator(
        model, train_config.batch_pairs, train_config.pairs_per_forward
    )
    records = {index: RoutingRecord(index, costs=pool.costs(index)) for index in order}
    allowed = set(range(scores.shape[1]) if allowed_modes is None else allowed_modes)
    stopped = False
    for index in order:
        record = records[index]
        chosen: int | None = None
        for mode in mode_order(record.costs):
            if mode not in allowed:
                continue
            extra = (
                float(score_seconds[index, mode]) * accounting.devices
                if score_seconds is not None
                else 0.0
            )
            estimate = (model.readout(record.costs[mode]) if charge_extraction else 0.0) + extra
            if enforce_budget and not ledger.affordable(estimate + estimator.total):
                stopped = True
                break
            measured = pool.extraction_seconds(index, mode) if charge_extraction else 0.0
            ledger.charge(
                "routing",
                accounting.cost(estimate, measured)
                + (extra if accounting.mode == "measured" else 0.0),
                "readouts",
            )
            record.margins[mode] = float(scores[index, mode])
            if record.margins[mode] >= threshold:
                chosen = mode
                break
        if stopped:
            break
        if chosen is None:
            if fallback_mode is None:
                record.status = EXCLUDED
                continue
            chosen = fallback_mode
        lengths = pool.lengths(index, chosen)
        if enforce_budget and not ledger.affordable(
            estimator.with_pair(train_position[index], lengths)
        ):
            record.status, record.mode = UNRETAINED, chosen
            break
        estimator.add(train_position[index], lengths)
        record.status, record.mode = RETAINED, chosen
    # Pairs after a budget stop keep the status "unprocessed".
    assert all(r.status in (RETAINED, EXCLUDED, UNRETAINED, UNPROCESSED) for r in records.values())
    return [records[index] for index in order]


def training_assignments(
    records: Sequence[RoutingRecord], train_position: Sequence[int]
) -> list[tuple[int, int]]:
    """Retained pairs with their modes, ordered by the fixed training permutation."""
    retained = [r for r in records if r.status == RETAINED]
    retained.sort(key=lambda r: train_position[r.index])
    return [(r.index, r.mode) for r in retained]


@dataclass
class RunResult:
    """Everything recorded for one adaptation run."""

    name: str
    seed: int
    threshold: float | None = None
    routing: dict[str, Any] = field(default_factory=dict)
    assignments: list[tuple[int, int]] = field(default_factory=list)
    train: dict[str, Any] = field(default_factory=dict)
    costs: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "threshold": self.threshold,
            "routing": self.routing,
            "assignments": [list(a) for a in self.assignments],
            "train": self.train,
            "costs": self.costs,
            "metrics": self.metrics,
            "extra": self.extra,
        }


def train_summary(log: TrainLog, pool: PoolData) -> dict[str, Any]:
    top = len(pool.modes) - 1
    return {
        "updates": log.updates,
        "pairs": log.pairs,
        "planned_pairs": log.planned_pairs,
        "visual_tokens": log.visual_tokens,
        "top_mode_share": log.mode_pairs.get(top, 0) / log.pairs if log.pairs else 0.0,
        "same_pair_top_tokens": sum(pool.costs(pair)[top] for pair, _mode in log.trained),
        "seconds": log.seconds,
        "stopped_by_budget": log.stopped_by_budget,
        "first_loss": log.losses[0] if log.losses else None,
        "last_loss": float(np.mean(log.losses[-5:])) if log.losses else None,
    }


def route_and_train(
    backbone: Backbone,
    pool: PoolData,
    scores: torch.Tensor,
    threshold: float,
    ledger: CostLedger,
    accounting: Accounting,
    train_config: TrainConfig,
    seed: int,
    name: str = "rat",
    candidates: Sequence[int] | None = None,
    **routing_options: Any,
) -> tuple[RunResult, list[RoutingRecord]]:
    """Algorithm S1 on the student pool: route, then one pass of Eq. (5) from ``theta_0``.

    ``candidates`` restricts the traversal to a subset of the pool (selectors that fix their
    subset before routing); the other pairs are never read and cost nothing.
    """
    order = fixed_permutation(len(pool), seed, stream=1)
    if candidates is not None:
        allowed = set(candidates)
        order = [index for index in order if index in allowed]
    train_order = fixed_permutation(len(pool), seed, stream=2)
    train_position = np.argsort(train_order).tolist()
    records = route_pool(
        pool,
        scores,
        threshold,
        order,
        train_position,
        ledger,
        accounting,
        train_config,
        **routing_options,
    )
    assignments = training_assignments(records, train_position)
    backbone.reset_adapter()
    log = train_pass(
        backbone,
        pool,
        assignments,
        train_config,
        ledger,
        accounting.cost_model,
        accounting.mode,
        accounting.devices,
    )
    result = RunResult(
        name=name,
        seed=seed,
        threshold=threshold,
        routing=routing_summary(records, top_mode=len(pool.modes) - 1),
        assignments=log.trained,
        train=train_summary(log, pool),
        costs=ledger.summary(unit=1.0),
    )
    return result, records


def evaluate(
    backbone: Backbone,
    pool: PoolData,
    batch_examples: int = 16,
    max_new_tokens: int = 128,
    pairs: Sequence[int] | None = None,
) -> tuple[dict[str, Any], list[tuple[str, str]], float]:
    """Paired metrics of the current student on ``pool`` at the evaluation mode ``H``."""
    top = len(pool.modes) - 1
    predictions, seconds = predict(backbone, pool, top, pairs, batch_examples, max_new_tokens)
    indices = range(len(pool)) if pairs is None else pairs
    answers = [pool.answers[i] for i in indices]
    records = [pool.records[i] for i in indices]
    return paired_metrics(predictions, answers, records), predictions, seconds


# --------------------------------------------------------------------------------------
# Threshold selection
# --------------------------------------------------------------------------------------


def select_threshold(
    backbone: Backbone,
    tune_pool: PoolData,
    tune_scores: torch.Tensor,
    dev_pool: PoolData,
    thresholds: Sequence[float],
    trial_budget: float,
    accounting: Accounting,
    train_config: TrainConfig,
    seed: int,
    batch_examples: int = 16,
    max_new_tokens: int = 128,
    **routing_options: Any,
) -> tuple[float, list[dict[str, Any]]]:
    """Appendix D.1: one equal-budget trial per threshold from the same initialisation.

    Every trial routes the tuning pool (whose features are cached and charged once to
    preparation), trains a fresh student and scores it by paired accuracy on the
    development pool at mode ``H``; routing, updates and development scoring share
    ``trial_budget``.  The highest score wins; ties favour the lower measured trial cost and
    then the smaller threshold.  The trial weights are discarded.
    """
    top = len(dev_pool.modes) - 1
    dev_estimate = accounting.cost_model.generation(
        [n for pair in range(len(dev_pool)) for n in dev_pool.lengths(pair, top)]
    )
    order = fixed_permutation(len(tune_pool), seed, stream=3)
    train_order = fixed_permutation(len(tune_pool), seed, stream=4)
    train_position = np.argsort(train_order).tolist()
    trials = []
    for threshold in thresholds:
        ledger = CostLedger(trial_budget)
        # Development scoring is reserved first so that routing and updates fit beside it.
        ledger.charge("preparation", dev_estimate, "dev_scoring")
        records = route_pool(
            tune_pool,
            tune_scores,
            threshold,
            order,
            train_position,
            ledger,
            accounting,
            train_config,
            charge_extraction=False,
            **routing_options,
        )
        backbone.reset_adapter()
        log = train_pass(
            backbone,
            tune_pool,
            training_assignments(records, train_position),
            train_config,
            ledger,
            accounting.cost_model,
            accounting.mode,
            accounting.devices,
        )
        metrics, _predictions, seconds = evaluate(
            backbone, dev_pool, batch_examples, max_new_tokens
        )
        if accounting.mode == "measured":
            ledger.preparation += seconds * accounting.devices - dev_estimate
        # Retention and the H share describe the gate on the whole tuning pool.
        passing = tune_scores >= threshold
        retained = passing.any(dim=1)
        low_pass = torch.zeros(len(tune_pool), dtype=torch.bool)
        for index in range(len(tune_pool)):
            first = next((m for m in mode_order(tune_pool.costs(index)) if passing[index, m]), None)
            low_pass[index] = first is not None and first != top
        trials.append(
            {
                "threshold": float(threshold),
                "retained_percent": 100.0 * float(retained.float().mean()),
                "top_mode_percent": 100.0
                * float((retained & ~low_pass).sum())
                / max(1, int(retained.sum())),
                "dev_pair": metrics["pair"],
                "cost": ledger.total,
                "trained_pairs": log.pairs,
                "updates": log.updates,
            }
        )
    backbone.reset_adapter()
    best = min(trials, key=lambda t: (-t["dev_pair"], t["cost"], t["threshold"]))
    return best["threshold"], trials


def student_margin_table(
    backbone: Backbone, pool: PoolData, batch_examples: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    """Branch margins ``N x M x 2`` of the initial student and scoring seconds ``N x M``."""
    backbone.reset_adapter()
    pairs = list(range(len(pool)))
    margins, seconds = [], []
    for mode in range(len(pool.modes)):
        values, spent = student_margins(backbone, pool, mode, pairs, batch_examples)
        margins.append(values)
        seconds.append(spent)
    return torch.stack(margins, dim=1), torch.stack(seconds, dim=1)


def token_saving_percent(assignments: Sequence[tuple[int, int]], pool: PoolData) -> float:
    """``100 [1 - sum_i c_i(a_i) / sum_i c_i(H)]`` over the trained pairs (Appendix E.3)."""
    top = len(pool.modes) - 1
    actual = sum(pool.costs(pair)[mode] for pair, mode in assignments)
    reference = sum(pool.costs(pair)[top] for pair, _mode in assignments)
    return 100.0 * (1.0 - actual / reference) if reference else 0.0


__all__ = [
    "Accounting",
    "RunResult",
    "evaluate",
    "extraction_cost",
    "fixed_permutation",
    "pair_scores",
    "prepare_reader",
    "reader_margin_table",
    "reference_budget",
    "route_and_train",
    "route_pool",
    "run_pilot",
    "select_threshold",
    "student_margin_table",
    "token_saving_percent",
    "train_summary",
    "training_assignments",
]
