"""Student adaptation (Section 4.4, Eq. (5), Appendix E.4).

The student starts from ``theta_0`` and makes one pass over the retained pairs in a fixed
shuffled order.  Both branches of a pair enter the same optimiser update; the loss is the
answer-only cross-entropy, summed over answer tokens (including the end token) and averaged
over examples.  Only the language-side LoRA parameters are updated.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from .backbones.base import Backbone
from .budget import CostLedger, CostModel
from .features import PoolData


@dataclass
class TrainConfig:
    """Optimisation settings of Appendix E.4."""

    learning_rate: float = 2e-5
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    warmup_fraction: float = 0.03
    max_grad_norm: float = 1.0
    #: Pairs per optimiser update (the paper uses 32 pairs, i.e. 64 examples).
    batch_pairs: int = 32
    #: Pairs per forward pass; an update accumulates ``batch_pairs / pairs_per_forward`` passes.
    pairs_per_forward: int = 1

    @classmethod
    def from_dict(cls, values: dict) -> TrainConfig:
        values = dict(values)
        if "betas" in values:
            values["betas"] = tuple(values["betas"])
        return cls(**values)


@dataclass
class TrainLog:
    updates: int = 0
    pairs: int = 0
    planned_pairs: int = 0
    visual_tokens: int = 0
    seconds: float = 0.0
    charged: float = 0.0
    stopped_by_budget: bool = False
    losses: list[float] = field(default_factory=list)
    mode_pairs: dict[int, int] = field(default_factory=dict)
    trained: list[tuple[int, int]] = field(default_factory=list)


def learning_rate_factor(update: int, total_updates: int, warmup_fraction: float) -> float:
    """Linear warm-up over ``ceil(warmup_fraction * total)`` updates, then cosine decay to zero.

    ``update`` counts from zero; the first update uses ``1 / warmup`` of the peak rate.
    """
    warmup = max(1, math.ceil(warmup_fraction * total_updates))
    if update < warmup:
        return (update + 1) / warmup
    span = max(1, total_updates - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * (update - warmup) / span))


def student_loss(nll: torch.Tensor) -> torch.Tensor:
    """Eq. (5) on a batch: per-example summed answer NLL, averaged over examples."""
    return nll.mean()


def train_pass(
    backbone: Backbone,
    pool: PoolData,
    assignments: Sequence[tuple[int, int]],
    config: TrainConfig,
    ledger: CostLedger | None = None,
    cost_model: CostModel | None = None,
    accounting: str = "measured",
    devices: int = 1,
    passes: int = 1,
    on_update: Callable[[int, int, float], None] | None = None,
) -> TrainLog:
    """Train the student on ``assignments = [(pair, mode), ...]`` in the given order.

    With a ledger the remaining budget is checked against the estimated cost before every
    update, and the realised cost (``accounting="measured"``, device seconds times
    ``devices``) or its estimate (``"modelled"``) is charged afterwards.  ``passes > 1``
    repeats the same order, as in the multi-pass conditions of the preliminary study.
    """
    if accounting not in ("measured", "modelled"):
        raise ValueError("accounting must be 'measured' or 'modelled'")
    log = TrainLog(planned_pairs=len(assignments) * passes)
    if not assignments:
        return log  # an empty retained set returns theta_0
    schedule = [pair for _ in range(passes) for pair in assignments]
    parameters = backbone.trainable_parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    total_updates = math.ceil(len(schedule) / config.batch_pairs)
    backbone.set_training(True)
    for update in range(total_updates):
        batch = schedule[update * config.batch_pairs : (update + 1) * config.batch_pairs]
        chunks = [
            batch[k : k + config.pairs_per_forward]
            for k in range(0, len(batch), config.pairs_per_forward)
        ]
        estimate = 0.0
        if cost_model is not None:
            estimate = cost_model.update_fixed + sum(
                cost_model.chunk([n for pair, mode in chunk for n in pool.lengths(pair, mode)])
                for chunk in chunks
            )
        if ledger is not None and not ledger.affordable(estimate):
            log.stopped_by_budget = True
            break
        tic = time.perf_counter()
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * learning_rate_factor(
                update, total_updates, config.warmup_fraction
            )
        optimizer.zero_grad(set_to_none=True)
        examples_in_update = 2 * len(batch)
        value = 0.0
        for chunk in chunks:
            examples = [
                pool.example(pair, branch, mode) for pair, mode in chunk for branch in (0, 1)
            ]
            loss = backbone.answer_nll(examples).sum() / examples_in_update
            loss.backward()
            value += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
        optimizer.step()
        if backbone.device.type == "mps":
            torch.mps.synchronize()
        elapsed = (time.perf_counter() - tic) * devices
        charged = elapsed if accounting == "measured" else estimate
        if ledger is not None:
            ledger.charge("training", charged, "student_updates")
        log.updates += 1
        log.pairs += len(batch)
        log.seconds += elapsed
        log.charged += charged
        log.losses.append(value)
        for pair, mode in batch:
            log.visual_tokens += pool.costs(pair)[mode]
            log.mode_pairs[mode] = log.mode_pairs.get(mode, 0) + 1
        log.trained.extend(batch)
        if on_update is not None:
            on_update(update + 1, total_updates, value)
    backbone.set_training(False)
    return log


@torch.no_grad()
def predict(
    backbone: Backbone,
    pool: PoolData,
    mode: int,
    pairs: Sequence[int] | None = None,
    batch_examples: int = 16,
    max_new_tokens: int = 128,
) -> tuple[list[tuple[str, str]], float]:
    """Greedy answers for both branches of every pair at a fixed mode.

    Returns ``[(answer of branch 0, answer of branch 1), ...]`` and the seconds spent.
    Examples are ordered by sequence length so that batches need little padding.
    """
    pairs = list(range(len(pool))) if pairs is None else list(pairs)
    items = [(position, branch) for position in range(len(pairs)) for branch in (0, 1)]
    items.sort(key=lambda item: -pool.lengths(pairs[item[0]], mode)[item[1]])
    answers: list[list[str]] = [["", ""] for _ in pairs]
    tic = time.perf_counter()
    for start in range(0, len(items), batch_examples):
        chunk = items[start : start + batch_examples]
        examples = [pool.example(pairs[p], b, mode, with_answer=False) for p, b in chunk]
        for (position, branch), text in zip(
            chunk, backbone.generate(examples, max_new_tokens), strict=True
        ):
            answers[position][branch] = text
    return [tuple(a) for a in answers], time.perf_counter() - tic  # type: ignore[misc]


@torch.no_grad()
def answer_losses(
    backbone: Backbone, pool: PoolData, mode: int, pairs: Sequence[int], batch_examples: int = 16
) -> torch.Tensor:
    """Per-pair answer loss of Eq. (5) (mean of the two branches) without adaptation."""
    backbone.set_training(False)
    out = torch.zeros(len(pairs))
    items = [(position, branch) for position in range(len(pairs)) for branch in (0, 1)]
    for start in range(0, len(items), batch_examples):
        chunk = items[start : start + batch_examples]
        examples = [pool.example(pairs[p], b, mode) for p, b in chunk]
        for (position, _branch), value in zip(
            chunk, backbone.answer_nll(examples).cpu().tolist(), strict=True
        ):
            out[position] += value / 2.0
    return out


@torch.no_grad()
def student_margins(
    backbone: Backbone, pool: PoolData, mode: int, pairs: Sequence[int], batch_examples: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-candidate log-likelihood margins of the student for both branches (Appendix E.3).

    Returns the margins (``len(pairs) x 2``) and the seconds spent on each pair.
    """
    backbone.set_training(False)
    margins = torch.zeros((len(pairs), 2))
    seconds = torch.zeros(len(pairs))
    per_batch = max(1, batch_examples // 2)
    for start in range(0, len(pairs), per_batch):
        chunk = pairs[start : start + per_batch]
        examples = [pool.example(pair, branch, mode) for pair in chunk for branch in (0, 1)]
        alternatives = [pool.answers[pair][1 - branch] for pair in chunk for branch in (0, 1)]
        tic = time.perf_counter()
        values = backbone.candidate_margins(examples, alternatives).cpu().view(len(chunk), 2)
        if backbone.device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - tic
        margins[start : start + len(chunk)] = values
        seconds[start : start + len(chunk)] = elapsed / len(chunk)
    return margins, seconds
