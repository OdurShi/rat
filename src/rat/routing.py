"""Minimum-margin input routing (Section 4.3, Eq. (4), Appendix C.2).

A pair is assigned the passing input mode with the fewest visual tokens; ties favour the
mode with less detail, and a pair that passes in no mode is excluded.  Modes are evaluated
lazily in increasing ``(cost, rank)`` order and evaluation stops at the first passing mode,
which gives exactly the choice of Eq. (4) for any ordering of the margin values.

The functions accept any number of modes, so the same rule serves the two modes ``{L, H}``
of the paper and longer resolution ladders.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

#: Status values of a candidate pair after Algorithm S1.
RETAINED = "retained"  # passed and admitted to the student training set
EXCLUDED = "excluded"  # failed in every mode
UNRETAINED = "unretained"  # passed, but its training cost no longer fits the budget
UNPROCESSED = "unprocessed"  # not reached, or its readout no longer fits the budget


def mode_order(costs: Sequence[int], ranks: Sequence[int] | None = None) -> list[int]:
    """Mode indices in increasing ``(cost, rank)`` order (Appendix C.2)."""
    ranks = range(len(costs)) if ranks is None else ranks
    return sorted(range(len(costs)), key=lambda a: (costs[a], ranks[a]))


def route(
    margins: Sequence[float],
    costs: Sequence[int],
    threshold: float,
    ranks: Sequence[int] | None = None,
) -> int | None:
    """Eq. (4): the cheapest mode whose margin reaches ``threshold``, or ``None``."""
    if threshold <= 0:
        raise ValueError("the margin threshold must be positive")
    for mode in mode_order(costs, ranks):
        if margins[mode] >= threshold:
            return mode
    return None


def lazy_route(
    margin_of: Callable[[int], float],
    costs: Sequence[int],
    threshold: float,
    ranks: Sequence[int] | None = None,
) -> tuple[int | None, dict[int, float]]:
    """Lazy evaluation of Eq. (4).

    ``margin_of(mode)`` is called in increasing cost order and never after the first passing
    mode.  Returns the decision and the margins that were evaluated.
    """
    if threshold <= 0:
        raise ValueError("the margin threshold must be positive")
    evaluated: dict[int, float] = {}
    for mode in mode_order(costs, ranks):
        evaluated[mode] = float(margin_of(mode))
        if evaluated[mode] >= threshold:
            return mode, evaluated
    return None, evaluated


@dataclass
class RoutingRecord:
    """Outcome of Algorithm S1 for one candidate pair."""

    index: int
    status: str = UNPROCESSED
    mode: int | None = None
    margins: dict[int, float] = field(default_factory=dict)
    costs: tuple[int, ...] = ()

    @property
    def tokens(self) -> int:
        """Visual tokens ``c_i(a*)`` of both branches at the assigned mode."""
        return 0 if self.mode is None else self.costs[self.mode]


def routing_summary(records: Sequence[RoutingRecord], top_mode: int) -> dict[str, float]:
    """Counts, share assigned to the highest-detail mode and token totals of a routing run."""
    retained = [r for r in records if r.status == RETAINED]
    processed = [r for r in records if r.status in (RETAINED, EXCLUDED, UNRETAINED)]
    tokens = sum(r.tokens for r in retained)
    same_pair_top = sum(r.costs[top_mode] for r in retained)
    return {
        "candidates": len(records),
        "processed": len(processed),
        "retained": len(retained),
        "excluded": sum(r.status == EXCLUDED for r in records),
        "unretained": sum(r.status == UNRETAINED for r in records),
        "unprocessed": sum(r.status == UNPROCESSED for r in records),
        "retained_share": len(retained) / len(processed) if processed else 0.0,
        "top_mode_share": (
            sum(r.mode == top_mode for r in retained) / len(retained) if retained else 0.0
        ),
        "visual_tokens": tokens,
        "same_pair_top_tokens": same_pair_top,
        "token_saving": 1.0 - tokens / same_pair_top if same_pair_top else 0.0,
    }


def permute_assignments_within_strata(
    modes: Sequence[int], costs: Sequence[tuple[int, ...]], seed: int
) -> list[int]:
    """Random-allocation control of Appendix E.3.

    Permutes the mode assignments among pairs with identical per-mode token costs.  The
    number of pairs per mode, the cumulative visual tokens and the pair identities are
    preserved exactly; strata with a single assignment value are unchanged.
    """
    if len(modes) != len(costs):
        raise ValueError("one cost tuple per assignment is required")
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 0x5712A7A])))
    strata: dict[tuple[int, ...], list[int]] = {}
    for position, cost in enumerate(costs):
        strata.setdefault(tuple(cost), []).append(position)
    out = list(modes)
    for key in sorted(strata):
        members = strata[key]
        order = generator.permutation(len(members))
        values = [modes[members[j]] for j in order]
        for position, value in zip(members, values, strict=True):
            out[position] = value
    return out
