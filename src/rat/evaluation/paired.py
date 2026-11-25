"""Paired-test metrics (Appendix A.3, D.1, E.1).

Generated pairs use exact matching after stripping leading and trailing whitespace from the
prediction and from the reference.  A pair is correct only if both branch answers are
correct (Eq. (S2), (S7)); single-image accuracy (Eq. (S9)) and the answer-collapse rate
(Eq. (S10)) are reported in addition.  Group results cover the familiar and held-out
rendering halves and the six task-by-rendering groups, whose minimum is the worst-group
paired accuracy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def normalize(text: str) -> str:
    """The whitespace rule applied to predictions and references."""
    return text.strip()


def branch_correct(
    predictions: Sequence[tuple[str, str]], answers: Sequence[tuple[str, str]]
) -> np.ndarray:
    """``e^b_i = 1[y_hat^b_i = y^b_i]`` as a boolean array ``N x 2``."""
    if len(predictions) != len(answers):
        raise ValueError("one prediction pair per reference pair is required")
    return np.array(
        [
            [normalize(p[0]) == normalize(a[0]), normalize(p[1]) == normalize(a[1])]
            for p, a in zip(predictions, answers, strict=True)
        ],
        dtype=bool,
    ).reshape(-1, 2)


def pair_correct(
    predictions: Sequence[tuple[str, str]], answers: Sequence[tuple[str, str]]
) -> np.ndarray:
    """``C_i``: both branch answers match (Eq. (S2))."""
    return branch_correct(predictions, answers).all(axis=1)


def collapsed(predictions: Sequence[tuple[str, str]]) -> np.ndarray:
    """Pairs with identical outputs on both branches, regardless of correctness (Eq. (S10))."""
    return np.array([normalize(p[0]) == normalize(p[1]) for p in predictions], dtype=bool)


def paired_metrics(
    predictions: Sequence[tuple[str, str]],
    answers: Sequence[tuple[str, str]],
    records: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Pair, Single and Collapse (in percent) and, with ``records``, the group results.

    ``records`` are the manifest entries of the pairs (fields ``family`` and ``rendering``).
    """
    correct = branch_correct(predictions, answers)
    pair = correct.all(axis=1)
    out: dict[str, Any] = {
        "pairs": int(len(pair)),
        "pair": 100.0 * float(pair.mean()) if len(pair) else 0.0,
        "single": 100.0 * float(correct.mean()) if len(pair) else 0.0,
        "collapse": 100.0 * float(collapsed(predictions).mean()) if len(pair) else 0.0,
    }
    if records is not None:
        families = np.array([r["family"] for r in records])
        renderings = np.array([r["rendering"] for r in records])
        groups: dict[str, float] = {}
        for rendering in sorted(set(renderings)):
            mask = renderings == rendering
            out[rendering] = 100.0 * float(pair[mask].mean())
            for family in sorted(set(families)):
                member = mask & (families == family)
                if member.any():
                    groups[f"{family}/{rendering}"] = 100.0 * float(pair[member].mean())
        for family in sorted(set(families)):
            out[family] = 100.0 * float(pair[families == family].mean())
        out["groups"] = groups
        out["worst"] = min(groups.values())
    return out


def accuracy_difference(
    first: Sequence[np.ndarray], second: Sequence[np.ndarray], members: np.ndarray | None = None
) -> float:
    """Eq. (S3): mean over runs of the paired-accuracy difference in percentage points.

    ``first`` and ``second`` hold one boolean pair-correctness array per run; differences are
    computed within each run and averaged across runs.  ``members`` restricts the pairs to a
    group ``G`` with equal weight for every pair.
    """
    if len(first) != len(second) or not first:
        raise ValueError("both conditions need the same positive number of runs")
    differences = []
    for a, b in zip(first, second, strict=True):
        if members is not None:
            a, b = a[members], b[members]
        differences.append(float(a.mean()) - float(b.mean()))
    return 100.0 * float(np.mean(differences))


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (zero for a single value) over seeds."""
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0
