"""Uncertainty of paired comparisons on a shared test set.

The paper reports means and standard deviations over three seeds.  Because all methods are
evaluated on the same pairs, the per-pair outcomes also give a paired bootstrap interval and
an exact McNemar test for the difference between two methods; these complement the seed
statistics when the number of seeds is small.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def paired_bootstrap(
    first: Sequence[np.ndarray],
    second: Sequence[np.ndarray],
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap interval of the paired-accuracy difference (percentage points).

    ``first`` and ``second`` hold one boolean pair-correctness array per seed; test pairs
    are resampled with replacement and the same resample is applied to every seed and both
    methods, so each test pair moves as a unit.
    """
    a = np.mean(np.stack([np.asarray(x, dtype=float) for x in first]), axis=0)
    b = np.mean(np.stack([np.asarray(x, dtype=float) for x in second]), axis=0)
    difference = a - b
    generator = np.random.Generator(np.random.PCG64(seed))
    count = len(difference)
    draws = np.empty(samples)
    for start in range(0, samples, 1000):
        index = generator.integers(0, count, size=(min(1000, samples - start), count))
        draws[start : start + index.shape[0]] = difference[index].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "difference": 100.0 * float(difference.mean()),
        "low": 100.0 * float(np.quantile(draws, tail)),
        "high": 100.0 * float(np.quantile(draws, 1.0 - tail)),
        "confidence": confidence,
    }


def mcnemar_exact(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """Exact two-sided McNemar test for two boolean outcome arrays on the same pairs."""
    first, second = np.asarray(first, dtype=bool), np.asarray(second, dtype=bool)
    only_first = int((first & ~second).sum())
    only_second = int((~first & second).sum())
    discordant = only_first + only_second
    if discordant == 0:
        return {"only_first": 0, "only_second": 0, "p_value": 1.0}
    smaller = min(only_first, only_second)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2.0**discordant
    return {"only_first": only_first, "only_second": only_second, "p_value": min(1.0, 2.0 * tail)}
