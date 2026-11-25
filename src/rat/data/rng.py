"""A small deterministic random stream.

The scene generator must give the same scenes on every platform and library version, because
a scene is identified by its integer seed alone.  NumPy guarantees the raw output of the
``PCG64`` bit generator and of ``SeedSequence``, but not of the distribution methods built on
top of them, so every draw here is derived from raw 64-bit words with explicit arithmetic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import numpy as np

T = TypeVar("T")

#: Mixed into every stream so that the scenes of this generator are its own.
GENERATOR_SALT = 0x52A7_2027


class Rng:
    """Deterministic stream keyed by a tuple of non-negative integers."""

    def __init__(self, *key: int) -> None:
        if any(k < 0 for k in key):
            raise ValueError("stream keys must be non-negative")
        self._bits = np.random.PCG64(np.random.SeedSequence([GENERATOR_SALT, *key]))

    def u64(self) -> int:
        return int(self._bits.random_raw())

    def uniform(self) -> float:
        """Uniform float in ``[0, 1)`` with 53 random bits."""
        return (self.u64() >> 11) / float(1 << 53)

    def randint(self, low: int, high: int) -> int:
        """Uniform integer in ``[low, high]`` (both inclusive)."""
        if high < low:
            raise ValueError(f"empty range [{low}, {high}]")
        span = high - low + 1
        # Rejection sampling removes the modulo bias.
        limit = (1 << 64) - ((1 << 64) % span)
        while True:
            word = self.u64()
            if word < limit:
                return low + word % span

    def chance(self, probability: float) -> bool:
        return self.uniform() < probability

    def choice(self, items: Sequence[T]) -> T:
        if not items:
            raise ValueError("cannot choose from an empty sequence")
        return items[self.randint(0, len(items) - 1)]

    def weighted(self, items: Sequence[T], weights: Sequence[float]) -> T:
        if len(items) != len(weights) or not items:
            raise ValueError("items and weights must be non-empty and of equal length")
        total = float(sum(weights))
        threshold = self.uniform() * total
        running = 0.0
        for item, weight in zip(items, weights, strict=True):
            running += weight
            if threshold < running:
                return item
        return items[-1]

    def shuffled(self, items: Sequence[T]) -> list[T]:
        """Fisher-Yates shuffle returning a new list."""
        out = list(items)
        for i in range(len(out) - 1, 0, -1):
            j = self.randint(0, i)
            out[i], out[j] = out[j], out[i]
        return out

    def sample(self, items: Sequence[T], count: int) -> list[T]:
        if count > len(items):
            raise ValueError("sample larger than population")
        return self.shuffled(items)[:count]


def permutation(size: int, *key: int) -> list[int]:
    """Deterministic permutation of ``range(size)`` for the stream ``key``."""
    return Rng(*key).shuffled(range(size))
