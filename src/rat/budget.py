"""Compute accounting (Section 4.4, Appendix D.2, E.2).

All costs are device time in seconds, summed over participating devices.  Reader fitting,
threshold trials, feature extraction, routing and student updates share one budget ``B``:

    C = C0 + C_route + C_train,      execute readout  <=>  C + E_hat_i(a) + R <= B        (S8)

``C0`` is the preparation cost, ``R = T_hat(S)`` the reservation for one training pass over
the retained set, ``E_hat_i(a)`` the estimated cost of the next readout.  Estimates come
from a pilot (:class:`CostModel`); realised costs are recorded after every operation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

CATEGORIES = ("preparation", "routing", "training")


@dataclass
class CostLedger:
    """Running total ``C`` against the budget ``B``, split as in Table 5."""

    budget: float
    preparation: float = 0.0
    routing: float = 0.0
    training: float = 0.0
    events: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return self.preparation + self.routing + self.training

    @property
    def remaining(self) -> float:
        return self.budget - self.total

    def affordable(self, amount: float) -> bool:
        """Whether ``C + amount <= B`` (with a relative tolerance for rounding)."""
        return self.total + amount <= self.budget * (1.0 + 1e-12)

    def charge(self, category: str, amount: float, event: str | None = None) -> None:
        if category not in CATEGORIES:
            raise KeyError(f"unknown cost category {category!r}")
        if amount < 0:
            raise ValueError("costs are non-negative")
        setattr(self, category, getattr(self, category) + amount)
        if event is not None:
            self.events[event] = self.events.get(event, 0.0) + amount

    def summary(self, unit: float = 3600.0) -> dict[str, float]:
        """Costs divided by ``unit`` (default: hours)."""
        return {
            "budget": self.budget / unit,
            "preparation": self.preparation / unit,
            "routing": self.routing / unit,
            "training": self.training / unit,
            "total": self.total / unit,
        }


@dataclass
class CostModel:
    """Operation-cost estimates obtained from a pilot (Appendix D.2).

    Feature extraction and reader scoring of one image cost
    ``extract_fixed + extract_per_token * K``; one student forward/backward chunk costs
    ``train_fixed + train_per_token * (examples * padded length)`` and every optimiser
    update adds ``update_fixed``.  Greedy decoding of one example costs
    ``generate_fixed + generate_per_token * prompt length``.
    """

    extract_fixed: float = 0.0
    extract_per_token: float = 1.0
    train_fixed: float = 0.0
    train_per_token: float = 1.0
    update_fixed: float = 0.0
    generate_fixed: float = 0.0
    generate_per_token: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def extraction(self, token_counts: Sequence[int]) -> float:
        """Estimated cost of encoding images with the given token counts."""
        return float(sum(self.extract_fixed + self.extract_per_token * k for k in token_counts))

    def readout(self, pair_tokens: int) -> float:
        """``E_hat_i(a)``: both images of a pair with ``pair_tokens`` visual tokens in total."""
        return 2.0 * self.extract_fixed + self.extract_per_token * pair_tokens

    def chunk(self, lengths: Sequence[int]) -> float:
        """One forward/backward pass over examples padded to the longest sequence."""
        return self.train_fixed + self.train_per_token * len(lengths) * max(lengths)

    def generation(self, lengths: Sequence[int]) -> float:
        return float(sum(self.generate_fixed + self.generate_per_token * n for n in lengths))


class TrainingCostEstimator:
    """``T_hat(S)``: estimated cost of one pass over the retained set.

    The retained pairs are taken in the fixed training order, split into forward chunks of
    ``pairs_per_forward`` pairs (both branches of a pair stay together) and into optimiser
    updates of ``batch_pairs`` pairs.  The estimate uses the full sequence lengths (visual
    tokens plus text) and the padding inside each chunk.
    """

    def __init__(self, model: CostModel, batch_pairs: int, pairs_per_forward: int) -> None:
        if batch_pairs % pairs_per_forward:
            raise ValueError("batch_pairs must be a multiple of pairs_per_forward")
        self.model = model
        self.batch_pairs = batch_pairs
        self.pairs_per_forward = pairs_per_forward
        self._positions = np.zeros(0, dtype=np.int64)
        self._lengths = np.zeros((0, 2), dtype=np.int64)
        self._total = 0.0

    def __len__(self) -> int:
        return len(self._positions)

    @property
    def total(self) -> float:
        return self._total

    def _cost(self, lengths: np.ndarray) -> float:
        count = len(lengths)
        if count == 0:
            return 0.0
        size = self.pairs_per_forward
        pad = (-count) % size
        longest = np.concatenate([lengths.max(axis=1), np.zeros(pad, dtype=np.int64)])
        chunk_max = longest.reshape(-1, size).max(axis=1)
        members = np.minimum(size, count - size * np.arange(len(chunk_max)))
        chunks = len(chunk_max) * self.model.train_fixed + self.model.train_per_token * float(
            (2 * members * chunk_max).sum()
        )
        updates = -(-count // self.batch_pairs)
        return chunks + updates * self.model.update_fixed

    def with_pair(self, position: int, lengths: tuple[int, int]) -> float:
        """``T_hat(S + {i})`` for a pair at ``position`` of the training order (no mutation)."""
        at = int(np.searchsorted(self._positions, position))
        merged = np.insert(self._lengths, at, np.asarray(lengths, dtype=np.int64), axis=0)
        return self._cost(merged)

    def add(self, position: int, lengths: tuple[int, int]) -> float:
        """Insert a pair and return the new reservation ``R = T_hat(S)``."""
        at = int(np.searchsorted(self._positions, position))
        self._positions = np.insert(self._positions, at, position)
        self._lengths = np.insert(self._lengths, at, np.asarray(lengths, dtype=np.int64), axis=0)
        self._total = self._cost(self._lengths)
        return self._total

    def pass_cost(self, lengths: Sequence[tuple[int, int]]) -> float:
        """Estimated one-pass cost of pairs given in training order."""
        return self._cost(np.asarray(lengths, dtype=np.int64).reshape(-1, 2))
