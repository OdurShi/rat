"""Pairwise margins (Section 4.3, Appendix C.1).

For a pair in one input mode the reader gives a 2 x 2 score matrix ``S`` (image branch x
answer candidate) with the correct answers on the diagonal.  The correct-answer margins are
``l_0 = S00 - S01`` and ``l_1 = S11 - S10``; the gate uses their minimum (Eq. (3)).
"""

from __future__ import annotations

import math

import torch

#: Thresholds of the paper: the weaker branch must prefer its correct answer with
#: probability 2/3, 4/5 or 9/10 (``sigma(log k) = k / (k + 1)``).
PAPER_THRESHOLDS: tuple[float, ...] = (math.log(2.0), math.log(4.0), math.log(9.0))


def branch_margins(scores: torch.Tensor) -> torch.Tensor:
    """``(l_0, l_1)`` for score matrices of shape ``... x 2 x 2``; output ``... x 2``."""
    if scores.shape[-2:] != (2, 2):
        raise ValueError("score matrices must have shape (..., 2, 2)")
    correct = torch.diagonal(scores, dim1=-2, dim2=-1)
    wrong = torch.diagonal(scores.flip(-1), dim1=-2, dim2=-1)
    return correct - wrong


def min_margin(scores: torch.Tensor) -> torch.Tensor:
    """Eq. (3): ``m = min_b l_b``, the margin of the weaker branch."""
    return branch_margins(scores).min(dim=-1).values


def mean_margin(scores: torch.Tensor) -> torch.Tensor:
    """Mean branch margin ``delta`` (score variant of Appendix E.3)."""
    return branch_margins(scores).mean(dim=-1)


def decompose(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eq. (S6): mean contrast ``delta``, branch imbalance ``beta`` and ``m = delta - |beta|``.

    ``delta`` is the part of the candidate preference that reverses with the image branch,
    ``beta`` the preference shared by both images.  The gate ``m >= gamma`` is equivalent to
    ``delta >= |beta| + gamma``.
    """
    margins = branch_margins(scores)
    delta = 0.5 * (margins[..., 0] + margins[..., 1])
    beta = 0.5 * (margins[..., 0] - margins[..., 1])
    return delta, beta, delta - beta.abs()


def passes(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """Whether both branches clear the threshold, ``m >= gamma`` with ``gamma > 0``."""
    if threshold <= 0:
        raise ValueError("the margin threshold must be positive")
    return min_margin(scores) >= threshold


def threshold_from_probability(probability: float) -> float:
    """Margin threshold at which the weaker branch has reader probability ``probability``.

    The reader's binary softmax probability of the correct answer on a branch is
    ``sigma(l_b)``, so ``m >= log(p / (1 - p))`` requires ``sigma(m) >= p``.
    """
    if not 0.5 < probability < 1.0:
        raise ValueError("the probability must lie in (0.5, 1) for a positive threshold")
    return math.log(probability / (1.0 - probability))


def probability_from_threshold(threshold: float) -> float:
    """Inverse of :func:`threshold_from_probability`."""
    return 1.0 / (1.0 + math.exp(-threshold))
