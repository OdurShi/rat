"""Visual contrast and edit scale of the preliminary study (Appendix A.2, A.4).

For a pair in input mode ``a`` the contrast is the average cosine dissimilarity between the
projected visual tokens of the two images at the same token position (Eq. (S1)).  The
projected edit scale is the square root of the edited-region area in the pixel units of the
low-resolution mode.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from ..backbones.base import Backbone
from ..data.pairs import edit_scale
from ..features import PoolData

#: Strict cut-offs of the small- and large-edit summaries (Table S1).
SMALL_EDIT = 2.0
LARGE_EDIT = 7.0


def pair_contrast(first: torch.Tensor, second: torch.Tensor) -> float:
    """Eq. (S1): ``D = mean_k (1 - cos(v0_k, v1_k))`` over the aligned valid tokens."""
    if first.shape != second.shape:
        raise ValueError("both branches of a pair share their token grid within a mode")
    cosine = torch.nn.functional.cosine_similarity(first.double(), second.double(), dim=-1)
    return float((1.0 - cosine).mean())


def pool_contrast(pool: PoolData, pairs: Sequence[int] | None = None) -> np.ndarray:
    """``D_i(a)`` for every pair and mode, shape ``N x M``."""
    pairs = range(len(pool)) if pairs is None else pairs
    out = np.zeros((len(pairs), len(pool.modes)))
    for row, pair in enumerate(pairs):
        for mode in range(len(pool.modes)):
            tokens, _grids = pool.fetch(pair, mode)
            out[row, mode] = pair_contrast(tokens[0], tokens[1])
    return out


def projected_edit_scales(backbone: Backbone, records: Sequence[dict[str, Any]]) -> np.ndarray:
    """``s_i``: edit scale of every pair in pixel units of the backbone's lowest mode."""
    low = backbone.modes[0]
    scales = []
    for record in records:
        native = (record["width"], record["height"])
        scales.append(
            edit_scale(record["edit_area"], native, backbone.token_grid(native, low).resized)
        )
    return np.asarray(scales)


def group_masks(
    scales: np.ndarray, small: float = SMALL_EDIT, large: float = LARGE_EDIT
) -> dict[str, np.ndarray]:
    """Small (``s < small``), large (``s > large``) and intermediate pairs."""
    return {
        "small": scales < small,
        "large": scales > large,
        "middle": (scales >= small) & (scales <= large),
    }


def median_ratio(contrast: np.ndarray, members: np.ndarray, low: int = 0, high: int = -1) -> float:
    """Median of the per-pair ratio ``D_i(H) / D_i(L)`` within a group."""
    ratios = contrast[members, high] / np.maximum(contrast[members, low], 1e-12)
    return float(np.median(ratios)) if members.any() else float("nan")


def quantile_bins(values: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Assign values to ``count`` quantile bins; returns bin indices and the bin edges."""
    edges = np.quantile(values, np.linspace(0.0, 1.0, count + 1))
    index = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, count - 1)
    return index, edges


def heatmap_cells(
    scales: np.ndarray,
    losses: np.ndarray,
    first: Sequence[np.ndarray],
    second: Sequence[np.ndarray],
    scale_bins: int = 8,
    loss_bins: int = 6,
) -> dict[str, np.ndarray]:
    """Cells of Fig. 2b: mean paired-accuracy difference by edit scale and initial loss.

    Columns group ``log10 s_i`` into quantile bins, rows the initial student's low-mode
    answer loss.  ``first`` and ``second`` hold one boolean pair-correctness array per run;
    each cell applies Eq. (S3) to its member pairs.  Returns the differences (rows x
    columns, in percentage points), the cell counts, the geometric midpoints of the scale
    bins and the arithmetic midpoints of the loss bins.
    """
    column, scale_edges = quantile_bins(np.log10(scales), scale_bins)
    row, loss_edges = quantile_bins(losses, loss_bins)
    difference = np.full((loss_bins, scale_bins), np.nan)
    counts = np.zeros((loss_bins, scale_bins), dtype=int)
    for r in range(loss_bins):
        for c in range(scale_bins):
            members = (row == r) & (column == c)
            counts[r, c] = int(members.sum())
            if counts[r, c]:
                runs = [
                    float(a[members].mean()) - float(b[members].mean())
                    for a, b in zip(first, second, strict=True)
                ]
                difference[r, c] = 100.0 * float(np.mean(runs))
    return {
        "difference": difference,
        "counts": counts,
        "scale_centres": 10 ** ((scale_edges[:-1] + scale_edges[1:]) / 2),
        "loss_centres": (loss_edges[:-1] + loss_edges[1:]) / 2,
    }
