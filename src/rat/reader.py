"""Shared candidate reader (Section 4.2, Appendix B.2, E.4).

For an image with ``K`` valid visual tokens the reader input is ``Z = [V, P]`` in
``R^{K x d}``: the projected tokens ``V`` of the frozen interface and normalised grid
coordinates ``P`` (Eq. (S4)), ``d = d_v + 2``.  For a question-conditioned candidate
embedding ``t`` (Eq. (S5)) a single attention readout computes

    alpha(Z, t) = softmax(Z W_A t / sqrt(d)),    r(Z, t) = alpha(Z, t)^T Z W_R t / sqrt(d)    (1)

with trainable ``W_A, W_R`` in ``R^{d x d_t}`` shared across input modes, image branches and
answer candidates.  The fitting loss is the binary cross-entropy of Eq. (2).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from .interface.geometry import TokenGrid


def grid_coordinates(grid: TokenGrid) -> torch.Tensor:
    """Normalised coordinates ``P`` of every token of a grid, in token order (Eq. (S4)).

    For a single ``h x w`` grid the token in row ``r`` and column ``c`` receives
    ``((2c + 1) / w - 1, (2r + 1) / h - 1)``.  Tiled interfaces apply the same rule inside
    the image region covered by each tile, so a coordinate is always the centre of the
    token's region in the image.
    """
    blocks = []
    for rows, cols, x0, y0, x1, y1 in grid.parts:
        c = (2 * torch.arange(cols, dtype=torch.float32) + 1) / (2 * cols)
        r = (2 * torch.arange(rows, dtype=torch.float32) + 1) / (2 * rows)
        x = x0 + c * (x1 - x0)
        y = y0 + r * (y1 - y0)
        blocks.append(torch.stack((x.repeat(rows), y.repeat_interleave(cols)), dim=-1))
    return torch.cat(blocks, dim=0)


def reader_input(tokens: torch.Tensor, grid: TokenGrid) -> torch.Tensor:
    """Concatenate projected tokens ``V`` (``K x d_v``) with their coordinates ``P``."""
    coordinates = grid_coordinates(grid).to(tokens.device)
    if coordinates.shape[0] != tokens.shape[0]:
        raise ValueError(f"grid has {coordinates.shape[0]} tokens, features have {tokens.shape[0]}")
    return torch.cat((tokens.float(), coordinates), dim=-1)


def pad_inputs(inputs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack variable-length ``Z`` matrices; returns ``(B x K_max x d, B x K_max mask)``."""
    longest = max(z.shape[0] for z in inputs)
    batch = inputs[0].new_zeros((len(inputs), longest, inputs[0].shape[1]))
    mask = torch.zeros((len(inputs), longest), dtype=torch.bool, device=inputs[0].device)
    for i, z in enumerate(inputs):
        batch[i, : z.shape[0]] = z
        mask[i, : z.shape[0]] = True
    return batch, mask


class CandidateReader(nn.Module):
    """Attention readout of Eq. (1) with parameters ``omega = (W_A, W_R)``."""

    def __init__(self, visual_dim: int, text_dim: int, gain: float = 0.01, seed: int | None = None):
        super().__init__()
        self.visual_dim, self.text_dim = visual_dim, text_dim
        self.dim = visual_dim + 2
        self.attention = nn.Parameter(torch.empty(self.dim, text_dim))  # W_A
        self.readout = nn.Parameter(torch.empty(self.dim, text_dim))  # W_R
        generator = None if seed is None else torch.Generator().manual_seed(seed)
        for weight in (self.attention, self.readout):
            nn.init.xavier_uniform_(weight, gain=gain, generator=generator)
        # Optional feature standardisation (identity by default: the paper keeps the native
        # scales of visual tokens and text embeddings).  Coordinates are never rescaled.
        self.register_buffer("input_shift", torch.zeros(self.dim))
        self.register_buffer("input_scale", torch.ones(self.dim))
        self.register_buffer("text_shift", torch.zeros(text_dim))
        self.register_buffer("text_scale", torch.ones(text_dim))

    def set_standardisation(
        self,
        visual_mean: torch.Tensor,
        visual_std: torch.Tensor,
        text_mean: torch.Tensor,
        text_std: torch.Tensor,
    ) -> None:
        """Standardise visual tokens and candidate embeddings per dimension before Eq. (1)."""
        device = self.attention.device
        self.input_shift[: self.visual_dim] = visual_mean.to(device)
        self.input_scale[: self.visual_dim] = visual_std.clamp_min(1e-6).to(device)
        self.text_shift.copy_(text_mean.to(device))
        self.text_scale.copy_(text_std.clamp_min(1e-6).to(device))

    def _prepare(
        self, inputs: torch.Tensor, candidates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = (inputs.float() - self.input_shift) / self.input_scale
        candidates = (candidates.float() - self.text_shift) / self.text_scale
        return inputs, candidates

    def attention_weights(
        self, inputs: torch.Tensor, mask: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        """``alpha`` for padded inputs ``B x K x d`` and candidates ``B x J x d_t``: ``B x J x K``.

        The softmax runs over the valid tokens only.
        """
        inputs, candidates = self._prepare(inputs, candidates)
        return self._attention(inputs, mask, candidates)

    def _attention(
        self, inputs: torch.Tensor, mask: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        logits = torch.einsum("bkt,bjt->bjk", inputs @ self.attention, candidates) / math.sqrt(
            self.dim
        )
        logits = logits.masked_fill(~mask[:, None, :], float("-inf"))
        return torch.softmax(logits, dim=-1)

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        """Scores ``r(Z_b, t_bj)`` of shape ``B x J``, computed in single precision."""
        inputs, candidates = self._prepare(inputs, candidates)
        alpha = self._attention(inputs, mask, candidates)
        values = torch.einsum("bkt,bjt->bjk", inputs @ self.readout, candidates) / math.sqrt(
            self.dim
        )
        return (alpha * values).sum(dim=-1)

    @torch.no_grad()
    def score_matrix(
        self, tokens: Sequence[torch.Tensor], grids: Sequence[TokenGrid], candidates: torch.Tensor
    ) -> torch.Tensor:
        """The 2 x 2 matrix ``S[b, j] = r(Z^b, t^j)`` of one pair in one mode.

        ``tokens`` and ``grids`` hold the two image branches, ``candidates`` the two
        candidate embeddings (``2 x d_t``).  The correct answers are on the diagonal.
        """
        device = self.attention.device
        inputs, mask = pad_inputs(
            [reader_input(v.to(device), g) for v, g in zip(tokens, grids, strict=True)]
        )
        both = candidates.to(device).float().unsqueeze(0).expand(len(tokens), -1, -1)
        return self(inputs, mask, both)


def reader_loss(scores: torch.Tensor) -> torch.Tensor:
    """Eq. (2) for score matrices of shape ``... x 2 x 2`` (branch x candidate).

    ``log(1 + exp(S[b, 1-b] - S[b, b]))`` averaged over all leading dimensions and both
    branches, i.e. the binary cross-entropy of the correct candidate.
    """
    correct = torch.diagonal(scores, dim1=-2, dim2=-1)
    wrong = torch.diagonal(scores.flip(-1), dim1=-2, dim2=-1)
    return nn.functional.softplus(wrong - correct).mean()


@dataclass
class ReaderFitConfig:
    """Fitting settings of Appendix E.4."""

    epochs: int = 10
    batch_pairs: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    init_gain: float = 0.01
    #: Pairs processed per forward pass; gradients are accumulated to the full batch.
    micro_pairs: int = 32
    #: Standardise visual tokens and candidate embeddings with statistics of the fitting
    #: pool.  ``False`` keeps the native feature scales, as in the paper.
    standardize: bool = False


@dataclass
class ReaderFitLog:
    losses: list[float] = field(default_factory=list)
    steps: int = 0


#: ``fetch(pair_index, mode_index)`` returns ``(tokens of both branches, grids of both branches)``.
FeatureFetcher = Callable[[int, int], tuple[Sequence[torch.Tensor], Sequence[TokenGrid]]]


def _pair_scores(
    reader: CandidateReader,
    fetch: FeatureFetcher,
    pairs: Sequence[int],
    mode: int,
    candidates: torch.Tensor,
) -> torch.Tensor:
    """Score matrices ``len(pairs) x 2 x 2`` for one mode."""
    device = reader.attention.device
    inputs, text = [], []
    for index in pairs:
        tokens, grids = fetch(index, mode)
        for branch in range(2):
            inputs.append(reader_input(tokens[branch].to(device), grids[branch]))
            text.append(candidates[index])
    padded, mask = pad_inputs(inputs)
    scores = reader(padded, mask, torch.stack(text).to(device))
    return scores.view(len(pairs), 2, 2)


def fit_reader(
    reader: CandidateReader,
    fetch: FeatureFetcher,
    candidates: torch.Tensor,
    num_modes: int,
    config: ReaderFitConfig | None = None,
    seed: int = 0,
    on_step: Callable[[int, float], None] | None = None,
) -> ReaderFitLog:
    """Minimise Eq. (2) over the reader-fitting pool.

    ``candidates`` has shape ``N x 2 x d_t`` (pair, candidate).  Every optimiser step uses
    ``batch_pairs`` pairs with both image branches and all ``num_modes`` input modes; the
    visual and text features are constants.  The reader of the final epoch is kept.
    """
    config = config or ReaderFitConfig()
    count = candidates.shape[0]
    if config.standardize:
        reader.set_standardisation(*feature_statistics(fetch, candidates, num_modes))
    optimizer = torch.optim.AdamW(
        reader.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    log = ReaderFitLog()
    reader.train()
    for _epoch in range(config.epochs):
        order = torch.randperm(count, generator=generator).tolist()
        for start in range(0, count, config.batch_pairs):
            batch = order[start : start + config.batch_pairs]
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for mode in range(num_modes):
                for offset in range(0, len(batch), config.micro_pairs):
                    chunk = batch[offset : offset + config.micro_pairs]
                    scores = _pair_scores(reader, fetch, chunk, mode, candidates)
                    # Equal weight for every pair, branch and mode of the batch.
                    loss = reader_loss(scores) * (len(chunk) / (len(batch) * num_modes))
                    loss.backward()
                    total += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(reader.parameters(), config.max_grad_norm)
            optimizer.step()
            log.losses.append(total)
            log.steps += 1
            if on_step is not None:
                on_step(log.steps, total)
    reader.eval()
    return log


@torch.no_grad()
def feature_statistics(
    fetch: FeatureFetcher, candidates: torch.Tensor, num_modes: int, max_pairs: int = 256
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-dimension mean and standard deviation of visual tokens and candidate embeddings.

    Visual statistics pool the tokens of both branches and all modes of evenly spaced pairs
    of the fitting pool; text statistics use every candidate embedding.
    """
    count = candidates.shape[0]
    step = max(1, count // max_pairs)
    total = squares = None
    tokens_seen = 0
    for index in range(0, count, step):
        for mode in range(num_modes):
            for tokens in fetch(index, mode)[0]:
                tokens = tokens.double()
                total = tokens.sum(0) if total is None else total + tokens.sum(0)
                squares = (tokens**2).sum(0) if squares is None else squares + (tokens**2).sum(0)
                tokens_seen += tokens.shape[0]
    mean = total / tokens_seen
    std = (squares / tokens_seen - mean**2).clamp_min(0).sqrt()
    text = candidates.reshape(-1, candidates.shape[-1]).double()
    return mean.float(), std.float(), text.mean(0).float(), text.std(0).float()


@torch.no_grad()
def score_pairs(
    reader: CandidateReader,
    fetch: FeatureFetcher,
    candidates: torch.Tensor,
    pairs: Iterable[int],
    mode: int,
    batch_pairs: int = 64,
) -> torch.Tensor:
    """Score matrices ``len(pairs) x 2 x 2`` of ``pairs`` in one mode with a fixed reader."""
    reader.eval()
    pairs = list(pairs)
    out = [
        _pair_scores(reader, fetch, pairs[i : i + batch_pairs], mode, candidates).cpu()
        for i in range(0, len(pairs), batch_pairs)
    ]
    return torch.cat(out) if out else torch.zeros((0, 2, 2))
