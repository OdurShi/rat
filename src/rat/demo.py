"""CPU quick start: reader fitting and minimum-margin routing on a simulated interface.

The demo uses real generated pairs (questions, answers, canvas sizes, edited areas) and the
real Qwen2.5-VL preprocessing geometry for their token counts, but replaces the visual
encoder and the language model by a small simulation so that it runs in seconds without
downloads: the token at the edited location carries the answer of its image with a strength
that grows with the projected size of the edit in the pixels of the input mode.  Small edits
are therefore readable only in the high-resolution mode, and very small ones in neither.

The simulation illustrates the mechanism; it is not a model of any particular backbone.
"""

from __future__ import annotations

import math
import time

import torch

from .data.pairs import edit_scale
from .data.splits import load_pool, pair_from_record
from .interface.geometry import TokenGrid, llava_grid, qwen_grid, tile_grid
from .margins import PAPER_THRESHOLDS, min_margin
from .reader import CandidateReader, ReaderFitConfig, fit_reader, score_pairs
from .routing import route

VISUAL_DIM = TEXT_DIM = 24
MODE_TOKENS = (256, 1024)
#: Projected edit scale (pixels of the mode) at which the evidence reaches 63% of its maximum.
LEGIBILITY_SCALE = 3.0
SIMULATED_TOKENS = 16


class SimulatedInterface:
    """Visual tokens and candidate embeddings for manifest records."""

    def __init__(self, records: list[dict], seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.records = records
        count = len(records)
        shared = torch.randn(count, 1, TEXT_DIM, generator=generator)
        self.candidates = shared + 0.5 * torch.randn(count, 2, TEXT_DIM, generator=generator)
        # A fixed map from answers to their visual trace; the reader has to recover it.
        trace = (
            torch.randn(TEXT_DIM, VISUAL_DIM, generator=torch.Generator().manual_seed(1234))
            / TEXT_DIM**0.5
        )
        self.grid = TokenGrid(((4, 4, -1.0, -1.0, 1.0, 1.0),), (112, 112))
        self.tokens: dict[tuple[int, int], list[torch.Tensor]] = {}
        self.scales = torch.zeros(count, len(MODE_TOKENS))
        self.costs = []
        for index, record in enumerate(records):
            native = (record["width"], record["height"])
            location = int(torch.randint(SIMULATED_TOKENS, (1,), generator=generator))
            background = 0.4 * torch.randn(SIMULATED_TOKENS, VISUAL_DIM, generator=generator)
            costs = []
            for mode, cap in enumerate(MODE_TOKENS):
                geometry = qwen_grid(native, cap)
                costs.append(2 * geometry.count)
                scale = edit_scale(record["edit_area"], native, geometry.resized)
                self.scales[index, mode] = scale
                strength = 6.0 * (1.0 - math.exp(-((scale / LEGIBILITY_SCALE) ** 2)))
                branches = []
                for branch in range(2):
                    tokens = background + 0.2 * torch.randn(
                        SIMULATED_TOKENS, VISUAL_DIM, generator=generator
                    )
                    tokens[location] += (
                        strength * (self.candidates[index, branch] - shared[index, 0]) @ trace
                    )
                    branches.append(tokens)
                self.tokens[index, mode] = branches
            self.costs.append(tuple(costs))

    def fetch(self, index: int, mode: int):
        return self.tokens[index, mode], [self.grid, self.grid]


def run(seed: int = 17, fit_pairs: int = 600, held_pairs: int = 600, log=print) -> dict:
    start = time.time()
    record = load_pool("train", 2)[1]
    pair = pair_from_record(record)
    log(
        f"Pair {record['pair_id']} ({pair.family}, {pair.width} x {pair.height} px, {pair.edit_area} edited pixels)"
    )
    log(f"  question: {pair.question}")
    log(f"  answers:  {pair.answers[0]!r} (image 0)  vs  {pair.answers[1]!r} (image 1)")
    log("Visual tokens per image (L / H):")
    log(f"  Qwen2.5-VL  {qwen_grid(pair.size, 256).count:>4} / {qwen_grid(pair.size, 1024).count}")
    log(f"  LLaVA-1.5   {llava_grid(pair.size, 'L').count:>4} / {llava_grid(pair.size, 'H').count}")
    log(
        f"  SmolVLM     {tile_grid(pair.size, 512, 512, 8).count:>4} / {tile_grid(pair.size, 1024, 512, 8).count}"
    )

    fit = SimulatedInterface(load_pool("reader_fit", fit_pairs), seed)
    held = SimulatedInterface(load_pool("tune", held_pairs), seed + 1)
    reader = CandidateReader(VISUAL_DIM, TEXT_DIM, seed=seed)
    config = ReaderFitConfig(epochs=20, batch_pairs=32, micro_pairs=32, learning_rate=3e-3)
    fit_log = fit_reader(reader, fit.fetch, fit.candidates, len(MODE_TOKENS), config, seed)
    log(
        f"\nReader fitted on {fit_pairs} pairs (simulated interface): "
        f"loss {fit_log.losses[0]:.3f} -> {sum(fit_log.losses[-5:]) / 5:.3f}"
    )
    margins = torch.stack(
        [
            min_margin(score_pairs(reader, held.fetch, held.candidates, range(held_pairs), mode))
            for mode in (0, 1)
        ],
        dim=1,
    )
    low_scale = held.scales[:, 0]
    small, large = low_scale < 2.0, low_scale > 7.0
    log(
        f"Held-out pairs: {held_pairs} ({int(small.sum())} with edit scale < 2 px at L, {int(large.sum())} above 7 px)"
    )
    log(
        "\n  gamma   retained   H share   tokens saved   small edits: skip / L / H    large edits: skip / L / H"
    )
    rows = []
    for threshold in PAPER_THRESHOLDS:
        decisions = [
            route(margins[i].tolist(), held.costs[i], threshold) for i in range(held_pairs)
        ]
        kept = [d for d in decisions if d is not None]
        tokens = sum(held.costs[i][d] for i, d in enumerate(decisions) if d is not None)
        all_high = sum(held.costs[i][1] for i, d in enumerate(decisions) if d is not None)

        def split(mask: torch.Tensor, decisions=decisions) -> str:
            chosen = [d for d, inside in zip(decisions, mask.tolist(), strict=True) if inside]
            parts = [
                sum(d is None for d in chosen),
                sum(d == 0 for d in chosen),
                sum(d == 1 for d in chosen),
            ]
            return " / ".join(f"{100 * p / max(1, len(chosen)):.0f}%" for p in parts)

        row = {
            "threshold": threshold,
            "retained_percent": 100 * len(kept) / held_pairs,
            "high_share_percent": 100 * sum(d == 1 for d in kept) / max(1, len(kept)),
            "token_saving_percent": 100 * (1 - tokens / all_high) if all_high else 0.0,
        }
        rows.append(row)
        log(
            f"  log {round(math.exp(threshold))}   {row['retained_percent']:6.1f}%   {row['high_share_percent']:6.1f}%"
            f"   {row['token_saving_percent']:10.1f}%   {split(small):>24}   {split(large):>24}"
        )
    log(
        f"\nDone in {time.time() - start:.0f}s.  Tokens are counted with the Qwen2.5-VL geometry of the real pairs."
    )
    return {"rows": rows, "first_loss": fit_log.losses[0], "final_loss": fit_log.losses[-1]}
