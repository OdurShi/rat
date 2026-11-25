"""Cached visual interfaces.

The visual encoder and connector are frozen, so the projected tokens of an image in a given
input mode are computed once and reused for reader fitting, routing and student updates
(Appendix D.2: "feature caching is available to all methods and charged at first
construction").  A :class:`FeatureStore` holds the tokens of one pool in one mode as a
half-precision memory map together with the device time measured when each image was
encoded; a :class:`PoolData` joins the stores of all modes with the manifest of the pool.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .backbones.base import Backbone, Example
from .data.splits import pair_from_record
from .interface.geometry import InputMode, TokenGrid


class FeatureStore:
    """Projected visual tokens of the images of one pool in one input mode."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / "meta.json").read_text())
        self.offsets = np.load(self.directory / "offsets.npy")
        self.seconds = np.load(self.directory / "seconds.npy")
        self.dim = int(self.meta["dim"])
        self._tokens = np.memmap(
            self.directory / "tokens.f16",
            dtype=np.float16,
            mode="r",
            shape=(int(self.offsets[-1]), self.dim),
        )

    def __len__(self) -> int:
        """Number of pairs."""
        return (len(self.offsets) - 1) // 2

    def count(self, pair: int, branch: int) -> int:
        row = 2 * pair + branch
        return int(self.offsets[row + 1] - self.offsets[row])

    def tokens(self, pair: int, branch: int) -> torch.Tensor:
        """Tokens ``K x d_v`` of one image as a single-precision tensor."""
        row = 2 * pair + branch
        block = self._tokens[self.offsets[row] : self.offsets[row + 1]]
        return torch.from_numpy(np.array(block, dtype=np.float32))

    def pair_seconds(self, pair: int) -> float:
        """Device time measured when both images of the pair were encoded."""
        return float(self.seconds[2 * pair] + self.seconds[2 * pair + 1])

    @staticmethod
    def exists(directory: str | Path) -> bool:
        return (Path(directory) / "meta.json").exists()

    @classmethod
    def build(
        cls,
        directory: str | Path,
        backbone: Backbone,
        records: Sequence[dict[str, Any]],
        mode: InputMode,
        detail: InputMode | None = None,
        batch_pairs: int = 8,
        progress: Callable[[int, int], None] | None = None,
    ) -> FeatureStore:
        """Encode every image of ``records`` in ``mode`` and write the store.

        The measured time covers preprocessing, the visual encoder and the connector; CPU
        rendering of the scenes is outside the adaptation budget and is not timed.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        counts = []
        for record in records:
            count = backbone.token_count((record["width"], record["height"]), mode)
            counts.extend((count, count))
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        tokens = np.memmap(
            directory / "tokens.f16",
            dtype=np.float16,
            mode="w+",
            shape=(int(offsets[-1]), backbone.visual_dim),
        )
        seconds = np.zeros(len(counts), dtype=np.float64)
        for start in range(0, len(records), batch_pairs):
            chunk = records[start : start + batch_pairs]
            images = [image for record in chunk for image in pair_from_record(record).images()]
            tic = time.perf_counter()
            features = backbone.encode_images(images, mode, detail)
            elapsed = time.perf_counter() - tic
            total = sum(f.shape[0] for f in features)
            for k, feature in enumerate(features):
                row = 2 * start + k
                if feature.shape[0] != counts[row]:
                    raise RuntimeError("the encoder returned an unexpected number of tokens")
                tokens[offsets[row] : offsets[row + 1]] = feature.numpy().astype(np.float16)
                # A batch's time is attributed to its images in proportion to their tokens.
                seconds[row] = elapsed * feature.shape[0] / total
            if progress is not None:
                progress(min(start + batch_pairs, len(records)), len(records))
        tokens.flush()
        del tokens
        np.save(directory / "offsets.npy", offsets)
        np.save(directory / "seconds.npy", seconds)
        meta = {
            "backbone": backbone.name,
            "mode": mode.name,
            "detail": None if detail is None else detail.name,
            "dim": backbone.visual_dim,
            "pairs": len(records),
            "first_pair": records[0]["pair_id"],
            "last_pair": records[-1]["pair_id"],
            "tokens": int(offsets[-1]),
            "seconds": float(seconds.sum()),
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))
        return cls(directory)


class PoolData:
    """A pool of pairs with its cached interfaces and text-side quantities.

    ``stores[a]`` is the store of mode ``backbone.modes[a]``; a missing store (``None``)
    means the pool is never used in that mode (for example evaluation pools, which only
    need ``H``).
    """

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        backbone: Backbone,
        stores: Sequence[FeatureStore | None],
        grid_modes: Sequence[InputMode] | None = None,
    ) -> None:
        self.records = list(records)
        self.backbone = backbone
        self.stores = list(stores)
        self.modes = list(grid_modes) if grid_modes is not None else list(backbone.modes)
        if len(self.stores) != len(self.modes):
            raise ValueError("one store (or None) per mode is required")
        for store in self.stores:
            if store is not None and len(store) != len(self.records):
                raise ValueError("a feature store does not match the number of pairs")
        self.sizes = [(r["width"], r["height"]) for r in self.records]
        self.questions = [r["question"] for r in self.records]
        self.answers = [tuple(r["answers"]) for r in self.records]
        self.candidates: torch.Tensor | None = None
        self._grids: dict[tuple, TokenGrid] = {}
        self._text: dict[int, tuple[int, int]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def grid(self, pair: int, mode: int) -> TokenGrid:
        key = (self.sizes[pair], mode)
        if key not in self._grids:
            self._grids[key] = self.backbone.token_grid(self.sizes[pair], self.modes[mode])
        return self._grids[key]

    def costs(self, pair: int) -> tuple[int, ...]:
        """``c_i(a)`` for every mode: visual tokens of both branches."""
        return tuple(2 * self.grid(pair, mode).count for mode in range(len(self.modes)))

    def fetch(self, pair: int, mode: int) -> tuple[list[torch.Tensor], list[TokenGrid]]:
        """Reader inputs of a pair in one mode: tokens and grids of both branches."""
        store = self._store(mode)
        grid = self.grid(pair, mode)
        return [store.tokens(pair, 0), store.tokens(pair, 1)], [grid, grid]

    def example(self, pair: int, branch: int, mode: int, with_answer: bool = True) -> Example:
        return Example(
            tokens=self._store(mode).tokens(pair, branch),
            grid=self.grid(pair, mode),
            question=self.questions[pair],
            answer=self.answers[pair][branch] if with_answer else None,
        )

    def extraction_seconds(self, pair: int, mode: int) -> float:
        return self._store(mode).pair_seconds(pair)

    def lengths(self, pair: int, mode: int) -> tuple[int, int]:
        """Teacher-forced sequence lengths of both branches at ``mode``."""
        if pair not in self._text:
            self._text[pair] = tuple(
                self.backbone.sequence_length(0, self.questions[pair], answer)
                for answer in self.answers[pair]
            )
        tokens = self.grid(pair, mode).count
        return tokens + self._text[pair][0], tokens + self._text[pair][1]

    def embed_candidates(self) -> float:
        """Compute ``t^j_i`` for both candidates of every pair; returns the seconds spent."""
        tic = time.perf_counter()
        questions = [q for q in self.questions for _ in range(2)]
        answers = [a for pair in self.answers for a in pair]
        embeddings = self.backbone.candidate_embeddings(questions, answers)
        self.candidates = embeddings.view(len(self.records), 2, -1)
        return time.perf_counter() - tic

    def _store(self, mode: int) -> FeatureStore:
        store = self.stores[mode]
        if store is None:
            raise KeyError(f"pool has no cached interface for mode {self.modes[mode]}")
        return store


def open_pool(
    backbone: Backbone,
    records: Sequence[dict[str, Any]],
    cache_dir: str | Path,
    name: str,
    modes: Sequence[InputMode] | None = None,
    build: bool = True,
    progress: Callable[[str, int, int], None] | None = None,
) -> PoolData:
    """Open (and if necessary build) the stores of a pool for the given modes."""
    wanted = list(backbone.modes) if modes is None else list(modes)
    stores: list[FeatureStore | None] = []
    for mode in backbone.modes:
        if mode not in wanted:
            stores.append(None)
            continue
        directory = Path(cache_dir) / f"{name}-{len(records)}-{mode.name}"
        if FeatureStore.exists(directory):
            stores.append(FeatureStore(directory))
        elif build:
            callback = (
                (lambda done, total, m=mode: progress(f"{name}/{m.name}", done, total))
                if progress
                else None
            )
            stores.append(FeatureStore.build(directory, backbone, records, mode, progress=callback))
        else:
            raise FileNotFoundError(directory)
    return PoolData(records, backbone, stores)
