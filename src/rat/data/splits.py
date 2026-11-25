"""Data partitions and generation manifests (Appendix B.1, E.1).

Every scene seed belongs to exactly one pool, and all edits and renderings of a seed inherit
that assignment.  The pools and their sizes are those of the paper:

=================  ======  =====================================================
pool               pairs   use
=================  ======  =====================================================
``reader_fit``     10,000  fitting the shared reader (``F``)
``tune``            5,000  threshold trials (``D_tune``)
``dev``             2,000  development evaluation (``D_dev``)
``train``          50,000  student candidate pool (``D``)
``test``            4,800  final paired evaluation, half with held-out renderings
``prelim_train``    2,000  preliminary study, training
``prelim_test``     4,800  preliminary study, evaluation
=================  ======  =====================================================

A pool is the concatenation of strata (task family x rendering family) taken in round-robin
order, so every prefix of a manifest is balanced across task families up to rounding.  Each
stratum owns a contiguous block of candidate seeds; rejected candidates are skipped.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from .pairs import Pair, build_pair
from .scenes import FAMILIES

#: Candidate seeds reserved for one stratum and for one pool.
STRATUM_BLOCK = 1_000_000
POOL_BLOCK = 10_000_000


@dataclass(frozen=True)
class PoolSpec:
    name: str
    index: int
    size: int
    #: Whether half of the pairs use the held-out rendering family.
    shifted_half: bool = False

    @property
    def strata(self) -> tuple[tuple[str, str], ...]:
        renderings = ("familiar", "heldout") if self.shifted_half else ("familiar",)
        return tuple((family, rendering) for rendering in renderings for family in FAMILIES)

    def first_seed(self, stratum: int) -> int:
        return self.index * POOL_BLOCK + stratum * STRATUM_BLOCK

    def quota(self, stratum: int, size: int | None = None) -> int:
        """Number of pairs of ``stratum`` in a pool prefix of ``size`` pairs."""
        size = self.size if size is None else size
        count = len(self.strata)
        return size // count + (1 if stratum < size % count else 0)


POOLS: dict[str, PoolSpec] = {
    spec.name: spec
    for spec in (
        PoolSpec("reader_fit", 1, 10_000),
        PoolSpec("tune", 2, 5_000),
        PoolSpec("dev", 3, 2_000),
        PoolSpec("train", 4, 50_000),
        PoolSpec("test", 5, 4_800, shifted_half=True),
        PoolSpec("prelim_train", 6, 2_000),
        PoolSpec("prelim_test", 7, 4_800),
    )
}


def pool_of_seed(seed: int) -> str:
    """Name of the pool that owns a scene seed."""
    for spec in POOLS.values():
        if spec.index == seed // POOL_BLOCK:
            return spec.name
    raise KeyError(f"seed {seed} does not belong to any pool")


def _build_record(task: tuple[str, int, str]) -> dict[str, Any] | None:
    pair = build_pair(*task)
    return None if pair is None else pair.record()


def _stratum_records(
    spec: PoolSpec, stratum: int, quota: int, pool, chunk: int
) -> list[dict[str, Any]]:
    family, rendering = spec.strata[stratum]
    records: list[dict[str, Any]] = []
    seed = spec.first_seed(stratum)
    limit = seed + STRATUM_BLOCK
    while len(records) < quota:
        if seed >= limit:
            raise RuntimeError(f"stratum {family}/{rendering} of {spec.name} ran out of seeds")
        tasks = [(family, s, rendering) for s in range(seed, min(seed + chunk, limit))]
        seed += len(tasks)
        results = (
            pool.map(_build_record, tasks, chunksize=32) if pool else map(_build_record, tasks)
        )
        records.extend(r for r in results if r is not None)
    return records[:quota]


def generate_pool(name: str, size: int | None = None, workers: int = 1) -> list[dict[str, Any]]:
    """Generate the manifest of a pool (or of its first ``size`` pairs).

    The result does not depend on ``workers``.  Records carry ``pair_id`` and ``pool`` in
    addition to the fields of :meth:`rat.data.pairs.Pair.record`.
    """
    spec = POOLS[name]
    size = spec.size if size is None else size
    if not 0 < size <= spec.size:
        raise ValueError(f"pool {name} holds {spec.size} pairs, requested {size}")
    pool = get_context("spawn").Pool(workers) if workers > 1 else None
    try:
        per_stratum = [
            _stratum_records(spec, k, spec.quota(k, size), pool, chunk=max(64, 2 * workers * 32))
            for k in range(len(spec.strata))
        ]
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    records = []
    for position in range(size):
        stratum, offset = position % len(spec.strata), position // len(spec.strata)
        record = dict(per_stratum[stratum][offset])
        records.append({"pair_id": f"{name}-{position:05d}", "pool": name, **record})
    return records


def write_manifest(records: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Write records as JSON lines; a ``.gz`` suffix selects reproducible gzip output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    if path.suffix == ".gz":
        with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(lines.encode("utf-8"))
    else:
        path.write_text(lines, encoding="utf-8")
    return path


def read_manifest(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Read a manifest written by :func:`write_manifest`."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    records = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if limit is not None and len(records) >= limit:
                break
            if line.strip():
                records.append(json.loads(line))
    return records


def pair_from_record(record: dict[str, Any], verify: bool = True) -> Pair:
    """Regenerate the pair of a manifest record from its seed.

    With ``verify`` the regenerated question, answers, canvas size and edited area must
    equal the stored ones; a mismatch means the generator differs from the one that wrote
    the manifest.
    """
    pair = build_pair(record["family"], record["seed"], record["rendering"])
    if pair is None:
        raise ValueError(f"record {record.get('pair_id')} does not regenerate a valid pair")
    if verify:
        fresh = pair.record()
        keys = ("question", "answers", "width", "height", "task", "layout", "font", "font_px")
        differing = [key for key in keys if fresh[key] != record[key]]
        if differing:
            raise ValueError(f"record {record.get('pair_id')} differs in {differing}")
    return pair


def iter_pairs(records: Iterable[dict[str, Any]], verify: bool = True) -> Iterator[Pair]:
    for record in records:
        yield pair_from_record(record, verify)


def default_manifest_dir() -> Path:
    """``data/manifests`` of a source checkout (the repository ships the manifests there)."""
    return Path(__file__).resolve().parents[3] / "data" / "manifests"


def load_pool(
    name: str, size: int | None = None, directory: str | Path | None = None
) -> list[dict[str, Any]]:
    """Load the shipped manifest of a pool, or generate it if the file is absent."""
    spec = POOLS[name]
    path = Path(directory) if directory else default_manifest_dir()
    file = path / f"{name}.jsonl.gz"
    if file.exists():
        records = read_manifest(file, limit=size)
        if size is None and len(records) != spec.size:
            raise ValueError(f"{file} holds {len(records)} records, expected {spec.size}")
        return records
    return generate_pool(name, size)
