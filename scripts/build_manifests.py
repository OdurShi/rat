"""Generate the manifests of all data pools into ``data/manifests``.

Usage::

    python scripts/build_manifests.py [--pools train test ...] [--workers 8] [--output data/manifests]

A manifest lists, for every pair of a pool, the scene seed that regenerates it together with
its question, both answers, canvas size, rendering choices and edited area.  The pool sizes
are those of Appendix E.1 of the paper.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

from rat.data.splits import POOLS, generate_pool, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pools", nargs="*", default=sorted(POOLS, key=lambda n: POOLS[n].index))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("data/manifests"))
    args = parser.parse_args()
    for name in args.pools:
        start = time.time()
        records = generate_pool(name, workers=args.workers)
        path = write_manifest(records, args.output / f"{name}.jsonl.gz")
        families = Counter(record["family"] for record in records)
        print(
            f"{name}: {len(records)} pairs {dict(families)} -> {path} "
            f"({path.stat().st_size / 1e6:.2f} MB, {time.time() - start:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
