"""Encode the images of the configured pools with the frozen visual interface.

Usage::

    python scripts/cache_features.py --config configs/laptop/smolvlm_256m.yaml [--pools train test]

Training-side pools are cached in every input mode, evaluation pools in the evaluation mode
only.  The stores are written to ``cache_dir`` and record the device time of every image.
"""

from __future__ import annotations

import argparse
import time

from rat.config import build_backbone, load_config
from rat.data.splits import load_pool
from rat.features import open_pool

#: Pools routed or trained in both modes; the remaining pools are only evaluated at H.
MULTI_MODE = ("reader_fit", "tune", "train", "prelim_train")
ALL = ("reader_fit", "tune", "dev", "train", "test", "prelim_train", "prelim_test")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--pools", nargs="*", default=list(ALL))
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    backbone = build_backbone(config)
    for name in args.pools:
        records = load_pool(name, config["data"][name], config["data"]["manifests"])
        modes = backbone.modes if name in MULTI_MODE else (backbone.eval_mode,)
        start = time.time()
        last = [start]

        def progress(label: str, done: int, total: int) -> None:
            if time.time() - last[0] > 60 or done == total:
                last[0] = time.time()
                print(f"  {label}: {done}/{total} pairs, {time.time() - start:.0f}s", flush=True)

        pool = open_pool(backbone, records, config["cache_dir"], name, modes, progress=progress)
        seconds = {
            m.name: round(s.meta["seconds"], 1) for m, s in zip(backbone.modes, pool.stores) if s
        }
        print(f"{name}: {len(records)} pairs, encode seconds {seconds}", flush=True)


if __name__ == "__main__":
    main()
