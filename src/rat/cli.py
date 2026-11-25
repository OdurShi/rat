"""Command-line interface: ``rat demo``, ``rat pairs``, ``rat geometry``, ``rat check-paper``."""

from __future__ import annotations

import argparse
from pathlib import Path


def _demo(args: argparse.Namespace) -> None:
    from .demo import run

    run(args.seed)


def _pairs(args: argparse.Namespace) -> None:
    from .data.splits import load_pool, pair_from_record

    args.out.mkdir(parents=True, exist_ok=True)
    for record in load_pool(args.pool, args.count):
        pair = pair_from_record(record)
        for branch, image in enumerate(pair.images()):
            image.save(args.out / f"{record['pair_id']}_{branch}.png")
        print(f"{record['pair_id']}: {pair.question}  ->  {pair.answers[0]} | {pair.answers[1]}")


def _geometry(args: argparse.Namespace) -> None:
    from .interface.geometry import llava_grid, qwen_grid, tile_grid

    size = (args.width, args.height)
    rows = (
        ("Qwen2.5-VL (256 / 1,024 tokens)", qwen_grid(size, 256), qwen_grid(size, 1024)),
        ("LLaVA-1.5 (168 / 336 pixels)", llava_grid(size, "L"), llava_grid(size, "H")),
        (
            "SmolVLM (512 / 1,024 pixels)",
            tile_grid(size, 512, 512, 8),
            tile_grid(size, 1024, 512, 8),
        ),
    )
    print(f"image {args.width} x {args.height}")
    for name, low, high in rows:
        print(
            f"  {name:<34} L: {low.count:>4} tokens at {low.resized}   "
            f"H: {high.count:>4} tokens at {high.resized}"
        )


def _check_paper(_args: argparse.Namespace) -> None:
    from .reported import check_consistency

    problems = check_consistency()
    print("\n".join(problems) if problems else "The reported tables are mutually consistent.")
    raise SystemExit(1 if problems else 0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rat", description="Representation-Aware Training")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="CPU quick start with the tiny built-in backbone")
    demo.add_argument("--seed", type=int, default=17)
    demo.set_defaults(run=_demo)
    pairs = commands.add_parser("pairs", help="render pairs of a pool to PNG files")
    pairs.add_argument("--pool", default="test")
    pairs.add_argument("--count", type=int, default=6)
    pairs.add_argument("--out", type=Path, default=Path("data/rendered"))
    pairs.set_defaults(run=_pairs)
    geometry = commands.add_parser(
        "geometry", help="visual tokens of an image size per backbone and mode"
    )
    geometry.add_argument("--width", type=int, required=True)
    geometry.add_argument("--height", type=int, required=True)
    geometry.set_defaults(run=_geometry)
    check = commands.add_parser("check-paper", help="check the reported tables for consistency")
    check.set_defaults(run=_check_paper)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
