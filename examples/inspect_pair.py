"""Look at one generated pair without a model.

    python examples/inspect_pair.py [--pool test] [--index 1] [--save pair]

Prints the question, both answers, the edited area, the projected edit scale and the visual
tokens of the pair for the three interface families, and optionally saves both images.
"""

from __future__ import annotations

import argparse

from rat.data.pairs import edit_scale
from rat.data.splits import load_pool, pair_from_record
from rat.interface.geometry import llava_grid, qwen_grid, tile_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pool", default="test")
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--save", help="write <save>_0.png and <save>_1.png")
    args = parser.parse_args()
    record = load_pool(args.pool, args.index + 1)[args.index]
    pair = pair_from_record(record)
    spec = pair.scenes[0].spec
    print(f"{record['pair_id']}: {pair.family} ({record['task']}), {spec.rendering} rendering")
    print(
        f"  canvas {pair.width} x {pair.height} px, {spec.font} font at {spec.font_px} px, layout {spec.layout}"
    )
    print(f"  question: {pair.question}")
    print(f"  answers:  {pair.answers[0]!r} | {pair.answers[1]!r}")
    print(f"  edited area: {pair.edit_area} pixels ({'fine' if pair.fine_edit else 'coarse'} edit)")
    grids = {
        "Qwen2.5-VL": (qwen_grid(pair.size, 256), qwen_grid(pair.size, 1024)),
        "LLaVA-1.5": (llava_grid(pair.size, "L"), llava_grid(pair.size, "H")),
        "SmolVLM": (tile_grid(pair.size, 512, 512, 8), tile_grid(pair.size, 1024, 512, 8)),
    }
    for name, (low, high) in grids.items():
        scales = [edit_scale(pair.edit_area, pair.size, grid.resized) for grid in (low, high)]
        print(
            f"  {name:<11} L: {2 * low.count:>4} tokens per pair, edit scale {scales[0]:5.2f} px"
            f"   H: {2 * high.count:>4} tokens, edit scale {scales[1]:5.2f} px"
        )
    if args.save:
        for branch, image in enumerate(pair.images()):
            image.save(f"{args.save}_{branch}.png")
        print(f"  wrote {args.save}_0.png and {args.save}_1.png")


if __name__ == "__main__":
    main()
