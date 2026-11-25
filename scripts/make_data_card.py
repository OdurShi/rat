"""Render sample pairs and compute the statistics of the shipped manifests.

Usage::

    python scripts/make_data_card.py

Writes ``data/samples/`` (PNG images and ``samples.jsonl``), the contact sheet
``assets/results/data_samples.png`` and ``data/statistics.json``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from rat.data.fonts import load_font
from rat.data.pairs import edit_scale
from rat.data.splits import POOLS, load_pool, pair_from_record
from rat.interface.geometry import llava_grid, qwen_grid, tile_grid


def statistics() -> dict:
    out = {}
    for name, spec in POOLS.items():
        records = load_pool(name)
        low = [qwen_grid((r["width"], r["height"]), 256) for r in records]
        high = [qwen_grid((r["width"], r["height"]), 1024) for r in records]
        scales = np.array(
            [
                edit_scale(r["edit_area"], (r["width"], r["height"]), g.resized)
                for r, g in zip(records, low, strict=True)
            ]
        )
        smol_high = [tile_grid((r["width"], r["height"]), 1024, 512, 8).count for r in records]
        out[name] = {
            "pairs": len(records),
            "expected_pairs": spec.size,
            "families": dict(Counter(r["family"] for r in records)),
            "renderings": dict(Counter(r["rendering"] for r in records)),
            "tasks": dict(sorted(Counter(f"{r['family']}/{r['task']}" for r in records).items())),
            "fine_edit_share": float(np.mean([r["fine_edit"] for r in records])),
            "qwen_tokens_per_image": {
                "L": float(np.mean([g.count for g in low])),
                "H": float(np.mean([g.count for g in high])),
            },
            "qwen_full_pool_tokens_millions": {
                "L": 2 * sum(g.count for g in low) / 1e6,
                "H": 2 * sum(g.count for g in high) / 1e6,
            },
            "llava_full_pool_tokens_millions": 2
            * len(records)
            * llava_grid((336, 336), "H").count
            / 1e6,
            "smolvlm_tokens_per_image": {"L": 64.0, "H": float(np.mean(smol_high))},
            "equal_cost_share_qwen": float(
                np.mean([a.count == b.count for a, b in zip(low, high, strict=True)])
            ),
            "edit_scale_L_px": {
                "share_below_2": float((scales < 2).mean()),
                "share_above_7": float((scales > 7).mean()),
                "octiles": [float(q) for q in np.quantile(scales, np.arange(1, 8) / 8)],
            },
        }
    return out


def contact_sheet(records: list[dict], path: Path, width: int = 1500) -> None:
    cell_w, cell_h, label_h, gap = width // 4, 250, 46, 10
    rows = (len(records) + 1) // 2
    sheet = Image.new("RGB", (width, rows * (cell_h + label_h + gap)), "white")
    draw = ImageDraw.Draw(sheet)
    font, bold = load_font("sans", 15), load_font("sans", 15, bold=True)
    for index, record in enumerate(records):
        pair = pair_from_record(record)
        x0 = (index % 2) * 2 * cell_w
        y0 = (index // 2) * (cell_h + label_h + gap)
        for branch, image in enumerate(pair.images()):
            thumb = image.copy()
            thumb.thumbnail((cell_w - 12, cell_h - 8), Image.Resampling.LANCZOS)
            x = x0 + branch * cell_w + (cell_w - thumb.width) // 2
            sheet.paste(thumb, (x, y0 + (cell_h - thumb.height) // 2))
            draw.rectangle(
                (
                    x - 1,
                    y0 + (cell_h - thumb.height) // 2 - 1,
                    x + thumb.width,
                    y0 + (cell_h + thumb.height) // 2,
                ),
                outline=(170, 170, 170),
            )
        draw.text(
            (x0 + 8, y0 + cell_h + 2),
            f"{record['family']} / {record['rendering']} rendering, {record['width']}x{record['height']} px",
            font=font,
            fill=(110, 110, 110),
        )
        draw.text(
            (x0 + 8, y0 + cell_h + 22),
            f"{record['question']}   {record['answers'][0]}  |  {record['answers'][1]}",
            font=bold,
            fill=(20, 20, 20),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, optimize=True)


def main() -> None:
    stats = statistics()
    Path("data/statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    for name, entry in stats.items():
        print(
            name,
            entry["pairs"],
            entry["qwen_tokens_per_image"],
            entry["edit_scale_L_px"]["share_below_2"],
            entry["edit_scale_L_px"]["share_above_7"],
        )
    # Two pairs per stratum of the test pool (three task families x two rendering families).
    test = load_pool("test")
    picked: list[dict] = []
    seen: Counter = Counter()
    for record in test:
        key = (record["family"], record["rendering"])
        if seen[key] < 2:
            seen[key] += 1
            picked.append(record)
    picked.sort(key=lambda r: (r["family"], r["rendering"], r["pair_id"]))
    samples = Path("data/samples")
    samples.mkdir(parents=True, exist_ok=True)
    with open(samples / "samples.jsonl", "w") as stream:
        for record in picked:
            pair = pair_from_record(record)
            for branch, image in enumerate(pair.images()):
                image.save(samples / f"{record['pair_id']}_{branch}.png", optimize=True)
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    contact_sheet(picked[::2], Path("assets/results/data_samples.png"))
    print(f"wrote {len(picked)} sample pairs")


if __name__ == "__main__":
    main()
