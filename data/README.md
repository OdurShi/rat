# Data

## Paired pools

`manifests/` holds the generation manifests of every pool with the sizes of Appendix E.1.
A manifest is a gzip-compressed JSON-lines file with one record per pair.

| Pool | File | Pairs | Use |
|---|---|---:|---|
| `reader_fit` | `manifests/reader_fit.jsonl.gz` | 10,000 | fitting the shared reader |
| `tune` | `manifests/tune.jsonl.gz` | 5,000 | threshold trials |
| `dev` | `manifests/dev.jsonl.gz` | 2,000 | development evaluation |
| `train` | `manifests/train.jsonl.gz` | 50,000 | student candidate pool |
| `test` | `manifests/test.jsonl.gz` | 4,800 | paired test set: 1,600 pairs per family, half with held-out renderings |
| `prelim_train` | `manifests/prelim_train.jsonl.gz` | 2,000 | preliminary study, training |
| `prelim_test` | `manifests/prelim_test.jsonl.gz` | 4,800 | preliminary study, evaluation |

Every pool is balanced over bar charts, numerical tables and text cards, and every prefix of
a manifest is balanced as well, so smaller experiments take the first pairs of a pool.  Scene
seeds are disjoint between pools.

A record looks like this:

```json
{"pair_id": "test-00001", "pool": "test", "family": "table", "seed": 51000000,
 "rendering": "familiar", "task": "argmax", "question": "Which city has the highest Units value?",
 "answers": ["Carver", "Hollis"], "width": 1232, "height": 924, "layout": "zebra",
 "font": "mono", "font_px": 18, "fine_edit": true, "edit_area": 295}
```

`family`, `seed` and `rendering` regenerate the pair; the other fields describe it and are
verified when the pair is rebuilt.  `edit_area` is the number of pixels that differ between
the two images at the native canvas size.

```python
from rat.data.splits import load_pool, pair_from_record

record = load_pool("test", 10)[1]
pair = pair_from_record(record)  # checks the record against the regenerated pair
image0, image1 = pair.images()  # PIL images at the native canvas size
print(pair.question, pair.answers)
```

```bash
rat pairs --pool test --count 12 --out data/rendered     # write PNG files
python scripts/build_manifests.py --workers 8            # rebuild every manifest (about five minutes)
python scripts/make_data_card.py                         # samples, contact sheet, statistics.json
```

## Samples and statistics

`samples/` contains twelve rendered pairs of the test pool (two per task family and
rendering family) with their records in `samples.jsonl`.  `statistics.json` lists, for every
pool, the task mix, the visual tokens per image under the Qwen2.5-VL and SmolVLM geometries
and the distribution of the projected edit scale.

| | Qwen2.5-VL, L | Qwen2.5-VL, H | LLaVA-1.5 | SmolVLM, L | SmolVLM, H |
|---|---:|---:|---:|---:|---:|
| Visual tokens per image (student pool, mean) | 241.1 | 843.1 | 576 | 64 | 320 |
| Visual tokens of one full-pool pass (millions) | 24.11 | 84.31 | 57.60 | 6.40 | 32.00 |

## Fonts

The renderer uses DejaVu Sans, DejaVu Sans Mono, DejaVu Serif and STIX General, shipped in
`src/rat/data/fonts/` with their licences (`LICENSE_DEJAVU`, `LICENSE_STIX`).

## Natural benchmarks

ChartQA, ChartQAPro, TextVQA, DocVQA, InfographicVQA, OCRBench v1/v2, MathVista, MMBench and
POPE are used for evaluation only and are not redistributed here.
`scripts/eval_benchmark.py` streams ChartQA, DocVQA, InfographicVQA and TextVQA from the
Hugging Face Hub; the scoring rules of all ten benchmarks are in
`src/rat/evaluation/benchmarks.py`.
