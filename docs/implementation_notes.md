# Implementation notes

This document maps the paper to the code, lists every detail the paper leaves open together
with the choice made here, describes what this repository adds, and keeps a record of the
runs that were executed.

Section, equation, table and figure numbers refer to the manuscript "Training
Vision-Language Models with Margin-Based Resolution Selection" (main text and
supplementary material, Appendices A-G).

## 1. Paper → code map

### Paired data (Section 4.1, Appendix B.1, E.1)

| Paper | Code |
|---|---|
| Pair `u = (I0, I1, q, y0, y1)` with a shared question and different answers | `rat/data/pairs.py`: `Pair`, `build_pair` |
| Scene: structured content, rendering specification, question; answer computed by the program | `rat/data/scenes.py`: `Scene`, `RenderSpec`, `sample_scene`, `compute_answer` |
| Edit of a single answer-determining variable; dependent elements (bar lengths, annotations, totals, displayed text) are recomputed | `scenes.py`: `sample_edit`, `_edit_chart`, `_edit_table`, `_edit_card`; `rat/data/render.py` |
| Bar charts, numerical tables, text cards | `render.py`: `_render_chart`, `_render_table`, `_render_card` |
| Font, layout and background sampled independently of the answer and shared within a pair | `scenes.py`: `sample_spec` (its own random stream) |
| Consistency checks: valid target, answer matches the data, marks agree with annotations, elements inside the canvas without unintended overlap; identical renderings removed | `rat/data/checks.py`: `check_scene`; `pairs.py`: `build_pair_report` |
| Pools `F`, `D_tune`, `D_dev`, `D`, final evaluation; preliminary-study pools; all variants of a scene stay in one pool | `rat/data/splits.py`: `POOLS`, `pool_of_seed`, `generate_pool` |
| Pool sizes 10,000 / 5,000 / 2,000 / 50,000 / 4,800 and 2,000 / 4,800 | `splits.py`: `POOLS`; shipped in `data/manifests/` |
| Test set: 1,600 pairs per family, half with held-out fonts, layouts and numerical ranges | `PoolSpec.shifted_half`, `scenes.py`: `LAYOUTS`, `NUMBER_RANGES`, `ID_DIGITS`; `rat/data/fonts.py` |
| Balance across task families up to rounding; manifest fixed before training | round-robin strata in `generate_pool`; `write_manifest`, `load_pool`, `pair_from_record` |
| Projected edit scale (Appendix A.2) | `pairs.py`: `changed_pixels`, `edit_scale`; `rat/analysis/contrast.py`: `projected_edit_scales` |

### Visual interface (Section 4.1, Appendix E.4)

| Paper | Code |
|---|---|
| Modes `A = {L, H}`; token counts from preprocessing geometry | `rat/interface/geometry.py`: `InputMode`, `TokenGrid`; `Backbone.token_grid`, `token_count`, `pair_cost` |
| Qwen2.5-VL: native aspect-ratio-preserving processor, minimum `4 x 28²` pixels, maximum `256 x 28²` (L) or `1,024 x 28²` (H) | `geometry.py`: `smart_resize`, `qwen_pixel_range`, `qwen_grid`; `rat/backbones/hf.py`: `QwenVLBackbone` |
| LLaVA-1.5: CLIP ViT-L/14-336, fixed 576-patch grid; `H` pads to a square 336-pixel input, `L` downsamples the canvas to 168 pixels and upsamples it back | `geometry.py`: `llava_grid`; `hf.py`: `LlavaBackbone._mode_image` |
| `X = phi_a(I)`: projected visual tokens at the interface of the language model | `Backbone.encode_images`; cached by `rat/features.py`: `FeatureStore` |
| Frozen visual encoder and connector; only language-side parameters are updated | `HFBackbone.__init__` (all parameters frozen), `add_adapter` (LoRA restricted to the language model) |

### Shared candidate reader (Section 4.2, Appendix B.2, E.4)

| Paper | Code |
|---|---|
| Eq. (S4): normalised grid coordinates | `rat/reader.py`: `grid_coordinates` |
| `Z = [V, P]`, `d = d_v + 2` | `reader_input`, `pad_inputs` |
| Eq. (S5): candidate embedding from `Question: q \n Answer: y`, mean final-layer state over the answer positions | `HFBackbone.candidate_embeddings`; `PoolData.embed_candidates` |
| Eq. (1): attention readout with `W_A`, `W_R` shared across modes, branches and candidates | `CandidateReader` |
| 2 x 2 score matrix `S_{i,a}` | `CandidateReader.score_matrix`, `score_pairs` |
| Eq. (2): binary cross-entropy over both branches and modes | `reader_loss`, `fit_reader` |
| 10 epochs, 128 pairs per batch, AdamW `3e-4`, weight decay `1e-4`, clipping 1.0, Xavier uniform with gain 0.01, FP32, final epoch | `ReaderFitConfig`; `configs/base.yaml` |
| Reader frozen before threshold selection and student training | `rat/pipeline.py`: `prepare_reader` |

### Margins and routing (Section 4.3, Appendix C)

| Paper | Code |
|---|---|
| Branch margins `l_b`, Eq. (3) minimum margin and gate `m >= gamma > 0` | `rat/margins.py`: `branch_margins`, `min_margin`, `passes` |
| Eq. (S6): `delta`, `beta`, `m = delta - |beta|` | `margins.decompose` |
| Invariances; identical inputs fail every positive threshold | `tests/test_reader.py` |
| Score scale `sigma(m)`; thresholds `log 2`, `log 4`, `log 9` | `margins.PAPER_THRESHOLDS`, `threshold_from_probability` |
| Eq. (4): cheapest passing mode, ties favour `L`, `⊥` for excluded pairs | `rat/routing.py`: `route`, `mode_order` |
| Appendix C.2: lazy routing in `(c_i(a), 1[a = H])` order is exact | `routing.lazy_route`; `tests/test_routing.py` |

### Budget, threshold selection and student adaptation (Section 4.3-4.4, Appendix D, E.2, E.4)

| Paper | Code |
|---|---|
| Eq. (S8): `C = C0 + C_route + C_train`, readout executed iff `C + E_hat + R <= B` | `rat/budget.py`: `CostLedger`; `pipeline.route_pool` |
| Operation-cost estimates from a pilot | `budget.CostModel`; `pipeline.run_pilot` |
| Reservation `R = T_hat(S)` with text lengths and padding; separate training permutation | `budget.TrainingCostEstimator`; `pipeline.fixed_permutation` |
| Algorithm S1, lines 1-14 (routing, exclusion, accepted-but-unretained, unprocessed) | `pipeline.route_pool`; statuses in `rat/routing.py` |
| Algorithm S1, lines 15-20 (one pass, pairs kept together, budget check before every update) | `rat/student.py`: `train_pass`; `pipeline.route_and_train` |
| Eq. (5): answer-only cross-entropy, summed over answer tokens including the end token, averaged over examples | `Backbone.answer_nll`, `student_loss` |
| LoRA rank 16, scaling 32, dropout 0.05 on query/key/value/output and MLP input/output projections | `LoraSettings`; `HFBackbone.add_adapter` |
| AdamW `2e-5`, `(0.9, 0.999)`, `eps = 1e-8`, weight decay 0.01, cosine decay, 3% warm-up, clipping 1.0, 32 pairs per update | `TrainConfig`, `learning_rate_factor`; `configs/base.yaml` |
| Appendix D.1: one equal-budget trial per threshold on `D_tune`, paired accuracy on `D_dev` at `H` (Eq. (S7)), ties by cost then smaller threshold | `pipeline.select_threshold` |
| `B_ref`: one fixed-`H` pass over the candidate pool including feature extraction; budgets 106 / 176 / 128 GPU-hours; trials receive `0.05 B_ref` | `pipeline.reference_budget`; `configs/backbones/*.yaml`; `routing.trial_fraction` |
| Preparation / routing / training decomposition (Table 5) | `CostLedger.summary`; `rat/experiments.py`: `Study.shared_preparation` |
| Greedy decoding at the fixed mode `H` | `Backbone.generate`; `student.predict`; `pipeline.evaluate` |

### Comparisons and controls (Section 5.1, Appendix E.2, E.3)

| Paper | Code |
|---|---|
| Unadapted backbone, full-pool passes at `L` and `H`, uniform selection within the budget | `experiments.Study.run_unadapted`, `run_random` |
| Retention fractions `{0.2, 0.5, 1.0}` and the common development-search allowance | `experiments.RETENTION_FRACTIONS`; `Study.run_random` |
| Pair-level combination of branch scores by the arithmetic mean | `rat/selection/baselines.py`: `pair_average` |
| CLIPScore, D2 Pruning, COINCIDE, XMAS, DOSE, DataProphet | `baselines.clipscore`, `d2_pruning`, `coincide`, `xmas`, `dose`, `dataprophet` |
| Filtering-only (`S_R`, all `H`) | `Study.controls_on_retained(..., ["filter_only"])` |
| Allocation-only (full pool, first passing mode, fallback to `H`) | `Study.run_allocation_only`; `route_pool(fallback_mode=...)` |
| Random allocation within strata of identical `(c_i(L), c_i(H))` | `routing.permute_assignments_within_strata` |
| Mean branch margin; initial-student candidate margin | `Study.run_gated(rule="mean")`, `run_gated(source="student")`; `Backbone.candidate_margins` |
| Equal-count random and answer-loss selection at `H` | `Study.controls_on_retained(..., ["equal_random", "equal_loss"])`; `student.answer_losses` |
| Token saving `100 [1 - Σ c_i(a_i) / Σ c_i(H)]` (Table 6) | `pipeline.token_saving_percent`; `routing.routing_summary` |

### Evaluation (Appendix A.3, E.1)

| Paper | Code |
|---|---|
| Whitespace rule, exact match, Eq. (S2), (S7), (S9), (S10) | `rat/evaluation/paired.py`: `normalize`, `pair_correct`, `branch_correct`, `collapsed`, `paired_metrics` |
| Familiar / held-out halves, six task-by-rendering groups, worst group | `paired_metrics(..., records)` |
| Eq. (S3): run-averaged differences within a group | `paired.accuracy_difference` |
| Mean and standard deviation over seeds 17, 29, 43 | `paired.mean_sd`; `scripts/report_study.py` |
| Table S2: benchmarks, splits and metrics | `rat/evaluation/benchmarks.py`: `BENCHMARKS`, `relaxed_accuracy`, `anls`, `vqa_accuracy`, `ocrbench_total`, `pope_f1`, `circular_accuracy` |

### Preliminary study (Section 3, Appendix A)

| Paper | Code |
|---|---|
| Conditions `L1x`, `L4x`, `H1x`, `H4x` and the sequence-length control `U1x` | `Study.run_preliminary`; `Backbone.encode_images(mode=H, detail=L)`; `Study.degraded_pool` |
| Eq. (S1): paired-token cosine dissimilarity, ratio `D_H / D_L` | `rat/analysis/contrast.py`: `pair_contrast`, `pool_contrast`, `median_ratio` |
| Strict cut-offs `s < 2` and `s > 7` | `contrast.SMALL_EDIT`, `LARGE_EDIT`, `group_masks` |
| Fig. 2b: eight quantile bins of `log10 s`, six bins of the initial `L`-mode answer loss, Eq. (S3) per cell | `contrast.heatmap_cells`, `quantile_bins` |

## 2. Unspecified details and choices

Each item states what the paper leaves open and what this repository does.

**Data generator**

1. *Generator.*  The paper describes the generator (Appendix B.1) but does not list its
   scenes.  The generator in `rat/data` is written from that description; the manifests in
   `data/manifests` are its output and identify every pair by task family, scene seed and
   rendering family.  The pool sizes, the family balance and the shifted half of the test
   set follow Appendix E.1.
2. *Questions.*  Charts ask for an annotated value, the category with the highest or lowest
   bar, or the number of bars; tables for a cell, a displayed column total or the row with
   the highest value in a column; cards for one field (identifier, code, date, time, seat,
   room, count) or the total of a receipt.  Answers are single tokens without spaces.
3. *Edits.*  60% of the edits are fine-grained (one digit or character, or a small change
   of a bar), the others redraw the value.  An edit of a chart value also changes the bar
   and its annotation; an edit of a table cell also changes a displayed total.
4. *Canvas and text size.*  The long side of the canvas is one of 560, 784, 1008, 1232,
   1456, 1680, 1904 pixels with aspect ratios 4:3, 3:2, 16:10, 1:1 or 3:4; the body text is
   7 to 30 pixels high, drawn log-uniformly.  Details of the same pool therefore range from
   illegible in the low-resolution mode to legible in both modes.
5. *Rendering families.*  Familiar renderings use DejaVu Sans and DejaVu Sans Mono, two
   layouts per task family and numbers of two or three digits or one decimal.  Held-out
   renderings use DejaVu Serif and STIX General, one further layout per family, four-digit
   numbers or two decimals and longer identifiers.  The fonts are shipped with the package.
6. *Edited-region area.*  The area is the number of pixels whose colour differs between the
   two renderings by more than 16 of 255 in some channel, at the native canvas size; the
   projected edit scale multiplies its square root by the linear scale of the low mode
   (geometric mean of the two axes).  With this definition a single changed digit of a
   glyph that is `f` pixels high in the low mode has a scale of about `0.56 f`.
   `data/statistics.json` lists the resulting distribution: about 6% of the pairs lie below
   2 pixels and 50% above 7 pixels, and one full-pool pass has 24.1 (L) and 84.3 (H)
   million Qwen tokens.
7. *Random numbers.*  Scenes are drawn from raw PCG64 words with explicit arithmetic
   (`rat/data/rng.py`), so a seed gives the same scene on every platform.  Rasterised
   pixels can differ slightly between FreeType versions.
8. *Overlength filter.*  All generated questions and answers are far shorter than the
   2,048-token limit, so no example is removed for length.

**Interface and reader**

9. *Dataset prompt.*  The student sees the image and
   `"{question}\nAnswer the question using a single word or phrase."` in the chat template
   of the backbone; the target is the answer followed by the end-of-turn token
   (`<|im_end|>` for Qwen, `</s>` for LLaVA, `<end_of_utterance>` for SmolVLM).
10. *Answer positions of Eq. (S5).*  The text is tokenised without special tokens; a token
    belongs to the answer if its characters end after the start of the answer string.
    The final-layer states are those after the last normalisation layer.
11. *Coordinates for tiled interfaces.*  For SmolVLM a token of a tile receives the centre
    of its region in the whole image, and the tokens of the global view use Eq. (S4) on the
    whole image.  For a single grid this is Eq. (S4).
12. *Feature cache.*  Tokens are stored in half precision and read back in single
    precision; the device time of every encoding call is stored with them and charged when a
    run first uses the features.
13. *Reader batches.*  A batch is processed in micro-batches (`reader.micro_pairs`) with
    accumulated gradients, which is equivalent to one batch.
14. *Feature standardisation.*  The paper keeps the native scales of visual tokens and
    candidate embeddings.  `reader.standardize: true` standardises both per dimension with
    statistics of the fitting pool (Section 3 of this document); the default is `false`.

**Routing, budget and training**

15. *Random orders.*  The traversal order of the student pool, its training order, and the
    two orders of the tuning pool are four independent permutations seeded by the run seed
    (`pipeline.fixed_permutation`, streams 1-4).
16. *Cost model.*  Feature extraction is linear in the token count; one forward/backward
    pass costs `train_fixed + train_per_token x examples x padded length`; greedy decoding
    is linear in the prompt length.  The coefficients come from the pilot.
17. *Accounting mode.*  `budget.accounting: measured` charges device seconds times the
    number of devices, as in the paper.  `modelled` charges the pilot's estimates instead,
    which makes runs on a shared machine repeatable; the wall-clock time is still recorded.
18. *Forward chunks.*  The paper trains one pair per GPU with four accumulation steps on
    eight GPUs.  `student.pairs_per_forward` sets the pairs per forward pass and
    `student.batch_pairs` the pairs per update; the reservation uses the same chunks.
19. *Development scoring inside a trial.*  The estimated cost of scoring the development
    pool is reserved before routing; with measured accounting it is replaced by the measured
    time afterwards.
20. *Retention and `H` share of a trial* are computed with the gate on the whole tuning
    pool (as in Table S5), independently of how many pairs the trial could train.
21. *Uniform selection.*  The final run of a uniform baseline keeps the first
    `fraction x |D|` pairs of the traversal order and trains them until the budget stops it.
22. *Costs of the controls.*  Filtering-only and random allocation inherit RAT's
    preparation and routing cost and add the extraction of features RAT did not need;
    equal-count random selection pays the pilot and the extraction of its pairs; answer-loss
    selection also pays the extraction and scoring of the whole pool.
23. *Student margin.*  The scoring time of the initial student is measured per pair and
    added to every readout of that control.
24. *New seeds.*  A seed draws LoRA `A` from the Kaiming-uniform law and sets `B` to zero
    with its own generator, so every method of a seed starts from the same `theta_0`.

**Evaluation**

25. *Maximum generation length.*  128 tokens for short answers as in Appendix E.1.  The
    laptop configurations use 24 tokens, which is longer than every reference answer of
    the generated pairs and therefore cannot change an exact-match result.
26. *Benchmark metrics.*  `rat/evaluation/benchmarks.py` implements the scoring rules of
    Table S2; `scripts/eval_benchmark.py` runs the benchmarks that are available as public
    Hugging Face datasets.

## 3. Additions of this repository

* **A third interface family.**  `SmolVLMBackbone` (256M and 500M) uses 512-pixel tiles;
  `L` is the single global view (64 tokens) and `H` the image resized to a longest edge of
  1,024 pixels (up to 320 tokens).  It makes student training runnable on a laptop.
* **Resolution ladders.**  `routing.route`, `lazy_route` and `route_pool` accept any number
  of modes; Eq. (4) and the exactness argument of Appendix C.2 do not depend on there being
  two.
* **Feature standardisation for the reader** (`reader.standardize`).  Adam moves every
  weight by about the learning rate per step, so the change of a score per step grows with
  the norms and the dimensions of the features.  The switch standardises every feature
  dimension with statistics of the fitting pool.  On SmolVLM-256M the fitting loss of
  Eq. (2) rose at the paper's learning rate and fell with the switch; on Qwen2.5-VL-3B it
  rose with and without it (Section 5).
* **Paired uncertainty.**  `rat/evaluation/stats.py` gives a paired bootstrap interval and
  an exact McNemar test for the difference between two methods on the shared test pairs.
* **Frozen-side study.**  `scripts/reader_study.py` measures contrast, reader margins and
  routing statistics without training a student, so the gate can be inspected on a new
  backbone in minutes.
* **Checks of the reported numbers.**  `rat/reported.py` verifies every value the paper
  states more than once against the tables in `results/paper`.
* **Quick start.**  `rat demo` fits the reader and routes real generated pairs on a
  simulated interface in a few seconds on a CPU.

## 4. Scope of the implementation

* The selectors of `rat/selection/baselines.py` implement the selection rules on arrays of
  features, scores and trajectories.  The proxy models that produce these arrays in the
  paper (CLIP, TinyLLaVA-2B, Vicuna-7B, InternVL3-2B) are not wrapped; `Study` runs RAT, its
  controls and the uniform baselines end to end.
* `scripts/eval_benchmark.py` covers ChartQA, DocVQA, InfographicVQA and TextVQA.  The
  metrics of the other benchmarks of Table S2 are implemented without a data loader.
* GPU-hours are recorded as device seconds per stage; peak memory is not recorded.

## 5. Record of executed runs

All runs below were made on one Apple-silicon laptop (Apple M5, 32 GB, PyTorch 2.15 on the
`mps` device, no CUDA) on 2026-11-25, with the generator and the manifests of this
repository.  They are small.  The numbers of the paper quoted in the README come from the
manuscript; runs with the paper's pool sizes and budgets, and student training of the
paper's backbones, are not part of this record.

**Test-suite and lint.**  `pytest` passes on Python 3.11 (72 tests; torch 2.15.0,
transformers 5.23.0, peft 0.21.3) and on Python 3.10 with the `dev` extra only (torch 2.14.1;
the two comparisons with the Transformers processors are skipped); `ruff check .` and
`ruff format --check .` are clean.  The tests cover, among others: Eq. (1) and (2) against
hand computations; the invariances of Appendix C.1 and the identical-input property; the
exactness of lazy routing on random margins and cost ties; the reservation estimate against
a direct computation; the statuses of Algorithm S1 with and without a budget stop; the
preservation of pairs, mode counts and tokens by the random-allocation control; every
method of `Study` on the tiny backbone; and the mutual consistency of the reported tables.

**Data.**  `scripts/build_manifests.py --workers 8` generated the seven manifests in about
five minutes; between 95.4% and 99.8% of the candidate seeds of a stratum pass the
consistency checks.
`data/statistics.json` holds the statistics of every pool.

**Quick start.**  The output of `rat demo` is reproduced in the README; it is identical on
Python 3.10 and 3.11.

**Backbone wrappers.**

* *SmolVLM-256M-Instruct* (language model in float32, visual interface in float16): the
  token counts predicted by `tile_grid` equal the model's for every encoded image; encoding
  takes 0.05 s (L, 64 tokens) and 0.23 s (H, 320 tokens) per image, one forward/backward
  pass 0.07 s (L) and 0.27 s (H) per example.
* *Qwen2.5-VL-3B-Instruct* (float16): the token counts predicted by `qwen_grid` equal the
  model's on 16 development images in both modes (228-256 tokens at L, 260-1,000 at H);
  encoding takes about 0.5 s (L) and 1.8 s (H) per image.  Greedy decoding from cached
  features answers 14 of these 16 images correctly at H and 10 at L before any adaptation;
  the six images it misses at L belong to the three pairs with the smallest text (8 pixels),
  and in two of these pairs both images receive the same answer.
* *LLaVA-1.5-7B*: the wrapper shares the code path of the other two and was not run here.

**Benchmark script.**  `scripts/eval_benchmark.py` streams ChartQA, DocVQA, InfographicVQA
and TextVQA with the identifiers in the script.  Unadapted Qwen2.5-VL-3B answers the first
100 test questions of ChartQA (all human-written) with 67.0% relaxed accuracy at H
(`results/benchmarks/`).

**Frozen-side study on Qwen2.5-VL-3B** (`results/reader_study/qwen2.5-vl-3b*`;
`configs/laptop/qwen2.5-vl-3b_reader.yaml`, float16; 800 fitting pairs of `reader_fit`, the
first 300 pairs of `tune` as held-out pool, seed 17).  A held-out image takes 241.6 visual
tokens at `L` (228-256) and 840.0 at `H` (240-1,024); encoding takes 0.38 s and 1.41 s per
image.

*Contrast (Eq. (S1)).*  Median `D_H / D_L` of the held-out pairs by projected edit scale `s`
in pixels at `L` (`assets/results/contrast_qwen2.5-vl-3b.png`):

| | s < 2 | 2 ≤ s ≤ 7 | s > 7 | All |
|---|---:|---:|---:|---:|
| Pairs | 27 | 135 | 138 | 300 |
| Median `D_H / D_L` | 3.27 | 0.96 | 0.86 | 0.90 |
| Pairs with `D_H > D_L` (%) | 85.2 | 48.1 | 26.1 | 41.3 |

Table S1 reports 2.70 for `s < 2` and 1.07 for `s > 7`.  The high-resolution mode keeps
about three times the contrast of the smallest edits, as in the paper; for larger edits the
two modes are close, with the ratio here below one.  By family the median ratio is 0.86 for
charts (median edit scale 61.6 pixels), 1.00 for tables (4.9) and 1.03 for text cards (3.9).

*Reader (Eq. (1), (2)).*  Ten epochs at the paper's learning rate of 3e-4 with 16 pairs per
update (500 updates); the reader has about 8.4 million weights on this backbone.

| Reader settings | Loss | Fit branch acc. L / H | Held-out branch acc. L / H | Held-out pair acc. L / H | Retained at log 2 / log 4 / log 9 |
|---|---:|---:|---:|---:|---:|
| Paper (native feature scales) | 0.693 → 1.055 | 0.574 / 0.565 | 0.505 / 0.503 | 0.070 / 0.053 | 0.7 / 0.3 / 0% |
| With `reader.standardize` | 0.693 → 1.229 | 0.558 / 0.551 | 0.510 / 0.517 | 0.030 / 0.040 | 2.0 / 0.3 / 0.3% |

With 800 fitting pairs the loss of Eq. (2) rises above its initial value in both settings,
the held-out branch accuracy stays at chance and the gate retains between zero and six of
the 300 pairs, so the routing statistics of Fig. 5 cannot be read from these runs.  The
paper fits the reader on 10,000 pairs with 128 pairs per update.

*Unadapted model at `H`* (`--student`; exact match, greedy decoding from the cached
features).  Paired accuracy 82.33%, single-image accuracy 87.00%, collapse 4.00% on the 300
held-out pairs, all with familiar renderings:

| Group | Pairs | Pair (%) | Single (%) | Collapse (%) |
|---|---:|---:|---:|---:|
| Charts | 100 | 78.00 | 84.50 | 1.00 |
| Tables | 100 | 82.00 | 86.00 | 10.00 |
| Text cards | 100 | 87.00 | 90.50 | 1.00 |
| Edit scale s < 2 | 27 | 62.96 | 68.52 | 7.41 |
| 2 ≤ s ≤ 7 | 135 | 86.67 | 90.00 | 6.67 |
| s > 7 | 138 | 81.88 | 87.68 | 0.72 |
| Font size on the canvas ≤ 9 px | 64 | 64.06 | | |
| 10-14 px | 101 | 83.17 | | |
| ≥ 15 px | 135 | 90.37 | | |

Twelve of the 53 failed pairs fail by the form of the answer only (a number word for a
digit, a full month name for its abbreviation, a dropped trailing zero); all twelve have an
edit scale above 7 pixels, and accepting them gives 86.33%.  Table S3 reports 57.42% paired
accuracy on familiar renderings, 70.81% single-image accuracy and 13.65% collapse for the
unadapted Qwen2.5-VL-3B on the paper's test pool: the pairs of this repository's generator
are easier for the unadapted model than the paper's, in line with their edit scales (6.5% of
the pairs of `prelim_test` are below 2 pixels here, `data/statistics.json`; 1,945 of 4,800,
or 40.5%, in Table S1).

**Learning rate of the laptop configuration.**  Stage `lr-pilot` trained 300 uniformly
selected pairs of the student pool at H for one pass (eight pairs per update) and scored
the 100-pair development pool:

| Learning rate | Dev Pair (%) | Dev Single (%) |
|---|---:|---:|
| 2e-5 | 34.0 | 40.5 |
| 1e-4 | 52.0 | 60.5 |
| 2e-4 | 49.0 | 58.5 |
| 5e-4 | 51.0 | 59.5 |

`configs/laptop/smolvlm_256m.yaml` uses 1e-4.  The three larger rates are within the noise
of a 100-pair pool; the paper's 2e-5 gives too few effective updates at this scale.

**Reader on SmolVLM-256M** (`results/reader_study/smolvlm-256m*`; 400 fitting pairs, 200
held-out pairs of the tuning pool, seed 17).

| Reader settings | Loss | Fit branch acc. L / H | Held-out branch acc. L / H | Retained at log 2 / log 4 / log 9 |
|---|---:|---:|---:|---:|
| Paper (native feature scales) | 0.693 → 0.746 | 0.52 / 0.52 | 0.50 / 0.51 | 0 / 0 / 0% |
| With `reader.standardize` | 0.693 → 0.595 | 0.64 / 0.68 | 0.54 / 0.58 | 2.0 / 1.0 / 0% |

The projected tokens of this backbone have a mean norm of about 150 and the candidate
embeddings of about 45.  At the paper's learning rate the loss of Eq. (2) rises; with
standardised features it falls and the high-resolution mode becomes more readable than the
low-resolution one, but a reader fitted on 400 pairs barely generalises, so the gate retains
almost nothing.  One token of this interface covers 64 x 64 pixels of a tile, which holds
several text lines of a typical scene.  The median contrast ratio `D_H / D_L` is 3.38 for
the 13 held-out pairs with an edit scale below 2 pixels and 0.73 for the 107 pairs above
7 pixels: `H` has five times as many tokens, most of them unchanged, so large edits are
diluted while small edits only become visible at `H`.

**Preliminary study on SmolVLM-256M** (`results/laptop_study/smolvlm-256m`;
`scripts/run_study.py --stages preliminary --seeds 17 29 43`).  300 training pairs of
`prelim_train`, 900 test pairs of `prelim_test` at fixed `H` input, LoRA rank 16 on the
135M language model, learning rate 1e-4, eight pairs per update; paired accuracy in percent,
mean ± standard deviation over three runs.  The edit scale `s` is in pixels of the 512-pixel
low mode.

| Condition | All 900 pairs | s < 2 (30) | s > 7 (519) | Lower third of s | Middle third | Upper third |
|---|---:|---:|---:|---:|---:|---:|
| Unadapted | 19.00 | 6.67 | 14.26 | 25.67 | 28.33 | 3.00 |
| L1× | 51.67 ± 0.22 | 6.67 ± 0.00 | 57.55 ± 0.29 | 38.00 ± 0.33 | 63.78 ± 0.38 | 53.22 ± 0.19 |
| L4× | 53.11 ± 0.29 | 10.00 ± 0.00 | 58.57 ± 0.67 | 41.11 ± 0.51 | 63.00 ± 0.58 | 55.22 ± 1.17 |
| U1× | 51.15 ± 0.61 | 10.00 ± 0.00 | 56.71 ± 0.95 | 38.56 ± 0.19 | 62.44 ± 1.26 | 52.44 ± 1.17 |
| H1× | 51.11 ± 0.11 | 6.67 ± 0.00 | 56.90 ± 0.29 | 38.00 ± 0.00 | 63.00 ± 0.33 | 52.33 ± 0.33 |
| H1× − L4× (pp) | −2.00 ± 0.19 | −3.33 | −1.67 ± 0.48 | −3.11 ± 0.51 | 0.00 ± 0.88 | −2.89 ± 1.02 |
| H1× − U1× (pp) | −0.04 ± 0.67 | −3.33 | 0.19 ± 1.07 | −0.56 ± 0.19 | 0.56 ± 1.54 | −0.11 ± 0.84 |

The 95% paired bootstrap interval of `H1× − L4×` over the test pairs is [−3.74, −0.26] and
that of `H1× − U1×` is [−0.85, +0.78].  At this scale the result of Section 3 does not
appear: every trained condition lifts paired accuracy from 19% to 51-53% regardless of the
training resolution, four low-resolution passes are slightly ahead of one high-resolution
pass, and pairs with an edit scale below 2 pixels stay unread in all conditions.  A few
hundred pairs and 38 to 150 updates of a 135M language model mainly teach the answer format
(the unadapted model ends 58% of its answers with a full stop, which exact matching rejects); they
do not change what the model reads from the image.  The paper's study uses 2,000 pairs and
a 3B backbone.
