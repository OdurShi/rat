# Study report: HuggingFaceTB/SmolVLM-256M-Instruct

## Preliminary study (layout of Table S1)

300 training pairs, 900 test pairs at H; paired accuracy in percent.  Edit scale s in pixels of the low mode; thirds are terciles of s (cut points 5.43 and 18.15).

| Statistic | all | s < 2 | s > 7 | lower third | middle third | upper third |
|---|---:|---:|---:|---:|---:|---:|
| Pairs | 900 | 30 | 519 | 300 | 300 | 300 |
| Unadapted accuracy (3 runs) | 19.00 ± 0.00 | 6.67 ± 0.00 | 14.26 ± 0.00 | 25.67 ± 0.00 | 28.33 ± 0.00 | 3.00 ± 0.00 |
| L1x accuracy (3 runs) | 51.67 ± 0.22 | 6.67 ± 0.00 | 57.55 ± 0.29 | 38.00 ± 0.33 | 63.78 ± 0.38 | 53.22 ± 0.19 |
| L4x accuracy (3 runs) | 53.11 ± 0.29 | 10.00 ± 0.00 | 58.57 ± 0.67 | 41.11 ± 0.51 | 63.00 ± 0.58 | 55.22 ± 1.17 |
| U1x accuracy (3 runs) | 51.15 ± 0.61 | 10.00 ± 0.00 | 56.71 ± 0.95 | 38.56 ± 0.19 | 62.44 ± 1.26 | 52.44 ± 1.17 |
| H1x accuracy (3 runs) | 51.11 ± 0.11 | 6.67 ± 0.00 | 56.90 ± 0.29 | 38.00 ± 0.00 | 63.00 ± 0.33 | 52.33 ± 0.33 |
| H1x − L4x (pp) | -2.00 ± 0.19 | -3.33 ± 0.00 | -1.67 ± 0.48 | -3.11 ± 0.51 | -0.00 ± 0.88 | -2.89 ± 1.02 |
| H1x − U1x (pp) | -0.04 ± 0.67 | -3.33 ± 0.00 | 0.19 ± 1.07 | -0.56 ± 0.19 | 0.56 ± 1.54 | -0.11 ± 0.84 |
| H1x − L1x (pp) | -0.56 ± 0.33 | 0.00 ± 0.00 | -0.64 ± 0.44 | 0.00 ± 0.33 | -0.78 ± 0.69 | -0.89 ± 0.51 |

## Paired differences

Difference of paired accuracy in percentage points, averaged over the common seeds, with a 95% paired bootstrap interval over test pairs; the McNemar p-value is for the first common seed.

| Comparison | Seeds | Difference | 95% interval | McNemar p |
|---|---:|---:|---:|---:|
| prelim_H1 − prelim_L4 | 3 | -2.00 | [-3.74, -0.26] | 0.117 |
| prelim_H1 − prelim_U1 | 3 | -0.04 | [-0.85, +0.78] | 1 |
| prelim_H1 − prelim_L1 | 3 | -0.56 | [-1.44, +0.33] | 0.169 |

