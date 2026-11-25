# Tables of the paper

The tables of the paper and its supplementary material as CSV files, one file per table.
They are the source of the tables and figures in the [README](../../README.md).

| File | Table |
|---|---|
| `table1_paired_accuracy_and_cost.csv` | Table 1: paired accuracy, GPU-hours and training visual tokens on three backbones |
| `table4_core_controls.csv` | Table 4: filtering, allocation and score controls on Qwen2.5-VL-3B |
| `table5_cost_decomposition.csv` | Table 5: preparation, routing and training cost of RAT |
| `table6_token_savings.csv` | Table 6: token savings from input allocation at fixed exposure |
| `tableS1_small_and_large_edits.csv` | Table S1: preliminary study by projected edit scale |
| `tableS3_paired_diagnostics_and_general_capabilities.csv` | Table S3: paired-test diagnostics, MathVista, MMBench, POPE |
| `tableS4_transfer.csv` | Table S4: transfer to nine chart, text, document and OCR metrics with standard deviations (its means are Tables 2 and 3) |
| `tableS5_threshold_trials.csv` | Table S5: threshold trials on Qwen2.5-VL-3B |

Empty cells are values the paper reports without a standard deviation.

```python
from rat.reported import best_baseline, check_consistency, load, value

table = load("T1")
print(value(table, "pair_mean", backbone="Qwen2.5-VL-3B", method="rat"))  # 82.61
print(best_baseline(table, "Qwen2.5-VL-3B"))  # ('xmas', 79.5)
print(check_consistency())  # []
```

`rat check-paper` runs the same check from the command line: every number the paper states
more than once (text, tables and figure annotations) is compared with these files.
