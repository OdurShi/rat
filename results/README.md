# Results

| Folder | Content |
|---|---|
| [`paper/`](paper) | The tables of the paper and its supplementary material as CSV files. They are the source of the tables and figures in the top-level README. |
| [`reader_study/`](reader_study) | Outputs of `scripts/reader_study.py`: visual contrast by edit scale, reader margins and routing statistics on a held-out pool (`summary.json`, one row per pair in `pairs.csv`). |
| [`laptop_study/`](laptop_study) | Outputs of `scripts/run_study.py` and `scripts/report_study.py` with the laptop-scale configuration: one JSON file per run (metrics, costs, assignments, predictions) and the summary tables. |
| [`benchmarks/`](benchmarks) | Outputs of `scripts/eval_benchmark.py`. |

The files outside `paper/` were produced by the code in this repository on the hardware and
at the scale recorded in
[`docs/implementation_notes.md`](../docs/implementation_notes.md#5-record-of-executed-runs).
