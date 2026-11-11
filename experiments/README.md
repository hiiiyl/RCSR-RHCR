# Twenty-seed experiments

This directory contains the simulator used for the revised RCSR-RHCR experiments and the recorded results for seeds 9400–9419. Tables 7 and 8 remain tables in this release.

## Run

Use Python 3.11 or later. From the repository root:

```sh
python -m pip install -r experiments/requirements.txt
python experiments/run.py --suite all --list
python experiments/summarize.py
python experiments/run.py --suite main --seeds 9400 --methods RCSR-RHCR --workers 1
```

`run.py` accepts `main`, `mechanism`, `window`, `scale`, `async`, `continuous`, and `cost`. Omit `--seeds` and `--methods` to run all recorded combinations in a suite. The full matrix contains 1,360 unique episodes and is computationally expensive. A short interface check is:

```sh
python experiments/run.py --suite continuous --seeds 9400 --methods RCSR-RHCR --steps 6 --limit 1 --workers 1
```

Short runs check execution only; their results do not reproduce the 1,000-step manuscript experiments. Rerun records are written under `reruns/`; existing published CSV files are never overwritten. Completed reruns are reused. Each episode runs in a fresh process. To measure computational cost, finish all other experiments first and use:

```sh
python experiments/run.py --suite cost --workers 1
```

Numerical-library thread counts are set to one. Timing results depend on hardware and load. The recorded cost experiment used an AMD Ryzen 7 5800H under Windows, Python 3.11.9 and NumPy 2.4.5. Original performance records used Python 3.12.14 and NumPy 2.3.5; supplementary records used Python 3.11.9 and NumPy 2.4.5.

## Method and interfaces

`simulator/src/urban_lifelong_planning/revision_experiment.py` runs the shared planning and execution loop. RCSR-RHCR combines component-wise legal graph replay, joint execution filtering, public task-history conditioning, supported marginal medoids, supported injective start assignment, and belief-aware reference selection. The common planner validates its path family and records an explicit hold when it cannot supply a valid result.

`main_runner.py` contains the eight main methods: RCSR-RHCR, RHCR-adapted, CC-RHCR, Index-only EP, deterministic replay, EMC-RHCR, spatial replay without graph constraints, and BaF-guided RHCR. Three additional controls isolate joint replay, history conditioning, and reference selection. CC-RHCR and BaF-guided RHCR are adaptations to the shared simulator; their implementations are in `concurrent_commit_adapter.py` and `literature_baselines.py`.

`extension/` supplies common-timestamp asynchronous observations and the continuous PD execution interface. Moderate links use delays 0–6 and loss probability 0.10; severe links use delays 0–10 and loss probability 0.20. Continuous execution uses the same graph reconstruction model with physical dynamics; it is a surrogate reconstruction model in that setting.

## Results and sample reuse

`results/per_seed_core_metrics.csv` has one row per unique episode:

| Suite | Unique episodes |
| --- | ---: |
| Main, eight methods × three delays × twenty seeds | 480 |
| Three mechanism controls × three delays × twenty seeds | 180 |
| Additional CC commitment windows | 160 |
| Additional fleets of 12 and 18 UAVs | 120 |
| Asynchronous communication | 120 |
| Continuous execution | 120 |
| Independent serial timing | 180 |

Window comparisons reuse main-panel results for commitment length two and the fixed RCSR reference, yielding 240 displayed method–seed–setting combinations. Fleet comparisons reuse the six-UAV main-panel results, yielding 180 displayed combinations. These reused records do not add independent episodes. The mechanism study similarly reuses the main RCSR results.

All episode groups have the same twenty seeds. The continuous RHCR-adapted runs for seeds 9405, 9417 and 9419 complete no tasks under moderate asynchronous feedback. Their P90 is undefined and stored as an empty field; throughput and other defined measurements still use all twenty runs. Missing values must not be replaced by zero.

The sixteen original seeds informed development; four additional seeds were run with unchanged methods and parameters. The pooled twenty-seed results are paired development results. Original valid records were retained and 398 supplementary runs added, including 162 serial timing runs. No claim of an independent held-out evaluation is made.

## Files and units

- `table7_continuous.tex`, `table8_contacts.tex`: the two retained manuscript tables.
- `core_*`: main comparison means, sample standard deviations, valid counts, and paired intervals.
- `mechanism_paired_comparisons.csv`: paired mechanism contrasts after equal-weight averaging over delays within each seed.
- `per_delay_paired_comparisons.csv`: matched-seed contrasts at each fixed delay.
- `window_*`, `scale_*`, `async_*`, `continuous_*`: setting-specific comparisons.
- `*_diagnostics_per_seed.csv`: age of information, completion, physical tracking and contact measurements underlying the extended experiments.
- `cost_per_seed.csv`: phase means and within-episode P95 values, in milliseconds; `cost_summary.csv` summarizes seeds.

Core metrics retain their source units: P90 in graph steps, planning-start error in grid cells, throughput in tasks per graph step, and intervention frequency in interventions per UAV-step. Published intervention summaries multiply the last field by 1,000. Physical tracking error is in metres; contacts distinguish onsets per episode from sampled exposure fractions. `simulator/analysis/metric_registry.json` defines recorded metrics.

`summarize.py` recalculates the core group statistics and paired contrasts wherever a same-setting RCSR reference exists. It checks all main-panel means against the published CSV. Bootstrap intervals use 10,000 seed resamples, random seed 20261107, and interpolated 2.5/97.5 percentiles. Mechanism, window and scale reference reuse is described above; their manuscript contrast tables are included directly. The CSV files contain the manuscript's recorded results; detailed task trajectories are produced by reruns.

## Random inputs

The twenty nominal random tapes are included under `simulator/tapes/`. They fix maps, deployment, tasks, deviations and energy inputs across methods. The recorded larger-fleet tapes are included under `tapes/n12_t1000/` and `tapes/n18_t1000/`. Missing or short-run tapes are generated deterministically with the original `formal_random_tape.py` implementation. Configurations use a 60 × 60 grid, 1,000 steps, 100-step warm-up, two active targets per UAV, planning horizon eight, and a two-step executed prefix unless the CC window is varied.
