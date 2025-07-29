# Fx-mL-v69

This project contains scripts for preparing Forex data, training models and running walk-forward simulations.

## Raw data

Place your EURUSD 5‑minute price file under `data/raw/` as `EURUSD5.csv`.  The
reference dataset spans from **1971‑01‑04 02:00** through **2024‑07‑12 23:55**
and uses the format:

```
date,time,open,high,low,close,volume
```

`prepare.py` automatically clamps the requested start date to this range. The
first and last timestamps are detected by reading only the first and last rows
of `EURUSD5.csv`, so very large files load quickly.

### CSV formatting

The loader expects two columns named `date` and `time` with values like
`2021.01.01` and `00:05`. Rows where these fields cannot be combined into a
valid timestamp are discarded with a warning. The first few offending rows are
printed so you can inspect them. Pass `strict=True` to `utils.load_data()` (or
use `--strict` with the CLI scripts) to abort instead of dropping them.

## Walk-forward validation

`walkforward.py` verifies that the requested simulation end date does not extend beyond the timestamps contained in the prepared dataset (`data/prepared.csv`).
`prepare.py` and `walkforward.py` both interpret `--end_date` inclusively, so
`--end_date 2024-06-30` loads bars through `2024-06-30 23:55` in the configured
timezone.  Each window now ends on the configured market close so the last day
of data is always included, preventing spurious "exceeds available data" errors.
If the simulation end falls outside the dates in `prepared.csv`, `walkforward.py`
now adjusts the end date to the last available timestamp and emits a warning.
Each iteration also checks the raw CSV before any scripts run. If its
simulation window would go past the last timestamp in `EURUSD5.csv` a warning
like:

```
Simulation window 2024-06-23–2024-06-30 exceeds available data ending 2024-06-21. Extend EURUSD5.csv or adjust stepback settings.
```

is printed and the entire iteration is skipped.

## Temporary outputs

Results are written under `output/` and run-specific folders like `models/run_*`. These files are regenerated each run and are not tracked in version control.

During a walk-forward run each helper script writes its full output to
`models/run_*/logs/<script>.log`. The file `output/output.txt` records only
important summary lines. By default it captures markers like `[prepare]` and
`[label]`, as well as sections labeled `Test Simulation Results` and `Full
Simulation Results`. Additional metric lines such as `Total Trades`, `Win Rate`,
`Profit Factor`, and `Average RR` are also included. Pass `--verbose` to
`walkforward.py` to disable this filtering and capture all output.
