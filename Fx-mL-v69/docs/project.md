# Trading System: Project Overhaul & Strategic Blueprint

> **Author:** David Stetler
> **Last Updated:** 2025‑05‑29 (Stacking v0 roadmap)

---

## 1. Vision

Build a **context‑aware, modular Forex trading system** that

* **Enters trades with high precision** (Class 1 signals only when we are very confident).
* **Optimises each trade's SL/TP levels dynamically** based on market regime.
* **Adapts weekly** through walk‑forward retraining.
* **Target metrics** (not strict acceptance):

  * Aim for **58 – 72 % win rate**
  * Average **1 : 2 to 1 : 3 risk‑reward**
  * **25 – 50 trades per week**
  * Seamless live execution via MT4/DWX

---

## 2. Strategic Principles

* **Precision‑first mindset** – predicting “no trade” (Class 0) is cheap; entering a bad trade is deadly.
* **Rare‑event classification** – embrace imbalance; design labels, loss functions, and metrics accordingly.
* **Separation of concerns** – direction (enter?) and exits (SL/TP) are solved by different models.
* **Market regimes matter** – treat each regime independently when it helps, share signal power when data are sparse.
* **Reproducibility > tinkering** – every experiment writes to its own run‑ID; artefacts are immutable.
* **Walk‑forward discipline** – weekly retrain + sim loop to keep the model honest.

---

## 3. Collaboration Model

| Role        | Responsibility                                                  |
| ----------- | --------------------------------------------------------------- |
| **ChatGPT** | Strategy, architecture, code review, spec writing               |
| **Cursor**  | Executes granular multi‑file edits & refactors based on prompts |
| **David**   | Product owner, tester, live‑trade operator                      |

Workflow: *decide spec → craft Cursor prompt → run tests → update **project.md***.

---

## 4. Pipeline (Live Status)

| Phase            | Script(s)             | Status     | Description                                                    |
| ---------------- | --------------------- | ---------- | -------------------------------------------------------------- |
| **Feature Prep** | `prepare.py`          | ✅ Stable   | Indicators, lags, regime clustering, scaling                   |
| **Labeling**     | `label.py`            | ✅ Stable  | Direction labels + SL/TP bucket + cooldown/ATR gates; `--train_end_date` masks future labels to avoid training leakage |
| **Base Models**  | `train_base.py` ✚ CNN | ✅ Working | LGBM per (regime, side) + global 1‑D CNN; output per‑bar probs |
| **Meta Model**   | `train_meta.py`       | ✅ Working | Logistic GBM stacks base probabilities + rule features         |
| **Edge Ranker**  | `train_ranker.py`     | ✅ Working | Ranks signals via meta prob × (TP−SL) and auto‑thresholds to 25‑50 trades/wk; generates holdout candidate trades even without labels |
| **Simulation**   | `simulate.py`         | ✅ Working  | Equity‑aware, single-position trade manager with per-side cooldown |
| **Walk‑Forward** | `walkforward.py`      | ✅ Working | 18‑month train window, weekly stepback loop; optional optimisation via `optimize.py` |

Add `--optimize` to enable a grid search before each iteration:

```bash
python walkforward.py --run OUTDIR --optimize
```

Use `--grid fast|mid|full` to pick a smaller search grid. Combine with
`--max_configs N` for quicker iterations:

```bash
walkforward.py --optimize --grid fast --max_configs 10
```

`--max_configs N` caps how many parameter sets are tested. A full search
can yield improved win rate or profit factor but takes much longer.

---

## 5. Standard Data & Artefact Schema

| Dataset             | Index (tz‑aware 5 min) | Key Columns                                                              | Produced by       |
| ------------------- | ---------------------- | ------------------------------------------------------------------------ | ----------------- |
| **prepared.csv**    | `datetime`             | `open, high, low, close, volume` + engineered features + `market_regime` | `prepare.py`      |
| **labeled.csv**     | same                   | all *prepared* cols + `label_long`, `label_short`, `bucket_id`           | `label.py`        |
| **probs\_base.csv** | same                   | `prob_long_lgbm`, `prob_short_lgbm`, `prob_cnn`, …                       | `train_base.py`   |
| **signals.csv**     | same                   | `edge_score`, `regime`, `sl_pips`, `tp_pips`, `meta_prob`                | `train_ranker.py` |
| **trades.csv**      | subset                 | `timestamp, price, side, sl_pips, tp_pips, regime`                       | `simulate.py`     |

Raw price files must include `date` and `time` columns plus the OHLCV fields. Any
additional feature columns are preserved. The loader combines `date` and `time`
into a timezone-aware index.

*Prices → float64 (5‑dec), pips → float32 positive, labels → int16 (−1 = no signal).  All engineered features are standard‑scaled; raw ATR/ADX/RSI keep a `_raw` suffix.*

Timestamps are localized using `market.timezone` (default **America/Chicago**).
The trading week spans from `market.weekly_open` to `market.weekly_close`
(Sunday 16:00 → Friday 16:00 by default).

---

## 6. Script Interface Contracts

### 6.1 `prepare.py`

*Purpose* – engineer features, cluster regimes, scale.
*CLI* – `python prepare.py --config config.yaml --start_date YYYY‑MM‑DD --end_date YYYY‑MM‑DD [--train_end_date YYYY‑MM‑DD] [--input_dir PATH] [--strict]`
`start_date` and `end_date` limit the raw data range. `train_end_date` controls the cutoff for fitting scalers and clustering. `--input_dir` overrides `data.input_dir` (defaults to `data/raw/`).
`--strict` raises errors instead of dropping malformed rows when loading CSVs.
`prepare.n_clusters` defaults to **3**. Use `auto` if you want the script to choose 2–4 clusters based on the date range.
*Outputs* – `${RUN}/data/prepared.csv`, `${RUN}/models/scaler.joblib`
*Perf target* – ≤ 15 s for 3 yrs on M2‑Max.

### 6.2 `label.py`

Generates SL/TP grid, labels `label_long/short`, enforces ATR & cooldown. Now accepts `--train_end_date` to mask labels that would look past the cutoff.
Outputs `${RUN}/data/labeled.csv`.
Perf target – ≤ 10 s, ≥ 5 signals / week.
*CLI* – `python label.py --run RUN --start_date YYYY‑MM‑DD [--train_end_date YYYY‑MM‑DD] [--strict] --end_date YYYY‑MM‑DD`

`--train_end_date` masks labels that would look past the cutoff to avoid leakage.
This masking is solely to prevent leakage when training models and has no effect on candidate trade generation.
`--strict` aborts instead of dropping malformed rows when reading CSVs.

### 6.3 `train_base.py`

Trains **LGBM** per (regime, side) and a global **1‑D CNN** on sliding windows.
The CNN model is saved to `${RUN}/models/base/cnn_model.h5` and its
predictions populate the `prob_cnn` column in `probs_base.csv` alongside the
LGBM probabilities.
*CLI* – `python train_base.py --run RUN --start_date YYYY‑MM‑DD --train_end_date YYYY‑MM‑DD --end_date YYYY‑MM‑DD`
`train_end_date` marks the split between training and simulation data.

### 6.4 `train_meta.py`

Stacks base probabilities + contextual features (ATR\_pct, hour, regime) into a logistic GBM; outputs `${RUN}/models/meta.pkl`.
*CLI* – `python train_meta.py --run RUN --start_date YYYY‑MM‑DD --train_end_date YYYY‑MM‑DD --end_date YYYY‑MM‑DD`

### 6.5 `train_ranker.py`

Reads `${RUN}/data/labeled.csv` and `${RUN}/data/probs_base.csv`, then loads
the saved meta models from `${RUN}/models/meta.pkl`.  Using SL/TP buckets from
`sltp.py`, it maps each predicted bucket to `(sl_pips, tp_pips)` and computes
`edge_score = meta_prob × (tp_pips − sl_pips)` for every signal.  A threshold is
auto‑selected so the historical trades above it average **25‑50 per week**.
The search first tries to find a threshold keeping both the training and
holdout periods within this range. If none exists it gradually lowers
`min_trades_per_week` until a valid threshold is found (emitting a warning when
relaxed); only then does it fall back to the training‑only logic.  The
chosen `edge_threshold` along with trades/week for both the training and
holdout periods are stored in `${RUN}/models/edge_threshold.json`.  If the
threshold exceeds all holdout scores, it is recomputed on the full dataset so at
least some holdout trades pass the filter.  You can bypass the search with
`--edge_threshold THR`, in which case the script simply reports the resulting
trades/week.  Filtered signals are saved.
Candidate trades are produced for bars beyond `--train_end_date` even if the
labels were masked.
*CLI* – `python train_ranker.py --run RUN --start_date YYYY‑MM‑DD --train_end_date YYYY‑MM‑DD --end_date YYYY‑MM‑DD [--edge_threshold THR]`
to `${RUN}/data/signals.csv` with columns `timestamp, edge_score, regime, sl_pips,
tp_pips, meta_prob, side`.

Full CLI/IO/table details for each script live in the canvas for quick
copy‑paste.

Example training workflow:

```bash
python prepare.py             # feature engineering
python label.py --run $RUN_ID # add labels
python train_base.py --run $RUN_ID
python train_meta.py --run $RUN_ID
```

### 6.6 Tuning the Ranker

Adjust trade volume in `config.yaml` under the `ranker` section:

```yaml
ranker:
  target_trades_per_week: 40
  min_trades_per_week: 25
  max_trades_per_week: 50
```

`train_ranker.py` uses these values when searching for the best
`edge_threshold`. You may override them with `--target_trades_per_week`,
`--min_trades_per_week` and `--max_trades_per_week`. `walkforward.py` accepts
the same options and forwards them to `train_ranker.py`, letting you test
different trade volumes across the entire loop. To experiment manually,
pass a threshold directly:

```bash
python train_ranker.py --run $RUN_ID --edge_threshold 0.018
```

You can also run the entire walk-forward loop with higher trade targets:

```bash
python walkforward.py --run OUTDIR \
  --target_trades_per_week 60 \
  --min_trades_per_week 50 \
  --max_trades_per_week 70
```

The script prints `trades/wk(train)` and `trades/wk(hold)` and writes the values
to `${RUN}/models/edge_threshold.json`. After tweaking the settings, rerun
`train_ranker.py` then `simulate.py` and inspect the reported trade counts to
decide on further adjustments.

---

## 7. Run‑ID & Directory Convention

```bash
RUN_ID=models/run_$(date +%Y%m%d_%H%M%S)
export RUN_ID
```

All scripts either read `$RUN_ID` or accept `--run`. No artefact ever overwrites another run.

---

## 8. Stacked‑Model Roadmap (v0)

1. **train\_base.py** – LGBM (× regime, side) + 1‑D CNN; write `probs_base.csv`.
2. **train\_meta.py** – logistic GBM on stacked probabilities + rules.
3. **train\_ranker.py** – compute edge scores, auto‑threshold to 25‑50 trades/wk.
4. **simulate.py** – sort by edge, enforce cooldown & equity risk sizing.
5. **walkforward.py** – weekly retrain + sim loop (18‑month train window, config‑driven stepback); log KPIs to `history.csv`.

---

## 9. Testing & CI Ladder

| Stage      | pytest suite                  | Gate                      |
| ---------- | ----------------------------- | ------------------------- |
| Prepare    | schema + NaN checks           | fail if `nan_pct > 0.1 %` |
| Label      | weekly signal count           | fail if `< 5` signals/wk  |
| Train‑Base | track precision (target ≥ 0.70) | warn if recall < 0.05     |
| Meta       | ROC‑AUC ≥ 0.80                | hard fail                 |
| Simulate   | track win rate, RR, trades/wk | fail if win rate < 58 % (goal: RR 2–3, 25–50 trades/wk) |

---

## 10. File Purpose Map

* `prepare.py` → feature engineering, clustering, scaling → **prepared.csv**
* `label.py`   → add future price cols + SL/TP grid → **labeled.csv**
* `train_base.py` → LGBM + CNN → per‑bar base probabilities
* `train_meta.py` → meta‑learner → `meta.pkl`
* `train_ranker.py` → edge scores + threshold → **signals.csv**
* `simulate.py` → equity‑aware back‑test → KPIs
* `walkforward.py` → orchestrates weekly retrain + sim
*CLI* – `python simulate.py --run RUN --start_date YYYY‑MM‑DD --end_date YYYY‑MM‑DD [--test_frac 0.2]`
*CLI* – `python walkforward.py --run OUTDIR [--stepback_weeks N]`
  `[--target_trades_per_week N] [--min_trades_per_week N] [--max_trades_per_week N]`
  * if `--window_weeks 1`, the script forwards `--test_frac 1.0` to `simulate.py`
  * internally calls `prepare.py` and `label.py` with a window‑specific `--train_end_date`
* `trade.py` (future) → live connector to MT4 via DWX
* `utils.py` → shared helpers (price math, directories, logging)

---

## 11. Trade Logic

> **Trade = High‑confidence Directional Signal + Best SL/TP from grid**

1. `label.py` builds the SL/TP grid from `sl_tp_grid` in `config.yaml`, then labels each bar while honouring `label.cooldown_min` and the optional
   `--train_end_date` cutoff. When `--strict` is supplied any CSV parsing errors abort instead of dropping rows.
   • SL = spread × SL multiplier (1.8 → 5.2)
   • TP = SL × TP multiplier (2.0 → 3.2)
2. `train_base.py` learns to predict high‑precision direction (Class 1) **per regime & side**.
3. `train_meta.py` combines multiple base probabilities; `train_ranker.py` converts this into an **edge score** and picks trades until the weekly cap hits.
4. `sltp.py` (to‑be‑replaced by `train_ranker.py`) trains on `labeled.csv` to predict the SL/TP bucket most likely to hit TP first. `walkforward.py` calls `label.py` with a window-specific `--train_end_date` so this model sees only historical data.
5. `simulate.py` merges signals with market data via a **left join** so every bar is processed. The simulation loops over all bars, opening new trades when `has_signal` is true and concurrent positions < `simulation.max_positions`. It uses continuous price data for SL/TP checks and enforces **per-side** cooldown timers from config. The default `simulation.max_positions` is **1**, preventing overlapping trades.
6. Simulations with zero executed trades now report a profit factor of `0.0` instead of infinity.

---

## 12. Lessons Learned

### A. Performance Gaps

* Early iterations showed 70–80 % precision in the deprecated `train.py` pipeline; later refactors lost that edge.
* SL/TP modelling adds noise; only useful **after** direction is rock‑solid.

### B. What Matters in Class 1

* False positives are far worse than missing positives.
* It’s better to skip a great trade than enter a bad one.

### C. Architecture Wins

* Per‑regime modelling outperforms global models.
* Separating direction and SL/TP keeps modules interpretable.
* Grid‑based labelling is transparent and debuggable.

### D. Failures to Avoid

* Rewriting core logic because of temporary metric dips.
* Letting model quantity outweigh signal quality.
* Ignoring lessons on imbalance and temporal leakage.

---

## 13. What We Abandoned

* PU‑learning approach (helped imbalance, but opaque).
* Joint direction + SL/TP modelling (tangled logic, poor interpretability).
* ROC‑first thresholding (precision‑first works better for us).

---

## 14. Example Cursor Prompt Format

Short prompt:

```python
# cursor prompt: generate train_ranker.py
"""
Create train_ranker.py that loads probs_base.csv + labeled.csv, trains a logistic
GBM meta‑model (features = base probs + ATR_pct + hour + regime), computes
edge_score and derives threshold for 25–50 trades/week. Save edge_threshold.json.
"""
```

Verbose iterative prompt (excerpt):

```python
"""
# === ITER 70.1.7 – align tests to PF=4.0 and add avg_rr metric ============
# Context: focusing on risk‑reward alongside profit factor.
# -------------------------------------------------------------------------
# 1. simulate.py
#    • After computing win_rate add:
#        avg_rr = df["tp_points"].mean() / df["sl_points"].mean()
# -------------------------------------------------------------------------
"""
```

Use whichever length suits the change.

---

## 15. Notes

* Fixed spread: **0.00013**.
* Spread is the only trading cost; no commission or slippage.
* SL multipliers = 1.8 → 5.2 (step 0.2).  TP multipliers = 2.0 → 3.2 (step 0.2).
* All scripts pull grid & parameters from **config.yaml** for consistency.
* The CNN's `sequence_length`, `batch_size`, `epochs` and `patience` are
  configurable under the **signal model parameters** section.
* `train_base.py` now records `cv_avg_precision` from time-series cross‑validation.
* `train_base.py` and `train_meta.py` expose a `--test_split` option to adjust the validation ratio.
* Walk-forward defaults: 18 month training window and 1 week simulation step (see `config.yaml`).
* `signal.min_signals_per_week` guards against weeks with too few candidate signals (default **8**).
* Latest defaults tweak signal gating:
  * `label.threshold` now **0.0010** and `max_sl_pips` **22**
  * `label.cooldown_min` and `simulation.cooldown_min` now **10 min**
  * Sample run: baseline config → **73 trades** (~172/wk); tuned config → **97 trades** (~227/wk).
* `optimize.py` now only tunes ranker trade-per-week settings. Set the regime cluster count via `prepare.n_clusters` in `config.yaml`.

Sample grid snippet:

```python
DEFAULT_GRID = {
    # ...
    "ranker.target_trades_per_week": [25, 35, 45],
    "ranker.min_trades_per_week": [20, 25, 30],
    "ranker.max_trades_per_week": [45, 50, 55],
}
```

---

## 16. Current Iteration & Next Steps

1. **Prep / Label** – finalise ATR & cooldown gates; vectorise price‑math util shared by all modules. *(in progress)*
2. **train\_base.py** – implement LGBM training + probability dump; build minimal CNN baseline. *(done)*
3. **train\_meta.py** – logistic GBM stacked learner; confirm ROC‑AUC ≥ 0.80. *(done)*
4. **train\_ranker.py** – edge‑score + dynamic threshold for 25‑50 trades/wk. *(done)*
5. **simulate.py** – patch overlapping‑trade bug; add equity‑risk sizing. *(done)*
6. **walkforward.py** – weekly loop; append KPIs to history. *(done)*
7. **CI** – integrate pytest suites & acceptance gates listed above. *(done)*
8. **optimize.py** – optional grid search for ranker settings. Clusters are fixed via `prepare.n_clusters`. *(new)*

*This document is the single source of truth – update it after every major decision or refactor.*
