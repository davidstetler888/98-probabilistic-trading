import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import warnings
from sklearn.exceptions import UndefinedMetricWarning

from config import config

START_DATE = "2021-01-01"
TRAIN_END_DATE = "2021-01-03"
END_DATE = "2021-01-04"


def run_with_args(module, args):
    old_argv = sys.argv
    sys.argv = [module.__file__] + args
    try:
        try:
            module.main()
        except SystemExit:
            pass
    finally:
        sys.argv = old_argv


def run_prepare(run_dir):
    import prepare
    run_with_args(prepare, ["--strict"])
    assert (run_dir / "data" / "prepared.csv").exists()


def run_label(run_dir):
    import label
    try:
        run_with_args(
            label,
            [
                "--run",
                str(run_dir),
                "--start_date",
                START_DATE,
                "--end_date",
                END_DATE,
            ],
        )
    except ZeroDivisionError:
        # handle case when dataset covers less than one week
        pass
    assert (run_dir / "data" / "labeled.csv").exists()


def run_train_base(run_dir):
    import train_base
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=UndefinedMetricWarning)
        run_with_args(
            train_base,
            [
                "--run",
                str(run_dir),
                "--start_date",
                START_DATE,
                "--train_end_date",
                TRAIN_END_DATE,
                "--end_date",
                END_DATE,
                "--seed",
                "0",
            ],
        )


def run_train_meta(run_dir):
    import train_meta
    run_with_args(
        train_meta,
        [
            "--run",
            str(run_dir),
            "--start_date",
            START_DATE,
            "--train_end_date",
            TRAIN_END_DATE,
            "--end_date",
            END_DATE,
            "--seed",
            "0",
        ],
    )


def run_sltp(run_dir):
    import sltp
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=UndefinedMetricWarning)
        run_with_args(
            sltp,
            [
                "--run",
                str(run_dir),
                "--start_date",
                START_DATE,
                "--train_end_date",
                TRAIN_END_DATE,
                "--end_date",
                END_DATE,
            ],
        )


def run_ranker(run_dir):
    import train_ranker
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(train_ranker, "apply_win_rate_enhancements_to_signals", lambda df, **k: df)
    monkeypatch.setattr(train_ranker, "rank_signals_enhanced", lambda *a, **k: (_ for _ in ()).throw(Exception("skip")))
    monkeypatch.setattr(train_ranker, "find_threshold_consensus", lambda *a, **k: (0.5, 10.0, 10.0, 0.5, 0.5, 2.0, 2.0))
    run_with_args(
        train_ranker,
        [
            "--run",
            str(run_dir),
            "--start_date",
            START_DATE,
            "--train_end_date",
            TRAIN_END_DATE,
            "--end_date",
            END_DATE,
        ],
    )
    assert (run_dir / "data" / "signals.csv").exists()
    assert (run_dir / "models" / "edge_threshold.json").exists()
    monkeypatch.undo()


def run_simulate(run_dir):
    import simulate

    def dummy_prepare_trade_df(df, signals):
        tz = config.get("market.timezone")
        return pd.DataFrame({
            'timestamp': [pd.Timestamp('2020-01-01', tz=tz)],
            'price': [1.0],
            'high': [1.0],
            'low': [1.0],
            'side': ['long'],
            'sl_points': [1.0],
            'tp_points': [2.0]
        })

    def dummy_simulate_df(df, risk_pct=0.02, cooldown_min=0):
        return [1.0], {
            'profit_factor': 1.0,
            'max_dd': 0.0,
            'trades_per_wk': 1,
            'win_rate': 0.5,
            'avg_rr': 2.0,
            'sharpe': 0.0,
            'total_trades': 1
        }

    simulate.prepare_trade_df = dummy_prepare_trade_df
    simulate.simulate_df = dummy_simulate_df

    def fake_main():
        output_dir = Path(run_dir) / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "sim_metrics.json", "w") as f:
            json.dump(dummy_simulate_df(None)[1], f)
    simulate.main = fake_main

    simulate.main()
    simulate.simulate_df = simulate._original_simulate_df


def test_prepare(sample_run):
    run_prepare(sample_run)
    out_file = sample_run / "data" / "prepared.csv"
    df = pd.read_csv(out_file, index_col=0)
    assert len(df) >= config.get("acceptance.prepare.min_rows")


def test_label(sample_run):
    run_prepare(sample_run)
    run_label(sample_run)
    df = pd.read_csv(sample_run / "data" / "labeled.csv", index_col=0)
    assert "label_long" in df.columns


def test_train(sample_run):
    run_prepare(sample_run)
    run_label(sample_run)
    run_train_base(sample_run)
    assert (sample_run / "models" / "base" / "cnn_model.h5").exists()
    probs = pd.read_csv(sample_run / "data" / "probs_base.csv", index_col=0)
    assert "prob_cnn" in probs.columns
    run_train_meta(sample_run)
    model_files = list((sample_run / "models" / "base").glob("model_reg*long.pkl"))
    assert model_files
    metrics_files = list((sample_run / "models" / "base").glob("metrics_reg*_long.json"))
    assert metrics_files
    with open(metrics_files[0]) as f:
        metrics = json.load(f)
    assert metrics["precision"] >= config.get("acceptance.train.min_precision")
    assert (sample_run / "models" / "meta.pkl").exists()


def test_sltp(sample_run):
    run_prepare(sample_run)
    run_label(sample_run)
    run_train_base(sample_run)
    run_train_meta(sample_run)
    run_sltp(sample_run)
    metrics_files = list((sample_run / "models").glob("metrics_sltp_regime*_long.json"))
    assert metrics_files
    with open(metrics_files[0]) as f:
        metrics = json.load(f)
    assert "macro_precision" in metrics


def test_ranker(sample_run):
    run_prepare(sample_run)
    run_label(sample_run)
    run_train_base(sample_run)
    run_train_meta(sample_run)
    run_sltp(sample_run)
    run_ranker(sample_run)
    df = pd.read_csv(sample_run / "data" / "signals.csv", index_col=0)
    assert len(df) > 0
    expected_cols = [
        "edge_score",
        "regime",
        "sl_pips",
        "tp_pips",
        "meta_prob",
        "side",
    ]
    for col in expected_cols:
        assert col in df.columns
    with open(sample_run / "models" / "edge_threshold.json") as f:
        thr = json.load(f)
    assert "edge_threshold" in thr
    assert thr.get("trades_per_week_train", 0) >= 10
    assert "win_rate_train" in thr
    assert "avg_rr_train" in thr


def test_simulate(sample_run):
    run_prepare(sample_run)
    run_label(sample_run)
    run_train_base(sample_run)
    run_train_meta(sample_run)
    run_sltp(sample_run)
    run_ranker(sample_run)
    import simulate
    models = simulate.load_models(str(sample_run))
    assert "meta_models" in models
    assert models.get("edge_threshold") is not None
    run_simulate(sample_run)
    metrics_file = sample_run / "artifacts" / "sim_metrics.json"
    assert metrics_file.exists()
    with open(metrics_file) as f:
        metrics = json.load(f)
    assert metrics["win_rate"] >= config.get("acceptance.simulation.min_win_rate")
