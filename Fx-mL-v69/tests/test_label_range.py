import os
import sys
from pathlib import Path
import pandas as pd
import pytest

from config import config
from tests.conftest import create_sample_csv


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


def test_label_date_slice(tmp_path, monkeypatch):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    create_sample_csv(input_dir / "EURUSD5.csv", rows=3500)

    run_dir = tmp_path / "run"
    os.environ["RUN_ID"] = str(run_dir)

    monkeypatch.setitem(config._config["data"], "input_dir", str(input_dir))
    monkeypatch.setitem(config._config["label"], "threshold", 0.0001)
    monkeypatch.setitem(config._config["label"], "max_sl_pips", 50)
    monkeypatch.setitem(config._config["label"], "min_rr", 1.0)

    acc = config._config.setdefault("acceptance", {})
    acc.setdefault("prepare", {}).update({"min_rows": 10, "max_nan_percent": 100})
    acc.setdefault("train", {}).update({"min_precision": 0})
    acc.setdefault("sltp", {}).update({"min_rr": 0})
    acc.setdefault("simulation", {}).update(
        {
            "min_win_rate": config.get("acceptance.simulation.min_win_rate"),
            "min_profit_factor": 0,
            "max_drawdown": 1,
            "min_trades_per_week": 0,
        }
    )
    monkeypatch.setitem(config._config["signal"], "epochs", 1)
    monkeypatch.setitem(config._config["signal"], "batch_size", 32)
    monkeypatch.setitem(config._config["signal"], "patience", 1)
    monkeypatch.setitem(config._config["signal"], "sequence_length", 16)
    import sltp

    monkeypatch.setattr(sltp, "MIN_SAMPLES", 10)

    import prepare
    import label

    run_with_args(prepare, ["--strict"])
    run_with_args(
        label,
        [
            "--run",
            str(run_dir),
            "--start_date",
            "2021-01-02",
            "--end_date",
            "2021-01-12",
        ],
    )

    df = pd.read_csv(run_dir / "data" / "labeled.csv", index_col=0, parse_dates=True)
    tz = config.get("market.timezone")
    assert df.index.min() >= pd.Timestamp("2021-01-02", tz=tz)
    assert df.index.max() >= pd.Timestamp("2021-01-10", tz=tz)
    assert df.index.max() <= pd.Timestamp("2021-01-12 23:59:59", tz=tz)


def test_labels_respect_cutoff(sample_run):
    run_dir = sample_run
    import prepare
    import label

    run_with_args(prepare, ["--strict"])
    run_with_args(
        label,
        [
            "--run",
            str(run_dir),
            "--start_date",
            "2021-01-01",
            "--train_end_date",
            "2021-01-02",
            "--end_date",
            "2021-01-03",
        ],
    )

    df = pd.read_csv(run_dir / "data" / "labeled.csv", index_col=0, parse_dates=True)
    tz = config.get("market.timezone")
    cutoff = pd.Timestamp("2021-01-02", tz=tz)
    after = df[df.index >= cutoff]
    assert (after["label_long"] == -1).all()
    assert (after["label_short"] == -1).all()
