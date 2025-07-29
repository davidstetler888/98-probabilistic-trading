import os
import sys
from pathlib import Path
import pandas as pd
import prepare
import label
import train_base
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from tests.conftest import create_sample_csv
from config import config


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


def test_train_base_date_range(tmp_path, monkeypatch):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    create_sample_csv(input_dir / "EURUSD5.csv", rows=1500)

    run_dir = tmp_path / "run"
    os.environ["RUN_ID"] = str(run_dir)

    monkeypatch.setitem(config._config["data"], "input_dir", str(input_dir))
    monkeypatch.setitem(config._config["signal"], "epochs", 1)
    monkeypatch.setitem(config._config["signal"], "batch_size", 32)
    monkeypatch.setitem(config._config["signal"], "patience", 1)

    run_with_args(prepare, ["--strict"])
    run_with_args(
        label,
        [
            "--run",
            str(run_dir),
            "--start_date",
            "2021-01-01",
            "--end_date",
            "2021-01-10",
        ],
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=UndefinedMetricWarning)
        run_with_args(
            train_base,
            [
                "--run",
                str(run_dir),
                "--start_date",
                "2021-01-01",
                "--train_end_date",
                "2021-01-05",
                "--end_date",
                "2021-01-10",
                "--seed",
                "0",
            ],
        )

    probs = pd.read_csv(Path(run_dir) / "data" / "probs_base.csv", index_col=0, parse_dates=True)
    tz = config.get("market.timezone")
    assert probs.index.min() >= pd.Timestamp("2021-01-01", tz=tz)
    assert probs.index.max() <= pd.Timestamp("2021-01-10", tz=tz)
    assert (probs.index > pd.Timestamp("2021-01-05", tz=tz)).any()
    assert probs.loc[probs.index > pd.Timestamp("2021-01-05", tz=tz)].notna().any().any()
