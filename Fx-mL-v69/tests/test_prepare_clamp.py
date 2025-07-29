import os
import sys
import pandas as pd

from config import config
from tests.conftest import create_sample_csv


def run_prepare_with_args(args):
    import prepare
    old = sys.argv
    sys.argv = [prepare.__file__] + args
    try:
        prepare.main()
    except SystemExit:
        pass
    finally:
        sys.argv = old


def test_prepare_start_date_clamped(tmp_path, monkeypatch):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    create_sample_csv(input_dir / "EURUSD5.csv", rows=1500)

    run_dir = tmp_path / "run"
    os.environ["RUN_ID"] = str(run_dir)

    monkeypatch.setitem(config._config["data"], "input_dir", str(input_dir))

    run_prepare_with_args([
        "--start_date",
        "2020-12-25",
        "--end_date",
        "2021-01-05",
        "--strict",
    ])

    df = pd.read_csv(run_dir / "data" / "prepared.csv", index_col=0, parse_dates=True)
    tz = config.get("market.timezone")
    assert df.index.min() >= pd.Timestamp("2021-01-01", tz=tz)


def test_no_negative_shift_features(tmp_path):
    from pathlib import Path
    import prepare

    csv_path = tmp_path / "sample.csv"
    create_sample_csv(csv_path, rows=100)

    df = pd.read_csv(csv_path)
    df.index = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.drop(columns=["date", "time"])

    df = prepare.calculate_indicators_and_lags(df)
    # Negative shifts would create NaNs at the end of the dataframe.
    assert not df.tail(1).isna().any().any()

