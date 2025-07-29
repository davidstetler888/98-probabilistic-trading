import os
import sys
import pandas as pd
from pathlib import Path
import prepare
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


def test_dynamic_n_clusters(tmp_path, monkeypatch):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    create_sample_csv(input_dir / "EURUSD5.csv", rows=1000)

    run_dir = tmp_path / "run"
    os.environ["RUN_ID"] = str(run_dir)

    monkeypatch.setitem(config._config["data"], "input_dir", str(input_dir))
    monkeypatch.setitem(config._config["prepare"], "n_clusters", "auto")

    run_with_args(prepare, ["--strict"])

    df = pd.read_csv(run_dir / "data" / "prepared.csv", index_col=0)
    assert df["market_regime"].nunique() <= 3


def test_fixed_n_clusters(tmp_path, monkeypatch):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    create_sample_csv(input_dir / "EURUSD5.csv", rows=1000)

    run_dir = tmp_path / "run"
    os.environ["RUN_ID"] = str(run_dir)

    monkeypatch.setitem(config._config["data"], "input_dir", str(input_dir))
    monkeypatch.setitem(config._config["prepare"], "n_clusters", 3)

    run_with_args(prepare, ["--strict"])

    df = pd.read_csv(run_dir / "data" / "prepared.csv", index_col=0)
    assert df["market_regime"].nunique() == 3
