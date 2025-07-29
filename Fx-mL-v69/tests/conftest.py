import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from config import config


def create_sample_csv(path: Path, rows: int = 2200) -> None:
    start = pd.Timestamp("2021-01-01")
    times = [start + pd.Timedelta(minutes=5*i) for i in range(rows)]
    data = {
        "date": [t.strftime("%Y.%m.%d") for t in times],
        "time": [t.strftime("%H:%M") for t in times],
        "open": np.linspace(1.1, 1.2, rows),
        "high": np.linspace(1.1, 1.2, rows) + 0.0002,
        "low": np.linspace(1.1, 1.2, rows) - 0.0002,
        "close": np.linspace(1.1, 1.2, rows) + 0.0001,
        "volume": np.random.randint(50, 100, rows),
    }
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)


@pytest.fixture()
def sample_run(tmp_path, monkeypatch):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    create_sample_csv(input_dir / "EURUSD5.csv", rows=1000)

    run_dir = tmp_path / "run"
    os.environ["RUN_ID"] = str(run_dir)

    # patch config paths
    monkeypatch.setitem(config._config["data"], "input_dir", str(input_dir))

    # make labeling easy
    monkeypatch.setitem(config._config["label"], "threshold", 0.0001)
    monkeypatch.setitem(config._config["label"], "max_sl_pips", 50)
    monkeypatch.setitem(config._config["label"], "min_rr", 1.0)
    monkeypatch.setitem(config._config["label"], "cooldown_min", 0)

    # loosen acceptance thresholds for tests
    acc = config._config.setdefault("acceptance", {})
    acc.setdefault("prepare", {}).update({"min_rows": 10, "max_nan_percent": 100})
    acc.setdefault("train", {}).update({"min_precision": 0})
    acc.setdefault("sltp", {}).update({"min_rr": 0})
    acc.setdefault("simulation", {}).update({
        "min_win_rate": 0,
        "min_profit_factor": 0,
        "max_drawdown": 1,
        "min_trades_per_week": 0,
    })
    monkeypatch.setitem(config._config["signal"], "epochs", 1)
    monkeypatch.setitem(config._config["signal"], "batch_size", 32)
    monkeypatch.setitem(config._config["signal"], "patience", 1)
    monkeypatch.setitem(config._config["signal"], "sequence_length", 16)
    import sltp
    monkeypatch.setattr(sltp, "MIN_SAMPLES", 10)
    return Path(run_dir)
