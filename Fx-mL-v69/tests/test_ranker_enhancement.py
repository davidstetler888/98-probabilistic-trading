import os
import pandas as pd
import numpy as np
from pathlib import Path

import train_ranker
from config import config
from tests.test_scripts import run_with_args

class DummyModel:
    def predict(self, X):
        return np.full(len(X), 0.5)


def test_win_rate_enhancements_reduce_signals(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    os.environ["RUN_ID"] = str(run_dir)

    tz = config.get("market.timezone")
    times = pd.date_range("2020-01-01", periods=4, freq="5min", tz=tz)
    labeled = pd.DataFrame(
        {
            "label_long": [1] * 4,
            "label_short": [0] * 4,
            "market_regime": [0] * 4,
            "hour": [0] * 4,
            "atr_pct": [0.1] * 4,
        },
        index=times,
    )
    probs = pd.DataFrame({"prob_cnn": [0.5] * 4}, index=times)

    monkeypatch.setattr(train_ranker, "load_data", lambda path, *a, **k: labeled if "labeled" in path else probs)
    monkeypatch.setattr(train_ranker, "load_meta_models", lambda *_: {"long": DummyModel(), "short": DummyModel()})
    monkeypatch.setattr(train_ranker, "load_sltp_models", lambda *_: {})
    monkeypatch.setattr(train_ranker, "ensure_dir", lambda p: Path(p).mkdir(parents=True, exist_ok=True))

    def dummy_enhance(df, target_trades_per_week=40):
        return df.iloc[: len(df) // 2]

    monkeypatch.setattr(train_ranker, "apply_win_rate_enhancements_to_signals", dummy_enhance)

    run_with_args(
        train_ranker,
        [
            "--run",
            str(run_dir),
            "--start_date",
            "2020-01-01",
            "--train_end_date",
            "2020-01-01",
            "--end_date",
            "2020-01-01",
            "--edge_threshold",
            "0",
        ],
    )

    df = pd.read_csv(run_dir / "data" / "signals.csv", index_col=0)
    assert len(df) == 4

