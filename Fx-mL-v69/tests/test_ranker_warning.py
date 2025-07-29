import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

import train_ranker
import simulate
from config import config
from contextlib import nullcontext


def run_with_args(module, args):
    old = sys.argv
    sys.argv = [module.__file__] + args
    try:
        module.main()
    except SystemExit:
        pass
    finally:
        sys.argv = old


class DummyModel:
    def predict(self, X):
        return np.full(len(X), 0.5)


@pytest.mark.parametrize(
    "tw_train, tw_hold, expect",
    [
        (1.0, 1.0, True),
        (9.0, 1.0, False),
        (1.0, 9.0, False),
    ],
)
def test_warning_low_trades(monkeypatch, tmp_path, tw_train, tw_hold, expect):
    run_dir = tmp_path / "run"
    os.environ["RUN_ID"] = str(run_dir)

    tz = config.get("market.timezone")
    times = pd.date_range("2020-01-01", periods=2, freq="5min", tz=tz)
    labeled = pd.DataFrame(
        {
            "label_long": [1, 1],
            "label_short": [0, 0],
            "market_regime": [0, 0],
            "hour": [0, 0],
            "atr_pct": [0.1, 0.1],
        },
        index=times,
    )
    probs = pd.DataFrame({"prob_cnn": [0.5, 0.5]}, index=times)

    monkeypatch.setattr(train_ranker, "load_data", lambda path, *a, **k: labeled if "labeled" in path else probs)
    monkeypatch.setattr(train_ranker, "load_meta_models", lambda *_: {"long": DummyModel()})
    monkeypatch.setattr(train_ranker, "load_sltp_models", lambda *_: {})
    monkeypatch.setattr(
        train_ranker,
        "find_threshold_consensus",
        lambda *a, **k: (0.5, tw_train, tw_hold, 0.5, 0.5, 2.0, 2.0),
    )
    monkeypatch.setattr(train_ranker, "ensure_dir", lambda p: Path(p).mkdir(parents=True, exist_ok=True))

    ctx = pytest.warns(UserWarning) if expect else nullcontext()
    with ctx:
        run_with_args(
            train_ranker,
            ["--run", str(run_dir), "--start_date", "2020-01-01", "--train_end_date", "2020-01-01", "--end_date", "2020-01-02"],
        )


def test_profit_factor_finite_with_no_trades():
    tz = config.get("market.timezone")
    times = pd.date_range("2020-01-01", periods=3, freq="5min", tz=tz)
    df = pd.DataFrame(
        {
            "timestamp": times,
            "price": [1.0] * 3,
            "high": [1.0] * 3,
            "low": [1.0] * 3,
            "side": [np.nan] * 3,
            "sl_points": [1.0] * 3,
            "tp_points": [2.0] * 3,
            "has_signal": [False] * 3,
        }
    )

    _, metrics = simulate.simulate_df(df, cooldown_min=0, max_positions=1)
    assert np.isfinite(metrics["profit_factor"])


def test_find_threshold_relaxes(monkeypatch):
    tz = config.get("market.timezone")
    idx = pd.date_range("2020-01-01", periods=3, freq="3D", tz=tz)
    scores = pd.Series([0.1, 0.2, 0.3], index=idx)
    wins = pd.Series([0.5, 0.5, 0.5], index=idx)
    rr = pd.Series([2.0, 2.0, 2.0], index=idx)

    with pytest.warns(UserWarning):
        train_ranker.find_threshold(
            scores,
            wins,
            rr,
            target=6.0,
            min_per_week=5.0,
            max_per_week=10.0,
            win_range=(0.0, 1.0),
            rr_range=(0.0, 3.0),
        )
