import pandas as pd
import numpy as np

from config import config
from sltp import prepare_training_data
import sltp


def test_find_best_bucket_dynamic(monkeypatch):
    monkeypatch.setattr(sltp, "SL_TP_PAIRS", [(10, 20), (20, 40)])

    base = {
        "close": [1.0, 1.0, 1.0],
        "future_high": [1.005, 1.005, 1.005],
        "future_low": [0.995, 0.995, 0.995],
    }

    df_london = pd.DataFrame(
        base | {
            "atr": [0.001, 0.001, 0.001],
            "session_london": [1, 1, 1],
            "session_asian": [0, 0, 0],
            "session_ny": [0, 0, 0],
            "session_overlap": [0, 0, 0],
        }
    )

    bucket, tp, sl, hit = sltp.find_best_bucket(df_london, True)
    assert bucket == 1
    assert hit

    df_asian = df_london.copy()
    df_asian["session_london"] = 0
    df_asian["session_asian"] = 1

    bucket2, tp2, sl2, hit2 = sltp.find_best_bucket(df_asian, True)
    assert bucket2 == 0
    assert hit2


def test_prepare_training_data_respects_cutoff():
    tz = config.get("market.timezone")
    times = pd.date_range("2021-01-01", periods=50, freq="5min", tz=tz)

    data = {
        "close": np.full(50, 1.0),
        "future_high": np.full(50, 1.001),
        "future_low": np.full(50, 0.999),
        "future_close": np.full(50, 1.0),
        "atr": np.full(50, 0.001),
        "label_long": [1 if i in (10, 38) else 0 for i in range(50)],
        "label_short": np.zeros(50, dtype=int),
        "market_regime": np.zeros(50, dtype=int),
    }
    df_full = pd.DataFrame(data, index=times)
    df_signals = df_full[df_full["label_long"] == 1]

    train_end = times[40]
    train_df, test_df = prepare_training_data(df_signals, df_full, train_end)

    remaining_idx = pd.Index(train_df.index.tolist() + test_df.index.tolist())
    assert len(remaining_idx) == 1
    assert remaining_idx[0] == times[10]
