import pandas as pd
import label
from config import config


def test_label_cooldown(monkeypatch):
    times = pd.date_range("2021-01-01", periods=3, freq="5min")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.001, 1.001, 1.001],
            "low": [0.999, 0.999, 0.999],
            "close": [1.0, 1.0, 1.0],
        },
        index=times,
    )

    monkeypatch.setitem(config._config["label"], "threshold", 0.0)
    monkeypatch.setitem(config._config["label"], "max_sl_pips", 100)
    monkeypatch.setitem(config._config["label"], "min_rr", 0.9)
    monkeypatch.setitem(config._config["label"], "cooldown_min", 10)

    labeled = label.label_trades(df, future_window=1)
    assert labeled.loc[times[0], "long_label"] == 1
    assert labeled.loc[times[1], "long_label"] == -1
