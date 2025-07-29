import pandas as pd
from pathlib import Path
import walkforward
from config import config


def test_load_raw_index_with_header(tmp_path, monkeypatch):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    times = [pd.Timestamp("2021-01-01") + pd.Timedelta(minutes=5 * i) for i in range(3)]
    df = pd.DataFrame(
        {
            "date": [t.strftime("%Y.%m.%d") for t in times],
            "time": [t.strftime("%H:%M") for t in times],
            "open": [1.1] * 3,
            "high": [1.1] * 3,
            "low": [1.1] * 3,
            "close": [1.1] * 3,
            "volume": [1] * 3,
        }
    )
    df.to_csv(input_dir / "EURUSD5.csv", index=False)
    monkeypatch.setitem(config._config["data"], "input_dir", str(input_dir))

    start_ts, end_ts = walkforward.load_raw_index()
    tz = config.get("market.timezone")
    expected_start = pd.Timestamp(f"{df['date'][0]} {df['time'][0]}").tz_localize(tz)
    expected_end = pd.Timestamp(f"{df['date'].iloc[-1]} {df['time'].iloc[-1]}").tz_localize(tz)
    assert start_ts == expected_start
    assert end_ts == expected_end
