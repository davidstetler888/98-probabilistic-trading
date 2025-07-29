import pandas as pd
import pytest
import pytz
from utils import (
    parse_start_date_arg,
    load_data,
    get_market_week_start,
    get_market_week_end,
)
from config import config


def test_parse_start_date_arg():
    assert parse_start_date_arg("2021-01-01") == "2021-01-01"
    assert parse_start_date_arg(None) is None
    with pytest.raises(ValueError):
        parse_start_date_arg("20210101")


def test_load_data_start_end(tmp_path):
    idx = pd.date_range("2021-01-01", periods=5, freq="D")
    df = pd.DataFrame({"val": range(5)}, index=idx)
    path = tmp_path / "df.csv"
    df.to_csv(path)
    loaded = load_data(str(path), end_date="2021-01-04", start_date="2021-01-02")
    tz = config.get("market.timezone")
    assert loaded.index.min() >= pd.Timestamp("2021-01-02", tz=tz)
    assert loaded.index.max() <= pd.Timestamp("2021-01-04", tz=tz)
    assert len(loaded) == 3


def test_get_market_week_start():
    tz = config.get("market.timezone")
    ts = pd.Timestamp("2021-01-08 17:00", tz=tz)
    week_start = get_market_week_start(ts)
    assert week_start == pd.Timestamp("2021-01-03 16:00", tz=tz)


def test_get_market_week_end():
    tz = config.get("market.timezone")
    ts = pd.Timestamp("2021-01-06 12:00", tz=tz)
    week_end = get_market_week_end(ts)
    assert week_end == pd.Timestamp("2021-01-08 16:00", tz=tz)


def test_load_data_non_datetime_index(tmp_path):
    path = tmp_path / "bad.csv"
    with open(path, "w") as f:
        f.write("val\n1\n2\n")
    loaded = load_data(str(path))
    tz = config.get("market.timezone")
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert loaded.index.tz == pytz.timezone(tz)


def test_load_data_parse_error(tmp_path):
    path = tmp_path / "bad_dates.csv"
    with open(path, "w") as f:
        f.write("idx,val\nfoo,1\nbar,2\n")
    with pytest.raises(ValueError):
        load_data(str(path))


def test_load_data_drop_bad_rows(tmp_path, capsys):
    path = tmp_path / "bad_rows.csv"
    with open(path, "w") as f:
        f.write("date,time,open,high,low,close,volume\n")
        f.write("2021.01.01,00:00,1,1,1,1,1\n")
        f.write("bad,00:05,1,1,1,1,1\n")
    loaded = load_data(str(path))
    out = capsys.readouterr().out
    assert len(loaded) == 1
    assert "1 rows with invalid date/time dropped" in out
    assert "bad,00:05" in out


def test_load_data_bad_index_preview(tmp_path, capsys):
    path = tmp_path / "bad_index.csv"
    with open(path, "w") as f:
        f.write("idx,val\nfoo,1\n2021-01-01,2\nbar,3\n")
    loaded = load_data(str(path))
    out = capsys.readouterr().out
    assert len(loaded) == 1
    assert "index values could not be parsed" in out
    assert "foo" in out


def test_load_data_strict(tmp_path):
    path = tmp_path / "strict.csv"
    with open(path, "w") as f:
        f.write("date,time,open,high,low,close,volume\n")
        f.write("bad,00:00,1,1,1,1,1\n")
    with pytest.raises(ValueError):
        load_data(str(path), strict=True)


def test_load_data_end_date_day_inclusive(tmp_path):
    idx = pd.DatetimeIndex(["2021-01-01 23:59", "2021-01-02 00:00", "2021-01-02 12:00"])
    df = pd.DataFrame({"val": [1, 2, 3]}, index=idx)
    path = tmp_path / "hf.csv"
    df.to_csv(path)
    tz = config.get("market.timezone")

    loaded = load_data(str(path), end_date="2021-01-01")
    assert len(loaded) == 1
    assert loaded.index.max() == pd.Timestamp("2021-01-01 23:59", tz=tz)

    loaded2 = load_data(str(path), end_date="2021-01-02")
    assert len(loaded2) == 3
    assert loaded2.index.max() == pd.Timestamp("2021-01-02 12:00", tz=tz)


def test_load_data_end_date_last_bar(tmp_path):
    """Verify ``end_date`` includes all 5-minute bars for that day."""
    idx = pd.date_range("2021-01-01", periods=600, freq="5min")
    df = pd.DataFrame(
        {
            "date": idx.strftime("%Y.%m.%d"),
            "time": idx.strftime("%H:%M"),
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1,
        }
    )
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    tz = config.get("market.timezone")

    loaded = load_data(str(path), end_date="2021-01-02")
    assert len(loaded) == 576
    assert loaded.index.max() == pd.Timestamp("2021-01-02 23:55", tz=tz)


def test_load_data_extra_cols(tmp_path, capsys):
    """CSV with extra feature columns should load without warnings."""
    data = {
        "date": ["2021.01.01", "2021.01.01"],
        "time": ["00:00", "00:05"],
        "open": [1.0, 1.0],
        "high": [1.0, 1.0],
        "low": [1.0, 1.0],
        "close": [1.0, 1.0],
        "volume": [1, 1],
        "feat": [10, 20],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "extra.csv"
    df.to_csv(path, index=False)
    loaded = load_data(str(path))
    out = capsys.readouterr().out
    assert len(loaded) == len(df)
    assert "Warning:" not in out
    assert "feat" in loaded.columns


def test_load_data_extra_cols_no_header(tmp_path, capsys):
    """Headerless CSV with extra columns should still load cleanly."""
    data = {
        "date": ["2021.01.01", "2021.01.01"],
        "time": ["00:00", "00:05"],
        "open": [1.0, 1.0],
        "high": [1.0, 1.0],
        "low": [1.0, 1.0],
        "close": [1.0, 1.0],
        "volume": [1, 1],
        "feat": [10, 20],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "extra_no_header.csv"
    df.to_csv(path, index=False, header=False)
    loaded = load_data(str(path))
    out = capsys.readouterr().out
    assert len(loaded) == len(df)
    assert "Warning:" not in out
    assert "extra_0" in loaded.columns


def test_load_data_recover_malformed_row(tmp_path):
    """Loader should recover timestamps with offsets and stray delimiters."""
    lines = [
        "date,time,open,high,low,close,volume",
        "2023.03.13,02:00,1,1,1,1,1",
        "2023-03-13 02:05:00-05:00;1;1;1;1;1;1",
        "2023.03.13,02:10,1,1,1,1,1",
    ]
    path = tmp_path / "messy.csv"
    path.write_text("\n".join(lines))

    df = load_data(str(path))
    assert len(df) == 3
    assert df.index.min() < df.index.max()
