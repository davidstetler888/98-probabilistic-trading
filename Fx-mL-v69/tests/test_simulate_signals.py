import os
import sys
import pandas as pd
import simulate
from config import config


def test_load_signals_respects_range(tmp_path):
    run_dir = tmp_path / "run"
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True)
    tz = config.get("market.timezone")
    idx = pd.date_range("2021-01-01", periods=3, freq="D", tz=tz)
    df = pd.DataFrame({"side": ["long"]*3, "sl_pips": [10]*3, "tp_pips": [20]*3}, index=idx)
    df.to_csv(data_dir / "signals.csv")

    out = simulate.load_signals(str(run_dir), end_date="2021-01-02", start_date="2021-01-01")

    assert len(out) == 2
    assert out.index.min() >= pd.Timestamp("2021-01-01", tz=tz)
    assert out.index.max() <= pd.Timestamp("2021-01-02 23:59:59", tz=tz)


def test_test_simulation_full_df(monkeypatch):
    tz = config.get("market.timezone")
    times = pd.date_range("2021-01-01", periods=3, freq="H", tz=tz)
    df = pd.DataFrame(
        {
            "timestamp": times,
            "price": [1.0]*3,
            "high": [1.0]*3,
            "low": [1.0]*3,
            "side": ["long"]*3,
            "sl_points": [1.0]*3,
            "tp_points": [2.0]*3,
        }
    )

    captured = {}

    def dummy_simulate_df(data, risk_pct=0.02, return_trades=True):
        captured["df"] = data.copy()
        metrics = {
            "total_trades": len(data),
            "win_rate": 0.5,
            "profit_factor": 1.0,
            "avg_rr": 2.0,
            "max_dd": 0.0,
            "sharpe": 0.0,
            "trades_per_wk": 1,
        }
        return [1.0]*len(data), metrics, data

    monkeypatch.setitem(simulate.__dict__, "simulate_df", dummy_simulate_df)
    monkeypatch.setattr(simulate, "_current_sim_fn", dummy_simulate_df)
    monkeypatch.setattr(simulate, "compute_weekly_summary", lambda df: pd.DataFrame())

    simulate.test_simulation(df, test_frac=1.0)
    assert captured["df"].equals(df)


def test_main_passes_same_df(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    os.environ["RUN_ID"] = str(run_dir)
    run_dir.mkdir()

    tz = config.get("market.timezone")
    times = pd.date_range("2021-01-01", periods=2, freq="H", tz=tz)
    base_df = pd.DataFrame({"close": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0]}, index=times)
    sig_df = pd.DataFrame({"side": ["long", "long"], "sl_pips": [10, 10], "tp_pips": [20, 20]}, index=times)

    monkeypatch.setattr(simulate, "load_data", lambda *a, **k: base_df)
    monkeypatch.setattr(simulate, "load_signals", lambda *a, **k: sig_df)

    def dummy_prepare(prepared, signals):
        df = prepared.join(signals, how="left")
        df["timestamp"] = df.index
        df["price"] = df["close"]
        df["sl_points"] = 1.0
        df["tp_points"] = 2.0
        df["side"] = "long"
        df["has_signal"] = True
        return df[["timestamp", "price", "high", "low", "side", "sl_points", "tp_points", "has_signal"]]

    monkeypatch.setattr(simulate, "prepare_trade_df", dummy_prepare)

    captured = {}
    def fake_test_sim(df, *a, **k):
        captured["df"] = df.copy()
        return {}
    monkeypatch.setattr(simulate, "test_simulation", fake_test_sim)

    monkeypatch.setitem(
        simulate.__dict__,
        "simulate_df",
        lambda df, **k: (
            [1.0] * len(df),
            {
                "total_trades": len(df),
                "win_rate": 0.5,
                "profit_factor": 1.0,
                "avg_rr": 2.0,
                "max_dd": 0.0,
                "sharpe": 0.0,
                "trades_per_wk": 1,
            },
            df,
        ),
    )
    monkeypatch.setattr(simulate, "_current_sim_fn", simulate.simulate_df)
    monkeypatch.setattr(simulate, "compute_weekly_summary", lambda df: pd.DataFrame())

    old = sys.argv
    sys.argv = [simulate.__file__, "--run", str(run_dir)]
    try:
        simulate.main()
    finally:
        sys.argv = old

    expected = dummy_prepare(base_df, sig_df)
    assert captured["df"].equals(expected)

