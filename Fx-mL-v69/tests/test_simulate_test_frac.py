import os
import sys

import pandas as pd
import simulate


def run_with_args(module, args):
    old = sys.argv
    sys.argv = [module.__file__] + args
    try:
        module.main()
    except SystemExit:
        pass
    finally:
        sys.argv = old


def test_simulate_test_frac_1(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    os.environ["RUN_ID"] = str(run_dir)

    times = pd.date_range("2021-01-01", periods=3, freq="5min")
    base_df = pd.DataFrame({"close": 1.0, "high": 1.0, "low": 1.0}, index=times)
    monkeypatch.setattr(simulate, "load_data", lambda *a, **k: base_df)

    signals = pd.DataFrame({"side": ["long"]*3, "sl_pips": [10]*3, "tp_pips": [20]*3}, index=times)
    monkeypatch.setattr(simulate, "load_signals", lambda *a, **k: signals)

    def dummy_prepare_trade_df(prepared, sig):
        df = prepared.join(sig, how="left")
        df["timestamp"] = df.index
        df["price"] = df["close"]
        df["high"] = df["high"]
        df["low"] = df["low"]
        df["side"] = "long"
        df["sl_points"] = 1.0
        df["tp_points"] = 2.0
        df["has_signal"] = True
        return df[["timestamp", "price", "high", "low", "side", "sl_points", "tp_points", "has_signal"]]

    monkeypatch.setattr(simulate, "prepare_trade_df", dummy_prepare_trade_df)

    def dummy_simulate_df(df, risk_pct=0.02, return_trades=False, **_):
        metrics = {
            "total_trades": len(df),
            "win_rate": 0.5,
            "profit_factor": 1.0,
            "avg_rr": 2.0,
            "max_dd": 0.1,
            "sharpe": 0.0,
            "trades_per_wk": 1,
        }
        trades = df if return_trades else None
        return [1.0]*len(df), metrics, trades

    monkeypatch.setattr(simulate, "simulate_df", dummy_simulate_df)

    run_with_args(simulate, ["--run", str(run_dir), "--test_frac", "1.0"])
    out = capsys.readouterr().out
    assert "Test Simulation Results" in out
    assert "Full Simulation Results" in out
