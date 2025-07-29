import sys
import json
from pathlib import Path
import os
import pandas as pd
import walkforward
from config import config
import importlib
import config as config_module


def test_default_acceptance_win_rate():
    """Ensure the default simulation win-rate gate matches the project goal."""
    # Reload config to avoid mutations from other tests
    importlib.reload(config_module)
    fresh = config_module.config
    assert fresh.get("acceptance.simulation.min_win_rate") == 0.58


def test_run_script_streams_large_output(tmp_path, monkeypatch):
    script = tmp_path / "spam.py"
    script.write_text("for i in range(1000):\n    print(i)")

    class Recorder:
        def __init__(self):
            self.calls = []

        def write(self, data):
            self.calls.append(data)

        def flush(self):
            pass

    rec = Recorder()
    monkeypatch.setattr(walkforward.sys, "stdout", rec)
    env = os.environ.copy()
    env["RUN_ID"] = str(tmp_path / "run")
    walkforward.run_script(str(script), [], env)
    assert len(rec.calls) > 1
    assert rec.calls[0].strip() == "0"
    assert rec.calls[-1].strip() == "999"
    log_file = Path(env["RUN_ID"]) / "logs" / "spam.log"
    assert log_file.exists()
    lines = log_file.read_text().splitlines()
    assert lines[0] == "0"
    assert lines[-1] == "999"


def test_walkforward_single_iteration(sample_run, monkeypatch, capsys):
    run_dir = sample_run
    # Patch run_script to avoid heavy training
    def dummy_run_script(script, args, env):
        rd = Path(env["RUN_ID"])
        (rd / "artifacts").mkdir(parents=True, exist_ok=True)
        with open(rd / "artifacts" / "sim_metrics.json", "w") as f:
            json.dump({"win_rate": 0.5, "profit_factor": 1.0, "trades_per_wk": 1}, f)
    monkeypatch.setattr(walkforward, "run_script", dummy_run_script)

    # Patch load_raw_index to match fixture range
    monkeypatch.setattr(walkforward, "load_raw_index", lambda: (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-04")))

    args = [
        walkforward.__file__,
        "--run",
        str(run_dir),
        "--train_window_months",
        "0",
        "--window_weeks",
        "1",
        "--stepback_weeks",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", args)
    walkforward.main()

    history = Path(run_dir) / "history.csv"
    assert not history.exists()
    captured = capsys.readouterr()
    assert "before available data" in captured.out


def test_walkforward_multiple_iterations(sample_run, monkeypatch):
    run_dir = sample_run

    run_ids = []

    def dummy_run_script(script, args, env):
        rd = Path(env["RUN_ID"])
        (rd / "artifacts").mkdir(parents=True, exist_ok=True)
        with open(rd / "artifacts" / "sim_metrics.json", "w") as f:
            json.dump({"win_rate": 0.5, "profit_factor": 1.0, "trades_per_wk": 1}, f)
        if script == "simulate.py":
            run_ids.append(env["RUN_ID"])

    monkeypatch.setattr(walkforward, "run_script", dummy_run_script)

    monkeypatch.setattr(
        walkforward,
        "load_raw_index",
        lambda: (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-20")),
    )

    args = [
        walkforward.__file__,
        "--run",
        str(run_dir),
        "--train_window_months",
        "0",
        "--window_weeks",
        "1",
        "--stepback_weeks",
        "2",
    ]
    monkeypatch.setattr(sys, "argv", args)
    walkforward.main()

    history = Path(run_dir) / "history.csv"
    assert history.exists()
    df = pd.read_csv(history)
    assert len(df) >= 2

    dates = sorted(pd.to_datetime(Path(p).name) for p in run_ids)
    assert len(dates) >= 2
    for d1, d2 in zip(dates, dates[1:]):
        assert (d2 - d1).days == 7


def test_walkforward_optimize(sample_run, monkeypatch, capsys):
    run_dir = sample_run

    def dummy_run_script(script, args, env):
        rd = Path(env["RUN_ID"])
        (rd / "artifacts").mkdir(parents=True, exist_ok=True)
        with open(rd / "artifacts" / "sim_metrics.json", "w") as f:
            json.dump({"win_rate": 0.5, "profit_factor": 1.0, "trades_per_wk": 1}, f)

    monkeypatch.setattr(walkforward, "run_script", dummy_run_script)
    monkeypatch.setattr(
        walkforward,
        "load_raw_index",
        lambda: (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-04")),
    )

    import optimize

    monkeypatch.setattr(
        optimize,
        "search_configs",
        lambda *args, **kwargs: ({"label.threshold": 0.0001}, {"win_rate": 0.6, "profit_factor": 1.1, "trades_per_wk": 1}),
    )

    args = [
        walkforward.__file__,
        "--run",
        str(run_dir),
        "--train_window_months",
        "0",
        "--window_weeks",
        "1",
        "--stepback_weeks",
        "1",
        "--optimize",
    ]
    monkeypatch.setattr(sys, "argv", args)
    walkforward.main()

    history = Path(run_dir) / "history.csv"
    assert not history.exists()
    captured = capsys.readouterr()
    assert "before available data" in captured.out


def test_walkforward_optimize_message(sample_run, monkeypatch, capsys):
    run_dir = sample_run

    def dummy_run_script(script, args, env):
        rd = Path(env["RUN_ID"])
        (rd / "artifacts").mkdir(parents=True, exist_ok=True)
        with open(rd / "artifacts" / "sim_metrics.json", "w") as f:
            json.dump({"win_rate": 0.5, "profit_factor": 1.0, "trades_per_wk": 1}, f)

    monkeypatch.setattr(walkforward, "run_script", dummy_run_script)
    monkeypatch.setattr(
        walkforward,
        "load_raw_index",
        lambda: (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-04")),
    )

    import optimize

    monkeypatch.setattr(
        optimize,
        "search_configs",
        lambda *args, **kwargs: ({"label.threshold": 0.0001}, {"win_rate": 0.6, "profit_factor": 1.1, "trades_per_wk": 1}),
    )

    args = [
        walkforward.__file__,
        "--run",
        str(run_dir),
        "--train_window_months",
        "0",
        "--window_weeks",
        "1",
        "--stepback_weeks",
        "1",
        "--optimize",
    ]
    monkeypatch.setattr(sys, "argv", args)
    walkforward.main()

    captured = capsys.readouterr()
    assert "Optimization enabled: searching configs (full)" in captured.out

def test_walkforward_optimize_applies_config(sample_run, monkeypatch, capsys):
    run_dir = sample_run
    recorded = []

    def dummy_run_script(script, args, env):
        recorded.append(config.get("label.threshold"))
        rd = Path(env["RUN_ID"])
        (rd / "artifacts").mkdir(parents=True, exist_ok=True)
        with open(rd / "artifacts" / "sim_metrics.json", "w") as f:
            json.dump({"win_rate": 0.5, "profit_factor": 1.0, "trades_per_wk": 1}, f)

    monkeypatch.setattr(walkforward, "run_script", dummy_run_script)
    monkeypatch.setattr(
        walkforward,
        "load_raw_index",
        lambda: (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-04")),
    )

    import optimize

    dummy_conf = {"label.threshold": 0.0002}
    dummy_metrics = {"win_rate": 0.6, "profit_factor": 1.1, "trades_per_wk": 1}
    monkeypatch.setattr(optimize, "search_configs", lambda *a, **k: (dummy_conf, dummy_metrics))

    args = [
        walkforward.__file__,
        "--run",
        str(run_dir),
        "--train_window_months",
        "0",
        "--window_weeks",
        "1",
        "--stepback_weeks",
        "1",
        "--optimize",
    ]
    monkeypatch.setattr(sys, "argv", args)
    walkforward.main()

    best_files = list(run_dir.glob("*/best_config.json"))
    assert not best_files
    history = Path(run_dir) / "history.csv"
    assert not history.exists()
    assert recorded == []
    captured = capsys.readouterr()
    assert "before available data" in captured.out


def test_walkforward_warns_when_window_exceeds_raw(sample_run, monkeypatch, capsys):
    run_dir = sample_run
    calls = []

    def dummy_run_script(script, args, env):
        calls.append((script, args))
        rd = Path(env["RUN_ID"])
        if script == "prepare.py":
            (rd / "data").mkdir(parents=True, exist_ok=True)
            tz = config.get("market.timezone")
            idx = pd.date_range("2021-01-24", periods=4, freq="D", tz=tz)
            pd.DataFrame(index=idx).to_csv(rd / "data" / "prepared.csv")
        elif script == "simulate.py":
            (rd / "artifacts").mkdir(parents=True, exist_ok=True)
            with open(rd / "artifacts" / "sim_metrics.json", "w") as f:
                json.dump({"win_rate": 0.5, "profit_factor": 1.0, "trades_per_wk": 1}, f)

    monkeypatch.setattr(walkforward, "run_script", dummy_run_script)
    tz = config.get("market.timezone")
    monkeypatch.setattr(
        walkforward,
        "load_raw_index",
        lambda: (
            pd.Timestamp("2021-01-20", tz=tz),
            pd.Timestamp("2021-02-05", tz=tz),
        ),
    )

    args = [
        walkforward.__file__,
        "--run",
        str(run_dir),
        "--train_window_months",
        "0",
        "--window_weeks",
        "1",
        "--stepback_weeks",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", args)
    walkforward.main()

    captured = capsys.readouterr()
    assert "adjusting simulation end from" in captured.out
    assert "due to limited data" in captured.out
    sim_call = [c for c in calls if c[0] == "simulate.py"][0]
    end_idx = sim_call[1].index("--end_date") + 1
    assert sim_call[1][end_idx] == "2021-01-27"


def test_history_written_when_sim_end_adjusted(sample_run, monkeypatch):
    run_dir = sample_run

    def dummy_run_script(script, args, env):
        rd = Path(env["RUN_ID"])
        if script == "prepare.py":
            (rd / "data").mkdir(parents=True, exist_ok=True)
            tz = config.get("market.timezone")
            idx = pd.date_range("2021-01-24", periods=4, freq="D", tz=tz)
            pd.DataFrame(index=idx).to_csv(rd / "data" / "prepared.csv")
        elif script == "simulate.py":
            (rd / "artifacts").mkdir(parents=True, exist_ok=True)
            with open(rd / "artifacts" / "sim_metrics.json", "w") as f:
                json.dump({"win_rate": 0.5, "profit_factor": 1.0, "trades_per_wk": 1}, f)

    monkeypatch.setattr(walkforward, "run_script", dummy_run_script)
    tz = config.get("market.timezone")
    monkeypatch.setattr(
        walkforward,
        "load_raw_index",
        lambda: (
            pd.Timestamp("2021-01-20", tz=tz),
            pd.Timestamp("2021-02-05", tz=tz),
        ),
    )

    args = [
        walkforward.__file__,
        "--run",
        str(run_dir),
        "--train_window_months",
        "0",
        "--window_weeks",
        "1",
        "--stepback_weeks",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", args)
    walkforward.main()

    hist = run_dir / "history.csv"
    assert hist.exists()
    df = pd.read_csv(hist)
    assert len(df) == 1


def test_prepare_passes_train_end_date(sample_run, monkeypatch):
    run_dir = sample_run
    calls = []

    def dummy_run_script(script, args, env):
        calls.append((script, args))
        rd = Path(env["RUN_ID"])
        (rd / "artifacts").mkdir(parents=True, exist_ok=True)
        with open(rd / "artifacts" / "sim_metrics.json", "w") as f:
            json.dump({"win_rate": 0.5, "profit_factor": 1.0, "trades_per_wk": 1}, f)

    monkeypatch.setattr(walkforward, "run_script", dummy_run_script)
    monkeypatch.setattr(
        walkforward,
        "load_raw_index",
        lambda: (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-20")),
    )

    args = [
        walkforward.__file__,
        "--run",
        str(run_dir),
        "--train_window_months",
        "0",
        "--window_weeks",
        "1",
        "--stepback_weeks",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", args)
    walkforward.main()

    tz = config.get("market.timezone")
    end_raw = pd.Timestamp("2021-01-20", tz=tz)
    train_end = walkforward.get_market_week_start(end_raw)
    if end_raw - train_end < pd.Timedelta(days=5):
        train_end -= pd.DateOffset(weeks=1)
    expected = train_end.strftime("%Y-%m-%d")

    prep_call = [c for c in calls if c[0] == "prepare.py"][0]
    idx = prep_call[1].index("--train_end_date") + 1
    assert prep_call[1][idx] == expected


def test_label_passes_train_end_date(sample_run, monkeypatch):
    run_dir = sample_run
    calls = []

    def dummy_run_script(script, args, env):
        calls.append((script, args))
        rd = Path(env["RUN_ID"])
        (rd / "artifacts").mkdir(parents=True, exist_ok=True)
        with open(rd / "artifacts" / "sim_metrics.json", "w") as f:
            json.dump({"win_rate": 0.5, "profit_factor": 1.0, "trades_per_wk": 1}, f)

    monkeypatch.setattr(walkforward, "run_script", dummy_run_script)
    monkeypatch.setattr(
        walkforward,
        "load_raw_index",
        lambda: (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-20")),
    )

    args = [
        walkforward.__file__,
        "--run",
        str(run_dir),
        "--train_window_months",
        "0",
        "--window_weeks",
        "1",
        "--stepback_weeks",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", args)
    walkforward.main()

    tz = config.get("market.timezone")
    end_raw = pd.Timestamp("2021-01-20", tz=tz)
    train_end = walkforward.get_market_week_start(end_raw)
    if end_raw - train_end < pd.Timedelta(days=5):
        train_end -= pd.DateOffset(weeks=1)
    expected = train_end.strftime("%Y-%m-%d")

    label_call = [c for c in calls if c[0] == "label.py"][0]
    idx = label_call[1].index("--train_end_date") + 1
    assert label_call[1][idx] == expected


def test_ranker_cli_overrides(sample_run, monkeypatch):
    run_dir = sample_run
    calls = []

    def dummy_run_script(script, args, env):
        calls.append((script, args))
        rd = Path(env["RUN_ID"])
        (rd / "artifacts").mkdir(parents=True, exist_ok=True)
        with open(rd / "artifacts" / "sim_metrics.json", "w") as f:
            json.dump({"win_rate": 0.5, "profit_factor": 1.0, "trades_per_wk": 1}, f)

    monkeypatch.setattr(walkforward, "run_script", dummy_run_script)
    monkeypatch.setattr(
        walkforward,
        "load_raw_index",
        lambda: (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-20")),
    )

    args = [
        walkforward.__file__,
        "--run",
        str(run_dir),
        "--train_window_months",
        "0",
        "--window_weeks",
        "1",
        "--stepback_weeks",
        "1",
        "--target_trades_per_week",
        "60",
        "--min_trades_per_week",
        "50",
        "--max_trades_per_week",
        "70",
    ]
    monkeypatch.setattr(sys, "argv", args)
    walkforward.main()

    ranker_call = [c for c in calls if c[0] == "train_ranker.py"][0]
    assert "--target_trades_per_week" in ranker_call[1]
    assert "--min_trades_per_week" in ranker_call[1]
    assert "--max_trades_per_week" in ranker_call[1]


def test_walkforward_grid_option(sample_run, monkeypatch):
    run_dir = sample_run

    def dummy_run_script(script, args, env):
        rd = Path(env["RUN_ID"])
        (rd / "artifacts").mkdir(parents=True, exist_ok=True)
        with open(rd / "artifacts" / "sim_metrics.json", "w") as f:
            json.dump({"win_rate": 0.5, "profit_factor": 1.0, "trades_per_wk": 1}, f)

    monkeypatch.setattr(walkforward, "run_script", dummy_run_script)
    monkeypatch.setattr(
        walkforward,
        "load_raw_index",
        lambda: (pd.Timestamp("2020-12-20"), pd.Timestamp("2021-01-31")),
    )

    import optimize

    recorded = []

    def dummy_search(grid, *args, **kwargs):
        recorded.append(grid)
        return {}, {}

    monkeypatch.setattr(optimize, "search_configs", dummy_search)

    args = [
        walkforward.__file__,
        "--run",
        str(run_dir),
        "--train_window_months",
        "0",
        "--window_weeks",
        "1",
        "--stepback_weeks",
        "1",
        "--optimize",
        "--grid",
        "fast",
    ]
    monkeypatch.setattr(sys, "argv", args)
    walkforward.main()

    assert recorded and recorded[0] == optimize.GRID_MAP["fast"]

