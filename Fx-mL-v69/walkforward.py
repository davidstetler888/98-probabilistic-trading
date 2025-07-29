import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
import io
import re
from contextlib import redirect_stdout, redirect_stderr

import pandas as pd

from copy import deepcopy
from config import config
import optimize
from optimize import DEFAULT_GRID, apply_overrides
from utils import (
    get_run_dir,
    make_run_dirs,
    get_market_week_start,
    get_market_week_end,
)


class Tee(io.TextIOBase):
    """Write output to multiple streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


class FilteredTee(io.TextIOBase):
    """Write output to console and selectively to a file."""

    def __init__(self, console, log_file, patterns: list[str], verbose: bool = False):
        self.console = console
        self.log_file = log_file
        self.patterns = [re.compile(p) for p in patterns]
        self.verbose = verbose
        self.buffer = ""

    def write(self, data):
        self.console.write(data)
        self.console.flush()
        if self.verbose:
            self.log_file.write(data)
            return len(data)

        for ch in data:
            self.buffer += ch
            if ch == "\n":
                line = self.buffer
                if any(p.search(line) for p in self.patterns):
                    self.log_file.write(line)
                self.buffer = ""
        return len(data)

    def flush(self):
        self.console.flush()
        if self.buffer:
            if self.verbose or any(p.search(self.buffer) for p in self.patterns):
                self.log_file.write(self.buffer)
            self.buffer = ""
        self.log_file.flush()


def format_td(td: timedelta) -> str:
    """Return HH:MM:SS string for a timedelta."""
    total = int(td.total_seconds())
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def parse_args():
    parser = argparse.ArgumentParser(description="Walk-forward backtest loop")
    parser.add_argument(
        "--train_window_months",
        type=int,
        default=config.get("walkforward.train_window_months"),
    )
    parser.add_argument(
        "--window_weeks",
        type=int,
        default=config.get("walkforward.window_weeks"),
    )
    parser.add_argument(
        "--stepback_weeks",
        type=int,
        default=config.get("walkforward.stepback_weeks", 12),
    )
    parser.add_argument("--run", type=str, help="Base directory for runs")
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Grid search configs before each iteration",
    )
    parser.add_argument(
        "--max_configs",
        type=int,
        default=None,
        help="Limit number of parameter sets tested when optimizing",
    )
    parser.add_argument(
        "--grid",
        choices=["fast", "mid", "full"],
        default="full",
        help="Which parameter grid to use when optimizing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Write all output to output.txt instead of filtering",
    )
    parser.add_argument(
        "--target_trades_per_week",
        type=float,
        default=None,
        help="Desired trades per week for ranker",
    )
    parser.add_argument(
        "--min_trades_per_week",
        type=float,
        default=None,
        help="Minimum trades per week for ranker",
    )
    parser.add_argument(
        "--max_trades_per_week",
        type=float,
        default=None,
        help="Maximum trades per week for ranker",
    )
    return parser.parse_args()


def load_raw_index():
    """Return the first and last timestamps in the raw CSV.

    Only the first and last lines are read to avoid loading the entire file.
    """
    input_dir = config.get("data.input_dir")
    path = next(Path(input_dir).glob("*.csv"))

    def parse_line(line: str) -> pd.Timestamp:
        date, time, *_ = line.strip().split(",")
        return pd.to_datetime(f"{date} {time}")

    with open(path, "r") as f:
        # Skip headers or any line that does not parse as a timestamp
        first_line = f.readline()
        while first_line:
            tokens = first_line.strip().split(",")
            if (
                len(tokens) >= 2
                and tokens[0].lower() == "date"
                and tokens[1].lower() == "time"
            ):
                first_line = f.readline()
                continue
            try:
                parse_line(first_line)
                break
            except Exception:
                first_line = f.readline()
        if not first_line:
            raise ValueError("No valid timestamp lines found")

    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell() - 1
        pos = end
        while pos >= 0:
            f.seek(pos)
            char = f.read(1)
            if char == b"\n" and pos != end:
                break
            pos -= 1
        f.seek(pos + 1)
        last_line = f.readline().decode()

    tz = config.get("market.timezone")
    start_ts = parse_line(first_line)
    end_ts = parse_line(last_line)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(tz)
    else:
        start_ts = start_ts.tz_convert(tz)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize(tz)
    else:
        end_ts = end_ts.tz_convert(tz)
    return start_ts, end_ts


def run_script(script: str, args: list[str], env: dict) -> None:
    """Run a helper script and stream its output."""
    run_dir = Path(env.get("RUN_ID", ""))
    log_dir = run_dir / "logs" if run_dir else None
    log_file = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{Path(script).stem}.log"
        log_file = open(log_path, "w")

    proc = subprocess.Popen(
        [sys.executable, script] + args,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_file:
                log_file.write(line)
    finally:
        if log_file:
            log_file.flush()
            log_file.close()
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, [sys.executable, script] + args
        )


def append_history(base: Path, end: pd.Timestamp, metrics: dict, best: dict | None = None) -> None:
    hist_path = base / "history.csv"
    exists = hist_path.exists()
    with open(hist_path, "a", newline="") as f:
        writer = csv.writer(f)
        header = [
            "end_date",
            "win_rate",
            "avg_rr",
            "total_trades",
            "profit_factor",
            "trades_per_wk",
        ]
        row = [
            end.strftime("%Y-%m-%d"),
            metrics.get("win_rate", 0),
            metrics.get("avg_rr", 0),
            metrics.get("total_trades", metrics.get("trades_per_wk", 0)),
            metrics.get("profit_factor", 0),
            metrics.get("trades_per_wk", 0),
        ]
        if best is not None:
            header.extend(["opt_win_rate", "opt_profit_factor", "opt_trades_per_wk"])
            row.extend([
                best.get("win_rate", 0),
                best.get("profit_factor", 0),
                best.get("trades_per_wk", 0),
            ])
        if not exists:
            writer.writerow(header)
        writer.writerow(row)


def main(args=None):
    if args is None:
        args = parse_args()
    if args.optimize:
        print(f"Optimization enabled: searching configs ({args.grid})")
    base_run = Path(args.run) if args.run else Path(get_run_dir())
    base_run.mkdir(parents=True, exist_ok=True)

    summary_rows: list[tuple[pd.Timestamp, dict]] = []
    start_time = datetime.now()

    def print_eta(completed: int) -> None:
        elapsed = datetime.now() - start_time
        avg = elapsed / completed if completed else timedelta(0)
        remaining = avg * (args.stepback_weeks - completed)
        print(
            f"Progress: {completed}/{args.stepback_weeks} iterations, "
            f"elapsed {format_td(elapsed)}, est. remaining {format_td(remaining)}"
        )

    start_raw, end_raw = load_raw_index()
    tz = config.get("market.timezone")
    if start_raw.tzinfo is None:
        start_raw = start_raw.tz_localize(tz)
    else:
        start_raw = start_raw.tz_convert(tz)
    if end_raw.tzinfo is None:
        end_raw = end_raw.tz_localize(tz)
    else:
        end_raw = end_raw.tz_convert(tz)

    # Find the last complete week in the data
    last_week_start = get_market_week_start(end_raw)
    
    # If we're not at the end of a complete week, go back one week
    if end_raw - last_week_start < pd.Timedelta(days=5):
        last_week_start -= pd.DateOffset(weeks=1)
    
    print(f"Data range: {start_raw.strftime('%Y-%m-%d')} to {end_raw.strftime('%Y-%m-%d')}")
    print(f"Last complete week starts: {last_week_start.strftime('%Y-%m-%d')}")
    
    # Calculate the earliest week we can start from
    # We need: train_window_months + window_weeks of data before the simulation
    earliest_possible_start = last_week_start - pd.DateOffset(weeks=args.stepback_weeks - 1)
    earliest_possible_start = earliest_possible_start - pd.DateOffset(months=args.train_window_months)
    
    if earliest_possible_start < start_raw:
        print(f"Warning: Earliest possible start {earliest_possible_start.strftime('%Y-%m-%d')} is before available data {start_raw.strftime('%Y-%m-%d')}")
        print("Adjusting stepback_weeks to fit available data...")
        
        # Calculate how many weeks we can actually go back
        available_weeks = (end_raw - start_raw).days // 7
        needed_weeks = args.train_window_months * 4 + args.window_weeks  # Rough estimate
        max_stepback = max(1, available_weeks - needed_weeks)
        
        if max_stepback < args.stepback_weeks:
            print(f"Reducing stepback_weeks from {args.stepback_weeks} to {max_stepback}")
            args.stepback_weeks = max_stepback

    # Start from the most recent week and work backwards
    for offset in range(args.stepback_weeks):
        iter_start = datetime.now()
        # Calculate the week we're processing (0 = most recent, 1 = second most recent, etc.)
        current_week_start = last_week_start - pd.DateOffset(weeks=offset)
        
        # Training period: from current_week_start back by train_window_months
        train_end = current_week_start
        train_start = train_end - pd.DateOffset(months=args.train_window_months)
        
        # Simulation period: from current_week_start forward by window_weeks
        sim_end = get_market_week_end(
            current_week_start + pd.DateOffset(weeks=args.window_weeks - 1)
        )
        
        # Verify all dates are within available data
        if train_start < start_raw:
            print(
                f"Skipping week starting {current_week_start.strftime('%Y-%m-%d')}: training start {train_start.strftime('%Y-%m-%d')} is before available data"
            )
            print_eta(offset + 1)
            continue
            
        if sim_end > end_raw:
            print(
                f"Skipping week starting {current_week_start.strftime('%Y-%m-%d')}: simulation end {sim_end.strftime('%Y-%m-%d')} exceeds available data"
            )
            print_eta(offset + 1)
            continue

        print(f"\nProcessing week {offset + 1}/{args.stepback_weeks}: {current_week_start.strftime('%Y-%m-%d')}")
        print(f"  Training: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
        print(f"  Simulation: {train_end.strftime('%Y-%m-%d')} to {sim_end.strftime('%Y-%m-%d')}")

        run_dir = base_run / train_end.strftime("%Y%m%d")
        env = os.environ.copy()
        env["RUN_ID"] = str(run_dir)
        make_run_dirs(str(run_dir))

        train_end_str = train_end.strftime("%Y-%m-%d")
        start_str = train_start.strftime("%Y-%m-%d")
        end_str = sim_end.strftime("%Y-%m-%d")

        best_conf = {}
        best_metrics = None
        base_cfg = deepcopy(config._config)
        if args.optimize:
            opt_dir = run_dir / "opt_tmp"
            opt_dir.mkdir(parents=True, exist_ok=True)
            cache_dir = run_dir / "opt_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            optimize.base_config = deepcopy(config._config)
            grid = optimize.GRID_MAP.get(args.grid, DEFAULT_GRID)
            best_conf, best_metrics = optimize.search_configs(
                grid,
                start_str,
                train_end_str,
                end_str,
                str(opt_dir),
                args.max_configs,
                str(cache_dir),
            )
            with open(run_dir / "best_config.json", "w") as f:
                json.dump({"best_conf": best_conf, "best_metrics": best_metrics}, f)
            config._config = apply_overrides(base_cfg, best_conf)

        run_script(
            "prepare.py",
            [
                "--start_date",
                start_str,
                "--train_end_date",
                train_end_str,
                "--end_date",
                end_str,
                "--input_dir",
                config.get("data.input_dir"),
            ],
            env,
        )
        run_script(
            "label.py",
            [
                "--run",
                str(run_dir),
                "--start_date",
                start_str,
                "--train_end_date",
                train_end_str,
                "--end_date",
                end_str,
            ],
            env,
        )
        model_args = [
            "--run",
            str(run_dir),
            "--start_date",
            start_str,
            "--train_end_date",
            train_end_str,
            "--end_date",
            end_str,
        ]
        run_script("train_base.py", model_args, env)
        run_script("train_meta.py", model_args, env)
        run_script("sltp.py", model_args, env)
        ranker_args = []
        if args.target_trades_per_week is not None:
            ranker_args += [
                "--target_trades_per_week",
                str(args.target_trades_per_week),
            ]
        if args.min_trades_per_week is not None:
            ranker_args += [
                "--min_trades_per_week",
                str(args.min_trades_per_week),
            ]
        if args.max_trades_per_week is not None:
            ranker_args += [
                "--max_trades_per_week",
                str(args.max_trades_per_week),
            ]
        run_script("train_ranker.py", model_args + ranker_args, env)

        # Verify that the simulation end date is within the prepared data range
        prepared_path = run_dir / "data" / "prepared.csv"
        if prepared_path.exists():
            df_dates = pd.read_csv(prepared_path, index_col=0, parse_dates=True, usecols=[0])
            tz = config.get("market.timezone")
            idx = df_dates.index
            # Ensure idx is a DatetimeIndex before checking timezone
            if not isinstance(idx, pd.DatetimeIndex):
                idx = pd.to_datetime(idx, utc=True)
            if idx.tz is None:
                idx = idx.tz_localize(tz)
            else:
                idx = idx.tz_convert(tz)
            max_date = idx.max()
            end_dt = pd.Timestamp(end_str, tz=tz)
            if end_dt > max_date:
                print(
                    f"Warning: adjusting simulation end from {end_str} to {max_date.strftime('%Y-%m-%d')} due to limited data"
                )
                end_dt = max_date
                end_str = end_dt.strftime("%Y-%m-%d")

        sim_args = ["--run", str(run_dir), "--start_date", train_end_str, "--end_date", end_str]
        if args.window_weeks == 1:
            sim_args += ["--test_frac", "1.0"]
        run_script("simulate.py", sim_args, env)

        metrics_file = run_dir / "artifacts" / "sim_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            append_history(base_run, train_end, metrics, best_metrics)
            summary_rows.append((train_end, metrics))

        print_eta(offset + 1)

        if args.optimize:
            config._config = base_cfg

    if summary_rows:
        print("\nSummary of results:")
        for end, metr in sorted(summary_rows):
            trades = metr.get("total_trades", metr.get("trades_per_wk", 0))
            print(
                f"{end.strftime('%Y-%m-%d')}: Win Rate {metr.get('win_rate', 0)*100:.1f}%, "
                f"Avg RR {metr.get('avg_rr', 0):.2f}, Trades {int(trades)}"
            )


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    patterns = [
        r"\[prepare\]",
        r"\[label\]",
        "Test Simulation Results",
        "Full Simulation Results",
        "Total Trades",
        "Win Rate",
        "Profit Factor",
        "Average RR",
    ]
    with open(output_dir / "output.txt", "w") as f:
        tee = FilteredTee(sys.stdout, f, patterns, verbose=args.verbose)
        with redirect_stdout(tee), redirect_stderr(tee):
            main(args)
