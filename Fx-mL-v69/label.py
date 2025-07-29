import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
from utils import (
    get_run_dir,
    make_run_dirs,
    parse_start_date_arg,
    parse_end_date_arg,
    load_data,
)
from config import config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Label trading data with SL/TP grid")
    parser.add_argument(
        "--start_date",
        type=str,
        required=False,
        help="Earliest bar to include (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=False,
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--train_end_date",
        type=str,
        required=False,
        help="Cutoff date for labeling (YYYY-MM-DD)",
    )
    parser.add_argument("--run", type=str, help="Run directory (overrides RUN_ID)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort if CSV parsing fails",
    )
    return parser.parse_args()


def create_sltp_grid():
    """Create grid of SL/TP pairs from config multipliers."""
    sl_multipliers = config.get("sl_tp_grid.sl_multipliers")
    tp_multipliers = config.get("sl_tp_grid.tp_multipliers")
    spread = config.get("sl_tp_grid.spread")

    # Create grid of (SL, TP) pairs
    grid = []
    for sl_mult in sl_multipliers:
        for tp_mult in tp_multipliers:
            sl_pips = sl_mult * spread * 10000  # Convert to pips
            tp_pips = tp_mult * spread * 10000
            grid.append((sl_pips, tp_pips))

    return grid


def label_trades(
    df: pd.DataFrame, future_window: int, train_end_date: str | None = None
) -> pd.DataFrame:
    """Label trades based on future price movements.

    Parameters use the ``label`` section of ``config.yaml``.
    Bars that fail to meet the criteria receive a label of ``-1``.
    When ``train_end_date`` is provided, labels that would require
    looking past this cutoff are set to ``-1``.
    """

    # Config-driven parameters
    threshold = config.get("label.threshold")
    max_sl_pips = config.get("label.max_sl_pips")
    min_rr = config.get("label.min_rr")

    max_sl = max_sl_pips / 10000.0  # convert pips to price
    tp_price = max_sl * min_rr

    # Future high/low/close arrays
    future_high = (
        df["high"]
        .rolling(window=future_window, min_periods=1)
        .max()
        .shift(-future_window)
    )
    future_low = (
        df["low"]
        .rolling(window=future_window, min_periods=1)
        .min()
        .shift(-future_window)
    )
    future_close = df["close"].shift(-future_window)

    long_profit = future_high - df["close"]
    long_loss = df["close"] - future_low
    short_profit = df["close"] - future_low
    short_loss = future_high - df["close"]

    cond_long = (
        (long_profit >= threshold)
        & (long_loss <= max_sl)
        & (long_profit / long_loss >= min_rr)
    )
    cond_short = (
        (short_profit >= threshold)
        & (short_loss <= max_sl)
        & (short_profit / short_loss >= min_rr)
    )

    # Use -1 to denote bars with no valid signal as described in docs/project.md
    df["long_label"] = np.where(cond_long, 1, -1)
    df["short_label"] = np.where(cond_short, 1, -1)

    if train_end_date:
        cutoff_ts = pd.Timestamp(train_end_date, tz=df.index.tz)
        cutoff_idx = df.index.searchsorted(cutoff_ts)
        mask = np.arange(len(df)) + future_window > cutoff_idx
        df.loc[df.index[mask], ["long_label", "short_label"]] = -1

    # ------------------------------------------------------------------
    #  Enforce cooldown between consecutive signals of the same side
    # ------------------------------------------------------------------
    cooldown_min = config.get("label.cooldown_min", 0)
    if cooldown_min > 0 and len(df) > 0:
        cooldown_delta = pd.Timedelta(minutes=cooldown_min)
        last_long_time = df.index[0] - cooldown_delta
        last_short_time = df.index[0] - cooldown_delta

        for ts in df.index:
            if df.at[ts, "long_label"] == 1:
                if ts - last_long_time >= cooldown_delta:
                    last_long_time = ts
                else:
                    df.at[ts, "long_label"] = -1

            if df.at[ts, "short_label"] == 1:
                if ts - last_short_time >= cooldown_delta:
                    last_short_time = ts
                else:
                    df.at[ts, "short_label"] = -1
    df["future_high"] = future_high
    df["future_low"] = future_low
    df["future_close"] = future_close

    return df


def main():
    """Main function."""
    args = parse_args()
    run_dir = args.run if args.run else get_run_dir()
    make_run_dirs(run_dir)

    prepared_path = Path(run_dir) / "data" / "prepared.csv"
    if not prepared_path.exists():
        print(f"Error: {prepared_path} not found. Run prepare.py first.")
        sys.exit(1)

    try:
        start_date = parse_start_date_arg(args.start_date)
        end_date = parse_end_date_arg(args.end_date)
        train_end_date = parse_end_date_arg(args.train_end_date)
    except ValueError as exc:
        raise SystemExit(f"Invalid date: {exc}") from exc

    df = load_data(
        str(prepared_path), end_date=end_date, start_date=start_date, strict=args.strict
    )
    # Create SL/TP grid
    grid = create_sltp_grid()
    # Label trades
    df = label_trades(
        df,
        future_window=config.get("label.future_window"),
        train_end_date=train_end_date,
    )
    df = df.rename(columns={"long_label": "label_long", "short_label": "label_short"})
    output_path = Path(run_dir) / "data" / "labeled.csv"
    df.to_csv(output_path)
    # Print summary
    signals_long = (df.label_long == 1).sum()
    signals_short = (df.label_short == 1).sum()
    total = len(df[(df.label_long == 1) | (df.label_short == 1)])
    weeks = (df.index[-1] - df.index[0]).days / 7
    print(
        f"[label] signals_long={signals_long:,}  signals_short={signals_short:,}  ≈{total/weeks:0.1f} signals/wk"
    )


if __name__ == "__main__":
    main()
