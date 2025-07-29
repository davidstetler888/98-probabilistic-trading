"""
Shared helper functions for the Forex trading pipeline.
Aligned with @project.md sections 5-10.
"""

from pathlib import Path
import os
import sys
from datetime import datetime
import logging
import re
import pandas as pd
from config import config

logger = logging.getLogger(__name__)

# Regex to extract timestamp from messy strings like
# "2023-03-13 02:00:00-05:00;1,2,3" or similar.
TS_PATTERN = re.compile(
    r"\d{4}[-./]\d{2}[-./]\d{2}(?:\s|T)\d{2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:?\d{2})?"
)

# ---------------------------------------------------------------------- #
#  Basic run-ID helpers
# ---------------------------------------------------------------------- #


def get_run_id() -> str:
    """Return RUN_ID from env var or raise with clear message."""
    run_id = os.environ.get("RUN_ID")
    if not run_id:
        raise RuntimeError(
            "RUN_ID environment variable not set. "
            "Export one, e.g.  RUN_ID=models/run_20250530_1630"
        )
    return run_id


def get_run_dir() -> str:
    """Get the run directory from RUN_ID environment variable."""
    run_id = get_run_id()
    return str(Path(run_id))


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)


def make_run_dirs(run_id: str) -> dict[str, Path]:
    """
    Ensure the standard subdirs exist and return them.

    Returns:
        dict: {"data": Path(...), "models": Path(...), "artifacts": Path(...)}
    """
    run_path = Path(run_id)
    dirs = {
        "data": run_path / "data",
        "models": run_path / "models",
        "artifacts": run_path / "artifacts",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


# ---------------------------------------------------------------------- #
#  Data loading and processing
# ---------------------------------------------------------------------- #


def parse_end_date_arg(end_date: str | None) -> str | None:
    """Parse and validate end_date argument."""
    if not end_date:
        return None
    try:
        datetime.strptime(end_date, "%Y-%m-%d")
        return end_date
    except ValueError:
        raise ValueError("end_date must be in YYYY-MM-DD format")


def parse_start_date_arg(start_date: str | None) -> str | None:
    """Parse and validate start_date argument."""
    if not start_date:
        return None
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        return start_date
    except ValueError:
        raise ValueError("start_date must be in YYYY-MM-DD format")


def load_data(
    filepath: str,
    end_date: str | None = None,
    start_date: str | None = None,
    *,
    strict: bool = False,
) -> pd.DataFrame:
    """Load data from CSV file, optionally filtering by date range.

    ``end_date`` may be provided as a ``YYYY-MM-DD`` string. In that case the
    filter is inclusive of the entire day by converting it to ``23:59:59`` in
    the configured timezone before comparison.
    """
    # First, inspect the header to decide how to parse. Allow mixed ',' and ';'
    # delimiters so malformed rows do not shift columns.
    peek = pd.read_csv(filepath, sep="[;,]", engine="python", nrows=0)
    cols_lower = {c.lower() for c in peek.columns}
    use_header = (
        {"date", "time"}.issubset(cols_lower)
        or {"open", "high", "low", "close", "volume"}.issubset(cols_lower)
    )
    if use_header:
        logger.debug("load_data using header row for %s", filepath)
        df_raw = pd.read_csv(filepath, sep="[;,]", engine="python")
    else:
        logger.debug("load_data treating %s as headerless", filepath)
        df_raw = pd.read_csv(filepath, sep="[;,]", engine="python", header=None)
        base_cols = ["date", "time", "open", "high", "low", "close", "volume"]
        if len(df_raw.columns) >= 7:
            extra_cols = [f"extra_{i}" for i in range(len(df_raw.columns) - 7)]
            df_raw.columns = base_cols + extra_cols

    def _parse_single(val: str) -> pd.Timestamp:
        m = TS_PATTERN.search(val)
        if not m:
            return pd.NaT
        ts = pd.to_datetime(m.group(0), errors="coerce")
        if ts.tzinfo is None:
            return ts.tz_localize(config.get("market.timezone"))
        return ts.tz_convert(config.get("market.timezone"))

    # Check if this looks like a standard OHLCV format with date/time
    if {"date", "time"}.issubset(df_raw.columns):
        # Format: date,time,open,high,low,close,volume[,...]
        
        # Combine date and time into datetime and attempt to recover
        dt_str = df_raw["date"].astype(str) + " " + df_raw["time"].astype(str)

        dt = dt_str.apply(_parse_single)
        bad_mask = dt.isna()
        if bad_mask.any():
            repaired = df_raw.loc[bad_mask, ["date", "time"]]
            preview = repaired.head(3).to_csv(index=False, header=False).strip()
            msg = f"{bad_mask.sum()} rows with invalid date/time dropped in {filepath}"
            if strict:
                raise ValueError(f"{msg}\n{preview}")
            print("Warning:", msg)
            print("Examples:\n" + preview)
            logger.debug("Dropped rows: %s", preview.replace("\n", "; "))
            df_raw = df_raw[~bad_mask].copy()
            dt = dt[~bad_mask]
        
        # Set the datetime as index
        df_raw.index = dt
        df_raw = df_raw.drop(columns=["date", "time"])
        df = df_raw
        
    else:
        # Try loading with a dedicated datetime index column
        try:
            df = pd.read_csv(filepath, index_col="datetime", parse_dates=True)
        except Exception:
            # Load without index to inspect columns
            df = pd.read_csv(filepath)
            if "datetime" in df.columns:
                df = df.set_index("datetime")
            elif "date" not in df.columns or "time" not in df.columns:
                # Assume first column stores the index
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                df.index.name = "datetime"

        tz = config.get("market.timezone")

        # If "date" and "time" columns exist, construct datetime from them
        if "date" in df.columns and "time" in df.columns:
            dt_str = df["date"].astype(str) + " " + df["time"].astype(str)
            dt = dt_str.apply(_parse_single)
            bad_mask = dt.isna()
            if bad_mask.any():
                preview = (
                    df.loc[bad_mask, ["date", "time"]]
                    .head(3)
                    .to_csv(index=False, header=False)
                    .strip()
                )
                msg = f"{bad_mask.sum()} rows with invalid date/time dropped in {filepath}"
                if strict:
                    raise ValueError(f"{msg}\n{preview}")
                print("Warning:", msg)
                print("Examples:\n" + preview)
                logger.debug("Dropped rows: %s", preview.replace("\n", "; "))
                df = df[~bad_mask].copy()
                dt = dt[~bad_mask]
            if dt.dt.tz is None:
                dt = dt.dt.tz_localize(tz)
            else:
                dt = dt.dt.tz_convert(tz)
            df.index = dt
            df = df.drop(columns=["date", "time"])
            parsed = dt
        else:
            parsed = pd.Index([_parse_single(str(v)) for v in df.index])
            if parsed.isna().all():
                # Fall back to numeric datetime parsing as before
                parsed = pd.to_datetime(df.index, errors="coerce")
            if parsed.isna().all():
                raise ValueError(
                    "index values could not be parsed as datetimes (all values NaN)"
                )
            if parsed.isna().any():
                bad_mask = parsed.isna()
                msg = f"{bad_mask.sum()} index values could not be parsed as datetimes"
                preview = ", ".join(map(str, df.index[bad_mask][:3]))
                if strict:
                    raise ValueError(f"{msg}: {preview}")
                print("Warning:", msg, "and will be dropped.")
                print("Examples:", preview)
                logger.debug("Dropped rows: %s", preview)
                df = df[~bad_mask].copy()
                parsed = parsed[~bad_mask]
            if parsed.tz is None:
                parsed = parsed.tz_localize(tz)
            else:
                parsed = parsed.tz_convert(tz)
            df.index = parsed

    # Apply timezone if needed
    tz = config.get("market.timezone")
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)

    if end_date and isinstance(end_date, str):
        try:
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            pass
        else:
            end_date = (
                pd.Timestamp(end_date, tz=tz)
                + pd.Timedelta(days=1)
                - pd.Timedelta(seconds=1)
            )
    if start_date and end_date:
        df = df[(df.index >= start_date) & (df.index <= end_date)]
    elif start_date:
        df = df[df.index >= start_date]
    elif end_date:
        df = df[df.index <= end_date]
    return df


def get_market_week_start(ts: pd.Timestamp) -> pd.Timestamp:
    """Return the start of the market week for ``ts`` in configured timezone."""
    tz = config.get("market.timezone")
    open_setting = config.get("market.weekly_open")
    day_str, time_str = open_setting.split()
    day_abbr = day_str[:3].upper()
    prev_day = {
        "MON": "SUN",
        "TUE": "MON",
        "WED": "TUE",
        "THU": "WED",
        "FRI": "THU",
        "SAT": "FRI",
        "SUN": "SAT",
    }[day_abbr]
    hours, minutes = map(int, time_str.split(":"))
    open_offset = pd.Timedelta(hours=hours, minutes=minutes)

    ts_local = ts.tz_convert(tz) if ts.tzinfo else ts.tz_localize(tz)
    ts_naive = ts_local.tz_localize(None)
    start_naive = (ts_naive - open_offset).to_period(
        f"W-{prev_day}"
    ).start_time + open_offset
    return start_naive.tz_localize(tz)


def get_market_week_end(ts: pd.Timestamp) -> pd.Timestamp:
    """Return the end of the market week for ``ts`` in configured timezone."""
    tz = config.get("market.timezone")
    close_setting = config.get("market.weekly_close")
    close_day, close_time = close_setting.split()
    close_hours, close_minutes = map(int, close_time.split(":"))

    open_setting = config.get("market.weekly_open")
    open_day, _ = open_setting.split()

    day_map = {
        "MON": 0,
        "TUE": 1,
        "WED": 2,
        "THU": 3,
        "FRI": 4,
        "SAT": 5,
        "SUN": 6,
    }

    open_num = day_map[open_day[:3].upper()]
    close_num = day_map[close_day[:3].upper()]

    week_start = get_market_week_start(ts)
    days_ahead = (close_num - open_num) % 7
    end_base = week_start + pd.Timedelta(days=days_ahead)
    end_naive = end_base.tz_convert(tz).tz_localize(None).normalize() + pd.Timedelta(
        hours=close_hours, minutes=close_minutes
    )
    return end_naive.tz_localize(tz)


# ---------------------------------------------------------------------- #
#  Convenience printing
# ---------------------------------------------------------------------- #


def prep_summary(df: pd.DataFrame) -> None:
    """Print a one-liner summary for prepare.py results."""
    regimes = (
        df["market_regime"].value_counts().sort_index().to_dict()
        if "market_regime" in df.columns
        else {}
    )
    print(
        f"[prepare] rows={len(df):,}   regimes={regimes}   "
        f"first={df.index.min()}   last={df.index.max()}"
    )


# ---------------------------------------------------------------------- #
#  (Placeholder) price utility stub — will be expanded later
# ---------------------------------------------------------------------- #


def price_to_pips(
    entry: float, exit_: float, side: str, *, spread: float = 0.0
) -> float:
    """Convert a price difference to pips, adjusting for spread.

    For long trades:
    - Entry price includes spread (entry + spread)
    - Exit price is as is
    - Profit = exit - (entry + spread)

    For short trades:
    - Entry price is as is
    - Exit price includes spread (exit + spread)
    - Profit = entry - (exit + spread)

    Args:
        entry: Entry price
        exit_: Exit price
        side: 'long' or 'short'
        spread: Spread in price units (default: 0.0)

    Returns:
        Profit/loss in pips
    """
    if side.lower() == "long":
        # For longs: entry price includes spread
        diff = exit_ - (entry + spread)
    else:
        # For shorts: exit price includes spread
        diff = entry - (exit_ + spread)

    return diff * 10_000  # EURUSD pip conversion (5-dec pricing)
