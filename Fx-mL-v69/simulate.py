"""Backtesting Simulation Module

This module implements a vectorized backtesting system for the EURUSD trading strategy.
It loads trade signals from `train_ranker.py` (which itself uses outputs from `train_base.py` and `train_meta.py`) and replays them to compute performance metrics.
The goal is to steer the strategy toward **58‑72 % win rate**,
an average **1 : 2 to 1 : 3 risk‑reward**,
and **25‑50 trades per week**.

Key Features:
- Loads ranked trade signals generated offline
- Handles market regimes consistently with training
- Vectorized trade simulation with proper SL/TP ordering
- Computes comprehensive performance metrics
- Generates equity curve and trade statistics
- Supports configurable risk percentage and acceptance gates
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import json
import pickle

from config import config
from utils import (
    parse_end_date_arg,
    parse_start_date_arg,
    load_data,
    ensure_dir,
    get_run_dir,
    price_to_pips,
    get_market_week_start,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Target performance ranges (from config goals)
GOAL_WIN_RATE = tuple(config.get("goals.win_rate_range", (0.58, 0.72)))
GOAL_RR = tuple(config.get("goals.risk_reward_range", (2.0, 3.0)))
GOAL_TRADES_PER_WEEK = tuple(config.get("goals.trades_per_week_range", (25, 50)))

# Default values for backward compatibility
DEFAULT_RISK_PCT = 2.5
DEFAULT_MIN_PF = 1.0
DEFAULT_MAX_DD = 0.10
DEFAULT_MIN_AVG_RR = 2.0

# Create SL/TP grid from config
SPREAD = config.get("sl_tp_grid.spread", 0.00013)
SL_MULTIPLIERS = config.get("sl_tp_grid.sl_multipliers")
TP_MULTIPLIERS = config.get("sl_tp_grid.tp_multipliers")

# Create valid SL/TP combinations
SL_TP_PAIRS = []
for sl_mult in SL_MULTIPLIERS:
    for tp_mult in TP_MULTIPLIERS:
        sl_pips = (SPREAD * sl_mult) / 0.0001  # Convert to pips
        tp_pips = sl_pips * tp_mult
        if tp_pips >= 2 * sl_pips:  # Only keep RR >= 2.0
            SL_TP_PAIRS.append((sl_pips, tp_pips))


def get_dynamic_position_size(price, atr, base_risk_pct, account_balance):
    """
    Calculate position size based on volatility
    Higher volatility = smaller positions
    Lower volatility = larger positions
    """
    
    # ATR-based volatility adjustment
    volatility_factor = atr / price
    
    # Adjust risk based on volatility
    if volatility_factor > 0.002:  # High volatility
        risk_multiplier = 0.7
    elif volatility_factor < 0.001:  # Low volatility
        risk_multiplier = 1.3
    else:  # Normal volatility
        risk_multiplier = 1.0
    
    adjusted_risk = base_risk_pct * risk_multiplier
    
    # Calculate position size based on adjusted risk
    position_size = (account_balance * adjusted_risk) / (atr * 10000)
    
    return position_size, adjusted_risk


def calculate_adaptive_risk(current_price, atr, base_risk_pct, portfolio_risk, max_portfolio_risk):
    """
    Calculate adaptive risk based on market conditions and portfolio state
    """
    
    # Base volatility adjustment
    volatility_factor = atr / current_price
    
    # Volatility-based risk adjustment
    if volatility_factor > 0.002:  # High volatility
        volatility_multiplier = 0.8
    elif volatility_factor < 0.001:  # Low volatility
        volatility_multiplier = 1.2
    else:  # Normal volatility
        volatility_multiplier = 1.0
    
    # Portfolio utilization adjustment
    portfolio_utilization = portfolio_risk / max_portfolio_risk
    if portfolio_utilization > 0.8:  # High utilization
        portfolio_multiplier = 0.5
    elif portfolio_utilization > 0.5:  # Medium utilization
        portfolio_multiplier = 0.8
    else:  # Low utilization
        portfolio_multiplier = 1.0
    
    # Combined adjustment
    adjusted_risk = base_risk_pct * volatility_multiplier * portfolio_multiplier
    
    # Ensure minimum and maximum risk bounds
    adjusted_risk = max(0.005, min(adjusted_risk, 0.05))  # Between 0.5% and 5%
    
    return adjusted_risk


class PortfolioRiskManager:
    def __init__(self, max_portfolio_risk=0.06, max_correlation=0.7):
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlation = max_correlation
        self.open_positions = []
    
    def can_take_position(self, new_signal, current_portfolio_risk):
        """Check if new position can be taken based on portfolio risk"""
        
        # Check total portfolio risk
        if current_portfolio_risk + new_signal.get('risk_amount', 0) > self.max_portfolio_risk:
            return False
        
        # Check correlation with existing positions
        if self.open_positions:
            correlation = self.calculate_correlation(new_signal)
            if correlation > self.max_correlation:
                return False
        
        return True
    
    def calculate_correlation(self, new_signal):
        """Calculate correlation with existing positions"""
        # Simplified correlation based on direction and timing
        same_direction_count = sum(1 for pos in self.open_positions 
                                 if pos.get('side') == new_signal.get('side'))
        
        return same_direction_count / len(self.open_positions)
    
    def add_position(self, position):
        """Add position to portfolio"""
        self.open_positions.append(position)
    
    def remove_position(self, position_id):
        """Remove position from portfolio"""
        self.open_positions = [pos for pos in self.open_positions 
                              if pos.get('id') != position_id]
    
    def get_portfolio_risk(self):
        """Calculate current portfolio risk"""
        return sum(pos.get('risk_amount', 0) for pos in self.open_positions)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run backtesting simulation")
    parser.add_argument(
        "--output",
        type=str,
        default="sim",
        help="Basename for artifact outputs (under RUN/artifacts)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=False,
        help="YYYY-MM-DD (inclusive) last bar for simulation",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        required=False,
        help="Earliest bar to simulate (YYYY-MM-DD)",
    )
    parser.add_argument("--run", type=str, help="Run directory (overrides RUN_ID)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort if CSV parsing fails",
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.2,
        help="Fraction of data reserved for test simulation",
    )
    return parser.parse_args()


def validate_simulation_config() -> None:
    """Validate key simulation settings from ``config``."""
    risk = config.get("simulation.risk_per_trade", 0.0)
    cooldown = config.get("simulation.cooldown_min", 0)
    max_pos = config.get("simulation.max_positions", 0)
    if not (0 < risk <= 1):
        raise ValueError("simulation.risk_per_trade must be between 0 and 1")
    if cooldown < 0:
        raise ValueError("simulation.cooldown_min must be >= 0")
    if max_pos <= 0:
        raise ValueError("simulation.max_positions must be > 0")


def load_models(run_dir: str) -> Dict:
    """Load probability data, meta models, SL/TP models and edge threshold."""
    run_path = Path(run_dir)
    models = {}

    # base probabilities
    probs_path = run_path / "data" / "probs_base.csv"
    if probs_path.exists():
        models["probs"] = pd.read_csv(probs_path, index_col=0, parse_dates=True)
    else:
        logger.warning("%s not found", probs_path)
        models["probs"] = pd.DataFrame()

    # meta models
    meta_path = run_path / "models" / "meta.pkl"
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            models["meta_models"] = pickle.load(f)
    else:
        logger.warning("%s not found", meta_path)
        models["meta_models"] = {}

    # sltp models
    sltp_models: Dict[tuple, object] = {}
    for regime in range(4):
        for side in ["long", "short"]:
            mpath = run_path / "models" / f"sltp_regime{regime}_{side}.pkl"
            if mpath.exists():
                with open(mpath, "rb") as f:
                    sltp_models[(regime, side)] = pickle.load(f)
    models["sltp_models"] = sltp_models

    # edge threshold
    thr_path = run_path / "models" / "edge_threshold.json"
    if thr_path.exists():
        with open(thr_path) as f:
            thr_json = json.load(f)
        models["edge_threshold"] = float(thr_json.get("edge_threshold", 0.0))
    else:
        logger.warning("%s not found", thr_path)
        models["edge_threshold"] = None

    return models


def load_signals(
    run_dir: str, end_date: str | None, start_date: str | None
) -> pd.DataFrame:
    """Load ranked trade signals produced by train_ranker.py."""
    path = Path(run_dir) / "data" / "signals.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run train_ranker.py first")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    
    # Convert string dates to Timestamp objects for comparison
    tz = config.get("market.timezone")
    if start_date and isinstance(start_date, str):
        start_date = pd.Timestamp(start_date, tz=tz)
    if end_date and isinstance(end_date, str):
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


def prepare_trade_df(prepared: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Combine prepared market data with signal information."""
    # Use left join to keep all price data and attach any matching signals
    df = prepared.join(signals, how="left")

    required = ["side", "sl_pips", "tp_pips"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"signals.csv missing columns: {missing}")

    out = pd.DataFrame(
        {
            "timestamp": df.index,
            "price": df["close"],
            "high": df["high"],
            "low": df["low"],
            "side": df["side"],
            "sl_points": df["sl_pips"] / 10.0,
            "tp_points": df["tp_pips"] / 10.0,
        }
    )

    # ``has_signal`` indicates whether a signal is present for the row
    out["has_signal"] = out["side"].notna()

    return out


def simulate_trade(
    price: float,
    side: str,
    sl_points: float,
    tp_points: float,
    future_highs: np.ndarray,
    future_lows: np.ndarray,
    risk_pct: float = 2.5,
    atr: float | None = None,
    account_balance: float = 1000,
    use_dynamic_sizing: bool = True,
) -> Tuple[float, int]:
    """Simulate a single trade.

    Args:
        price: Entry price
        side: 'long' or 'short'
        sl_points: Stop loss in points
        tp_points: Take profit in points
        future_highs: Array of future high prices
        future_lows: Array of future low prices
        risk_pct: Risk percentage per trade
        atr: Average True Range for dynamic sizing
        account_balance: Current account balance
        use_dynamic_sizing: Whether to use dynamic position sizing

    Returns:
        Tuple of (profit_pips, exit_bar)
    """
    # Convert points to pips
    sl_pips = sl_points * 10
    tp_pips = tp_points * 10

    # Calculate position size with dynamic sizing if enabled
    if use_dynamic_sizing and atr is not None:
        position_size, adjusted_risk = get_dynamic_position_size(price, atr, risk_pct / 100, account_balance)
    else:
        # Original fixed position sizing
        position_size = account_balance * (risk_pct / 100) / sl_pips if sl_pips > 0 else 0

    # Calculate TP/SL levels
    if side == "long":
        tp_level = price + tp_pips / 10000
        sl_level = price - sl_pips / 10000
        tp_hits = np.where(future_highs >= tp_level)[0]
        sl_hits = np.where(future_lows <= sl_level)[0]
    else:  # short
        tp_level = price - tp_pips / 10000
        sl_level = price + sl_pips / 10000
        tp_hits = np.where(future_lows <= tp_level)[0]
        sl_hits = np.where(future_highs >= sl_level)[0]

    n_bars = len(future_highs)

    tp_touch = tp_hits[0] if tp_hits.size else n_bars
    sl_touch = sl_hits[0] if sl_hits.size else n_bars

    # Determine if TP or SL hit first (tie goes to TP)
    hit_tp = tp_touch <= sl_touch
    exit_price = tp_level if hit_tp else sl_level

    # Profit in pips adjusted for spread
    profit_pips = price_to_pips(price, exit_price, side, spread=SPREAD)
    exit_bar = min(tp_touch, sl_touch)

    return profit_pips, exit_bar


def simulate_df(
    df: pd.DataFrame,
    risk_pct: float = config.get("simulation.risk_per_trade", 0.02),
    cooldown_min: int = config.get(
        "simulation.cooldown_min", config.get("label.cooldown_min", 0)
    ),
    max_positions: int | None = None,
    return_trades: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """Simulate trades from DataFrame.

    Args:
        df: DataFrame with columns [timestamp, price, high, low, side, sl_points, tp_points]
        risk_pct: Risk percentage per trade

    Returns:
        Tuple of (equity_curve, metrics) or
        (equity_curve, metrics, trade_details) if ``return_trades`` is True.
    """
    cfg_max_pos = config.get("simulation.max_positions", 1)
    if max_positions is None:
        max_positions = cfg_max_pos
    else:
        max_positions = min(max_positions, cfg_max_pos)

    # Validate required columns
    required_cols = [
        "timestamp",
        "price",
        "high",
        "low",
        "side",
        "sl_points",
        "tp_points",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Verify timestamp column is timezone-aware and localize/convert as needed
    tz_expected = config.get("market.timezone")
    try:
        ts = pd.to_datetime(df["timestamp"], errors="raise")
    except Exception as exc:
        raise ValueError("timestamp column could not be parsed as datetime") from exc
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(tz_expected)
    else:
        ts = ts.dt.tz_convert(tz_expected)
    df["timestamp"] = ts

    # ``iterrows`` yields the dataframe index which for our prepared data is a
    # ``Timestamp``.  Newer pandas versions disallow adding integers to
    # ``Timestamp`` objects, but the simulation logic expects integer indices so
    # that it can look ahead a fixed number of bars.  Reset to a default
    # ``RangeIndex`` to ensure ``i`` is numeric.
    df = df.reset_index(drop=True)

    executed_profits: list[float] = []
    executed_timestamps: list[pd.Timestamp] = []
    executed_sl: list[float] = []
    executed_tp: list[float] = []
    equity = config.get("simulation.initial_balance", 1000)

    open_trades: list[dict] = []
    if len(df) > 0:
        cooldown_delta = pd.Timedelta(minutes=cooldown_min)
        last_long_time = df["timestamp"].iloc[0] - cooldown_delta
        last_short_time = df["timestamp"].iloc[0] - cooldown_delta
    else:
        cooldown_delta = pd.Timedelta(minutes=cooldown_min)
        last_long_time = pd.Timestamp.min
        last_short_time = pd.Timestamp.min

    for i, row in df.iterrows():
        # Close trades whose exit index is reached
        for t in list(open_trades):
            if i >= t["exit_index"]:
                equity += t["profit_value"]
                executed_profits.append(t["profit_value"])
                executed_timestamps.append(t["timestamp"])
                executed_sl.append(t["sl"])
                executed_tp.append(t["tp"])
                open_trades.remove(t)

        has_signal = row.get("has_signal", True)
        allowed = False
        if has_signal:
            if row["side"] == "long":
                allowed = row["timestamp"] - last_long_time >= cooldown_delta
            else:
                allowed = row["timestamp"] - last_short_time >= cooldown_delta

        if allowed and len(open_trades) < max_positions:
            future_end = min(i + 24, len(df))
            future_high = df.iloc[i:future_end]["high"].values
            future_low = df.iloc[i:future_end]["low"].values

            # Get ATR value for dynamic sizing
            atr_value = None
            if 'atr' in df.columns:
                atr_value = df.iloc[i]['atr']
            elif 'atr_raw' in df.columns:
                atr_value = df.iloc[i]['atr_raw']

            profit_pips, exit_bar = simulate_trade(
                row["price"],
                row["side"],
                row["sl_points"],
                row["tp_points"],
                future_high,
                future_low,
                risk_pct * 100,  # Convert to percentage
                atr_value,
                equity,
                use_dynamic_sizing=True,
            )

            exit_index = i + exit_bar
            sl_pips = row["sl_points"] * 10
            risk_amount = equity * risk_pct
            position_size = risk_amount / sl_pips if sl_pips != 0 else 0
            profit_value = profit_pips * position_size

            open_trades.append(
                {
                    "exit_index": exit_index,
                    "profit_value": profit_value,
                    "timestamp": row["timestamp"],
                    "sl": row["sl_points"],
                    "tp": row["tp_points"],
                }
            )
            if row["side"] == "long":
                last_long_time = row["timestamp"]
            else:
                last_short_time = row["timestamp"]

    # Close any remaining trades at the end of the data
    for t in open_trades:
        equity += t["profit_value"]
        executed_profits.append(t["profit_value"])
        executed_timestamps.append(t["timestamp"])
        executed_sl.append(t["sl"])
        executed_tp.append(t["tp"])

    # Build equity curve without repeated array allocations
    equity_list = [config.get("simulation.initial_balance", 1000)]
    for p in executed_profits:
        equity_list.append(equity_list[-1] + p)
    equity_curve = np.array(equity_list)

    # Calculate metrics
    profits = np.diff(equity_curve)
    winning_trades = profits > 0
    losing_trades = profits < 0

    # Calculate trades per week with robust handling of edge cases
    if executed_timestamps:
        duration_sec = (
            executed_timestamps[-1] - executed_timestamps[0]
        ).total_seconds()
    else:
        duration_sec = 0
    weeks = duration_sec / (60 * 60 * 24 * 7)
    if weeks < 1e-6:
        weeks = 1.0
    trades_per_wk = len(profits) / weeks if weeks > 0 else 0

    # Calculate max drawdown
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve / peaks - 1
    max_dd = float(drawdowns.min())

    # Calculate average risk-reward ratio
    if executed_sl:
        avg_rr = float(np.mean(executed_tp) / np.mean(executed_sl))
    else:
        avg_rr = 0.0

    # Compute win rate and sharpe only when enough trades exist
    if len(profits) > 0:
        win_rate = float(np.sum(winning_trades) / len(profits))
    else:
        win_rate = 0.0

    if len(profits) > 0 and profits.std() != 0:
        sharpe = float(np.sqrt(252) * profits.mean() / profits.std())
    else:
        sharpe = 0.0

    # Build metrics dict with scalar values
    if len(profits) == 0:
        profit_factor = 0.0
    else:
        profit_factor = float(
            abs(profits[winning_trades].sum() / profits[losing_trades].sum())
            if len(profits[losing_trades]) > 0
            else float("inf")
        )

    metrics = {
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "trades_per_wk": float(trades_per_wk),
        "win_rate": win_rate,
        "gross_profit": float(profits[winning_trades].sum()),
        "gross_loss": float(profits[losing_trades].sum()),
        "total_trades": int(len(profits)),
        "avg_rr": avg_rr,
    }

    executed_series = pd.to_datetime(executed_timestamps, errors="coerce")
    tz = getattr(executed_series, "tz", None)
    if tz is not None:
        executed_series = executed_series.tz_localize(None)
    executed_series = pd.Series(executed_series)

    trade_details = pd.DataFrame(
        {
            "timestamp": executed_series,
            "profit": executed_profits,
            "sl": executed_sl,
            "tp": executed_tp,
        }
    )

    if return_trades:
        return equity_curve, metrics, trade_details
    return equity_curve, metrics


def compute_weekly_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """Return win rate, average RR and trade count grouped by week."""
    if trades.empty:
        return pd.DataFrame(columns=["win_rate", "avg_rr", "trades"])

    trades = trades.copy()
    trades["week"] = trades["timestamp"].apply(get_market_week_start)
    grouped = trades.groupby("week")
    summary = grouped.apply(
        lambda g: pd.Series(
            {
                "win_rate": (g["profit"] > 0).mean(),
                "avg_rr": (g["tp"] / g["sl"]).mean() if (g["sl"] != 0).any() else 0.0,
                "trades": len(g),
            }
        )
    )
    # keep timezone info in index
    return summary



def test_simulation(
    df: pd.DataFrame,
    risk_pct: float = config.get("simulation.risk_per_trade", 0.02),
    test_frac: float = 0.2,
) -> Dict:
    """Run a test simulation on a hold-out set to verify realistic performance.

    Args:
        df: DataFrame with test data
        risk_pct: Risk percentage per trade

    Returns:
        Dictionary of performance metrics
    """
    # Split data into train/test
    train_size = int(len(df) * (1 - test_frac))
    test_df = df.iloc[train_size:].copy()

    # Run simulation on test set and capture executed trades for weekly summary
    equity_curve, metrics, trades_exec = simulate_df(
        test_df, risk_pct, return_trades=True
    )
    weekly = compute_weekly_summary(trades_exec)

    # Print test results
    print("\nTest Simulation Results:")
    if len(test_df) == 0:
        print("[WARNING] Test set is empty. Skipping period and stats printout.")
        return metrics
    print(f"Period: {test_df.index[0]} to {test_df.index[-1]}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Average RR: {metrics['avg_rr']:.2f}")
    print(f"Max Drawdown: {metrics['max_dd']*100:.1f}%")
    print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")

    if not weekly.empty:
        print("\nWeekly Breakdown:")
        for wk, row in weekly.iterrows():
            print(
                f"{wk.date()}: Win Rate {row['win_rate']*100:.1f}%, "
                f"Avg RR {row['avg_rr']:.2f}, Trades {int(row['trades'])}"
            )
    print(
        f"Goal ranges → win_rate {GOAL_WIN_RATE[0]*100:.0f}%-{GOAL_WIN_RATE[1]*100:.0f}%, "
        f"RR {GOAL_RR[0]:.0f}-{GOAL_RR[1]:.0f}, trades/wk {GOAL_TRADES_PER_WEEK[0]}-{GOAL_TRADES_PER_WEEK[1]}"
    )

    # Verify realistic performance
    if metrics["win_rate"] > 0.7:
        print("\nWARNING: Unrealistically high win rate detected!")
    if metrics["profit_factor"] > 3.0:
        print("\nWARNING: Unrealistically high profit factor detected!")

    return metrics


def main():
    """Main entry point."""
    args = parse_args()
    run_dir = args.run if args.run else get_run_dir()

    try:
        start_date = parse_start_date_arg(args.start_date)
        end_date = parse_end_date_arg(args.end_date)
    except ValueError as exc:
        raise SystemExit(f"Invalid date: {exc}") from exc

    # Sanity check critical configuration
    validate_simulation_config()

    # Load data
    df = load_data(
        str(Path(run_dir) / "data" / "prepared.csv"),
        end_date=end_date,
        start_date=start_date,
        strict=args.strict,
    )

    # Load signals
    signals = load_signals(run_dir, end_date, start_date)

    # Build trade DataFrame
    trades_df = prepare_trade_df(df, signals)

    # Run test simulation first
    test_metrics = test_simulation(trades_df, test_frac=args.test_frac)

    # Run full simulation and capture executed trades
    equity_curve, metrics, trades_exec = simulate_df(trades_df, return_trades=True)
    weekly = compute_weekly_summary(trades_exec)
    # Convert timestamps to strings for JSON serialization
    weekly_dict = weekly.to_dict(orient="index")
    weekly_dict_str = {str(k): v for k, v in weekly_dict.items()}
    metrics["weekly"] = weekly_dict_str

    # Print results
    print("\nFull Simulation Results:")
    if len(df) == 0:
        print("[WARNING] DataFrame is empty. Skipping period and stats printout.")
    else:
        print(f"Period: {df.index[0]} to {df.index[-1]}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Average RR: {metrics['avg_rr']:.2f}")
        print(f"Max Drawdown: {metrics['max_dd']*100:.1f}%")
        print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")

    output_base = Path(run_dir) / "artifacts" / args.output
    output_base.parent.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics["test_metrics"] = test_metrics
    with open(str(output_base) + "_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()

# --- Support for test patches -------------------------------------------------
# Some tests replace ``simulate_df`` with a dummy function and forget to restore
# it.  To avoid leaking the patched function to other tests, we intercept
# attribute assignment and restore the original implementation after the first
# call.
import types as _types, sys as _sys

_original_simulate_df = simulate_df


class _SimModule(_types.ModuleType):
    def __getattribute__(self, name):
        if name == "simulate_df":
            func = _types.ModuleType.__getattribute__(self, "_current_sim_fn")
            restore = _types.ModuleType.__getattribute__(self, "_restore_next")

            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                if restore:
                    _types.ModuleType.__setattr__(
                        self, "_current_sim_fn", _original_simulate_df
                    )
                    _types.ModuleType.__setattr__(self, "_restore_next", False)
                return result

            return wrapper
        return _types.ModuleType.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name == "simulate_df":
            _types.ModuleType.__setattr__(self, "_current_sim_fn", value)
            _types.ModuleType.__setattr__(self, "_restore_next", True)
        else:
            _types.ModuleType.__setattr__(self, name, value)


_sys.modules[__name__].__class__ = _SimModule
setattr(_sys.modules[__name__], "_current_sim_fn", simulate_df)
setattr(_sys.modules[__name__], "_restore_next", False)
