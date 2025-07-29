"""SL/TP Model for Optimal Trade Exit Points

This module implements a LightGBM model to predict the optimal SL/TP bucket
for each trade signal. For each signal, it simulates all valid SL/TP combinations
from the config grid and selects the best performing one.

Key Design Points:
- Simulates all valid SL/TP combinations per signal
- Selects best bucket based on risk-reward score
- Trains separate models per (regime, side) combination
- Uses macro precision and per-bucket accuracy for evaluation
- Requires minimum 200 samples per model
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score
import lightgbm as lgb
import pickle, json, os

from config import config
from utils import (
    parse_end_date_arg,
    parse_start_date_arg,
    ensure_dir,
    get_run_dir,
    make_run_dirs,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === Training Configuration ===
MIN_SAMPLES = 200  # Minimum samples required to train a model
TEST_SPLIT = 0.2   # Validation split ratio
CHUNK_SIZE = 10000  # Process data in chunks

# Get grid parameters from config
SPREAD = config.get('sl_tp_grid.spread')
SL_MULTIPLIERS = config.get('sl_tp_grid.sl_multipliers')
TP_MULTIPLIERS = config.get('sl_tp_grid.tp_multipliers')

# Create valid SL/TP combinations
SL_TP_PAIRS = []
for sl_mult in SL_MULTIPLIERS:
    for tp_mult in TP_MULTIPLIERS:
        sl_pips = (SPREAD * sl_mult) / 0.0001  # Convert to pips
        tp_pips = sl_pips * tp_mult
        if tp_pips >= 2 * sl_pips:  # Only keep RR >= 2.0
            SL_TP_PAIRS.append((sl_pips, tp_pips))

# Update LightGBM parameters for new number of buckets
params = {
    'objective': 'multiclass',
    'num_class': len(SL_TP_PAIRS),  # Number of valid SL/TP combinations
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'min_data_in_leaf': 30,
    'min_sum_hessian_in_leaf': 1e-3,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'max_depth': -1,
    'num_threads': -1,
    'early_stopping_rounds': 50  # Add early stopping
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SL/TP models for each regime and side")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--start_date",
        type=str,
        required=False,
        help="Earliest bar to include (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--train_end_date",
        type=str,
        required=False,
        help="Final bar used for training (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date", type=str, required=False,
        help="YYYY-MM-DD (inclusive) last bar for training"
    )
    parser.add_argument("--run", type=str, help="Run directory (overrides RUN_ID)")
    return parser.parse_args()

def load_labeled_data(
    run_dir: str, end_date: str | None, start_date: str | None
) -> pd.DataFrame:
    """Load labeled data from CSV file.
    
    Args:
        run_dir: Path to run directory
        end_date: End date filter (YYYY-MM-DD)
        start_date: Start date filter (YYYY-MM-DD)
        
    Returns:
        DataFrame with labeled data
    """
    df = pd.read_csv(Path(run_dir) / "data" / "labeled.csv", index_col=0, parse_dates=True)
    
    # Convert string dates to Timestamp objects for comparison
    tz = config.get("market.timezone")
    if start_date and isinstance(start_date, str):
        start_date = pd.Timestamp(start_date, tz=tz)
    if end_date and isinstance(end_date, str):
        end_date = pd.Timestamp(end_date, tz=tz)
    
    # Filter by date range
    if start_date and end_date:
        df = df[(df.index >= start_date) & (df.index <= end_date)]
    elif start_date:
        df = df[df.index >= start_date]
    elif end_date:
        df = df[df.index <= end_date]
    
    # Assert required columns exist
    required_cols = [
        'label_long', 'label_short', 'atr', 'future_high', 'future_low',
        'future_close', 'market_regime'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df

def simulate_bucket(
    df: pd.DataFrame,
    bucket_idx: int,
    is_long: bool
) -> Tuple[float, float, bool]:
    """Simulate a single SL/TP bucket for a signal.
    
    Args:
        df: DataFrame with price data (window of future bars)
        bucket_idx: Index of SL/TP pair to simulate
        is_long: True for long trades, False for short
        
    Returns:
        Tuple of (tp_pips, sl_pips, hit_tp)
    """
    # Get SL/TP values for this bucket
    sl_pips, tp_pips = SL_TP_PAIRS[bucket_idx]
    
    # Calculate TP/SL levels
    if is_long:
        tp_level = df['close'].iloc[0] + tp_pips/10000
        sl_level = df['close'].iloc[0] - sl_pips/10000
    else:
        tp_level = df['close'].iloc[0] - tp_pips/10000
        sl_level = df['close'].iloc[0] + sl_pips/10000
    
    # Find first touch within available bars
    n_bars = len(df)
    tp_touch = n_bars
    sl_touch = n_bars
    
    for i in range(n_bars):
        if is_long:
            if df['future_high'].iloc[i] >= tp_level:
                tp_touch = i
                break
            if df['future_low'].iloc[i] <= sl_level:
                sl_touch = i
                break
        else:
            if df['future_low'].iloc[i] <= tp_level:
                tp_touch = i
                break
            if df['future_high'].iloc[i] >= sl_level:
                sl_touch = i
                break
    
    hit_tp = tp_touch < sl_touch
    return tp_pips, sl_pips, hit_tp

def find_best_bucket(
    df: pd.DataFrame,
    is_long: bool
) -> Tuple[int, float, float, bool]:
    """Find the best performing bucket for a signal.

    Buckets are scored using risk-reward ratio weighted by recent ATR
    (volatility) and trading session characteristics.
    
    Args:
        df: DataFrame with price data
        is_long: True for long trades, False for short
        
    Returns:
        Tuple of (best_bucket_idx, tp_pips, sl_pips, hit_tp)
    """
    # Track performance for each bucket
    bucket_stats = []

    # Get current price and ATR
    current_price = df['close'].iloc[0]
    current_atr = df['atr'].iloc[0]

    # Determine active session (if available)
    session = None
    if 'session_overlap' in df.columns and df['session_overlap'].iloc[0] == 1:
        session = 'overlap'
    elif 'session_london' in df.columns and df['session_london'].iloc[0] == 1:
        session = 'london'
    elif 'session_ny' in df.columns and df['session_ny'].iloc[0] == 1:
        session = 'ny'
    elif 'session_asian' in df.columns and df['session_asian'].iloc[0] == 1:
        session = 'asian'
    
    # Try all valid SL/TP combinations
    atr_pips = current_atr / 0.0001 if current_atr > 0 else 1.0

    for bucket_idx in range(len(SL_TP_PAIRS)):
        sl_pips, tp_pips = SL_TP_PAIRS[bucket_idx]
        
        # Calculate TP/SL levels
        if is_long:
            tp_level = current_price + tp_pips/10000
            sl_level = current_price - sl_pips/10000
        else:
            tp_level = current_price - tp_pips/10000
            sl_level = current_price + sl_pips/10000
        
        # Find first touch within available bars
        n_bars = len(df)
        tp_touch = n_bars
        sl_touch = n_bars
        
        for i in range(n_bars):
            if is_long:
                if df['future_high'].iloc[i] >= tp_level:
                    tp_touch = i
                    break
                if df['future_low'].iloc[i] <= sl_level:
                    sl_touch = i
                    break
            else:
                if df['future_low'].iloc[i] <= tp_level:
                    tp_touch = i
                    break
                if df['future_high'].iloc[i] >= sl_level:
                    sl_touch = i
                    break
        
        hit_tp = tp_touch < sl_touch
        if hit_tp:
            rr_ratio = tp_pips / sl_pips

            size_ratio = sl_pips / atr_pips if atr_pips > 0 else 1.0
            atr_weight = 1 / (1 + abs(size_ratio - 1))

            session_weight = 1.0
            if session in ('london', 'ny', 'overlap'):
                session_weight = size_ratio ** 1.5
            elif session == 'asian':
                session_weight = size_ratio ** -1.5

            score = rr_ratio * atr_weight * session_weight

            bucket_stats.append({
                'bucket': bucket_idx,
                'sl_pips': sl_pips,
                'tp_pips': tp_pips,
                'score': score,
                'hit_tp': True
            })
    
    # If no winning buckets, return -1
    if not bucket_stats:
        return -1, 0, 0, False
    
    # Sort by weighted score (descending) and take the top bucket
    best_bucket = sorted(bucket_stats, key=lambda x: x['score'], reverse=True)[0]
    return best_bucket['bucket'], best_bucket['tp_pips'], best_bucket['sl_pips'], True

def prepare_training_data(
    df_signals: pd.DataFrame,
    df_full: pd.DataFrame,
    train_end_date: str | pd.Timestamp | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare training data with best bucket labels.
    
    Args:
        df_signals: Filtered DataFrame for a specific regime/side
        df_full: Full labeled dataset (indexed by datetime)
        train_end_date: Optional cutoff for lookahead windows
        
    Returns:
        Tuple of (train_df, test_df) with best bucket labels added
    """
    # Ensure df_full.index is a timezone-aware DatetimeIndex
    if not isinstance(df_full.index, pd.DatetimeIndex):
        df_full.index = pd.to_datetime(df_full.index, utc=True)
    tz = config.get("market.timezone")
    if df_full.index.tz is None:
        df_full.index = df_full.index.tz_localize(tz)
    else:
        df_full.index = df_full.index.tz_convert(tz)
    tz = df_full.index.tz
    
    # Create a copy to avoid modifying the original
    df = df_signals.copy()
    
    # Initialize columns for best bucket results
    df['best_bucket'] = -1
    df['best_tp_pips'] = np.nan
    df['best_sl_pips'] = np.nan
    df['best_hit_tp'] = False
    valid_idx: List[pd.Timestamp] = []
    
    # Track bucket distribution and price movement
    bucket_counts = []
    price_moves = []
    
    # Process each signal
    for i, (idx, row) in enumerate(df.iterrows()):
        # Find this signal's position in df_full
        try:
            full_loc = df_full.index.get_loc(idx)
        except KeyError:
            continue  # skip if not found
            
        # Get window of future bars
        window_df = df_full.iloc[full_loc:full_loc+24]
        if len(window_df) < 2:  # need at least 2 bars to simulate
            continue
        if train_end_date is not None:
            end_ts = (
                pd.Timestamp(train_end_date, tz=tz)
                if not isinstance(train_end_date, pd.Timestamp)
                else train_end_date.tz_convert(tz)
            )
            if window_df.index[-1] > end_ts:
                continue
            
        # Calculate max price movement in first bar
        first_bar_high = window_df['future_high'].iloc[0]
        first_bar_low = window_df['future_low'].iloc[0]
        first_bar_move = (first_bar_high - first_bar_low) * 10000  # convert to pips
        price_moves.append(first_bar_move)
            
        # Determine if this is a long or short signal
        is_long = row['label_long'] == 1
        
        # Find best bucket for this signal
        best_bucket, tp_pips, sl_pips, hit_tp = find_best_bucket(window_df, is_long)
        
        # Update results
        df.loc[idx, 'best_bucket'] = best_bucket
        df.loc[idx, 'best_tp_pips'] = tp_pips
        df.loc[idx, 'best_sl_pips'] = sl_pips
        df.loc[idx, 'best_hit_tp'] = hit_tp
        
        if best_bucket >= 0:
            bucket_counts.append(best_bucket)
        valid_idx.append(idx)
    
    # Split into train/test sets (80/20)
    df = df.loc[valid_idx]
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    # Print statistics
    print(f"\nProcessed {len(df)} signals")
    print(f"Average first bar movement: {np.mean(price_moves):.1f} pips")
    if bucket_counts:
        print("\nBucket distribution:")
        for bucket in sorted(set(bucket_counts)):
            count = bucket_counts.count(bucket)
            pct = (count / len(bucket_counts)) * 100
            sl_pips, tp_pips = SL_TP_PAIRS[bucket]
            print(f"Bucket {bucket}: {count} signals ({pct:.1f}%) - SL: {sl_pips:.1f} pips, TP: {tp_pips:.1f} pips")
    
    return train_df, test_df

def filter_by_regime_side(
    df: pd.DataFrame,
    regime: int,
    side: str
) -> pd.DataFrame:
    """Filter data for specific regime and trading side.
    
    Args:
        df: Input DataFrame
        regime: Regime ID (0-3)
        side: 'long' or 'short'
        
    Returns:
        Filtered DataFrame
    """
    # Filter by regime
    df = df[df['market_regime'] == regime].copy()
    
    # Filter by side
    if side == 'long':
        df = df[df['label_long'] == 1]
    else:  # short
        df = df[df['label_short'] == 1]
    
    return df

def train_per_regime_side(
    df: pd.DataFrame,
    regime: int,
    side: str,
    seed: int
) -> Tuple[LGBMClassifier, Dict]:
    """Train SL/TP model for specific regime and side.
    
    Args:
        df: Training data (already filtered for regime and side)
        regime: Regime ID (0-3)
        side: 'long' or 'short'
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (trained model, metrics dictionary)
    """
    # Check minimum samples
    if len(df) < MIN_SAMPLES:
        logger.warning(f"Insufficient samples for regime {regime}, {side} side: {len(df)} < {MIN_SAMPLES}")
        return None, None
    
    # Load features from config
    feature_cols = config.get('sltp.features')
    
    # Prepare data
    X = df[feature_cols]
    y = df['best_bucket']
    
    # Time-based split
    split_idx = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Drop validation samples whose label is not in training set
    valid_labels = set(y_train.unique())
    val_mask = y_val.isin(valid_labels)
    X_val = X_val[val_mask]
    y_val = y_val[val_mask]
    
    logger.info(f"Train set: {len(X_train):,} rows")
    logger.info(f"Val set: {len(X_val):,} rows")
    
    # Initialize and train model
    model = LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False)
        ]
    )
    
    # Get predictions and calculate metrics
    val_pred = model.predict(X_val)
    
    # Calculate per-bucket metrics
    bucket_metrics = {}
    for bucket in range(len(SL_TP_PAIRS)):
        mask = y_val == bucket
        if mask.any():
            y_true_bin = y_val == bucket
            if len(np.unique(y_true_bin)) > 1:
                prec = precision_score(y_true_bin, val_pred == bucket)
                rec = recall_score(y_true_bin, val_pred == bucket)
            else:  # avoid UndefinedMetricWarning when only one class
                prec = 0.0
                rec = 0.0
            bucket_metrics[f'bucket_{bucket}'] = {
                'precision': float(prec),
                'recall': float(rec),
                'support': int(mask.sum())
            }
    
    # Calculate macro metrics
    if len(np.unique(y_val)) > 1:
        macro_prec = precision_score(y_val, val_pred, average='macro', zero_division=0)
        macro_rec = recall_score(y_val, val_pred, average='macro', zero_division=0)
    else:
        macro_prec = 0.0
        macro_rec = 0.0
    
    # Create metrics dictionary
    metrics = {
        'macro_precision': float(macro_prec),
        'macro_recall': float(macro_rec),
        'bucket_metrics': bucket_metrics,
        'best_iteration': int(model.best_iteration_),
        'train_samples': {
            'total': len(y_train),
            'per_bucket': y_train.value_counts().to_dict()
        },
        'val_samples': {
            'total': len(y_val),
            'per_bucket': y_val.value_counts().to_dict()
        }
    }
    
    return model, metrics

def save_artifacts(
    model: LGBMClassifier,
    metrics: Dict,
    regime: int,
    side: str,
    run_dir: str
) -> None:
    """Save model, metrics, and feature importances."""
    # Save model
    model_path = Path(run_dir) / 'models' / f'sltp_regime{regime}_{side}.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metrics
    metrics_path = Path(run_dir) / 'models' / f'metrics_sltp_regime{regime}_{side}.json'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    # Save feature importances
    feature_cols = config.get('sltp.features')
    importances = dict(zip(feature_cols, [int(x) for x in model.feature_importances_]))
    importance_dir = Path(run_dir) / 'artifacts' / 'feature_importance'
    importance_dir.mkdir(parents=True, exist_ok=True)
    importance_path = importance_dir / f"sltp_regime{regime}_{side}.json"
    with open(importance_path, 'w') as f:
        json.dump(importances, f, indent=2)

def main():
    """Main training function."""
    args = parse_args()
    seed = args.seed
    try:
        start_date = parse_start_date_arg(args.start_date)
        end_date = parse_end_date_arg(args.end_date)
        train_end_date = parse_end_date_arg(args.train_end_date) or end_date
    except ValueError as exc:
        raise SystemExit(f"Invalid date: {exc}") from exc

    logger.info("Starting SL/TP model training...")
    logger.info(f"Using seed: {seed}")
    if end_date:
        logger.info(f"Training until: {end_date}")
    
    # Load data
    logger.info("Loading labeled data...")
    run_dir = args.run if args.run else get_run_dir()
    make_run_dirs(run_dir)
    df_all = load_labeled_data(run_dir, end_date, start_date)
    
    # Convert train_end_date to Timestamp if it's a string
    tz = config.get("market.timezone")
    if train_end_date and isinstance(train_end_date, str):
        train_end_date = pd.Timestamp(train_end_date, tz=tz)
    
    df = df_all[df_all.index <= train_end_date] if train_end_date else df_all
    logger.info(f"Loaded {len(df):,} samples")
    
    # Get unique regimes
    regimes = sorted(df['market_regime'].unique())
    logger.info(f"Found regimes: {regimes}")
    
    # Train models for each regime and side
    for regime in regimes:
        logger.info(f"\nProcessing regime {regime}...")
        
        for side in ['long', 'short']:
            logger.info(f"\nTraining {side} model for regime {regime}...")
            
            # Filter data for this regime/side
            df_filtered = filter_by_regime_side(df, regime, side)
            if len(df_filtered) < MIN_SAMPLES:
                logger.warning(f"Not enough samples for regime {regime} {side} ({len(df_filtered)} < {MIN_SAMPLES})")
                continue
            
            logger.info(f"Found {len(df_filtered):,} samples for regime {regime} {side}")
            
            # Process data in chunks
            train_parts: List[pd.DataFrame] = []
            test_parts: List[pd.DataFrame] = []
            for chunk_start in range(0, len(df_filtered), CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, len(df_filtered))
                chunk = df_filtered.iloc[chunk_start:chunk_end]

                # Prepare training data for this chunk
                train_chunk, test_chunk = prepare_training_data(
                    chunk, df_all, train_end_date
                )
                if len(train_chunk) > 0:
                    train_parts.append(train_chunk)
                    test_parts.append(test_chunk)
                # Clear references to reduce memory usage
                del chunk, train_chunk, test_chunk

            if not train_parts:
                logger.warning(f"No valid training data for regime {regime} {side}")
                continue

            # Combine chunk results
            train_df = pd.concat(train_parts)
            test_df = pd.concat(test_parts)
            train_parts.clear()
            test_parts.clear()
            
            logger.info(f"Training on {len(train_df):,} samples, testing on {len(test_df):,} samples")
            
            # Train model
            model, metrics = train_per_regime_side(train_df, regime, side, seed)
            
            # Save artifacts
            save_artifacts(model, metrics, regime, side, run_dir)
            
            logger.info(f"Completed training for regime {regime} {side}")
            logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
    
    logger.info("\nSL/TP model training completed!")

if __name__ == "__main__":
    main() 
