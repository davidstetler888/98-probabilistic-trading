import argparse
import json
import pickle
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from config import config
from utils import (
    parse_end_date_arg,
    parse_start_date_arg,
    load_data,
    get_run_dir,
    make_run_dirs,
    ensure_dir,
)
from sltp import SL_TP_PAIRS
from ranker_integration import apply_win_rate_enhancements_to_signals


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank trade signals via meta probabilities and SL/TP models"
    )
    parser.add_argument("--run", type=str, help="Run directory (overrides RUN_ID)")
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
    parser.add_argument("--end_date", type=str, required=False, help="YYYY-MM-DD last date for signals")
    parser.add_argument(
        "--target_trades_per_week",
        type=float,
        required=False,
        help="Desired trades per week",
    )
    parser.add_argument(
        "--min_trades_per_week",
        type=float,
        required=False,
        help="Minimum trades per week",
    )
    parser.add_argument(
        "--max_trades_per_week",
        type=float,
        required=False,
        help="Maximum trades per week",
    )
    parser.add_argument(
        "--edge_threshold",
        type=float,
        required=False,
        help="Use this edge score threshold instead of auto search",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort if CSV parsing fails",
    )
    return parser.parse_args()


def get_dynamic_threshold(probs, target_trades_per_week, total_signals, weeks):
    """
    Calculate threshold based on target trades per week rather than fixed criteria
    """
    current_trades_per_week = total_signals / weeks
    
    if current_trades_per_week < target_trades_per_week:
        # Be more aggressive - lower threshold
        percentile = min(90, 70 + (target_trades_per_week - current_trades_per_week) * 2)
    else:
        # Be more conservative - higher threshold
        percentile = max(50, 70 - (current_trades_per_week - target_trades_per_week) * 2)
    
    return np.percentile(probs, percentile)


def find_edge_threshold_dynamic(probs, target_trades_per_week=40):
    """Enhanced threshold finding with dynamic adjustment"""
    weeks = len(probs) / (7 * 24 * 12)  # 5-minute bars
    
    # Start with percentile-based approach
    threshold = get_dynamic_threshold(probs, target_trades_per_week, len(probs), weeks)
    
    # Apply minimum viability check
    valid_signals = probs >= threshold
    if valid_signals.sum() < (target_trades_per_week * 0.5 * weeks):
        # Too few signals, lower threshold
        threshold = np.percentile(probs, 60)
    
    return threshold


def calculate_signal_score(row):
    """
    Calculate composite signal score based on multiple factors
    """
    score = 0
    
    # Base probability score (0-40 points)
    score += row['meta_prob'] * 40
    
    # Risk-reward bonus (0-20 points)
    rr_ratio = row['tp_pips'] / row['sl_pips']
    score += min(rr_ratio / 3.0, 1.0) * 20
    
    # Volatility bonus (0-15 points) - use raw ATR if available
    if 'atr_raw' in row.index:
        volatility_score = min(row['atr_raw'] / 0.0015, 1.0) * 15
    else:
        volatility_score = 7.5  # Default medium score
    score += volatility_score
    
    # Session bonus (0-10 points)
    hour = row.name.hour if hasattr(row.name, 'hour') else 12
    if 8 <= hour < 13 or 13 <= hour < 18:  # London or NY
        score += 10
    elif 22 <= hour or hour < 8:  # Asian
        score += 5
    
    # Multi-timeframe alignment bonus (0-15 points)
    if 'mtf_alignment' in row.index:
        score += row['mtf_alignment'] * 15
    elif 'htf_15min_trend' in row.index and 'htf_30min_trend' in row.index:
        # Calculate alignment based on trend directions
        if row['side'] == 'long':
            if row['htf_15min_trend'] > 0 and row['htf_30min_trend'] > 0:
                score += 15
            elif row['htf_15min_trend'] > 0 or row['htf_30min_trend'] > 0:
                score += 7.5
        else:  # short
            if row['htf_15min_trend'] < 0 and row['htf_30min_trend'] < 0:
                score += 15
            elif row['htf_15min_trend'] < 0 or row['htf_30min_trend'] < 0:
                score += 7.5
    else:
        # Default medium score if no HTF data
        score += 7.5
    
    return score


def get_session_multiplier(timestamp):
    """Get session-specific threshold multiplier"""
    hour = timestamp.hour
    
    # Asian session (lower volatility, more aggressive)
    if 22 <= hour or hour < 8:
        return 0.85
    
    # London session (high volatility, standard)
    elif 8 <= hour < 13:
        return 1.0
    
    # NY session (high volatility, standard)
    elif 13 <= hour < 18:
        return 1.0
    
    # Overlap periods (highest volatility, more conservative)
    elif 12 <= hour < 14:  # London-NY overlap
        return 1.15
    
    # Off-hours (very low volatility, more aggressive)
    else:
        return 0.9


def apply_session_filters(df):
    """Apply session-specific threshold adjustments"""
    if df.empty:
        return df
    df['session_multiplier'] = df.index.map(get_session_multiplier)

    # Adjust edge thresholds based on session
    if 'edge_score' in df.columns:
        df['edge_score_adjusted'] = df['edge_score'] * df['session_multiplier']
    
    return df


def rank_signals_enhanced(signals_df, target_trades_per_week):
    """Enhanced signal ranking with multiple factors"""
    
    # Calculate composite scores
    signals_df['signal_score'] = signals_df.apply(calculate_signal_score, axis=1)
    
    # Sort by score (descending)
    signals_df = signals_df.sort_values('signal_score', ascending=False)
    
    # Select top signals based on target
    weeks = len(signals_df) / (7 * 24 * 12)
    target_signals = int(target_trades_per_week * weeks)
    
    selected_signals = signals_df.head(target_signals)
    
    return selected_signals


def load_meta_models(run_dir: str) -> dict:
    model_path = Path(run_dir) / "models" / "meta.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_sltp_models(run_dir: str) -> dict:
    models = {}
    base = Path(run_dir) / "models"
    for regime in range(4):
        for side in ["long", "short"]:
            path = base / f"sltp_regime{regime}_{side}.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    models[(regime, side)] = pickle.load(f)
    return models


def predict_sltp_bucket(model, features: pd.DataFrame) -> np.ndarray:
    try:
        return model.predict(features)
    except Exception:
        # In case model does not support predict on empty frame
        return np.array([])


def find_threshold(
    scores: pd.Series,
    win_probs: pd.Series,
    rr: pd.Series,
    target: float,
    min_per_week: float,
    max_per_week: float,
    win_range: tuple[float, float],
    rr_range: tuple[float, float],
) -> tuple[float, float, float, float]:
    if scores.empty:
        return 0.0, 0.0, 0.0, 0.0
    unique = np.sort(scores.unique())
    start, end = scores.index.min(), scores.index.max()
    weeks = max((end - start).days / 7, 1e-9)

    original_min = min_per_week
    best = None

    while min_per_week >= 0:
        best_thr = unique[0]
        best_diff = float("inf")
        best_tw = 0.0
        best_wr = 0.0
        best_rr = 0.0
        for thr in unique:
            mask = scores >= thr
            tw = mask.sum() / weeks
            wr = win_probs[mask].mean() if mask.any() else 0.0
            rr_val = rr[mask].mean() if mask.any() else 0.0
            if (
                min_per_week <= tw <= max_per_week
                and win_range[0] <= wr <= win_range[1]
                and rr_range[0] <= rr_val <= rr_range[1]
                and abs(tw - target) < best_diff
            ):
                best_diff = abs(tw - target)
                best_thr = thr
                best_tw = tw
                best_wr = wr
                best_rr = rr_val
        if best_diff != float("inf"):
            best = (best_thr, best_tw, best_wr, best_rr)
            break
        min_per_week -= 1

    if best is None:
        # no threshold met the range even after relaxing; choose closest to target
        best_thr = unique[0]
        best_diff = float("inf")
        best_tw = 0.0
        for thr in unique:
            tw = (scores >= thr).sum() / weeks
            diff = abs(tw - target)
            if diff < best_diff:
                best_diff = diff
                best_thr = thr
                best_tw = tw
        mask = scores >= best_thr
        best_wr = win_probs[mask].mean() if mask.any() else 0.0
        best_rr = rr[mask].mean() if mask.any() else 0.0
        best = (best_thr, best_tw, best_wr, best_rr)

    if min_per_week + 1 <= original_min:
        warnings.warn(
            f"Relaxed min_trades_per_week from {original_min} to {min_per_week + 1}",
            UserWarning,
        )

    thr, tw, wr_val, rr_val = best
    return float(thr), float(tw), float(wr_val), float(rr_val)


def find_threshold_consensus(
    train_scores: pd.Series,
    hold_scores: pd.Series,
    train_win_probs: pd.Series,
    hold_win_probs: pd.Series,
    train_rr: pd.Series,
    hold_rr: pd.Series,
    target: float,
    min_per_week: float,
    max_per_week: float,
    win_range: tuple[float, float],
    rr_range: tuple[float, float],
) -> tuple[float, float, float, float, float, float, float] | None:
    """Return threshold meeting range on train & hold, or None."""
    if train_scores.empty:
        return None
    if hold_scores.empty:
        return None

    unique = np.sort(
        np.unique(np.concatenate([train_scores.values, hold_scores.values]))
    )
    start_train, end_train = train_scores.index.min(), train_scores.index.max()
    weeks_train = max((end_train - start_train).days / 7, 1e-9)
    start_hold, end_hold = hold_scores.index.min(), hold_scores.index.max()
    weeks_hold = max((end_hold - start_hold).days / 7, 1e-9)

    original_min = min_per_week
    best = None

    while min_per_week >= 0 and best is None:
        best_diff = float("inf")
        for thr in unique:
            mask_train = train_scores >= thr
            mask_hold = hold_scores >= thr
            tw_train = mask_train.sum() / weeks_train
            tw_hold = mask_hold.sum() / weeks_hold
            wr_train = train_win_probs[mask_train].mean() if mask_train.any() else 0.0
            wr_hold = hold_win_probs[mask_hold].mean() if mask_hold.any() else 0.0
            rr_train_val = train_rr[mask_train].mean() if mask_train.any() else 0.0
            rr_hold_val = hold_rr[mask_hold].mean() if mask_hold.any() else 0.0
            if (
                min_per_week <= tw_train <= max_per_week
                and min_per_week <= tw_hold <= max_per_week
                and win_range[0] <= wr_train <= win_range[1]
                and win_range[0] <= wr_hold <= win_range[1]
                and rr_range[0] <= rr_train_val <= rr_range[1]
                and rr_range[0] <= rr_hold_val <= rr_range[1]
            ):
                diff = abs(tw_train - target)
                if diff < best_diff:
                    best_diff = diff
                    best = (
                        float(thr),
                        float(tw_train),
                        float(tw_hold),
                        float(wr_train),
                        float(wr_hold),
                        float(rr_train_val),
                        float(rr_hold_val),
                    )
        if best is None:
            min_per_week -= 1

    if best is None:
        # fallback: choose threshold closest to target with at least one holdout trade
        best_diff = float("inf")
        for thr in unique:
            mask_train = train_scores >= thr
            mask_hold = hold_scores >= thr
            if not mask_hold.any():
                continue
            tw_train = mask_train.sum() / weeks_train
            tw_hold = mask_hold.sum() / weeks_hold
            wr_train = train_win_probs[mask_train].mean() if mask_train.any() else 0.0
            wr_hold = hold_win_probs[mask_hold].mean() if mask_hold.any() else 0.0
            rr_train_val = train_rr[mask_train].mean() if mask_train.any() else 0.0
            rr_hold_val = hold_rr[mask_hold].mean() if mask_hold.any() else 0.0
            diff = abs(tw_train - target)
            if diff < best_diff:
                best_diff = diff
                best = (
                    float(thr),
                    float(tw_train),
                    float(tw_hold),
                    float(wr_train),
                    float(wr_hold),
                    float(rr_train_val),
                    float(rr_hold_val),
                )

    if min_per_week + 1 <= original_min:
        warnings.warn(
            f"Relaxed min_trades_per_week from {original_min} to {min_per_week + 1}",
            UserWarning,
        )

    return best


def main():
    args = parse_args()
    run_dir = args.run if args.run else get_run_dir()
    make_run_dirs(run_dir)
    start_date = parse_start_date_arg(args.start_date)
    end_date = parse_end_date_arg(args.end_date)
    train_end_date = parse_end_date_arg(args.train_end_date) or end_date

    target = config.get("ranker.target_trades_per_week", 40)
    min_per_week = config.get("ranker.min_trades_per_week", 25)
    max_per_week = config.get("ranker.max_trades_per_week", 50)
    if args.target_trades_per_week is not None:
        target = args.target_trades_per_week
    if args.min_trades_per_week is not None:
        min_per_week = args.min_trades_per_week
    if args.max_trades_per_week is not None:
        max_per_week = args.max_trades_per_week

    labeled = load_data(
        str(Path(run_dir) / "data" / "labeled.csv"), end_date, start_date, strict=args.strict
    )
    probs = load_data(
        str(Path(run_dir) / "data" / "probs_base.csv"), end_date, start_date, strict=args.strict
    )
    df_all = labeled.join(probs, how="inner")
    dropped_labeled = len(labeled) - len(df_all)
    dropped_probs = len(probs) - len(df_all)
    if dropped_labeled or dropped_probs:
        print(
            f"[ranker] join discarded {dropped_labeled + dropped_probs} rows "
            f"({dropped_labeled} labeled, {dropped_probs} probs)"
        )
    df_train = df_all[df_all.index <= train_end_date] if train_end_date else df_all

    meta_models = load_meta_models(run_dir)

    missing_models = {s for s in ["long", "short"] if s not in meta_models}
    if missing_models:
        warnings.warn(
            "Meta models missing for {} - using base probabilities".format(
                ", ".join(sorted(missing_models))
            ),
            UserWarning,
        )

    sltp_models = load_sltp_models(run_dir)

    feature_cols = list(probs.columns) + ["atr_pct", "hour", "market_regime"]
    sltp_features = config.get("sltp.features")

    signal_frames = []
    for side in ["long", "short"]:
        if side in meta_models:
            meta_prob = meta_models[side].predict(df_all[feature_cols])
        else:
            col = f"prob_{side}_lgbm"
            if col in df_all:
                meta_prob = df_all[col].values
            else:
                meta_prob = np.full(len(df_all), 0.5)
            print(f"[ranker] {side} model missing - using base probabilities")

        df_side = df_all.copy()
        df_side["meta_prob"] = meta_prob
        df_side["side"] = side
        
        # Initialize bucket column with NaN
        df_side["bucket"] = np.nan
        
        # Try to get SL/TP predictions from models
        for regime in df_side["market_regime"].unique():
            model = sltp_models.get((regime, side))
            if model is None:
                continue
            rows = df_side["market_regime"] == regime
            feats = df_side.loc[rows, sltp_features]
            preds = predict_sltp_bucket(model, feats)
            if len(preds) == len(feats):
                df_side.loc[rows, "bucket"] = preds
        
        # Use default SL/TP values for rows without bucket predictions
        # Default to bucket 6 (SL: 3.1 pips, TP: 10.0 pips) for missing predictions
        default_bucket = 6
        missing_buckets = df_side["bucket"].isna().sum()
        df_side.loc[df_side["bucket"].isna(), "bucket"] = default_bucket
        if missing_buckets:
            print(f"[ranker] {side} missing sltp: {missing_buckets} rows")
        
        df_side["sl_pips"] = df_side["bucket"].apply(
            lambda b: SL_TP_PAIRS[int(b)][0] if pd.notna(b) else SL_TP_PAIRS[default_bucket][0]
        )
        df_side["tp_pips"] = df_side["bucket"].apply(
            lambda b: SL_TP_PAIRS[int(b)][1] if pd.notna(b) else SL_TP_PAIRS[default_bucket][1]
        )
        df_side["edge_score"] = df_side["meta_prob"] * (
            df_side["tp_pips"] - df_side["sl_pips"]
        )
        signal_frames.append(
            df_side[["edge_score", "market_regime", "sl_pips", "tp_pips", "meta_prob", "side"]]
        )
        print(f"[ranker] {side} candidates={len(df_side)}")

    if not signal_frames:
        print("No signals found")
        return

    signals = pd.concat(signal_frames).sort_index()
    signals["rr"] = signals["tp_pips"] / signals["sl_pips"]

    # Apply win rate enhancements before further filtering
    enhanced = apply_win_rate_enhancements_to_signals(
        signals, target_trades_per_week=target
    )
    if not enhanced.empty:
        signals = enhanced

    # Apply session-specific adjustments
    signals = apply_session_filters(signals)

    train_mask = signals.index <= train_end_date if train_end_date else slice(None)
    train_scores = signals.loc[train_mask, "edge_score"]
    hold_scores = (
        signals.loc[~train_mask, "edge_score"] if train_end_date else pd.Series(dtype=float)
    )
    train_win_probs = signals.loc[train_mask, "meta_prob"]
    hold_win_probs = signals.loc[~train_mask, "meta_prob"] if train_end_date else pd.Series(dtype=float)
    train_rr = signals.loc[train_mask, "rr"]
    hold_rr = signals.loc[~train_mask, "rr"] if train_end_date else pd.Series(dtype=float)

    weeks_train = max((train_scores.index.max() - train_scores.index.min()).days / 7, 1e-9)
    weeks_hold = (
        max((hold_scores.index.max() - hold_scores.index.min()).days / 7, 1e-9)
        if not hold_scores.empty
        else 0.0
    )

    win_range = tuple(config.get("goals.win_rate_range", (0.0, 1.0)))
    rr_range = tuple(config.get("goals.risk_reward_range", (0.0, 10.0)))

    if args.edge_threshold is not None:
        thr = args.edge_threshold
        mask_train = train_scores >= thr
        mask_hold = hold_scores >= thr
        trades_per_week_train = mask_train.sum() / weeks_train
        trades_per_week_hold = mask_hold.sum() / weeks_hold if weeks_hold else 0.0
        win_rate_train = train_win_probs[mask_train].mean() if mask_train.any() else 0.0
        win_rate_hold = hold_win_probs[mask_hold].mean() if weeks_hold and mask_hold.any() else 0.0
        avg_rr_train = train_rr[mask_train].mean() if mask_train.any() else 0.0
        avg_rr_hold = hold_rr[mask_hold].mean() if weeks_hold and mask_hold.any() else 0.0
    else:
        # Try enhanced ranking first
        try:
            print("[ranker] Using enhanced signal ranking...")
            signals_enhanced = rank_signals_enhanced(signals, target)
            thr = signals_enhanced['edge_score'].min() if not signals_enhanced.empty else 0.0
            
            # Calculate metrics for enhanced ranking
            mask_train = (signals.index <= train_end_date) if train_end_date else signals.index.notna()
            mask_hold = (signals.index > train_end_date) if train_end_date else pd.Series(False, index=signals.index)
            
            enhanced_train = signals_enhanced[signals_enhanced.index.isin(signals[mask_train].index)]
            enhanced_hold = signals_enhanced[signals_enhanced.index.isin(signals[mask_hold].index)]
            
            trades_per_week_train = len(enhanced_train) / weeks_train
            trades_per_week_hold = len(enhanced_hold) / weeks_hold if weeks_hold else 0.0
            win_rate_train = enhanced_train['meta_prob'].mean() if not enhanced_train.empty else 0.0
            win_rate_hold = enhanced_hold['meta_prob'].mean() if not enhanced_hold.empty else 0.0
            avg_rr_train = enhanced_train['rr'].mean() if not enhanced_train.empty else 0.0
            avg_rr_hold = enhanced_hold['rr'].mean() if not enhanced_hold.empty else 0.0
            
            print(f"[ranker] Enhanced ranking: threshold={thr:.6f}, trades/wk(train)={trades_per_week_train:.1f}")
            
        except Exception as e:
            print(f"[ranker] Enhanced ranking failed: {e}, falling back to traditional method")
            
            # Fallback to traditional consensus-based approach
            consensus = (
                find_threshold_consensus(
                    train_scores,
                    hold_scores,
                    train_win_probs,
                    hold_win_probs,
                    train_rr,
                    hold_rr,
                    target,
                    min_per_week,
                    max_per_week,
                    win_range,
                    rr_range,
                )
                if weeks_hold
                else None
            )
            
            if consensus is not None:
                (
                    thr,
                    trades_per_week_train,
                    trades_per_week_hold,
                    win_rate_train,
                    win_rate_hold,
                    avg_rr_train,
                    avg_rr_hold,
                ) = consensus
            else:
                # Try dynamic thresholding
                try:
                    print("[ranker] Using dynamic thresholding...")
                    thr = find_edge_threshold_dynamic(train_scores, target)
                    mask_train = train_scores >= thr
                    mask_hold = hold_scores >= thr
                    trades_per_week_train = mask_train.sum() / weeks_train
                    trades_per_week_hold = mask_hold.sum() / weeks_hold if weeks_hold else 0.0
                    win_rate_train = train_win_probs[mask_train].mean() if mask_train.any() else 0.0
                    win_rate_hold = hold_win_probs[mask_hold].mean() if weeks_hold and mask_hold.any() else 0.0
                    avg_rr_train = train_rr[mask_train].mean() if mask_train.any() else 0.0
                    avg_rr_hold = hold_rr[mask_hold].mean() if weeks_hold and mask_hold.any() else 0.0
                    print(f"[ranker] Dynamic threshold: {thr:.6f}, trades/wk(train)={trades_per_week_train:.1f}")
                    
                except Exception as e:
                    print(f"[ranker] Dynamic thresholding failed: {e}, using traditional approach")
                    
                    (
                        thr,
                        trades_per_week_train,
                        win_rate_train,
                        avg_rr_train,
                    ) = find_threshold(
                        train_scores,
                        train_win_probs,
                        train_rr,
                        target,
                        min_per_week,
                        max_per_week,
                        win_range,
                        rr_range,
                    )
                    mask_hold = hold_scores >= thr
                    trades_per_week_hold = mask_hold.sum() / weeks_hold if weeks_hold else 0.0
                    win_rate_hold = hold_win_probs[mask_hold].mean() if weeks_hold and mask_hold.any() else 0.0
                    avg_rr_hold = hold_rr[mask_hold].mean() if weeks_hold and mask_hold.any() else 0.0
                    if weeks_hold and thr > hold_scores.max():
                        alt_thr, _, _, _ = find_threshold(
                            signals["edge_score"],
                            signals["meta_prob"],
                            signals["rr"],
                            target,
                            min_per_week,
                            max_per_week,
                            win_range,
                            rr_range,
                        )
                        thr = alt_thr if alt_thr <= hold_scores.max() else float(hold_scores.max())
                        mask_train = train_scores >= thr
                        mask_hold = hold_scores >= thr
                        trades_per_week_train = mask_train.sum() / weeks_train
                        trades_per_week_hold = mask_hold.sum() / weeks_hold
                        win_rate_train = train_win_probs[mask_train].mean() if mask_train.any() else 0.0
                        win_rate_hold = hold_win_probs[mask_hold].mean() if mask_hold.any() else 0.0
                        avg_rr_train = train_rr[mask_train].mean() if mask_train.any() else 0.0
                        avg_rr_hold = hold_rr[mask_hold].mean() if mask_hold.any() else 0.0

    # Warn only if both train and hold trade rates fall below the configured
    # minimum.  Default remains 10 when the config key is absent.
    min_signals = config.get("signal.min_signals_per_week", 10)
    if trades_per_week_train < min_signals and trades_per_week_hold < min_signals:
        warnings.warn(
            f"Trades per week below minimum ({min_signals}); consider adjusting target_trades_per_week",
            UserWarning,
        )

    thr_path = Path(run_dir) / "models" / "edge_threshold.json"
    ensure_dir(thr_path.parent)
    with open(thr_path, "w") as f:
        json.dump(
            {
                "edge_threshold": thr,
                "trades_per_week_train": trades_per_week_train,
                "trades_per_week_hold": trades_per_week_hold,
                "win_rate_train": win_rate_train,
                "win_rate_hold": win_rate_hold,
                "avg_rr_train": avg_rr_train,
                "avg_rr_hold": avg_rr_hold,
                "target_trades_per_week": target,
                "min_trades_per_week": min_per_week,
                "max_trades_per_week": max_per_week,
            },
            f,
            indent=2,
        )

    total_signals = len(signals)
    train_before = (signals.index <= train_end_date).sum() if train_end_date else total_signals
    hold_before = total_signals - train_before
    filtered = signals[signals["edge_score"] >= thr].copy()
    discarded = total_signals - len(filtered)
    train_after = (filtered.index <= train_end_date).sum() if train_end_date else len(filtered)
    hold_after = len(filtered) - train_after
    print(
        f"[ranker] threshold discarded {discarded} signals "
        f"({train_before}->{train_after} train, {hold_before}->{hold_after} hold)"
    )
    out = filtered.rename(columns={"market_regime": "regime"})
    out.index.name = "timestamp"
    out.to_csv(Path(run_dir) / "data" / "signals.csv")

    if weeks_hold and trades_per_week_hold < min_per_week:
        warnings.warn(
            "Holdout trades per week below minimum", UserWarning
        )

    print(
        "edge_threshold={:.6f}  trades/wk(train)={:.1f}  trades/wk(hold)={:.1f}  "
        "win_rate(train)={:.2f}  win_rate(hold)={:.2f}  avg_rr(train)={:.2f}  avg_rr(hold)={:.2f}  "
        "target={} range=({},{})".format(
            thr,
            trades_per_week_train,
            trades_per_week_hold,
            win_rate_train,
            win_rate_hold,
            avg_rr_train,
            avg_rr_hold,
            target,
            min_per_week,
            max_per_week,
        )
    )
    print("Signals generated")


if __name__ == "__main__":
    main()
