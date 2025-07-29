#!/usr/bin/env python3
"""
Enhanced Ranker with Win Rate Optimization Integration

This enhanced ranker prioritizes signal quality over quantity to achieve
60%+ win rates while maintaining 30+ trades per week.

Key Features:
1. Integration with WinRateOptimizer
2. Quality-first signal ranking
3. Dynamic threshold adjustment
4. Session-based optimization
5. Market regime filtering
6. Advanced risk management
"""

import argparse
import json
import pickle
from pathlib import Path
import warnings
import sys

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

# Import the win rate optimizer
try:
    from win_rate_optimizer import WinRateOptimizer
except ImportError:
    print("Warning: WinRateOptimizer not found. Using standard ranking.")
    WinRateOptimizer = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enhanced ranker with win rate optimization"
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
        default=30,
        help="Target trades per week (default: 30)",
    )
    parser.add_argument(
        "--min_trades_per_week",
        type=float,
        default=20,
        help="Minimum trades per week (default: 20)",
    )
    parser.add_argument(
        "--max_trades_per_week",
        type=float,
        default=40,
        help="Maximum trades per week (default: 40)",
    )
    parser.add_argument(
        "--target_win_rate",
        type=float,
        default=0.60,
        help="Target win rate (default: 0.60)",
    )
    parser.add_argument(
        "--edge_threshold",
        type=float,
        required=False,
        help="Use this edge score threshold instead of auto search",
    )
    parser.add_argument(
        "--use_win_rate_optimizer",
        action="store_true",
        help="Use win rate optimizer for signal filtering",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort if CSV parsing fails",
    )
    return parser.parse_args()


def calculate_enhanced_quality_score(signal_data: pd.Series) -> float:
    """
    Calculate enhanced quality score optimized for win rate
    
    Args:
        signal_data: Row from signals dataframe
        
    Returns:
        Quality score between 0-100
    """
    score = 0
    
    # 1. Base probability (0-35 points) - Primary factor
    meta_prob = signal_data.get('meta_prob', 0.5)
    edge_score = signal_data.get('edge_score', 0.5)
    
    # Weight both meta_prob and edge_score
    combined_prob = (meta_prob * 0.7) + (edge_score * 0.3)
    
    if combined_prob >= 0.75:
        score += 35
    elif combined_prob >= 0.70:
        score += 30
    elif combined_prob >= 0.65:
        score += 25
    elif combined_prob >= 0.60:
        score += 20
    elif combined_prob >= 0.55:
        score += 15
    else:
        score += 0
    
    # 2. Risk-Reward premium (0-25 points)
    rr_ratio = signal_data.get('rr', 2.0)
    tp_pips = signal_data.get('tp_pips', 20)
    sl_pips = signal_data.get('sl_pips', 10)
    
    if rr_ratio >= 3.5:
        score += 25
    elif rr_ratio >= 3.0:
        score += 22
    elif rr_ratio >= 2.5:
        score += 18
    elif rr_ratio >= 2.2:
        score += 15
    elif rr_ratio >= 2.0:
        score += 10
    else:
        score += 0
    
    # 3. Market regime optimization (0-20 points)
    regime = signal_data.get('market_regime', 1)
    regime_scores = {
        0: 20,  # Optimal regime (trending, low volatility)
        1: 16,  # Good regime (moderate conditions)
        2: 8,   # Challenging regime (high volatility)
        3: 2    # Difficult regime (choppy markets)
    }
    score += regime_scores.get(regime, 5)
    
    # 4. Session quality (0-12 points)
    try:
        timestamp = signal_data.name
        hour = timestamp.hour if hasattr(timestamp, 'hour') else 12
        weekday = timestamp.weekday() if hasattr(timestamp, 'weekday') else 1
    except:
        hour = 12
        weekday = 1
    
    # Avoid weekends and focus on best sessions
    if weekday >= 5:  # Weekend
        score += 0
    elif 12 <= hour < 14:  # London/NY overlap
        score += 12
    elif 8 <= hour < 13:   # London session
        score += 10
    elif 13 <= hour < 17:  # NY session
        score += 8
    elif 2 <= hour < 8:    # Asian session
        score += 4
    else:                  # Off-hours
        score += 0
    
    # 5. Volatility assessment (0-8 points)
    atr_pct = signal_data.get('atr_pct', 0.5)
    if atr_pct <= 0.25:      # Very low volatility
        score += 8
    elif atr_pct <= 0.4:     # Low volatility
        score += 6
    elif atr_pct <= 0.6:     # Moderate volatility
        score += 4
    elif atr_pct <= 0.8:     # High volatility
        score += 2
    else:                    # Extreme volatility
        score += 0
    
    return score


def apply_quality_first_ranking(signals_df: pd.DataFrame, 
                               target_trades_per_week: int = 30,
                               target_win_rate: float = 0.60) -> pd.DataFrame:
    """
    Apply quality-first ranking to maximize win rate
    
    Args:
        signals_df: Input signals DataFrame
        target_trades_per_week: Target number of trades per week
        target_win_rate: Target win rate
        
    Returns:
        Ranked and filtered signals DataFrame
    """
    if signals_df.empty:
        return signals_df
    
    print(f"[EnhancedRanker] Starting quality-first ranking with {len(signals_df)} signals")
    
    # Calculate enhanced quality scores
    signals_df['quality_score'] = signals_df.apply(calculate_enhanced_quality_score, axis=1)
    
    # Apply minimum quality thresholds
    min_quality_score = 60  # Only top 40% of signals
    min_meta_prob = 0.60    # Minimum probability
    min_rr = 2.0           # Minimum risk-reward
    
    # Progressive filtering
    quality_filtered = signals_df[signals_df['quality_score'] >= min_quality_score].copy()
    prob_filtered = quality_filtered[quality_filtered['meta_prob'] >= min_meta_prob].copy()
    rr_filtered = prob_filtered[prob_filtered['rr'] >= min_rr].copy()
    
    print(f"[EnhancedRanker] After quality filtering: {len(rr_filtered)} signals")
    
    # Apply market regime filtering
    preferred_regimes = [0, 1]  # Only best regimes
    regime_filtered = rr_filtered[rr_filtered['market_regime'].isin(preferred_regimes)].copy()
    
    print(f"[EnhancedRanker] After regime filtering: {len(regime_filtered)} signals")
    
    # Apply session filtering
    session_filtered = apply_session_quality_filter(regime_filtered)
    
    print(f"[EnhancedRanker] After session filtering: {len(session_filtered)} signals")
    
    # Calculate target number of trades
    if not session_filtered.empty:
        weeks = (session_filtered.index.max() - session_filtered.index.min()).days / 7
        target_trades = max(int(target_trades_per_week * weeks), 20)  # Minimum 20 trades
        
        # Sort by quality score and select top trades
        sorted_signals = session_filtered.sort_values('quality_score', ascending=False)
        
        # Apply temporal distribution
        final_signals = apply_temporal_distribution(sorted_signals, target_trades)
        
        print(f"[EnhancedRanker] Final selection: {len(final_signals)} signals")
        
        return final_signals
    
    return session_filtered


def apply_session_quality_filter(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Apply session-based quality filtering"""
    if signals_df.empty:
        return signals_df
    
    def is_quality_session(timestamp):
        try:
            hour = timestamp.hour
            weekday = timestamp.weekday()
            
            # Avoid weekends
            if weekday >= 5:
                return False
            
            # Focus on London and NY sessions
            return (8 <= hour < 17)
        except:
            return False
    
    # Filter by session quality
    session_filtered = signals_df[signals_df.index.map(is_quality_session)].copy()
    
    # Limit trades per session to maintain quality
    max_trades_per_session = 6
    session_groups = session_filtered.groupby(session_filtered.index.floor('4H'))
    
    quality_sessions = []
    for session_time, group in session_groups:
        # Take only the best signals from each session
        session_best = group.nlargest(max_trades_per_session, 'quality_score')
        quality_sessions.append(session_best)
    
    return pd.concat(quality_sessions) if quality_sessions else pd.DataFrame()


def apply_temporal_distribution(signals_df: pd.DataFrame, target_trades: int) -> pd.DataFrame:
    """Apply temporal distribution to avoid trade clustering"""
    if signals_df.empty:
        return signals_df
    
    # Sort by timestamp
    sorted_signals = signals_df.sort_index()
    
    # Apply minimum time gap between trades
    min_gap_minutes = 30  # 30 minutes minimum gap
    distributed_signals = []
    last_trade_time = None
    
    for idx, signal in sorted_signals.iterrows():
        if last_trade_time is None:
            distributed_signals.append(signal)
            last_trade_time = idx
        else:
            time_diff = (idx - last_trade_time).total_seconds() / 60
            if time_diff >= min_gap_minutes:
                distributed_signals.append(signal)
                last_trade_time = idx
                
                # Stop if we have enough trades
                if len(distributed_signals) >= target_trades:
                    break
    
    return pd.DataFrame(distributed_signals) if distributed_signals else pd.DataFrame()


def calculate_dynamic_threshold(signals_df: pd.DataFrame, 
                              target_trades_per_week: int = 30,
                              target_win_rate: float = 0.60) -> float:
    """
    Calculate dynamic threshold based on quality distribution
    
    Args:
        signals_df: Input signals DataFrame
        target_trades_per_week: Target trades per week
        target_win_rate: Target win rate
        
    Returns:
        Dynamic threshold value
    """
    if signals_df.empty:
        return 0.0
    
    # Calculate weeks
    weeks = (signals_df.index.max() - signals_df.index.min()).days / 7
    target_trades = target_trades_per_week * weeks
    
    # Calculate quality scores
    quality_scores = signals_df.apply(calculate_enhanced_quality_score, axis=1)
    
    # Determine percentile based on target
    if len(quality_scores) < target_trades:
        # Not enough signals, use lower percentile
        percentile = 50
    else:
        # Calculate percentile to achieve target trades
        percentile = (1 - target_trades / len(quality_scores)) * 100
        percentile = max(60, min(90, percentile))  # Clamp between 60-90%
    
    threshold = np.percentile(quality_scores, percentile)
    
    print(f"[EnhancedRanker] Dynamic threshold: {threshold:.2f} (percentile: {percentile:.1f}%)")
    
    return threshold


def main():
    """Enhanced ranker main function"""
    args = parse_args()
    
    print("🎯 Enhanced Ranker with Win Rate Optimization")
    print("=" * 50)
    
    # Initialize win rate optimizer if requested
    optimizer = None
    if args.use_win_rate_optimizer and WinRateOptimizer is not None:
        optimizer = WinRateOptimizer(
            target_win_rate=args.target_win_rate,
            min_trades_per_week=args.min_trades_per_week
        )
        print(f"✅ Win Rate Optimizer initialized (target: {args.target_win_rate*100:.1f}%)")
    
    # Get run directory
    run_dir = get_run_dir(args.run)
    make_run_dirs(run_dir)
    
    # Parse dates
    start_date = parse_start_date_arg(args.start_date)
    train_end_date = parse_end_date_arg(args.train_end_date)
    end_date = parse_end_date_arg(args.end_date)
    
    # Load signals
    signal_frames = []
    for side in ["long", "short"]:
        path = Path(run_dir) / "data" / f"signals_{side}.csv"
        if not path.exists():
            continue
        
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = df.sort_index()
            
            # Filter by date range
            if start_date is not None:
                df = df[df.index >= start_date]
            if end_date is not None:
                df = df[df.index <= end_date]
            
            if df.empty:
                continue
                
            # Add side column
            df["side"] = side
            
            # Calculate RR if not present
            if "rr" not in df.columns and "tp_pips" in df.columns and "sl_pips" in df.columns:
                df["rr"] = df["tp_pips"] / df["sl_pips"]
            
            signal_frames.append(df)
            print(f"[EnhancedRanker] {side} signals loaded: {len(df)}")
            
        except Exception as e:
            print(f"Error loading {side} signals: {e}")
            if args.strict:
                raise
    
    if not signal_frames:
        print("No signals found")
        return
    
    # Combine signals
    signals = pd.concat(signal_frames).sort_index()
    print(f"[EnhancedRanker] Total signals loaded: {len(signals)}")
    
    # Apply quality-first ranking
    if optimizer:
        print("🔧 Applying win rate optimizer...")
        ranked_signals = optimizer.apply_ultra_strict_filtering(signals)
    else:
        print("🔧 Applying quality-first ranking...")
        ranked_signals = apply_quality_first_ranking(
            signals, 
            target_trades_per_week=args.target_trades_per_week,
            target_win_rate=args.target_win_rate
        )
    
    # Calculate final threshold
    if not ranked_signals.empty:
        threshold = ranked_signals['edge_score'].min()
        
        # Calculate metrics
        weeks = (ranked_signals.index.max() - ranked_signals.index.min()).days / 7
        trades_per_week = len(ranked_signals) / weeks
        avg_quality = ranked_signals['quality_score'].mean() if 'quality_score' in ranked_signals.columns else 0
        avg_prob = ranked_signals['meta_prob'].mean()
        avg_rr = ranked_signals['rr'].mean()
        
        print(f"\n📊 Final Results:")
        print(f"  • Threshold: {threshold:.6f}")
        print(f"  • Trades per week: {trades_per_week:.1f}")
        print(f"  • Average quality score: {avg_quality:.1f}/100")
        print(f"  • Average probability: {avg_prob:.3f}")
        print(f"  • Average RR: {avg_rr:.2f}")
        
        # Save threshold
        threshold_data = {
            "edge_threshold": threshold,
            "trades_per_week": trades_per_week,
            "avg_quality_score": avg_quality,
            "avg_probability": avg_prob,
            "avg_rr": avg_rr,
            "target_trades_per_week": args.target_trades_per_week,
            "target_win_rate": args.target_win_rate,
            "optimization_method": "quality_first_ranking"
        }
        
        threshold_path = Path(run_dir) / "models" / "edge_threshold.json"
        ensure_dir(threshold_path.parent)
        with open(threshold_path, 'w') as f:
            json.dump(threshold_data, f, indent=2)
        
        # Save ranked signals
        output_signals = ranked_signals.rename(columns={"market_regime": "regime"})
        output_signals.index.name = "timestamp"
        output_signals.to_csv(Path(run_dir) / "data" / "signals.csv")
        
        print(f"✅ Enhanced ranking complete: {len(ranked_signals)} signals selected")
        
        # Generate report if optimizer was used
        if optimizer:
            report = optimizer.generate_optimization_report(signals, ranked_signals)
            report_path = Path(run_dir) / "win_rate_optimization_report.md"
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"📄 Optimization report saved to: {report_path}")
    
    else:
        print("⚠️ No signals passed quality filtering")
        
        # Save empty results
        threshold_data = {
            "edge_threshold": 0.0,
            "trades_per_week": 0.0,
            "avg_quality_score": 0.0,
            "avg_probability": 0.0,
            "avg_rr": 0.0,
            "target_trades_per_week": args.target_trades_per_week,
            "target_win_rate": args.target_win_rate,
            "optimization_method": "quality_first_ranking",
            "warning": "No signals passed quality filtering"
        }
        
        threshold_path = Path(run_dir) / "models" / "edge_threshold.json"
        ensure_dir(threshold_path.parent)
        with open(threshold_path, 'w') as f:
            json.dump(threshold_data, f, indent=2)


if __name__ == "__main__":
    main()