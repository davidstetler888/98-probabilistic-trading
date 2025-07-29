
# Enhanced Signal Ranking Functions for Win Rate Optimization
# Add these functions to train_ranker.py

def calculate_enhanced_signal_score(row):
    """
    Calculate enhanced signal score based on multiple factors
    """
    score = 0
    
    # Base probability score (0-40 points)
    score += row['meta_prob'] * 40
    
    # Risk-reward bonus (0-20 points)
    if 'tp_pips' in row.index and 'sl_pips' in row.index:
        rr_ratio = row['tp_pips'] / row['sl_pips'] if row['sl_pips'] > 0 else 0
        score += min(rr_ratio / 3.0, 1.0) * 20
    
    # Session quality bonus (0-10 points)
    hour = row.name.hour if hasattr(row.name, 'hour') else 12
    if 8 <= hour < 13 or 13 <= hour < 18:  # London or NY
        score += 10
    elif 22 <= hour or hour < 8:  # Asian
        score += 5
    
    # Volatility bonus (0-15 points)
    if 'atr' in row.index:
        # Optimal volatility around 1.2 pips
        vol_optimal = 0.0012
        vol_current = row['atr']
        vol_diff = abs(vol_current - vol_optimal) / vol_optimal
        vol_score = max(0, 15 * (1 - vol_diff))
        score += min(15, vol_score)
    else:
        score += 7.5  # Default medium score
    
    # Multi-timeframe alignment (0-15 points)
    if 'htf_15min_trend' in row.index and 'htf_30min_trend' in row.index:
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
        score += 7.5  # Default score
    
    return score

def find_dynamic_threshold(signals, target_trades_per_week):
    """
    Use percentile-based approach for consistent trade volume
    """
    if signals.empty:
        return 0.0
    
    # Calculate target number of signals
    weeks = len(signals) / (7 * 24 * 12)  # 5-minute bars
    target_signals = int(target_trades_per_week * weeks)
    
    # Calculate enhanced scores
    signals['enhanced_score'] = signals.apply(calculate_enhanced_signal_score, axis=1)
    
    # Sort by enhanced score and take top N
    sorted_signals = signals.sort_values('enhanced_score', ascending=False)
    
    if len(sorted_signals) >= target_signals:
        selected_signals = sorted_signals.head(target_signals)
        threshold = selected_signals['edge_score'].min()
    else:
        # Not enough signals, use 70th percentile of edge scores
        threshold = signals['edge_score'].quantile(0.7)
    
    return threshold

def apply_session_multipliers(signals):
    """
    Apply session-specific adjustments to signal scores
    """
    def get_session_multiplier(timestamp):
        hour = timestamp.hour
        if 8 <= hour < 12:      # London session
            return 1.1
        elif 13 <= hour < 17:   # NY session  
            return 1.05
        elif 22 <= hour or hour < 7:  # Asian session
            return 0.95
        else:  # Overlap periods
            return 1.15
    
    signals['session_multiplier'] = signals.index.map(get_session_multiplier)
    signals['edge_score_adjusted'] = signals['edge_score'] * signals['session_multiplier']
    
    return signals
