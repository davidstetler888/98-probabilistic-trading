# Additional Code Optimizations for Increased Trade Frequency

## Advanced Algorithmic Improvements

### 1. **Dynamic Edge Thresholding** (High Impact)

**Current Issue**: Fixed edge threshold in `train_ranker.py` is too restrictive.

**Solution**: Implement percentile-based signal selection:

```python
# In train_ranker.py, modify the edge threshold calculation
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

# Replace the existing threshold calculation around line 430
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
```

### 2. **Multi-Timeframe Signal Generation** (High Impact)

**Current Issue**: Only 5-minute signals generated.

**Solution**: Add 15-minute and 30-minute timeframes:

```python
# In prepare.py, add multi-timeframe processing
def prepare_multi_timeframe(df):
    """Generate features for multiple timeframes"""
    
    # Original 5-minute data
    df_5min = df.copy()
    
    # 15-minute aggregation
    df_15min = df.resample('15T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # 30-minute aggregation
    df_30min = df.resample('30T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Add higher timeframe features to 5-minute data
    df_5min['htf_15min_trend'] = df_15min['close'].pct_change().reindex(df_5min.index, method='ffill')
    df_5min['htf_30min_trend'] = df_30min['close'].pct_change().reindex(df_5min.index, method='ffill')
    
    return df_5min, df_15min, df_30min

# In label.py, add multi-timeframe labeling
def label_multi_timeframe(df_5min, df_15min, df_30min):
    """Generate labels for multiple timeframes"""
    
    # 5-minute labels (existing)
    labels_5min = label_data(df_5min, config)
    
    # 15-minute labels (different criteria)
    config_15min = config.copy()
    config_15min['label']['threshold'] = 0.0015  # Higher threshold for 15min
    config_15min['label']['future_window'] = 8   # 2 hours in 15min bars
    labels_15min = label_data(df_15min, config_15min)
    
    # 30-minute labels (different criteria) 
    config_30min = config.copy()
    config_30min['label']['threshold'] = 0.0020  # Higher threshold for 30min
    config_30min['label']['future_window'] = 4   # 2 hours in 30min bars
    labels_30min = label_data(df_30min, config_30min)
    
    return labels_5min, labels_15min, labels_30min
```

### 3. **Session-Specific Optimization** (Medium Impact)

**Current Issue**: Same criteria applied to all trading sessions.

**Solution**: Different thresholds for different sessions:

```python
# In simulate.py, add session-specific logic
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
    df['session_multiplier'] = df['timestamp'].apply(get_session_multiplier)
    
    # Adjust edge thresholds based on session
    if 'edge_prob' in df.columns:
        df['edge_prob_adjusted'] = df['edge_prob'] * df['session_multiplier']
    
    return df
```

### 4. **Volatility-Based Position Sizing** (Medium Impact)

**Current Issue**: Fixed position sizing regardless of market conditions.

**Solution**: Dynamic position sizing based on volatility:

```python
# In simulate.py, enhance position sizing
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
    
    # Calculate position size
    position_size = (account_balance * adjusted_risk) / (atr * 10000)
    
    return position_size, adjusted_risk
```

### 5. **Improved Signal Ranking** (High Impact)

**Current Issue**: Simple edge threshold-based ranking.

**Solution**: Multi-factor ranking system:

```python
# In train_ranker.py, enhance signal ranking
def calculate_signal_score(row):
    """
    Calculate composite signal score based on multiple factors
    """
    score = 0
    
    # Base probability score (0-40 points)
    score += row['edge_prob'] * 40
    
    # Risk-reward bonus (0-20 points)
    rr_ratio = row['tp_pips'] / row['sl_pips']
    score += min(rr_ratio / 3.0, 1.0) * 20
    
    # Volatility bonus (0-15 points)
    volatility_score = min(row['atr'] / 0.0015, 1.0) * 15
    score += volatility_score
    
    # Session bonus (0-10 points)
    hour = row.index.hour
    if 8 <= hour < 13 or 13 <= hour < 18:  # London or NY
        score += 10
    elif 22 <= hour or hour < 8:  # Asian
        score += 5
    
    # Trend alignment bonus (0-15 points)
    if 'htf_15min_trend' in row and 'htf_30min_trend' in row:
        if row['side'] == 'long':
            if row['htf_15min_trend'] > 0 and row['htf_30min_trend'] > 0:
                score += 15
        else:  # short
            if row['htf_15min_trend'] < 0 and row['htf_30min_trend'] < 0:
                score += 15
    
    return score

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
```

### 6. **Risk Management Enhancements** (Medium Impact)

**Current Issue**: Basic risk management.

**Solution**: Advanced portfolio-based risk management:

```python
# In simulate.py, add portfolio risk management
class PortfolioRiskManager:
    def __init__(self, max_portfolio_risk=0.06, max_correlation=0.7):
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlation = max_correlation
        self.open_positions = []
    
    def can_take_position(self, new_signal, current_portfolio_risk):
        """Check if new position can be taken based on portfolio risk"""
        
        # Check total portfolio risk
        if current_portfolio_risk + new_signal['risk_amount'] > self.max_portfolio_risk:
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
                                 if pos['side'] == new_signal['side'])
        
        return same_direction_count / len(self.open_positions)
    
    def add_position(self, position):
        """Add position to portfolio"""
        self.open_positions.append(position)
    
    def remove_position(self, position_id):
        """Remove position from portfolio"""
        self.open_positions = [pos for pos in self.open_positions 
                              if pos['id'] != position_id]
```

### 7. **Performance Monitoring & Auto-Adjustment** (Low Impact, High Value)

**Solution**: Automatic parameter adjustment based on performance:

```python
# Create performance_monitor.py
class PerformanceMonitor:
    def __init__(self):
        self.performance_history = []
        self.adjustment_threshold = 0.1  # 10% deviation from target
    
    def monitor_and_adjust(self, current_performance, target_metrics):
        """Monitor performance and suggest adjustments"""
        
        trades_per_week = current_performance['trades_per_week']
        target_trades = target_metrics['target_trades_per_week']
        
        deviation = abs(trades_per_week - target_trades) / target_trades
        
        if deviation > self.adjustment_threshold:
            return self.suggest_adjustments(current_performance, target_metrics)
        
        return None
    
    def suggest_adjustments(self, current, target):
        """Suggest parameter adjustments"""
        suggestions = []
        
        if current['trades_per_week'] < target['target_trades_per_week']:
            suggestions.append({
                'parameter': 'simulation.max_weekly_trades',
                'current': current['max_weekly_trades'],
                'suggested': min(current['max_weekly_trades'] * 1.2, 100),
                'reason': 'Increase trade capacity'
            })
            
            suggestions.append({
                'parameter': 'label.threshold',
                'current': current['threshold'],
                'suggested': current['threshold'] * 0.9,
                'reason': 'Lower entry threshold'
            })
        
        elif current['trades_per_week'] > target['target_trades_per_week']:
            suggestions.append({
                'parameter': 'label.threshold',
                'current': current['threshold'],
                'suggested': current['threshold'] * 1.1,
                'reason': 'Raise entry threshold'
            })
        
        return suggestions
```

## Implementation Priority

### **Week 1: Core Optimizations**
1. Apply existing `apply_optimizations.py` script
2. Implement dynamic edge thresholding
3. Add session-specific multipliers

### **Week 2: Signal Enhancement**
1. Add multi-timeframe signal generation
2. Implement enhanced signal ranking
3. Add volatility-based position sizing

### **Week 3: Risk Management**
1. Implement portfolio risk management
2. Add performance monitoring
3. Fine-tune all parameters

### **Week 4: Advanced Features**
1. Add automatic parameter adjustment
2. Implement correlation-based filtering
3. Add regime-specific optimizations

## Expected Additional Impact

| Optimization | Expected Trade Increase | Risk Level |
|--------------|------------------------|------------|
| Dynamic Thresholding | 50-100% | Low |
| Multi-Timeframe | 100-200% | Medium |
| Session-Specific | 20-30% | Low |
| Enhanced Ranking | 30-50% | Low |
| Portfolio Risk Mgmt | 0-20% | Low |
| Performance Monitor | 10-20% | Low |

## Testing Strategy

1. **A/B Testing**: Run parallel systems with different optimizations
2. **Gradual Rollout**: Implement one optimization at a time
3. **Performance Tracking**: Monitor all key metrics continuously
4. **Rollback Plan**: Maintain ability to revert any changes

## Code Quality Considerations

1. **Modular Design**: Each optimization as separate module
2. **Configuration Driven**: All parameters in config files
3. **Logging**: Comprehensive logging for debugging
4. **Testing**: Unit tests for all new functions
5. **Documentation**: Clear documentation for all changes

These additional optimizations, combined with the existing optimization scripts, should easily achieve the target of 25-50 trades per week while maintaining profitability.