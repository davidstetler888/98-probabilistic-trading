# Win Rate Optimization Strategy
*Analysis and Recommendations for Achieving Consistent 58-75% Win Rate*

## Current Situation Analysis

### Performance Summary from output.txt:
- **Signal Generation**: 352+ signals/week being generated
- **Actual Trades**: 1-6 trades/week (99.7% filter rate)
- **Win Rate**: Highly volatile (20% to 100%)
- **RR**: Excellent and consistent (2.4-3.2)
- **Trade Frequency**: Far below target (1-6 vs 25-50 trades/week)

### Root Cause Identification:
1. **Over-aggressive filtering**: Edge threshold too restrictive in `train_ranker.py`
2. **Cherry-picking effect**: Only taking ultra-conservative trades creates unstable win rates
3. **Single-trade constraint**: Current max_positions=1 limits opportunities
4. **Conservative labeling**: Very strict signal validation criteria

## Strategic Options for Win Rate Improvement

### Option 1: Volume-Based Stabilization (RECOMMENDED)
**Theory**: Increase trade volume to achieve statistical consistency and more stable win rates.

**Target**: 25-40 trades/week with 60-70% win rate

**Implementation**:
1. **Relax edge threshold** using percentile-based selection (top 15-20% of signals)
2. **Increase position limits** from 1 to 2-3 simultaneous trades
3. **Reduce cooldown** from 10 to 5 minutes
4. **Optimize labeling criteria** for slightly more permissive signal generation

**Expected Results**:
- Win rate: 60-70% (more stable than current 20-100% volatility)
- Trades/week: 25-40 (meets target)
- RR: 2.0-2.5 (slight decrease but still excellent)
- Overall profitability: SIGNIFICANTLY HIGHER due to volume increase

### Option 2: Quality Enhancement Approach
**Theory**: Improve signal quality rather than quantity to achieve higher win rates.

**Target**: 10-15 trades/week with 75-85% win rate

**Implementation**:
1. **Enhanced meta-model features** with higher timeframe analysis
2. **Advanced signal scoring** incorporating market regime, volatility, and session factors
3. **Improved SL/TP prediction** using dynamic volatility-based adjustments
4. **Multi-timeframe signal confirmation**

**Expected Results**:
- Win rate: 75-85% (very high)
- Trades/week: 10-15 (below target but higher than current)
- RR: 2.5-3.0 (maintained or improved)
- Lower volume but higher accuracy

### Option 3: Hybrid Adaptive Strategy
**Theory**: Dynamically adjust strategy based on market conditions and recent performance.

**Target**: 20-35 trades/week with 65-75% win rate

**Implementation**:
1. **Adaptive thresholding** based on recent win rate performance
2. **Session-specific optimization** (more aggressive in Asian, conservative in London)
3. **Market regime adjustments** with different criteria per regime
4. **Performance-based position sizing**

## Detailed Implementation Plan

### Phase 1: Immediate Wins (Week 1)
```yaml
# Configuration adjustments:
simulation:
  max_positions: 2              # Increase from 1
  cooldown_min: 5              # Reduce from 10
  max_weekly_trades: 40        # Increase from 20

ranker:
  target_trades_per_week: 35   # Increase from 25
  min_trades_per_week: 20      # Increase from 15

label:
  threshold: 0.0008            # Reduce from 0.0010
  min_rr: 1.8                  # Reduce from 2.0
```

### Phase 2: Enhanced Signal Selection (Week 2)
**Modify `train_ranker.py`** to use percentile-based thresholding:

```python
def find_edge_threshold_enhanced(signals, target_trades_per_week):
    """
    Use percentile-based approach for more consistent trade volume
    """
    weeks = len(signals) / (7 * 24 * 12)
    target_signals = int(target_trades_per_week * weeks)
    
    # Sort by edge score and take top X%
    sorted_signals = signals.sort_values('edge_score', ascending=False)
    selected_signals = sorted_signals.head(target_signals)
    
    threshold = selected_signals['edge_score'].min() if not selected_signals.empty else 0.0
    
    return threshold, selected_signals
```

### Phase 3: Multi-Factor Signal Scoring (Week 3)
**Enhanced signal quality assessment**:

```python
def calculate_win_probability_score(row):
    """
    Calculate enhanced win probability based on multiple factors
    """
    base_score = row['meta_prob']
    
    # Time-of-day bonus (London/NY sessions are generally higher quality)
    session_bonus = 0.05 if 8 <= row.name.hour <= 17 else 0.0
    
    # Volatility adjustment (moderate volatility preferred)
    if 'atr' in row.index:
        vol_optimal = 0.0012  # Optimal ATR level
        vol_diff = abs(row['atr'] - vol_optimal) / vol_optimal
        vol_bonus = max(0, 0.1 * (1 - vol_diff))
    else:
        vol_bonus = 0.0
    
    # Multi-timeframe alignment bonus
    if 'htf_15min_trend' in row.index and 'htf_30min_trend' in row.index:
        if row['side'] == 'long':
            alignment = (row['htf_15min_trend'] > 0) and (row['htf_30min_trend'] > 0)
        else:
            alignment = (row['htf_15min_trend'] < 0) and (row['htf_30min_trend'] < 0)
        alignment_bonus = 0.1 if alignment else -0.05
    else:
        alignment_bonus = 0.0
    
    final_score = base_score + session_bonus + vol_bonus + alignment_bonus
    return min(1.0, max(0.0, final_score))
```

### Phase 4: Advanced Risk Management (Week 4)
**Dynamic position sizing based on win rate confidence**:

```python
def get_dynamic_position_size(signal_score, current_balance, base_risk=0.02):
    """
    Adjust position size based on signal confidence
    """
    # High confidence signals get larger positions
    confidence_multiplier = 0.5 + (signal_score * 1.0)  # 0.5x to 1.5x
    
    # Recent performance adjustment
    recent_win_rate = get_recent_win_rate()  # Implement this
    performance_multiplier = 0.8 + (recent_win_rate * 0.4)  # 0.8x to 1.2x
    
    adjusted_risk = base_risk * confidence_multiplier * performance_multiplier
    
    # Cap at reasonable limits
    final_risk = max(0.01, min(0.04, adjusted_risk))
    
    return final_risk
```

## Win Rate Improvement Techniques

### 1. Signal Quality Enhancement
- **Add higher timeframe confirmation** (15min, 30min trends)
- **Include momentum indicators** (RSI divergence, MACD crossovers)
- **Market structure analysis** (support/resistance levels)
- **Volume confirmation** (above-average volume for signal validation)

### 2. Market Regime Optimization
```python
def get_regime_specific_threshold(regime, base_threshold):
    """
    Adjust thresholds based on market regime characteristics
    """
    regime_multipliers = {
        0: 1.1,  # Trending up - more conservative on longs
        1: 1.0,  # Neutral - standard threshold
        2: 1.1,  # Trending down - more conservative on shorts
        3: 0.9   # High volatility - slightly more aggressive
    }
    
    return base_threshold * regime_multipliers.get(regime, 1.0)
```

### 3. Session-Based Optimization
```python
def get_session_multiplier(timestamp):
    """
    Adjust signal strength based on trading session
    """
    hour = timestamp.hour
    
    if 8 <= hour < 12:      # London session
        return 1.1  # Higher quality signals
    elif 13 <= hour < 17:   # NY session  
        return 1.05 # Good quality signals
    elif 22 <= hour or hour < 7:  # Asian session
        return 0.95  # Lower quality, need higher threshold
    else:  # Overlap periods
        return 1.15  # Best quality signals
```

### 4. Adaptive Threshold Management
```python
def get_adaptive_threshold(recent_performance, base_threshold):
    """
    Adjust threshold based on recent win rate performance
    """
    recent_win_rate = recent_performance.get('win_rate', 0.6)
    target_win_rate = 0.65
    
    if recent_win_rate < target_win_rate:
        # Win rate too low, be more selective
        multiplier = 1.0 + (target_win_rate - recent_win_rate) * 2
    else:
        # Win rate good, can be more aggressive
        multiplier = max(0.8, 1.0 - (recent_win_rate - target_win_rate) * 1)
    
    return base_threshold * multiplier
```

## Monitoring and Risk Management

### Key Performance Indicators
1. **Win Rate Stability**: Target 60-70% with <10% weekly variation
2. **Trade Volume**: 25-40 trades/week consistently
3. **Average RR**: Maintain above 2.0
4. **Maximum Drawdown**: Monitor for increases above 12%
5. **Profit Factor**: Keep above 1.5

### Early Warning Signals
- Win rate drops below 55% for 2+ consecutive weeks
- Trade volume falls below 15 trades/week
- Drawdown exceeds 15%
- RR drops below 1.8

### Rollback Criteria
- Win rate drops below 50%
- Drawdown exceeds 20%
- Profit factor drops below 1.0
- Consecutive losing weeks > 3

## Expected Outcomes by Phase

| Phase | Timeline | Trades/Week | Win Rate | Avg RR | Max DD |
|-------|----------|-------------|----------|--------|--------|
| Current | - | 1-6 | 20-100% | 2.4-3.2 | <5% |
| Phase 1 | Week 1 | 15-25 | 60-75% | 2.2-2.8 | 5-8% |
| Phase 2 | Week 2 | 20-30 | 62-72% | 2.0-2.6 | 6-10% |
| Phase 3 | Week 3 | 25-35 | 65-75% | 2.0-2.5 | 7-11% |
| Phase 4 | Week 4 | 30-40 | 65-70% | 2.0-2.4 | 8-12% |

## Implementation Commands

### Quick Start (Phase 1):
```bash
# Apply immediate optimizations
python apply_optimizations.py

# Run with enhanced configuration
python walkforward.py --run path/to/output --stepback_weeks 4 --optimize --grid fast

# Monitor results
tail -f output/output.txt
```

### Advanced Implementation (Phases 2-4):
```bash
# Enhanced ranker with dynamic thresholding
python train_ranker.py --run $RUN_ID --target_trades_per_week 35

# Multi-timeframe signal generation (requires code modifications)
python prepare.py --multi_timeframe

# Advanced simulation with dynamic position sizing
python simulate.py --run $RUN_ID --dynamic_sizing
```

## Key Success Factors

1. **Gradual Implementation**: Start with Phase 1, monitor for 1 week before proceeding
2. **Continuous Monitoring**: Track all KPIs, not just win rate
3. **Data-Driven Decisions**: Use performance metrics to guide adjustments
4. **Maintain Risk Discipline**: Never sacrifice risk management for higher win rates
5. **A/B Testing**: Run parallel systems to compare performance

## Bottom Line

Your current **100% win rate with 1-6 trades/week is unsustainable and unprofitable**. By implementing this strategy, you can achieve:

- **Consistent 60-70% win rate** (much more stable than current 20-100% volatility)
- **25-40 trades/week** (meets your target range)
- **Maintained RR ratios** (2.0-2.5 range)
- **Significantly higher overall profitability** due to increased volume

The key insight is that **statistical consistency through higher volume will produce much better overall results than trying to maintain perfect win rates with minimal trades**.

**Recommended Next Steps:**
1. Implement Phase 1 changes immediately
2. Monitor performance for 1 week
3. Proceed to Phase 2 if results are stable
4. Continue gradual implementation while monitoring all KPIs