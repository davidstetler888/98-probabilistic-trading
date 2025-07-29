# Trade Frequency Analysis & Optimization Recommendations

## Executive Summary

**Current State**: Your trading system generates ~352 signals/week but only executes 1-6 trades/week (99.7% filtering rate), achieving 100% win rate - indicating severe over-filtering.

**Target**: Increase to 25-50 trades/week while maintaining 58-75% win rate.

**Root Cause**: Multiple layers of overly conservative constraints are creating a bottleneck that's blocking profitable trades.

## 🔍 Key Bottlenecks Identified

### 1. **CRITICAL: Simulation Hard Cap**
```yaml
# Current constraint in config.yaml
simulation:
  max_weekly_trades: 20  # ❌ BLOCKS trades even if ranker approves them
  max_daily_trades: 5    # ❌ Only 25 trades/week maximum possible
```

### 2. **Ranker Over-filtering**
- Edge threshold calculation is too restrictive
- 352 signals/week filtered down to 1-6 trades (99.7% rejection rate)
- Perfect 100% win rate indicates excessive conservatism

### 3. **Conservative Operational Constraints**
```yaml
simulation:
  max_positions: 2       # ❌ Limits concurrent trades
  cooldown_min: 10       # ❌ 10-minute gaps between trades
  position_size: 0.1     # ❌ Large position sizes with high risk requirements
```

### 4. **Strict Labeling Criteria**
```yaml
label:
  threshold: 0.0010      # ❌ 1.0 pip minimum (very tight)
  max_sl_pips: 22        # ❌ Narrow SL range
  min_rr: 2.0            # ❌ High RR requirement
  cooldown_min: 10       # ❌ Additional cooldown layer
```

## 🚀 Immediate High-Impact Optimizations

### **Phase 1: Critical Constraint Removal (Expected: 3-5x increase)**

#### 1. Remove Simulation Trade Caps
```yaml
simulation:
  max_weekly_trades: 50  # ✅ Increase from 20 (CRITICAL)
  max_daily_trades: 10   # ✅ Increase from 5
  max_positions: 3       # ✅ Allow more concurrent positions
  cooldown_min: 5        # ✅ Reduce from 10 minutes
```

#### 2. Optimize Ranker Parameters
```yaml
ranker:
  target_trades_per_week: 45  # ✅ Increase from 40
  min_trades_per_week: 30     # ✅ Increase from 25
  max_trades_per_week: 60     # ✅ Increase from 50
```

#### 3. Relax Labeling Criteria
```yaml
label:
  threshold: 0.0008      # ✅ Reduce to 0.8 pips
  max_sl_pips: 25        # ✅ Increase from 22
  min_rr: 1.8            # ✅ Reduce from 2.0
  cooldown_min: 5        # ✅ Reduce from 10
```

### **Phase 2: Risk-Reward Optimization (Expected: Additional 2-3x increase)**

#### 4. Expand SL/TP Grid
```yaml
sl_tp_grid:
  tp_multipliers: [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2]  # ✅ Add lower RR options
  sl_multipliers: [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6]  # ✅ Expand range
```

#### 5. Adjust Acceptance Criteria
```yaml
acceptance:
  simulation:
    min_win_rate: 0.55     # ✅ Reduce from 0.58
    min_profit_factor: 1.3  # ✅ Reduce from 1.5
    min_trades_per_week: 30 # ✅ Increase from 25
```

### **Phase 3: Advanced Algorithmic Improvements**

#### 6. Implement Dynamic Edge Thresholding
- **Current**: Fixed edge threshold that's too restrictive
- **Solution**: Implement percentile-based selection (e.g., top 20% of signals)
- **Code location**: `train_ranker.py` lines 430-440

#### 7. Multi-Timeframe Signal Generation
- **Current**: Only 5-minute signals
- **Solution**: Add 15-minute and 30-minute timeframes
- **Expected impact**: 2-3x more signal opportunities

#### 8. Session-Specific Optimization
- **Current**: Same criteria for all sessions
- **Solution**: Different thresholds for Asian/London/NY sessions
- **Rationale**: Different volatility patterns require different approaches

## 📊 Implementation Strategy

### **Week 1: Quick Wins (Existing Solutions)**
```bash
# Use the existing optimization script
python apply_optimizations.py
```

This automatically applies Phase 1 optimizations and creates backups.

### **Week 2: Monitor and Adjust**
- Expected: 15-25 trades/week
- Monitor win rate (target: 60-75%)
- Watch for drawdown increases

### **Week 3: Advanced Optimizations**
- Implement dynamic thresholding
- Add multi-timeframe signals
- Optimize session-specific parameters

### **Week 4: Fine-tuning**
- Adjust based on performance data
- Implement additional risk management
- Consider portfolio-based position sizing

## 🎯 Expected Results

| Phase | Trades/Week | Win Rate | Avg RR | Drawdown |
|-------|-------------|----------|--------|----------|
| Current | 1-6 | 100% | 2.4-3.2 | <5% |
| Phase 1 | 15-25 | 65-75% | 2.0-2.5 | 5-10% |
| Phase 2 | 25-35 | 60-70% | 1.8-2.2 | 8-12% |
| Phase 3 | 35-50 | 58-65% | 1.8-2.0 | 10-15% |

## 🔧 Code Changes Required

### **Immediate (Using Existing Scripts)**
```bash
# Apply all Phase 1 optimizations
python apply_optimizations.py

# Run the pipeline
python main.py
```

### **Advanced (Custom Development)**

#### 1. Dynamic Thresholding in `train_ranker.py`
```python
# Add percentile-based threshold selection
def get_dynamic_threshold(probs, target_percentile=0.8):
    """Select threshold based on percentile rather than fixed criteria"""
    return np.percentile(probs, target_percentile * 100)
```

#### 2. Multi-Timeframe Signal Generation
```python
# In prepare.py, add multiple timeframe processing
def prepare_multi_timeframe(df):
    """Generate signals on 5min, 15min, 30min timeframes"""
    tf_5min = df.resample('5T').agg(ohlc_dict)
    tf_15min = df.resample('15T').agg(ohlc_dict)
    tf_30min = df.resample('30T').agg(ohlc_dict)
    return tf_5min, tf_15min, tf_30min
```

#### 3. Session-Specific Optimization
```python
# In simulate.py, add session-specific thresholds
def get_session_threshold(timestamp, base_threshold):
    """Adjust thresholds based on trading session"""
    if is_asian_session(timestamp):
        return base_threshold * 0.9  # More aggressive in Asian session
    elif is_london_session(timestamp):
        return base_threshold * 1.1  # More conservative in London
    return base_threshold
```

## ⚠️ Risk Management

### **Monitoring Metrics**
- **Win Rate**: Should stay above 55% (currently 100%)
- **Drawdown**: Monitor for increases above 15%
- **Sharpe Ratio**: Track risk-adjusted returns
- **Profit Factor**: Maintain above 1.3

### **Rollback Criteria**
- Win rate drops below 50% for 2+ weeks
- Drawdown exceeds 20%
- Profit factor drops below 1.0
- Sharpe ratio becomes negative

### **Gradual Implementation**
1. Start with Phase 1 (lowest risk)
2. Monitor for 1-2 weeks
3. Implement Phase 2 if results are stable
4. Phase 3 only if system proves robust

## 🏆 Key Success Factors

1. **Start with Existing Optimizations**: Use `apply_optimizations.py` for immediate gains
2. **Monitor Continuously**: Track all metrics, not just trade count
3. **Gradual Approach**: Don't implement all changes at once
4. **Backup Everything**: Always maintain ability to rollback
5. **Test in Parallel**: Consider running old and new configs side-by-side

## 📈 Bottom Line

Your system is **over-optimized for precision at the expense of recall**. The 100% win rate with only 1-6 trades/week indicates you're leaving significant profits on the table. By systematically relaxing constraints and implementing the suggested optimizations, you can realistically achieve 25-50 trades/week while maintaining profitable win rates above 58%.

**Next Steps:**
1. Run `python apply_optimizations.py` immediately
2. Execute the pipeline and monitor results
3. Implement advanced optimizations based on performance data
4. Consider hiring a quantitative developer if advanced algorithmic changes are needed

The infrastructure is solid - you just need to "turn up the volume" on trade execution while maintaining risk management discipline.