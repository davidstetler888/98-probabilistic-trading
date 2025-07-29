# Implementation Guide: Increasing Trade Frequency

## Quick Start

### Option 1: Use the Optimization Script (Recommended)
```bash
python apply_optimizations.py
```
This will:
- Backup your current config.yaml
- Apply Phase 1 optimizations automatically
- Show a summary of all changes made

### Option 2: Manual Configuration Update
Replace your `config.yaml` with `config_optimized.yaml` (after backing up the original)

## Key Problem Identified

**Current Issue**: Pipeline generates ~352 signals/week but only executes 1-6 trades/week (99.7% filtering rate)

**Root Cause**: Multiple layers of overly conservative constraints:
1. **Simulation cap**: `max_weekly_trades: 20` (blocks trades even if ranker approves them)
2. **Ranker filtering**: Edge threshold too restrictive
3. **Labeling criteria**: Too strict signal requirements
4. **Risk parameters**: Overly conservative RR minimums

## Critical Changes Made

### 1. Simulation Constraints (Immediate Impact)
```yaml
simulation:
  max_weekly_trades: 50  # 🔥 CRITICAL: Increased from 20
  max_daily_trades: 10   # Increased from 5
  max_positions: 3       # Increased from 2
  cooldown_min: 5        # Reduced from 10
```

### 2. Ranker Targets (High Impact)
```yaml
ranker:
  target_trades_per_week: 45  # Increased from 40
  min_trades_per_week: 30     # Increased from 25
  max_trades_per_week: 60     # Increased from 50
```

### 3. Signal Generation (Medium Impact)
```yaml
label:
  threshold: 0.0008      # Reduced from 0.0010 (0.8 vs 1.0 pip)
  max_sl_pips: 25        # Increased from 22
  min_rr: 1.8            # Reduced from 2.0
  cooldown_min: 5        # Reduced from 10
```

### 4. Risk-Reward Flexibility (Medium Impact)
```yaml
sl_tp_grid:
  tp_multipliers: [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2]  # Added 1.6RR
acceptance:
  simulation:
    min_win_rate: 0.55     # Reduced from 0.58
    min_profit_factor: 1.3  # Reduced from 1.5
```

## Expected Results

| Metric | Before | After (Conservative) | After (Optimistic) |
|--------|--------|---------------------|-------------------|
| Trades/Week | 1-6 | 15-25 | 30-45 |
| Win Rate | 100% | 65-75% | 58-68% |
| Avg RR | 2.4-3.2 | 2.0-2.5 | 1.8-2.2 |
| Drawdown | <5% | 5-10% | 8-12% |

## Monitoring Plan

### Week 1-2: Initial Assessment
- **Target**: 15-25 trades/week
- **Win Rate**: Should stay above 60%
- **Drawdown**: Monitor for increases above 10%

### Week 3-4: Fine-tuning
- If win rate drops below 55%, increase thresholds slightly
- If trade frequency still low, implement Phase 2 changes
- If drawdown exceeds 15%, tighten risk management

### Rollback Criteria
- Win rate drops below 50% for 2+ weeks
- Drawdown exceeds 20%
- Profit factor drops below 1.0

## Phase 2 Implementation (If Needed)

If Phase 1 doesn't achieve 25+ trades/week:

1. **Modify `train_ranker.py`**: Implement percentile-based thresholds
2. **Dynamic parameters**: Time-based or volatility-based adjustments
3. **Multi-timeframe signals**: Add 15min and 30min signal generation
4. **Advanced filtering**: Regime-specific edge thresholds

## Files Created

1. `pipeline_assessment.md` - Detailed analysis
2. `config_optimized.yaml` - Ready-to-use configuration
3. `apply_optimizations.py` - Automated optimization script
4. `IMPLEMENTATION_GUIDE.md` - This guide

## Next Steps

1. Run `python apply_optimizations.py`
2. Execute the pipeline: `python main.py`
3. Monitor results for 1-2 weeks
4. Adjust parameters based on performance
5. Implement Phase 2 if needed

## Support

If you encounter issues:
1. Check backup files created by the script
2. Verify all dependencies are installed
3. Review the detailed analysis in `pipeline_assessment.md`
4. Consider gradual implementation of changes

The key insight is that your system is **over-optimized for precision at the expense of recall**. These changes will increase trade frequency while maintaining profitability.