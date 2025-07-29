# Single-Trade Constraint Configuration Update

## Summary
Modified both `config.yaml` and `config_optimized.yaml` to enforce a **true single-trade constraint** where only one trade can be open at any given time.

## Key Changes Made

### 1. **Core Constraint Applied**
```yaml
# Both config files now enforce:
simulation:
  max_positions: 1    # ✅ ENFORCES SINGLE TRADE CONSTRAINT
```

### 2. **Adjusted Trade Capacity Settings**

#### **config.yaml (Conservative)**
```yaml
simulation:
  max_positions: 1        # Changed from 2
  cooldown_min: 5         # Reduced from 10 (less restrictive since only 1 trade)
  max_daily_trades: 8     # Increased from 5
  max_weekly_trades: 35   # Increased from 20

ranker:
  target_trades_per_week: 30  # Reduced from 40
  min_trades_per_week: 20     # Reduced from 25
  max_trades_per_week: 40     # Reduced from 50

goals:
  trades_per_week_range: [20, 40]  # Reduced from [25, 50]
```

#### **config_optimized.yaml (Aggressive)**
```yaml
simulation:
  max_positions: 1        # Changed from 3
  cooldown_min: 3         # Further reduced from 5
  max_daily_trades: 15    # Increased from 10
  max_weekly_trades: 75   # Increased from 50

ranker:
  target_trades_per_week: 40  # Adjusted from 45
  min_trades_per_week: 25     # Adjusted from 30
  max_trades_per_week: 50     # Adjusted from 60

goals:
  trades_per_week_range: [25, 50]  # Adjusted from [30, 60]
```

### 3. **Acceptance Criteria Adjustments**
```yaml
# config.yaml
acceptance:
  simulation:
    min_trades_per_week: 20  # Reduced from 25

# config_optimized.yaml  
acceptance:
  simulation:
    min_trades_per_week: 25  # Adjusted from 30
```

## Impact Analysis

### **✅ Benefits of Single-Trade Constraint**
1. **Simplified Risk Management**: No need to manage correlation between multiple open positions
2. **Clearer Performance Attribution**: Each trade's impact is isolated
3. **Reduced Maximum Drawdown Risk**: Lower exposure at any given time
4. **Easier to Understand**: Trade sequence is linear and predictable

### **⚠️ Trade-offs**
1. **Lower Total Trade Volume**: Fewer opportunities for profit
2. **Opportunity Cost**: May miss profitable trades while one is already open
3. **Reduced Diversification**: Cannot hedge with opposite positions

### **Expected Results**
- **Trade Frequency**: 20-40 trades/week (conservative) or 25-50 trades/week (optimized)
- **Risk Profile**: Lower overall portfolio risk
- **Win Rate**: Should maintain current levels (54-87% as seen in your recent results)
- **Profit Factor**: May improve due to better risk management

## Verification Steps

### **How to Verify Single-Trade Constraint is Working**
1. **Check Simulation Output**: Look for overlapping trade timestamps
2. **Monitor `open_trades` List**: Should never exceed length of 1
3. **Validate Trade Sequence**: Each trade should close before the next opens

### **Key Simulation Logic**
```python
# In simulate.py, this check now enforces single trade:
if allowed and len(open_trades) < max_positions:  # max_positions = 1
    # Only opens new trade if no trades are currently open
```

## Next Steps

1. **Test the Configuration**: Run the pipeline with both configs to compare results
2. **Monitor Performance**: Ensure trade frequency targets are met
3. **Adjust if Needed**: May need to further reduce cooldown or increase trade caps if frequency is too low

## Rollback Plan

If you need to revert to multi-position trading:
```yaml
simulation:
  max_positions: 2    # Or 3 for more aggressive approach
  cooldown_min: 10    # Original conservative setting
```

The single-trade constraint is now properly enforced across all configuration files. Your system will only open one trade at a time, ensuring the risk management behavior you expected.