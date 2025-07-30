# LIVE TRADING DEBUG SYSTEM USAGE

## Files Created:
1. `live_trading_debug_logger.py` - Core debugging and logging system
2. `enhanced_live_trading.py` - Enhanced live trading with comprehensive logging
3. `analyze_debug_logs.py` - Script to analyze debug logs

## Usage Instructions:

### 1. On Windows VM (Live Trading):
```bash
# Run enhanced live trading (replaces live_trading.py)
python enhanced_live_trading.py
```

### 2. After Trading Session:
```bash
# Create a zip file of debug logs
zip -r debug_logs.zip debug_logs/
```

### 3. Transfer to Mac:
- Download `debug_logs.zip` from VM
- Extract to your Mac development directory

### 4. On Mac (Analysis):
```bash
# Analyze the latest debug session
python analyze_debug_logs.py
```

## What Gets Logged:

### System Status:
- MT5 connection attempts and status
- Trading system initialization
- Position tracking updates
- Single trade constraint checks

### Trade Decisions:
- All trade decisions with expected value and confidence
- Action taken (buy/sell/no_action)
- Reasons for decisions

### Expected Value:
- Detailed expected value calculations
- Probabilistic labeling results
- Ensemble prediction details

### Errors:
- All errors and exceptions
- System failures
- Connection issues

### Performance:
- Trade execution results
- Position management
- Risk management events

## Debug Output Location:
- All logs saved to: `debug_logs/session_YYYYMMDD_HHMMSS/`
- Each session gets its own directory
- Summary file: `debug_summary.json`

## Key Benefits:
1. **Comprehensive Logging**: Every aspect of the trading system is logged
2. **Session Isolation**: Each trading session gets its own log directory
3. **Easy Analysis**: Structured JSON logs for easy parsing
4. **Error Tracking**: All errors and exceptions are captured
5. **Performance Monitoring**: Track expected value and confidence over time

## Next Steps:
1. Deploy `enhanced_live_trading.py` to your Windows VM
2. Run a trading session
3. Download the debug logs
4. Analyze on your Mac to identify issues

## Debugging the Current Issues:

### Issue #1: Expected Value Showing 0.0
The debug system will log:
- Expected value calculations in `expected_value.log`
- Ensemble predictions in `ensemble_predictions.log`
- Trade decisions with EV in `trade_decisions.log`

### Issue #2: All Buy Signals
The debug system will log:
- Signal generation details in `signal_generation.log`
- Direction predictions in `ensemble_predictions.log`
- Trade decisions with actions in `trade_decisions.log`

### Issue #3: Single Trade Constraint Violation
The debug system will log:
- Constraint checks in `system_status.log`
- Position tracking in `system_status.log`
- Trade execution attempts in `system_status.log`

## Analysis Commands:

### Basic Analysis:
```bash
python analyze_debug_logs.py
```

### Manual Analysis:
```bash
# Check specific log files
cat debug_logs/session_YYYYMMDD_HHMMSS/trade_decisions.log
cat debug_logs/session_YYYYMMDD_HHMMSS/errors.log
cat debug_logs/session_YYYYMMDD_HHMMSS/system_status.log
```

### JSON Analysis:
```bash
# Convert logs to readable format
jq '.' debug_logs/session_YYYYMMDD_HHMMSS/trade_decisions.log
```

## Expected Debug Output:

### Successful Session:
```
🔍 DEBUG LOG ANALYSIS
============================================================
📁 Latest session: debug_logs/session_20250730_143022
🔍 Analyzing debug session: debug_logs/session_20250730_143022
============================================================
📊 Session ID: 20250730_143022
📅 Session Start: 2025-07-30T14:30:22.123456

📋 System Status Analysis:
   Total entries: 45
   Events:
     mt5_connection_attempt: 1
     mt5_connected: 1
     market_data_received: 12
     single_trade_constraint_check: 12
     trade_executed: 3
   ✅ MT5 connection attempts detected
   ✅ Trades executed
   ✅ Single trade constraint checks performed

📋 Trade Decisions Analysis:
   Total decisions: 12
   Actions:
     buy: 3
     no_action: 9
   Expected Value Stats:
     Mean: 0.000456
     Min: 0.000123
     Max: 0.000789
     Zero count: 0
   Confidence Stats:
     Mean: 0.623
     Min: 0.512
     Max: 0.745
```

### Problem Session:
```
📋 Trade Decisions Analysis:
   Total decisions: 15
   Actions:
     buy: 15
     sell: 0
   Expected Value Stats:
     Mean: 0.000000
     Min: 0.000000
     Max: 0.000000
     Zero count: 15  ← ISSUE IDENTIFIED
   Confidence Stats:
     Mean: 0.523
     Min: 0.456
     Max: 0.612

📋 Errors Analysis:
   Total errors: 3
   Error types:
     expected_value_calculation_error: 2
     ensemble_prediction_error: 1
```

## Troubleshooting Guide:

### If Expected Value is Always 0:
1. Check `expected_value.log` for calculation errors
2. Check `ensemble_predictions.log` for prediction issues
3. Verify probabilistic labeling in `probabilistic_labels.log`

### If All Signals are Buy:
1. Check `signal_generation.log` for direction bias
2. Check `ensemble_predictions.log` for direction predictions
3. Verify feature engineering in `system_status.log`

### If Single Trade Constraint is Violated:
1. Check `system_status.log` for constraint check failures
2. Check `errors.log` for constraint-related errors
3. Verify position tracking in `system_status.log`

## File Transfer Commands:

### From Windows VM:
```bash
# Create zip file
powershell Compress-Archive -Path debug_logs -DestinationPath debug_logs.zip

# Or use 7zip if available
7z a debug_logs.zip debug_logs/
```

### To Mac:
```bash
# Extract zip file
unzip debug_logs.zip

# Analyze
python analyze_debug_logs.py
```

## Quick Start:
1. Copy `enhanced_live_trading.py` and `live_trading_debug_logger.py` to VM
2. Run: `python enhanced_live_trading.py`
3. Let it run for 30-60 minutes
4. Stop with Ctrl+C
5. Download `debug_logs.zip`
6. Extract and run: `python analyze_debug_logs.py`
7. Review the analysis output for issues 