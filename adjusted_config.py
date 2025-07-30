
# Adjusted Signal Criteria Configuration
# Generated: 2025-07-29 20:18:34

# Signal Generation Criteria
SIGNAL_CRITERIA = {
    'min_expected_value': 0.0002,  # Minimum expected value (pips)
    'min_confidence': 0.5,          # Minimum confidence (0-1)
    'min_signal_quality': 0.3,                             # Minimum signal quality
    'max_spread_impact': 0.0002,                           # Maximum spread impact
}

# Position Sizing
POSITION_SIZING = {
    'base_risk_per_trade': 0.02,  # Base risk per trade
    'confidence_multiplier': 1.0,                           # Confidence multiplier
    'max_position_size': 0.05,                              # Maximum position size
    'min_position_size': 0.01,                              # Minimum position size
}

# Risk Management
RISK_MANAGEMENT = {
    'max_daily_risk': 0.05,        # Maximum daily risk
    'max_drawdown': 0.15,                                   # Maximum drawdown
    'max_positions': 3,                                     # Maximum open positions
    'cooldown_after_loss': 300,                             # Cooldown after loss (seconds)
}

# Trading Parameters
TRADING_PARAMS = {
    'symbol': 'EURUSD.PRO',
    'timeframe': 'M5',
    'check_interval': 60,                                   # Check interval (seconds)
    'min_trade_interval': 300,                              # Minimum time between trades
}

# Performance Targets
PERFORMANCE_TARGETS = {
    'min_win_rate': 0.58,                                   # Minimum win rate
    'min_profit_factor': 1.3,                               # Minimum profit factor
    'min_trades_per_week': 25,                              # Minimum trades per week
    'max_trades_per_week': 50,                              # Maximum trades per week
}
