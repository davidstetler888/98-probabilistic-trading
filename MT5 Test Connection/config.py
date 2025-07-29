# Trading Bot Configuration

# Trading Parameters
SYMBOL = "EURUSD.PRO"  # Trading symbol
LOT_SIZE = 0.01    # Lot size for trades

# Timing Parameters
TRADE_HOLD_TIME = 10    # Seconds to keep trade open
WAIT_TIME = 20         # Seconds to wait between cycles

# Trading Settings
MAX_CYCLES = 5         # Maximum number of trading cycles (None for infinite)
ORDER_TYPE = "BUY"     # "BUY" or "SELL" - type of orders to place

# Risk Management
MAX_POSITIONS = 1      # Maximum number of open positions at once
STOP_LOSS_PIPS = 50    # Stop loss in pips (0 to disable)
TAKE_PROFIT_PIPS = 50  # Take profit in pips (0 to disable)

# MT5 Settings
DEVIATION = 20         # Price deviation for orders
MAGIC_NUMBER = 234000  # Magic number for order identification 