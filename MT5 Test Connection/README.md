# MT5 Python Test

This repository contains Python code for testing MetaTrader 5 integration with automated trading capabilities.

## Features

- **Automated Trading**: Places trades automatically with configurable timing
- **Risk Management**: Built-in position management and safety features
- **Comprehensive Logging**: Detailed logs for monitoring and debugging
- **Configurable Parameters**: Easy customization of trading settings
- **Error Handling**: Robust error handling and recovery mechanisms

## Trading Pattern

The bot follows this pattern:
1. Places a trade (BUY order by default)
2. Keeps the position open for 10 seconds
3. Closes the position
4. Waits 20 seconds
5. Repeats the cycle

## Prerequisites

1. **MetaTrader 5 Terminal**: Must be installed and running
2. **Python 3.7+**: Required for running the script
3. **Trading Account**: Active MT5 account with sufficient funds
4. **Internet Connection**: For real-time market data

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/davidstetler888/MT5-Python-Test.git
   cd MT5-Python-Test
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure settings** (optional):
   Edit `config.py` to customize trading parameters

## Usage

### Basic Usage

Run the trading bot with default settings:

```bash
python mt5_trading_bot.py
```

### Configuration

Edit `config.py` to customize:

- **SYMBOL**: Trading instrument (default: "EURUSD")
- **LOT_SIZE**: Trade size (default: 0.01)
- **TRADE_HOLD_TIME**: Seconds to keep trade open (default: 10)
- **WAIT_TIME**: Seconds between cycles (default: 20)
- **MAX_CYCLES**: Maximum trading cycles (default: 5)

### Safety Features

- **Position Limits**: Maximum 1 open position at a time
- **Automatic Cleanup**: Closes all positions on exit
- **Error Recovery**: Handles connection and order failures
- **Logging**: Comprehensive logging to `trading_bot.log`

## Important Notes

⚠️ **WARNING**: This is for testing purposes only. Trading involves risk of financial loss.

### Before Running

1. **Ensure MT5 is running** and logged into your account
2. **Check account balance** and ensure sufficient funds
3. **Verify symbol availability** in your MT5 terminal
4. **Test with small lot sizes** first
5. **Monitor the bot** while it's running

### Safety Checklist

- [ ] MT5 terminal is running and connected
- [ ] Account has sufficient funds
- [ ] Symbol is available for trading
- [ ] Lot size is appropriate for your account
- [ ] You understand the risks involved

## Logging

The bot creates detailed logs in `trading_bot.log` including:
- Connection status
- Order placement and execution
- Position management
- Errors and warnings
- Trading cycle information

## Troubleshooting

### Common Issues

1. **"MT5 initialization failed"**
   - Ensure MT5 terminal is running
   - Check if MT5 is properly installed

2. **"Trading account not available"**
   - Log into your MT5 account
   - Verify account credentials

3. **"Symbol not found"**
   - Check if the symbol is available in your MT5
   - Verify symbol name spelling

4. **"Order failed"**
   - Check account balance
   - Verify symbol is tradeable
   - Check market hours

### Getting Help

1. Check the log file `trading_bot.log` for detailed error messages
2. Verify all prerequisites are met
3. Test with different symbols or smaller lot sizes

## Development

### Project Structure

```
MT5-Python-Test/
├── mt5_trading_bot.py    # Main trading script
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── trading_bot.log      # Log file (created when running)
```

### Customization

The `MT5TradingBot` class is designed to be easily extensible:

- Add new order types
- Implement different trading strategies
- Add risk management features
- Customize logging and monitoring

## License

This project is for educational and testing purposes. Use at your own risk.

## Disclaimer

This software is provided "as is" without warranty. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. 