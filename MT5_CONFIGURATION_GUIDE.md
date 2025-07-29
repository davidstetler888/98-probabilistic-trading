# MT5 Configuration Guide for Revolutionary Trading System

## 🔧 Complete MT5 Setup Instructions

### Step 1: Install MetaTrader 5 Python Package

```bash
pip install MetaTrader5
```

### Step 2: Get Your MT5 Credentials

You need these details from your MT5 broker:

1. **Login ID** (Account Number)
2. **Password** 
3. **Server Name** (e.g., "ICMarkets-Demo", "Pepperstone-Live", etc.)
4. **Symbol** (usually "EURUSD")

### Step 3: Configure MT5 Connection

Edit the `live_trading.py` file and update these lines:

```python
# In live_trading.py, around line 39, uncomment and update:
authorized = mt5.login(
    login=YOUR_LOGIN_ID,           # Replace with your account number
    password=YOUR_PASSWORD,        # Replace with your password
    server=YOUR_SERVER_NAME        # Replace with your server name
)
```

**Example Configuration:**
```python
authorized = mt5.login(
    login=12345678,                # Your MT5 account number
    password="your_password_here", # Your MT5 password
    server="ICMarkets-Demo"        # Your broker's server name
)
```

### Step 4: Test MT5 Connection

Run this test to verify your connection:

```bash
python3 test_mt5_connection.py
```

### Step 5: Configure Trading Parameters

Update these settings in `live_trading.py`:

```python
# Trading parameters (around line 133)
def run_live_trading(self, symbol="EURUSD", check_interval=60):
    """
    symbol: Trading pair (default: "EURUSD")
    check_interval: How often to check for new signals (default: 60 seconds)
    """
```

### Step 6: Risk Management Settings

Configure your risk parameters:

```python
# In phase3_live_trading_preparation.py
risk_config = {
    'max_position_size': 0.05,     # 5% of balance per trade
    'max_daily_risk': 0.02,        # 2% maximum daily risk
    'max_drawdown': 0.12,          # 12% maximum drawdown
    'stop_loss_pips': 20,          # 20 pips stop loss
    'take_profit_pips': 40,        # 40 pips take profit
}
```

## 📋 Common MT5 Broker Configurations

### Demo Account Examples:

**IC Markets Demo:**
```python
authorized = mt5.login(
    login=12345678,
    password="demo_password",
    server="ICMarkets-Demo"
)
```

**Pepperstone Demo:**
```python
authorized = mt5.login(
    login=12345678,
    password="demo_password", 
    server="Pepperstone-Demo"
)
```

**FXCM Demo:**
```python
authorized = mt5.login(
    login=12345678,
    password="demo_password",
    server="FXCM-Demo"
)
```

### Live Account Examples:

**IC Markets Live:**
```python
authorized = mt5.login(
    login=12345678,
    password="your_live_password",
    server="ICMarkets-Live"
)
```

## 🔍 How to Find Your MT5 Credentials

### 1. Open MetaTrader 5 Terminal
- Launch your MT5 terminal
- Look at the top of the terminal window

### 2. Find Account Information
- **Account Number**: Usually displayed in the top toolbar
- **Server**: Shown in the account info panel
- **Password**: The password you use to login to MT5

### 3. Alternative Method
- In MT5, go to **File → Login to Trade Account**
- This will show your account details

## ⚠️ Important Security Notes

1. **Never commit credentials to Git**
   - Keep your password secure
   - Use environment variables for production

2. **Start with Demo Account**
   - Test everything on demo first
   - Only move to live after thorough testing

3. **Monitor Your Account**
   - Check positions regularly
   - Monitor balance and equity

## 🚀 Quick Start Commands

### 1. Test Connection
```bash
python3 test_mt5_connection.py
```

### 2. Train System (if not done)
```bash
python3 train_system.py EURUSD.PRO_M5.csv
```

### 3. Start Live Trading
```bash
python3 live_trading.py
```

### 4. Monitor Performance
```bash
python3 monitor_performance.py
```

## 🔧 Troubleshooting

### Common Issues:

**1. "Failed to initialize MT5"**
- Make sure MT5 terminal is running
- Check if MT5 is installed correctly

**2. "Failed to login"**
- Verify your credentials
- Check if server name is correct
- Ensure account is active

**3. "No data received"**
- Check internet connection
- Verify symbol name (should be "EURUSD")
- Ensure market is open

### Support Commands:

```bash
# Check MT5 installation
python3 -c "import MetaTrader5; print('MT5 installed successfully')"

# Test basic connection
python3 -c "import MetaTrader5 as mt5; print('MT5 initialized:', mt5.initialize())"

# List available symbols
python3 -c "import MetaTrader5 as mt5; mt5.initialize(); print(mt5.symbols_get()[:5])"
```

## 📞 Need Help?

If you encounter issues:

1. **Check MT5 terminal is running**
2. **Verify your credentials**
3. **Test with demo account first**
4. **Check broker's MT5 documentation**

Your revolutionary trading system is ready - just configure MT5 and you'll be live trading! 🎯 