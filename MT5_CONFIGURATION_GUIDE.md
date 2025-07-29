# MT5 Configuration Guide for Revolutionary Trading System

## 🔧 Simple MT5 Setup Instructions (No Credentials Required!)

### Step 1: Install MetaTrader 5 Python Package

```bash
pip install MetaTrader5
```

### Step 2: Start Your MT5 Terminal

1. **Launch MetaTrader 5** on your computer
2. **Log into your account** in the MT5 terminal
3. **Keep MT5 running** - the Python script will use your existing session

### Step 3: Test MT5 Connection

Run this test to verify your connection:

```bash
python3 test_mt5_connection.py
```

### Step 4: Start Live Trading

Once the connection test passes:

```bash
python3 live_trading.py
```

## 🎯 Key Benefits of This Approach

✅ **No Credentials in Code** - More secure
✅ **Uses Existing MT5 Session** - No additional login needed
✅ **Simpler Setup** - Just start MT5 and run the script
✅ **More Reliable** - Uses your established MT5 connection

## 📋 What You Need

**Before Running:**
1. ✅ **MT5 Terminal Running** - Must be launched and logged in
2. ✅ **Active Account** - Logged into your MT5 account
3. ✅ **Internet Connection** - For real-time data
4. ✅ **Python MT5 Package** - `pip install MetaTrader5`

**No Need For:**
❌ **Login credentials in Python code**
❌ **Server configuration**
❌ **Complex setup procedures**

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

## 🔍 How It Works

1. **MT5 Terminal** - You log in normally through MT5
2. **Python Script** - Connects to your existing MT5 session
3. **Trading** - Uses your authenticated session for orders and data

## ⚠️ Important Notes

1. **MT5 Must Be Running** - The script connects to your existing MT5 session
2. **Stay Logged In** - Don't log out of MT5 while the script is running
3. **Demo First** - Test with demo account before live trading
4. **Monitor Positions** - Check your MT5 terminal for open positions

## 🔧 Troubleshooting

### Common Issues:

**1. "Failed to initialize MT5"**
- Make sure MT5 terminal is running
- Check if MT5 is installed correctly

**2. "Account info not available"**
- Make sure you're logged into MT5 terminal
- Check if your account is active

**3. "No tick data for EURUSD"**
- Check internet connection
- Verify EURUSD is available in your MT5
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

1. **Check MT5 terminal is running and logged in**
2. **Verify EURUSD symbol is available**
3. **Test with demo account first**
4. **Check internet connection**

## 🎉 Ready to Trade!

Your revolutionary trading system is now configured with the simplest possible MT5 connection:

1. **Start MT5** and log in
2. **Test connection**: `python3 test_mt5_connection.py`
3. **Start trading**: `python3 live_trading.py`

That's it! No complex configuration needed! 🚀 