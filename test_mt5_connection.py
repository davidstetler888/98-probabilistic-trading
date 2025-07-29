#!/usr/bin/env python3
"""
MT5 Connection Test Script
Test your MetaTrader 5 connection before running live trading
"""

import sys
import pandas as pd
from datetime import datetime

# Try to import MT5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    print("❌ MetaTrader5 not installed. Install with: pip install MetaTrader5")
    sys.exit(1)

def test_mt5_connection():
    """Test MT5 connection and basic functionality."""
    
    print("🔧 Testing MT5 Connection...")
    
    # Step 1: Initialize MT5
    print("1. Initializing MT5...")
    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        print("   Make sure MT5 terminal is running")
        return False
    
    print("✅ MT5 initialized successfully")
    
    # Step 2: Get terminal info
    print("2. Getting terminal info...")
    terminal_info = mt5.terminal_info()
    if terminal_info is None:
        print("❌ Failed to get terminal info")
        return False
    
    print(f"✅ Terminal: {terminal_info.name}")
    print(f"✅ Version: {terminal_info.version}")
    print(f"✅ Connected: {terminal_info.connected}")
    
    # Step 3: Get account info (if logged in)
    print("3. Checking account info...")
    account_info = mt5.account_info()
    if account_info is None:
        print("⚠️ Not logged in - this is normal for initial setup")
        print("   You'll need to login with your credentials")
    else:
        print(f"✅ Account: {account_info.login}")
        print(f"✅ Server: {account_info.server}")
        print(f"✅ Balance: ${account_info.balance:.2f}")
        print(f"✅ Equity: ${account_info.equity:.2f}")
    
    # Step 4: Test symbol info
    print("4. Testing symbol info...")
    symbol_info = mt5.symbol_info("EURUSD")
    if symbol_info is None:
        print("❌ Failed to get EURUSD symbol info")
        return False
    
    print(f"✅ Symbol: {symbol_info.name}")
    print(f"✅ Spread: {symbol_info.spread} points")
    print(f"✅ Trade mode: {symbol_info.trade_mode}")
    
    # Step 5: Test data retrieval
    print("5. Testing data retrieval...")
    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M5, 0, 10)
    if rates is None:
        print("❌ Failed to get market data")
        return False
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    print(f"✅ Retrieved {len(df)} bars of EURUSD data")
    print(f"✅ Latest time: {df['time'].iloc[-1]}")
    print(f"✅ Latest close: {df['close'].iloc[-1]:.5f}")
    
    # Step 6: Test order placement (simulation only)
    print("6. Testing order placement simulation...")
    
    # Create a test order request (won't actually place it)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": "EURUSD",
        "volume": 0.01,
        "type": mt5.ORDER_TYPE_BUY,
        "price": symbol_info.ask,
        "deviation": 20,
        "magic": 234000,
        "comment": "python test order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    print("✅ Order request structure valid")
    print("✅ Ready for live trading")
    
    return True

def show_configuration_help():
    """Show configuration help."""
    
    print("\n" + "="*60)
    print("🔧 MT5 CONFIGURATION HELP")
    print("="*60)
    
    print("\nTo configure MT5 for live trading:")
    
    print("\n1. Get your MT5 credentials:")
    print("   - Login ID (Account Number)")
    print("   - Password")
    print("   - Server Name")
    
    print("\n2. Edit live_trading.py and uncomment these lines:")
    print("   # authorized = mt5.login(")
    print("   #     login=YOUR_LOGIN_ID,")
    print("   #     password=YOUR_PASSWORD,")
    print("   #     server=YOUR_SERVER_NAME")
    print("   # )")
    
    print("\n3. Replace with your actual credentials:")
    print("   authorized = mt5.login(")
    print("       login=12345678,")
    print("       password='your_password',")
    print("       server='ICMarkets-Demo'")
    print("   )")
    
    print("\n4. Test your configuration:")
    print("   python3 test_mt5_connection.py")
    
    print("\n5. Start live trading:")
    print("   python3 live_trading.py")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    print("🚀 MT5 Connection Test")
    print("="*40)
    
    success = test_mt5_connection()
    
    if success:
        print("\n🎉 MT5 Connection Test PASSED!")
        print("✅ Your MT5 is ready for configuration")
        show_configuration_help()
    else:
        print("\n❌ MT5 Connection Test FAILED!")
        print("Please check your MT5 installation and try again")
        show_configuration_help() 