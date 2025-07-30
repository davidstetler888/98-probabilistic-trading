#!/usr/bin/env python3
"""
Diagnostic Tool for No Trades Issue
Identifies why the revolutionary trading system isn't executing trades.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

def diagnose_no_trades():
    """Diagnose why no trades are being executed."""
    print("🔍 DIAGNOSING NO TRADES ISSUE")
    print("="*50)
    
    # 1. Check MT5 Connection
    print("\n1️⃣ CHECKING MT5 CONNECTION...")
    if not MT5_AVAILABLE:
        print("❌ MetaTrader5 not installed")
        print("   Solution: pip install MetaTrader5")
        return
    
    if not mt5.initialize():
        print(f"❌ Failed to initialize MT5: {mt5.last_error()}")
        print("   Solution: Make sure MT5 terminal is running and logged in")
        return
    
    print("✅ MT5 connected successfully")
    
    # 2. Check Account Status
    print("\n2️⃣ CHECKING ACCOUNT STATUS...")
    account_info = mt5.account_info()
    if account_info is None:
        print("❌ Cannot get account info")
        return
    
    print(f"✅ Account: {account_info.login}")
    print(f"✅ Server: {account_info.server}")
    print(f"✅ Balance: ${account_info.balance:,.2f}")
    print(f"✅ Equity: ${account_info.equity:,.2f}")
    print(f"✅ Free Margin: ${account_info.margin_free:,.2f}")
    
    if account_info.margin_free < 100:
        print("⚠️ Low free margin - may prevent trades")
    
    # 3. Check Symbol Availability
    print("\n3️⃣ CHECKING SYMBOL AVAILABILITY...")
    symbol_info = mt5.symbol_info("EURUSD.PRO")
    if symbol_info is None:
        print("❌ EURUSD.PRO symbol not found")
        print("   Trying to select symbol...")
        if mt5.symbol_select("EURUSD.PRO", True):
            symbol_info = mt5.symbol_info("EURUSD.PRO")
            if symbol_info is None:
                print("❌ Still cannot find EURUSD.PRO symbol")
                return
        else:
            print("❌ Failed to select EURUSD.PRO symbol")
            return
    
    print("✅ EURUSD.PRO symbol available")
    print(f"✅ Spread: {symbol_info.spread} points")
    print(f"✅ Trade mode: {symbol_info.trade_mode}")
    
    if symbol_info.trade_mode == 0:
        print("❌ Trading disabled for this symbol")
        return
    
    # 4. Check Market Hours
    print("\n4️⃣ CHECKING MARKET HOURS...")
    current_time = datetime.now()
    print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Forex market hours (simplified)
    weekday = current_time.weekday()
    hour = current_time.hour
    
    if weekday >= 5:  # Weekend
        print("❌ Market closed (weekend)")
        print("   Forex market is closed on weekends")
        return
    elif hour < 5 or hour > 23:  # Outside main hours
        print("⚠️ Outside main trading hours")
        print("   Main forex hours: 5:00-23:00 UTC")
    else:
        print("✅ Market should be open")
    
    # 5. Check Recent Data
    print("\n5️⃣ CHECKING RECENT MARKET DATA...")
    rates = mt5.copy_rates_from_pos("EURUSD.PRO", mt5.TIMEFRAME_M5, 0, 10)
    if rates is None:
        print("❌ Cannot get market data")
        return
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    print(f"✅ Latest data: {len(df)} bars")
    print(f"✅ Latest time: {df['time'].iloc[-1]}")
    print(f"✅ Latest close: {df['close'].iloc[-1]:.5f}")
    
    # Check if data is recent
    latest_time = df['time'].iloc[-1]
    time_diff = datetime.now() - latest_time
    
    if time_diff.total_seconds() > 300:  # 5 minutes
        print(f"⚠️ Data may be stale (last update: {time_diff.total_seconds():.0f}s ago)")
    
    # 6. Check Trade History
    print("\n6️⃣ CHECKING TRADE HISTORY...")
    from_date = datetime.now() - timedelta(days=1)
    history = mt5.history_deals_get(from_date, datetime.now())
    
    if history is None:
        print("❌ Cannot get trade history")
        return
    
    today_trades = [deal for deal in history if deal.time >= from_date.timestamp()]
    print(f"✅ Trades today: {len(today_trades)}")
    
    if today_trades:
        print("Recent trades:")
        for trade in today_trades[-5:]:  # Last 5 trades
            trade_time = datetime.fromtimestamp(trade.time)
            print(f"   {trade_time.strftime('%H:%M:%S')} - {trade.symbol} {deal.type} {deal.volume:.2f} P&L: ${deal.profit:+.2f}")
    else:
        print("❌ No trades executed today")
    
    # 7. Check if Live Trading System is Running
    print("\n7️⃣ CHECKING LIVE TRADING SYSTEM...")
    print("   This requires checking if your live_trading.py script is running")
    print("   Check your VM for:")
    print("   - Python processes running live_trading.py")
    print("   - Any error messages in the console")
    print("   - System logs")
    
    # 8. Check Signal Generation
    print("\n8️⃣ CHECKING SIGNAL GENERATION...")
    print("   The system should be generating signals if:")
    print("   - Market data is available")
    print("   - Models are trained")
    print("   - Signal criteria are met")
    print("   - Risk management allows trading")
    
    # 9. Common Issues and Solutions
    print("\n9️⃣ COMMON ISSUES AND SOLUTIONS...")
    print("\n🔧 If no trades are executing:")
    print("   1. Check if live_trading.py is running on your VM")
    print("   2. Verify MT5 terminal is open and logged in")
    print("   3. Check market hours (forex is closed weekends)")
    print("   4. Verify EURUSD.PRO symbol is available")
    print("   5. Check account has sufficient margin")
    print("   6. Review system logs for errors")
    print("   7. Ensure models were trained properly")
    print("   8. Check signal generation criteria")
    
    print("\n📞 Next Steps:")
    print("   1. Check your VM console for live_trading.py output")
    print("   2. Look for any error messages")
    print("   3. Verify the system is actually running")
    print("   4. Check if signals are being generated")
    print("   5. Review risk management settings")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    diagnose_no_trades() 