#!/usr/bin/env python3
"""
Check Live Trading System Status
Verifies if the system is running and generating signals.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

def check_system_status():
    """Check if the live trading system is running and generating signals."""
    print("🔍 CHECKING LIVE TRADING SYSTEM STATUS")
    print("="*50)
    
    # 1. Check MT5 Connection
    print("\n1️⃣ MT5 CONNECTION:")
    if not MT5_AVAILABLE:
        print("❌ MetaTrader5 not available")
        return False
    
    if not mt5.initialize():
        print("❌ MT5 not connected")
        return False
    
    print("✅ MT5 connected")
    
    # 2. Check Account Info
    account_info = mt5.account_info()
    if account_info:
        print(f"✅ Account: {account_info.login}")
        print(f"✅ Balance: ${account_info.balance:,.2f}")
    else:
        print("❌ Cannot get account info")
        return False
    
    # 3. Check Symbol
    symbol_info = mt5.symbol_info("EURUSD.PRO")
    if symbol_info:
        print(f"✅ EURUSD.PRO available (Spread: {symbol_info.spread} points)")
    else:
        print("❌ EURUSD.PRO not available")
        return False
    
    # 4. Check Recent Market Data
    print("\n2️⃣ MARKET DATA:")
    rates = mt5.copy_rates_from_pos("EURUSD.PRO", mt5.TIMEFRAME_M5, 0, 100)
    if rates is None:
        print("❌ No market data available")
        return False
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    print(f"✅ Market data: {len(df)} bars")
    print(f"✅ Latest time: {df['time'].iloc[-1]}")
    print(f"✅ Latest close: {df['close'].iloc[-1]:.5f}")
    
    # Check data freshness
    latest_time = df['time'].iloc[-1]
    time_diff = datetime.now() - latest_time
    if time_diff.total_seconds() > 300:  # 5 minutes
        print(f"⚠️ Data may be stale ({time_diff.total_seconds():.0f}s old)")
    else:
        print("✅ Data is recent")
    
    # 5. Check Trade History
    print("\n3️⃣ TRADE HISTORY:")
    from_date = datetime.now() - timedelta(days=1)
    history = mt5.history_deals_get(from_date, datetime.now())
    
    if history:
        today_trades = [deal for deal in history if deal.time >= from_date.timestamp()]
        print(f"✅ Trades today: {len(today_trades)}")
        
        if today_trades:
            print("Recent trades:")
            for trade in today_trades[-3:]:
                trade_time = datetime.fromtimestamp(trade.time)
                print(f"   {trade_time.strftime('%H:%M:%S')} - {trade.symbol} {trade.volume:.2f} P&L: ${trade.profit:+.2f}")
        else:
            print("❌ No trades today")
    else:
        print("❌ Cannot get trade history")
    
    # 6. Check Open Positions
    print("\n4️⃣ OPEN POSITIONS:")
    positions = mt5.positions_get()
    if positions:
        print(f"✅ Open positions: {len(positions)}")
        for pos in positions:
            print(f"   {pos.symbol} {pos.type} {pos.volume:.2f} P&L: ${pos.profit:+.2f}")
    else:
        print("✅ No open positions")
    
    # 7. System Status Assessment
    print("\n5️⃣ SYSTEM STATUS ASSESSMENT:")
    
    # Check if it's weekend
    if datetime.now().weekday() >= 5:
        print("⚠️ Weekend - Forex market closed")
        print("   This is normal - no trades expected")
        return True
    
    # Check market hours
    current_hour = datetime.now().hour
    if current_hour < 5 or current_hour > 23:
        print("⚠️ Outside main trading hours")
        print("   Main forex hours: 5:00-23:00 UTC")
    
    # Check if we have recent data but no trades
    if time_diff.total_seconds() < 300 and len(today_trades) == 0:
        print("❌ ISSUE DETECTED: Recent data but no trades")
        print("   Possible causes:")
        print("   1. Live trading system not running")
        print("   2. Signal generation issues")
        print("   3. Risk management blocking trades")
        print("   4. Model training issues")
        return False
    
    print("✅ System appears to be functioning normally")
    return True

def test_signal_generation():
    """Test if the system can generate signals."""
    print("\n6️⃣ TESTING SIGNAL GENERATION:")
    
    try:
        # Import the live trading system
        from phase3_live_trading_preparation import LiveTradingSystem
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=1000, freq='5T')
        prices = 1.1000 + np.cumsum(np.random.normal(0, 0.0001, 1000))
        
        training_data = pd.DataFrame({
            'open': prices,
            'close': prices + np.random.normal(0, 0.0001, 1000),
            'high': prices + np.abs(np.random.normal(0, 0.0003, 1000)),
            'low': prices - np.abs(np.random.normal(0, 0.0003, 1000)),
            'volume': np.random.randint(100, 1000, 1000)
        }, index=dates)
        
        # Initialize system
        print("   Initializing system...")
        live_system = LiveTradingSystem()
        success = live_system.initialize_system(training_data)
        
        if not success:
            print("   ❌ Failed to initialize system")
            return False
        
        print("   ✅ System initialized")
        
        # Test signal generation
        print("   Testing signal generation...")
        recent_data = training_data.tail(100)
        trade_decision = live_system.process_market_data(recent_data)
        
        print(f"   Signal generated: {trade_decision['action']}")
        if trade_decision['action'] != 'no_action':
            print(f"   Direction: {trade_decision.get('direction', 'N/A')}")
            print(f"   Confidence: {trade_decision.get('confidence', 0):.3f}")
            print(f"   Expected Value: {trade_decision.get('expected_value', 0):.6f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing signal generation: {e}")
        return False

def main():
    """Main function."""
    print("🎯 Live Trading System Status Check")
    print("="*50)
    
    # Check basic system status
    status_ok = check_system_status()
    
    if status_ok:
        # Test signal generation
        signal_ok = test_signal_generation()
        
        print("\n📋 SUMMARY:")
        if signal_ok:
            print("✅ System appears to be working correctly")
            print("   If no trades are executing, check:")
            print("   1. Is live_trading.py running on your VM?")
            print("   2. Are there any error messages?")
            print("   3. Is it weekend (market closed)?")
            print("   4. Are signal criteria being met?")
        else:
            print("❌ Signal generation issues detected")
            print("   Check system initialization and model training")
    else:
        print("❌ System status issues detected")
        print("   Check MT5 connection and market data")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main() 