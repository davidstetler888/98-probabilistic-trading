#!/usr/bin/env python3
"""
Simple test script to verify MT5 connection and basic functionality
without placing any trades.
"""

import MetaTrader5 as mt5
import logging
from config import SYMBOL

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mt5_connection():
    """Test basic MT5 connection and functionality"""
    
    print("=== MT5 Connection Test ===")
    
    try:
        # Initialize MT5
        print("1. Initializing MT5...")
        if not mt5.initialize():
            print(f"❌ MT5 initialization failed: {mt5.last_error()}")
            return False
        print("✅ MT5 initialized successfully")
        
        # Check terminal info
        print("2. Checking terminal info...")
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            print("❌ Terminal info not available")
            return False
        print(f"✅ Terminal: {terminal_info.name}")
        # Check if version attribute exists before accessing it
        if hasattr(terminal_info, 'version'):
            print(f"   Version: {terminal_info.version}")
        else:
            print("   Version: Not available")
        print(f"   Connected: {terminal_info.connected}")
        
        # Check account info
        print("3. Checking account info...")
        account_info = mt5.account_info()
        if account_info is None:
            print("❌ Account info not available")
            return False
        print(f"✅ Account: {account_info.login}")
        print(f"   Server: {account_info.server}")
        print(f"   Balance: {account_info.balance}")
        print(f"   Equity: {account_info.equity}")
        print(f"   Margin: {account_info.margin}")
        print(f"   Free Margin: {account_info.margin_free}")
        
        # Check symbol info
        print("4. Checking symbol info...")
        symbol = SYMBOL
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"❌ Symbol {symbol} not found")
            print("   Trying to select symbol...")
            if mt5.symbol_select(symbol, True):
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    print(f"❌ Still cannot find symbol {symbol}")
                    return False
            else:
                print(f"❌ Failed to select symbol {symbol}")
                return False
        
        print(f"✅ Symbol: {symbol_info.name}")
        print(f"   Digits: {symbol_info.digits}")
        print(f"   Spread: {symbol_info.spread}")
        print(f"   Trade mode: {symbol_info.trade_mode}")
        print(f"   Visible: {symbol_info.visible}")
        
        # Check tick data
        print("5. Checking tick data...")
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"❌ No tick data for {symbol}")
            return False
        print(f"✅ Tick data available")
        print(f"   Bid: {tick.bid}")
        print(f"   Ask: {tick.ask}")
        print(f"   Time: {tick.time}")
        
        print("\n🎉 All tests passed! MT5 is ready for trading.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False
        
    finally:
        # Shutdown MT5
        mt5.shutdown()
        print("MT5 connection closed")

if __name__ == "__main__":
    test_mt5_connection() 