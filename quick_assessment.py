#!/usr/bin/env python3
"""
Quick Performance Assessment
Immediate snapshot of live trading performance for the revolutionary system.
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

def quick_assessment():
    """Perform a quick assessment of live trading performance."""
    print("🎯 QUICK PERFORMANCE ASSESSMENT")
    print("="*50)
    
    # Connect to MT5
    if MT5_AVAILABLE:
        if not mt5.initialize():
            print("❌ Failed to connect to MT5")
            return
        print("✅ Connected to MT5")
    else:
        print("⚠️ Running in simulation mode")
    
    # Get account info
    if MT5_AVAILABLE:
        account_info = mt5.account_info()
        if account_info:
            balance = account_info.balance
            equity = account_info.equity
            profit = account_info.profit
        else:
            balance = equity = profit = 0
    else:
        balance = equity = 10000.0
        profit = 0.0
    
    print(f"\n💰 Account Status:")
    print(f"   Balance: ${balance:,.2f}")
    print(f"   Equity: ${equity:,.2f}")
    print(f"   P&L: ${profit:+,.2f}")
    
    # Get positions
    if MT5_AVAILABLE:
        positions = mt5.positions_get()
        if positions is None:
            positions = []
    else:
        positions = []
    
    print(f"\n📋 Open Positions: {len(positions)}")
    if positions:
        total_pnl = sum(pos.profit for pos in positions)
        print(f"   Total Position P&L: ${total_pnl:+,.2f}")
        
        for pos in positions:
            print(f"   {pos.symbol} {pos.type} {pos.volume:.2f} @ {pos.price_current:.5f} "
                  f"P&L: ${pos.profit:+,.2f}")
    
    # Get recent trades
    if MT5_AVAILABLE:
        from_date = datetime.now() - timedelta(days=7)
        history = mt5.history_deals_get(from_date, datetime.now())
        
        if history:
            trades = [deal for deal in history if deal.entry == 1]  # Entry deals only
            print(f"\n📊 Recent Trades (7 days): {len(trades)}")
            
            if trades:
                winning_trades = [t for t in trades if t.profit > 0]
                losing_trades = [t for t in trades if t.profit < 0]
                
                win_rate = len(winning_trades) / len(trades) if trades else 0
                total_profit = sum(t.profit for t in winning_trades)
                total_loss = abs(sum(t.profit for t in losing_trades))
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
                
                print(f"   Win Rate: {win_rate:.1%}")
                print(f"   Profit Factor: {profit_factor:.2f}")
                print(f"   Total Profit: ${total_profit:,.2f}")
                print(f"   Total Loss: ${total_loss:,.2f}")
                
                # Performance assessment
                print(f"\n🎯 Quick Assessment:")
                if win_rate >= 0.58:
                    print("   ✅ Win Rate: EXCELLENT (58%+)")
                elif win_rate >= 0.50:
                    print("   ⚠️ Win Rate: GOOD (50%+)")
                else:
                    print("   ❌ Win Rate: NEEDS IMPROVEMENT (<50%)")
                
                if profit_factor >= 1.3:
                    print("   ✅ Profit Factor: EXCELLENT (1.3+)")
                elif profit_factor >= 1.1:
                    print("   ⚠️ Profit Factor: GOOD (1.1+)")
                else:
                    print("   ❌ Profit Factor: NEEDS IMPROVEMENT (<1.1)")
                
                if len(trades) >= 25:
                    print("   ✅ Trade Frequency: EXCELLENT (25+ trades/week)")
                elif len(trades) >= 15:
                    print("   ⚠️ Trade Frequency: GOOD (15+ trades/week)")
                else:
                    print("   ❌ Trade Frequency: LOW (<15 trades/week)")
            else:
                print("   No trades in the last 7 days")
        else:
            print(f"\n📊 Recent Trades: No trade history available")
    else:
        print(f"\n📊 Recent Trades: Simulation mode - no real data")
    
    print(f"\n" + "="*50)
    print("💡 For detailed monitoring, run: python3 monitor_live_trading.py")

if __name__ == "__main__":
    quick_assessment() 