#!/usr/bin/env python3
"""
Live Trading Performance Monitor
Comprehensive monitoring and assessment tool for the revolutionary trading system.
"""

import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️ MetaTrader5 not available - running in simulation mode")

class LiveTradingMonitor:
    """
    Comprehensive monitoring system for live trading performance.
    Provides real-time assessment, alerts, and performance analysis.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.performance_history = []
        self.alerts = []
        self.start_time = datetime.now()
        
    def _get_default_config(self) -> Dict:
        """Get monitoring configuration."""
        return {
            'performance_thresholds': {
                'min_win_rate': 0.58,      # 58% minimum win rate
                'min_profit_factor': 1.3,   # 1.3 minimum profit factor
                'max_drawdown': 0.15,       # 15% maximum drawdown
                'min_sharpe': 1.0,          # 1.0 minimum Sharpe ratio
                'min_trades_per_week': 25,  # 25 minimum trades per week
                'max_trades_per_week': 50   # 50 maximum trades per week
            },
            'alert_thresholds': {
                'win_rate_warning': 0.55,   # Warning below 55%
                'profit_factor_warning': 1.1, # Warning below 1.1
                'drawdown_warning': 0.10,   # Warning above 10%
                'no_trades_hours': 4        # Alert if no trades in 4 hours
            },
            'monitoring_interval': 60,      # Check every 60 seconds
            'performance_window': 100,      # Rolling window for metrics
            'export_interval': 3600        # Export performance every hour
        }
    
    def connect_to_mt5(self) -> bool:
        """Connect to MetaTrader 5."""
        if not MT5_AVAILABLE:
            print("❌ MetaTrader5 not available")
            return False
            
        if not mt5.initialize():
            print(f"❌ Failed to initialize MT5: {mt5.last_error()}")
            return False
            
        return True
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get current account information."""
        if not MT5_AVAILABLE:
            return {
                'balance': 10000.0,
                'equity': 10000.0,
                'profit': 0.0,
                'margin': 0.0,
                'free_margin': 10000.0
            }
        
        account_info = mt5.account_info()
        if account_info is None:
            return {}
            
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'profit': account_info.profit,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free
        }
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current open positions."""
        if not MT5_AVAILABLE:
            return []
        
        positions = mt5.positions_get()
        if positions is None:
            return []
            
        return [
            {
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'buy' if pos.type == 0 else 'sell',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'price_current': pos.price_current,
                'profit': pos.profit,
                'swap': pos.swap,
                'time': datetime.fromtimestamp(pos.time)
            }
            for pos in positions
        ]
    
    def get_trade_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get trade history for the specified number of days."""
        if not MT5_AVAILABLE:
            return self._get_mock_trade_history()
        
        from_date = datetime.now() - timedelta(days=days)
        history = mt5.history_deals_get(from_date, datetime.now())
        
        if history is None:
            return []
            
        trades = []
        for deal in history:
            if deal.entry == 1:  # Only entry deals
                trades.append({
                    'ticket': deal.ticket,
                    'symbol': deal.symbol,
                    'type': 'buy' if deal.type == 0 else 'sell',
                    'volume': deal.volume,
                    'price': deal.price,
                    'profit': deal.profit,
                    'swap': deal.swap,
                    'time': datetime.fromtimestamp(deal.time),
                    'comment': deal.comment
                })
        
        return trades
    
    def _get_mock_trade_history(self) -> List[Dict[str, Any]]:
        """Generate mock trade history for testing."""
        trades = []
        base_time = datetime.now() - timedelta(days=7)
        
        for i in range(20):
            trade_time = base_time + timedelta(hours=i*2)
            profit = np.random.normal(10, 50)  # Random profit/loss
            
            trades.append({
                'ticket': 1000 + i,
                'symbol': 'EURUSD.PRO',
                'type': 'buy' if np.random.random() > 0.5 else 'sell',
                'volume': 0.01,
                'price': 1.1000 + np.random.normal(0, 0.001),
                'profit': profit,
                'swap': -0.5,
                'time': trade_time,
                'comment': 'python_ea'
            })
        
        return trades
    
    def calculate_performance_metrics(self, trades: List[Dict[str, Any]], 
                                    account_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_profit': 0.0,
                'total_loss': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'trades_per_week': 0.0,
                'avg_trade_duration': 0.0
            }
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['profit'] > 0]
        losing_trades = [t for t in trades if t['profit'] < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        total_profit = sum(t['profit'] for t in winning_trades)
        total_loss = abs(sum(t['profit'] for t in losing_trades))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        avg_win = total_profit / len(winning_trades) if winning_trades else 0
        avg_loss = total_loss / len(losing_trades) if losing_trades else 0
        
        # Drawdown calculation
        equity_curve = []
        running_equity = account_info['balance']
        
        for trade in sorted(trades, key=lambda x: x['time']):
            running_equity += trade['profit']
            equity_curve.append(running_equity)
        
        if equity_curve:
            peak = max(equity_curve)
            current_equity = equity_curve[-1]
            max_drawdown = (peak - current_equity) / peak if peak > 0 else 0
        else:
            max_drawdown = 0
        
        # Sharpe ratio (simplified)
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve)
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Trades per week
        if trades:
            first_trade = min(t['time'] for t in trades)
            last_trade = max(t['time'] for t in trades)
            days_elapsed = (last_trade - first_trade).days + 1
            trades_per_week = (total_trades / days_elapsed) * 7 if days_elapsed > 0 else 0
        else:
            trades_per_week = 0
        
        # Average trade duration
        if len(trades) >= 2:
            durations = []
            for i in range(1, len(trades)):
                duration = (trades[i]['time'] - trades[i-1]['time']).total_seconds() / 3600  # hours
                durations.append(duration)
            avg_trade_duration = np.mean(durations) if durations else 0
        else:
            avg_trade_duration = 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trades_per_week': trades_per_week,
            'avg_trade_duration': avg_trade_duration
        }
    
    def check_performance_thresholds(self, metrics: Dict[str, Any]) -> Dict[str, bool]:
        """Check if performance meets target thresholds."""
        thresholds = self.config['performance_thresholds']
        
        return {
            'win_rate_ok': metrics['win_rate'] >= thresholds['min_win_rate'],
            'profit_factor_ok': metrics['profit_factor'] >= thresholds['min_profit_factor'],
            'drawdown_ok': metrics['max_drawdown'] <= thresholds['max_drawdown'],
            'sharpe_ok': metrics['sharpe_ratio'] >= thresholds['min_sharpe'],
            'trades_per_week_ok': (thresholds['min_trades_per_week'] <= 
                                 metrics['trades_per_week'] <= 
                                 thresholds['max_trades_per_week'])
        }
    
    def generate_alerts(self, metrics: Dict[str, Any], 
                       positions: List[Dict[str, Any]]) -> List[str]:
        """Generate performance alerts."""
        alerts = []
        alert_thresholds = self.config['alert_thresholds']
        
        # Performance alerts
        if metrics['win_rate'] < alert_thresholds['win_rate_warning']:
            alerts.append(f"⚠️ Win rate warning: {metrics['win_rate']:.1%} (target: 58%+)")
        
        if metrics['profit_factor'] < alert_thresholds['profit_factor_warning']:
            alerts.append(f"⚠️ Profit factor warning: {metrics['profit_factor']:.2f} (target: 1.3+)")
        
        if metrics['max_drawdown'] > alert_thresholds['drawdown_warning']:
            alerts.append(f"⚠️ Drawdown warning: {metrics['max_drawdown']:.1%} (max: 15%)")
        
        # Trading activity alerts
        if metrics['total_trades'] == 0:
            alerts.append("⚠️ No trades executed yet")
        elif metrics['trades_per_week'] < 10:
            alerts.append(f"⚠️ Low trading activity: {metrics['trades_per_week']:.1f} trades/week")
        
        # Position alerts
        if len(positions) > 3:
            alerts.append(f"⚠️ High position count: {len(positions)} open positions")
        
        return alerts
    
    def print_performance_report(self, metrics: Dict[str, Any], 
                               positions: List[Dict[str, Any]],
                               alerts: List[str]):
        """Print comprehensive performance report."""
        print("\n" + "="*60)
        print("🎯 REVOLUTIONARY TRADING SYSTEM - LIVE PERFORMANCE REPORT")
        print("="*60)
        
        # Account Status
        account_info = self.get_account_info()
        print(f"\n💰 ACCOUNT STATUS:")
        print(f"   Balance: ${account_info['balance']:,.2f}")
        print(f"   Equity: ${account_info['equity']:,.2f}")
        print(f"   P&L: ${account_info['profit']:+,.2f}")
        print(f"   Free Margin: ${account_info['free_margin']:,.2f}")
        
        # Performance Metrics
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   Total Trades: {metrics['total_trades']}")
        print(f"   Win Rate: {metrics['win_rate']:.1%} {'✅' if metrics['win_rate'] >= 0.58 else '❌'}")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f} {'✅' if metrics['profit_factor'] >= 1.3 else '❌'}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.1%} {'✅' if metrics['max_drawdown'] <= 0.15 else '❌'}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f} {'✅' if metrics['sharpe_ratio'] >= 1.0 else '❌'}")
        print(f"   Trades/Week: {metrics['trades_per_week']:.1f} {'✅' if 25 <= metrics['trades_per_week'] <= 50 else '❌'}")
        
        # Trade Analysis
        if metrics['total_trades'] > 0:
            print(f"\n📈 TRADE ANALYSIS:")
            print(f"   Total Profit: ${metrics['total_profit']:,.2f}")
            print(f"   Total Loss: ${metrics['total_loss']:,.2f}")
            print(f"   Avg Win: ${metrics['avg_win']:,.2f}")
            print(f"   Avg Loss: ${metrics['avg_loss']:,.2f}")
            print(f"   Avg Trade Duration: {metrics['avg_trade_duration']:.1f} hours")
        
        # Current Positions
        print(f"\n📋 CURRENT POSITIONS: {len(positions)}")
        if positions:
            total_position_pnl = sum(p['profit'] for p in positions)
            print(f"   Total Position P&L: ${total_position_pnl:+,.2f}")
            
            for pos in positions:
                print(f"   {pos['symbol']} {pos['type'].upper()} {pos['volume']:.2f} @ {pos['price_current']:.5f} "
                      f"P&L: ${pos['profit']:+,.2f}")
        
        # Alerts
        if alerts:
            print(f"\n🚨 ALERTS:")
            for alert in alerts:
                print(f"   {alert}")
        else:
            print(f"\n✅ No alerts - System performing well!")
        
        # Performance Assessment
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        thresholds_met = sum(1 for v in self.check_performance_thresholds(metrics).values() if v)
        total_thresholds = 5
        
        if thresholds_met == total_thresholds:
            print("   🎉 EXCELLENT: All performance targets met!")
        elif thresholds_met >= 4:
            print("   ✅ GOOD: Most performance targets met")
        elif thresholds_met >= 3:
            print("   ⚠️ FAIR: Some performance targets met")
        else:
            print("   ❌ POOR: Most performance targets not met")
        
        print(f"   Targets Met: {thresholds_met}/{total_thresholds}")
        
        print("\n" + "="*60)
    
    def export_performance_data(self, metrics: Dict[str, Any], 
                              positions: List[Dict[str, Any]]):
        """Export performance data to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_report_{timestamp}.json"
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'positions': positions,
            'account_info': self.get_account_info(),
            'alerts': self.alerts
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"📁 Performance data exported to: {filename}")
    
    def run_monitoring(self, duration_hours: int = 24):
        """Run continuous monitoring for specified duration."""
        print("🚀 Starting Live Trading Performance Monitor...")
        print(f"⏱️ Monitoring duration: {duration_hours} hours")
        print("Press Ctrl+C to stop early")
        
        end_time = datetime.now() + timedelta(hours=duration_hours)
        last_export = datetime.now()
        
        try:
            while datetime.now() < end_time:
                # Get current data
                account_info = self.get_account_info()
                positions = self.get_positions()
                trades = self.get_trade_history(days=7)
                
                # Calculate metrics
                metrics = self.calculate_performance_metrics(trades, account_info)
                
                # Generate alerts
                alerts = self.generate_alerts(metrics, positions)
                self.alerts.extend(alerts)
                
                # Print report
                self.print_performance_report(metrics, positions, alerts)
                
                # Export data periodically
                if (datetime.now() - last_export).total_seconds() >= self.config['export_interval']:
                    self.export_performance_data(metrics, positions)
                    last_export = datetime.now()
                
                # Wait for next check
                print(f"\n⏳ Next check in {self.config['monitoring_interval']} seconds...")
                time.sleep(self.config['monitoring_interval'])
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        
        print("✅ Monitoring completed")


def main():
    """Main function to run the monitoring system."""
    print("🎯 Revolutionary Trading System - Performance Monitor")
    print("="*60)
    
    # Initialize monitor
    monitor = LiveTradingMonitor()
    
    # Connect to MT5
    if monitor.connect_to_mt5():
        print("✅ Connected to MetaTrader 5")
    else:
        print("⚠️ Running in simulation mode")
    
    # Run monitoring
    monitor.run_monitoring(duration_hours=24)


if __name__ == "__main__":
    main() 