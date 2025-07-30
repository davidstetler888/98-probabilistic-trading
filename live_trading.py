"""
Live Trading Script for Revolutionary Trading System
Updated to use simple MT5 connection without explicit login credentials
ENFORCES SINGLE TRADE CONSTRAINT with cooldown period
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from phase3_live_trading_preparation import LiveTradingSystem
import warnings
warnings.filterwarnings('ignore')

# Try to import MT5, but don't fail if not available
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    print("⚠️ MetaTrader5 not installed. Install with: pip install MetaTrader5")
    MT5_AVAILABLE = False

class LiveTrader:
    def __init__(self, trained_system: LiveTradingSystem):
        self.system = trained_system
        self.is_running = False
        self.last_bar_time = None
        self.trade_count = 0
        self.profit_loss = 0.0
        
        # SINGLE TRADE CONSTRAINT ENFORCEMENT
        self.last_trade_time = None
        self.cooldown_minutes = 5  # 5-minute cooldown between trades
        self.max_positions = 1     # Only one trade open at a time
        self.open_positions = []   # Track open positions
        
    def connect_mt5(self):
        """Connect to MetaTrader 5 using simple initialization."""
        if not MT5_AVAILABLE:
            print("❌ MetaTrader5 not available")
            return False
            
        print("🔧 Connecting to MetaTrader 5...")
        
        # Simple initialization - uses existing MT5 terminal session
        if not mt5.initialize():
            print(f"❌ Failed to initialize MT5: {mt5.last_error()}")
            print("   Make sure MT5 terminal is running and logged in")
            return False
        
        # Check terminal info
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            print("❌ Terminal info not available")
            return False
            
        print(f"✅ Connected to MT5 Terminal: {terminal_info.name}")
        print(f"✅ Connected: {terminal_info.connected}")
        
        # Check account info
        account_info = mt5.account_info()
        if account_info is None:
            print("❌ Account info not available")
            print("   Make sure you're logged into MT5 terminal")
            return False
            
        print(f"✅ Account: {account_info.login}")
        print(f"✅ Server: {account_info.server}")
        print(f"✅ Balance: ${account_info.balance:.2f}")
        print(f"✅ Equity: ${account_info.equity:.2f}")
        
        # Check symbol availability
        symbol_info = mt5.symbol_info("EURUSD.PRO")
        if symbol_info is None:
            print("❌ EURUSD.PRO symbol not found")
            print("   Trying to select symbol...")
            if mt5.symbol_select("EURUSD.PRO", True):
                symbol_info = mt5.symbol_info("EURUSD.PRO")
                if symbol_info is None:
                    print("❌ Still cannot find EURUSD.PRO symbol")
                    return False
            else:
                print("❌ Failed to select EURUSD.PRO symbol")
                return False
        
        print(f"✅ Symbol: {symbol_info.name}")
        print(f"✅ Spread: {symbol_info.spread} points")
        print(f"✅ Trade mode: {symbol_info.trade_mode}")
        
        return True
    
    def get_latest_data(self, symbol="EURUSD.PRO", timeframe=None, bars=100):
        if MT5_AVAILABLE and timeframe is None:
            timeframe = mt5.TIMEFRAME_M5
        """Get latest market data."""
        if not MT5_AVAILABLE:
            # Create mock data for testing
            return self._create_mock_data()
            
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df
    
    def _create_mock_data(self):
        """Create mock data for testing when MT5 is not available."""
        # Create realistic mock data
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=8)
        
        timestamps = pd.date_range(start=start_time, end=end_time, freq='5T')
        np.random.seed(42)  # For reproducible results
        
        # Create realistic EURUSD price movements
        base_price = 1.09350
        prices = []
        for i in range(len(timestamps)):
            if i == 0:
                price = base_price
            else:
                # Small random walk
                change = np.random.normal(0, 0.0001)
                price = prices[-1] + change
            prices.append(price)
        
        df = pd.DataFrame({
            'open': [p - np.random.uniform(0, 0.00005) for p in prices],
            'high': [p + np.random.uniform(0, 0.0001) for p in prices],
            'low': [p - np.random.uniform(0, 0.0001) for p in prices],
            'close': prices,
            'tick_volume': np.random.randint(50, 200, len(timestamps)),
            'spread': np.random.uniform(0.0001, 0.0003, len(timestamps)),
            'real_volume': np.random.randint(50, 200, len(timestamps))
        }, index=timestamps)
        
        return df
    
    def process_market_data(self, df: pd.DataFrame):
        """Process market data and get trading decision."""
        return self.system.process_market_data(df)
    
    def check_single_trade_constraint(self):
        """Check if we can place a new trade based on single trade constraint."""
        current_time = datetime.now()
        
        # Check if we have open positions
        if MT5_AVAILABLE:
            positions = mt5.positions_get(symbol="EURUSD.PRO")
            if positions is None:
                positions = []
            open_positions = len(positions)
        else:
            # Mock position tracking
            open_positions = len(self.open_positions)
        
        # SINGLE TRADE CONSTRAINT: Only allow one position at a time
        if open_positions >= self.max_positions:
            return False, f"Single trade constraint: {open_positions} positions open (max: {self.max_positions})"
        
        # Check cooldown period
        if self.last_trade_time is not None:
            time_since_last_trade = current_time - self.last_trade_time
            cooldown_delta = timedelta(minutes=self.cooldown_minutes)
            
            if time_since_last_trade < cooldown_delta:
                remaining_cooldown = cooldown_delta - time_since_last_trade
                return False, f"Cooldown period: {remaining_cooldown.seconds//60}m {remaining_cooldown.seconds%60}s remaining"
        
        return True, "Trade allowed"
    
    def execute_trade(self, trade_decision):
        """Execute trade decision with single trade constraint enforcement."""
        if trade_decision['action'] == 'no_action':
            return True
        
        # CHECK SINGLE TRADE CONSTRAINT
        can_trade, reason = self.check_single_trade_constraint()
        if not can_trade:
            print(f"🚫 Trade blocked: {reason}")
            return False
        
        if not MT5_AVAILABLE:
            # Mock trade execution for testing
            print(f"🔸 MOCK TRADE: {trade_decision['action']} {trade_decision['volume']} {trade_decision['symbol']}")
            print(f"   Price: {trade_decision['price']:.5f}")
            print(f"   SL: {trade_decision['stop_loss']:.5f}, TP: {trade_decision['take_profit']:.5f}")
            print(f"   Confidence: {trade_decision['confidence']:.3f}")
            
            # Track mock position
            mock_position = {
                'ticket': self.trade_count + 1,
                'symbol': trade_decision['symbol'],
                'type': trade_decision['action'],
                'volume': trade_decision['volume'],
                'price': trade_decision['price'],
                'sl': trade_decision['stop_loss'],
                'tp': trade_decision['take_profit'],
                'time': datetime.now()
            }
            self.open_positions.append(mock_position)
            
            self.trade_count += 1
            self.last_trade_time = datetime.now()
            return True
        
        # Place order via MT5
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": trade_decision['symbol'],
            "volume": trade_decision['volume'],
            "type": mt5.ORDER_TYPE_BUY if trade_decision['action'] == 'buy' else mt5.ORDER_TYPE_SELL,
            "price": trade_decision['price'],
            "sl": trade_decision['stop_loss'],
            "tp": trade_decision['take_profit'],
            "comment": f"Conf:{trade_decision['confidence']:.3f},EV:{trade_decision['expected_value']:.6f}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ Trade executed: {trade_decision['action']} {trade_decision['volume']} {trade_decision['symbol']}")
            self.trade_count += 1
            self.last_trade_time = datetime.now()
            return True
        else:
            print(f"❌ Trade failed: {result.comment}")
            return False
    
    def update_position_tracking(self):
        """Update position tracking for single trade constraint."""
        if MT5_AVAILABLE:
            # Get current positions from MT5
            positions = mt5.positions_get(symbol="EURUSD.PRO")
            if positions is None:
                positions = []
            
            # Update open positions count
            current_open_positions = len(positions)
            
            # If positions were closed, reset cooldown
            if current_open_positions == 0 and len(self.open_positions) > 0:
                print("📊 All positions closed - cooldown period active")
                self.open_positions = []
        else:
            # Mock position tracking - simulate position closure after some time
            current_time = datetime.now()
            self.open_positions = [
                pos for pos in self.open_positions 
                if (current_time - pos['time']).total_seconds() < 3600  # Close after 1 hour for testing
            ]
    
    def run_live_trading(self, symbol="EURUSD.PRO", check_interval=60):
        """Run live trading loop with single trade constraint enforcement."""
        print(f"🚀 Starting live trading for {symbol}...")
        print(f"🛡️ SINGLE TRADE CONSTRAINT: Max {self.max_positions} position, {self.cooldown_minutes}min cooldown")
        
        if MT5_AVAILABLE:
            if not self.connect_mt5():
                print("⚠️ Running in simulation mode (no MT5 connection)")
        else:
            print("⚠️ Running in simulation mode (MT5 not available)")
        
        self.system.start_trading()
        self.is_running = True
        
        print("✅ Live trading started. Press Ctrl+C to stop.")
        print(f"📊 Check interval: {check_interval} seconds")
        
        try:
            while self.is_running:
                # Update position tracking
                self.update_position_tracking()
                
                # Get latest data
                df = self.get_latest_data(symbol)
                if df is None:
                    print("❌ Failed to get market data")
                    time.sleep(check_interval)
                    continue
                
                # Check if we have new data
                latest_time = df.index[-1]
                if self.last_bar_time == latest_time:
                    time.sleep(check_interval)
                    continue
                
                self.last_bar_time = latest_time
                
                # Process data and get trading decision
                trade_decision = self.process_market_data(df)
                
                if trade_decision:
                    # Check single trade constraint before executing
                    can_trade, reason = self.check_single_trade_constraint()
                    
                    if can_trade:
                        print(f"📊 {latest_time}: {trade_decision['action']} - {trade_decision.get('reason', 'Signal generated')}")
                        
                        # Execute trade if signal
                        if trade_decision['action'] != 'no_action':
                            self.execute_trade(trade_decision)
                    else:
                        print(f"📊 {latest_time}: Signal generated but blocked - {reason}")
                
                # Update performance metrics
                self.system.update_performance()
                
                # Get system status
                status = self.system.get_system_status()
                if status['alerts']:
                    print(f"⚠️ Alerts: {status['alerts']}")
                
                # Print periodic status
                if self.trade_count % 10 == 0 and self.trade_count > 0:
                    print(f"📈 Total trades: {self.trade_count}")
                    if status['performance_metrics']:
                        metrics = status['performance_metrics']
                        print(f"   Win rate: {metrics.get('win_rate', 0):.2%}")
                        print(f"   Profit factor: {metrics.get('profit_factor', 0):.2f}")
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Live trading stopped by user")
        except Exception as e:
            print(f"❌ Error in live trading: {e}")
        finally:
            self.stop_live_trading()
    
    def stop_live_trading(self):
        """Stop live trading."""
        self.is_running = False
        if MT5_AVAILABLE:
            mt5.shutdown()
        print("✅ Live trading stopped")

def main():
    """Main function."""
    print("🚀 Revolutionary Trading System - Live Trading")
    print("=" * 50)
    
    # Load trained system
    print("Loading trained system...")
    try:
        # Create sample training data for demonstration
        print("Creating sample training data...")
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='5T')
        sample_data = pd.DataFrame({
            'open': np.random.uniform(1.09, 1.10, len(dates)),
            'high': np.random.uniform(1.09, 1.10, len(dates)),
            'low': np.random.uniform(1.09, 1.10, len(dates)),
            'close': np.random.uniform(1.09, 1.10, len(dates)),
            'volume': np.random.randint(50, 200, len(dates))
        }, index=dates)
        
        # Initialize live trading system
        print("Initializing and training system...")
        live_system = LiveTradingSystem()
        live_system.initialize_system(sample_data)
        
        # Create live trader
        trader = LiveTrader(live_system)
        
        # Configuration
        symbol = "EURUSD.PRO"
        check_interval = 60  # 60 seconds
        
        print(f"✅ System initialized successfully!")
        print(f"Configuration:")
        print(f"   Symbol: {symbol}")
        print(f"   Check interval: {check_interval} seconds")
        print(f"   MT5 available: {MT5_AVAILABLE}")
        
        # Start live trading
        trader.run_live_trading(symbol=symbol, check_interval=check_interval)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 