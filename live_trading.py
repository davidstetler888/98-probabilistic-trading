"""
Live Trading Script for Revolutionary Trading System
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
        
    def connect_mt5(self):
        """Connect to MetaTrader 5."""
        if not MT5_AVAILABLE:
            print("❌ MetaTrader5 not available")
            return False
            
        if not mt5.initialize():
            print("❌ Failed to initialize MT5")
            return False
        
        # Login to your account (update with your credentials)
        # Uncomment and update these lines with your MT5 credentials:
        # authorized = mt5.login(login=YOUR_LOGIN, password=YOUR_PASSWORD, server=YOUR_SERVER)
        # if not authorized:
        #     print("❌ Failed to login to MT5")
        #     return False
        
        print("✅ Connected to MetaTrader 5")
        print(f"Terminal info: {mt5.terminal_info()}")
        return True
    
    def get_latest_data(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5, bars=100):
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
        """Create mock data for testing without MT5."""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='5T')
        prices = 1.1000 + np.cumsum(np.random.normal(0, 0.0001, 100))
        
        df = pd.DataFrame({
            'open': prices,
            'close': prices + np.random.normal(0, 0.0001, 100),
            'high': prices + np.abs(np.random.normal(0, 0.0003, 100)),
            'low': prices - np.abs(np.random.normal(0, 0.0003, 100)),
            'volume': np.random.randint(100, 1000, 100)
        }, index=dates)
        
        # Ensure OHLC relationship
        df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
        df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
        
        return df
    
    def process_market_data(self, df: pd.DataFrame):
        """Process market data and generate trading signals."""
        if len(df) < 50:  # Need minimum data for features
            return None
        
        # Process with trading system
        trade_decision = self.system.process_market_data(df)
        
        return trade_decision
    
    def execute_trade(self, trade_decision):
        """Execute trade decision."""
        if trade_decision['action'] == 'no_action':
            return True
        
        if not MT5_AVAILABLE:
            # Mock trade execution for testing
            print(f"🔸 MOCK TRADE: {trade_decision['action']} {trade_decision['volume']} {trade_decision['symbol']}")
            print(f"   Price: {trade_decision['price']:.5f}")
            print(f"   SL: {trade_decision['stop_loss']:.5f}, TP: {trade_decision['take_profit']:.5f}")
            print(f"   Confidence: {trade_decision['confidence']:.3f}")
            self.trade_count += 1
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
            return True
        else:
            print(f"❌ Trade failed: {result.comment}")
            return False
    
    def run_live_trading(self, symbol="EURUSD", check_interval=60):
        """Run live trading loop."""
        print(f"🚀 Starting live trading for {symbol}...")
        
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
                    print(f"📊 {latest_time}: {trade_decision['action']} - {trade_decision.get('reason', 'Signal generated')}")
                    
                    # Execute trade if signal
                    if trade_decision['action'] != 'no_action':
                        self.execute_trade(trade_decision)
                
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
            print("\n🛑 Stopping live trading...")
            self.stop_live_trading()
    
    def stop_live_trading(self):
        """Stop live trading."""
        self.is_running = False
        self.system.stop_trading()
        if MT5_AVAILABLE:
            mt5.shutdown()
        print("✅ Live trading stopped")
        print(f"📊 Final stats: {self.trade_count} trades executed")

def main():
    """Main function to run live trading."""
    
    print("🚀 Revolutionary Trading System - Live Trading")
    print("=" * 50)
    
    # Load your trained system
    print("Loading trained system...")
    
    # For demonstration, we'll create a new system and train it
    # In production, you'd load a saved trained system
    from phase3_live_trading_preparation import LiveTradingSystem
    
    # Create sample data for demonstration
    # Replace this with your actual training data
    print("Creating sample training data...")
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=5000, freq='5T')
    prices = 1.1000 + np.cumsum(np.random.normal(0, 0.0001, 5000))
    
    training_data = pd.DataFrame({
        'open': prices,
        'close': prices + np.random.normal(0, 0.0001, 5000),
        'high': prices + np.abs(np.random.normal(0, 0.0003, 5000)),
        'low': prices - np.abs(np.random.normal(0, 0.0003, 5000)),
        'volume': np.random.randint(100, 1000, 5000)
    }, index=dates)
    
    # Ensure OHLC relationship
    training_data['high'] = np.maximum(training_data['high'], np.maximum(training_data['open'], training_data['close']))
    training_data['low'] = np.minimum(training_data['low'], np.minimum(training_data['open'], training_data['close']))
    
    # Initialize and train system
    print("Initializing and training system...")
    live_system = LiveTradingSystem()
    success = live_system.initialize_system(training_data)
    
    if not success:
        print("❌ Failed to initialize system")
        return
    
    print("✅ System initialized successfully!")
    
    # Create live trader
    trader = LiveTrader(live_system)
    
    # Configuration
    symbol = "EURUSD"
    check_interval = 60  # Check every 60 seconds
    
    print(f"\nConfiguration:")
    print(f"Symbol: {symbol}")
    print(f"Check interval: {check_interval} seconds")
    print(f"MT5 available: {MT5_AVAILABLE}")
    
    if not MT5_AVAILABLE:
        print("\n⚠️ Running in simulation mode")
        print("To enable real trading:")
        print("1. Install MT5: pip install MetaTrader5")
        print("2. Update login credentials in the script")
        print("3. Restart the system")
    
    # Run live trading
    print(f"\nStarting live trading...")
    trader.run_live_trading(symbol=symbol, check_interval=check_interval)

if __name__ == "__main__":
    main() 