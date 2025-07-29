"""
Quick test of live trading functionality
"""
import pandas as pd
import numpy as np
from phase3_live_trading_preparation import LiveTradingSystem
import warnings
warnings.filterwarnings('ignore')

def test_live_trading():
    """Test live trading functionality."""
    
    print("🧪 Testing Live Trading Functionality...")
    
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
    
    # Ensure OHLC relationship
    training_data['high'] = np.maximum(training_data['high'], np.maximum(training_data['open'], training_data['close']))
    training_data['low'] = np.minimum(training_data['low'], np.minimum(training_data['open'], training_data['close']))
    
    # Initialize system
    print("Initializing live trading system...")
    live_system = LiveTradingSystem()
    success = live_system.initialize_system(training_data)
    
    if not success:
        print("❌ Failed to initialize system")
        return False
    
    # Start trading
    live_system.start_trading()
    
    # Test processing market data
    print("Testing market data processing...")
    
    # Create test market data
    test_data = pd.DataFrame({
        'open': [1.1050, 1.1052, 1.1051],
        'close': [1.1052, 1.1051, 1.1053],
        'high': [1.1055, 1.1056, 1.1057],
        'low': [1.1048, 1.1049, 1.1050],
        'volume': [500, 600, 700]
    }, index=pd.date_range('2024-01-20', periods=3, freq='5T'))
    
    # Process market data
    trade_decision = live_system.process_market_data(test_data)
    
    print(f"Trade decision: {trade_decision}")
    
    # Test system status
    status = live_system.get_system_status()
    print(f"System status: {status['system_running']}")
    print(f"MT5 connected: {status['mt5_connected']}")
    
    # Stop trading
    live_system.stop_trading()
    
    print("✅ Live trading functionality test completed!")
    return True

if __name__ == "__main__":
    test_live_trading() 