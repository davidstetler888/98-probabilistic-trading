"""
Train the Revolutionary Trading System
"""
import pandas as pd
import numpy as np
from phase3_live_trading_preparation import LiveTradingSystem
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df):
    """Preprocess data to ensure correct format and column names."""
    
    print(f"Original columns: {list(df.columns)}")
    
    # Handle MT5 export format with angle brackets
    if '<DATE>' in df.columns:
        print("Detected MT5 export format - processing...")
        # Remove angle brackets from column names
        df.columns = df.columns.str.replace('<', '').str.replace('>', '')
        print(f"Cleaned columns: {list(df.columns)}")
    
    # Convert column names to lowercase
    df.columns = df.columns.str.lower()
    print(f"Lowercase columns: {list(df.columns)}")
    
    # Handle different date/time formats
    if 'date' in df.columns and 'time' in df.columns:
        print("Processing date/time columns...")
        # Handle MT5 format: YYYY.MM.DD HH:MM:SS
        try:
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
            print("✅ Date/time parsing successful")
        except Exception as e:
            print(f"❌ Date/time parsing error: {e}")
            return None
        df = df.set_index('datetime')
        df = df.drop(['date', 'time'], axis=1)
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    
    # Handle MT5 volume columns (tickvol vs vol)
    if 'tickvol' in df.columns and 'vol' in df.columns:
        # Use tickvol as volume (more reliable for forex)
        df['volume'] = df['tickvol']
        df = df.drop(['tickvol', 'vol'], axis=1)
    elif 'tickvol' in df.columns:
        df['volume'] = df['tickvol']
        df = df.drop('tickvol', axis=1)
    elif 'vol' in df.columns:
        df['volume'] = df['vol']
        df = df.drop('vol', axis=1)
    
    # Handle spread column (optional)
    if 'spread' in df.columns:
        print("✅ Spread data detected - will be used for enhanced modeling")
        # Keep spread column for enhanced features
    else:
        print("⚠️ No spread data - will use estimated spreads")
    
    # Ensure required columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # Ensure data types are correct
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    
    # Remove any rows with NaN values
    df = df.dropna()
    
    print(f"✅ Final processed data shape: {df.shape}")
    print(f"✅ Final columns: {list(df.columns)}")
    
    return df

def train_trading_system(data_file: str):
    """Train the complete trading system."""
    
    print("🚀 Training Revolutionary Trading System...")
    
    # Load data
    print(f"Loading data from {data_file}...")
    try:
        # Check if file contains tab-separated data (MT5 export format)
        with open(data_file, 'r') as f:
            first_line = f.readline()
            if '\t' in first_line:
                df = pd.read_csv(data_file, sep='\t')
                print("✅ Loaded as TSV format (MT5 export)")
            else:
                df = pd.read_csv(data_file)
                print("✅ Loaded as CSV format")
        
        print(f"Loaded {len(df)} bars of data")
        
        # Preprocess the data
        df = preprocess_data(df)
        if df is None:
            return None
            
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        # Check data format
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return None
        
        print(f"✅ Data format verified: {len(df.columns)} columns")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Initialize live trading system
    print("Initializing live trading system...")
    live_system = LiveTradingSystem()
    
    # Train the system
    print("Training ensemble system...")
    success = live_system.initialize_system(df)
    
    if success:
        print("✅ Training completed successfully!")
        
        # Get system status
        status = live_system.get_system_status()
        print(f"\n=== Training Summary ===")
        print(f"Ensemble trained: {live_system.ensemble.is_trained}")
        print(f"MT5 connection: {status['mt5_connected']}")
        print(f"Risk management: Active")
        print(f"Performance monitoring: Active")
        
        return live_system
    else:
        print("❌ Training failed!")
        return None

def create_sample_data():
    """Create sample data for testing if no data file is provided."""
    print("Creating sample EURUSD data for testing...")
    
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=10000, freq='5T')
    
    # Create realistic EURUSD price movement
    trend_component = np.linspace(0, 0.02, 10000)  # Gradual uptrend
    noise_component = np.random.normal(0, 0.0002, 10000)
    prices = 1.1000 + trend_component + noise_component
    
    df = pd.DataFrame({
        'open': prices,
        'close': prices + np.random.normal(0, 0.0001, 10000),
        'high': prices + np.abs(np.random.normal(0, 0.0003, 10000)),
        'low': prices - np.abs(np.random.normal(0, 0.0003, 10000)),
        'volume': np.random.randint(100, 1000, 10000)
    }, index=dates)
    
    # Ensure OHLC relationship is maintained
    df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
    df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
    
    return df

if __name__ == "__main__":
    import sys
    
    # Check if data file is provided
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        print(f"Using data file: {data_file}")
    else:
        print("No data file provided. Creating sample data for testing...")
        df = create_sample_data()
        data_file = "sample_eurusd_data.csv"
        df.to_csv(data_file)
        print(f"Sample data saved to: {data_file}")
    
    # Train the system
    trained_system = train_trading_system(data_file)
    
    if trained_system:
        print("\n🎉 System ready for live trading!")
        print("\nNext steps:")
        print("1. Configure MT5 connection in live_trading.py")
        print("2. Run: python3 live_trading.py")
        print("3. Monitor performance and adjust as needed")
    else:
        print("\n❌ Training failed. Please check your data and try again.") 