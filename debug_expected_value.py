#!/usr/bin/env python3
"""
Debug Expected Value Calculation
Identify why expected value is showing 0.0 in live trading
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_test_data():
    """Create realistic EURUSD test data similar to live trading conditions."""
    print("Creating test data...")
    
    # Create 1000 bars of realistic EURUSD data
    dates = pd.date_range(start='2025-07-30 06:00:00', periods=1000, freq='5T')
    np.random.seed(42)
    
    # Realistic EURUSD price movements
    base_price = 1.15500
    prices = []
    for i in range(len(dates)):
        if i == 0:
            price = base_price
        else:
            # Small random walk with some trend
            change = np.random.normal(0, 0.0001)
            if i > 500:  # Add some trend after 500 bars
                change += 0.00005
            price = prices[-1] + change
        prices.append(price)
    
    df = pd.DataFrame({
        'open': [p - np.random.uniform(0, 0.00005) for p in prices],
        'high': [p + np.random.uniform(0, 0.0001) for p in prices],
        'low': [p - np.random.uniform(0, 0.0001) for p in prices],
        'close': prices,
        'volume': np.random.randint(50, 200, len(dates))
    }, index=dates)
    
    # Ensure OHLC relationship
    df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
    df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
    
    print(f"✅ Created {len(df)} bars of test data")
    print(f"   Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")
    return df

def test_probabilistic_labeling(df):
    """Test the probabilistic labeling system."""
    print("\n🧪 Testing Probabilistic Labeling System...")
    
    try:
        from probabilistic_labeling import ProbabilisticLabelingSystem
        
        # Initialize the system
        labeling_system = ProbabilisticLabelingSystem()
        
        # Create probabilistic labels
        print("Creating probabilistic labels...")
        labels = labeling_system.create_probabilistic_labels(df)
        
        print(f"✅ Labels created successfully")
        print(f"   Total labels: {len(labels)}")
        print(f"   Positive signals: {(labels['is_signal'] == 1).sum()}")
        print(f"   Signal rate: {(labels['is_signal'] == 1).sum() / len(labels):.2%}")
        
        # Check expected value distribution
        if 'expected_value' in labels.columns:
            ev_stats = labels['expected_value'].describe()
            print(f"\n📊 Expected Value Statistics:")
            print(f"   Mean: {ev_stats['mean']:.6f}")
            print(f"   Min: {ev_stats['min']:.6f}")
            print(f"   Max: {ev_stats['max']:.6f}")
            print(f"   Std: {ev_stats['std']:.6f}")
            
            # Check for zero expected values
            zero_ev_count = (labels['expected_value'] == 0).sum()
            print(f"   Zero EV count: {zero_ev_count} ({zero_ev_count/len(labels):.2%})")
            
            # Show sample of positive signals
            positive_signals = labels[labels['is_signal'] == 1]
            if len(positive_signals) > 0:
                print(f"\n📈 Sample Positive Signals:")
                for i, (idx, row) in enumerate(positive_signals.head(3).iterrows()):
                    print(f"   Signal {i+1}: EV={row['expected_value']:.6f}, Conf={row.get('confidence', 'N/A'):.3f}")
        else:
            print("❌ Expected value column not found in labels")
            
        return labels
        
    except Exception as e:
        print(f"❌ Error in probabilistic labeling: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_live_trading_system(df):
    """Test the live trading system's expected value calculation."""
    print("\n🧪 Testing Live Trading System...")
    
    try:
        from phase3_live_trading_preparation import LiveTradingSystem
        
        # Initialize the system
        print("Initializing live trading system...")
        live_system = LiveTradingSystem()
        success = live_system.initialize_system(df)
        
        if not success:
            print("❌ Failed to initialize live trading system")
            return None
        
        print("✅ Live trading system initialized")
        
        # Test market data processing
        print("Testing market data processing...")
        trade_decision = live_system.process_market_data(df)
        
        if trade_decision:
            print(f"✅ Trade decision generated:")
            print(f"   Action: {trade_decision.get('action', 'N/A')}")
            print(f"   Expected Value: {trade_decision.get('expected_value', 'N/A')}")
            print(f"   Confidence: {trade_decision.get('confidence', 'N/A')}")
            print(f"   Reason: {trade_decision.get('reason', 'N/A')}")
        else:
            print("❌ No trade decision generated")
            
        return trade_decision
        
    except Exception as e:
        print(f"❌ Error in live trading system: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_signal_generation(df):
    """Test signal generation and expected value calculation."""
    print("\n🧪 Testing Signal Generation...")
    
    try:
        # Test the ensemble prediction
        from phase2_ensemble_architecture import AdvancedEnsembleSystem
        from enhanced_features import EnhancedFeatureEngineering
        
        # Create features
        print("Creating enhanced features...")
        feature_engineer = EnhancedFeatureEngineering()
        features = feature_engineer.create_enhanced_features(df)
        print(f"✅ Created {features.shape[1]} features")
        
        # Initialize ensemble (mock for testing)
        print("Initializing ensemble system...")
        ensemble = AdvancedEnsembleSystem()
        
        # Test prediction
        print("Testing ensemble prediction...")
        predictions = ensemble.predict_ensemble(features.iloc[-10:])  # Last 10 bars
        
        if predictions:
            print(f"✅ Ensemble predictions generated:")
            print(f"   Final prediction: {predictions.get('final_prediction', 'N/A')}")
            print(f"   Ensemble confidence: {predictions.get('ensemble_confidence', 'N/A')}")
            print(f"   Expected value: {predictions.get('expected_value', 'N/A')}")
        else:
            print("❌ No ensemble predictions generated")
            
        return predictions
        
    except Exception as e:
        print(f"❌ Error in signal generation: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_live_trading_issues():
    """Analyze the specific issues from the live trading report."""
    print("\n🔍 ANALYZING LIVE TRADING ISSUES")
    print("=" * 50)
    
    # Issues from the report
    issues = {
        "expected_value_zero": "All trades show EV:0. - Expected value calculation failing",
        "low_confidence": "Confidence levels 50-65% - May be too low for quality signals",
        "all_buy_signals": "All signals are buy signals - No short signals generated",
        "stop_loss_hits": "Multiple stop loss hits - Risk management working but win rate poor"
    }
    
    print("📋 Issues Identified in Live Trading Report:")
    for i, (issue, description) in enumerate(issues.items(), 1):
        print(f"   {i}. {issue}: {description}")
    
    print("\n🎯 Debugging Strategy:")
    print("   1. Test expected value calculation in isolation")
    print("   2. Verify probabilistic labeling system")
    print("   3. Check ensemble prediction pipeline")
    print("   4. Analyze signal generation criteria")
    print("   5. Review confidence thresholds")

def main():
    """Main debugging function."""
    print("🔧 DEBUGGING EXPECTED VALUE CALCULATION")
    print("=" * 60)
    
    # Analyze issues
    analyze_live_trading_issues()
    
    # Create test data
    df = create_test_data()
    
    # Test probabilistic labeling
    labels = test_probabilistic_labeling(df)
    
    # Test live trading system
    trade_decision = test_live_trading_system(df)
    
    # Test signal generation
    predictions = test_signal_generation(df)
    
    # Summary
    print("\n📊 DEBUG SUMMARY")
    print("=" * 60)
    
    if labels is not None:
        print("✅ Probabilistic labeling system: WORKING")
    else:
        print("❌ Probabilistic labeling system: FAILED")
    
    if trade_decision is not None:
        print("✅ Live trading system: WORKING")
    else:
        print("❌ Live trading system: FAILED")
    
    if predictions is not None:
        print("✅ Signal generation: WORKING")
    else:
        print("❌ Signal generation: FAILED")
    
    print("\n🎯 Next Steps:")
    print("   1. Review the debug output above")
    print("   2. Identify which component is failing")
    print("   3. Fix the expected value calculation")
    print("   4. Test with the fixed system")
    print("   5. Resume live trading only after fixes")

if __name__ == "__main__":
    main() 