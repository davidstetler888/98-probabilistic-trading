#!/usr/bin/env python3
"""
Debug Live Trading Issues
Focus on identifying why expected value shows 0.0 and why all signals are buy signals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_realistic_test_data():
    """Create realistic test data that should generate signals."""
    print("Creating realistic test data...")
    
    # Create data with clear trends that should generate signals
    dates = pd.date_range(start='2025-07-30 06:00:00', periods=500, freq='5T')
    np.random.seed(42)
    
    # Create trending data that should generate buy signals
    base_price = 1.15500
    prices = []
    
    for i in range(len(dates)):
        if i == 0:
            price = base_price
        else:
            # Create a clear uptrend that should generate buy signals
            trend = 0.0001  # 1 pip per bar uptrend
            noise = np.random.normal(0, 0.00005)  # Small noise
            price = prices[-1] + trend + noise
        prices.append(price)
    
    df = pd.DataFrame({
        'open': [p - np.random.uniform(0, 0.00002) for p in prices],
        'high': [p + np.random.uniform(0, 0.00005) for p in prices],
        'low': [p - np.random.uniform(0, 0.00005) for p in prices],
        'close': prices,
        'volume': np.random.randint(50, 200, len(dates))
    }, index=dates)
    
    # Ensure OHLC relationship
    df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
    df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
    
    print(f"✅ Created {len(df)} bars of trending test data")
    print(f"   Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")
    print(f"   Total movement: {(df['close'].iloc[-1] - df['close'].iloc[0])*10000:.1f} pips")
    return df

def test_live_trading_with_signals():
    """Test the live trading system with data that should generate signals."""
    print("\n🧪 Testing Live Trading System with Signal Generation...")
    
    try:
        from phase3_live_trading_preparation import LiveTradingSystem
        
        # Create test data
        df = create_realistic_test_data()
        
        # Initialize the system
        print("Initializing live trading system...")
        live_system = LiveTradingSystem()
        success = live_system.initialize_system(df)
        
        if not success:
            print("❌ Failed to initialize live trading system")
            return None
        
        print("✅ Live trading system initialized")
        
        # Start trading to enable signal generation
        print("Starting trading system...")
        live_system.start_trading()
        
        # Test market data processing
        print("Testing market data processing...")
        trade_decision = live_system.process_market_data(df)
        
        if trade_decision:
            print(f"✅ Trade decision generated:")
            print(f"   Action: {trade_decision.get('action', 'N/A')}")
            print(f"   Expected Value: {trade_decision.get('expected_value', 'N/A')}")
            print(f"   Confidence: {trade_decision.get('confidence', 'N/A')}")
            print(f"   Reason: {trade_decision.get('reason', 'N/A')}")
            
            # Check if expected value is properly calculated
            if trade_decision.get('expected_value') == 0 or trade_decision.get('expected_value') == 'N/A':
                print("❌ Expected value is 0 or N/A - This is the issue!")
            else:
                print(f"✅ Expected value is properly calculated: {trade_decision.get('expected_value')}")
        else:
            print("❌ No trade decision generated")
            
        return trade_decision
        
    except Exception as e:
        print(f"❌ Error in live trading system: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_ensemble_prediction():
    """Test the ensemble prediction system directly."""
    print("\n🧪 Testing Ensemble Prediction System...")
    
    try:
        from phase2_ensemble_architecture import AdvancedEnsembleSystem
        from enhanced_features import EnhancedFeatureEngineering
        from probabilistic_labeling import ProbabilisticLabelingSystem
        
        # Create test data
        df = create_realistic_test_data()
        
        # Create features
        print("Creating enhanced features...")
        feature_engineer = EnhancedFeatureEngineering()
        features = feature_engineer.create_enhanced_features(df)
        print(f"✅ Created {features.shape[1]} features")
        
        # Create labels
        print("Creating probabilistic labels...")
        labeling_system = ProbabilisticLabelingSystem()
        labels = labeling_system.create_probabilistic_labels(df)
        print(f"✅ Created labels with {len(labels)} samples")
        
        # Initialize and train ensemble
        print("Initializing and training ensemble...")
        ensemble = AdvancedEnsembleSystem()
        ensemble.train_ensemble(df)  # This should train the ensemble
        print("✅ Ensemble trained")
        
        # Test prediction
        print("Testing ensemble prediction...")
        predictions = ensemble.predict_ensemble(features.iloc[-5:])  # Last 5 bars
        
        if predictions:
            print(f"✅ Ensemble predictions generated:")
            print(f"   Final prediction: {predictions.get('final_prediction', 'N/A')}")
            print(f"   Ensemble confidence: {predictions.get('ensemble_confidence', 'N/A')}")
            print(f"   Expected value: {predictions.get('expected_value', 'N/A')}")
            
            # Check expected value
            if predictions.get('expected_value') == 0 or predictions.get('expected_value') == 'N/A':
                print("❌ Expected value is 0 or N/A in ensemble predictions")
            else:
                print(f"✅ Expected value calculated: {predictions.get('expected_value')}")
        else:
            print("❌ No ensemble predictions generated")
            
        return predictions
        
    except Exception as e:
        print(f"❌ Error in ensemble prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_probabilistic_labeling_detailed():
    """Test probabilistic labeling with detailed analysis."""
    print("\n🧪 Testing Probabilistic Labeling in Detail...")
    
    try:
        from probabilistic_labeling import ProbabilisticLabelingSystem
        
        # Create test data
        df = create_realistic_test_data()
        
        # Initialize the system
        labeling_system = ProbabilisticLabelingSystem()
        
        # Create probabilistic labels
        print("Creating probabilistic labels...")
        labels = labeling_system.create_probabilistic_labels(df)
        
        print(f"✅ Labels created successfully")
        print(f"   Total labels: {len(labels)}")
        
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
            
            # Show sample of highest expected values
            top_ev = labels.nlargest(5, 'expected_value')
            print(f"\n📈 Top 5 Expected Values:")
            for i, (idx, row) in enumerate(top_ev.iterrows()):
                print(f"   {i+1}. EV={row['expected_value']:.6f}, Signal={row.get('is_signal', 'N/A')}")
            
            # Check signal distribution
            if 'is_signal' in labels.columns:
                signal_count = labels['is_signal'].value_counts()
                print(f"\n📊 Signal Distribution:")
                print(f"   No signal (0): {signal_count.get(0, 0)}")
                print(f"   Signal (1): {signal_count.get(1, 0)}")
                
                # Check direction if available
                if 'direction' in labels.columns:
                    direction_count = labels['direction'].value_counts()
                    print(f"\n📊 Direction Distribution:")
                    for direction, count in direction_count.items():
                        print(f"   {direction}: {count}")
        else:
            print("❌ Expected value column not found in labels")
            
        return labels
        
    except Exception as e:
        print(f"❌ Error in probabilistic labeling: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_live_trading_report_issues():
    """Analyze the specific issues from the live trading report."""
    print("\n🔍 ANALYZING LIVE TRADING REPORT ISSUES")
    print("=" * 60)
    
    print("📋 Issues from Trading Report (July 30, 2025):")
    print("   1. Expected Value: All trades show 'EV:0.' - Calculation failing")
    print("   2. Confidence: 50-65% range - May be too low for quality")
    print("   3. Direction: All signals are buy signals - No short signals")
    print("   4. Performance: 10% win rate, 0.02 profit factor - Catastrophic")
    print("   5. Stop Loss: Multiple -$20.00 hits - Risk management working")
    
    print("\n🎯 Root Cause Analysis:")
    print("   • Expected value showing 0.0 suggests calculation error")
    print("   • All buy signals suggest direction prediction issue")
    print("   • Low confidence suggests signal quality problem")
    print("   • Poor performance suggests fundamental signal quality issue")

def main():
    """Main debugging function."""
    print("🔧 DEBUGGING LIVE TRADING ISSUES")
    print("=" * 60)
    
    # Analyze issues
    analyze_live_trading_report_issues()
    
    # Test probabilistic labeling in detail
    labels = test_probabilistic_labeling_detailed()
    
    # Test ensemble prediction
    predictions = test_ensemble_prediction()
    
    # Test live trading system
    trade_decision = test_live_trading_with_signals()
    
    # Summary and recommendations
    print("\n📊 DEBUG SUMMARY")
    print("=" * 60)
    
    print("🔍 Key Findings:")
    if labels is not None:
        print("✅ Probabilistic labeling system: WORKING")
        if 'expected_value' in labels.columns:
            zero_ev_count = (labels['expected_value'] == 0).sum()
            if zero_ev_count > 0:
                print(f"⚠️  {zero_ev_count} labels have zero expected value")
    else:
        print("❌ Probabilistic labeling system: FAILED")
    
    if predictions is not None:
        print("✅ Ensemble prediction system: WORKING")
        if predictions.get('expected_value') == 0:
            print("❌ Expected value is 0 in ensemble predictions")
    else:
        print("❌ Ensemble prediction system: FAILED")
    
    if trade_decision is not None:
        print("✅ Live trading system: WORKING")
        if trade_decision.get('expected_value') == 0 or trade_decision.get('expected_value') == 'N/A':
            print("❌ Expected value is 0 or N/A in live trading")
    else:
        print("❌ Live trading system: FAILED")
    
    print("\n🎯 RECOMMENDATIONS:")
    print("   1. Check expected value calculation in probabilistic labeling")
    print("   2. Verify ensemble prediction pipeline")
    print("   3. Review signal generation criteria")
    print("   4. Test with fixed system before resuming live trading")
    print("   5. Consider adjusting confidence thresholds")

if __name__ == "__main__":
    main() 