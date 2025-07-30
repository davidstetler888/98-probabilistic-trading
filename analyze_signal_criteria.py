#!/usr/bin/env python3
"""
Analyze Signal Criteria
Analyzes why signals are being rejected and suggests adjustments.
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

def analyze_signal_criteria():
    """Analyze why signals are being rejected."""
    print("🔍 ANALYZING SIGNAL CRITERIA")
    print("="*50)
    
    # Get recent market data
    if not MT5_AVAILABLE or not mt5.initialize():
        print("❌ MT5 not available")
        return
    
    print("📊 Getting recent market data...")
    rates = mt5.copy_rates_from_pos("EURUSD.PRO", mt5.TIMEFRAME_M5, 0, 1000)
    if rates is None:
        print("❌ Cannot get market data")
        return
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    print(f"✅ Got {len(df)} bars of market data")
    print(f"✅ Time range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    
    # Test signal generation with different criteria
    print("\n🎯 TESTING SIGNAL GENERATION...")
    
    try:
        from phase3_live_trading_preparation import LiveTradingSystem
        
        # Initialize system
        print("   Initializing system...")
        live_system = LiveTradingSystem()
        success = live_system.initialize_system(df)
        
        if not success:
            print("   ❌ Failed to initialize system")
            return
        
        print("   ✅ System initialized")
        
        # Test with recent data
        recent_data = df.tail(100)
        print(f"   Testing with {len(recent_data)} recent bars...")
        
        # Get ensemble predictions
        features = live_system.feature_engineer.create_enhanced_features(recent_data)
        predictions = live_system.ensemble.predict_ensemble(features)
        
        print(f"   ✅ Generated predictions for {len(predictions['final_prediction'])} samples")
        
        # Analyze predictions
        final_pred = predictions['final_prediction']
        confidence = predictions['ensemble_confidence']
        
        print(f"\n📈 PREDICTION ANALYSIS:")
        print(f"   Expected Value Range: {final_pred.min():.6f} to {final_pred.max():.6f}")
        print(f"   Average Expected Value: {final_pred.mean():.6f}")
        print(f"   Confidence Range: {confidence.min():.3f} to {confidence.max():.3f}")
        print(f"   Average Confidence: {confidence.mean():.3f}")
        
        # Check signal criteria
        print(f"\n🎯 SIGNAL CRITERIA ANALYSIS:")
        
        # Current criteria (from the system)
        min_expected_value = 0.0004  # 4 pips minimum
        min_confidence = 0.7         # 70% confidence minimum
        
        print(f"   Current Minimum Expected Value: {min_expected_value:.6f} (4 pips)")
        print(f"   Current Minimum Confidence: {min_confidence:.1%}")
        
        # Count signals that meet criteria
        high_ev_signals = final_pred >= min_expected_value
        high_conf_signals = confidence >= min_confidence
        both_criteria = high_ev_signals & high_conf_signals
        
        print(f"\n📊 SIGNAL COUNTS:")
        print(f"   High Expected Value (≥{min_expected_value:.6f}): {high_ev_signals.sum()}/{len(final_pred)} ({high_ev_signals.sum()/len(final_pred)*100:.1f}%)")
        print(f"   High Confidence (≥{min_confidence:.1%}): {high_conf_signals.sum()}/{len(final_pred)} ({high_conf_signals.sum()/len(final_pred)*100:.1f}%)")
        print(f"   Both Criteria Met: {both_criteria.sum()}/{len(final_pred)} ({both_criteria.sum()/len(final_pred)*100:.1f}%)")
        
        # Suggest adjustments
        print(f"\n💡 SUGGESTED ADJUSTMENTS:")
        
        # Lower expected value threshold
        lower_ev_thresholds = [0.0002, 0.0003, 0.0004, 0.0005]
        print(f"   Expected Value Thresholds:")
        for threshold in lower_ev_thresholds:
            count = (final_pred >= threshold).sum()
            percentage = count / len(final_pred) * 100
            print(f"     ≥{threshold:.6f} ({threshold*10000:.1f} pips): {count} signals ({percentage:.1f}%)")
        
        # Lower confidence threshold
        lower_conf_thresholds = [0.5, 0.6, 0.7, 0.8]
        print(f"   Confidence Thresholds:")
        for threshold in lower_conf_thresholds:
            count = (confidence >= threshold).sum()
            percentage = count / len(final_pred) * 100
            print(f"     ≥{threshold:.1%}: {count} signals ({percentage:.1f}%)")
        
        # Combined analysis
        print(f"\n🎯 COMBINED THRESHOLD ANALYSIS:")
        for ev_thresh in [0.0002, 0.0003, 0.0004]:
            for conf_thresh in [0.5, 0.6, 0.7]:
                count = ((final_pred >= ev_thresh) & (confidence >= conf_thresh)).sum()
                percentage = count / len(final_pred) * 100
                print(f"   EV≥{ev_thresh:.6f} & Conf≥{conf_thresh:.1%}: {count} signals ({percentage:.1f}%)")
        
        # Test with adjusted criteria
        print(f"\n🧪 TESTING ADJUSTED CRITERIA:")
        
        # Test with lower thresholds
        test_ev_threshold = 0.0002  # 2 pips
        test_conf_threshold = 0.5   # 50% confidence
        
        test_signals = (final_pred >= test_ev_threshold) & (confidence >= test_conf_threshold)
        test_count = test_signals.sum()
        test_percentage = test_count / len(final_pred) * 100
        
        print(f"   With EV≥{test_ev_threshold:.6f} & Conf≥{test_conf_threshold:.1%}:")
        print(f"   Signals: {test_count}/{len(final_pred)} ({test_percentage:.1f}%)")
        
        if test_count > 0:
            print(f"   ✅ Would generate {test_count} signals with adjusted criteria")
            
            # Show sample signals
            signal_indices = np.where(test_signals)[0]
            print(f"\n📋 SAMPLE SIGNALS (first 3):")
            for i in range(min(3, len(signal_indices))):
                idx = signal_indices[i]
                print(f"   Signal {i+1}: EV={final_pred[idx]:.6f}, Conf={confidence[idx]:.3f}")
        else:
            print(f"   ❌ Still no signals with adjusted criteria")
        
        # Market conditions analysis
        print(f"\n📊 MARKET CONDITIONS ANALYSIS:")
        
        # Calculate some basic market metrics
        recent_prices = recent_data['close'].values
        price_changes = np.diff(recent_prices)
        volatility = np.std(price_changes) * 10000  # In pips
        
        print(f"   Recent Volatility: {volatility:.2f} pips")
        print(f"   Price Range: {recent_prices.max() - recent_prices.min():.5f}")
        print(f"   Average Price Change: {np.mean(np.abs(price_changes)) * 10000:.2f} pips")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if test_count > 0:
            print(f"   1. ✅ Consider lowering signal criteria to generate more trades")
            print(f"   2. ✅ Current market conditions may support trading")
            print(f"   3. ✅ System is working correctly - just needs adjustment")
        else:
            print(f"   1. ⚠️ Market conditions may be unfavorable for trading")
            print(f"   2. ⚠️ Consider waiting for better market conditions")
            print(f"   3. ⚠️ System may need retraining with current market data")
        
        print(f"\n🔧 NEXT STEPS:")
        print(f"   1. Run this analysis during active trading hours (5:00-23:00 UTC)")
        print(f"   2. Consider adjusting signal criteria if needed")
        print(f"   3. Monitor for better market conditions")
        print(f"   4. Check if models need retraining with recent data")
        
    except Exception as e:
        print(f"❌ Error analyzing signal criteria: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function."""
    print("🎯 Signal Criteria Analysis")
    print("="*50)
    
    analyze_signal_criteria()
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main() 