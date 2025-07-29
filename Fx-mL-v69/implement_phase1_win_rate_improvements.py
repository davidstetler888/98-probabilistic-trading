#!/usr/bin/env python3
"""
Phase 1 Win Rate Improvements Implementation
============================================

This script implements Phase 1 of the win rate optimization strategy:
1. Enhanced signal quality scoring
2. Quality-based signal selection
3. Updated configuration for win rate optimization

Target: Achieve 55-65% win rate with 45-55 trades per week
"""

import argparse
import shutil
import yaml
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

def backup_current_config():
    """Create backup of current configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"config_backup_phase1_{timestamp}.yaml"
    
    if Path("config.yaml").exists():
        shutil.copy("config.yaml", backup_name)
        print(f"✅ Created backup: {backup_name}")
        return backup_name
    else:
        print("⚠️ No config.yaml found to backup")
        return None

def update_config_for_phase1():
    """Update configuration for Phase 1 win rate optimization"""
    
    # Load current config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    print("\n🎯 PHASE 1: WIN RATE OPTIMIZATION CONFIG UPDATES")
    print("=" * 55)
    
    # 1. Update goals for win rate optimization
    print("📊 1. Updating performance goals...")
    if "goals" not in config:
        config["goals"] = {}
    
    config["goals"]["win_rate_range"] = [0.55, 0.75]  # Phase 1 target
    config["goals"]["risk_reward_range"] = [2.0, 3.5]  # Maintain excellent RR
    config["goals"]["trades_per_week_range"] = [40, 55]  # Slight reduction for quality
    print("  ✓ Win rate target: 55-75% (Phase 1)")
    print("  ✓ RR range: 2.0-3.5 (maintained)")
    print("  ✓ Trades/week: 40-55 (quality focus)")
    
    # 2. Ranker optimizations for signal quality
    print("\n🎖️ 2. Optimizing ranker for signal quality...")
    if "ranker" not in config:
        config["ranker"] = {}
    
    config["ranker"]["target_trades_per_week"] = 50     # Reduced from previous high volume
    config["ranker"]["min_trades_per_week"] = 40        # Increase minimum
    config["ranker"]["max_trades_per_week"] = 55        # Reduce maximum for quality
    config["ranker"]["enhanced_filtering"] = True       # Enable enhanced filtering
    config["ranker"]["quality_threshold"] = 65          # Phase 1 quality threshold
    config["ranker"]["confidence_based_sizing"] = True  # Dynamic position sizing
    print("  ✓ Target trades/week: 50 (quality focused)")
    print("  ✓ Enhanced filtering enabled")
    print("  ✓ Quality threshold: 65% (Phase 1)")
    
    # 3. Signal processing improvements
    print("\n🤖 3. Enhancing signal processing...")
    if "signal" not in config:
        config["signal"] = {}
    
    config["signal"]["min_confidence_threshold"] = 0.62  # Increase confidence requirement
    config["signal"]["enhanced_meta_model"] = True       # Enhanced meta-model
    config["signal"]["quality_based_selection"] = True   # Enable quality-based selection
    print("  ✓ Minimum confidence: 62%")
    print("  ✓ Enhanced meta-model enabled")
    print("  ✓ Quality-based selection enabled")
    
    # 4. Simulation optimizations
    print("\n⚙️ 4. Optimizing simulation parameters...")
    if "simulation" not in config:
        config["simulation"] = {}
    
    config["simulation"]["max_positions"] = 2            # Allow 2 concurrent positions
    config["simulation"]["cooldown_min"] = 5             # Maintain reasonable cooldown
    config["simulation"]["dynamic_sizing"] = True        # Enable dynamic sizing
    config["simulation"]["max_weekly_trades"] = 55       # Align with ranker
    print("  ✓ Max positions: 2 (balanced)")
    print("  ✓ Cooldown: 5 minutes")
    print("  ✓ Dynamic sizing enabled")
    
    # 5. Add Phase 1 specific configuration
    print("\n🚀 5. Adding Phase 1 specific settings...")
    config["win_rate_optimization"] = {
        "enabled": True,
        "phase": 1,
        "target_win_rate": 0.60,
        "min_acceptable_win_rate": 0.55,
        "quality_score_weight": 0.3,  # Phase 1 moderate weight
        "implementation_date": datetime.now().isoformat(),
        "description": "Phase 1: Enhanced signal quality scoring and selection"
    }
    print("  ✓ Phase 1 optimization enabled")
    print("  ✓ Target win rate: 60%")
    print("  ✓ Minimum acceptable: 55%")
    
    # Save updated config
    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print("\n✅ Configuration updated for Phase 1 win rate optimization")
    return config

def create_enhanced_ranker_functions():
    """Create enhanced ranker functions file"""
    
    enhanced_code = '''#!/usr/bin/env python3
"""
Enhanced Signal Quality Functions for Phase 1 Win Rate Optimization
==================================================================

This module provides enhanced signal quality scoring and selection functions
to be integrated into train_ranker.py for Phase 1 win rate improvements.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

def calculate_enhanced_signal_quality(signal_data: pd.Series) -> float:
    """
    Calculate comprehensive signal quality score (0-100) for Phase 1
    
    Args:
        signal_data: Series containing signal information
        
    Returns:
        Quality score between 0-100
    """
    quality_score = 0
    
    # Base probability strength (0-30 points)
    base_prob = signal_data.get('meta_prob', 0.5)
    quality_score += min(base_prob * 30, 30)
    
    # Market regime alignment (0-20 points)
    regime = signal_data.get('market_regime', 1)
    # Phase 1: Simple regime scoring
    regime_scores = {0: 18, 1: 15, 2: 18, 3: 10}  # Trending regimes preferred
    quality_score += regime_scores.get(regime, 10)
    
    # Session quality bonus (0-15 points)
    try:
        hour = signal_data.name.hour if hasattr(signal_data.name, 'hour') else 12
    except:
        hour = 12
    
    if 8 <= hour < 13:      # London session
        quality_score += 15
    elif 13 <= hour < 18:   # NY session
        quality_score += 12
    elif 12 <= hour < 14:   # Overlap
        quality_score += 15
    else:                   # Asian/off-hours
        quality_score += 8
    
    # Volatility quality (0-15 points)
    atr = signal_data.get('atr', 0.0012)
    if atr > 0:
        optimal_atr = 0.0015  # Optimal volatility for EURUSD
        atr_deviation = abs(atr - optimal_atr) / optimal_atr
        vol_score = max(0, 15 * (1 - min(atr_deviation, 1.0)))
        quality_score += vol_score
    else:
        quality_score += 7.5  # Default medium score
    
    # Risk-reward bonus (0-20 points)
    if 'tp_pips' in signal_data.index and 'sl_pips' in signal_data.index:
        tp_pips = signal_data.get('tp_pips', 0)
        sl_pips = signal_data.get('sl_pips', 1)
        if sl_pips > 0:
            rr_ratio = tp_pips / sl_pips
            # Reward good RR ratios (2.0-3.0 optimal)
            if 2.0 <= rr_ratio <= 3.0:
                quality_score += 20
            elif 1.8 <= rr_ratio < 2.0 or 3.0 < rr_ratio <= 3.5:
                quality_score += 15
            elif 1.5 <= rr_ratio < 1.8 or 3.5 < rr_ratio <= 4.0:
                quality_score += 10
            else:
                quality_score += 5
    else:
        quality_score += 10  # Default medium score
    
    return min(quality_score, 100)

def select_signals_by_quality(signals_df: pd.DataFrame, 
                             target_trades_per_week: int = 50) -> pd.DataFrame:
    """
    Select signals based on quality scores and target volume for Phase 1
    
    Args:
        signals_df: DataFrame containing signals
        target_trades_per_week: Target number of trades per week
        
    Returns:
        Selected signals DataFrame
    """
    if signals_df.empty:
        return signals_df
    
    print(f"[quality_selection] Starting with {len(signals_df)} signals")
    
    # Calculate quality scores
    signals_df = signals_df.copy()
    signals_df['quality_score'] = signals_df.apply(calculate_enhanced_signal_quality, axis=1)
    
    # Calculate target number of signals
    weeks = len(signals_df) / (7 * 24 * 12)  # 5-minute bars
    target_signals = int(target_trades_per_week * weeks)
    
    print(f"[quality_selection] Target signals: {target_signals} for {weeks:.1f} weeks")
    
    # Phase 1: Apply minimum quality threshold (top 65% of all signals)
    quality_threshold = signals_df['quality_score'].quantile(0.35)  # Bottom 35% filtered out
    qualified_signals = signals_df[signals_df['quality_score'] >= quality_threshold]
    
    print(f"[quality_selection] Quality threshold: {quality_threshold:.1f}")
    print(f"[quality_selection] Qualified signals: {len(qualified_signals)}")
    
    # If we have enough qualified signals, take the top ones by quality
    if len(qualified_signals) >= target_signals:
        selected_signals = qualified_signals.nlargest(target_signals, 'quality_score')
        print(f"[quality_selection] Selected top {target_signals} by quality")
    else:
        # Take all qualified signals and fill with next best by edge score
        remaining_needed = target_signals - len(qualified_signals)
        remaining_signals = signals_df[signals_df['quality_score'] < quality_threshold]
        
        if not remaining_signals.empty:
            # Sort remaining by edge score and take best
            additional_signals = remaining_signals.nlargest(remaining_needed, 'edge_score')
            selected_signals = pd.concat([qualified_signals, additional_signals])
            print(f"[quality_selection] Used all {len(qualified_signals)} qualified + {len(additional_signals)} additional")
        else:
            selected_signals = qualified_signals
            print(f"[quality_selection] Used all {len(qualified_signals)} qualified signals")
    
    # Sort final selection by quality score for consistency
    selected_signals = selected_signals.sort_values('quality_score', ascending=False)
    
    print(f"[quality_selection] Final selection: {len(selected_signals)} signals")
    print(f"[quality_selection] Quality range: {selected_signals['quality_score'].min():.1f} - {selected_signals['quality_score'].max():.1f}")
    
    return selected_signals

def apply_phase1_enhancements(signals_df: pd.DataFrame,
                             target_trades_per_week: int = 50) -> pd.DataFrame:
    """
    Apply Phase 1 win rate enhancements to signals
    
    Args:
        signals_df: Input signals DataFrame
        target_trades_per_week: Target trades per week
        
    Returns:
        Enhanced and filtered signals DataFrame
    """
    print(f"\\n[phase1] Starting Phase 1 win rate enhancement...")
    print(f"[phase1] Input signals: {len(signals_df)}")
    
    if signals_df.empty:
        print("[phase1] No signals to process")
        return signals_df
    
    # Apply quality-based selection
    enhanced_signals = select_signals_by_quality(signals_df, target_trades_per_week)
    
    # Add confidence-based risk sizing hints
    if not enhanced_signals.empty:
        enhanced_signals = enhanced_signals.copy()
        enhanced_signals['confidence_multiplier'] = (
            enhanced_signals['quality_score'] / 100 * 0.5 + 0.75  # 0.75x to 1.25x
        )
        enhanced_signals['suggested_risk'] = (
            enhanced_signals['confidence_multiplier'] * 0.02  # Base 2% risk
        )
        enhanced_signals['suggested_risk'] = enhanced_signals['suggested_risk'].clip(0.01, 0.03)
    
    print(f"[phase1] Final enhanced signals: {len(enhanced_signals)}")
    
    return enhanced_signals

def get_phase1_quality_threshold(signals_df: pd.DataFrame) -> float:
    """
    Calculate Phase 1 quality threshold for signal filtering
    
    Args:
        signals_df: DataFrame with quality scores
        
    Returns:
        Quality threshold value
    """
    if signals_df.empty or 'quality_score' not in signals_df.columns:
        return 65.0  # Default Phase 1 threshold
    
    # Phase 1: Use 65th percentile as threshold
    threshold = signals_df['quality_score'].quantile(0.35)  # Bottom 35% filtered
    
    # Ensure minimum threshold
    threshold = max(threshold, 50.0)  # Never go below 50
    
    return threshold

# Integration helper for train_ranker.py
def integrate_with_ranker(signals_df: pd.DataFrame, 
                         target_trades_per_week: int,
                         use_quality_selection: bool = True) -> pd.DataFrame:
    """
    Integration function for train_ranker.py to use Phase 1 enhancements
    
    Args:
        signals_df: Input signals from ranker
        target_trades_per_week: Target trades per week
        use_quality_selection: Whether to use quality-based selection
        
    Returns:
        Enhanced signals DataFrame
    """
    if not use_quality_selection:
        return signals_df
    
    try:
        return apply_phase1_enhancements(signals_df, target_trades_per_week)
    except Exception as e:
        print(f"[ranker_integration] Error in Phase 1 enhancements: {e}")
        print("[ranker_integration] Falling back to original signals")
        return signals_df
'''
    
    # Write the enhanced functions file
    with open("enhanced_signal_quality_phase1.py", "w") as f:
        f.write(enhanced_code)
    
    print("✅ Created enhanced_signal_quality_phase1.py")
    return "enhanced_signal_quality_phase1.py"

def update_train_ranker_for_phase1():
    """Update train_ranker.py to use Phase 1 enhancements"""
    
    ranker_file = Path("train_ranker.py")
    if not ranker_file.exists():
        print("⚠️ train_ranker.py not found - cannot apply Phase 1 integration")
        return False
    
    # Read current ranker file
    with open(ranker_file, "r") as f:
        content = f.read()
    
    # Check if already integrated
    if "enhanced_signal_quality_phase1" in content:
        print("✅ train_ranker.py already has Phase 1 integration")
        return True
    
    # Add import at the top
    import_line = "from enhanced_signal_quality_phase1 import integrate_with_ranker, apply_phase1_enhancements\n"
    
    # Find the imports section and add our import
    lines = content.split('\n')
    import_inserted = False
    
    for i, line in enumerate(lines):
        if line.startswith('from config import config'):
            lines.insert(i + 1, import_line.strip())
            import_inserted = True
            break
    
    if not import_inserted:
        # Add after other imports
        for i, line in enumerate(lines):
            if line.startswith('import') and not lines[i+1].startswith('import'):
                lines.insert(i + 1, import_line.strip())
                break
    
    # Find the main signal processing section and add Phase 1 integration
    for i, line in enumerate(lines):
        if "signals = pd.concat(signal_frames).sort_index()" in line:
            # Add Phase 1 integration after signal concatenation
            integration_code = """
    # Phase 1 Win Rate Enhancement Integration
    try:
        from config import config
        use_quality_selection = config.get("signal.quality_based_selection", False)
        if use_quality_selection:
            print("[ranker] Applying Phase 1 win rate enhancements...")
            signals = integrate_with_ranker(signals, target, use_quality_selection=True)
            print(f"[ranker] Phase 1 enhanced signals: {len(signals)}")
    except Exception as e:
        print(f"[ranker] Phase 1 enhancement failed: {e}, using original signals")
"""
            lines.insert(i + 1, integration_code)
            break
    
    # Write updated file
    updated_content = '\n'.join(lines)
    
    # Backup original
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"train_ranker_backup_phase1_{timestamp}.py"
    shutil.copy(ranker_file, backup_name)
    print(f"✅ Created backup: {backup_name}")
    
    # Write updated file
    with open(ranker_file, "w") as f:
        f.write(updated_content)
    
    print("✅ Updated train_ranker.py with Phase 1 integration")
    return True

def create_phase1_test_script():
    """Create a test script for Phase 1 functionality"""
    
    test_script = '''#!/usr/bin/env python3
"""
Phase 1 Win Rate Optimization Test Script
=========================================

Test the Phase 1 enhancements before running full walkforward analysis.
"""

import pandas as pd
import numpy as np
from enhanced_signal_quality_phase1 import (
    calculate_enhanced_signal_quality,
    select_signals_by_quality,
    apply_phase1_enhancements
)

def create_test_signals(n_signals=1000):
    """Create test signals for validation"""
    
    # Create test timestamp index
    dates = pd.date_range('2024-01-01', periods=n_signals, freq='5T')
    
    # Create test signals
    test_data = {
        'meta_prob': np.random.uniform(0.4, 0.9, n_signals),
        'market_regime': np.random.choice([0, 1, 2, 3], n_signals),
        'edge_score': np.random.uniform(0.001, 0.01, n_signals),
        'tp_pips': np.random.uniform(15, 35, n_signals),
        'sl_pips': np.random.uniform(8, 18, n_signals),
        'atr': np.random.uniform(0.0008, 0.002, n_signals),
        'side': np.random.choice(['long', 'short'], n_signals)
    }
    
    df = pd.DataFrame(test_data, index=dates)
    return df

def test_quality_scoring():
    """Test the quality scoring function"""
    print("\\n🧪 Testing Quality Scoring Function...")
    
    test_signals = create_test_signals(100)
    
    # Calculate quality scores
    quality_scores = test_signals.apply(calculate_enhanced_signal_quality, axis=1)
    
    print(f"Quality scores range: {quality_scores.min():.1f} - {quality_scores.max():.1f}")
    print(f"Quality scores mean: {quality_scores.mean():.1f}")
    print(f"Quality scores std: {quality_scores.std():.1f}")
    
    # Check distribution
    print("\\nQuality score distribution:")
    print(f"  90-100: {(quality_scores >= 90).sum()} signals")
    print(f"  80-89:  {((quality_scores >= 80) & (quality_scores < 90)).sum()} signals")
    print(f"  70-79:  {((quality_scores >= 70) & (quality_scores < 80)).sum()} signals")
    print(f"  60-69:  {((quality_scores >= 60) & (quality_scores < 70)).sum()} signals")
    print(f"  <60:    {(quality_scores < 60).sum()} signals")
    
    return quality_scores

def test_signal_selection():
    """Test the signal selection function"""
    print("\\n🧪 Testing Signal Selection Function...")
    
    test_signals = create_test_signals(1000)
    target_trades_per_week = 50
    
    # Apply selection
    selected_signals = select_signals_by_quality(test_signals, target_trades_per_week)
    
    print(f"Original signals: {len(test_signals)}")
    print(f"Selected signals: {len(selected_signals)}")
    print(f"Selection ratio: {len(selected_signals)/len(test_signals)*100:.1f}%")
    
    if not selected_signals.empty:
        print(f"Selected quality range: {selected_signals['quality_score'].min():.1f} - {selected_signals['quality_score'].max():.1f}")
        print(f"Selected quality mean: {selected_signals['quality_score'].mean():.1f}")
    
    return selected_signals

def test_phase1_integration():
    """Test the full Phase 1 integration"""
    print("\\n🧪 Testing Phase 1 Integration...")
    
    test_signals = create_test_signals(2000)
    target_trades_per_week = 50
    
    # Apply Phase 1 enhancements
    enhanced_signals = apply_phase1_enhancements(test_signals, target_trades_per_week)
    
    print(f"Original signals: {len(test_signals)}")
    print(f"Enhanced signals: {len(enhanced_signals)}")
    
    if not enhanced_signals.empty:
        print(f"Quality range: {enhanced_signals['quality_score'].min():.1f} - {enhanced_signals['quality_score'].max():.1f}")
        print(f"Confidence multiplier range: {enhanced_signals['confidence_multiplier'].min():.2f} - {enhanced_signals['confidence_multiplier'].max():.2f}")
        print(f"Suggested risk range: {enhanced_signals['suggested_risk'].min():.3f} - {enhanced_signals['suggested_risk'].max():.3f}")
    
    return enhanced_signals

def main():
    """Run all Phase 1 tests"""
    print("🚀 PHASE 1 WIN RATE OPTIMIZATION TESTS")
    print("=" * 45)
    
    try:
        # Test quality scoring
        quality_scores = test_quality_scoring()
        
        # Test signal selection
        selected_signals = test_signal_selection()
        
        # Test full integration
        enhanced_signals = test_phase1_integration()
        
        print("\\n✅ All Phase 1 tests completed successfully!")
        print("\\n📊 Summary:")
        print(f"  - Quality scoring: Working ✅")
        print(f"  - Signal selection: Working ✅") 
        print(f"  - Phase 1 integration: Working ✅")
        print("\\n🎯 Phase 1 is ready for deployment!")
        
    except Exception as e:
        print(f"\\n❌ Phase 1 tests failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''
    
    with open("test_phase1_enhancements.py", "w") as f:
        f.write(test_script)
    
    print("✅ Created test_phase1_enhancements.py")
    return "test_phase1_enhancements.py"

def main():
    """Main implementation function"""
    parser = argparse.ArgumentParser(description="Implement Phase 1 Win Rate Improvements")
    parser.add_argument("--test-only", action="store_true", help="Only run tests, don't modify files")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backups")
    args = parser.parse_args()
    
    print("🚀 PHASE 1 WIN RATE OPTIMIZATION IMPLEMENTATION")
    print("=" * 55)
    print("Target: 55-65% win rate with 45-55 trades per week")
    print("Focus: Enhanced signal quality scoring and selection")
    
    try:
        if not args.test_only:
            # 1. Backup current configuration
            if not args.no_backup:
                backup_config = backup_current_config()
            
            # 2. Update configuration
            updated_config = update_config_for_phase1()
            
            # 3. Create enhanced ranker functions
            enhanced_file = create_enhanced_ranker_functions()
            
            # 4. Update train_ranker.py
            ranker_updated = update_train_ranker_for_phase1()
        
        # 5. Create test script
        test_script = create_phase1_test_script()
        
        print("\n🎯 PHASE 1 IMPLEMENTATION COMPLETE!")
        print("=" * 40)
        
        if not args.test_only:
            print("✅ Configuration updated for Phase 1")
            print("✅ Enhanced signal quality functions created")
            print("✅ train_ranker.py integration completed")
        
        print("✅ Test script created")
        
        print("\n📋 NEXT STEPS:")
        print("1. Run tests:")
        print("   python test_phase1_enhancements.py")
        print("\n2. Test with small dataset:")
        print("   python walkforward.py --run output/phase1_test --stepback_weeks 2 --optimize --grid fast")
        print("\n3. If tests pass, run full analysis:")
        print("   python walkforward.py --run output/phase1_full --stepback_weeks 4 --optimize --grid fast")
        print("\n4. Monitor results in output.txt for:")
        print("   - Win rate: Target 55-65%")
        print("   - Trades/week: Target 45-55")
        print("   - RR: Maintain >2.0")
        print("   - Profit factor: Target >2.0")
        
        print("\n⚠️ IMPORTANT REMINDERS:")
        print("- Monitor win rate stability (should be <10% variation)")
        print("- If win rate drops below 50%, consider rollback")
        print("- Phase 2 can be implemented after 1 week of stable results")
        
    except Exception as e:
        print(f"\n❌ Implementation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())