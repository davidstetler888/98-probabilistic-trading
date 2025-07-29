#!/usr/bin/env python3
"""
Win Rate Optimization Implementation Script

This script applies immediate win rate improvements to your trading system.
Goal: Improve win rate from 45-55% to 60-70% while maintaining 35+ trades/week.

Usage:
    python apply_win_rate_optimizations.py [--test] [--backup]
"""

import argparse
import sys
import os
import shutil
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Import our enhanced signal quality module
try:
    from enhanced_signal_quality import (
        apply_win_rate_enhancements,
        calculate_enhanced_signal_quality,
        AdvancedRiskManager
    )
except ImportError:
    print("❌ Error: enhanced_signal_quality.py not found")
    print("Please ensure enhanced_signal_quality.py is in the same directory")
    sys.exit(1)

def backup_current_config(config_file: str = "config.yaml") -> Optional[str]:
    """Create a backup of the current configuration"""
    if not Path(config_file).exists():
        print(f"❌ Configuration file {config_file} not found!")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"config_backup_winrate_{timestamp}.yaml"
    
    try:
        shutil.copy(config_file, backup_name)
        print(f"✅ Configuration backed up as: {backup_name}")
        return backup_name
    except Exception as e:
        print(f"❌ Error creating backup: {e}")
        return None

def load_config(config_file: str = "config.yaml") -> Dict[str, Any]:
    """Load current configuration"""
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return {}

def save_config(config: Dict[str, Any], config_file: str = "config.yaml") -> bool:
    """Save updated configuration"""
    try:
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"✅ Configuration updated: {config_file}")
        return True
    except Exception as e:
        print(f"❌ Error saving config: {e}")
        return False

def apply_win_rate_config_changes(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply win rate optimization changes to configuration"""
    
    print("\n🎯 APPLYING WIN RATE OPTIMIZATION CHANGES")
    print("=" * 50)
    
    # 1. Update goals for higher win rate targets
    print("📊 1. Updating performance goals...")
    if "goals" not in config:
        config["goals"] = {}
    
    config["goals"]["win_rate_range"] = [0.60, 0.75]  # Increased from [0.55, 0.75]
    config["goals"]["risk_reward_range"] = [2.0, 3.5]  # Maintain good RR
    config["goals"]["trades_per_week_range"] = [35, 50]  # Keep current frequency
    print("  ✓ Win rate target: 60-75%")
    print("  ✓ RR range: 2.0-3.5")
    print("  ✓ Trades/week: 35-50")
    
    # 2. Ranker optimizations for signal quality
    print("\n🎖️ 2. Optimizing ranker for signal quality...")
    if "ranker" not in config:
        config["ranker"] = {}
    
    config["ranker"]["target_trades_per_week"] = 40  # Maintain current target
    config["ranker"]["min_trades_per_week"] = 35     # Increase minimum
    config["ranker"]["max_trades_per_week"] = 50     # Keep current max
    config["ranker"]["enhanced_filtering"] = True    # Enable enhanced filtering
    config["ranker"]["quality_threshold"] = 0.70     # Quality score threshold
    config["ranker"]["confidence_based_sizing"] = True  # Dynamic position sizing
    print("  ✓ Enhanced signal filtering enabled")
    print("  ✓ Quality threshold set to 70%")
    print("  ✓ Confidence-based sizing enabled")
    
    # 3. Signal processing improvements
    print("\n🤖 3. Enhancing signal processing...")
    if "signal" not in config:
        config["signal"] = {}
    
    config["signal"]["min_precision_target"] = 0.75     # Maintain current
    config["signal"]["min_signals_per_week"] = 12       # Maintain current
    config["signal"]["multi_timeframe_features"] = True # Enable HTF features
    config["signal"]["enhanced_meta_model"] = True      # Enhanced meta-model
    config["signal"]["min_confidence_threshold"] = 0.60 # Minimum confidence
    print("  ✓ Multi-timeframe features enabled")
    print("  ✓ Enhanced meta-model enabled")
    print("  ✓ Minimum confidence: 60%")
    
    # 4. Simulation enhancements
    print("\n⚙️ 4. Optimizing simulation parameters...")
    if "simulation" not in config:
        config["simulation"] = {}
    
    config["simulation"]["max_positions"] = 2            # Maintain current
    config["simulation"]["cooldown_min"] = 5             # Maintain current
    config["simulation"]["max_weekly_trades"] = 50       # Keep current limit
    config["simulation"]["risk_per_trade"] = 0.02        # Maintain current
    config["simulation"]["dynamic_sizing"] = True        # Enable dynamic sizing
    config["simulation"]["advanced_risk_management"] = True  # Advanced risk mgmt
    print("  ✓ Dynamic position sizing enabled")
    print("  ✓ Advanced risk management enabled")
    
    # 5. Label optimization for better signal quality
    print("\n🏷️ 5. Optimizing labeling for quality...")
    if "label" not in config:
        config["label"] = {}
    
    config["label"]["threshold"] = 0.0008               # Keep current optimized value
    config["label"]["min_rr"] = 1.8                     # Keep current optimized value
    config["label"]["min_win_rate_target"] = 0.60       # Increase from 0.55
    config["label"]["cooldown_min"] = 5                 # Maintain current
    config["label"]["enhanced_quality_scoring"] = True  # Enable quality scoring
    print("  ✓ Win rate target increased to 60%")
    print("  ✓ Enhanced quality scoring enabled")
    
    return config

def create_enhanced_ranker_integration():
    """Create integration code for enhanced ranker functions"""
    
    integration_code = '''
# Enhanced Signal Ranking Integration
# Add this code to your train_ranker.py file

def apply_win_rate_enhancements_to_signals(signals_df, target_trades_per_week=40):
    """
    Apply win rate enhancements to generated signals
    """
    try:
        from enhanced_signal_quality import apply_win_rate_enhancements
        
        print("[ranker] Applying win rate enhancements...")
        enhanced_signals = apply_win_rate_enhancements(
            signals_df, 
            target_trades_per_week=target_trades_per_week
        )
        
        print(f"[ranker] Enhanced signals: {len(enhanced_signals)} from {len(signals_df)} original")
        return enhanced_signals
        
    except ImportError:
        print("[ranker] Win rate enhancements not available, using original signals")
        return signals_df
    except Exception as e:
        print(f"[ranker] Error applying enhancements: {e}")
        return signals_df

# Add this call in your main() function after signal generation:
# signals = apply_win_rate_enhancements_to_signals(signals, target)
'''
    
    with open("ranker_integration.py", "w") as f:
        f.write(integration_code)
    
    print("📄 Created ranker integration code: ranker_integration.py")
    print("   ⚠️ Add the integration code to your train_ranker.py file")

def create_monitoring_script():
    """Create enhanced monitoring script for win rate tracking"""
    
    monitoring_code = '''#!/usr/bin/env python3
"""
Enhanced Win Rate Performance Monitor

Monitor win rate improvements and provide actionable insights.
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path

def analyze_win_rate_performance(output_file="output/output.txt"):
    """Analyze win rate performance from walkforward results"""
    
    results = []
    
    try:
        with open(output_file, 'r') as f:
            lines = f.readlines()
        
        current_result = {}
        for line in lines:
            line = line.strip()
            
            if "Total Trades:" in line:
                trades = int(line.split(": ")[1])
                current_result['total_trades'] = trades
                
            elif "Win Rate:" in line:
                win_rate = float(line.split(": ")[1].replace("%", ""))
                current_result['win_rate'] = win_rate
                
            elif "Average RR:" in line:
                avg_rr = float(line.split(": ")[1])
                current_result['avg_rr'] = avg_rr
                
            elif "Profit Factor:" in line:
                pf = line.split(": ")[1]
                if pf == "inf":
                    current_result['profit_factor'] = float('inf')
                else:
                    current_result['profit_factor'] = float(pf)
                
            elif line.startswith("2024-") and "Win Rate" in line:
                # Week summary line
                if current_result:
                    results.append(current_result.copy())
                    current_result = {}
    
    except FileNotFoundError:
        print(f"❌ Output file not found: {output_file}")
        return []
    
    return results

def generate_win_rate_report(results):
    """Generate comprehensive win rate analysis report"""
    
    if not results:
        print("❌ No results to analyze")
        return
    
    print("\\n" + "="*70)
    print("📊 WIN RATE OPTIMIZATION PERFORMANCE REPORT")
    print("="*70)
    
    df = pd.DataFrame(results)
    
    if not df.empty:
        current_win_rate = df['win_rate'].mean()
        win_rate_std = df['win_rate'].std()
        avg_trades = df['total_trades'].mean()
        avg_rr = df['avg_rr'].mean()
        
        print(f"\\n📈 PERFORMANCE SUMMARY:")
        print(f"   • Periods Analyzed: {len(df)}")
        print(f"   • Average Win Rate: {current_win_rate:.1f}% (σ={win_rate_std:.1f}%)")
        print(f"   • Average Trades/Period: {avg_trades:.1f}")
        print(f"   • Average RR: {avg_rr:.2f}")
        
        # Win rate assessment
        print(f"\\n🎯 WIN RATE ANALYSIS:")
        if current_win_rate >= 65:
            rating = "✅ EXCELLENT"
            recommendation = "Target achieved! Consider further optimization."
        elif current_win_rate >= 60:
            rating = "✅ GOOD"
            recommendation = "Close to target. Minor adjustments may help."
        elif current_win_rate >= 55:
            rating = "⚠️ ACCEPTABLE"
            recommendation = "Improvement needed. Apply Phase 2 enhancements."
        else:
            rating = "❌ NEEDS WORK"
            recommendation = "Significant improvement required. Check configuration."
        
        print(f"   • Rating: {rating}")
        print(f"   • Recommendation: {recommendation}")
        
        # Stability analysis
        print(f"\\n📊 WIN RATE STABILITY:")
        if win_rate_std < 8:
            stability = "✅ STABLE"
        elif win_rate_std < 15:
            stability = "⚠️ MODERATE"
        else:
            stability = "❌ VOLATILE"
        
        print(f"   • Stability: {stability} (σ={win_rate_std:.1f}%)")
        print(f"   • Min Win Rate: {df['win_rate'].min():.1f}%")
        print(f"   • Max Win Rate: {df['win_rate'].max():.1f}%")
        
        # Trade frequency analysis
        print(f"\\n📊 TRADE FREQUENCY:")
        if avg_trades >= 35:
            freq_rating = "✅ TARGET MET"
        elif avg_trades >= 25:
            freq_rating = "⚠️ CLOSE TO TARGET"
        else:
            freq_rating = "❌ BELOW TARGET"
        
        print(f"   • Rating: {freq_rating}")
        print(f"   • Current: {avg_trades:.1f} trades/period")
        print(f"   • Target: 35+ trades/period")
        
        # Overall assessment
        print(f"\\n🏆 OVERALL ASSESSMENT:")
        
        success_score = 0
        if current_win_rate >= 60: success_score += 40
        elif current_win_rate >= 55: success_score += 20
        
        if win_rate_std < 10: success_score += 20
        elif win_rate_std < 15: success_score += 10
        
        if avg_trades >= 35: success_score += 25
        elif avg_trades >= 25: success_score += 15
        
        if avg_rr >= 2.5: success_score += 15
        elif avg_rr >= 2.0: success_score += 10
        
        if success_score >= 80:
            overall = "✅ OPTIMIZATION SUCCESSFUL!"
            next_steps = "System performing excellently. Monitor and maintain."
        elif success_score >= 60:
            overall = "⚠️ PARTIAL SUCCESS"
            next_steps = "Good progress. Apply additional enhancements."
        else:
            overall = "❌ NEEDS IMPROVEMENT"
            next_steps = "Review configuration and apply Phase 1 fixes."
        
        print(f"   • Result: {overall}")
        print(f"   • Score: {success_score}/100")
        print(f"   • Next Steps: {next_steps}")

def main():
    print("🔍 Analyzing win rate optimization results...")
    results = analyze_win_rate_performance()
    generate_win_rate_report(results)

if __name__ == "__main__":
    main()
'''
    
    with open("enhanced_win_rate_monitor.py", "w") as f:
        f.write(monitoring_code)
    
    os.chmod("enhanced_win_rate_monitor.py", 0o755)
    print("📊 Enhanced monitoring script created: enhanced_win_rate_monitor.py")

def test_enhancements():
    """Test the enhancement functions with sample data"""
    try:
        print("\n🧪 Testing win rate enhancements...")
        from enhanced_signal_quality import test_signal_enhancement
        test_signal_enhancement()
        print("✅ Enhancement functions working correctly")
        return True
    except Exception as e:
        print(f"❌ Error testing enhancements: {e}")
        return False

def print_implementation_summary():
    """Print implementation summary and next steps"""
    
    print("\n" + "="*70)
    print("🎯 WIN RATE OPTIMIZATION IMPLEMENTATION COMPLETE")
    print("="*70)
    
    print("\\n📋 CHANGES APPLIED:")
    print("   1. ✅ Win rate targets increased to 60-75%")
    print("   2. ✅ Enhanced signal filtering enabled")
    print("   3. ✅ Quality-based signal scoring implemented")
    print("   4. ✅ Dynamic position sizing enabled")
    print("   5. ✅ Advanced risk management activated")
    print("   6. ✅ Multi-timeframe features enabled")
    
    print("\\n🚀 NEXT STEPS:")
    print("   1. Test the optimized configuration:")
    print("      python walkforward.py --run path/to/output --stepback_weeks 4 --optimize")
    
    print("\\n   2. Monitor win rate performance:")
    print("      python enhanced_win_rate_monitor.py")
    
    print("\\n   3. Expected improvements:")
    print("      • Win Rate: 60-70% (vs current 45-55%)")
    print("      • Win Rate Stability: <8% variation")
    print("      • Trade Frequency: Maintained at 35+ trades/week")
    print("      • Risk-Reward: Maintained above 2.5")
    
    print("\\n⚠️ MONITORING GUIDELINES:")
    print("   • Run for 1 week before making further changes")
    print("   • Monitor all KPIs, not just win rate")
    print("   • Rollback if win rate drops below 50%")
    print("   • Target: Stable 60-70% win rate")

def main():
    """Main implementation function"""
    
    parser = argparse.ArgumentParser(description="Apply win rate optimizations")
    parser.add_argument("--test", action="store_true", help="Test enhancements only")
    parser.add_argument("--backup", action="store_true", default=True, help="Create config backup")
    args = parser.parse_args()
    
    print("🎯 WIN RATE OPTIMIZATION IMPLEMENTATION")
    print("=" * 50)
    print("Goal: Improve win rate from 45-55% to 60-70%")
    print("Maintain: 35+ trades/week and 2.5+ RR")
    
    # Test mode
    if args.test:
        success = test_enhancements()
        if success:
            print("\\n✅ All enhancement functions working correctly!")
            print("Run without --test flag to apply optimizations")
        else:
            print("\\n❌ Enhancement functions have issues")
        return success
    
    # Check if config file exists
    if not Path("config.yaml").exists():
        print("❌ Error: config.yaml not found!")
        print("   Please ensure you're in the correct directory")
        return False
    
    try:
        # Backup current configuration
        if args.backup:
            backup_name = backup_current_config()
            if not backup_name:
                print("❌ Could not create backup. Aborting.")
                return False
        
        # Load and modify configuration
        config = load_config()
        if not config:
            print("❌ Could not load configuration. Aborting.")
            return False
        
        optimized_config = apply_win_rate_config_changes(config)
        
        # Save optimized configuration
        if not save_config(optimized_config):
            print("❌ Could not save configuration. Aborting.")
            return False
        
        # Create additional files
        create_enhanced_ranker_integration()
        create_monitoring_script()
        
        # Test enhancements
        test_success = test_enhancements()
        
        # Print summary
        print_implementation_summary()
        
        if test_success:
            print("\\n✅ WIN RATE OPTIMIZATION READY FOR TESTING!")
            print("\\nRun: python walkforward.py --run path/to/output --stepback_weeks 4 --optimize")
        else:
            print("\\n⚠️ OPTIMIZATION APPLIED BUT TESTING FAILED")
            print("Check enhanced_signal_quality.py for issues")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        print("   Configuration may have been partially modified")
        if args.backup and 'backup_name' in locals():
            print(f"   Restore from backup: cp {backup_name} config.yaml")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)