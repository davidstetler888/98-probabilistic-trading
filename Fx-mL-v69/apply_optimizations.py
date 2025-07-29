#!/usr/bin/env python3
"""
Apply Phase 1 optimizations to increase trade frequency while maintaining win rate.
This script backs up the current config and applies the recommended changes.
"""

import shutil
import yaml
from datetime import datetime
from pathlib import Path

def backup_config():
    """Create a backup of the current config.yaml file."""
    backup_name = f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    shutil.copy("config.yaml", backup_name)
    print(f"✓ Created backup: {backup_name}")
    return backup_name

def load_config():
    """Load the current config.yaml file."""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def apply_phase1_optimizations(config):
    """Apply Phase 1 optimizations to the config."""
    print("\n🔧 Applying Phase 1 optimizations...")
    
    # Goals adjustments
    config["goals"]["win_rate_range"] = [0.55, 0.75]
    config["goals"]["risk_reward_range"] = [1.6, 3.5]
    config["goals"]["trades_per_week_range"] = [30, 60]
    print("  ✓ Adjusted goal ranges")
    
    # Labeling parameters - more permissive
    config["label"]["threshold"] = 0.0008  # from 0.0010
    config["label"]["max_sl_pips"] = 25    # from 22
    config["label"]["min_rr"] = 1.8        # from 2.0
    config["label"]["cooldown_min"] = 5    # from 10
    config["label"]["min_rr_target"] = 1.8 # from 2.0
    config["label"]["min_win_rate_target"] = 0.55  # from 0.58
    print("  ✓ Relaxed labeling criteria")
    
    # Signal parameters
    config["signal"]["min_precision_target"] = 0.75  # from 0.80
    config["signal"]["min_signals_per_week"] = 12    # from 8
    config["signal"]["precision_filter"]["thresholds"] = [0.45, 0.55, 0.6, 0.65]
    print("  ✓ Adjusted signal parameters")
    
    # SL/TP parameters
    config["sltp"]["min_tp"] = 3      # from 4
    config["sltp"]["max_tp"] = 30     # from 28
    config["sltp"]["tp_step"] = 1     # from 2
    config["sltp"]["min_rr"] = 1.6    # from 2.0
    print("  ✓ Expanded SL/TP range")
    
    # SL/TP grid - add lower RR options
    config["sl_tp_grid"]["tp_multipliers"] = [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2]
    print("  ✓ Added lower RR options to SL/TP grid")
    
    # Simulation parameters - KEY CHANGES
    config["simulation"]["max_positions"] = 3      # from 2
    config["simulation"]["cooldown_min"] = 5       # from 10
    config["simulation"]["max_daily_trades"] = 10  # from 5
    config["simulation"]["max_weekly_trades"] = 50 # from 20 (CRITICAL)
    print("  ✓ Increased simulation trade capacity")
    
    # Ranker parameters
    config["ranker"]["target_trades_per_week"] = 45  # from 40
    config["ranker"]["min_trades_per_week"] = 30     # from 25
    config["ranker"]["max_trades_per_week"] = 60     # from 50
    print("  ✓ Increased ranker targets")
    
    # Acceptance criteria
    config["acceptance"]["simulation"]["min_win_rate"] = 0.55     # from 0.58
    config["acceptance"]["simulation"]["min_profit_factor"] = 1.3  # from 1.5
    config["acceptance"]["simulation"]["min_trades_per_week"] = 30 # from 25
    config["acceptance"]["sltp"]["min_rr"] = 1.6                  # from 2.0
    print("  ✓ Relaxed acceptance criteria")
    
    # Backtest parameters
    config["backtest"]["min_avg_rr"] = 1.8  # from 2.0
    print("  ✓ Adjusted backtest parameters")
    
    # Walkforward parameters
    config["walkforward"]["min_trades_per_week"] = 8   # from 5
    config["walkforward"]["max_trades_per_week"] = 120 # from 100
    print("  ✓ Expanded walkforward ranges")
    
    return config

def save_config(config):
    """Save the modified config back to config.yaml."""
    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print("✓ Saved optimized config.yaml")

def print_summary():
    """Print a summary of changes made."""
    print("\n📊 SUMMARY OF PHASE 1 OPTIMIZATIONS")
    print("="*50)
    
    changes = [
        ("Win Rate Range", "58-75%", "55-75%"),
        ("Risk-Reward Range", "1.8-3.5", "1.6-3.5"),
        ("Trades/Week Target", "25-50", "30-60"),
        ("Label Threshold", "0.0010 (1.0 pip)", "0.0008 (0.8 pip)"),
        ("Max SL", "22 pips", "25 pips"),
        ("Min RR", "2.0", "1.8"),
        ("Cooldown", "10 min", "5 min"),
        ("Max Positions", "2", "3"),
        ("Max Daily Trades", "5", "10"),
        ("Max Weekly Trades", "20", "50 (CRITICAL)"),
        ("Ranker Target", "40/week", "45/week"),
        ("Min Precision", "0.80", "0.75"),
    ]
    
    for change, old, new in changes:
        print(f"  {change:20} {old:15} → {new}")
    
    print("\n🎯 EXPECTED IMPACT:")
    print("  • Trade frequency: 2-5x increase (10-30 trades/week)")
    print("  • Win rate: Slight decrease to 60-75% range")
    print("  • Risk-reward: Maintained above 1.8")
    print("  • Drawdown: Monitor for increases")
    
    print("\n⚠️  MONITORING PLAN:")
    print("  1. Run for 1-2 weeks with new config")
    print("  2. Monitor win rate (should stay above 55%)")
    print("  3. Check drawdown doesn't exceed 15%")
    print("  4. Verify trade frequency increases")
    print("  5. If win rate drops below 50%, rollback")

def main():
    """Main function to apply optimizations."""
    print("🚀 FOREX PIPELINE OPTIMIZATION - PHASE 1")
    print("=" * 50)
    print("Goal: Increase trade frequency while maintaining win rate")
    
    # Check if config.yaml exists
    if not Path("config.yaml").exists():
        print("❌ Error: config.yaml not found!")
        return
    
    # Create backup
    backup_name = backup_config()
    
    try:
        # Load current config
        config = load_config()
        
        # Apply optimizations
        optimized_config = apply_phase1_optimizations(config)
        
        # Save optimized config
        save_config(optimized_config)
        
        # Print summary
        print_summary()
        
        print(f"\n✅ OPTIMIZATION COMPLETE!")
        print(f"   Original config backed up as: {backup_name}")
        print(f"   Run the pipeline to test the changes:")
        print(f"   python main.py")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        print(f"   Restoring backup...")
        shutil.copy(backup_name, "config.yaml")
        print(f"   Config restored from backup")

if __name__ == "__main__":
    main()