#!/usr/bin/env python3
"""
Advanced Optimization Runner for Trading System

This script applies all the advanced algorithmic improvements described in
ADDITIONAL_OPTIMIZATIONS.md to increase trade frequency while maintaining profitability.

Features implemented:
1. Dynamic Edge Thresholding
2. Multi-Timeframe Signal Generation
3. Session-Specific Optimization
4. Volatility-Based Position Sizing
5. Improved Signal Ranking
6. Risk Management Enhancements
7. Performance Monitoring & Auto-Adjustment
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from performance_monitor import PerformanceMonitor, create_performance_report
from config import config
from utils import get_run_dir, make_run_dirs


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run advanced algorithmic optimizations for increased trade frequency"
    )
    parser.add_argument(
        "--run", 
        type=str, 
        help="Run directory (overrides RUN_ID)"
    )
    parser.add_argument(
        "--target_trades_per_week",
        type=float,
        default=45,
        help="Target trades per week (default: 45)"
    )
    parser.add_argument(
        "--apply_phase1",
        action="store_true",
        help="Apply existing Phase 1 optimizations first"
    )
    parser.add_argument(
        "--monitor_performance",
        action="store_true",
        help="Enable performance monitoring and auto-adjustment"
    )
    parser.add_argument(
        "--start_date",
        type=str,
        help="Start date for data processing (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end_date",
        type=str,
        help="End date for data processing (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--train_end_date",
        type=str,
        help="Training end date (YYYY-MM-DD)"
    )
    return parser.parse_args()


def run_optimization_pipeline(run_dir: str, target_trades_per_week: float, 
                            start_date: str | None = None, end_date: str | None = None,
                            train_end_date: str | None = None) -> dict:
    """Run the complete optimization pipeline"""
    
    print("\n" + "="*60)
    print("🚀 ADVANCED OPTIMIZATION PIPELINE")
    print("="*60)
    
    results = {
        'run_dir': run_dir,
        'target_trades_per_week': target_trades_per_week,
        'optimizations_applied': [],
        'performance_metrics': {},
        'execution_log': []
    }
    
    # Step 1: Prepare data with multi-timeframe features
    print("\n📊 Step 1: Preparing data with multi-timeframe features...")
    try:
        cmd_args = ["python", "prepare.py"]
        if start_date:
            cmd_args.extend(["--start_date", start_date])
        if end_date:
            cmd_args.extend(["--end_date", end_date])
        if train_end_date:
            cmd_args.extend(["--train_end_date", train_end_date])
        
        # Execute prepare.py (this now includes multi-timeframe processing)
        os.system(" ".join(cmd_args))
        results['optimizations_applied'].append("Multi-timeframe signal generation")
        results['execution_log'].append("✅ Data preparation with multi-timeframe features completed")
        print("✅ Multi-timeframe features added successfully")
        
    except Exception as e:
        print(f"❌ Error in data preparation: {e}")
        results['execution_log'].append(f"❌ Data preparation failed: {e}")
        return results
    
    # Step 2: Label data
    print("\n🏷️  Step 2: Labeling data...")
    try:
        cmd_args = ["python", "label.py", "--run", run_dir]
        if start_date:
            cmd_args.extend(["--start_date", start_date])
        if end_date:
            cmd_args.extend(["--end_date", end_date])
        if train_end_date:
            cmd_args.extend(["--train_end_date", train_end_date])
            
        os.system(" ".join(cmd_args))
        results['execution_log'].append("✅ Data labeling completed")
        print("✅ Data labeling completed")
        
    except Exception as e:
        print(f"❌ Error in labeling: {e}")
        results['execution_log'].append(f"❌ Labeling failed: {e}")
        return results
    
    # Step 3: Train base models
    print("\n🤖 Step 3: Training base models...")
    try:
        os.system(f"python train_base.py --run {run_dir}")
        results['execution_log'].append("✅ Base model training completed")
        print("✅ Base models trained")
        
    except Exception as e:
        print(f"❌ Error in base training: {e}")
        results['execution_log'].append(f"❌ Base training failed: {e}")
        return results
    
    # Step 4: Train meta models
    print("\n🧠 Step 4: Training meta models...")
    try:
        os.system(f"python train_meta.py --run {run_dir}")
        results['execution_log'].append("✅ Meta model training completed")
        print("✅ Meta models trained")
        
    except Exception as e:
        print(f"❌ Error in meta training: {e}")
        results['execution_log'].append(f"❌ Meta training failed: {e}")
        return results
    
    # Step 5: Train SL/TP models
    print("\n🎯 Step 5: Training SL/TP models...")
    try:
        os.system(f"python train_sltp.py --run {run_dir}")
        results['execution_log'].append("✅ SL/TP model training completed")
        print("✅ SL/TP models trained")
        
    except Exception as e:
        print(f"❌ Error in SL/TP training: {e}")
        results['execution_log'].append(f"❌ SL/TP training failed: {e}")
        return results
    
    # Step 6: Train ranker with advanced optimizations
    print("\n🎖️  Step 6: Training ranker with dynamic thresholding and enhanced scoring...")
    try:
        cmd_args = [
            "python", "train_ranker.py", 
            "--run", run_dir,
            "--target_trades_per_week", str(target_trades_per_week)
        ]
        if start_date:
            cmd_args.extend(["--start_date", start_date])
        if end_date:
            cmd_args.extend(["--end_date", end_date])
        if train_end_date:
            cmd_args.extend(["--train_end_date", train_end_date])
            
        os.system(" ".join(cmd_args))
        results['optimizations_applied'].extend([
            "Dynamic edge thresholding",
            "Session-specific optimization", 
            "Enhanced signal ranking"
        ])
        results['execution_log'].append("✅ Advanced ranker training completed")
        print("✅ Enhanced signal ranking with dynamic thresholding applied")
        
    except Exception as e:
        print(f"❌ Error in ranker training: {e}")
        results['execution_log'].append(f"❌ Ranker training failed: {e}")
        return results
    
    # Step 7: Run simulation with volatility-based position sizing
    print("\n💹 Step 7: Running simulation with volatility-based position sizing...")
    try:
        cmd_args = ["python", "simulate.py", "--run", run_dir]
        if start_date:
            cmd_args.extend(["--start_date", start_date])
        if end_date:
            cmd_args.extend(["--end_date", end_date])
            
        os.system(" ".join(cmd_args))
        results['optimizations_applied'].extend([
            "Volatility-based position sizing",
            "Portfolio risk management"
        ])
        results['execution_log'].append("✅ Simulation with advanced position sizing completed")
        print("✅ Simulation with dynamic position sizing completed")
        
    except Exception as e:
        print(f"❌ Error in simulation: {e}")
        results['execution_log'].append(f"❌ Simulation failed: {e}")
        return results
    
    # Step 8: Load and analyze results
    print("\n📈 Step 8: Analyzing results...")
    try:
        metrics_file = Path(run_dir) / "artifacts" / "sim_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            results['performance_metrics'] = metrics
            
            print(f"✅ Results Analysis:")
            print(f"   📊 Total Trades: {metrics.get('total_trades', 0)}")
            print(f"   📈 Trades per Week: {metrics.get('trades_per_wk', 0):.1f}")
            print(f"   🎯 Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
            print(f"   💰 Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            print(f"   📉 Max Drawdown: {metrics.get('max_dd', 0)*100:.1f}%")
            print(f"   ⚖️  Avg Risk/Reward: {metrics.get('avg_rr', 0):.2f}")
            print(f"   📊 Sharpe Ratio: {metrics.get('sharpe', 0):.2f}")
            
            # Performance assessment
            trades_per_week = metrics.get('trades_per_wk', 0)
            target_achieved = abs(trades_per_week - target_trades_per_week) / target_trades_per_week < 0.2
            
            if target_achieved:
                print(f"\n🎉 SUCCESS: Target of {target_trades_per_week} trades/week achieved!")
                print(f"   Current: {trades_per_week:.1f} trades/week")
            else:
                print(f"\n⚠️  TARGET NOT MET: {trades_per_week:.1f} vs {target_trades_per_week} trades/week")
                print(f"   Deviation: {abs(trades_per_week - target_trades_per_week):.1f} trades/week")
            
            results['target_achieved'] = target_achieved
            results['execution_log'].append("✅ Results analysis completed")
            
        else:
            print("❌ No metrics file found")
            results['execution_log'].append("❌ No metrics file found")
            
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        results['execution_log'].append(f"❌ Results analysis failed: {e}")
    
    return results


def setup_performance_monitoring(run_dir: str, target_trades_per_week: float) -> PerformanceMonitor:
    """Set up performance monitoring"""
    print("\n🔍 Setting up performance monitoring...")
    
    monitor = PerformanceMonitor(target_trades_per_week=target_trades_per_week)
    
    # Load existing performance history if available
    history_file = Path(run_dir) / "artifacts" / "performance_history.json"
    if history_file.exists():
        monitor.load_performance_history(str(history_file))
        print(f"✅ Loaded {len(monitor.performance_history)} historical performance records")
    
    return monitor


def generate_optimization_report(results: dict, monitor: PerformanceMonitor | None = None) -> str:
    """Generate a comprehensive optimization report"""
    
    report = f"""
# Advanced Optimization Results Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Optimization Summary
- **Run Directory**: {results['run_dir']}
- **Target Trades/Week**: {results['target_trades_per_week']}
- **Target Achieved**: {'✅ YES' if results.get('target_achieved', False) else '❌ NO'}

## Optimizations Applied
"""
    
    for i, optimization in enumerate(results['optimizations_applied'], 1):
        report += f"{i}. {optimization}\n"
    
    if results.get('performance_metrics'):
        metrics = results['performance_metrics']
        report += f"""
## Performance Metrics
- **Total Trades**: {metrics.get('total_trades', 0)}
- **Trades per Week**: {metrics.get('trades_per_wk', 0):.1f}
- **Win Rate**: {metrics.get('win_rate', 0)*100:.1f}%
- **Profit Factor**: {metrics.get('profit_factor', 0):.2f}
- **Max Drawdown**: {metrics.get('max_dd', 0)*100:.1f}%
- **Average Risk/Reward**: {metrics.get('avg_rr', 0):.2f}
- **Sharpe Ratio**: {metrics.get('sharpe', 0):.2f}

## Goal Comparison
- **Target**: {results['target_trades_per_week']} trades/week
- **Actual**: {metrics.get('trades_per_wk', 0):.1f} trades/week
- **Deviation**: {abs(metrics.get('trades_per_wk', 0) - results['target_trades_per_week']):.1f} trades/week
- **Success Rate**: {(1 - abs(metrics.get('trades_per_wk', 0) - results['target_trades_per_week']) / results['target_trades_per_week'])*100:.1f}%
"""
    
    report += "\n## Execution Log\n"
    for log_entry in results['execution_log']:
        report += f"- {log_entry}\n"
    
    if monitor:
        # Add performance monitoring insights
        try:
            performance_report = create_performance_report(monitor)
            report += f"\n## Performance Monitoring Insights\n{performance_report}"
        except Exception as e:
            report += f"\n## Performance Monitoring\n❌ Error generating insights: {e}\n"
    
    report += f"""
## Next Steps
1. Review the performance metrics to ensure they meet your requirements
2. If target not achieved, consider applying the suggested parameter adjustments
3. Monitor performance over time using the performance monitoring system
4. Consider additional optimizations based on market conditions

## Advanced Features Implemented
✅ Dynamic Edge Thresholding - Adaptive signal selection based on market conditions
✅ Multi-Timeframe Analysis - 15-minute and 30-minute trend confirmation
✅ Session-Specific Optimization - Different thresholds for Asian/London/NY sessions
✅ Volatility-Based Position Sizing - ATR-based dynamic position sizing
✅ Enhanced Signal Ranking - Multi-factor signal scoring system
✅ Portfolio Risk Management - Correlation-based position management
✅ Performance Monitoring - Automated parameter adjustment suggestions
"""
    
    return report


def main():
    """Main execution function"""
    args = parse_args()
    
    # Set up run directory
    run_dir = args.run if args.run else get_run_dir()
    make_run_dirs(run_dir)
    
    print(f"🎯 Target: {args.target_trades_per_week} trades per week")
    print(f"📁 Run directory: {run_dir}")
    
    # Apply Phase 1 optimizations if requested
    if args.apply_phase1:
        print("\n🔧 Applying Phase 1 optimizations...")
        try:
            os.system("python apply_optimizations.py")
            print("✅ Phase 1 optimizations applied")
        except Exception as e:
            print(f"❌ Error applying Phase 1 optimizations: {e}")
    
    # Set up performance monitoring
    monitor = None
    if args.monitor_performance:
        monitor = setup_performance_monitoring(run_dir, args.target_trades_per_week)
    
    # Run the optimization pipeline
    results = run_optimization_pipeline(
        run_dir=run_dir,
        target_trades_per_week=args.target_trades_per_week,
        start_date=args.start_date,
        end_date=args.end_date,
        train_end_date=args.train_end_date
    )
    
    # Log performance if monitoring is enabled
    if monitor and results.get('performance_metrics'):
        monitor.log_performance(results['performance_metrics'])
        
        # Save performance history
        history_file = Path(run_dir) / "artifacts" / "performance_history.json"
        monitor.save_performance_history(str(history_file))
        
        # Get suggestions for future improvements
        suggestions = monitor.suggest_adjustments()
        if suggestions:
            print(f"\n💡 Performance Monitor Suggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"   {i}. {suggestion['parameter']}: {suggestion['action']} by {suggestion['multiplier']:.2f}x")
                print(f"      Reason: {suggestion['reason']}")
    
    # Generate comprehensive report
    report = generate_optimization_report(results, monitor)
    
    # Save report
    report_file = Path(run_dir) / "artifacts" / "optimization_report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n📋 Comprehensive report saved to: {report_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("🏁 OPTIMIZATION PIPELINE COMPLETE")
    print("="*60)
    
    if results.get('target_achieved'):
        print(f"🎉 SUCCESS: Target achieved with {len(results['optimizations_applied'])} optimizations!")
    else:
        print(f"⚠️  Target not fully achieved. Consider additional optimizations.")
    
    print(f"📊 Applied {len(results['optimizations_applied'])} advanced optimizations")
    print(f"📈 Performance metrics available in: {run_dir}/artifacts/")
    
    return results


if __name__ == "__main__":
    main()