"""Performance Monitoring and Auto-Adjustment Module

This module monitors the performance of the trading system and automatically suggests
parameter adjustments to optimize trade frequency while maintaining profitability.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

from config import config


class PerformanceMonitor:
    """Monitor performance and suggest adjustments to optimize trade frequency"""
    
    def __init__(self, target_trades_per_week: float = 40):
        self.target_trades_per_week = target_trades_per_week
        self.performance_history = []
        self.adjustment_threshold = 0.2  # 20% deviation from target
        self.min_win_rate = 0.55
        self.min_profit_factor = 1.3
        self.max_drawdown = 0.15
        
    def log_performance(self, metrics: Dict, timestamp: datetime = None):
        """Log performance metrics"""
        if timestamp is None:
            timestamp = datetime.now()
            
        performance_record = {
            'timestamp': timestamp.isoformat(),
            'trades_per_week': metrics.get('trades_per_wk', 0),
            'win_rate': metrics.get('win_rate', 0),
            'profit_factor': metrics.get('profit_factor', 0),
            'max_drawdown': metrics.get('max_dd', 0),
            'avg_rr': metrics.get('avg_rr', 0),
            'total_trades': metrics.get('total_trades', 0),
            'sharpe': metrics.get('sharpe', 0),
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only last 50 records
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]
    
    def analyze_performance(self) -> Dict:
        """Analyze current performance vs targets"""
        if not self.performance_history:
            return {'status': 'no_data', 'message': 'No performance data available'}
        
        latest = self.performance_history[-1]
        
        # Calculate recent averages (last 10 records)
        recent_records = self.performance_history[-10:]
        avg_trades_per_week = sum(r['trades_per_week'] for r in recent_records) / len(recent_records)
        avg_win_rate = sum(r['win_rate'] for r in recent_records) / len(recent_records)
        avg_profit_factor = sum(r['profit_factor'] for r in recent_records) / len(recent_records)
        avg_max_drawdown = sum(r['max_drawdown'] for r in recent_records) / len(recent_records)
        
        # Calculate deviations
        trades_deviation = abs(avg_trades_per_week - self.target_trades_per_week) / self.target_trades_per_week
        
        analysis = {
            'current_trades_per_week': avg_trades_per_week,
            'target_trades_per_week': self.target_trades_per_week,
            'trades_deviation': trades_deviation,
            'win_rate': avg_win_rate,
            'profit_factor': avg_profit_factor,
            'max_drawdown': avg_max_drawdown,
            'needs_adjustment': trades_deviation > self.adjustment_threshold,
            'performance_quality': self._assess_performance_quality(avg_win_rate, avg_profit_factor, avg_max_drawdown),
        }
        
        return analysis
    
    def _assess_performance_quality(self, win_rate: float, profit_factor: float, max_drawdown: float) -> str:
        """Assess overall performance quality"""
        if (win_rate >= self.min_win_rate and 
            profit_factor >= self.min_profit_factor and 
            max_drawdown <= self.max_drawdown):
            return 'excellent'
        elif (win_rate >= self.min_win_rate * 0.9 and 
              profit_factor >= self.min_profit_factor * 0.8 and 
              max_drawdown <= self.max_drawdown * 1.2):
            return 'good'
        elif (win_rate >= self.min_win_rate * 0.8 and 
              profit_factor >= self.min_profit_factor * 0.6):
            return 'acceptable'
        else:
            return 'poor'
    
    def suggest_adjustments(self) -> List[Dict]:
        """Suggest parameter adjustments based on performance analysis"""
        analysis = self.analyze_performance()
        
        if not analysis.get('needs_adjustment', False):
            return []
        
        suggestions = []
        current_trades = analysis['current_trades_per_week']
        target_trades = analysis['target_trades_per_week']
        performance_quality = analysis['performance_quality']
        
        # If we're getting too few trades
        if current_trades < target_trades:
            adjustment_factor = min(target_trades / current_trades, 2.0)  # Cap at 2x
            
            # Suggest increasing trade capacity
            suggestions.append({
                'parameter': 'simulation.max_weekly_trades',
                'action': 'increase',
                'multiplier': adjustment_factor,
                'reason': f'Increase trade capacity to reach {target_trades:.1f} trades/week from {current_trades:.1f}',
                'priority': 'high'
            })
            
            # Suggest lowering thresholds if performance quality is good
            if performance_quality in ['excellent', 'good']:
                suggestions.append({
                    'parameter': 'label.threshold',
                    'action': 'decrease',
                    'multiplier': 0.9,
                    'reason': 'Lower entry threshold to generate more signals',
                    'priority': 'medium'
                })
                
                suggestions.append({
                    'parameter': 'ranker.min_trades_per_week',
                    'action': 'increase',
                    'multiplier': 1.2,
                    'reason': 'Increase minimum trades to be more aggressive',
                    'priority': 'medium'
                })
        
        # If we're getting too many trades
        elif current_trades > target_trades:
            adjustment_factor = max(target_trades / current_trades, 0.5)  # Cap at 0.5x
            
            suggestions.append({
                'parameter': 'label.threshold',
                'action': 'increase',
                'multiplier': 1.1,
                'reason': f'Raise entry threshold to reduce trades from {current_trades:.1f} to {target_trades:.1f}',
                'priority': 'medium'
            })
            
            suggestions.append({
                'parameter': 'ranker.target_trades_per_week',
                'action': 'decrease',
                'multiplier': adjustment_factor,
                'reason': 'Reduce target trades to match current capacity',
                'priority': 'low'
            })
        
        # Performance-based adjustments
        if performance_quality == 'poor':
            suggestions.append({
                'parameter': 'label.threshold',
                'action': 'increase',
                'multiplier': 1.2,
                'reason': 'Increase threshold to improve signal quality',
                'priority': 'high'
            })
            
            suggestions.append({
                'parameter': 'simulation.max_positions',
                'action': 'decrease',
                'multiplier': 0.8,
                'reason': 'Reduce position count to improve risk management',
                'priority': 'medium'
            })
        
        return suggestions
    
    def apply_suggestions(self, suggestions: List[Dict], config_path: str = None) -> Dict:
        """Apply parameter adjustments to configuration"""
        if config_path is None:
            config_path = "config.yaml"
        
        # Load current config
        current_config = dict(config._config) if hasattr(config, '_config') else {}
        
        applied_changes = []
        
        for suggestion in suggestions:
            param_path = suggestion['parameter']
            action = suggestion['action']
            multiplier = suggestion['multiplier']
            
            # Parse parameter path
            keys = param_path.split('.')
            
            # Navigate to the parameter
            current_value = current_config
            for key in keys[:-1]:
                if key not in current_value:
                    current_value[key] = {}
                current_value = current_value[key]
            
            final_key = keys[-1]
            old_value = current_value.get(final_key, 0)
            
            # Calculate new value
            if action == 'increase':
                new_value = old_value * multiplier
            elif action == 'decrease':
                new_value = old_value / multiplier
            else:
                new_value = old_value
            
            # Apply bounds checking
            new_value = self._apply_bounds(param_path, new_value)
            
            # Update config
            current_value[final_key] = new_value
            
            applied_changes.append({
                'parameter': param_path,
                'old_value': old_value,
                'new_value': new_value,
                'reason': suggestion['reason']
            })
        
        return {
            'applied_changes': applied_changes,
            'config': current_config
        }
    
    def _apply_bounds(self, param_path: str, value: float) -> float:
        """Apply reasonable bounds to parameter values"""
        bounds = {
            'simulation.max_weekly_trades': (10, 100),
            'simulation.max_daily_trades': (2, 20),
            'simulation.max_positions': (1, 5),
            'simulation.cooldown_min': (1, 30),
            'label.threshold': (0.0005, 0.002),
            'ranker.target_trades_per_week': (10, 80),
            'ranker.min_trades_per_week': (5, 60),
            'ranker.max_trades_per_week': (20, 100),
        }
        
        if param_path in bounds:
            min_val, max_val = bounds[param_path]
            return max(min_val, min(value, max_val))
        
        return value
    
    def save_performance_history(self, file_path: str):
        """Save performance history to file"""
        with open(file_path, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
    
    def load_performance_history(self, file_path: str):
        """Load performance history from file"""
        if Path(file_path).exists():
            with open(file_path, 'r') as f:
                self.performance_history = json.load(f)
    
    def get_performance_summary(self) -> Dict:
        """Get a summary of recent performance"""
        if not self.performance_history:
            return {'status': 'no_data'}
        
        recent = self.performance_history[-10:]
        
        return {
            'total_records': len(self.performance_history),
            'recent_avg_trades_per_week': sum(r['trades_per_week'] for r in recent) / len(recent) if recent else 0,
            'recent_avg_win_rate': sum(r['win_rate'] for r in recent) / len(recent) if recent else 0,
            'recent_avg_profit_factor': sum(r['profit_factor'] for r in recent) / len(recent) if recent else 0,
            'recent_avg_max_drawdown': sum(r['max_drawdown'] for r in recent) / len(recent) if recent else 0,
            'target_trades_per_week': self.target_trades_per_week,
            'performance_trend': self._calculate_trend(),
        }
    
    def _calculate_trend(self) -> str:
        """Calculate performance trend"""
        if len(self.performance_history) < 5:
            return 'insufficient_data'
        
        recent_trades = [r['trades_per_week'] for r in self.performance_history[-5:]]
        
        if len(recent_trades) < 2:
            return 'insufficient_data'
        
        # Simple trend calculation: compare first half to second half
        mid = len(recent_trades) // 2
        first_half_avg = sum(recent_trades[:mid]) / mid if mid > 0 else 0
        second_half_avg = sum(recent_trades[mid:]) / (len(recent_trades) - mid)
        
        diff = second_half_avg - first_half_avg
        
        if diff > 1.0:
            return 'improving'
        elif diff < -1.0:
            return 'declining'
        else:
            return 'stable'


def create_performance_report(monitor: PerformanceMonitor, output_path: str = None) -> str:
    """Create a detailed performance report"""
    analysis = monitor.analyze_performance()
    suggestions = monitor.suggest_adjustments()
    summary = monitor.get_performance_summary()
    
    report = f"""
# Trading System Performance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Summary
- Current Trades/Week: {analysis.get('current_trades_per_week', 0):.1f}
- Target Trades/Week: {analysis.get('target_trades_per_week', 0):.1f}
- Deviation: {analysis.get('trades_deviation', 0)*100:.1f}%
- Win Rate: {analysis.get('win_rate', 0)*100:.1f}%
- Profit Factor: {analysis.get('profit_factor', 0):.2f}
- Max Drawdown: {analysis.get('max_drawdown', 0)*100:.1f}%
- Performance Quality: {analysis.get('performance_quality', 'unknown').upper()}

## Trend Analysis
- Performance Trend: {summary.get('performance_trend', 'unknown').upper()}
- Total Records: {summary.get('total_records', 0)}

## Adjustment Recommendations
"""
    
    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            report += f"""
### {i}. {suggestion['parameter']} ({suggestion['priority'].upper()} Priority)
- Action: {suggestion['action'].upper()}
- Multiplier: {suggestion['multiplier']:.2f}
- Reason: {suggestion['reason']}
"""
    else:
        report += "\nNo adjustments recommended at this time.\n"
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report


if __name__ == "__main__":
    # Example usage
    monitor = PerformanceMonitor(target_trades_per_week=40)
    
    # Simulate some performance data
    sample_metrics = {
        'trades_per_wk': 15.5,
        'win_rate': 0.62,
        'profit_factor': 1.8,
        'max_dd': 0.08,
        'avg_rr': 2.3,
        'total_trades': 45,
        'sharpe': 1.2
    }
    
    monitor.log_performance(sample_metrics)
    
    # Get suggestions
    suggestions = monitor.suggest_adjustments()
    
    # Generate report
    report = create_performance_report(monitor)
    print(report)