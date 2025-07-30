#!/usr/bin/env python3
"""
Debug Log Analysis Script
Analyze debug logs from live trading sessions
"""

import json
import pandas as pd
import os
from datetime import datetime
import glob

def analyze_debug_session(session_dir):
    """Analyze a debug session directory."""
    print(f"🔍 Analyzing debug session: {session_dir}")
    print("=" * 60)
    
    # Load summary
    summary_file = os.path.join(session_dir, 'debug_summary.json')
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        print(f"📊 Session ID: {summary.get('session_id', 'Unknown')}")
        print(f"📅 Session Start: {summary.get('session_start', 'Unknown')}")
    
    # Analyze each log file
    log_files = {
        'system_status': 'System Status',
        'signal_generation': 'Signal Generation',
        'expected_value': 'Expected Value',
        'trade_decisions': 'Trade Decisions',
        'ensemble_predictions': 'Ensemble Predictions',
        'probabilistic_labels': 'Probabilistic Labels',
        'errors': 'Errors',
        'performance': 'Performance'
    }
    
    for log_file, description in log_files.items():
        file_path = os.path.join(session_dir, f'{log_file}.log')
        if os.path.exists(file_path):
            analyze_log_file(file_path, description)
        else:
            print(f"❌ {description} log not found")
    
    print("\n" + "=" * 60)

def analyze_log_file(file_path, description):
    """Analyze a specific log file."""
    print(f"\n📋 {description} Analysis:")
    print("-" * 40)
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            print("   Empty log file")
            return
        
        # Parse JSON lines
        entries = []
        for line in lines:
            try:
                entry = json.loads(line.strip())
                entries.append(entry)
            except json.JSONDecodeError:
                continue
        
        if not entries:
            print("   No valid JSON entries found")
            return
        
        # Analyze based on log type
        if 'system_status' in file_path:
            analyze_system_status(entries)
        elif 'trade_decisions' in file_path:
            analyze_trade_decisions(entries)
        elif 'expected_value' in file_path:
            analyze_expected_value(entries)
        elif 'errors' in file_path:
            analyze_errors(entries)
        elif 'ensemble_predictions' in file_path:
            analyze_ensemble_predictions(entries)
        else:
            print(f"   {len(entries)} entries found")
            
    except Exception as e:
        print(f"   Error analyzing file: {e}")

def analyze_system_status(entries):
    """Analyze system status entries."""
    events = [entry.get('status', {}).get('event') for entry in entries]
    event_counts = pd.Series(events).value_counts()
    
    print(f"   Total entries: {len(entries)}")
    print("   Events:")
    for event, count in event_counts.items():
        print(f"     {event}: {count}")
    
    # Check for specific issues
    if 'mt5_connection_attempt' in events:
        print("   ✅ MT5 connection attempts detected")
    if 'trade_executed' in events:
        print("   ✅ Trades executed")
    if 'single_trade_constraint_check' in events:
        print("   ✅ Single trade constraint checks performed")

def analyze_trade_decisions(entries):
    """Analyze trade decision entries."""
    actions = []
    expected_values = []
    confidences = []
    
    for entry in entries:
        decision = entry.get('trade_decision', {})
        action = decision.get('action')
        ev = decision.get('expected_value')
        conf = decision.get('confidence')
        
        if action:
            actions.append(action)
        if ev is not None and ev != 'N/A':
            expected_values.append(ev)
        if conf is not None and conf != 'N/A':
            confidences.append(conf)
    
    print(f"   Total decisions: {len(entries)}")
    
    if actions:
        action_counts = pd.Series(actions).value_counts()
        print("   Actions:")
        for action, count in action_counts.items():
            print(f"     {action}: {count}")
    
    if expected_values:
        ev_series = pd.Series(expected_values)
        print(f"   Expected Value Stats:")
        print(f"     Mean: {ev_series.mean():.6f}")
        print(f"     Min: {ev_series.min():.6f}")
        print(f"     Max: {ev_series.max():.6f}")
        print(f"     Zero count: {(ev_series == 0).sum()}")
    
    if confidences:
        conf_series = pd.Series(confidences)
        print(f"   Confidence Stats:")
        print(f"     Mean: {conf_series.mean():.3f}")
        print(f"     Min: {conf_series.min():.3f}")
        print(f"     Max: {conf_series.max():.3f}")

def analyze_expected_value(entries):
    """Analyze expected value entries."""
    print(f"   Total entries: {len(entries)}")
    # Add specific expected value analysis here

def analyze_errors(entries):
    """Analyze error entries."""
    if not entries:
        print("   No errors logged")
        return
    
    print(f"   Total errors: {len(entries)}")
    
    error_types = [entry.get('error', {}).get('event') for entry in entries]
    error_counts = pd.Series(error_types).value_counts()
    
    print("   Error types:")
    for error_type, count in error_counts.items():
        print(f"     {error_type}: {count}")

def analyze_ensemble_predictions(entries):
    """Analyze ensemble prediction entries."""
    print(f"   Total entries: {len(entries)}")
    # Add specific ensemble analysis here

def find_latest_session():
    """Find the latest debug session directory."""
    debug_logs_dir = "debug_logs"
    if not os.path.exists(debug_logs_dir):
        return None
    
    session_dirs = glob.glob(os.path.join(debug_logs_dir, "session_*"))
    if not session_dirs:
        return None
    
    # Return the most recent session
    latest_session = max(session_dirs, key=os.path.getctime)
    return latest_session

def main():
    """Main analysis function."""
    print("🔍 DEBUG LOG ANALYSIS")
    print("=" * 60)
    
    # Find latest session
    latest_session = find_latest_session()
    if latest_session:
        print(f"📁 Latest session: {latest_session}")
        analyze_debug_session(latest_session)
    else:
        print("❌ No debug sessions found")
    
    # List all sessions
    debug_logs_dir = "debug_logs"
    if os.path.exists(debug_logs_dir):
        session_dirs = glob.glob(os.path.join(debug_logs_dir, "session_*"))
        if session_dirs:
            print(f"\n📁 All debug sessions:")
            for session_dir in sorted(session_dirs, key=os.path.getctime, reverse=True):
                session_name = os.path.basename(session_dir)
                ctime = datetime.fromtimestamp(os.path.getctime(session_dir))
                print(f"   {session_name} - {ctime.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 