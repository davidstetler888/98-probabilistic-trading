#!/usr/bin/env python3
"""
Restart Live Trading System
Restarts the live trading system with adjusted signal criteria.
"""

import subprocess
import time
import signal
import os
import sys

def find_live_trading_process():
    """Find running live_trading.py process."""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        for line in lines:
            if 'live_trading.py' in line and 'python' in line:
                parts = line.split()
                if len(parts) > 1:
                    return parts[1]  # Return PID
        return None
    except:
        return None

def stop_live_trading():
    """Stop the live trading system."""
    print("🛑 Stopping live trading system...")
    
    pid = find_live_trading_process()
    if pid:
        try:
            os.kill(int(pid), signal.SIGTERM)
            print(f"✅ Sent stop signal to process {pid}")
            time.sleep(3)  # Wait for graceful shutdown
            
            # Check if still running
            if find_live_trading_process():
                print("⚠️ Process still running, forcing stop...")
                os.kill(int(pid), signal.SIGKILL)
                time.sleep(1)
        except Exception as e:
            print(f"❌ Error stopping process: {e}")
    else:
        print("✅ No live trading process found")

def start_live_trading():
    """Start the live trading system."""
    print("🚀 Starting live trading system with adjusted criteria...")
    
    try:
        # Start in background
        process = subprocess.Popen([
            sys.executable, 'live_trading.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print(f"✅ Started live trading system (PID: {process.pid})")
        print("📊 System is now running with adjusted signal criteria:")
        print("   - Expected Value: ≥2 pips (was 4 pips)")
        print("   - Confidence: ≥50% (was 70%)")
        print("   - Expected signal rate: 80% (was 2%)")
        
        return process
    except Exception as e:
        print(f"❌ Error starting live trading: {e}")
        return None

def check_system_status():
    """Check if the system is running properly."""
    print("\n🔍 Checking system status...")
    
    time.sleep(5)  # Wait for system to initialize
    
    pid = find_live_trading_process()
    if pid:
        print(f"✅ Live trading system is running (PID: {pid})")
        return True
    else:
        print("❌ Live trading system is not running")
        return False

def main():
    """Main function to restart the live trading system."""
    print("🔄 RESTARTING LIVE TRADING SYSTEM")
    print("="*50)
    
    print("📊 Signal Criteria Adjustments Applied:")
    print("   ✅ Expected Value: 4 pips → 2 pips")
    print("   ✅ Confidence: 70% → 50%")
    print("   ✅ Symbol: EURUSD → EURUSD.PRO")
    print("   📈 Expected Results: 2% → 80% signal rate")
    
    # Stop current system
    stop_live_trading()
    
    # Start new system
    process = start_live_trading()
    
    if process:
        # Check status
        if check_system_status():
            print("\n🎉 SUCCESS! Live trading system restarted with adjusted criteria")
            print("\n📋 Next Steps:")
            print("   1. Monitor the system output for signals")
            print("   2. Check for trade execution")
            print("   3. Monitor performance metrics")
            print("   4. Run quick_assessment.py to check results")
            
            print("\n⚠️ Important Notes:")
            print("   - System will generate 40x more trading opportunities")
            print("   - Risk management is still active")
            print("   - Position sizing based on confidence")
            print("   - Monitor closely for first few hours")
            
            print("\n🔧 Monitoring Commands:")
            print("   python quick_assessment.py          # Quick performance check")
            print("   python monitor_live_trading.py      # Detailed monitoring")
            print("   python analyze_signal_criteria.py   # Signal analysis")
        else:
            print("\n❌ Failed to start live trading system")
            print("   Check for errors and try again")
    else:
        print("\n❌ Failed to restart live trading system")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main() 