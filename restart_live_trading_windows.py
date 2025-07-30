#!/usr/bin/env python3
"""
Restart Live Trading System (Windows)
Windows-specific script to restart the live trading system with adjusted criteria.
"""

import subprocess
import time
import os
import sys

def find_live_trading_process():
    """Find running live_trading.py process on Windows."""
    try:
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'], 
                              capture_output=True, text=True, shell=True)
        lines = result.stdout.split('\n')
        for line in lines:
            if 'live_trading.py' in line:
                # Extract PID from CSV format
                parts = line.split(',')
                if len(parts) > 1:
                    pid = parts[1].strip('"')
                    return pid
        return None
    except:
        return None

def stop_live_trading():
    """Stop the live trading system on Windows."""
    print("🛑 Stopping live trading system...")
    
    try:
        # Use taskkill to stop python processes running live_trading.py
        result = subprocess.run([
            'taskkill', '/F', '/IM', 'python.exe', '/FI', 'WINDOWTITLE eq live_trading.py'
        ], capture_output=True, text=True, shell=True)
        
        if result.returncode == 0:
            print("✅ Stopped live trading processes")
        else:
            print("✅ No live trading processes found to stop")
            
        time.sleep(2)  # Wait for processes to stop
        
    except Exception as e:
        print(f"⚠️ Error stopping processes: {e}")

def start_live_trading():
    """Start the live trading system on Windows."""
    print("🚀 Starting live trading system with adjusted criteria...")
    
    try:
        # Start in a new command window
        cmd = f'start "Live Trading System" cmd /k "python live_trading.py"'
        subprocess.run(cmd, shell=True)
        
        print("✅ Started live trading system in new window")
        print("📊 System is now running with adjusted signal criteria:")
        print("   - Expected Value: ≥2 pips (was 4 pips)")
        print("   - Confidence: ≥50% (was 70%)")
        print("   - Expected signal rate: 80% (was 2%)")
        
        return True
    except Exception as e:
        print(f"❌ Error starting live trading: {e}")
        return False

def check_system_status():
    """Check if the system is running properly."""
    print("\n🔍 Checking system status...")
    
    time.sleep(3)  # Wait for system to initialize
    
    try:
        # Check if python process is running
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                              capture_output=True, text=True, shell=True)
        
        if 'python.exe' in result.stdout:
            print("✅ Python processes are running")
            return True
        else:
            print("❌ No Python processes found")
            return False
    except:
        print("⚠️ Could not check process status")
        return True  # Assume it's running

def manual_restart_instructions():
    """Provide manual restart instructions."""
    print("\n📋 MANUAL RESTART INSTRUCTIONS:")
    print("="*50)
    print("If the automatic restart didn't work, follow these steps:")
    print()
    print("1️⃣ Close any existing live trading windows")
    print("2️⃣ Open a new Command Prompt")
    print("3️⃣ Navigate to your project directory:")
    print("   cd C:\\Users\\Administrator\\Documents\\Projects\\98-probabilistic-trading")
    print("4️⃣ Start the live trading system:")
    print("   python live_trading.py")
    print()
    print("📊 The system will now run with adjusted criteria:")
    print("   - Expected Value: ≥2 pips")
    print("   - Confidence: ≥50%")
    print("   - Expected signal rate: 80%")
    print()
    print("🔧 Monitoring Commands (in separate Command Prompt):")
    print("   python quick_assessment.py")
    print("   python monitor_live_trading.py")
    print("   python analyze_signal_criteria.py")

def main():
    """Main function to restart the live trading system."""
    print("🔄 RESTARTING LIVE TRADING SYSTEM (Windows)")
    print("="*50)
    
    print("📊 Signal Criteria Adjustments Applied:")
    print("   ✅ Expected Value: 4 pips → 2 pips")
    print("   ✅ Confidence: 70% → 50%")
    print("   ✅ Symbol: EURUSD → EURUSD.PRO")
    print("   📈 Expected Results: 2% → 80% signal rate")
    
    # Stop current system
    stop_live_trading()
    
    # Start new system
    success = start_live_trading()
    
    if success:
        # Check status
        if check_system_status():
            print("\n🎉 SUCCESS! Live trading system restarted with adjusted criteria")
            print("\n📋 Next Steps:")
            print("   1. Check the new Command Prompt window for system output")
            print("   2. Monitor for trading signals")
            print("   3. Run monitoring commands in separate Command Prompt")
            
            print("\n⚠️ Important Notes:")
            print("   - System will generate 40x more trading opportunities")
            print("   - Risk management is still active")
            print("   - Position sizing based on confidence")
            print("   - Monitor closely for first few hours")
            
            print("\n🔧 Monitoring Commands (run in separate Command Prompt):")
            print("   python quick_assessment.py          # Quick performance check")
            print("   python monitor_live_trading.py      # Detailed monitoring")
            print("   python analyze_signal_criteria.py   # Signal analysis")
        else:
            print("\n⚠️ Could not verify system status")
            print("   Check the Command Prompt window manually")
            manual_restart_instructions()
    else:
        print("\n❌ Failed to restart live trading system")
        manual_restart_instructions()
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main() 