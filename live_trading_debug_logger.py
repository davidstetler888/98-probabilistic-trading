#!/usr/bin/env python3
"""
Live Trading Debug Logger
Comprehensive logging and debugging system for live trading
Outputs detailed information to files for analysis
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class LiveTradingDebugLogger:
    """Comprehensive debug logger for live trading system."""
    
    def __init__(self, log_dir="debug_logs"):
        self.log_dir = log_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging directories and files."""
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create session-specific directory
        self.session_dir = os.path.join(self.log_dir, f"session_{self.session_id}")
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Initialize log files
        self.log_files = {
            'system_status': os.path.join(self.session_dir, 'system_status.log'),
            'signal_generation': os.path.join(self.session_dir, 'signal_generation.log'),
            'expected_value': os.path.join(self.session_dir, 'expected_value.log'),
            'trade_decisions': os.path.join(self.session_dir, 'trade_decisions.log'),
            'ensemble_predictions': os.path.join(self.session_dir, 'ensemble_predictions.log'),
            'probabilistic_labels': os.path.join(self.session_dir, 'probabilistic_labels.log'),
            'errors': os.path.join(self.session_dir, 'errors.log'),
            'performance': os.path.join(self.session_dir, 'performance.log')
        }
        
        # Create summary file
        self.summary_file = os.path.join(self.session_dir, 'debug_summary.json')
        
        print(f"🔧 Debug Logger initialized")
        print(f"   Session ID: {self.session_id}")
        print(f"   Log directory: {self.session_dir}")
        
    def log_system_status(self, status_data):
        """Log system status information."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'status': status_data
        }
        
        with open(self.log_files['system_status'], 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    def log_signal_generation(self, signal_data):
        """Log signal generation details."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'signal_data': signal_data
        }
        
        with open(self.log_files['signal_generation'], 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    def log_expected_value(self, ev_data):
        """Log expected value calculation details."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'expected_value_data': ev_data
        }
        
        with open(self.log_files['expected_value'], 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    def log_trade_decision(self, decision_data):
        """Log trade decision details."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'trade_decision': decision_data
        }
        
        with open(self.log_files['trade_decisions'], 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    def log_ensemble_predictions(self, prediction_data):
        """Log ensemble prediction details."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'ensemble_predictions': prediction_data
        }
        
        with open(self.log_files['ensemble_predictions'], 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    def log_probabilistic_labels(self, labels_data):
        """Log probabilistic labeling details."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'probabilistic_labels': labels_data
        }
        
        with open(self.log_files['probabilistic_labels'], 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    def log_error(self, error_data):
        """Log error information."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'error': error_data
        }
        
        with open(self.log_files['errors'], 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    def log_performance(self, performance_data):
        """Log performance metrics."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'performance': performance_data
        }
        
        with open(self.log_files['performance'], 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    def create_debug_summary(self):
        """Create a comprehensive debug summary."""
        summary = {
            'session_id': self.session_id,
            'session_start': datetime.now().isoformat(),
            'log_files': self.log_files,
            'session_dir': self.session_dir
        }
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        return summary

def create_enhanced_live_trading_with_debugging():
    """Create an enhanced version of live trading with comprehensive debugging."""
    
    enhanced_code = '''
#!/usr/bin/env python3
"""
Enhanced Live Trading with Comprehensive Debugging
Modified version of live_trading.py with detailed logging
"""

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import time
import signal
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our debug logger
from live_trading_debug_logger import LiveTradingDebugLogger

class EnhancedLiveTrader:
    """Enhanced live trader with comprehensive debugging."""
    
    def __init__(self):
        self.logger = LiveTradingDebugLogger()
        self.last_trade_time = None
        self.cooldown_minutes = 5
        self.max_positions = 1
        self.open_positions = []
        self.system = None
        self.is_running = False
        
        # Log initialization
        self.logger.log_system_status({
            'event': 'initialization',
            'max_positions': self.max_positions,
            'cooldown_minutes': self.cooldown_minutes
        })
        
    def connect_mt5(self):
        """Connect to MetaTrader 5 with detailed logging."""
        try:
            self.logger.log_system_status({
                'event': 'mt5_connection_attempt',
                'timestamp': datetime.now().isoformat()
            })
            
            if not mt5.initialize():
                error_msg = f"Failed to initialize MT5: {mt5.last_error()}"
                self.logger.log_error({
                    'event': 'mt5_initialization_failed',
                    'error': error_msg
                })
                return False
                
            # Get terminal info
            terminal_info = mt5.terminal_info()
            account_info = mt5.account_info()
            
            connection_status = {
                'event': 'mt5_connected',
                'terminal': terminal_info.name if terminal_info else 'Unknown',
                'account': account_info.login if account_info else 'Unknown',
                'server': account_info.server if account_info else 'Unknown',
                'balance': account_info.balance if account_info else 0,
                'equity': account_info.equity if account_info else 0
            }
            
            self.logger.log_system_status(connection_status)
            return True
            
        except Exception as e:
            self.logger.log_error({
                'event': 'mt5_connection_exception',
                'error': str(e)
            })
            return False
    
    def get_latest_data(self, symbol="EURUSD.PRO", bars=100):
        """Get latest market data with logging."""
        try:
            # Get latest data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars)
            
            if rates is None or len(rates) == 0:
                self.logger.log_error({
                    'event': 'no_market_data',
                    'symbol': symbol,
                    'bars_requested': bars
                })
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.set_index('time')
            
            # Log data summary
            data_summary = {
                'event': 'market_data_received',
                'symbol': symbol,
                'bars_count': len(df),
                'time_range': {
                    'start': df.index[0].isoformat(),
                    'end': df.index[-1].isoformat()
                },
                'price_range': {
                    'min': float(df['close'].min()),
                    'max': float(df['close'].max())
                }
            }
            
            self.logger.log_system_status(data_summary)
            return df
            
        except Exception as e:
            self.logger.log_error({
                'event': 'market_data_error',
                'error': str(e),
                'symbol': symbol
            })
            return None
    
    def check_single_trade_constraint(self):
        """Check single trade constraint with logging."""
        constraint_check = {
            'event': 'single_trade_constraint_check',
            'open_positions': len(self.open_positions),
            'max_positions': self.max_positions,
            'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None,
            'cooldown_minutes': self.cooldown_minutes
        }
        
        # Check if we have open positions
        if len(self.open_positions) >= self.max_positions:
            constraint_check['result'] = False
            constraint_check['reason'] = 'max_positions_reached'
            self.logger.log_system_status(constraint_check)
            return False, "Single trade constraint: Maximum positions reached"
        
        # Check cooldown period
        if self.last_trade_time:
            time_since_last_trade = datetime.now() - self.last_trade_time
            cooldown_remaining = timedelta(minutes=self.cooldown_minutes) - time_since_last_trade
            
            if cooldown_remaining.total_seconds() > 0:
                constraint_check['result'] = False
                constraint_check['reason'] = 'cooldown_period'
                constraint_check['cooldown_remaining_seconds'] = cooldown_remaining.total_seconds()
                self.logger.log_system_status(constraint_check)
                return False, f"Single trade constraint: Cooldown period ({cooldown_remaining.total_seconds():.0f}s remaining)"
        
        constraint_check['result'] = True
        constraint_check['reason'] = 'constraint_satisfied'
        self.logger.log_system_status(constraint_check)
        return True, "Single trade constraint satisfied"
    
    def process_market_data(self, df):
        """Process market data with comprehensive logging."""
        try:
            # Log data processing start
            self.logger.log_system_status({
                'event': 'market_data_processing_start',
                'data_bars': len(df),
                'timestamp': datetime.now().isoformat()
            })
            
            # Check if system is initialized
            if not self.system:
                self.logger.log_error({
                    'event': 'system_not_initialized',
                    'message': 'Trading system not initialized'
                })
                return None
            
            # Check single trade constraint
            constraint_ok, constraint_reason = self.check_single_trade_constraint()
            if not constraint_ok:
                return {
                    'action': 'no_action',
                    'reason': constraint_reason,
                    'expected_value': None,
                    'confidence': None
                }
            
            # Process with ensemble system
            trade_decision = self.system.process_market_data(df)
            
            # Log trade decision
            if trade_decision:
                self.logger.log_trade_decision({
                    'action': trade_decision.get('action'),
                    'expected_value': trade_decision.get('expected_value'),
                    'confidence': trade_decision.get('confidence'),
                    'reason': trade_decision.get('reason'),
                    'constraint_check': constraint_reason
                })
            else:
                self.logger.log_trade_decision({
                    'action': 'no_decision',
                    'reason': 'system_no_decision',
                    'constraint_check': constraint_reason
                })
            
            return trade_decision
            
        except Exception as e:
            self.logger.log_error({
                'event': 'market_data_processing_error',
                'error': str(e)
            })
            return None
    
    def execute_trade(self, action, symbol="EURUSD.PRO", lot_size=0.01):
        """Execute trade with detailed logging."""
        try:
            # Check constraint again before execution
            constraint_ok, constraint_reason = self.check_single_trade_constraint()
            if not constraint_ok:
                self.logger.log_error({
                    'event': 'trade_execution_blocked',
                    'reason': constraint_reason
                })
                return False
            
            # Log trade execution attempt
            trade_execution = {
                'event': 'trade_execution_attempt',
                'action': action,
                'symbol': symbol,
                'lot_size': lot_size,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.log_system_status(trade_execution)
            
            # Execute trade (mock for now)
            if action == 'buy':
                # Mock buy order
                position_id = f"BUY_{int(time.time())}"
                self.open_positions.append({
                    'id': position_id,
                    'type': 'buy',
                    'symbol': symbol,
                    'lot_size': lot_size,
                    'open_time': datetime.now(),
                    'open_price': 1.15500  # Mock price
                })
                
                self.last_trade_time = datetime.now()
                
                # Log successful trade
                self.logger.log_system_status({
                    'event': 'trade_executed',
                    'position_id': position_id,
                    'action': action,
                    'symbol': symbol,
                    'lot_size': lot_size,
                    'open_positions_count': len(self.open_positions)
                })
                
                return True
                
            elif action == 'sell':
                # Mock sell order
                position_id = f"SELL_{int(time.time())}"
                self.open_positions.append({
                    'id': position_id,
                    'type': 'sell',
                    'symbol': symbol,
                    'lot_size': lot_size,
                    'open_time': datetime.now(),
                    'open_price': 1.15500  # Mock price
                })
                
                self.last_trade_time = datetime.now()
                
                # Log successful trade
                self.logger.log_system_status({
                    'event': 'trade_executed',
                    'position_id': position_id,
                    'action': action,
                    'symbol': symbol,
                    'lot_size': lot_size,
                    'open_positions_count': len(self.open_positions)
                })
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.log_error({
                'event': 'trade_execution_error',
                'error': str(e),
                'action': action
            })
            return False
    
    def update_position_tracking(self):
        """Update position tracking with logging."""
        try:
            # Mock position updates (in real system, check MT5 positions)
            current_time = datetime.now()
            
            # Log position tracking
            position_summary = {
                'event': 'position_tracking_update',
                'open_positions_count': len(self.open_positions),
                'timestamp': current_time.isoformat()
            }
            
            if self.open_positions:
                position_summary['positions'] = [
                    {
                        'id': pos['id'],
                        'type': pos['type'],
                        'symbol': pos['symbol'],
                        'lot_size': pos['lot_size'],
                        'open_time': pos['open_time'].isoformat(),
                        'duration_minutes': (current_time - pos['open_time']).total_seconds() / 60
                    }
                    for pos in self.open_positions
                ]
            
            self.logger.log_system_status(position_summary)
            
        except Exception as e:
            self.logger.log_error({
                'event': 'position_tracking_error',
                'error': str(e)
            })
    
    def run_live_trading(self, symbol="EURUSD.PRO", check_interval=60):
        """Run live trading with comprehensive debugging."""
        try:
            self.logger.log_system_status({
                'event': 'live_trading_start',
                'symbol': symbol,
                'check_interval': check_interval,
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"🚀 Starting enhanced live trading for {symbol}...")
            print(f"📊 Debug logs will be saved to: {self.logger.session_dir}")
            
            # Connect to MT5
            if not self.connect_mt5():
                print("❌ Failed to connect to MT5")
                return
            
            # Initialize trading system
            print("Initializing trading system...")
            from phase3_live_trading_preparation import LiveTradingSystem
            
            # Get initial data for training
            initial_data = self.get_latest_data(symbol, bars=1000)
            if initial_data is None:
                print("❌ Failed to get initial data")
                return
            
            self.system = LiveTradingSystem()
            success = self.system.initialize_system(initial_data)
            
            if not success:
                print("❌ Failed to initialize trading system")
                return
            
            self.system.start_trading()
            self.is_running = True
            
            self.logger.log_system_status({
                'event': 'trading_system_initialized',
                'success': success,
                'initial_data_bars': len(initial_data)
            })
            
            print("✅ Enhanced live trading started. Press Ctrl+C to stop.")
            print(f"📊 Check interval: {check_interval} seconds")
            
            # Main trading loop
            while self.is_running:
                try:
                    # Get latest data
                    df = self.get_latest_data(symbol, bars=100)
                    if df is None:
                        continue
                    
                    # Update position tracking
                    self.update_position_tracking()
                    
                    # Process market data
                    trade_decision = self.process_market_data(df)
                    
                    # Execute trade if decision made
                    if trade_decision and trade_decision.get('action') in ['buy', 'sell']:
                        action = trade_decision.get('action')
                        expected_value = trade_decision.get('expected_value')
                        confidence = trade_decision.get('confidence')
                        
                        print(f"📈 Signal: {action.upper()} | EV: {expected_value:.6f} | Conf: {confidence:.3f}")
                        
                        # Execute trade
                        success = self.execute_trade(action, symbol)
                        if success:
                            print(f"✅ Trade executed: {action.upper()}")
                        else:
                            print(f"❌ Trade execution failed: {action.upper()}")
                    
                    # Wait for next check
                    time.sleep(check_interval)
                    
                except KeyboardInterrupt:
                    print("\\n🛑 Stopping enhanced live trading...")
                    self.is_running = False
                    break
                except Exception as e:
                    self.logger.log_error({
                        'event': 'main_loop_error',
                        'error': str(e)
                    })
                    print(f"❌ Error in main loop: {e}")
                    time.sleep(check_interval)
            
            # Create debug summary
            self.logger.create_debug_summary()
            print(f"📊 Debug summary created: {self.logger.summary_file}")
            
        except Exception as e:
            self.logger.log_error({
                'event': 'live_trading_error',
                'error': str(e)
            })
            print(f"❌ Error in live trading: {e}")

def main():
    """Main function for enhanced live trading."""
    trader = EnhancedLiveTrader()
    
    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\\n🛑 Received interrupt signal, shutting down...")
        trader.is_running = False
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run live trading
    symbol = "EURUSD.PRO"
    check_interval = 60  # seconds
    
    trader.run_live_trading(symbol=symbol, check_interval=check_interval)

if __name__ == "__main__":
    main()
'''
    
    return enhanced_code

def create_debug_analysis_script():
    """Create a script to analyze debug logs."""
    
    analysis_script = '''
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
    
    print("\\n" + "=" * 60)

def analyze_log_file(file_path, description):
    """Analyze a specific log file."""
    print(f"\\n📋 {description} Analysis:")
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
            print(f"\\n📁 All debug sessions:")
            for session_dir in sorted(session_dirs, key=os.path.getctime, reverse=True):
                session_name = os.path.basename(session_dir)
                ctime = datetime.fromtimestamp(os.path.getctime(session_dir))
                print(f"   {session_name} - {ctime.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
'''
    
    return analysis_script

def main():
    """Create the debugging system."""
    print("🔧 CREATING LIVE TRADING DEBUG SYSTEM")
    print("=" * 60)
    
    # Create the debug logger
    print("✅ Created live_trading_debug_logger.py")
    
    # Create enhanced live trading script
    enhanced_code = create_enhanced_live_trading_with_debugging()
    with open('enhanced_live_trading.py', 'w') as f:
        f.write(enhanced_code)
    print("✅ Created enhanced_live_trading.py")
    
    # Create debug analysis script
    analysis_script = create_debug_analysis_script()
    with open('analyze_debug_logs.py', 'w') as f:
        f.write(analysis_script)
    print("✅ Created analyze_debug_logs.py")
    
    # Create usage instructions
    instructions = '''
# LIVE TRADING DEBUG SYSTEM USAGE

## Files Created:
1. `live_trading_debug_logger.py` - Core debugging and logging system
2. `enhanced_live_trading.py` - Enhanced live trading with comprehensive logging
3. `analyze_debug_logs.py` - Script to analyze debug logs

## Usage Instructions:

### 1. On Windows VM (Live Trading):
```bash
# Run enhanced live trading (replaces live_trading.py)
python enhanced_live_trading.py
```

### 2. After Trading Session:
```bash
# Create a zip file of debug logs
zip -r debug_logs.zip debug_logs/
```

### 3. Transfer to Mac:
- Download `debug_logs.zip` from VM
- Extract to your Mac development directory

### 4. On Mac (Analysis):
```bash
# Analyze the latest debug session
python analyze_debug_logs.py
```

## What Gets Logged:

### System Status:
- MT5 connection attempts and status
- Trading system initialization
- Position tracking updates
- Single trade constraint checks

### Trade Decisions:
- All trade decisions with expected value and confidence
- Action taken (buy/sell/no_action)
- Reasons for decisions

### Expected Value:
- Detailed expected value calculations
- Probabilistic labeling results
- Ensemble prediction details

### Errors:
- All errors and exceptions
- System failures
- Connection issues

### Performance:
- Trade execution results
- Position management
- Risk management events

## Debug Output Location:
- All logs saved to: `debug_logs/session_YYYYMMDD_HHMMSS/`
- Each session gets its own directory
- Summary file: `debug_summary.json`

## Key Benefits:
1. **Comprehensive Logging**: Every aspect of the trading system is logged
2. **Session Isolation**: Each trading session gets its own log directory
3. **Easy Analysis**: Structured JSON logs for easy parsing
4. **Error Tracking**: All errors and exceptions are captured
5. **Performance Monitoring**: Track expected value and confidence over time

## Next Steps:
1. Deploy `enhanced_live_trading.py` to your Windows VM
2. Run a trading session
3. Download the debug logs
4. Analyze on your Mac to identify issues
'''
    
    with open('DEBUG_SYSTEM_USAGE.md', 'w') as f:
        f.write(instructions)
    print("✅ Created DEBUG_SYSTEM_USAGE.md")
    
    print("\n🎯 DEBUG SYSTEM READY!")
    print("=" * 60)
    print("📋 Next Steps:")
    print("   1. Copy enhanced_live_trading.py to your Windows VM")
    print("   2. Run a trading session with enhanced logging")
    print("   3. Download debug_logs.zip from VM")
    print("   4. Extract and analyze on your Mac")
    print("   5. Use analyze_debug_logs.py to identify issues")

if __name__ == "__main__":
    main() 