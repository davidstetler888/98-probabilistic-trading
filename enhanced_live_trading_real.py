#!/usr/bin/env python3
"""
Enhanced Live Trading with Real MT5 Orders
Modified version with actual MT5 trade execution
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

class EnhancedLiveTraderReal:
    """Enhanced live trader with real MT5 orders and comprehensive debugging."""
    
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
            'cooldown_minutes': self.cooldown_minutes,
            'trading_mode': 'real_mt5_orders'
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
                'equity': account_info.equity if account_info else 0,
                'trading_allowed': terminal_info.trade_allowed if terminal_info else False
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
        """Check single trade constraint with real MT5 positions."""
        try:
            # Get real MT5 positions
            positions = mt5.positions_get()
            current_positions = len(positions) if positions else 0
            
            constraint_check = {
                'event': 'single_trade_constraint_check',
                'open_positions': current_positions,
                'max_positions': self.max_positions,
                'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None,
                'cooldown_minutes': self.cooldown_minutes
            }
            
            # Check if we have open positions
            if current_positions >= self.max_positions:
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
            
        except Exception as e:
            self.logger.log_error({
                'event': 'constraint_check_error',
                'error': str(e)
            })
            return False, f"Constraint check error: {e}"
    
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
        """Execute real MT5 trade with detailed logging."""
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
            
            # Get current market price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                self.logger.log_error({
                    'event': 'no_market_price',
                    'symbol': symbol
                })
                return False
            
            # Prepare order request
            if action == 'buy':
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
                sl = price - 0.0020  # 20 pips stop loss
                tp = price + 0.0040  # 40 pips take profit
            elif action == 'sell':
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
                sl = price + 0.0020  # 20 pips stop loss
                tp = price - 0.0040  # 40 pips take profit
            else:
                self.logger.log_error({
                    'event': 'invalid_action',
                    'action': action
                })
                return False
            
            # Create order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 234000,
                "comment": f"EV:{trade_execution.get('expected_value', 0):.6f}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.log_error({
                    'event': 'order_failed',
                    'retcode': result.retcode,
                    'comment': result.comment,
                    'action': action,
                    'symbol': symbol
                })
                return False
            
            # Log successful trade
            self.logger.log_system_status({
                'event': 'trade_executed',
                'order_id': result.order,
                'action': action,
                'symbol': symbol,
                'lot_size': lot_size,
                'price': price,
                'stop_loss': sl,
                'take_profit': tp,
                'retcode': result.retcode
            })
            
            self.last_trade_time = datetime.now()
            return True
            
        except Exception as e:
            self.logger.log_error({
                'event': 'trade_execution_error',
                'error': str(e),
                'action': action
            })
            return False
    
    def update_position_tracking(self):
        """Update position tracking with real MT5 positions."""
        try:
            # Get real MT5 positions
            positions = mt5.positions_get()
            current_time = datetime.now()
            
            # Log position tracking
            position_summary = {
                'event': 'position_tracking_update',
                'open_positions_count': len(positions) if positions else 0,
                'timestamp': current_time.isoformat()
            }
            
            if positions:
                position_summary['positions'] = [
                    {
                        'ticket': pos.ticket,
                        'symbol': pos.symbol,
                        'type': 'buy' if pos.type == 0 else 'sell',
                        'volume': pos.volume,
                        'price_open': pos.price_open,
                        'price_current': pos.price_current,
                        'profit': pos.profit,
                        'swap': pos.swap,
                        'time': datetime.fromtimestamp(pos.time).isoformat()
                    }
                    for pos in positions
                ]
            
            self.logger.log_system_status(position_summary)
            
        except Exception as e:
            self.logger.log_error({
                'event': 'position_tracking_error',
                'error': str(e)
            })
    
    def run_live_trading(self, symbol="EURUSD.PRO", check_interval=60):
        """Run live trading with real MT5 orders and comprehensive debugging."""
        try:
            self.logger.log_system_status({
                'event': 'live_trading_start',
                'symbol': symbol,
                'check_interval': check_interval,
                'timestamp': datetime.now().isoformat(),
                'trading_mode': 'real_mt5_orders'
            })
            
            print(f"🚀 Starting enhanced live trading with REAL MT5 orders for {symbol}...")
            print(f"📊 Debug logs will be saved to: {self.logger.session_dir}")
            print(f"⚠️  WARNING: This will place REAL trades with REAL money!")
            
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
                            print(f"✅ REAL trade executed: {action.upper()}")
                        else:
                            print(f"❌ Trade execution failed: {action.upper()}")
                    
                    # Wait for next check
                    time.sleep(check_interval)
                    
                except KeyboardInterrupt:
                    print("\n🛑 Stopping enhanced live trading...")
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
    """Main function for enhanced live trading with real orders."""
    trader = EnhancedLiveTraderReal()
    
    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n🛑 Received interrupt signal, shutting down...")
        trader.is_running = False
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run live trading
    symbol = "EURUSD.PRO"
    check_interval = 60  # seconds
    
    trader.run_live_trading(symbol=symbol, check_interval=check_interval)

if __name__ == "__main__":
    main() 