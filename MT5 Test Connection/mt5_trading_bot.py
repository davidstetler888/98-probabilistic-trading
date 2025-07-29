import MetaTrader5 as mt5
import time
import datetime
from typing import Optional, Dict, Any
import logging
from config import SYMBOL, STOP_LOSS_PIPS, TAKE_PROFIT_PIPS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MT5TradingBot:
    def __init__(self, symbol: str = SYMBOL, lot_size: float = 0.01):
        """
        Initialize the MT5 Trading Bot
        
        Args:
            symbol: Trading symbol (default: from config)
            lot_size: Size of the trade in lots (default: 0.01)
        """
        self.symbol = symbol
        self.lot_size = lot_size
        self.connected = False
        
    def get_error_description(self, retcode: int) -> str:
        """
        Get detailed error description for MT5 return codes
        
        Args:
            retcode: MT5 return code
            
        Returns:
            str: Error description
        """
        error_descriptions = {
            10004: "TRADE_RETCODE_REQUOTE - Requote",
            10006: "TRADE_RETCODE_REJECT - Request rejected",
            10007: "TRADE_RETCODE_CANCEL - Request canceled by trader",
            10008: "TRADE_RETCODE_PLACED - Order placed",
            10009: "TRADE_RETCODE_DONE - Request completed",
            10010: "TRADE_RETCODE_DONE_PARTIAL - Request partially completed",
            10011: "TRADE_RETCODE_ERROR - Request processing error",
            10012: "TRADE_RETCODE_TIMEOUT - Request canceled by timeout",
            10013: "TRADE_RETCODE_INVALID - Invalid request",
            10014: "TRADE_RETCODE_INVALID_VOLUME - Invalid volume",
            10015: "TRADE_RETCODE_INVALID_PRICE - Invalid price",
            10016: "TRADE_RETCODE_INVALID_STOPS - Invalid stops",
            10017: "TRADE_RETCODE_TRADE_DISABLED - Trading disabled",
            10018: "TRADE_RETCODE_MARKET_CLOSED - Market closed",
            10019: "TRADE_RETCODE_NO_MONEY - Not enough money",
            10020: "TRADE_RETCODE_PRICE_CHANGED - Price changed",
            10021: "TRADE_RETCODE_PRICE_OFF - Off quotes",
            10022: "TRADE_RETCODE_BROKER_BUSY - Broker busy",
            10023: "TRADE_RETCODE_REQUOTE - Requote",
            10024: "TRADE_RETCODE_ORDER_LOCKED - Order locked",
            10025: "TRADE_RETCODE_LONG_ONLY - Long positions only",
            10026: "TRADE_RETCODE_SHORT_ONLY - Short positions only",
            10027: "TRADE_RETCODE_CLOSE_ONLY - Close only",
            10028: "TRADE_RETCODE_FIFO_CLOSE - FIFO close",
            10029: "TRADE_RETCODE_INVALID_FILL - Invalid fill",
            10030: "TRADE_RETCODE_INVALID_MODIFICATION - Invalid modification",
            10031: "TRADE_RETCODE_INVALID_ORDER - Invalid order",
            10032: "TRADE_RETCODE_INVALID_ACCOUNT - Invalid account",
            10033: "TRADE_RETCODE_INVALID_SYMBOL - Invalid symbol",
            10034: "TRADE_RETCODE_INVALID_VOLUME - Invalid volume",
            10035: "TRADE_RETCODE_INVALID_PRICE - Invalid price",
            10036: "TRADE_RETCODE_INVALID_STOPS - Invalid stops",
            10037: "TRADE_RETCODE_INVALID_TRADE_PARAMETERS - Invalid trade parameters",
            10038: "TRADE_RETCODE_INVALID_TRADE_PARAMETERS - Invalid trade parameters",
            10039: "TRADE_RETCODE_INVALID_TRADE_PARAMETERS - Invalid trade parameters",
            10040: "TRADE_RETCODE_INVALID_TRADE_PARAMETERS - Invalid trade parameters",
            10041: "TRADE_RETCODE_INVALID_TRADE_PARAMETERS - Invalid trade parameters",
            10042: "TRADE_RETCODE_INVALID_TRADE_PARAMETERS - Invalid trade parameters",
            10043: "TRADE_RETCODE_INVALID_TRADE_PARAMETERS - Invalid trade parameters",
            10044: "TRADE_RETCODE_INVALID_TRADE_PARAMETERS - Invalid trade parameters",
            10045: "TRADE_RETCODE_INVALID_TRADE_PARAMETERS - Invalid trade parameters",
            10046: "TRADE_RETCODE_INVALID_TRADE_PARAMETERS - Invalid trade parameters",
            10047: "TRADE_RETCODE_INVALID_TRADE_PARAMETERS - Invalid trade parameters",
            10048: "TRADE_RETCODE_INVALID_TRADE_PARAMETERS - Invalid trade parameters",
            10049: "TRADE_RETCODE_INVALID_TRADE_PARAMETERS - Invalid trade parameters",
            10050: "TRADE_RETCODE_INVALID_TRADE_PARAMETERS - Invalid trade parameters",
        }
        
        return error_descriptions.get(retcode, f"Unknown error code: {retcode}")

    def connect_to_mt5(self) -> bool:
        """
        Connect to MetaTrader 5
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Initialize MT5
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Check if MT5 is running
            if not mt5.terminal_info():
                logger.error("MT5 terminal is not running")
                return False
            
            # Check if trading is allowed
            if not mt5.account_info():
                logger.error("Trading account not available")
                return False
            
            logger.info("Successfully connected to MetaTrader 5")
            self.connected = True
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to MT5: {e}")
            return False
    
    def get_symbol_info(self) -> Optional[Dict[str, Any]]:
        """
        Get symbol information
        
        Returns:
            Dict containing symbol info or None if failed
        """
        try:
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logger.error(f"Symbol {self.symbol} not found")
                return None
            
            if not symbol_info.visible:
                logger.info(f"Symbol {self.symbol} is not visible, adding it")
                if not mt5.symbol_select(self.symbol, True):
                    logger.error(f"Failed to select symbol {self.symbol}")
                    return None
            
            return {
                'name': symbol_info.name,
                'point': symbol_info.point,
                'digits': symbol_info.digits,
                'spread': symbol_info.spread,
                'trade_mode': symbol_info.trade_mode
            }
            
        except Exception as e:
            logger.error(f"Error getting symbol info: {e}")
            return None
    
    def place_buy_order(self) -> Optional[int]:
        """
        Place a buy order with stop loss and take profit
        
        Returns:
            int: Order ticket if successful, None otherwise
        """
        try:
            # Get current price and symbol info
            tick = mt5.symbol_info_tick(self.symbol)
            symbol_info = mt5.symbol_info(self.symbol)
            if tick is None or symbol_info is None:
                logger.error(f"Failed to get tick data for {self.symbol}")
                return None
            
            # Calculate stop loss and take profit prices
            point = symbol_info.point
            digits = symbol_info.digits
            min_stop_level = symbol_info.trade_stops_level * point
            
            # Calculate SL and TP prices with minimum distance validation
            sl_price = 0
            tp_price = 0
            
            if STOP_LOSS_PIPS > 0:
                sl_distance = STOP_LOSS_PIPS * point
                if sl_distance >= min_stop_level:
                    sl_price = tick.ask - sl_distance
                    sl_price = round(sl_price, digits)
                else:
                    logger.warning(f"Stop Loss distance ({sl_distance}) too small, minimum required: {min_stop_level}")
                    sl_price = tick.ask - min_stop_level
                    sl_price = round(sl_price, digits)
            
            if TAKE_PROFIT_PIPS > 0:
                tp_distance = TAKE_PROFIT_PIPS * point
                if tp_distance >= min_stop_level:
                    tp_price = tick.ask + tp_distance
                    tp_price = round(tp_price, digits)
                else:
                    logger.warning(f"Take Profit distance ({tp_distance}) too small, minimum required: {min_stop_level}")
                    tp_price = tick.ask + min_stop_level
                    tp_price = round(tp_price, digits)
            
            logger.info(f"Order details - Ask: {tick.ask}, SL: {sl_price}, TP: {tp_price}")
            
            # Prepare the order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.lot_size,
                "type": mt5.ORDER_TYPE_BUY,
                "price": tick.ask,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 20,
                "magic": 234000,
                "comment": "python script order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send the order
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_desc = self.get_error_description(result.retcode)
                logger.error(f"Order failed, return code: {result.retcode} - {error_desc}")
                logger.error(f"Error details: {result.comment}")
                return None
            
            logger.info(f"Buy order placed successfully. Ticket: {result.order}")
            if sl_price > 0:
                logger.info(f"Stop Loss set at: {sl_price}")
            if tp_price > 0:
                logger.info(f"Take Profit set at: {tp_price}")
            return result.order
            
        except Exception as e:
            logger.error(f"Error placing buy order: {e}")
            return None
    
    def place_sell_order(self) -> Optional[int]:
        """
        Place a sell order with stop loss and take profit
        
        Returns:
            int: Order ticket if successful, None otherwise
        """
        try:
            # Get current price and symbol info
            tick = mt5.symbol_info_tick(self.symbol)
            symbol_info = mt5.symbol_info(self.symbol)
            if tick is None or symbol_info is None:
                logger.error(f"Failed to get tick data for {self.symbol}")
                return None
            
            # Calculate stop loss and take profit prices
            point = symbol_info.point
            digits = symbol_info.digits
            min_stop_level = symbol_info.trade_stops_level * point
            
            # Calculate SL and TP prices (opposite direction for sell orders)
            sl_price = 0
            tp_price = 0
            
            if STOP_LOSS_PIPS > 0:
                sl_distance = STOP_LOSS_PIPS * point
                if sl_distance >= min_stop_level:
                    sl_price = tick.bid + sl_distance
                    sl_price = round(sl_price, digits)
                else:
                    logger.warning(f"Stop Loss distance ({sl_distance}) too small, minimum required: {min_stop_level}")
                    sl_price = tick.bid + min_stop_level
                    sl_price = round(sl_price, digits)
            
            if TAKE_PROFIT_PIPS > 0:
                tp_distance = TAKE_PROFIT_PIPS * point
                if tp_distance >= min_stop_level:
                    tp_price = tick.bid - tp_distance
                    tp_price = round(tp_price, digits)
                else:
                    logger.warning(f"Take Profit distance ({tp_distance}) too small, minimum required: {min_stop_level}")
                    tp_price = tick.bid - min_stop_level
                    tp_price = round(tp_price, digits)
            
            logger.info(f"Order details - Bid: {tick.bid}, SL: {sl_price}, TP: {tp_price}")
            
            # Prepare the order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.lot_size,
                "type": mt5.ORDER_TYPE_SELL,
                "price": tick.bid,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 20,
                "magic": 234000,
                "comment": "python script order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send the order
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_desc = self.get_error_description(result.retcode)
                logger.error(f"Order failed, return code: {result.retcode} - {error_desc}")
                logger.error(f"Error details: {result.comment}")
                return None
            
            logger.info(f"Sell order placed successfully. Ticket: {result.order}")
            if sl_price > 0:
                logger.info(f"Stop Loss set at: {sl_price}")
            if tp_price > 0:
                logger.info(f"Take Profit set at: {tp_price}")
            return result.order
            
        except Exception as e:
            logger.error(f"Error placing sell order: {e}")
            return None
    
    def close_position(self, ticket: int) -> bool:
        """
        Close a position by ticket
        
        Args:
            ticket: Order ticket to close
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get position information
            position = mt5.positions_get(ticket=ticket)
            if not position:
                logger.error(f"Position with ticket {ticket} not found")
                return False
            
            position = position[0]
            
            # Determine close type and price
            if position.type == mt5.POSITION_TYPE_BUY:
                close_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(self.symbol).bid
            else:
                close_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(self.symbol).ask
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": position.volume,
                "type": close_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": "python script close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send close order
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Close order failed, return code: {result.retcode}")
                return False
            
            logger.info(f"Position closed successfully. Ticket: {ticket}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def get_open_positions(self) -> list:
        """
        Get all open positions
        
        Returns:
            list: List of open positions
        """
        try:
            positions = mt5.positions_get()
            return positions if positions else []
        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
            return []
    
    def check_position_status(self, ticket: int) -> bool:
        """
        Check if a position is still open
        
        Args:
            ticket: Order ticket to check
            
        Returns:
            bool: True if position is still open, False if closed
        """
        try:
            positions = mt5.positions_get(ticket=ticket)
            return len(positions) > 0
        except Exception as e:
            logger.error(f"Error checking position status: {e}")
            return False
    
    def run_trading_cycle(self):
        """
        Run one complete trading cycle:
        1. Place a trade
        2. Keep it open for 10 seconds (or until SL/TP hit)
        3. Close the trade (if still open)
        4. Wait 20 seconds before next cycle
        """
        try:
            # Place a buy order
            logger.info("Starting new trading cycle...")
            ticket = self.place_buy_order()
            
            if ticket is None:
                logger.error("Failed to place order, skipping cycle")
                return
            
            # Keep position open for 10 seconds, checking for SL/TP
            logger.info(f"Position opened with ticket {ticket}. Monitoring for 10 seconds...")
            start_time = time.time()
            
            while time.time() - start_time < 10:
                # Check if position was closed by SL/TP
                if not self.check_position_status(ticket):
                    logger.info(f"Position {ticket} was closed by Stop Loss or Take Profit")
                    break
                time.sleep(1)  # Check every second
            
            # If position is still open, close it manually
            if self.check_position_status(ticket):
                logger.info("Closing position manually...")
                if self.close_position(ticket):
                    logger.info("Position closed successfully")
                else:
                    logger.error("Failed to close position")
            else:
                logger.info("Position already closed by SL/TP")
            
            # Wait 20 seconds before next cycle
            logger.info("Waiting 20 seconds before next cycle...")
            time.sleep(20)
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    def run_continuous_trading(self, max_cycles: int = None):
        """
        Run continuous trading cycles
        
        Args:
            max_cycles: Maximum number of cycles to run (None for infinite)
        """
        if not self.connected:
            logger.error("Not connected to MT5. Please connect first.")
            return
        
        logger.info(f"Starting continuous trading on {self.symbol}")
        logger.info(f"Lot size: {self.lot_size}")
        logger.info(f"Max cycles: {max_cycles if max_cycles else 'Infinite'}")
        
        cycle_count = 0
        
        try:
            while True:
                if max_cycles and cycle_count >= max_cycles:
                    logger.info(f"Reached maximum cycles ({max_cycles}). Stopping.")
                    break
                
                cycle_count += 1
                logger.info(f"=== Cycle {cycle_count} ===")
                
                self.run_trading_cycle()
                
        except KeyboardInterrupt:
            logger.info("Trading stopped by user")
        except Exception as e:
            logger.error(f"Unexpected error in continuous trading: {e}")
        finally:
            # Close any remaining positions
            positions = self.get_open_positions()
            if positions:
                logger.info("Closing remaining positions...")
                for position in positions:
                    self.close_position(position.ticket)
    
    def disconnect(self):
        """Disconnect from MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MetaTrader 5")


def main():
    """Main function to run the trading bot"""
    # Create trading bot instance
    bot = MT5TradingBot(symbol=SYMBOL, lot_size=0.01)
    
    try:
        # Connect to MT5
        if not bot.connect_to_mt5():
            logger.error("Failed to connect to MT5. Exiting.")
            return
        
        # Get symbol info
        symbol_info = bot.get_symbol_info()
        if symbol_info:
            logger.info(f"Symbol info: {symbol_info}")
        
        # Run continuous trading (5 cycles for testing)
        bot.run_continuous_trading(max_cycles=5)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        bot.disconnect()


if __name__ == "__main__":
    main() 