"""Live trading engine that runs continuously"""
import time
import schedule
from datetime import datetime
from typing import Dict, List
import logging

from ..core.signal_generator import PowerOf3SignalGenerator
from ..data.live_feed import LiveDataFeed
from ..brokers.oanda_broker import OandaBroker
from ..database.repository import TradingDatabase
from ..utils.signal_formatter import SignalFormatter

class LiveTradingEngine:
    def __init__(self, config: Dict):
        self.config = config
        self.signal_generator = PowerOf3SignalGenerator()
        self.data_feed = LiveDataFeed(config['api_keys']['twelve_data'])
        self.broker = OandaBroker(
            config['api_keys']['oanda'],
            config['oanda_account_id']
        )
        self.database = TradingDatabase(config['database'])
        self.running = False
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start the live trading engine"""
        self.running = True
        self.logger.info("🚀 Live Trading Engine Started")
        
        # Schedule signal generation every 5 minutes
        schedule.every(5).minutes.do(self._generate_and_execute_signals)
        
        # Schedule position monitoring every minute
        schedule.every(1).minutes.do(self._monitor_positions)
        
        # Main loop
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(10)  # Check every 10 seconds
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Stopping trading engine...")
                self.running = False
                
            except Exception as e:
                self.logger.error(f"Engine error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _generate_and_execute_signals(self):
        """Generate signals and execute trades"""
        try:
            symbols = self.config['trading']['symbols']
            account_balance = self.broker.get_account_balance()
            
            for symbol in symbols:
                # Get live data
                df = self.data_feed.get_live_data(symbol)
                
                if df.empty:
                    continue
                
                # Generate signals
                signals = self.signal_generator.generate_signals(
                    symbol, df, account_balance
                )
                
                # Execute signals
                for signal in signals:
                    if signal.quality_score >= 7.0:  # Only trade high-quality signals
                        self._execute_signal(signal)
                        
        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")
    
    def _execute_signal(self, signal):
        """Execute a trading signal"""
        try:
            # Log signal to database
            signal_id = self.database.log_signal(signal.to_dict())
            
            # Place order with broker
            if self.config.get('live_trading', False):
                order_id = self.broker.place_order(signal.to_dict())
                
                if order_id:
                    self.logger.info(f"✅ Order placed: {signal.symbol} {signal.signal_type}")
                    # Update database with order ID
                    self.database.update_signal_order_id(signal_id, order_id)
                else:
                    self.logger.error(f"❌ Failed to place order: {signal.symbol}")
            else:
                self.logger.info(f"📝 Paper trade: {signal.symbol} {signal.signal_type}")
                
        except Exception as e:
            self.logger.error(f"Execution error: {e}")
    
    def _monitor_positions(self):
        """Monitor open positions"""
        # Implementation for position monitoring
        pass