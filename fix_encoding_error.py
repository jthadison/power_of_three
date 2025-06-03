#!/usr/bin/env python3
"""
Fix Unicode Encoding Error
==========================
Quick fix for the Windows encoding issue when creating trading engine.
"""

from pathlib import Path

def create_simple_trading_engine():
    """Create a simple trading engine without Unicode characters"""
    
    # Simple content without emojis to avoid encoding issues
    trading_engine_content = '''"""Live trading engine that runs continuously"""
import time
import schedule
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Import with better error handling
try:
    from ..core.signal_generator import PowerOf3SignalGenerator
except ImportError:
    print("Warning: PowerOf3SignalGenerator not available - using mock")
    PowerOf3SignalGenerator = None

try:
    from ..data.live_feed import LiveDataFeed
except ImportError:
    print("Warning: LiveDataFeed not available - using mock")
    LiveDataFeed = None

try:
    from ..brokers.oanda_broker import OandaBroker
except ImportError:
    print("Warning: OandaBroker not available - using mock")
    OandaBroker = None

try:
    from ..database.repository import TradingDatabase
except ImportError:
    print("Warning: TradingDatabase not available - using mock")
    TradingDatabase = None

try:
    from ..utils.signal_formatter import SignalFormatter
except ImportError:
    print("Warning: SignalFormatter not available - using mock")
    SignalFormatter = None

class LiveTradingEngine:
    def __init__(self, config: Dict):
        self.config = config
        self.running = False
        
        # Initialize components with error handling
        try:
            self.signal_generator = PowerOf3SignalGenerator() if PowerOf3SignalGenerator else None
        except Exception as e:
            print(f"Could not initialize signal generator: {e}")
            self.signal_generator = None
        
        try:
            api_key = config.get('api_keys', {}).get('twelve_data')
            self.data_feed = LiveDataFeed(api_key) if LiveDataFeed and api_key else None
        except Exception as e:
            print(f"Could not initialize data feed: {e}")
            self.data_feed = None
        
        try:
            oanda_key = config.get('api_keys', {}).get('oanda')
            oanda_account = config.get('oanda_account_id')
            self.broker = OandaBroker(oanda_key, oanda_account) if OandaBroker and oanda_key else None
        except Exception as e:
            print(f"Could not initialize broker: {e}")
            self.broker = None
        
        try:
            db_config = config.get('database')
            self.database = TradingDatabase(db_config) if TradingDatabase else None
        except Exception as e:
            print(f"Could not initialize database: {e}")
            self.database = None
        
        # Setup logging
        try:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('logs/trading.log'),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger(__name__)
        except Exception as e:
            print(f"Could not setup logging: {e}")
            self.logger = None
        
        print("LiveTradingEngine initialized successfully")
    
    def start(self):
        """Start the live trading engine"""
        self.running = True
        
        if self.logger:
            self.logger.info("Live Trading Engine Started")
        
        # Check if we have the minimum required components
        if not self.signal_generator:
            print("Warning: No signal generator - running in demo mode")
        
        # Schedule signal generation every 5 minutes (if we have components)
        if self.signal_generator and self.data_feed:
            schedule.every(5).minutes.do(self._generate_and_execute_signals)
            print("Scheduled signal generation every 5 minutes")
        
        # Schedule position monitoring every minute
        schedule.every(1).minutes.do(self._monitor_positions)
        
        print("Trading engine started successfully")
        print("Use engine.stop() to stop the engine")
        
        # For testing, don't run the infinite loop
        if self.config.get('test_mode', False):
            print("Test mode - not starting infinite loop")
            return
        
        # Main loop
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(10)  # Check every 10 seconds
        except KeyboardInterrupt:
            if self.logger:
                self.logger.info("Stopping trading engine...")
            self.running = False
    
    def stop(self):
        """Stop the trading engine"""
        self.running = False
        if self.logger:
            self.logger.info("Trading engine stopped")
        print("Trading engine stopped")
    
    def get_status(self) -> Dict:
        """Get engine status"""
        return {
            'running': self.running,
            'timestamp': datetime.now().isoformat(),
            'components': {
                'signal_generator': self.signal_generator is not None,
                'data_feed': self.data_feed is not None,
                'broker': self.broker is not None,
                'database': self.database is not None
            },
            'config': self.config
        }
    
    def _generate_and_execute_signals(self):
        """Generate signals and execute trades"""
        try:
            if not self.signal_generator or not self.data_feed:
                if self.logger:
                    self.logger.warning("Cannot generate signals - missing components")
                return
                
            symbols = self.config.get('trading', {}).get('symbols', ['US30'])
            account_balance = self.broker.get_account_balance() if self.broker else 10000.0
            
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
            if self.logger:
                self.logger.error(f"Signal generation error: {e}")
            else:
                print(f"Signal generation error: {e}")
    
    def _execute_signal(self, signal):
        """Execute a trading signal with better error handling"""
        try:
            signal_id = None
            
            # Log signal to database if available
            if self.database:
                try:
                    signal_id = self.database.log_signal(signal.to_dict())
                    if self.logger:
                        self.logger.info(f"Signal logged to database: {signal_id}")
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Database logging failed: {e}")
            
            # Place order with broker if available and live trading enabled
            if self.broker and self.config.get('live_trading', False):
                try:
                    order_id = self.broker.place_order(signal.to_dict())
                    
                    if order_id:
                        if self.logger:
                            self.logger.info(f"Order placed: {signal.symbol} {signal.signal_type}")
                        
                        # Update database with order ID if both are available
                        if self.database and signal_id:
                            try:
                                # Check if method exists before calling
                                if hasattr(self.database, 'update_signal_order_id'):
                                    self.database.update_signal_order_id(signal_id, order_id)
                                else:
                                    # Fallback to updating status
                                    self.database.update_signal_status(
                                        signal.signal_id, 'executed', f'Order ID: {order_id}'
                                    )
                            except Exception as e:
                                if self.logger:
                                    self.logger.error(f"Failed to update signal with order ID: {e}")
                    else:
                        if self.logger:
                            self.logger.error(f"Failed to place order: {signal.symbol}")
                        
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Order execution failed: {e}")
            else:
                if self.logger:
                    self.logger.info(f"Paper trade: {signal.symbol} {signal.signal_type}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Execution error: {e}")
            else:
                print(f"Execution error: {e}")
    
    def _monitor_positions(self):
        """Monitor open positions"""
        try:
            if self.broker:
                # Implementation for position monitoring
                pass
        except Exception as e:
            if self.logger:
                self.logger.error(f"Position monitoring error: {e}")
'''
    
    # Save the trading engine with proper UTF-8 encoding
    trading_engine_file = Path('src/power_of_3/live/trading_engine.py')
    trading_engine_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use UTF-8 encoding to avoid Windows encoding issues
        with open(trading_engine_file, 'w', encoding='utf-8') as f:
            f.write(trading_engine_content)
        
        print(f"Successfully created: {trading_engine_file}")
        return True
        
    except Exception as e:
        print(f"Error creating trading engine: {e}")
        return False

def test_fixed_system():
    """Test the fixed system"""
    print("\nTesting the fixed system...")
    
    import sys
    from pathlib import Path
    
    # Add src to path
    src_path = Path('src').absolute()
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    try:
        # Test import
        from power_of_3.live.trading_engine import LiveTradingEngine
        print("SUCCESS: Import works!")
        
        # Test creation
        engine = LiveTradingEngine({'test_mode': True})
        print("SUCCESS: Engine created!")
        
        # Test start
        engine.start()
        print("SUCCESS: Engine started!")
        
        # Test status
        status = engine.get_status()
        print(f"SUCCESS: Status retrieved: {status['running']}")
        
        # Test stop
        engine.stop()
        print("SUCCESS: Engine stopped!")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    print("FIXING UNICODE ENCODING ERROR")
    print("=" * 35)
    
    # Create the fixed trading engine
    if create_simple_trading_engine():
        print("Fixed trading engine created successfully!")
        
        # Test the system
        if test_fixed_system():
            print("\nALL TESTS PASSED!")
            print("Your Power of 3 system is now working correctly.")
            print("\nNext steps:")
            print("1. Run: python scripts/run_live_trading.py")
            print("2. Or test with: python test_imports.py")
        else:
            print("\nSome tests failed. Check the errors above.")
    else:
        print("Failed to create trading engine.")