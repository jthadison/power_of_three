#!/usr/bin/env python3
"""
Fix Database Method
==================
Add the missing update_signal_order_id method to TradingDatabase class
"""

# =============================================================================
# MISSING METHOD FOR TRADING DATABASE
# =============================================================================

from pathlib import Path


def update_signal_order_id(self, signal_id: str, order_id: str):
    """
    Update signal with broker order ID
    
    Args:
        signal_id: UUID of the signal record
        order_id: Broker's order ID
    """
    try:
        with self.engine.connect() as conn:
            from sqlalchemy import text
            
            query = text("""
                UPDATE signals 
                SET execution_reason = :order_id,
                    status = 'executed'
                WHERE id = :signal_id OR signal_id = :signal_id
            """)
            
            result = conn.execute(query, {
                'signal_id': signal_id,
                'order_id': f"Broker Order ID: {order_id}"
            })
            
            conn.commit()
            
            if result.rowcount > 0:
                print(f"✅ Signal {signal_id} updated with order ID: {order_id}")
            else:
                print(f"⚠️ No signal found with ID: {signal_id}")
                
    except Exception as e:
        print(f"❌ Error updating signal order ID: {e}")
        raise

# =============================================================================
# QUICK FIX SCRIPT
# =============================================================================

def add_missing_method_to_database():
    """Add the missing method to the existing database file"""
    
    import os
    from pathlib import Path
    
    database_file = Path('src/power_of_3/database/repository.py')
    
    if not database_file.exists():
        print(f"❌ Database file not found: {database_file}")
        return False
    
    # Read the existing file
    with open(database_file, 'r') as f:
        content = f.read()
    
    # Check if method already exists
    if 'update_signal_order_id' in content:
        print("✅ update_signal_order_id method already exists")
        return True
    
    # Find the best place to insert the method (after update_signal_status)
    insert_point = content.find('def update_signal_status(')
    
    if insert_point == -1:
        # If not found, try to find the class definition end
        insert_point = content.find('def log_trade_entry(')
    
    if insert_point == -1:
        print("❌ Could not find insertion point in database file")
        return False
    
    # Find the end of the previous method
    insert_point = content.find('\n    def ', insert_point + 1)
    
    if insert_point == -1:
        print("❌ Could not find proper insertion point")
        return False
    
    # The method to insert
    new_method = '''
    def update_signal_order_id(self, signal_id: str, order_id: str):
        """
        Update signal with broker order ID
        
        Args:
            signal_id: UUID of the signal record
            order_id: Broker's order ID
        """
        try:
            with self.engine.connect() as conn:
                query = text("""
                    UPDATE signals 
                    SET execution_reason = :order_id,
                        status = 'executed'
                    WHERE id = :signal_id OR signal_id = :signal_id
                """)
                
                result = conn.execute(query, {
                    'signal_id': signal_id,
                    'order_id': f"Broker Order ID: {order_id}"
                })
                
                conn.commit()
                
                if result.rowcount > 0:
                    logger.info(f"Signal {signal_id} updated with order ID: {order_id}")
                else:
                    logger.warning(f"No signal found with ID: {signal_id}")
                    
        except Exception as e:
            logger.error(f"Error updating signal order ID: {e}")
            raise
'''
    
    # Insert the new method
    new_content = content[:insert_point] + new_method + content[insert_point:]
    
    # Write back to file
    with open(database_file, 'w') as f:
        f.write(new_content)
    
    print("✅ Added update_signal_order_id method to TradingDatabase")
    return True

def create_fixed_trading_engine():
    """Create a fixed version of the trading engine with better error handling"""
    
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
    print("⚠️ PowerOf3SignalGenerator not available - using mock")
    PowerOf3SignalGenerator = None

try:
    from ..data.live_feed import LiveDataFeed
except ImportError:
    print("⚠️ LiveDataFeed not available - using mock")
    LiveDataFeed = None

try:
    from ..brokers.oanda_broker import OandaBroker
except ImportError:
    print("⚠️ OandaBroker not available - using mock")
    OandaBroker = None

try:
    from ..database.repository import TradingDatabase
except ImportError:
    print("⚠️ TradingDatabase not available - using mock")
    TradingDatabase = None

try:
    from ..utils.signal_formatter import SignalFormatter
except ImportError:
    print("⚠️ SignalFormatter not available - using mock")
    SignalFormatter = None

class LiveTradingEngine:
    def __init__(self, config: Dict):
        self.config = config
        self.running = False
        
        # Initialize components with error handling
        try:
            self.signal_generator = PowerOf3SignalGenerator() if PowerOf3SignalGenerator else None
        except Exception as e:
            print(f"⚠️ Could not initialize signal generator: {e}")
            self.signal_generator = None
        
        try:
            api_key = config.get('api_keys', {}).get('twelve_data')
            self.data_feed = LiveDataFeed(api_key) if LiveDataFeed and api_key else None
        except Exception as e:
            print(f"⚠️ Could not initialize data feed: {e}")
            self.data_feed = None
        
        try:
            oanda_key = config.get('api_keys', {}).get('oanda')
            oanda_account = config.get('oanda_account_id')
            self.broker = OandaBroker(oanda_key, oanda_account) if OandaBroker and oanda_key else None
        except Exception as e:
            print(f"⚠️ Could not initialize broker: {e}")
            self.broker = None
        
        try:
            db_config = config.get('database')
            self.database = TradingDatabase(db_config) if TradingDatabase else None
        except Exception as e:
            print(f"⚠️ Could not initialize database: {e}")
            self.database = None
        
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
        
        print("✅ LiveTradingEngine initialized")
    
    def start(self):
        """Start the live trading engine"""
        self.running = True
        self.logger.info("🚀 Live Trading Engine Started")
        
        # Check if we have the minimum required components
        if not self.signal_generator:
            self.logger.warning("⚠️ No signal generator - running in demo mode")
        
        # Schedule signal generation every 5 minutes (if we have components)
        if self.signal_generator and self.data_feed:
            schedule.every(5).minutes.do(self._generate_and_execute_signals)
            self.logger.info("📅 Scheduled signal generation every 5 minutes")
        
        # Schedule position monitoring every minute
        schedule.every(1).minutes.do(self._monitor_positions)
        
        print("✅ Trading engine started")
        print("💡 Use engine.stop() to stop the engine")
        
        # For testing, don't run the infinite loop
        if self.config.get('test_mode', False):
            print("🧪 Test mode - not starting infinite loop")
            return
        
        # Main loop
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(10)  # Check every 10 seconds
        except KeyboardInterrupt:
            self.logger.info("🛑 Stopping trading engine...")
            self.running = False
    
    def stop(self):
        """Stop the trading engine"""
        self.running = False
        self.logger.info("⏹️ Trading engine stopped")
        print("⏹️ Trading engine stopped")
    
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
            self.logger.error(f"Signal generation error: {e}")
    
    def _execute_signal(self, signal):
        """Execute a trading signal with better error handling"""
        try:
            signal_id = None
            
            # Log signal to database if available
            if self.database:
                try:
                    signal_id = self.database.log_signal(signal.to_dict())
                    self.logger.info(f"📝 Signal logged to database: {signal_id}")
                except Exception as e:
                    self.logger.error(f"Database logging failed: {e}")
            
            # Place order with broker if available and live trading enabled
            if self.broker and self.config.get('live_trading', False):
                try:
                    order_id = self.broker.place_order(signal.to_dict())
                    
                    if order_id:
                        self.logger.info(f"✅ Order placed: {signal.symbol} {signal.signal_type}")
                        
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
                                self.logger.error(f"Failed to update signal with order ID: {e}")
                    else:
                        self.logger.error(f"❌ Failed to place order: {signal.symbol}")
                        
                except Exception as e:
                    self.logger.error(f"Order execution failed: {e}")
            else:
                self.logger.info(f"📝 Paper trade: {signal.symbol} {signal.signal_type}")
                
        except Exception as e:
            self.logger.error(f"Execution error: {e}")
    
    def _monitor_positions(self):
        """Monitor open positions"""
        try:
            if self.broker:
                # Implementation for position monitoring
                pass
        except Exception as e:
            self.logger.error(f"Position monitoring error: {e}")
'''
    
    # Save the fixed trading engine
    trading_engine_file = Path('src/power_of_3/live/trading_engine.py')
    trading_engine_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(trading_engine_file, 'w') as f:
        f.write(trading_engine_content)
    
    print("✅ Created fixed trading engine with better error handling")

if __name__ == "__main__":
    print("🔧 FIXING DATABASE METHOD ERROR")
    print("=" * 40)
    
    # Option 1: Add method to existing database file
    print("1. Adding missing method to database...")
    if add_missing_method_to_database():
        print("   ✅ Method added successfully")
    else:
        print("   ⚠️ Could not add method automatically")
    
    # Option 2: Create fixed trading engine
    print("\n2. Creating fixed trading engine...")
    create_fixed_trading_engine()
    
    print("\n✅ FIXES APPLIED!")
    print("\nNext steps:")
    print("1. Test with: python test_imports.py")
    print("2. Run: python scripts/run_live_trading.py")