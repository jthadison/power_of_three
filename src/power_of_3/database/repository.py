"""
Database Integration for Power of 3 Trading System
==================================================

This module provides database connectivity and data persistence
for the Power of 3 trading system using PostgreSQL.

Features:
- Signal logging and tracking
- Trade execution recording
- Performance metrics calculation
- Risk management monitoring
- Historical analysis queries
"""

import psycopg2
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, date, timedelta
import pytz
from typing import Dict, List, Optional, Tuple
import logging
import os
from dataclasses import asdict
import uuid
from power_of_3_signal_generator import TradingSession

# Import Power of 3 classes (if available)
try:
    from power_of_3_signal_generator import (
        PowerOf3Signal, ManipulationPattern, LiquidityZone, 
        TradingSession, SignalType, SignalQuality
    )
    POWER_OF_3_AVAILABLE = True
except ImportError:
    POWER_OF_3_AVAILABLE = False

logger = logging.getLogger(__name__)

class TradingDatabase:
    """Main database interface for Power of 3 trading system"""
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize database connection
        
        Args:
            connection_string: PostgreSQL connection string
                              If None, will build from environment variables
        """
        if connection_string is None:
            connection_string = self._build_connection_string()
        
        self.connection_string = connection_string
        self.engine = create_engine(connection_string)
        self._test_connection()
    
    def _build_connection_string(self) -> str:
        """Build connection string from environment variables"""
        host = os.getenv('DB_HOST', 'localhost')
        port = os.getenv('DB_PORT', '5432')
        database = os.getenv('DB_NAME', 'trading_db')
        username = os.getenv('DB_USER', 'postgres')
        password = os.getenv('DB_PASSWORD', '')
        
        return f"postgresql://{username}:{password}@{host}:{port}/{database}"
    
    def _test_connection(self):
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                logger.info("Database connection successful")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    # =========================================================================
    # SIGNAL MANAGEMENT
    # =========================================================================
    
    def log_signal(self, signal: 'PowerOf3Signal', session_id: Optional[str] = None) -> str:
        """
        Log a Power of 3 signal to database
        
        Args:
            signal: PowerOf3Signal object
            session_id: UUID of trading session (optional)
            
        Returns:
            str: UUID of inserted signal record
        """
        try:
            with self.engine.connect() as conn:
                # Get or create session
                if session_id is None:
                    session_id = self._get_or_create_session(signal.session, signal.generated_at)
                
                # Insert signal
                query = text("""
                    INSERT INTO signals (
                        signal_id, symbol, signal_type, session_id, quality_score,
                        quality_level, confidence, entry_price, stop_loss,
                        take_profit_1, take_profit_2, take_profit_3,
                        risk_reward_ratio, position_size, risk_amount,
                        generated_at, expires_at, status
                    ) VALUES (
                        :signal_id, :symbol, :signal_type, :session_id, :quality_score,
                        :quality_level, :confidence, :entry_price, :stop_loss,
                        :take_profit_1, :take_profit_2, :take_profit_3,
                        :risk_reward_ratio, :position_size, :risk_amount,
                        :generated_at, :expires_at, 'pending'
                    ) RETURNING id
                """)
                
                result = conn.execute(query, {
                    'signal_id': signal.signal_id,
                    'symbol': signal.symbol,
                    'signal_type': signal.signal_type.value,
                    'session_id': session_id,
                    'quality_score': signal.quality_score,
                    'quality_level': signal.quality_level.value,
                    'confidence': signal.confidence,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit_1': signal.take_profit_1,
                    'take_profit_2': signal.take_profit_2,
                    'take_profit_3': signal.take_profit_3,
                    'risk_reward_ratio': signal.risk_reward_ratio,
                    'position_size': signal.position_size,
                    'risk_amount': signal.risk_amount,
                    'generated_at': signal.generated_at,
                    'expires_at': signal.expires_at
                })
                
                conn.commit()
                row = result.fetchone()
                if row is None:
                    raise ValueError("No signal ID returned from the database.")
                signal_uuid = row[0]
                logger.info(f"Signal logged: {signal.signal_id}")
                return str(signal_uuid)
                
        except Exception as e:
            logger.error(f"Error logging signal: {e}")
            raise
    
    def update_signal_status(self, signal_id: str, status: str, reason: Optional[str] = None):
        """Update signal status (executed, expired, cancelled)"""
        try:
            with self.engine.connect() as conn:
                query = text("""
                    UPDATE signals 
                    SET status = :status, execution_reason = :reason
                    WHERE signal_id = :signal_id
                """)
                
                conn.execute(query, {
                    'signal_id': signal_id,
                    'status': status,
                    'reason': reason
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating signal status: {e}")
            raise
    
    # =========================================================================
    # TRADE MANAGEMENT
    # =========================================================================
    
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

    def log_trade_entry(self, trade_data: Dict) -> str:
        """
        Log trade entry to database
        
        Args:
            trade_data: Dictionary containing trade entry details
            
        Returns:
            str: UUID of trade entry record
        """
        try:
            with self.engine.connect() as conn:
                query = text("""
                    INSERT INTO trade_entries (
                        signal_id, symbol, trade_type, entry_price, entry_time,
                        position_size, stop_loss, take_profit_1, take_profit_2,
                        take_profit_3, risk_amount, account_balance_at_entry,
                        risk_percentage, execution_method, slippage_pips,
                        commission, spread, notes
                    ) VALUES (
                        :signal_id, :symbol, :trade_type, :entry_price, :entry_time,
                        :position_size, :stop_loss, :take_profit_1, :take_profit_2,
                        :take_profit_3, :risk_amount, :account_balance_at_entry,
                        :risk_percentage, :execution_method, :slippage_pips,
                        :commission, :spread, :notes
                    ) RETURNING id
                """)
                
                result = conn.execute(query, trade_data)
                conn.commit()
                row = result.fetchone()
                if row is None:
                    raise ValueError("No trade ID returned from the database.")
                trade_id = row[0]
                
                logger.info(f"Trade entry logged: {trade_data['symbol']} {trade_data['trade_type']}")
                return str(trade_id)
                
        except Exception as e:
            logger.error(f"Error logging trade entry: {e}")
            raise
    
    def log_trade_exit(self, exit_data: Dict) -> str:
        """Log trade exit to database"""
        try:
            with self.engine.connect() as conn:
                query = text("""
                    INSERT INTO trade_exits (
                        trade_entry_id, exit_type, exit_price, exit_time,
                        position_size_closed, remaining_position, gross_pnl,
                        commission, net_pnl, pips_gained, percentage_return,
                        slippage_pips, execution_method, notes
                    ) VALUES (
                        :trade_entry_id, :exit_type, :exit_price, :exit_time,
                        :position_size_closed, :remaining_position, :gross_pnl,
                        :commission, :net_pnl, :pips_gained, :percentage_return,
                        :slippage_pips, :execution_method, :notes
                    ) RETURNING id
                """)
                
                result = conn.execute(query, exit_data)
                conn.commit()
                
                # Update trade entry status if fully closed
                if exit_data.get('remaining_position', 0) == 0:
                    self._update_trade_entry_status(
                        exit_data['trade_entry_id'], 
                        'closed' if exit_data['exit_type'] != 'stop_loss' else 'stopped_out'
                    )
                else:
                    self._update_trade_entry_status(exit_data['trade_entry_id'], 'partially_closed')
                
                row = result.fetchone()
                if row is None:
                    raise ValueError("No exit ID returned from the database.")
                exit_id = row[0]
                logger.info(f"Trade exit logged: {exit_data['exit_type']}")
                return str(exit_id)
                
        except Exception as e:
            logger.error(f"Error logging trade exit: {e}")
            raise
    
    def _update_trade_entry_status(self, trade_id: str, status: str):
        """Update trade entry status"""
        with self.engine.connect() as conn:
            query = text("UPDATE trade_entries SET status = :status WHERE id = :trade_id")
            conn.execute(query, {'trade_id': trade_id, 'status': status})
            conn.commit()
    
    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================
    
    def _get_or_create_session(self, session_type: 'TradingSession', timestamp: datetime) -> str:
        """Get existing session or create new one"""
        try:
            with self.engine.connect() as conn:
                session_date = timestamp.date()
                
                # Try to find existing session
                query = text("""
                    SELECT id FROM trading_sessions 
                    WHERE session_date = :session_date AND session_type = :session_type
                """)
                
                result = conn.execute(query, {
                    'session_date': session_date,
                    'session_type': session_type.value
                })
                
                existing = result.fetchone()
                if existing:
                    return str(existing[0])
                
                # Create new session
                # Calculate session start/end times based on type and date
                est_tz = pytz.timezone('US/Eastern')
                session_date_dt = datetime.combine(session_date, datetime.min.time())
                session_date_dt = est_tz.localize(session_date_dt)
                
                if session_type.value == 'london_open':
                    start_time = session_date_dt.replace(hour=2, minute=0)
                    end_time = session_date_dt.replace(hour=5, minute=0)
                elif session_type.value == 'new_york_open':
                    start_time = session_date_dt.replace(hour=7, minute=0)
                    end_time = session_date_dt.replace(hour=10, minute=0)
                else:  # london_close
                    start_time = session_date_dt.replace(hour=10, minute=0)
                    end_time = session_date_dt.replace(hour=12, minute=0)
                
                create_query = text("""
                    INSERT INTO trading_sessions (
                        session_date, session_type, start_time, end_time
                    ) VALUES (
                        :session_date, :session_type, :start_time, :end_time
                    ) RETURNING id
                """)
                
                result = conn.execute(create_query, {
                    'session_date': session_date,
                    'session_type': session_type.value,
                    'start_time': start_time,
                    'end_time': end_time
                })
                
                conn.commit()
                row = result.fetchone()
                if row is None:
                    raise ValueError("No session ID returned from the database.")
                session_id = row[0]
                return str(session_id)
                
        except Exception as e:
            logger.error(f"Error managing session: {e}")
            raise
    
    # =========================================================================
    # PERFORMANCE ANALYTICS
    # =========================================================================
    
    def get_daily_performance(self, days: int = 30) -> pd.DataFrame:
        """Get daily performance metrics for last N days"""
        query = """
        SELECT * FROM daily_performance 
        WHERE trade_date >= CURRENT_DATE - INTERVAL '%s days'
        ORDER BY trade_date DESC
        """ % days
        
        return pd.read_sql(query, self.engine)
    
    def get_session_performance(self, days: int = 30) -> pd.DataFrame:
        """Get session performance breakdown"""
        query = """
        SELECT * FROM session_performance 
        WHERE session_date >= CURRENT_DATE - INTERVAL '%s days'
        ORDER BY session_date DESC, session_type
        """ % days
        
        return pd.read_sql(query, self.engine)
    
    def get_symbol_performance(self) -> pd.DataFrame:
        """Get performance breakdown by symbol"""
        query = "SELECT * FROM symbol_performance ORDER BY total_pnl DESC"
        return pd.read_sql(query, self.engine)
    
    def get_open_trades(self) -> pd.DataFrame:
        """Get current open trades"""
        query = "SELECT * FROM open_trades_summary ORDER BY entry_time DESC"
        return pd.read_sql(query, self.engine)
    
    def calculate_performance_metrics(self, start_date: date, end_date: date, 
                                     period_type: str = 'daily') -> Dict:
        """Calculate comprehensive performance metrics for a period"""
        try:
            with self.engine.connect() as conn:
                # Get all closed trades in period
                query = text("""
                    SELECT 
                        te.symbol, te.trade_type, te.entry_price, te.entry_time,
                        te.position_size, te.risk_amount, te.account_balance_at_entry,
                        tex.exit_price, tex.exit_time, tex.net_pnl, tex.gross_pnl,
                        tex.pips_gained, s.quality_score, s.risk_reward_ratio
                    FROM trade_entries te
                    JOIN trade_exits tex ON te.id = tex.trade_entry_id
                    LEFT JOIN signals s ON te.signal_id = s.id
                    WHERE te.entry_time::date BETWEEN :start_date AND :end_date
                    AND tex.exit_type IN ('take_profit_1', 'take_profit_2', 'take_profit_3', 
                                         'stop_loss', 'manual_close')
                """)
                
                trades_df = pd.read_sql(query, conn, params={
                    'start_date': start_date,
                    'end_date': end_date
                })
                
                if trades_df.empty:
                    return self._empty_performance_metrics()
                
                # Calculate metrics
                total_trades = len(trades_df)
                winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
                losing_trades = len(trades_df[trades_df['net_pnl'] < 0])
                
                gross_profit = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum()
                gross_loss = trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum()
                net_profit = trades_df['net_pnl'].sum()
                
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
                avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
                profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else 999.99
                
                # Advanced metrics
                returns = trades_df['net_pnl'] / trades_df['account_balance_at_entry']
                sharpe_ratio = returns.mean() / returns.std() if returns.std() != 0 else 0
                
                # Calculate max drawdown
                cumulative_pnl = trades_df['net_pnl'].cumsum()
                running_max = cumulative_pnl.cummax()
                drawdown = cumulative_pnl - running_max
                max_drawdown = abs(drawdown.min())
                
                metrics = {
                    'period_start': start_date,
                    'period_end': end_date,
                    'period_type': period_type,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': round(win_rate, 2),
                    'gross_profit': round(gross_profit, 2),
                    'gross_loss': round(gross_loss, 2),
                    'net_profit': round(net_profit, 2),
                    'average_win': round(avg_win, 2),
                    'average_loss': round(avg_loss, 2),
                    'largest_win': round(trades_df['net_pnl'].max(), 2),
                    'largest_loss': round(trades_df['net_pnl'].min(), 2),
                    'profit_factor': round(profit_factor, 2),
                    'sharpe_ratio': round(sharpe_ratio, 3),
                    'max_drawdown': round(max_drawdown, 2),
                    'avg_signal_quality': round(trades_df['quality_score'].mean(), 2),
                    'avg_risk_reward': round(trades_df['risk_reward_ratio'].mean(), 2)
                }
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            raise
    
    def _empty_performance_metrics(self) -> Dict:
        """Return empty performance metrics structure"""
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0.0, 'gross_profit': 0.0, 'gross_loss': 0.0,
            'net_profit': 0.0, 'average_win': 0.0, 'average_loss': 0.0,
            'largest_win': 0.0, 'largest_loss': 0.0, 'profit_factor': 0.0,
            'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'avg_signal_quality': 0.0,
            'avg_risk_reward': 0.0
        }
    
    def save_performance_metrics(self, metrics: Dict):
        """Save calculated performance metrics to database"""
        try:
            with self.engine.connect() as conn:
                query = text("""
                    INSERT INTO performance_metrics (
                        period_start, period_end, period_type, starting_balance,
                        ending_balance, net_profit, gross_profit, gross_loss,
                        total_trades, winning_trades, losing_trades, win_rate,
                        average_win, average_loss, largest_win, largest_loss,
                        profit_factor, sharpe_ratio, max_drawdown, avg_signal_quality,
                        avg_risk_reward
                    ) VALUES (
                        :period_start, :period_end, :period_type, :starting_balance,
                        :ending_balance, :net_profit, :gross_profit, :gross_loss,
                        :total_trades, :winning_trades, :losing_trades, :win_rate,
                        :average_win, :average_loss, :largest_win, :largest_loss,
                        :profit_factor, :sharpe_ratio, :max_drawdown, :avg_signal_quality,
                        :avg_risk_reward
                    )
                """)
                
                conn.execute(query, metrics)
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")
            raise
    
    # =========================================================================
    # RISK MANAGEMENT
    # =========================================================================
    
    def update_daily_risk(self, date: date, risk_data: Dict):
        """Update daily risk management data"""
        try:
            with self.engine.connect() as conn:
                # Check if record exists
                check_query = text("SELECT id FROM risk_management WHERE date = :date")
                result = conn.execute(check_query, {'date': date})
                
                if result.fetchone():
                    # Update existing
                    query = text("""
                        UPDATE risk_management SET
                            account_balance = :account_balance,
                            total_risk_amount = :total_risk_amount,
                            current_daily_risk = :current_daily_risk,
                            open_positions = :open_positions,
                            current_drawdown = :current_drawdown,
                            risk_status = :risk_status
                        WHERE date = :date
                    """)
                else:
                    # Insert new
                    query = text("""
                        INSERT INTO risk_management (
                            date, account_balance, total_risk_amount, max_daily_risk,
                            current_daily_risk, open_positions, current_drawdown, risk_status
                        ) VALUES (
                            :date, :account_balance, :total_risk_amount, :max_daily_risk,
                            :current_daily_risk, :open_positions, :current_drawdown, :risk_status
                        )
                    """)
                
                risk_data['date'] = date
                conn.execute(query, risk_data)
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating risk management: {e}")
            raise
    
    # =========================================================================
    # REPORTING AND ANALYSIS
    # =========================================================================
    
    def generate_trading_report(self, days: int = 30) -> Dict:
        """Generate comprehensive trading report"""
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        report = {
            'period': f"{start_date} to {end_date}",
            'daily_performance': self.get_daily_performance(days).to_dict('records'),
            'session_performance': self.get_session_performance(days).to_dict('records'),
            'symbol_performance': self.get_symbol_performance().to_dict('records'),
            'open_trades': self.get_open_trades().to_dict('records'),
            'performance_metrics': self.calculate_performance_metrics(start_date, end_date)
        }
        
        return report
    
    def get_signal_quality_analysis(self, days: int = 30) -> pd.DataFrame:
        """Analyze signal quality vs actual performance"""
        query = f"""
        SELECT 
            s.quality_level,
            s.quality_score,
            COUNT(te.id) as trades_taken,
            AVG(tex.net_pnl) as avg_pnl,
            SUM(tex.net_pnl) as total_pnl,
            AVG(CASE WHEN tex.net_pnl > 0 THEN 1.0 ELSE 0.0 END) * 100 as win_rate,
            AVG(s.risk_reward_ratio) as avg_expected_rr,
            AVG(ABS(tex.net_pnl / te.risk_amount)) as avg_actual_rr
        FROM signals s
        JOIN trade_entries te ON s.id = te.signal_id
        JOIN trade_exits tex ON te.id = tex.trade_entry_id
        WHERE s.generated_at >= CURRENT_DATE - INTERVAL '{days} days'
        GROUP BY s.quality_level, s.quality_score
        ORDER BY s.quality_score DESC
        """
        
        return pd.read_sql(query, self.engine)
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'engine'):
            self.engine.dispose()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def setup_database():
    """Setup database with schema (run once)"""
    try:
        db = TradingDatabase()
        logger.info("Database setup completed successfully")
        return db
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise

def create_sample_data(db: TradingDatabase):
    """Create sample data for testing"""
    try:
        # Sample session
        session_id = db._get_or_create_session(
            TradingSession.NEW_YORK_OPEN, 
            datetime.now(pytz.timezone('US/Eastern'))
        )
        
        # Sample trade entry
        trade_data = {
            'signal_id': None,
            'symbol': 'US30',
            'trade_type': 'long',
            'entry_price': 42250.50,
            'entry_time': datetime.now(pytz.timezone('US/Eastern')),
            'position_size': 0.1,
            'stop_loss': 42200.00,
            'take_profit_1': 42350.00,
            'take_profit_2': 42450.00,
            'take_profit_3': 42550.00,
            'risk_amount': 200.00,
            'account_balance_at_entry': 10000.00,
            'risk_percentage': 2.0,
            'execution_method': 'manual',
            'slippage_pips': 0.5,
            'commission': 5.00,
            'spread': 2.0,
            'notes': 'Sample trade for testing'
        }
        
        trade_id = db.log_trade_entry(trade_data)
        
        # Sample exit
        exit_data = {
            'trade_entry_id': trade_id,
            'exit_type': 'take_profit_1',
            'exit_price': 42350.00,
            'exit_time': datetime.now(pytz.timezone('US/Eastern')),
            'position_size_closed': 0.1,
            'remaining_position': 0.0,
            'gross_pnl': 99.50,
            'commission': 5.00,
            'net_pnl': 94.50,
            'pips_gained': 99.5,
            'percentage_return': 0.945,
            'slippage_pips': 0.0,
            'execution_method': 'manual',
            'notes': 'Sample exit'
        }
        
        db.log_trade_exit(exit_data)
        
        logger.info("Sample data created successfully")
        
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")
        raise

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize database
        db = TradingDatabase()
        
        # Generate report
        report = db.generate_trading_report(30)
        print("Trading Report Generated:")
        print(f"Period: {report['period']}")
        print(f"Performance Metrics: {report['performance_metrics']}")
        
        # Close connection
        db.close()
        
    except Exception as e:
        print(f"Error: {e}")