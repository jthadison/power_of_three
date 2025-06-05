# Power of 3 Backtesting Engine
# Save the 'Integrated Backtesting Engine' artifact as this file
"""
Power of 3 Integrated Backtesting Engine
========================================

Backtesting engine that integrates with your existing architecture.

File Location: src/power_of_3/backtesting/engine.py
"""

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.power_of_3.data.providers.data_provider import DataProvider

from ..config.settings import load_config
from ..core.signal_generator import PowerOf3SignalGenerator

# Import from your existing architecture
#from src.power_of_3.core.signal_generator import PowerOf3SignalGenerator
from ..core.types import PowerOf3Signal
from ..database.repository import TradingDatabase

# Try to import your data provider
try:
    from ..data.providers.data_provider import DataProvider
    DATA_PROVIDER_AVAILABLE = True
except ImportError:
    DATA_PROVIDER_AVAILABLE = False
    # Fallback data provider using yfinance
    import yfinance as yf

logger = logging.getLogger(__name__)

@dataclass
class BacktestTrade:
    """Individual backtest trade record"""
    trade_id: str
    signal_id: str
    symbol: str
    signal_type: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_type: Optional[str] = None  # 'tp1', 'tp2', 'tp3', 'sl', 'manual'
    position_size: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    quality_score: float = 0.0
    risk_reward_ratio: float = 0.0
    status: str = 'open'  # 'open', 'closed', 'stopped_out'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Convert datetime objects to strings
        if self.entry_time:
            result['entry_time'] = self.entry_time.isoformat()
        if self.exit_time:
            result['exit_time'] = self.exit_time.isoformat()
        return result

@dataclass
class BacktestResults:
    """Complete backtest results"""
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: timedelta = timedelta()
    avg_signal_quality: float = 0.0
    avg_risk_reward: float = 0.0
    trades: Optional[List[BacktestTrade]] = None
    equity_curve: Optional[List[Dict]] = None
    session_performance: Optional[Dict] = None
    
    def __post_init__(self):
        if self.trades is None:
            self.trades = []
        if self.equity_curve is None:
            self.equity_curve = []
        if self.session_performance is None:
            self.session_performance = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['start_date'] = self.start_date.isoformat()
        result['end_date'] = self.end_date.isoformat()
        result['avg_trade_duration'] = str(self.avg_trade_duration)
        result['trades'] = [trade.to_dict() for trade in self.trades] if self.trades else []
        return result

class FallbackDataProvider:
    """Fallback data provider using yfinance if DataProvider not available"""
    
    #@staticmethod
    def get_historical_data(self, symbol: str, start_date: str, end_date: str, 
                           interval: str = '5min') -> pd.DataFrame:
        """Get historical data using yfinance"""
        try:
            # Symbol mapping for yfinance
            symbol_map = {
                'US30': '^DJI',
                'NAS100': '^IXIC',
                'SPX500': '^GSPC',
                'XAUUSD': 'GC=F'
            }
            
            yf_symbol = symbol_map.get(symbol, symbol)
            
            # Interval mapping
            interval_map = {
                '1min': '1m',
                '5min': '5m',
                '15min': '15m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d'
            }
            
            yf_interval = interval_map.get(interval, '5m')
            
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(start=start_date, end=end_date, interval=yf_interval)
            
            df = self._standardize_column_names(df)
            
            if df.empty:
                logger.warning(f"No data retrieved for {symbol}")
                return pd.DataFrame()
            
            # Ensure column names are correct
            if 'Adj Close' in df.columns:
                df = df.drop('Adj Close', axis=1)
            
            logger.info(f"Retrieved {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {e}")
            return pd.DataFrame()
        
    def _standardize_column_names(self, df):
        """
        Standardize column names from different data providers
        Yahoo Finance: 'High', 'Low', 'Open', 'Close', 'Volume' 
        Power of 3 expects: 'high', 'low', 'open', 'close', 'volume'
        """
        if df.empty:
            return df
        
        # Column mapping from provider format to Power of 3 format
        column_mapping = {
            # Yahoo Finance format -> Power of 3 format
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            
            # Handle other possible formats
            'OPEN': 'open',
            'HIGH': 'high',
            'LOW': 'low', 
            'CLOSE': 'close',
            'VOLUME': 'volume',
            
            # Some providers use different names
            'Adj Close': 'close',  # Yahoo sometimes has this
            'close_price': 'close',
            'high_price': 'high',
            'low_price': 'low',
            'open_price': 'open'
        }
        
        # Only rename columns that exist in the DataFrame
        columns_to_rename = {
            old_name: new_name 
            for old_name, new_name in column_mapping.items() 
            if old_name in df.columns
        }
        
        if columns_to_rename:
            df = df.rename(columns=columns_to_rename)
            logger.info(f"Standardized columns: {list(columns_to_rename.keys())} -> {list(columns_to_rename.values())}")
        
        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns after standardization: {missing_columns}")
        
        return df

class PowerOf3BacktestEngine:
    """
    Integrated backtesting engine for Power of 3 trading system.
    Uses your existing signal generator, database, and configuration.
    """
    
    def __init__(self, config: Optional[Dict] = None, use_database: bool = False):
        """
        Initialize backtesting engine
        
        Args:
            config: Configuration dictionary (will load from settings if None)
            use_database: Whether to use database for trade logging
        """
        self.config = config or load_config()
        self.use_database = use_database
        
        # Initialize components from your existing architecture
        self.signal_generator = PowerOf3SignalGenerator(
            min_risk_reward=self.config.get('trading', {}).get('min_risk_reward', 5.0),
            max_risk_percent=self.config.get('trading', {}).get('max_risk_percent', 2.0)
        )
        
        # Initialize data provider
        if DATA_PROVIDER_AVAILABLE:
            self.data_provider = DataProvider()
        else:
            self.data_provider = FallbackDataProvider()
            logger.info("Using fallback data provider (yfinance)")
        
        # Initialize database if requested
        self.database = None
        if use_database:
            try:
                self.database = TradingDatabase()
                logger.info("Database integration enabled")
            except Exception as e:
                logger.warning(f"Database initialization failed: {e}")
                self.use_database = False
        
        # Backtest state
        self.current_balance = 0.0
        self.initial_balance = 0.0
        self.open_trades = {}
        self.closed_trades = []
        self.equity_curve = []
        
    async def run_backtest(self, symbol: str, start_date: str, end_date: str,
                          initial_balance: float = 10000.0,
                          timeframe: str = '5min') -> BacktestResults:
        """
        Run backtest for a single symbol
        
        Args:
            symbol: Trading symbol (e.g., 'US30', 'XAUUSD')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            initial_balance: Starting account balance
            timeframe: Data timeframe ('5min', '15min', '1h', etc.)
            
        Returns:
            BacktestResults object with complete results
        """
        logger.info(f"Starting backtest: {symbol} from {start_date} to {end_date}")
        
        # Initialize backtest state
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.open_trades = {}
        self.closed_trades = []
        self.equity_curve = []
        
        # Get historical data
        df = self._get_historical_data(symbol, start_date, end_date, timeframe)
        
        df = self._standardize_column_names(df)
        
        print(f"🔍 DEBUG - Columns after standardization: {df.columns.tolist()}")
        if 'High' in df.columns:
            print("❌ ERROR: 'High' still exists after standardization!")
        if 'high' not in df.columns:
            print("❌ ERROR: 'high' column missing!")

        # Verify the data is properly formatted
        print(f"Sample data:\n{df.head(2)}")
        
        if df.empty:
            raise ValueError(f"No historical data available for {symbol}")
        
        logger.info(f"Processing {len(df)} bars for backtesting")
        
        # Process each bar
        for i in range(len(df)):
            current_bar = df.iloc[i]
            current_time = df.index[i]
            
            # Get data up to current bar for signal generation
            historical_data = df.iloc[:i+1]
            
            # Skip if insufficient data for signal generation
            if len(historical_data) < 100:
                continue
            
            # Generate signals using your existing signal generator
            signals = self.signal_generator.generate_signals(
                symbol, historical_data, self.current_balance
            )
            
            # Process new signals
            for signal in signals:
                if signal.quality_score >= 7.0:  # Only trade high-quality signals
                    self._open_trade(signal, current_bar, current_time)
            
            # Check existing trades for exits
            self._process_open_trades(current_bar, current_time)
            
            # Update equity curve
            self._update_equity_curve(current_time)
        
        # Close any remaining open trades at end of backtest
        final_bar = df.iloc[-1]
        final_time = df.index[-1]
        for trade_id in list(self.open_trades.keys()):
            self._close_trade(trade_id, final_bar['Close'], final_time, 'manual')
        
        # Calculate final results
        results = self._calculate_results(symbol, start_date, end_date)
        
        # Save to database if enabled
        if self.use_database and self.database:
            await self._save_results_to_database(results)
        
        logger.info(f"Backtest completed: {results.total_trades} trades, "
                   f"{results.win_rate:.1f}% win rate, "
                   f"{results.total_pnl_pct:.2f}% return")
        
        return results
    
    def _standardize_column_names(self, df):
        """
        Standardize column names from different data providers
        Yahoo Finance: 'High', 'Low', 'Open', 'Close', 'Volume' 
        Power of 3 expects: 'high', 'low', 'open', 'close', 'volume'
        """
        if df.empty:
            return df
        
        # Column mapping from provider format to Power of 3 format
        column_mapping = {
            # Yahoo Finance format -> Power of 3 format
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            
            # Handle other possible formats
            'OPEN': 'open',
            'HIGH': 'high',
            'LOW': 'low', 
            'CLOSE': 'close',
            'VOLUME': 'volume',
            
            # Some providers use different names
            'Adj Close': 'close',  # Yahoo sometimes has this
            'close_price': 'close',
            'high_price': 'high',
            'low_price': 'low',
            'open_price': 'open'
        }
        
        # Only rename columns that exist in the DataFrame
        columns_to_rename = {
            old_name: new_name 
            for old_name, new_name in column_mapping.items() 
            if old_name in df.columns
        }
        
        if columns_to_rename:
            df = df.rename(columns=columns_to_rename)
            logger.info(f"Standardized columns: {list(columns_to_rename.keys())} -> {list(columns_to_rename.values())}")
        
        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns after standardization: {missing_columns}")
        
        return df
    
    async def run_multi_symbol_backtest(self, symbols: List[str], start_date: str, 
                                       end_date: str, initial_balance: float = 10000.0,
                                       timeframe: str = '5min') -> Dict[str, BacktestResults]:
        """Run backtest on multiple symbols"""
        results = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Running backtest for {symbol}")
                result = await self.run_backtest(
                    symbol, start_date, end_date, initial_balance, timeframe
                )
                results[symbol] = result
            except Exception as e:
                logger.error(f"Error backtesting {symbol}: {e}")
                continue
        
        return results
    
    def _get_historical_data(self, symbol: str, start_date: str, 
                                   end_date: str, timeframe: str) -> pd.DataFrame:
        """Get historical data using available data provider"""
        try:
            if DATA_PROVIDER_AVAILABLE:
                # Use your existing data provider
                df = self.data_provider.get_historical_data(
                    symbol, start_date, end_date, timeframe
                )
            else:
                # Use fallback provider
                df = self.data_provider.get_historical_data(
                    symbol, start_date, end_date, timeframe
                )
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()
    
    def _open_trade(self, signal: PowerOf3Signal, current_bar: pd.Series, 
                   current_time: datetime):
        """Open a new trade based on signal"""
        try:
            # Calculate position size based on risk management
            risk_amount = self.current_balance * (signal.risk_amount / 10000.0)  # Assuming signal.risk_amount is in dollars
            position_size = signal.position_size
            
            # Create trade record
            trade = BacktestTrade(
                trade_id=f"{signal.signal_id}_{len(self.closed_trades)}",
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                signal_type=signal.signal_type.value,
                entry_time=current_time,
                entry_price=signal.entry_price,
                position_size=position_size,
                quality_score=signal.quality_score,
                risk_reward_ratio=signal.risk_reward_ratio,
                status='open'
            )
            
            self.open_trades[trade.trade_id] = {
                'trade': trade,
                'signal': signal
            }
            
            logger.debug(f"Opened trade: {trade.trade_id} at {signal.entry_price}")
            
        except Exception as e:
            logger.error(f"Error opening trade: {e}")
    
    def _process_open_trades(self, current_bar: pd.Series, current_time: datetime):
        """Process all open trades for potential exits"""
        trades_to_close = []
        
        for trade_id, trade_data in self.open_trades.items():
            trade = trade_data['trade']
            signal = trade_data['signal']
            
            current_price = current_bar['Close']
            high_price = current_bar['High']
            low_price = current_bar['Low']
            
            # Check for exits based on signal type
            if trade.signal_type == 'long':
                # Check stop loss
                if low_price <= signal.stop_loss:
                    trades_to_close.append((trade_id, signal.stop_loss, 'sl'))
                # Check take profits (in order)
                elif high_price >= signal.take_profit_1:
                    trades_to_close.append((trade_id, signal.take_profit_1, 'tp1'))
                elif high_price >= signal.take_profit_2:
                    trades_to_close.append((trade_id, signal.take_profit_2, 'tp2'))
                elif high_price >= signal.take_profit_3:
                    trades_to_close.append((trade_id, signal.take_profit_3, 'tp3'))
            
            elif trade.signal_type == 'short':
                # Check stop loss
                if high_price >= signal.stop_loss:
                    trades_to_close.append((trade_id, signal.stop_loss, 'sl'))
                # Check take profits (in order)
                elif low_price <= signal.take_profit_1:
                    trades_to_close.append((trade_id, signal.take_profit_1, 'tp1'))
                elif low_price <= signal.take_profit_2:
                    trades_to_close.append((trade_id, signal.take_profit_2, 'tp2'))
                elif low_price <= signal.take_profit_3:
                    trades_to_close.append((trade_id, signal.take_profit_3, 'tp3'))
        
        # Close trades
        for trade_id, exit_price, exit_type in trades_to_close:
            self._close_trade(trade_id, exit_price, current_time, exit_type)
    
    def _close_trade(self, trade_id: str, exit_price: float, 
                    exit_time: datetime, exit_type: str):
        """Close a trade and calculate P&L"""
        if trade_id not in self.open_trades:
            return
        
        trade_data = self.open_trades[trade_id]
        trade = trade_data['trade']
        
        # Update trade record
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_type = exit_type
        trade.status = 'stopped_out' if exit_type == 'sl' else 'closed'
        
        # Calculate P&L
        if trade.signal_type == 'long':
            trade.pnl = (exit_price - trade.entry_price) * trade.position_size
        else:  # short
            trade.pnl = (trade.entry_price - exit_price) * trade.position_size
        
        trade.pnl_pct = (trade.pnl / self.current_balance) * 100
        
        # Update balance
        self.current_balance += trade.pnl
        
        # Move to closed trades
        self.closed_trades.append(trade)
        del self.open_trades[trade_id]
        
        logger.debug(f"Closed trade: {trade_id} P&L: {trade.pnl:.2f}")
    
    def _update_equity_curve(self, current_time: datetime):
        """Update equity curve with current balance"""
        total_pnl = sum(trade.pnl for trade in self.closed_trades)
        unrealized_pnl = sum(
            self._calculate_unrealized_pnl(trade_data) 
            for trade_data in self.open_trades.values()
        )
        
        equity_point = {
            'timestamp': current_time.isoformat(),
            'balance': self.current_balance,
            'total_pnl': total_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_equity': self.current_balance + unrealized_pnl
        }
        
        self.equity_curve.append(equity_point)
    
    def _calculate_unrealized_pnl(self, trade_data: Dict) -> float:
        """Calculate unrealized P&L for open trade"""
        # This would need current market price, but in backtest we use last close
        # For simplicity, return 0 here - in real implementation you'd use current bar price
        return 0.0
    
    def _calculate_results(self, symbol: str, start_date: str, end_date: str) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        
        if not self.closed_trades:
            return BacktestResults(
                symbol=symbol,
                start_date=datetime.fromisoformat(start_date),
                end_date=datetime.fromisoformat(end_date),
                initial_balance=self.initial_balance,
                final_balance=self.current_balance,
                trades=self.closed_trades,
                equity_curve=self.equity_curve
            )
        
        # Basic metrics
        total_trades = len(self.closed_trades)
        winning_trades = sum(1 for trade in self.closed_trades if trade.pnl > 0)
        losing_trades = sum(1 for trade in self.closed_trades if trade.pnl < 0)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = sum(trade.pnl for trade in self.closed_trades)
        total_pnl_pct = (total_pnl / self.initial_balance) * 100
        
        wins = [trade.pnl for trade in self.closed_trades if trade.pnl > 0]
        losses = [trade.pnl for trade in self.closed_trades if trade.pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0
        
        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.99
        
        # Drawdown calculation
        equity_values = [point['total_equity'] for point in self.equity_curve]
        if equity_values:
            running_max = np.maximum.accumulate(equity_values)
            drawdowns = np.array(equity_values) - running_max
            max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0
            max_drawdown_pct = (max_drawdown / self.initial_balance) * 100
        else:
            max_drawdown = 0
            max_drawdown_pct = 0
        
        # Sharpe ratio (simplified)
        if len(self.closed_trades) > 1:
            returns = [trade.pnl_pct for trade in self.closed_trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Average trade duration
        durations = []
        for trade in self.closed_trades:
            if trade.exit_time and trade.entry_time:
                duration = trade.exit_time - trade.entry_time
                durations.append(duration)
        
        if durations:
            mean_duration_seconds = float(np.mean([duration.total_seconds() for duration in durations]))
            avg_trade_duration = timedelta(seconds=mean_duration_seconds)
        else:
            avg_trade_duration = timedelta()
        
        # Power of 3 specific metrics
        avg_signal_quality = np.mean([trade.quality_score for trade in self.closed_trades])
        avg_risk_reward = np.mean([trade.risk_reward_ratio for trade in self.closed_trades])
        
        # Session performance
        session_performance = self._calculate_session_performance()
        
        return BacktestResults(
            symbol=symbol,
            start_date=datetime.fromisoformat(start_date),
            end_date=datetime.fromisoformat(end_date),
            initial_balance=self.initial_balance,
            final_balance=self.current_balance,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=float(sharpe_ratio),
            profit_factor=profit_factor,
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration=avg_trade_duration,
            avg_signal_quality=float(avg_signal_quality),
            avg_risk_reward=float(avg_risk_reward),
            trades=self.closed_trades,
            equity_curve=self.equity_curve,
            session_performance=session_performance
        )
    
    def _calculate_session_performance(self) -> Dict:
        """Calculate performance by Power of 3 sessions"""
        session_stats = {}
        
        for session in ['london_open', 'new_york_open', 'london_close']:
            session_trades = [
                trade for trade in self.closed_trades 
                if self._get_session_from_time(trade.entry_time) == session
            ]
            
            if session_trades:
                session_pnl = sum(trade.pnl for trade in session_trades)
                session_wins = sum(1 for trade in session_trades if trade.pnl > 0)
                session_total = len(session_trades)
                session_win_rate = (session_wins / session_total) * 100
                
                session_stats[session] = {
                    'trades': session_total,
                    'wins': session_wins,
                    'win_rate': session_win_rate,
                    'total_pnl': session_pnl,
                    'avg_pnl': session_pnl / session_total
                }
            else:
                session_stats[session] = {
                    'trades': 0, 'wins': 0, 'win_rate': 0,
                    'total_pnl': 0, 'avg_pnl': 0
                }
        
        return session_stats
    
    def _get_session_from_time(self, timestamp: datetime) -> str:
        """Determine Power of 3 session from timestamp"""
        import pytz
        
        # Convert to EST
        est_tz = pytz.timezone('US/Eastern')
        if timestamp.tzinfo is None:
            timestamp = est_tz.localize(timestamp)
        else:
            timestamp = timestamp.astimezone(est_tz)
        
        hour = timestamp.hour
        
        if 2 <= hour < 5:
            return 'london_open'
        elif 7 <= hour < 10:
            return 'new_york_open'
        elif 10 <= hour < 12:
            return 'london_close'
        else:
            return 'other'
    
    async def _save_results_to_database(self, results: BacktestResults):
        """Save backtest results to database"""
        if not self.database:
            return
        
        try:
            # Save performance metrics
            metrics = {
                'period_start': results.start_date,
                'period_end': results.end_date,
                'period_type': 'backtest',
                'starting_balance': results.initial_balance,
                'ending_balance': results.final_balance,
                'net_profit': results.total_pnl,
                'gross_profit': sum(t.pnl for t in (results.trades or []) if t.pnl > 0),
                'gross_loss': sum(t.pnl for t in (results.trades or []) if t.pnl < 0),
                'total_trades': results.total_trades,
                'winning_trades': results.winning_trades,
                'losing_trades': results.losing_trades,
                'win_rate': results.win_rate,
                'average_win': results.avg_win,
                'average_loss': results.avg_loss,
                'largest_win': results.largest_win,
                'largest_loss': results.largest_loss,
                'profit_factor': results.profit_factor,
                'sharpe_ratio': results.sharpe_ratio,
                'max_drawdown': results.max_drawdown,
                'avg_signal_quality': results.avg_signal_quality,
                'avg_risk_reward': results.avg_risk_reward
            }
            
            self.database.save_performance_metrics(metrics)
            logger.info("Backtest results saved to database")
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            
    

# Usage example
if __name__ == "__main__":
    async def run_example():
        # Initialize engine
        engine = PowerOf3BacktestEngine(use_database=False)
        
        # Run backtest
        results = await engine.run_backtest(
            symbol='US30',
            start_date='2024-01-01',
            end_date='2024-03-31',
            initial_balance=10000.0,
            timeframe='5min'
        )
        
        # Print results
        print(f"Backtest Results for {results.symbol}:")
        print(f"Total Trades: {results.total_trades}")
        print(f"Win Rate: {results.win_rate:.1f}%")
        print(f"Total Return: {results.total_pnl_pct:.2f}%")
        print(f"Max Drawdown: {results.max_drawdown_pct:.2f}%")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Profit Factor: {results.profit_factor:.2f}")
    
    # Run the example
    asyncio.run(run_example())