"""
Fallback Data Provider for Backtesting
=====================================
"""

import yfinance as yf
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class DataProvider:
    """Fallback data provider using Yahoo Finance"""
    
    def __init__(self):
        self.yahoo_symbols = {
            'US30': '^DJI',
            'NAS100': '^IXIC',
            'SPX500': '^GSPC',
            'XAUUSD': 'GC=F'
        }
    
    def get_historical_data(self, symbol: str, start_date: str, 
                                 end_date: str, interval: str = '1h') -> pd.DataFrame:
        """Get historical data using Yahoo Finance"""
        try:
            yahoo_symbol = self.yahoo_symbols.get(symbol, symbol)
            
            # Convert interval to Yahoo format
            yahoo_interval = {
                '1min': '1m',
                '5min': '5m',
                '15min': '15m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d'
            }.get(interval, '1h')
            
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(start=start_date, end=end_date, interval=yahoo_interval)
            
            logger.info(f"Fetched {len(df)} bars for {symbol} from Yahoo Finance")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
