"""Live data feed for real-time trading"""
import time
import pandas as pd
from typing import Dict
import yfinance as yf
import requests

class LiveDataFeed:
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def get_live_data(self, symbol: str, timeframe: str = '5m') -> pd.DataFrame:
        """Get live OHLCV data"""
        try:
            # Use yfinance for live data (free)
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval=timeframe)
            
            # Rename columns to match your system
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            return data.tail(100)  # Last 100 bars
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('regularMarketPrice', 0.0)
        except:
            return 0.0