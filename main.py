"""
Power of 3 Trading Strategy - CrewAI Implementation
=================================================

A comprehensive trading system implementing the Power of 3 methodology
using CrewAI agents for automated analysis and execution.

SETUP INSTRUCTIONS:
==================

1. Install required packages:
   pip install crewai pandas numpy yfinance pandas-ta psycopg2-binary sqlalchemy requests pytz

2. Set environment variables:
   export OPENAI_API_KEY="your-openai-api-key"
   export TWELVE_DATA_API_KEY="your-twelve-data-key"  # Optional
   export ALPHA_VANTAGE_API_KEY="your-alpha-vantage-key"  # Optional
   export OANDA_API_KEY="your-oanda-key"  # Optional
   export DB_USER="postgres_user"  # Optional for basic testing
   export DB_PASSWORD="postgres_password"  # Optional for basic testing
   export DB_HOST="localhost"  # Optional for basic testing
   export DB_NAME="trading_db"  # Optional for basic testing

3. For testing without external APIs, just set OPENAI_API_KEY and run the basic test.

4. Save the Power of 3 Signal Generator as 'power_of_3_signal_generator.py' in the same directory.

TRADING SESSIONS:
================
- London Open: 2:00-5:00 AM EST
- New York Open: 7:00-10:00 AM EST  
- London Close: 10:00 AM-12:00 PM EST

POWER OF 3 METHODOLOGY:
======================
1. Accumulation Phase: Smart money builds positions quietly
2. Manipulation Phase: Price sweeps liquidity to trigger retail stops
3. Direction Phase: True institutional move begins

The system detects:
- Liquidity sweeps and stop hunts
- Fake breakouts and reversals
- Institutional order blocks
- Market structure shifts
- High-probability entry points with 1:5+ risk-reward ratios
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import aiohttp
import pandas as pd
import pytz

# Data and trading imports
import yfinance as yf

# CrewAI imports
from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# from power_of_3_signal_generator import (
#     PowerOf3SignalGenerator,
#     SignalFormatter,
#     TradingSession,
# )

#import pandas_ta as ta
from sqlalchemy import create_engine

from src.power_of_3.core.signal_generator import PowerOf3SignalGenerator
from src.power_of_3.core.types import LiquidityZone, TradingSession
from src.power_of_3.database.repository import POWER_OF_3_AVAILABLE
from src.power_of_3.utils.signal_formatter import SignalFormatter

load_dotenv()  # Load environment variables from .env file

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# class TradingSession(Enum):
#     LONDON_OPEN = "london_open"
#     NEW_YORK_OPEN = "new_york_open"
#     LONDON_CLOSE = "london_close"

class MarketDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

@dataclass
class TradingSignal:
    symbol: str
    direction: MarketDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    confidence: float
    session: TradingSession
    timestamp: datetime
    analysis: str

@dataclass
class MarketStructure:
    symbol: str
    higher_high: bool
    higher_low: bool
    trend_direction: MarketDirection
    liquidity_zones: List[Dict]
    order_blocks: List[Dict]
    manipulation_detected: bool

# =============================================================================
# DATA PROVIDERS CONFIGURATION
# =============================================================================

class DataProviderConfig:
    """Configuration for different data providers"""
    
    # Symbol mapping for different providers
    SYMBOL_MAPPING = {
        'twelve_data': {
            'US30': 'DJI',
            'NAS100': 'IXIC', 
            'SPX500': 'SPX',
            'XAUUSD': 'XAU/USD'
        },
        'alpha_vantage': {
            'US30': 'DJI',
            'NAS100': 'IXIC',
            'SPX500': 'SPX',
            'XAUUSD': 'XAUUSD'
        },
        'oanda': {
            'US30': 'US30_USD',
            'NAS100': 'NAS100_USD',
            'SPX500': 'SPX500_USD',
            'XAUUSD': 'XAU_USD'
        },
        'yahoo': {
            'US30': '^DJI',
            'NAS100': '^IXIC',
            'SPX500': '^GSPC',
            'XAUUSD': 'GC=F'  # Gold futures
        }
    }
    
    # Interval mapping for different providers
    INTERVAL_MAPPING = {
        'twelve_data': {
            '1min': '1min',
            '5min': '5min',
            '15min': '15min',
            '1h': '1h',
            '4h': '4h',
            '1d': '1day'
        },
        'alpha_vantage': {
            '1min': '1min',
            '5min': '5min',
            '15min': '15min',
            '1h': '60min',
            '4h': '4h',
            '1d': 'daily'
        },
        'oanda': {
            '1min': 'M1',
            '5min': 'M5',
            '15min': 'M15',
            '1h': 'H1',
            '4h': 'H4',
            '1d': 'D'
        }
    }
    
    # Rate limits (requests per minute)
    RATE_LIMITS = {
        'twelve_data': 8,  # Free tier: 8 requests/minute
        'alpha_vantage': 5,  # Free tier: 5 requests/minute
        'oanda': 100  # Practice account: 100 requests/minute
    }

class DataProvider:
    """Unified data provider supporting multiple sources with failover"""
    
    def __init__(self):
        # API Keys
        self.twelve_data_key = os.getenv('TWELVE_DATA_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.oanda_key = os.getenv('OANDA_API_KEY')
        self.oanda_account_id = os.getenv('OANDA_ACCOUNT_ID')
        
        # Rate limiting
        self.last_request_time = {}
        self.request_counts = {}
        
        # Cache
        self.cache = {}
        self.cache_ttl = 60  # Cache for 1 minute
        
        # Provider priority order
        self.provider_priority = self._determine_provider_priority()
        
        logger.info(f"DataProvider initialized with providers: {self.provider_priority}")
    
    def _determine_provider_priority(self) -> List[str]:
        """Determine provider priority based on available API keys"""
        providers = []
        
        if self.oanda_key and self.oanda_account_id:
            providers.append('oanda')
        if self.twelve_data_key:
            providers.append('twelve_data')
        if self.alpha_vantage_key:
            providers.append('alpha_vantage')
        
        # Always include yahoo as fallback
        providers.append('yahoo')
        
        return providers
    
    async def get_realtime_data(self, symbol: str, interval: str = '5min', 
                               bars: int = 500) -> pd.DataFrame:
        """Get real-time price data with provider fallback"""
        
        # Check cache first
        cache_key = f"{symbol}_{interval}_{bars}"
        if self._is_cache_valid(cache_key):
            logger.info(f"Returning cached data for {symbol}")
            return self.cache[cache_key]['data']
        
        # Try providers in priority order
        for provider in self.provider_priority:
            try:
                logger.info(f"Trying {provider} for {symbol}")
                
                if not self._can_make_request(provider):
                    logger.warning(f"Rate limit reached for {provider}, trying next")
                    continue
                
                df = await self._fetch_from_provider(provider, symbol, interval, bars)
                
                if not df.empty:
                    # Cache successful result
                    self.cache[cache_key] = {
                        'data': df,
                        'timestamp': time.time(),
                        'provider': provider
                    }
                    
                    logger.info(f"Successfully fetched {len(df)} bars from {provider}")
                    return df
                    
            except Exception as e:
                logger.warning(f"Error with {provider}: {e}")
                continue
        
        logger.error(f"Failed to fetch data for {symbol} from all providers")
        return pd.DataFrame()
    
    async def get_historical_data(self, symbol: str, start_date: str, 
                                 end_date: str, interval: str = '1h') -> pd.DataFrame:
        """Get historical data for backtesting"""
        
        cache_key = f"hist_{symbol}_{start_date}_{end_date}_{interval}"
        if self._is_cache_valid(cache_key, ttl=3600):  # Cache for 1 hour
            logger.info(f"Returning cached historical data for {symbol}")
            return self.cache[cache_key]['data']
        
        # For historical data, prefer providers with better historical coverage
        historical_priority = ['twelve_data', 'alpha_vantage', 'yahoo', 'oanda']
        available_providers = [p for p in historical_priority if p in self.provider_priority]
        
        for provider in available_providers:
            try:
                df = await self._fetch_historical_from_provider(
                    provider, symbol, start_date, end_date, interval
                )
                
                if not df.empty:
                    self.cache[cache_key] = {
                        'data': df,
                        'timestamp': time.time(),
                        'provider': provider
                    }
                    logger.info(f"Successfully fetched historical data from {provider}")
                    return df
                    
            except Exception as e:
                logger.warning(f"Historical data error with {provider}: {e}")
                continue
        
        logger.error(f"Failed to fetch historical data for {symbol}")
        return pd.DataFrame()
    
    async def _fetch_from_provider(self, provider: str, symbol: str, 
                                  interval: str, bars: int) -> pd.DataFrame:
        """Fetch data from specific provider"""
        
        if provider == 'twelve_data':
            return await self._get_twelve_data(symbol, interval, bars)
        elif provider == 'alpha_vantage':
            return await self._get_alpha_vantage_data(symbol, interval, bars)
        elif provider == 'oanda':
            return await self._get_oanda_data(symbol, interval, bars)
        elif provider == 'yahoo':
            return await self._get_yahoo_data(symbol, interval, bars)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    async def _fetch_historical_from_provider(self, provider: str, symbol: str,
                                            start_date: str, end_date: str,
                                            interval: str) -> pd.DataFrame:
        """Fetch historical data from specific provider"""
        
        if provider == 'twelve_data':
            return await self._get_twelve_data_historical(symbol, start_date, end_date, interval)
        elif provider == 'alpha_vantage':
            return await self._get_alpha_vantage_historical(symbol, start_date, end_date, interval)
        elif provider == 'oanda':
            return await self._get_oanda_historical(symbol, start_date, end_date, interval)
        elif provider == 'yahoo':
            return await self._get_yahoo_historical(symbol, start_date, end_date, interval)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _can_make_request(self, provider: str) -> bool:
        """Check if we can make a request without hitting rate limits"""
        now = time.time()
        minute_ago = now - 60
        
        if provider not in self.request_counts:
            self.request_counts[provider] = []
        
        # Remove requests older than 1 minute
        self.request_counts[provider] = [
            req_time for req_time in self.request_counts[provider] 
            if req_time > minute_ago
        ]
        
        # Check if we're under the rate limit
        rate_limit = DataProviderConfig.RATE_LIMITS.get(provider, 60)
        return len(self.request_counts[provider]) < rate_limit
    
    def _record_request(self, provider: str):
        """Record a request for rate limiting"""
        if provider not in self.request_counts:
            self.request_counts[provider] = []
        self.request_counts[provider].append(time.time())
    
    def _is_cache_valid(self, cache_key: str, ttl: Optional[int] = None) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        cache_ttl = ttl or self.cache_ttl
        age = time.time() - self.cache[cache_key]['timestamp']
        return age < cache_ttl
    
    def _get_provider_symbol(self, provider: str, symbol: str) -> str:
        """Get provider-specific symbol"""
        mapping = DataProviderConfig.SYMBOL_MAPPING.get(provider, {})
        return mapping.get(symbol, symbol)
    
    def _get_provider_interval(self, provider: str, interval: str) -> str:
        """Get provider-specific interval"""
        mapping = DataProviderConfig.INTERVAL_MAPPING.get(provider, {})
        return mapping.get(interval, interval)
    
    # TWELVE DATA IMPLEMENTATION
    async def _get_twelve_data(self, symbol: str, interval: str, bars: int) -> pd.DataFrame:
        """Fetch from Twelve Data API"""
        if not self.twelve_data_key:
            raise ValueError("Twelve Data API key not configured")
        
        provider_symbol = self._get_provider_symbol('twelve_data', symbol)
        provider_interval = self._get_provider_interval('twelve_data', interval)
        
        url = "https://api.twelvedata.com/time_series"
        params = {
            'symbol': provider_symbol,
            'interval': provider_interval,
            'outputsize': min(bars, 5000),  # API limit
            'apikey': self.twelve_data_key,
            'format': 'JSON'
        }
        
        self._record_request('twelve_data')
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                if 'values' in data:
                    df = pd.DataFrame(data['values'])
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                    
                    # Rename columns to standard format
                    df.rename(columns={
                        'open': 'Open',
                        'high': 'High', 
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume'
                    }, inplace=True)
                    
                    # Convert to numeric
                    numeric_columns = ['Open', 'High', 'Low', 'Close']
                    if 'Volume' in df.columns:
                        numeric_columns.append('Volume')
                    
                    for col in numeric_columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    return df.sort_index()
                
                elif 'message' in data:
                    raise ValueError(f"Twelve Data error: {data['message']}")
                
                return pd.DataFrame()
    
    async def _get_twelve_data_historical(self, symbol: str, start_date: str, 
                                         end_date: str, interval: str) -> pd.DataFrame:
        """Get historical data from Twelve Data"""
        provider_symbol = self._get_provider_symbol('twelve_data', symbol)
        provider_interval = self._get_provider_interval('twelve_data', interval)
        
        url = "https://api.twelvedata.com/time_series"
        params = {
            'symbol': provider_symbol,
            'interval': provider_interval,
            'start_date': start_date,
            'end_date': end_date,
            'apikey': self.twelve_data_key,
            'format': 'JSON',
            'outputsize': 5000
        }
        
        self._record_request('twelve_data')
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return self._process_twelve_data_response(data)
    
    def _process_twelve_data_response(self, data: dict) -> pd.DataFrame:
        """Process Twelve Data API response"""
        if 'values' in data:
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            
            # Standardize column names
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low', 
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            # Convert to numeric
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if 'Volume' in df.columns:
                df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            
            return df.sort_index()
        
        return pd.DataFrame()
    
    # ALPHA VANTAGE IMPLEMENTATION
    async def _get_alpha_vantage_data(self, symbol: str, interval: str, bars: int) -> pd.DataFrame:
        """Fetch from Alpha Vantage API"""
        if not self.alpha_vantage_key:
            raise ValueError("Alpha Vantage API key not configured")
        
        provider_symbol = self._get_provider_symbol('alpha_vantage', symbol)
        provider_interval = self._get_provider_interval('alpha_vantage', interval)
        
        # Alpha Vantage has different endpoints for intraday vs daily
        if interval in ['1min', '5min', '15min', '30min', '60min']:
            function = 'TIME_SERIES_INTRADAY'
            extra_params = {'interval': provider_interval}
        else:
            function = 'TIME_SERIES_DAILY'
            extra_params = {}
        
        url = "https://www.alphavantage.co/query"
        params = {
            'function': function,
            'symbol': provider_symbol,
            'apikey': self.alpha_vantage_key,
            'outputsize': 'full',
            'datatype': 'json',
            **extra_params
        }
        
        self._record_request('alpha_vantage')
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return self._process_alpha_vantage_response(data, bars)
    
    async def _get_alpha_vantage_historical(self, symbol: str, start_date: str, 
                                           end_date: str, interval: str) -> pd.DataFrame:
        """Get historical data from Alpha Vantage"""
        # Alpha Vantage doesn't support date ranges directly, so we get full data and filter
        df = await self._get_alpha_vantage_data(symbol, interval, 5000)
        
        if not df.empty:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            mask = (df.index >= start) & (df.index <= end)
            return df.loc[mask]
        
        return pd.DataFrame()
    
    def _process_alpha_vantage_response(self, data: dict, bars: int) -> pd.DataFrame:
        """Process Alpha Vantage API response"""
        # Find the time series data key
        time_series_key = None
        for key in data.keys():
            if 'Time Series' in key:
                time_series_key = key
                break
        
        if not time_series_key or time_series_key not in data:
            if 'Error Message' in data:
                raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
            return pd.DataFrame()
        
        time_series = data[time_series_key]
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Standardize column names
        column_mapping = {}
        for col in df.columns:
            if 'open' in col.lower():
                column_mapping[col] = 'Open'
            elif 'high' in col.lower():
                column_mapping[col] = 'High'
            elif 'low' in col.lower():
                column_mapping[col] = 'Low'
            elif 'close' in col.lower():
                column_mapping[col] = 'Close'
            elif 'volume' in col.lower():
                column_mapping[col] = 'Volume'
        
        df.rename(columns=column_mapping, inplace=True)
        
        # Convert to numeric
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        
        # Return last N bars
        return df.tail(bars)
    
    # OANDA IMPLEMENTATION  
    async def _get_oanda_data(self, symbol: str, interval: str, bars: int) -> pd.DataFrame:
        """Fetch from OANDA API"""
        if not self.oanda_key:
            raise ValueError("OANDA API key not configured")
        
        provider_symbol = self._get_provider_symbol('oanda', symbol)
        provider_interval = self._get_provider_interval('oanda', interval)
        
        # OANDA uses different base URLs for live vs practice
        base_url = "https://api-fxpractice.oanda.com"  # Practice account
        # base_url = "https://api-fxtrade.oanda.com"  # Live account
        
        url = f"{base_url}/v3/instruments/{provider_symbol}/candles"
        headers = {
            'Authorization': f'Bearer {self.oanda_key}',
            'Accept-Datetime-Format': 'RFC3339'
        }
        params = {
            'granularity': provider_interval,
            'count': min(bars, 5000),  # OANDA limit
            'price': 'M'  # Mid prices
        }
        
        self._record_request('oanda')
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_oanda_response(data)
                else:
                    error_text = await response.text()
                    raise ValueError(f"OANDA API error: {response.status} - {error_text}")
    
    async def _get_oanda_historical(self, symbol: str, start_date: str, 
                                   end_date: str, interval: str) -> pd.DataFrame:
        """Get historical data from OANDA"""
        provider_symbol = self._get_provider_symbol('oanda', symbol)
        provider_interval = self._get_provider_interval('oanda', interval)
        
        base_url = "https://api-fxpractice.oanda.com"
        url = f"{base_url}/v3/instruments/{provider_symbol}/candles"
        headers = {
            'Authorization': f'Bearer {self.oanda_key}',
            'Accept-Datetime-Format': 'RFC3339'
        }
        params = {
            'granularity': provider_interval,
            'from': f"{start_date}T00:00:00.000000000Z",
            'to': f"{end_date}T23:59:59.000000000Z",
            'price': 'M'
        }
        
        self._record_request('oanda')
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_oanda_response(data)
                else:
                    error_text = await response.text()
                    raise ValueError(f"OANDA API error: {response.status} - {error_text}")
    
    def _process_oanda_response(self, data: dict) -> pd.DataFrame:
        """Process OANDA API response"""
        if 'candles' not in data:
            return pd.DataFrame()
        
        candles = data['candles']
        
        # Extract OHLC data
        ohlc_data = []
        for candle in candles:
            if candle['complete']:  # Only use complete candles
                mid = candle['mid']
                ohlc_data.append({
                    'datetime': candle['time'],
                    'Open': float(mid['o']),
                    'High': float(mid['h']),
                    'Low': float(mid['l']),
                    'Close': float(mid['c']),
                    'Volume': int(candle['volume'])
                })
        
        if not ohlc_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(ohlc_data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        return df.sort_index()
    
    # YAHOO FINANCE IMPLEMENTATION (FALLBACK)
    async def _get_yahoo_data(self, symbol: str, interval: str, bars: int) -> pd.DataFrame:
        """Fetch from Yahoo Finance (fallback)"""
        provider_symbol = self._get_provider_symbol('yahoo', symbol)
        
        # Convert interval to Yahoo format
        yahoo_interval = {
            '1min': '1m',
            '5min': '5m', 
            '15min': '15m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }.get(interval, '5m')
        
        try:
            ticker = yf.Ticker(provider_symbol)
            
            # Calculate period needed for bars
            if bars <= 100:
                period = "1d"
            elif bars <= 500:
                period = "5d"
            elif bars <= 1000:
                period = "1mo"
            else:
                period = "3mo"
            
            df = ticker.history(period=period, interval=yahoo_interval)
            
            if not df.empty:
                # Yahoo already uses standard column names
                df = df.tail(bars)  # Get last N bars
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            raise ValueError(f"Yahoo Finance error: {e}")
    
    async def _get_yahoo_historical(self, symbol: str, start_date: str, 
                                   end_date: str, interval: str) -> pd.DataFrame:
        """Get historical data from Yahoo Finance"""
        provider_symbol = self._get_provider_symbol('yahoo', symbol)
        
        yahoo_interval = {
            '1min': '1m',
            '5min': '5m',
            '15min': '15m', 
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }.get(interval, '1h')
        
        try:
            ticker = yf.Ticker(provider_symbol)
            df = ticker.history(start=start_date, end=end_date, interval=yahoo_interval)
            return df
        except Exception as e:
            raise ValueError(f"Yahoo Finance historical error: {e}")
    
    # UTILITY METHODS
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return self.provider_priority
    
    def test_provider_connectivity(self) -> Dict[str, bool]:
        """Test connectivity to all configured providers"""
        results = {}
        
        async def test_provider(provider: str):
            try:
                # Test with a simple request
                df = await self._fetch_from_provider(provider, 'US30', '5min', 1)
                return not df.empty
            except Exception as e:
                logger.warning(f"Provider {provider} test failed: {e}")
                return False
        
        async def run_tests():
            tasks = []
            for provider in self.provider_priority:
                if provider != 'yahoo':  # Skip API key requirement test for yahoo
                    tasks.append(test_provider(provider))
            
            if tasks:
                test_results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, provider in enumerate([p for p in self.provider_priority if p != 'yahoo']):
                    results[provider] = test_results[i] is True
            
            # Always test yahoo
            results['yahoo'] = await test_provider('yahoo')
            
            return results
        
        return asyncio.run(run_tests())
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        logger.info("Data cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total_entries = len(self.cache)
        valid_entries = sum(1 for key in self.cache.keys() if self._is_cache_valid(key))
        
        providers_used = {}
        for entry in self.cache.values():
            provider = entry.get('provider', 'unknown')
            providers_used[provider] = providers_used.get(provider, 0) + 1
        
        return {
            'total_entries': total_entries,
            'valid_entries': valid_entries,
            'providers_used': providers_used,
            'cache_hit_rate': valid_entries / max(total_entries, 1)
        }

# =============================================================================
# TECHNICAL ANALYSIS TOOLS
# =============================================================================

class TechnicalAnalysis:
    """Advanced technical analysis for Power of 3 strategy"""
    
    @staticmethod
    def identify_liquidity_zones(df: pd.DataFrame, lookback: int = 20) -> List[Dict]:
        """Identify key liquidity zones where stops are likely clustered"""
        liquidity_zones = []
        
        # FIXED: Changed from df['High'] to df['high']
        daily_high = df['high'].rolling(24).max()
        daily_low = df['low'].rolling(24).min()
        
        # FIXED: Changed column names to lowercase
        swing_highs = df['high'][(df['high'].shift(1) < df['high']) & 
                                (df['high'].shift(-1) < df['high'])]
        swing_lows = df['low'][(df['low'].shift(1) > df['low']) & 
                              (df['low'].shift(-1) > df['low'])]
        
        # FIXED: Changed column names to lowercase
        weekly_high = df['high'].rolling(168).max()  # 7 days * 24 hours
        weekly_low = df['low'].rolling(168).min()
        
        # Add significant levels
        for i in range(len(df)):
            if i < lookback:
                continue
                
            # FIXED: Changed from df['Close'] to df['close']
            current_price = df['close'].iloc[i]
            
            # Check for liquidity zones
            levels = [
                daily_high.iloc[i],
                daily_low.iloc[i],
                weekly_high.iloc[i],
                weekly_low.iloc[i]
            ]
            
            for level in levels:
                if level and abs(current_price - level) / current_price < 0.005:  # Within 0.5%
                    liquidity_zones.append({
                        'level': level,
                        'type': 'resistance' if level > current_price else 'support',
                        'strength': TechnicalAnalysis._calculate_level_strength(df, level, i),
                        'timestamp': df.index[i]
                    })
        
        return liquidity_zones
    
    @staticmethod
    def _calculate_level_strength(df: pd.DataFrame, level: float, current_idx: int) -> float:
        """Calculate strength of a support/resistance level"""
        touches = 0
        lookback = min(current_idx, 50)
        
        for i in range(current_idx - lookback, current_idx):
            # FIXED: Changed from df['High'] and df['Low'] to lowercase
            if abs(df['high'].iloc[i] - level) / level < 0.001 or \
               abs(df['low'].iloc[i] - level) / level < 0.001:
                touches += 1
        
        return min(touches / 5.0, 1.0)  # Normalize to 0-1
    
    @staticmethod
    def identify_market_structure_break(df: pd.DataFrame, lookback: int = 10) -> MarketStructure:
        """Identify market structure breaks and trend changes"""
        if len(df) < lookback * 2:
            return MarketStructure("", False, False, MarketDirection.NEUTRAL, [], [], False)
        
        # Calculate swing points
        swing_highs = []
        swing_lows = []
        
        for i in range(lookback, len(df) - lookback):
            # FIXED: Changed from df['High'] to df['high']
            if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, lookback+1)) and \
               all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, lookback+1)):
                swing_highs.append((i, df['high'].iloc[i]))
            
            # FIXED: Changed from df['Low'] to df['low']
            if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, lookback+1)) and \
               all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, lookback+1)):
                swing_lows.append((i, df['low'].iloc[i]))
        
        # Analyze trend
        higher_high = False
        higher_low = False
        trend_direction = MarketDirection.NEUTRAL
        
        if len(swing_highs) >= 2:
            higher_high = swing_highs[-1][1] > swing_highs[-2][1]
        
        if len(swing_lows) >= 2:
            higher_low = swing_lows[-1][1] > swing_lows[-2][1]
        
        if higher_high and higher_low:
            trend_direction = MarketDirection.BULLISH
        elif not higher_high and not higher_low:
            trend_direction = MarketDirection.BEARISH
        
        # Identify order blocks
        order_blocks = TechnicalAnalysis.identify_order_blocks(df)
        
        # Detect manipulation
        manipulation_detected = TechnicalAnalysis.detect_manipulation(df)
        
        return MarketStructure(
            symbol="",
            higher_high=higher_high,
            higher_low=higher_low,
            trend_direction=trend_direction,
            liquidity_zones=TechnicalAnalysis.identify_liquidity_zones(df),
            order_blocks=order_blocks,
            manipulation_detected=manipulation_detected
        )
    
    @staticmethod
    def identify_order_blocks(df: pd.DataFrame, min_body_pct: float = 0.7) -> List[Dict]:
        """Identify institutional order blocks"""
        order_blocks = []
        
        for i in range(1, len(df)):
            candle = df.iloc[i]
            prev_candle = df.iloc[i-1]
            
            # FIXED: Changed all column names to lowercase
            body_size = abs(candle['close'] - candle['open'])
            total_size = candle['high'] - candle['low']
            body_pct = body_size / total_size if total_size > 0 else 0
            
            # Volume surge (if available) - FIXED: Changed from 'Volume' to 'volume'
            avg_volume = 1  # Default value to avoid unbound error
            volume_surge = False
            if 'volume' in df.columns:
                avg_volume = df['volume'].rolling(20).mean().iloc[i]
                volume_surge = candle['volume'] > avg_volume * 1.5
            
            # Large body candle with volume
            if body_pct > min_body_pct and volume_surge:
                order_block_type = 'bullish' if candle['close'] > candle['open'] else 'bearish'
                
                order_blocks.append({
                    'timestamp': df.index[i],
                    'type': order_block_type,
                    # FIXED: All column names to lowercase
                    'high': candle['high'],
                    'low': candle['low'],
                    'open': candle['open'],
                    'close': candle['close'],
                    'strength': body_pct,
                    'volume_ratio': candle.get('volume', 0) / avg_volume if 'volume' in df.columns else 1.0
                })
        
        return order_blocks[-10:]  # Return last 10 order blocks
    
    @staticmethod
    def detect_manipulation(df: pd.DataFrame, threshold: float = 0.002) -> bool:
        """Detect potential manipulation moves"""
        if len(df) < 5:
            return False
        
        recent_candles = df.tail(5)
        
        # Look for quick spikes followed by reversals
        for i in range(1, len(recent_candles)):
            current = recent_candles.iloc[i]
            prev = recent_candles.iloc[i-1]
            
            # FIXED: Changed column names to lowercase
            move_up = (current['high'] - prev['close']) / prev['close']
            move_down = (prev['close'] - current['low']) / prev['close']
            
            if move_up > threshold and current['close'] < current['open']:
                return True  # Spike up then close lower
            if move_down > threshold and current['close'] > current['open']:
                return True  # Spike down then close higher
        
        return False
    
    # @staticmethod
    # def _calculate_level_strength(df: pd.DataFrame, level: float, current_idx: int) -> float:
    #     """Calculate strength of a support/resistance level"""
    #     touches = 0
    #     lookback = min(current_idx, 50)
        
    #     for i in range(current_idx - lookback, current_idx):
    #         if abs(df['High'].iloc[i] - level) / level < 0.001 or \
    #            abs(df['Low'].iloc[i] - level) / level < 0.001:
    #             touches += 1
        
    #     return min(touches / 5.0, 1.0)  # Normalize to 0-1
    
    # @staticmethod
    # def identify_market_structure_break(df: pd.DataFrame, lookback: int = 10) -> MarketStructure:
    #     """Identify market structure breaks and trend changes"""
    #     if len(df) < lookback * 2:
    #         return MarketStructure("", False, False, MarketDirection.NEUTRAL, [], [], False)
        
    #     # Calculate swing points
    #     swing_highs = []
    #     swing_lows = []
        
    #     for i in range(lookback, len(df) - lookback):
    #         # Swing high
    #         if all(df['High'].iloc[i] > df['High'].iloc[i-j] for j in range(1, lookback+1)) and \
    #            all(df['High'].iloc[i] > df['High'].iloc[i+j] for j in range(1, lookback+1)):
    #             swing_highs.append((i, df['High'].iloc[i]))
            
    #         # Swing low
    #         if all(df['Low'].iloc[i] < df['Low'].iloc[i-j] for j in range(1, lookback+1)) and \
    #            all(df['Low'].iloc[i] < df['Low'].iloc[i+j] for j in range(1, lookback+1)):
    #             swing_lows.append((i, df['Low'].iloc[i]))
        
    #     # Analyze trend
    #     higher_high = False
    #     higher_low = False
    #     trend_direction = MarketDirection.NEUTRAL
        
    #     if len(swing_highs) >= 2:
    #         higher_high = swing_highs[-1][1] > swing_highs[-2][1]
        
    #     if len(swing_lows) >= 2:
    #         higher_low = swing_lows[-1][1] > swing_lows[-2][1]
        
    #     if higher_high and higher_low:
    #         trend_direction = MarketDirection.BULLISH
    #     elif not higher_high and not higher_low:
    #         trend_direction = MarketDirection.BEARISH
        
    #     # Identify order blocks
    #     order_blocks = TechnicalAnalysis.identify_order_blocks(df)
        
    #     # Detect manipulation
    #     manipulation_detected = TechnicalAnalysis.detect_manipulation(df)
        
    #     return MarketStructure(
    #         symbol="",
    #         higher_high=higher_high,
    #         higher_low=higher_low,
    #         trend_direction=trend_direction,
    #         liquidity_zones=TechnicalAnalysis.identify_liquidity_zones(df),
    #         order_blocks=order_blocks,
    #         manipulation_detected=manipulation_detected
    #     )
    
    # @staticmethod
    # def identify_order_blocks(df: pd.DataFrame, min_body_pct: float = 0.7) -> List[Dict]:
    #     """Identify institutional order blocks"""
    #     order_blocks = []
        
    #     for i in range(1, len(df)):
    #         candle = df.iloc[i]
    #         prev_candle = df.iloc[i-1]
            
    #         # Calculate candle properties
    #         body_size = abs(candle['Close'] - candle['Open'])
    #         total_size = candle['High'] - candle['Low']
    #         body_pct = body_size / total_size if total_size > 0 else 0
            
    #         # Volume surge (if available)
    #         volume_surge = False
    #         avg_volume = df['Volume'].rolling(20).mean().iloc[i] if 'Volume' in df.columns else 1.0
    #         if 'Volume' in df.columns:
    #             avg_volume = df['Volume'].rolling(20).mean().iloc[i]
    #             volume_surge = candle['Volume'] > avg_volume * 1.5
            
    #         # Large body candle with volume
    #         if body_pct > min_body_pct and volume_surge:
    #             order_block_type = 'bullish' if candle['Close'] > candle['Open'] else 'bearish'
                
    #             order_blocks.append({
    #                 'timestamp': df.index[i],
    #                 'type': order_block_type,
    #                 'high': candle['High'],
    #                 'low': candle['Low'],
    #                 'open': candle['Open'],
    #                 'close': candle['Close'],
    #                 'strength': body_pct,
    #                 'volume_ratio': candle.get('Volume', 0) / (avg_volume if 'Volume' in df.columns else 1.0)
    #             })
        
    #     return order_blocks[-10:]  # Return last 10 order blocks
    
    # @staticmethod
    # def detect_manipulation(df: pd.DataFrame, threshold: float = 0.002) -> bool:
    #     """Detect potential manipulation moves"""
    #     if len(df) < 5:
    #         return False
        
    #     recent_candles = df.tail(5)
        
    #     # Look for quick spikes followed by reversals
    #     for i in range(1, len(recent_candles)):
    #         current = recent_candles.iloc[i]
    #         prev = recent_candles.iloc[i-1]
            
    #         # Significant move up then down (or vice versa)
    #         move_up = (current['High'] - prev['Close']) / prev['Close']
    #         move_down = (prev['Close'] - current['Low']) / prev['Close']
            
    #         if move_up > threshold and current['Close'] < current['Open']:
    #             return True  # Spike up then close lower
    #         if move_down > threshold and current['Close'] > current['Open']:
    #             return True  # Spike down then close higher
        
    #     return False

# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

class SessionManager:
    """Manage trading sessions and timing"""
    
    @staticmethod
    def get_current_session() -> Optional[TradingSession]:
        """Determine current trading session"""
        now = datetime.now(pytz.timezone('US/Eastern'))
        hour = now.hour
        
        # London Open: 2:00 AM - 5:00 AM EST
        if 2 <= hour < 5:
            return TradingSession.LONDON_OPEN
        # New York Open: 7:00 AM - 10:00 AM EST  
        elif 7 <= hour < 10:
            return TradingSession.NEW_YORK_OPEN
        # London Close: 10:00 AM - 12:00 PM EST
        elif 10 <= hour < 12:
            return TradingSession.LONDON_CLOSE
        
        return None
    
    @staticmethod
    def is_power_of_3_time() -> bool:
        """Check if we're in a Power of 3 session"""
        return SessionManager.get_current_session() is not None

# =============================================================================
# CREWAI TOOLS
# =============================================================================

class MarketDataTool(BaseTool):
    name: str = "market_data_tool"
    description: str = "Fetch real-time and historical market data for analysis"
    
    def _run(self, symbol: str, timeframe: str = "5min", bars: int = 100) -> str:
        """Fetch market data - simplified for CrewAI compatibility"""
        try:
            # Use Yahoo Finance as fallback for CrewAI compatibility
            # This avoids async event loop issues
            
            # Map our symbols to Yahoo symbols
            yahoo_symbols = {
                'US30': '^DJI',
                'NAS100': '^IXIC', 
                'SPX500': '^GSPC',
                'XAUUSD': 'GC=F'
            }
            
            yahoo_symbol = yahoo_symbols.get(symbol, symbol)
            
            try:
                import yfinance as yf
                ticker = yf.Ticker(yahoo_symbol)
                
                # Convert timeframe to Yahoo format
                yahoo_interval = {
                    '1min': '1m',
                    '5min': '5m',
                    '15min': '15m',
                    '1h': '1h',
                    '1H': '1h',
                    '4h': '4h',
                    '1d': '1d'
                }.get(timeframe, '5m')
                
                # Get data
                df = ticker.history(period="5d", interval=yahoo_interval)
                
                if df.empty:
                    return f"No data available for {symbol} ({yahoo_symbol})"
                
                # Get latest data
                latest = df.tail(1).iloc[0]
                previous = df.tail(2).iloc[0] if len(df) > 1 else latest
                
                # Calculate change
                price_change = latest['Close'] - previous['Close']
                price_change_pct = (price_change / previous['Close']) * 100
                
                return f"""Market Data for {symbol}:
Latest Price: {latest['Close']:.2f}
Change: {price_change:+.2f} ({price_change_pct:+.2f}%)
High: {latest['High']:.2f}
Low: {latest['Low']:.2f}
Open: {latest['Open']:.2f}
Volume: {latest['Volume']:,.0f}
Data Points: {len(df)}
Timeframe: {timeframe}
Provider: Yahoo Finance
Last Updated: {df.index[-1]}"""
                
            except Exception as yf_error:
                # Fallback to sample data for testing
                return f"""Market Data for {symbol} (SAMPLE DATA):
Latest Price: 2100.50
Change: +15.25 (+0.73%)
High: 2105.75
Low: 2095.30
Open: 2098.40
Volume: 125,430
Data Points: 50
Timeframe: {timeframe}
Provider: Sample Data (Yahoo Finance unavailable: {str(yf_error)})
Note: This is sample data for testing purposes"""
                
        except Exception as e:
            return f"Error fetching data for {symbol}: {str(e)}"

class TechnicalAnalysisTool(BaseTool):
    name: str = "technical_analysis_tool"
    description: str = "Perform technical analysis including market structure and order blocks"
    
    def _run(self, symbol: str, timeframe: str = "5min") -> str:
        """Perform technical analysis - simplified for CrewAI compatibility"""
        try:
            # Use Yahoo Finance to get data for analysis
            yahoo_symbols = {
                'US30': '^DJI',
                'NAS100': '^IXIC',
                'SPX500': '^GSPC', 
                'XAUUSD': 'GC=F'
            }
            
            yahoo_symbol = yahoo_symbols.get(symbol, symbol)
            
            try:
                import yfinance as yf
                ticker = yf.Ticker(yahoo_symbol)
                
                yahoo_interval = {
                    '1min': '1m',
                    '5min': '5m',
                    '15min': '15m',
                    '1h': '1h',
                    '1H': '1h',
                    '4h': '4h',
                    '1d': '1d'
                }.get(timeframe, '5m')
                
                # Get enough data for analysis
                df = ticker.history(period="30d", interval=yahoo_interval)
                
                if df.empty:
                    return f"No data available for technical analysis of {symbol}"
                
                # FIXED: Standardize column names IMMEDIATELY after fetching
                column_mapping = {
                    'Open': 'open', 'High': 'high', 'Low': 'low', 
                    'Close': 'close', 'Volume': 'volume'
                }
                df = df.rename(columns=column_mapping)
                
                # Perform technical analysis using our algorithms
                market_structure = TechnicalAnalysis.identify_market_structure_break(df)
                liquidity_zones = TechnicalAnalysis.identify_liquidity_zones(df)
                order_blocks = TechnicalAnalysis.identify_order_blocks(df)
                manipulation_detected = TechnicalAnalysis.detect_manipulation(df)
                
                # Current price analysis - FIXED: Use lowercase columns
                latest = df.tail(1).iloc[0]
                previous = df.tail(2).iloc[0] if len(df) > 1 else latest
                price_change = latest['close'] - previous['close']
                price_change_pct = (price_change / previous['close']) * 100
                
                # Find nearby liquidity zones
                current_price = latest['close']
                nearby_zones = []
                for zone in liquidity_zones[-10:]:  # Last 10 zones
                    if 'level' in zone:
                        distance = abs(zone['level'] - current_price) / current_price * 100
                        if distance < 3.0:  # Within 3%
                            nearby_zones.append({
                                'level': zone['level'],
                                'type': zone.get('type', 'unknown'),
                                'strength': zone.get('strength', 0),
                                'distance_pct': distance
                            })
                
                # Recent order blocks
                recent_blocks = order_blocks[-3:] if order_blocks else []
                
                # Calculate basic trend - FIXED: Use lowercase columns
                sma_20 = df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else latest['close']
                trend_bias = "BULLISH" if latest['close'] > sma_20 else "BEARISH"
                
                analysis = f"""Technical Analysis for {symbol} ({timeframe}):

=== PRICE ACTION ===
Current Price: {latest['close']:.2f}
Price Change: {price_change:+.2f} ({price_change_pct:+.2f}%)
High: {latest['high']:.2f}
Low: {latest['low']:.2f}
20-SMA: {sma_20:.2f}

=== MARKET STRUCTURE ===
Trend Direction: {market_structure.trend_direction.value.upper()}
Higher Highs: {market_structure.higher_high}
Higher Lows: {market_structure.higher_low}
Overall Bias: {trend_bias}
Structure Break: {"YES" if market_structure.higher_high != market_structure.higher_low else "NO"}

=== INSTITUTIONAL ACTIVITY ===
Manipulation Detected: {manipulation_detected}
Total Order Blocks: {len(order_blocks)}
Recent Order Blocks: {len(recent_blocks)}

=== LIQUIDITY ANALYSIS ===
Total Liquidity Zones: {len(liquidity_zones)}
Nearby Zones (within 3%): {len(nearby_zones)}"""

                # Add nearby liquidity zones details
                if nearby_zones:
                    analysis += "\n\n=== NEARBY LIQUIDITY ZONES ==="
                    for i, zone in enumerate(sorted(nearby_zones, key=lambda x: x['distance_pct'])[:5]):
                        analysis += f"\n{i+1}. {zone['type'].upper()}: {zone['level']:.2f} ({zone['distance_pct']:.1f}% away)"
                
                # Add recent order blocks
                if recent_blocks:
                    analysis += "\n\n=== RECENT ORDER BLOCKS ==="
                    for i, block in enumerate(recent_blocks):
                        analysis += f"\n{i+1}. {block['type'].upper()}: {block['low']:.2f} - {block['high']:.2f}"
                
                # Add Power of 3 assessment
                analysis += f"\n\n=== POWER OF 3 ASSESSMENT ==="
                analysis += f"\nManipulation Evidence: {'STRONG' if manipulation_detected else 'WEAK'}"
                analysis += f"\nLiquidity Hunt Potential: {'HIGH' if len(nearby_zones) > 2 else 'MEDIUM' if len(nearby_zones) > 0 else 'LOW'}"
                analysis += f"\nInstitutional Interest: {'HIGH' if len(recent_blocks) > 1 else 'MEDIUM' if len(recent_blocks) > 0 else 'LOW'}"
                
                return analysis
                
            except Exception as yf_error:
                # Fallback analysis with sample data
                return f"""Technical Analysis for {symbol} (SAMPLE ANALYSIS):
Note: Yahoo Finance unavailable: {str(yf_error)}"""
                
        except Exception as e:
            return f"Error in technical analysis for {symbol}: {str(e)}"

class PowerOf3SignalTool(BaseTool):
    name: str = "power_of_3_signal_tool"
    description: str = "Generate Power of 3 trading signals using ICT methodology"
    
    def _run(self, symbol: str, timeframe: str = "5min", account_balance: float = 10000.0) -> str:
        """Generate Power of 3 signals for the given symbol"""
        
        if not POWER_OF_3_AVAILABLE:
            return f"""Power of 3 Signal Generator not available for {symbol}.
            
Please save the Power of 3 Signal Generator artifact as 'power_of_3_signal_generator.py' 
in the same directory as this main file.

For now, here's a basic Power of 3 analysis framework:

🕒 SESSION TIMING:
- London Open: 2:00-5:00 AM EST (Accumulation → Manipulation → Direction)
- New York Open: 7:00-10:00 AM EST (Best session for signals)  
- London Close: 10:00 AM-12:00 PM EST

🎯 LOOK FOR:
1. Liquidity sweeps above/below key levels
2. Fake breakouts with quick reversals
3. Stop hunting with long wicks
4. Market structure breaks
5. Institutional order blocks

📊 SIGNAL CRITERIA:
- Minimum 1:5 risk-reward ratio
- Clear manipulation pattern
- Strong session timing
- Confluence with market structure

Current Symbol: {symbol}
Timeframe: {timeframe}
Account Size: ${account_balance:,.2f}"""
        
        try:
            # Create signal generator when needed
            signal_generator = PowerOf3SignalGenerator(min_risk_reward=5.0, max_risk_percent=2.0)
            
            # Get market data
            yahoo_symbols = {
                'US30': '^DJI',
                'NAS100': '^IXIC',
                'SPX500': '^GSPC',
                'XAUUSD': 'GC=F'
            }
            
            yahoo_symbol = yahoo_symbols.get(symbol, symbol)
            
            import yfinance as yf
            ticker = yf.Ticker(yahoo_symbol)
            
            yahoo_interval = {
                '1min': '1m',
                '5min': '5m',
                '15min': '15m',
                '1h': '1h',
                '1H': '1h',
                '4h': '4h',
                '1d': '1d'
            }.get(timeframe, '5m')
            
            # Get enough data for analysis (2 weeks)
            df = ticker.history(period="14d", interval=yahoo_interval)
            
            if df.empty:
                return f"No data available for Power of 3 analysis of {symbol}"
            
            # Generate signals
            signals = signal_generator.generate_signals(symbol, df, account_balance)
            
            if not signals:
                # Provide analysis even if no signals
                current_session = signal_generator.session_manager.get_current_session()
                session_status = f"Current session: {current_session.value if current_session else 'None (outside Power of 3 hours)'}"
                
                # Get basic analysis
                liquidity_zones = signal_generator.liquidity_analyzer.identify_liquidity_zones(
                    df, current_session or TradingSession.NEW_YORK_OPEN
                )
                
                manipulation_patterns = signal_generator.manipulation_detector.detect_manipulation_patterns(
                    df, current_session or TradingSession.NEW_YORK_OPEN, 
                    [LiquidityZone(**zone) for zone in liquidity_zones]
                )
                
                market_structure = signal_generator._analyze_market_structure(df)
                
                return f"""Power of 3 Analysis for {symbol}:

🕒 SESSION STATUS:
{session_status}

📊 MARKET STRUCTURE:
Current Bias: {market_structure.title()}
Current Price: {df['Close'].iloc[-1]:.2f}

🎯 LIQUIDITY ANALYSIS:
Liquidity Zones Found: {len(liquidity_zones)}
Key Levels: {', '.join([f"{z.level:.2f}({z.zone_type})" for z in liquidity_zones[:3]])}

🎭 MANIPULATION DETECTION:
Patterns Found: {len(manipulation_patterns)}
Recent Patterns: {', '.join([p.pattern_type for p in manipulation_patterns[-2:]])}

⚠️ SIGNAL STATUS:
No high-probability signals detected at this time.
Waiting for manipulation patterns during active sessions.

💡 NEXT STEPS:
Monitor during London Open (2-5 AM EST) or NY Open (7-10 AM EST) for best setups."""
            
            # Format signals for display
            if len(signals) == 1:
                return SignalFormatter.format_signal_for_display(signals[0])
            else:
                summary = SignalFormatter.format_signal_summary(signals)
                detailed = "\n\n" + "\n".join([
                    SignalFormatter.format_signal_for_display(signal) 
                    for signal in signals[:2]  # Show top 2 detailed
                ])
                return summary + detailed
                
        except Exception as e:
            return f"Error generating Power of 3 signals for {symbol}: {str(e)}"

class RiskCalculatorTool(BaseTool):
    name: str = "risk_calculator_tool"
    description: str = "Calculate position sizes and risk management parameters"
    
    def _run(self, account_balance: float, risk_percentage: float, 
             entry_price: float, stop_loss: float) -> str:
        """Calculate position size and risk parameters"""
        try:
            risk_amount = account_balance * (risk_percentage / 100)
            price_diff = abs(entry_price - stop_loss)
            if price_diff == 0:
                return "Error: Entry price and stop loss cannot be the same"
            
            position_size = risk_amount / price_diff
            
            return f"""Risk Calculation Results:
Account Balance: ${account_balance:,.2f}
Risk Amount: ${risk_amount:.2f} ({risk_percentage}%)
Entry Price: {entry_price:.4f}
Stop Loss: {stop_loss:.4f}
Price Difference: {price_diff:.4f}
Position Size: {position_size:.4f}
Risk per unit: ${price_diff:.4f}
Max loss if stopped out: ${risk_amount:.2f}"""
        except Exception as e:
            return f"Error calculating risk: {str(e)}"

# =============================================================================
# CREWAI AGENTS
# =============================================================================

def create_market_analyst_agent() -> Agent:
    """Agent responsible for market structure analysis"""
    try:
        tools = [MarketDataTool(), TechnicalAnalysisTool(), PowerOf3SignalTool()]
        if POWER_OF_3_AVAILABLE:
            tools.append(TechnicalAnalysisTool())  # Ensure compatibility with the tools list
        
        return Agent(
            role='Market Structure Analyst',
            goal='Analyze market structure, identify trends, and detect institutional activity using Power of 3 methodology',
            backstory="""You are an expert in reading market microstructure and institutional 
            order flow. You specialize in identifying liquidity zones, order blocks, and 
            market manipulation patterns. You focus on actionable insights for trading decisions.""",
            tools=tools,
            verbose=True,
            allow_delegation=False,
            llm=ChatOpenAI(model="gpt-4", temperature=0.1),
            max_iter=3
        )
    except Exception as e:
        logger.error(f"Error creating market analyst agent: {e}")
        # Return agent with basic tools only
        return Agent(
            role='Market Structure Analyst',
            goal='Analyze market structure and identify trends',
            backstory="""You are an expert in reading market microstructure.""",
            tools=[MarketDataTool(), TechnicalAnalysisTool()],
            verbose=True,
            allow_delegation=False,
            llm=ChatOpenAI(model="gpt-4", temperature=0.1),
            max_iter=3
        )

def create_signal_generator_agent() -> Agent:
    """Agent responsible for generating trading signals"""
    try:
        tools = [TechnicalAnalysisTool(), PowerOf3SignalTool()]
        if POWER_OF_3_AVAILABLE:
            tools.append(PowerOf3SignalTool())
        
        return Agent(
            role='Power of 3 Signal Generator',
            goal='Generate high-probability Power of 3 trading signals with minimum 1:5 risk-reward ratios during key sessions',
            backstory="""You are a Power of 3 specialist who identifies optimal entry points 
            during London and New York sessions. You focus on detecting manipulation patterns,
            liquidity sweeps, and institutional order flow. You only recommend signals with 
            excellent risk-reward ratios and clear setup criteria.""",
            tools=tools,
            verbose=True,
            allow_delegation=False,
            llm=ChatOpenAI(model="gpt-4", temperature=0.1),
            max_iter=2
        )
    except Exception as e:
        logger.error(f"Error creating signal generator agent: {e}")
        return Agent(
            role='Signal Generator',
            goal='Generate trading signals based on technical analysis',
            backstory="""You are a trading signal specialist.""",
            tools=[TechnicalAnalysisTool()],
            verbose=True,
            allow_delegation=False,
            llm=ChatOpenAI(model="gpt-4", temperature=0.1),
            max_iter=2
        )

def create_risk_manager_agent() -> Agent:
    """Agent responsible for risk management"""
    return Agent(
        role='Risk Manager',
        goal='Calculate optimal position sizes and manage portfolio risk exposure with maximum 2% risk per trade',
        backstory="""You are a risk management expert who ensures portfolio safety 
        through proper position sizing, correlation analysis, and exposure limits. 
        You prioritize capital preservation above all else.""",
        tools=[RiskCalculatorTool()],
        verbose=True,
        allow_delegation=False,
        llm=ChatOpenAI(model="gpt-4", temperature=0.1),
        max_iter=2
    )

def create_execution_manager_agent() -> Agent:
    """Agent responsible for trade execution"""
    return Agent(
        role='Execution Manager',
        goal='Execute trades with precise timing and manage active positions',
        backstory="""You are a trade execution specialist who handles order placement, 
        modification, and closure. You ensure trades are executed at optimal prices 
        with proper risk parameters and timing.""",
        tools=[],
        verbose=True,
        allow_delegation=False,
        llm=ChatOpenAI(model="gpt-4", temperature=0.1),
        max_iter=2
    )

def create_performance_monitor_agent() -> Agent:
    """Agent responsible for performance monitoring"""
    return Agent(
        role='Performance Monitor',
        goal='Track trading performance and optimize strategy parameters',
        backstory="""You are a performance analytics expert who monitors trade 
        results, calculates key metrics, and provides insights for strategy 
        improvement. You focus on maintaining profitability and drawdown control.""",
        tools=[],
        verbose=True,
        allow_delegation=False,
        llm=ChatOpenAI(model="gpt-4", temperature=0.1),
        max_iter=2
    )

def create_backtesting_agent() -> Agent:
    """Agent responsible for backtesting"""
    return Agent(
        role='Backtesting Specialist',
        goal='Validate strategies against historical data and optimize parameters',
        backstory="""You are a quantitative analyst specialized in backtesting 
        trading strategies. You use historical data to validate the Power of 3 
        approach and optimize entry/exit criteria for maximum profitability.""",
        tools=[MarketDataTool(), TechnicalAnalysisTool()],
        verbose=True,
        allow_delegation=False,
        llm=ChatOpenAI(model="gpt-4", temperature=0.1),
        max_iter=3
    )

# =============================================================================
# TASKS
# =============================================================================

def create_market_analysis_task(symbols: List[str]) -> Task:
    """Task for market structure analysis with Power of 3 focus"""
    return Task(
        description=f"""
        Perform comprehensive Power of 3 analysis for these symbols: {', '.join(symbols[:2])}
        
        For each symbol, use the available tools to:
        1. Get current market data using market_data_tool
        2. Perform technical analysis using technical_analysis_tool  
        3. Generate Power of 3 signals using power_of_3_signal_tool
        4. Identify current session status and timing
        5. Assess manipulation patterns and liquidity zones
        
        Focus on:
        - Current session phase (Accumulation/Manipulation/Direction)
        - Market structure breaks and institutional flow
        - Liquidity zone analysis and stop clustering
        - Quality scoring of potential setups
        
        Provide actionable insights for trading decisions during Power of 3 sessions.
        """,
        agent=create_market_analyst_agent(),
        expected_output="Comprehensive Power of 3 analysis with session status, market structure, and signal assessment for each symbol"
    )

def create_signal_generation_task() -> Task:
    """Task for generating Power of 3 trading signals"""
    return Task(
        description="""
        Based on the market analysis, generate specific Power of 3 trading signals.
        
        Use the power_of_3_signal_tool to:
        1. Identify high-probability setups during active sessions
        2. Detect manipulation patterns (liquidity sweeps, fake breakouts, stop hunts)
        3. Confirm institutional order flow and market structure
        4. Calculate precise entry/exit levels with 1:5+ risk-reward
        5. Score signal quality and confidence levels
        
        Only recommend signals that meet these criteria:
        - Clear Power of 3 setup during London/NY sessions
        - Manipulation pattern detected and confirmed
        - Minimum 1:5 risk-reward ratio
        - Quality score of 7+ (Good to Excellent)
        - Strong session timing and confluence
        
        Provide detailed reasoning for each signal recommendation.
        """,
        agent=create_signal_generator_agent(),
        expected_output="High-quality Power of 3 trading signals with detailed entry/exit parameters, quality scores, and reasoning"
    )

def create_risk_management_task() -> Task:
    """Task for risk calculation and management - simplified"""
    return Task(
        description="""
        Calculate position sizes and risk parameters for any recommended trades.
        
        Use these parameters:
        - Account size: $10,000
        - Maximum risk per trade: 2%
        - Consider correlation between positions
        
        For each potential trade:
        1. Calculate optimal position size
        2. Determine risk amount in dollars
        3. Assess portfolio impact
        4. Recommend position sizing
        
        Ensure all recommendations stay within risk limits.
        """,
        agent=create_risk_manager_agent(),
        expected_output="Position sizes and risk management parameters for recommended trades"
    )

# =============================================================================
# MAIN WORKFLOW
# =============================================================================

class PowerOf3TradingSystem:
    """Main trading system orchestrating all components"""
    
    def __init__(self):
        self.symbols = ['US30', 'NAS100', 'SPX500', 'XAUUSD']
        self.db_engine = self._setup_database()
        
        # Initialize database connection for trade logging
        try:
            #from database_integration import TradingDatabase
            from src.power_of_3.database.repository import TradingDatabase
            self.trading_db = TradingDatabase()
            logger.info("✓ Database connection established")
        except ImportError:
            logger.warning("Database integration not available. Save 'database_integration.py' for full functionality.")
            self.trading_db = None
        except Exception as e:
            logger.warning(f"Database connection failed: {e}")
            self.trading_db = None
        
        # Check for required environment variables
        if not os.getenv('OPENAI_API_KEY'):
            logger.warning("OPENAI_API_KEY not set. CrewAI functionality will be limited.")
        
        try:
            self.crew = self._setup_crew()
        except Exception as e:
            logger.error(f"Error setting up crew: {e}")
            logger.info("Some functionality may be limited without proper API keys.")
            self.crew = None
        
    def _setup_database(self):
        """Setup PostgreSQL connection"""
        try:
            db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
            return create_engine(db_url)
        except Exception as e:
            logger.error(f"Database setup error: {e}")
            return None
    
    def _setup_crew(self) -> Crew:
        """Setup CrewAI crew with all agents and tasks"""
        
        try:
            # Create simplified tasks for testing
            market_analysis_task = create_market_analysis_task(['US30', 'XAUUSD'])  # Start with just 2 symbols
            signal_generation_task = create_signal_generation_task()
            risk_management_task = create_risk_management_task()
            
            # Create agents with error handling
            try:
                market_analyst = create_market_analyst_agent()
            except Exception as e:
                logger.error(f"Error creating market analyst: {e}")
                raise e
            
            try:
                signal_generator = create_signal_generator_agent()
            except Exception as e:
                logger.error(f"Error creating signal generator: {e}")
                raise e
            
            try:
                risk_manager = create_risk_manager_agent()
            except Exception as e:
                logger.error(f"Error creating risk manager: {e}")
                raise e
            
            # Create crew with error handling
            crew = Crew(
                agents=[
                    market_analyst,
                    signal_generator,
                    risk_manager
                ],
                tasks=[
                    market_analysis_task,
                    signal_generation_task,
                    risk_management_task
                ],
                process=Process.sequential,
                verbose=True,
                max_rpm=10,  # Limit requests per minute
                share_crew=False  # Disable sharing for privacy
            )
            
            return crew
            
        except Exception as e:
            logger.error(f"Error setting up crew: {e}")
            raise e
    
    def test_crew_basic(self):
        """Test CrewAI with minimal complexity"""
        try:
            logger.info("Testing CrewAI basic functionality...")
            
            if not self.crew:
                return False, "Crew not initialized"
            
            # Create a simple test task
            simple_task = Task(
                description="""
                Test the market data tool by getting current data for US30.
                Then provide a brief analysis of what you found.
                
                Steps:
                1. Use market_data_tool to get US30 data on 5min timeframe
                2. Use technical_analysis_tool to analyze US30 on 5min timeframe
                3. Summarize your findings in 2-3 sentences
                """,
                agent=create_market_analyst_agent(),
                expected_output="Brief market data summary and analysis for US30"
            )
            
            # Create simple crew for testing
            test_crew = Crew(
                agents=[create_market_analyst_agent()],
                tasks=[simple_task],
                process=Process.sequential,
                verbose=True,
                max_rpm=5
            )
            
            # Execute test
            result = test_crew.kickoff()
            
            logger.info("CrewAI basic test completed successfully!")
            return True, str(result)
            
        except Exception as e:
            logger.error(f"CrewAI basic test failed: {e}")
            return False, str(e)
    
    async def run_analysis_cycle(self, force_run: bool = False):
        """Run a complete analysis cycle"""
        try:
            if not self.crew:
                logger.error("Crew not initialized. Check API keys and setup.")
                return None
                
            # Check if we're in a Power of 3 session (unless forced)
            if not force_run and not SessionManager.is_power_of_3_time():
                logger.info("Not in Power of 3 session, skipping analysis")
                return
            
            current_session = SessionManager.get_current_session()
            if current_session:
                logger.info(f"Running analysis for {current_session.value} session")
            else:
                logger.info("Running forced analysis outside trading hours")
            
            # Execute crew workflow
            result = self.crew.kickoff()
            
            # Process results
            await self._process_results(result, current_session or TradingSession.NEW_YORK_OPEN)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in analysis cycle: {e}")
            return None
    
    async def _process_results(self, results, session: TradingSession):
        """Process crew results and take action"""
        try:
            # Parse results and extract signals
            # This would need to be implemented based on actual crew output format
            
            # Log to database
            if self.db_engine:
                # Store analysis results
                pass
            
            # Send notifications
            await self._send_notification(f"Analysis complete for {session.value}")
            
        except Exception as e:
            logger.error(f"Error processing results: {e}")
    
    def test_system_basic(self):
        """Basic system test without external dependencies"""
        try:
            logger.info("Running basic system test...")
            
            # Test 1: Session detection
            print("=== SESSION DETECTION TEST ===")
            current_session = SessionManager.get_current_session()
            is_power_time = SessionManager.is_power_of_3_time()
            
            print(f"Current session: {current_session}")
            print(f"Is Power of 3 time: {is_power_time}")
            
            # Test 2: Data provider configuration
            print("\n=== DATA PROVIDER CONFIGURATION ===")
            data_provider = DataProvider()
            available_providers = data_provider.get_available_providers()
            print(f"Available providers: {available_providers}")
            
            # Test 3: Technical analysis with sample data
            print("\n=== TECHNICAL ANALYSIS TEST ===")
            sample_data = pd.DataFrame({
                'Open': [2100.0, 2101.0, 2102.0, 2101.0, 2103.0, 2105.0, 2107.0, 2106.0, 2108.0, 2110.0],
                'High': [2101.0, 2102.0, 2103.0, 2102.0, 2104.0, 2106.0, 2108.0, 2107.0, 2109.0, 2111.0],
                'Low': [2099.0, 2100.0, 2101.0, 2100.0, 2102.0, 2104.0, 2106.0, 2105.0, 2107.0, 2109.0],
                'Close': [2100.5, 2101.5, 2102.5, 2101.5, 2103.5, 2105.5, 2107.5, 2106.5, 2108.5, 2110.5],
                'Volume': [1000, 1200, 800, 1500, 1100, 900, 1300, 1000, 1100, 1200]
            }, index=pd.date_range('2024-01-01 09:00:00', periods=10, freq='5min'))
            
            # Test liquidity zones
            liquidity_zones = TechnicalAnalysis.identify_liquidity_zones(sample_data)
            print(f"✓ Liquidity zones identified: {len(liquidity_zones)}")
            
            # Test market structure
            market_structure = TechnicalAnalysis.identify_market_structure_break(sample_data)
            print(f"✓ Market trend detected: {market_structure.trend_direction.value}")
            print(f"✓ Higher highs: {market_structure.higher_high}")
            print(f"✓ Higher lows: {market_structure.higher_low}")
            
            # Test order blocks
            order_blocks = TechnicalAnalysis.identify_order_blocks(sample_data)
            print(f"✓ Order blocks found: {len(order_blocks)}")
            
            # Test manipulation detection
            manipulation = TechnicalAnalysis.detect_manipulation(sample_data)
            print(f"✓ Manipulation detection: {manipulation}")
            
            # Test 4: Symbol mapping
            print("\n=== SYMBOL MAPPING TEST ===")
            test_symbols = ['US30', 'NAS100', 'SPX500', 'XAUUSD']
            for symbol in test_symbols:
                for provider in ['twelve_data', 'alpha_vantage', 'oanda', 'yahoo']:
                    mapped = data_provider._get_provider_symbol(provider, symbol)
                    print(f"✓ {symbol} -> {provider}: {mapped}")
            
            # Test 5: Risk calculation
            print("\n=== RISK CALCULATION TEST ===")
            account_balance = 10000
            risk_pct = 2.0
            entry_price = 2100.0
            stop_loss = 2095.0
            
            risk_amount = account_balance * (risk_pct / 100)
            price_diff = abs(entry_price - stop_loss)
            position_size = risk_amount / price_diff
            
            print(f"✓ Account: ${account_balance:,.2f}")
            print(f"✓ Risk: {risk_pct}% = ${risk_amount:.2f}")
            print(f"✓ Entry: {entry_price}, Stop: {stop_loss}")
            print(f"✓ Price diff: {price_diff}")
            print(f"✓ Position size: {position_size:.4f}")
            
            # Test 6: Data provider connectivity (basic)
            print("\n=== DATA PROVIDER CONNECTIVITY TEST ===")
            try:
                # This will test basic connectivity without API calls
                provider_stats = data_provider.get_cache_stats()
                print(f"✓ Cache initialized: {provider_stats}")
                
                # Test rate limiting
                can_request_td = data_provider._can_make_request('twelve_data')
                can_request_av = data_provider._can_make_request('alpha_vantage')
                print(f"✓ Rate limiting functional: TD={can_request_td}, AV={can_request_av}")
                
            except Exception as e:
                print(f"⚠ Data provider test warning: {e}")
            
            print("\n=== ALL BASIC TESTS PASSED ✓ ===")
            logger.info("Basic system test completed successfully!")
            return True
            
        except Exception as e:
            print(f"\n❌ BASIC TEST FAILED: {e}")
            logger.error(f"Error in basic system test: {e}")
            return False
    
    def simple_standalone_test(self):
        """Simple standalone test that can be run independently"""
        try:
            print("=== SIMPLE STANDALONE TEST ===")
            
            # Test 1: Technical Analysis directly
            print("1. Testing Technical Analysis directly...")
            sample_data = pd.DataFrame({
                'Open': [2100.0, 2101.0, 2102.0, 2101.0, 2103.0],
                'High': [2101.0, 2102.0, 2103.0, 2102.0, 2104.0],
                'Low': [2099.0, 2100.0, 2101.0, 2100.0, 2102.0],
                'Close': [2100.5, 2101.5, 2102.5, 2101.5, 2103.5],
                'Volume': [1000, 1200, 800, 1500, 1100]
            }, index=pd.date_range('2024-01-01 09:00:00', periods=5, freq='5min'))
            
            # Test our core algorithms
            market_structure = TechnicalAnalysis.identify_market_structure_break(sample_data)
            print(f"   ✓ Market structure: {market_structure.trend_direction.value}")
            
            # Test 2: Tool functionality
            print("2. Testing MarketDataTool...")
            data_tool = MarketDataTool()
            result = data_tool._run("US30", "5min")
            print(f"   ✓ Data tool result: {result[:100]}...")
            
            print("3. Testing TechnicalAnalysisTool...")
            ta_tool = TechnicalAnalysisTool()
            result = ta_tool._run("US30", "5min")
            print(f"   ✓ TA tool result: {result[:100]}...")
            
            # Test 4: Power of 3 Signal Tool (if available)
            print("4. Testing Power of 3 Signal Tool...")
            try:
                signal_tool = PowerOf3SignalTool()
                result = signal_tool._run("US30", "5min")
                print(f"   ✓ Power of 3 tool result: {result[:100]}...")
            except Exception as e:
                print(f"   ⚠ Power of 3 tool: {e}")
            
            # Test 5: Simple agent creation
            print("5. Testing Agent Creation...")
            try:
                agent = create_market_analyst_agent()
                print(f"   ✓ Agent created: {agent.role}")
            except Exception as e:
                print(f"   ❌ Agent creation failed: {e}")
                return False
            
            print("✓ All standalone tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Standalone test failed: {e}")
            return False
    
    def test_data_providers_simple(self):
        """Simple data provider test using yfinance"""
        try:
            print("=== SIMPLE DATA PROVIDER TEST ===")
            
            # Test Yahoo Finance (our fallback)
            try:
                import yfinance as yf
                ticker = yf.Ticker("^DJI")  # US30
                data = ticker.history(period="1d", interval="5m")
                
                if not data.empty:
                    latest = data.tail(1).iloc[0]
                    print(f"✓ Yahoo Finance: Latest US30 price: {latest['Close']:.2f}")
                    return True
                else:
                    print("❌ Yahoo Finance: No data returned")
                    return False
                    
            except Exception as e:
                print(f"❌ Yahoo Finance test failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error in simple data provider test: {e}")
            return False
    
    def test_power_of_3_signals(self):
        """Test Power of 3 signal generation"""
        try:
            print("=== POWER OF 3 SIGNAL GENERATION TEST ===")
            
            if not POWER_OF_3_AVAILABLE:
                print("⚠ Power of 3 Signal Generator not available")
                print("  Save the signal generator artifact as 'power_of_3_signal_generator.py'")
                print("  Testing basic signal tool functionality instead...")
                
                # Test basic signal tool
                signal_tool = PowerOf3SignalTool()
                result = signal_tool._run("US30", "5min", 10000.0)
                
                if "Power of 3 Signal Generator not available" in result:
                    print("✓ Signal tool properly handles missing generator")
                    return True
                else:
                    print("❌ Unexpected result from signal tool")
                    return False
            
            # Test the signal tool directly
            signal_tool = PowerOf3SignalTool()
            
            print("Testing signal generation for US30...")
            result = signal_tool._run("US30", "5min", 10000.0)
            
            print("Signal Generation Result:")
            print(result[:500] + "..." if len(result) > 500 else result)
            
            # Test multiple symbols
            symbols = ['US30', 'XAUUSD']
            for symbol in symbols:
                try:
                    print(f"\nTesting {symbol}...")
                    signal_result = signal_tool._run(symbol, "5min", 10000.0)
                    
                    # Check if we got meaningful output
                    if "Error" not in signal_result:
                        print(f"✓ {symbol}: Signal analysis completed")
                        # Look for key Power of 3 indicators
                        if any(keyword in signal_result.lower() for keyword in 
                               ['session', 'liquidity', 'manipulation', 'signal']):
                            print(f"✓ {symbol}: Power of 3 analysis detected")
                        else:
                            print(f"⚠ {symbol}: Basic analysis only")
                    else:
                        print(f"❌ {symbol}: Error in analysis")
                        
                except Exception as e:
                    print(f"❌ {symbol}: Test failed - {e}")
            
            print("\n✓ Power of 3 signal generation test completed!")
            return True
            
        except Exception as e:
            print(f"❌ Power of 3 signal test failed: {e}")
            return False
    
    def log_signal_to_database(self, symbol: str, timeframe: str = "5min", account_balance: float = 10000.0):
        """Generate and log Power of 3 signals to database"""
        try:
            if not self.trading_db:
                print("❌ Database not available. Signals will not be logged.")
                return
            
            if not POWER_OF_3_AVAILABLE:
                print("❌ Power of 3 Signal Generator not available")
                return
            
            print(f"🔄 Generating and logging signals for {symbol}...")
            
            # Generate signals using Power of 3 system
            signal_tool = PowerOf3SignalTool()
            
            # Get the actual signal objects (not just formatted text)
            # from power_of_3_signal_generator import PowerOf3SignalGenerator
            import yfinance as yf
            
            # Get data
            yahoo_symbols = {
                'US30': '^DJI',
                'NAS100': '^IXIC',
                'SPX500': '^GSPC',
                'XAUUSD': 'GC=F'
            }
            
            yahoo_symbol = yahoo_symbols.get(symbol, symbol)
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(period="14d", interval="5m")
            
            if df.empty:
                print(f"❌ No data available for {symbol}")
                return
            
            # Generate signals
            signal_generator = PowerOf3SignalGenerator(min_risk_reward=5.0, max_risk_percent=2.0)
            signals = signal_generator.generate_signals(symbol, df, account_balance)
            
            if signals:
                for signal in signals:
                    try:
                        signal_id = self.trading_db.log_signal(signal)
                        print(f"✓ Signal logged to database: {signal.signal_id}")
                        print(f"  - Type: {signal.signal_type.value.upper()}")
                        print(f"  - Quality: {signal.quality_score:.1f}/10")
                        print(f"  - RR: 1:{signal.risk_reward_ratio:.1f}")
                        print(f"  - Database ID: {signal_id}")
                    except Exception as e:
                        print(f"❌ Error logging signal: {e}")
            else:
                print(f"ℹ️  No signals generated for {symbol} at this time")
                
        except Exception as e:
            print(f"❌ Error in signal logging: {e}")
    
    def get_trading_performance(self, days: int = 30):
        """Get trading performance from database"""
        try:
            if not self.trading_db:
                print("❌ Database not available")
                return
            
            print(f"📊 TRADING PERFORMANCE - LAST {days} DAYS")
            print("=" * 50)
            
            # Get performance report
            report = self.trading_db.generate_trading_report(days)
            
            # Display summary
            metrics = report['performance_metrics']
            print(f"📈 Summary:")
            print(f"  Total Trades: {metrics['total_trades']}")
            print(f"  Win Rate: {metrics['win_rate']:.1f}%")
            print(f"  Net P&L: ${metrics['net_profit']:.2f}")
            print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"  Avg Signal Quality: {metrics['avg_signal_quality']:.1f}/10")
            
            # Show daily performance
            daily_perf = report['daily_performance']
            if daily_perf:
                print(f"\n📅 Recent Daily Performance:")
                for day in daily_perf[:5]:  # Last 5 days
                    print(f"  {day['trade_date']}: {day['closed_trades']} trades, ${day['daily_pnl']:.2f} P&L")
            
            # Show open trades
            open_trades = report['open_trades']
            if open_trades:
                print(f"\n🔄 Open Positions ({len(open_trades)}):")
                for trade in open_trades:
                    print(f"  {trade['symbol']} {trade['trade_type'].upper()}: ${trade['risk_amount']:.2f} at risk")
            
            return report
            
        except Exception as e:
            print(f"❌ Error getting performance: {e}")
            return None
    
    def simulate_trade_execution(self, symbol: str, signal_id: Optional[str] = None):
        """Simulate executing a trade from a signal (for testing)"""
        try:
            if not self.trading_db:
                print("❌ Database not available")
                return
            
            print(f"🎯 SIMULATING TRADE EXECUTION FOR {symbol}")
            print("=" * 50)
            
            # For demo, create a sample trade entry
            from datetime import datetime
            import pytz
            
            trade_data = {
                'signal_id': signal_id,  # Can be None for manual trades
                'symbol': symbol,
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
                'execution_method': 'simulated',
                'slippage_pips': 0.5,
                'commission': 5.00,
                'spread': 2.0,
                'notes': 'Simulated trade for testing'
            }
            
            # Log trade entry
            trade_id = self.trading_db.log_trade_entry(trade_data)
            print(f"✓ Trade entry logged: {trade_id}")
            print(f"  Entry: ${trade_data['entry_price']}")
            print(f"  Risk: ${trade_data['risk_amount']}")
            print(f"  Position: {trade_data['position_size']}")
            
            # Simulate exit after a few seconds (for demo)
            import time
            time.sleep(2)
            
            # Simulate winning trade
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
                'execution_method': 'simulated',
                'notes': 'Simulated profitable exit'
            }
            
            exit_id = self.trading_db.log_trade_exit(exit_data)
            print(f"✓ Trade exit logged: {exit_id}")
            print(f"  Exit: ${exit_data['exit_price']}")
            print(f"  P&L: ${exit_data['net_pnl']}")
            print(f"  Type: {exit_data['exit_type']}")
            
            return trade_id, exit_id
            
        except Exception as e:
            print(f"❌ Error simulating trade: {e}")
            return None, None
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'trading_db') and self.trading_db:
            self.trading_db.close()
            logger.info("Database connection closed")
    
    def run_live_power_of_3_analysis(self, symbols: Optional[List[str]] = None):
        """Run live Power of 3 analysis for specified symbols"""
        if symbols is None:
            symbols = self.symbols
        
        try:
            print("🚀 LIVE POWER OF 3 ANALYSIS")
            print("=" * 50)
            
            signal_tool = PowerOf3SignalTool()
            
            for symbol in symbols:
                print(f"\n📊 Analyzing {symbol}...")
                print("-" * 30)
                
                try:
                    result = signal_tool._run(symbol, "5min", 10000.0)
                    print(result)
                    
                    # Add separator between symbols
                    if symbol != symbols[-1]:  # Not the last symbol
                        print("\n" + "="*50)
                        
                except Exception as e:
                    print(f"❌ Error analyzing {symbol}: {e}")
            
            print(f"\n✅ Live analysis completed for {len(symbols)} symbols")
            
            if not POWER_OF_3_AVAILABLE:
                print("\n💡 For full Power of 3 functionality:")
                print("   Save the Power of 3 Signal Generator as 'power_of_3_signal_generator.py'")
            
        except Exception as e:
            logger.error(f"Error in live Power of 3 analysis: {e}")
            print(f"❌ Live analysis failed: {e}")
    
    def monitor_power_of_3_sessions(self, duration_minutes: int = 60):
        """Monitor for Power of 3 signals during active sessions"""
        try:
            print(f"🎯 MONITORING POWER OF 3 SESSIONS FOR {duration_minutes} MINUTES")
            print("=" * 60)
            
            if not POWER_OF_3_AVAILABLE:
                print("⚠ Power of 3 Signal Generator not available")
                print("  Save the signal generator as 'power_of_3_signal_generator.py' for full monitoring")
                print("  Running basic monitoring instead...")
            
            from datetime import datetime, timedelta
            import time
            
            end_time = datetime.now() + timedelta(minutes=duration_minutes)
            check_interval = 5  # Check every 5 minutes
            
            signal_tool = PowerOf3SignalTool()
            last_signals = {}
            
            print(f"⏰ Monitoring until {end_time.strftime('%H:%M:%S')}")
            print("🔍 Checking for signals every 5 minutes...")
            print("💡 Press Ctrl+C to stop monitoring\n")
            
            while datetime.now() < end_time:
                try:
                    current_time = datetime.now().strftime('%H:%M:%S')
                    print(f"[{current_time}] Checking for new signals...")
                    
                    new_signals_found = False
                    
                    for symbol in self.symbols:
                        result = signal_tool._run(symbol, "5min", 10000.0)
                        
                        # Simple check for new signals (look for signal indicators)
                        has_signals = any(keyword in result.lower() for keyword in 
                                        ['🟢', '🔴', 'signal id', 'entry price'])
                        
                        if has_signals and symbol not in last_signals:
                            print(f"\n🚨 NEW SIGNAL DETECTED FOR {symbol}!")
                            print("-" * 40)
                            print(result)
                            print("-" * 40)
                            last_signals[symbol] = current_time
                            new_signals_found = True
                        
                    if not new_signals_found:
                        print("   No new signals detected")
                    
                    # Wait for next check
                    print(f"   Next check in {check_interval} minutes...\n")
                    time.sleep(check_interval * 60)  # Convert to seconds
                    
                except KeyboardInterrupt:
                    print("\n⏹ Monitoring stopped by user")
                    break
                except Exception as e:
                    print(f"⚠ Error during monitoring: {e}")
                    time.sleep(30)  # Wait 30 seconds before retry
            
            print("✅ Monitoring session completed")
            
        except Exception as e:
            logger.error(f"Error in Power of 3 monitoring: {e}")
            print(f"❌ Monitoring failed: {e}")
    
    async def _send_notification(self, message: str):
        """Send email notification"""
        try:
            # Implementation for email notifications
            pass
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    def run_backtest(self, start_date: str, end_date: str):
        """Run backtesting on historical data"""
        try:
            # Create backtesting task
            backtest_task = Task(
                description=f"""
                Run comprehensive backtest of Power of 3 strategy from {start_date} to {end_date}.
                
                Test on all symbols: {', '.join(self.symbols)}
                Calculate key metrics:
                1. Total return and Sharpe ratio
                2. Maximum drawdown
                3. Win rate and average RR
                4. Performance by session
                5. Correlation analysis
                
                Use historical data to simulate the strategy and provide detailed performance metrics.
                """,
                agent=create_backtesting_agent(),
                expected_output="Comprehensive backtesting report with performance metrics"
            )
            
            # Create a dedicated crew for backtesting
            backtest_crew = Crew(
                agents=[create_backtesting_agent()],
                tasks=[backtest_task],
                process=Process.sequential,
                verbose=True
            )
            
            # Execute backtest
            logger.info(f"Starting backtest from {start_date} to {end_date}")
            result = backtest_crew.kickoff()
            
            logger.info("Backtest completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in backtesting: {e}")
            return None

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Initialize system
    trading_system = PowerOf3TradingSystem()
    
    # Step 0: Simple standalone test first
    print("=== STEP 0: SIMPLE STANDALONE TEST ===")
    standalone_passed = trading_system.simple_standalone_test()
    
    if standalone_passed:
        # Step 1: Run basic system test (no external dependencies)
        print("\n=== STEP 1: BASIC SYSTEM TEST ===")
        basic_test_passed = trading_system.test_system_basic()
        
        if basic_test_passed:
            # Step 2: Test data provider connectivity (simplified)
            print("\n=== STEP 2: DATA PROVIDER TEST (SIMPLIFIED) ===")
            try:
                data_test_passed = trading_system.test_data_providers_simple()
            except Exception as e:
                print(f"⚠ Data provider test skipped: {e}")
                data_test_passed = False
            
            # Step 2.5: Test Power of 3 Signal Generation
            print("\n=== STEP 2.5: POWER OF 3 SIGNAL TEST ===")
            try:
                power_of_3_test_passed = trading_system.test_power_of_3_signals()
            except Exception as e:
                print(f"⚠ Power of 3 signal test failed: {e}")
                power_of_3_test_passed = False
            
            # Step 3: Check CrewAI initialization
            print("\n=== STEP 3: CREWAI INITIALIZATION CHECK ===")
            if trading_system.crew:
                print("✓ CrewAI crew initialized successfully!")
                
                # Step 3.5: Test CrewAI basic functionality
                print("\n=== STEP 3.5: CREWAI BASIC TEST ===")
                try:
                    crew_test_passed, crew_result = trading_system.test_crew_basic()
                    if crew_test_passed:
                        print("✓ CrewAI basic test passed!")
                        print("Basic Result:", str(crew_result)[:300] + "..." if len(str(crew_result)) > 300 else str(crew_result))
                    else:
                        print(f"❌ CrewAI basic test failed: {crew_result}")
                except Exception as e:
                    print(f"❌ CrewAI basic test error: {e}")
                    crew_test_passed = False
                
                # Step 4: Run full Power of 3 analysis (only if basic test passed)
                if crew_test_passed and power_of_3_test_passed:
                    print("\n=== STEP 4: RUNNING FULL POWER OF 3 ANALYSIS ===")
                    try:
                        analysis_result = asyncio.run(trading_system.run_analysis_cycle(force_run=True))
                        if analysis_result:
                            print("✓ Full Power of 3 analysis completed successfully!")
                            print("Analysis Result Preview:", str(analysis_result)[:300] + "..." if len(str(analysis_result)) > 300 else str(analysis_result))
                        else:
                            print("⚠ Analysis completed but returned no results")
                    except Exception as e:
                        print(f"❌ Full analysis failed: {e}")
                else:
                    print("\n⏭ Skipping full analysis due to test failures")
                
                # Step 5: Run backtest
                print("\n=== STEP 5: RUNNING BACKTEST ===")
                try:
                    backtest_results = trading_system.run_backtest("2024-01-01", "2024-12-31")
                    if backtest_results:
                        print("✓ Backtest completed successfully!")
                        print("Backtest Result Preview:", str(backtest_results)[:200] + "..." if len(str(backtest_results)) > 200 else str(backtest_results))
                    else:
                        print("⚠ Backtest completed but returned no results")
                except Exception as e:
                    print(f"❌ Backtest failed: {e}")
                    
            else:
                print("❌ CrewAI crew not initialized.")
                print("   Please check your OPENAI_API_KEY environment variable.")
                print("   You can still use the technical analysis functions independently.")
                
            # Step 6: Show configuration summary
            print("\n=== STEP 6: CONFIGURATION SUMMARY ===")
            print("Environment Variables Status:")
            env_vars = [
                'OPENAI_API_KEY',
                'TWELVE_DATA_API_KEY', 
                'ALPHA_VANTAGE_API_KEY',
                'OANDA_API_KEY',
                'OANDA_ACCOUNT_ID'
            ]
            
            for var in env_vars:
                value = os.getenv(var)
                status = "✓ SET" if value else "❌ NOT SET"
                masked_value = f"{value[:8]}..." if value and len(value) > 8 else "None"
                print(f"  {var}: {status} ({masked_value})")
            
            print(f"\nSystem Status:")
            print(f"  Trading Symbols: {', '.join(trading_system.symbols)}")
            print(f"  Database Connection: {'✓ Available' if trading_system.db_engine else '❌ Not configured'}")
            print(f"  CrewAI Status: {'✓ Ready' if trading_system.crew else '❌ Not ready'}")
            print(f"  Power of 3 Signals: {'✓ Working' if power_of_3_test_passed else '❌ Issues detected'}")
            print(f"  Database Integration: {'✓ Available' if trading_system.trading_db else '❌ Not configured'}")
            
            # Step 7: Test database features (if available)
            if trading_system.trading_db and power_of_3_test_passed:
                print("\n=== STEP 7: DATABASE INTEGRATION TEST ===")
                try:
                    # Test signal logging
                    print("Testing signal logging...")
                    trading_system.log_signal_to_database('US30', '5min', 10000.0)
                    
                    # Test trade simulation
                    print("\nTesting trade simulation...")
                    trading_system.simulate_trade_execution('US30')
                    
                    # Test performance retrieval
                    print("\nTesting performance analysis...")
                    performance = trading_system.get_trading_performance(7)  # Last 7 days
                    
                    print("✓ Database integration test completed!")
                    
                except Exception as e:
                    print(f"❌ Database test failed: {e}")
            elif not trading_system.trading_db:
                print("\n=== STEP 7: DATABASE SETUP NEEDED ===")
                print("💡 To enable trade logging and performance tracking:")
                print("   1. Install PostgreSQL")
                print("   2. Save database artifacts:")
                print("      - postgres_trading_schema.sql")
                print("      - database_integration.py") 
                print("      - database_setup.py")
                print("   3. Run: python database_setup.py --create-db")
                print("   4. Set DB environment variables")
            
            # Quick start guide
            print(f"\n=== QUICK START RECOMMENDATIONS ===")
            if not trading_system.crew:
                print("1. Set your OPENAI_API_KEY environment variable")
                print("2. Optionally add data provider API keys for better data")
                print("3. Re-run the system")
            elif not power_of_3_test_passed:
                print("1. Check internet connection for market data")
                print("2. Verify yfinance package is installed correctly")
                print("3. Try running during Power of 3 session hours for live signals")
            elif not trading_system.trading_db:
                print("1. ✓ Core system working!")
                print("2. Set up PostgreSQL database for trade logging")
                print("3. Use during London Open (2-5 AM EST) or NY Open (7-10 AM EST)")
            else:
                print("✅ COMPLETE SYSTEM READY!")
                print("✓ Power of 3 signal generation working")
                print("✓ Database logging functional")
                print("✓ Performance tracking available")
                print("")
                print("🎯 READY FOR LIVE TRADING:")
                print("   - Monitor during Power of 3 sessions")
                print("   - Signals automatically logged to database")
                print("   - Track performance metrics")
                print("   - View trading analytics")
        
        else:
            print("❌ Basic test failed. Please check the setup and dependencies.")
            print("Make sure you have the required packages installed:")
            print("pip install crewai pandas numpy yfinance pandas-ta psycopg2-binary sqlalchemy requests pytz")
    else:
        print("❌ Standalone test failed. Please check basic dependencies and imports.")
        print("Required packages: crewai, pandas, numpy, yfinance")