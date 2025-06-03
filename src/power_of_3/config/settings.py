"""
Configuration Management for Power of 3 Trading System
=====================================================
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables"""
    load_dotenv()
    
    return {
        'database': {
            'url': os.getenv('DATABASE_URL', 'postgresql://localhost:5432/trading_db'),
            'pool_size': int(os.getenv('DB_POOL_SIZE', '5')),
        },
        'trading': {
            'max_risk_percent': float(os.getenv('MAX_RISK_PERCENT', '2.0')),
            'account_size': float(os.getenv('ACCOUNT_SIZE', '10000')),
            'symbols': os.getenv('SYMBOLS', 'US30,NAS100,SPX500,XAUUSD').split(','),
        },
        'api_keys': {
            'openai': os.getenv('OPENAI_API_KEY'),
            'twelve_data': os.getenv('TWELVE_DATA_API_KEY'),
            'oanda': os.getenv('OANDA_API_KEY'),
        }
    }