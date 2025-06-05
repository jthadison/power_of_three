"""
Liquidity Zone Detection for Power of 3 Trading System
=====================================================
Identifies key support/resistance levels and liquidity pools.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.power_of_3.core.types import TradingSession


class LiquidityZoneDetector:
    """Detects liquidity zones and support/resistance levels"""
    
    def __init__(self, lookback_periods: int = 50):
        self.lookback_periods = lookback_periods
    
    def identify_liquidity_zones(self, df: pd.DataFrame, session: Optional[TradingSession] = None) -> List[Dict]:
        """Identify key liquidity zones from price data"""
        zones = []
        
        # Use default session if none provided
        if session is None:
            session = TradingSession.NEW_YORK_OPEN
        
        # Current timestamp for last_test
        current_time = datetime.now() if df.empty else df.index[-1]
        
        # Find swing highs and lows
        swing_highs = self._find_swing_highs(df)
        swing_lows = self._find_swing_lows(df)
        
        # Convert to liquidity zones with CORRECT field names
        for level in swing_highs:
            zones.append({
                'level': level,                    # Changed from 'price' to 'level'
                'zone_type': 'resistance',         # Changed from 'type' to 'zone_type'
                'strength': self._calculate_zone_strength(df, level),
                'session_created': session,        # Added required field
                'touches': self._count_touches(df, level),
                'last_test': current_time,         # Added required field
                'broken': False                    # Added required field with default
            })
            
        for level in swing_lows:
            zones.append({
                'level': level,                    # Changed from 'price' to 'level'
                'zone_type': 'support',            # Changed from 'type' to 'zone_type'
                'strength': self._calculate_zone_strength(df, level),
                'session_created': session,        # Added required field
                'touches': self._count_touches(df, level),
                'last_test': current_time,         # Added required field
                'broken': False                    # Added required field with default
            })
        
        # Sort by strength
        return sorted(zones, key=lambda x: x['strength'], reverse=True)
    
    def _find_swing_highs(self, df: pd.DataFrame, window: int = 5) -> List[float]:
        """Find swing high points"""
        highs = []
        if len(df) < window * 2 + 1:
            return highs
            
        for i in range(window, len(df) - window):
            if all(df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, window+1)):
                highs.append(df['high'].iloc[i])
        return highs
    
    def _find_swing_lows(self, df: pd.DataFrame, window: int = 5) -> List[float]:
        """Find swing low points"""
        lows = []
        if len(df) < window * 2 + 1:
            return lows
            
        for i in range(window, len(df) - window):
            if all(df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, window+1)):
                lows.append(df['low'].iloc[i])
        return lows
    
    def _calculate_zone_strength(self, df: pd.DataFrame, level: float, tolerance: float = 0.001) -> float:
        """Calculate the strength of a liquidity zone"""
        touches = 0
        volume_at_level = 0
        
        for _, row in df.iterrows():
            if abs(row['high'] - level) / level <= tolerance or abs(row['low'] - level) / level <= tolerance:
                touches += 1
                volume_at_level += row.get('volume', 1)
        
        # Normalize strength to 0-1 range
        strength = touches * (volume_at_level / len(df)) if len(df) > 0 else 0
        return min(strength, 1.0)
    
    def _count_touches(self, df: pd.DataFrame, level: float, tolerance: float = 0.001) -> int:
        """Count how many times price touched this level"""
        touches = 0
        for _, row in df.iterrows():
            if abs(row['high'] - level) / level <= tolerance or abs(row['low'] - level) / level <= tolerance:
                touches += 1
        return touches
    
    def detect_liquidity_sweeps(self, df: pd.DataFrame, session: TradingSession = TradingSession.NEW_YORK_OPEN) -> List[Dict]:
        """Detect liquidity sweeps (stop hunts)"""
        sweeps = []
        zones = self.identify_liquidity_zones(df, session)
        
        for zone in zones[:10]:  # Check top 10 zones
            level = zone['level']              # Updated field name
            zone_type = zone['zone_type']      # Updated field name
            
            # Look for price action that swept liquidity
            for i in range(1, len(df)):
                current = df.iloc[i]
                previous = df.iloc[i-1]
                
                if zone_type == 'resistance':
                    # Check for liquidity sweep above resistance
                    if current['high'] > level and previous['high'] <= level:
                        if current['close'] < level:  # But closed back below
                            sweeps.append({
                                'type': 'resistance_sweep',
                                'level': level,
                                'timestamp': current.name,
                                'sweep_high': current['high']
                            })
                
                elif zone_type == 'support':
                    # Check for liquidity sweep below support  
                    if current['low'] < level and previous['low'] >= level:
                        if current['close'] > level:  # But closed back above
                            sweeps.append({
                                'type': 'support_sweep',
                                'level': level,
                                'timestamp': current.name,
                                'sweep_low': current['low']
                            })
        
        return sweeps
        """Detect liquidity sweeps (stop hunts)"""
        sweeps = []
        zones = self.identify_liquidity_zones(df)
        
        for zone in zones[:10]:  # Check top 10 zones
            level = zone['price']
            zone_type = zone['type']
            
            # Look for price action that swept liquidity
            for i in range(1, len(df)):
                current = df.iloc[i]
                previous = df.iloc[i-1]
                
                if zone_type == 'resistance':
                    # Check for liquidity sweep above resistance
                    if current['high'] > level and previous['high'] <= level:
                        if current['close'] < level:  # But closed back below
                            sweeps.append({
                                'type': 'resistance_sweep',
                                'level': level,
                                'timestamp': current.name,
                                'sweep_high': current['high']
                            })
                
                elif zone_type == 'support':
                    # Check for liquidity sweep below support  
                    if current['low'] < level and previous['low'] >= level:
                        if current['close'] > level:  # But closed back above
                            sweeps.append({
                                'type': 'support_sweep',
                                'level': level,
                                'timestamp': current.name,
                                'sweep_low': current['low']
                            })
        
        return sweeps