# =============================================================================
# MANIPULATION DETECTOR
# =============================================================================

from datetime import datetime
from typing import List
import pandas as pd

from src.power_of_3.core.types import LiquidityZone, ManipulationPattern, TradingSession


class ManipulationDetector:
    """Detects institutional manipulation patterns"""
    
    @staticmethod
    def detect_manipulation_patterns(df: pd.DataFrame, session: TradingSession,
                                   liquidity_zones: List[LiquidityZone]) -> List[ManipulationPattern]:
        """Detect various manipulation patterns"""
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        # 1. Liquidity sweeps
        sweep_patterns = ManipulationDetector._detect_liquidity_sweeps(
            df, liquidity_zones, session
        )
        patterns.extend(sweep_patterns)
        
        # 2. Fake breakouts
        breakout_patterns = ManipulationDetector._detect_fake_breakouts(
            df, liquidity_zones, session
        )
        patterns.extend(breakout_patterns)
        
        # 3. Stop hunts
        stop_hunt_patterns = ManipulationDetector._detect_stop_hunts(
            df, session
        )
        patterns.extend(stop_hunt_patterns)
        
        return patterns
    
    @staticmethod
    def _detect_liquidity_sweeps(df: pd.DataFrame, zones: List[LiquidityZone],
                               session: TradingSession) -> List[ManipulationPattern]:
        """Detect liquidity sweeps above/below key levels"""
        patterns = []
        recent_data = df.tail(20)  # Look at last 20 bars
        
        for zone in zones:
            if zone.broken:  # Skip already broken zones
                continue
            
            for i in range(1, len(recent_data)):
                current_bar = recent_data.iloc[i]
                previous_bar = recent_data.iloc[i-1]
                
                # Check for sweep above resistance
                if (zone.zone_type == 'resistance' and 
                    current_bar['High'] > zone.level and
                    current_bar['Close'] < zone.level and
                    previous_bar['High'] <= zone.level):
                    
                    # Confirm it's a manipulation (quick move back below)
                    reversal_strength = (zone.level - current_bar['Close']) / zone.level
                    
                    if reversal_strength >= 0.002:  # At least 0.2% reversal
                        patterns.append(ManipulationPattern(
                            pattern_type='liquidity_sweep',
                            start_time=pd.to_datetime(str(previous_bar.name)) if not isinstance(previous_bar.name, datetime) else previous_bar.name,
                            end_time=pd.to_datetime(str(current_bar.name)) if not isinstance(current_bar.name, datetime) else current_bar.name,
                            trigger_level=zone.level,
                            reversal_level=current_bar['Close'],
                            strength=min(reversal_strength * 100, 1.0),
                            volume_surge=ManipulationDetector._check_volume_surge(
                                recent_data, i
                            ),
                            session=session
                        ))
                        zone.broken = True
                
                # Check for sweep below support
                elif (zone.zone_type == 'support' and 
                      current_bar['Low'] < zone.level and
                      current_bar['Close'] > zone.level and
                      previous_bar['Low'] >= zone.level):
                    
                    reversal_strength = (current_bar['Close'] - zone.level) / zone.level
                    
                    if reversal_strength >= 0.002:
                        patterns.append(ManipulationPattern(
                            pattern_type='liquidity_sweep',
                            start_time=pd.to_datetime(str(previous_bar.name)) if not isinstance(previous_bar.name, datetime) else previous_bar.name,
                            end_time=pd.to_datetime(str(current_bar.name)) if not isinstance(current_bar.name, datetime) else current_bar.name,
                            trigger_level=zone.level,
                            reversal_level=current_bar['Close'],
                            strength=min(reversal_strength * 100, 1.0),
                            volume_surge=ManipulationDetector._check_volume_surge(
                                recent_data, i
                            ),
                            session=session
                        ))
                        zone.broken = True
        
        return patterns
    
    @staticmethod
    def _detect_fake_breakouts(df: pd.DataFrame, zones: List[LiquidityZone],
                             session: TradingSession) -> List[ManipulationPattern]:
        """Detect fake breakouts that quickly reverse"""
        patterns = []
        recent_data = df.tail(15)
        
        for i in range(2, len(recent_data)):
            current_bar = recent_data.iloc[i]
            prev_bar = recent_data.iloc[i-1]
            prev_prev_bar = recent_data.iloc[i-2]
            
            # Look for strong moves followed by immediate reversals
            
            # Upside fake breakout
            if (current_bar['High'] > prev_bar['High'] and
                prev_bar['High'] > prev_prev_bar['High'] and
                current_bar['Close'] < prev_bar['Open']):
                
                breakout_size = (current_bar['High'] - prev_prev_bar['High']) / prev_prev_bar['High']
                reversal_size = (current_bar['High'] - current_bar['Close']) / current_bar['High']
                
                if breakout_size >= 0.003 and reversal_size >= 0.004:  # Significant move and reversal
                    patterns.append(ManipulationPattern(
                        pattern_type='fake_breakout',
                        start_time=pd.to_datetime(str(prev_prev_bar.name)) if not isinstance(prev_prev_bar.name, datetime) else prev_prev_bar.name,
                        end_time=pd.to_datetime(str(current_bar.name)) if not isinstance(current_bar.name, datetime) else current_bar.name,
                        trigger_level=current_bar['High'],
                        reversal_level=current_bar['Close'],
                        strength=min((reversal_size + breakout_size) * 50, 1.0),
                        volume_surge=ManipulationDetector._check_volume_surge(recent_data, i),
                        session=session
                    ))
            
            # Downside fake breakout
            elif (current_bar['Low'] < prev_bar['Low'] and
                  prev_bar['Low'] < prev_prev_bar['Low'] and
                  current_bar['Close'] > prev_bar['Open']):
                
                breakout_size = (prev_prev_bar['Low'] - current_bar['Low']) / prev_prev_bar['Low']
                reversal_size = (current_bar['Close'] - current_bar['Low']) / current_bar['Low']
                
                if breakout_size >= 0.003 and reversal_size >= 0.004:
                    patterns.append(ManipulationPattern(
                        pattern_type='fake_breakout',
                        start_time=pd.to_datetime(str(prev_prev_bar.name)) if not isinstance(prev_prev_bar.name, datetime) else prev_prev_bar.name,
                        end_time=pd.to_datetime(str(current_bar.name)) if not isinstance(current_bar.name, datetime) else current_bar.name,
                        trigger_level=current_bar['Low'],
                        reversal_level=current_bar['Close'],
                        strength=min((reversal_size + breakout_size) * 50, 1.0),
                        volume_surge=ManipulationDetector._check_volume_surge(recent_data, i),
                        session=session
                    ))
        
        return patterns
    
    @staticmethod
    def _detect_stop_hunts(df: pd.DataFrame, session: TradingSession) -> List[ManipulationPattern]:
        """Detect stop hunting patterns (spikes followed by reversals)"""
        patterns = []
        recent_data = df.tail(10)
        
        for i in range(1, len(recent_data)):
            current_bar = recent_data.iloc[i]
            prev_bar = recent_data.iloc[i-1]
            
            # Calculate wick sizes relative to body
            body_size = abs(current_bar['Close'] - current_bar['Open'])
            upper_wick = current_bar['High'] - max(current_bar['Open'], current_bar['Close'])
            lower_wick = min(current_bar['Open'], current_bar['Close']) - current_bar['Low']
            
            # Look for long wicks (>3x body size) indicating rejections
            if body_size > 0:  # Avoid division by zero
                upper_wick_ratio = upper_wick / body_size
                lower_wick_ratio = lower_wick / body_size
                
                # Upper wick rejection (bearish)
                if (upper_wick_ratio >= 3.0 and 
                    current_bar['Close'] < current_bar['Open'] and
                    upper_wick / current_bar['High'] >= 0.4):  # Wick is 40%+ of high
                    
                    patterns.append(ManipulationPattern(
                        pattern_type='stop_hunt',
                        start_time=pd.to_datetime(str(current_bar.name)) if not isinstance(current_bar.name, datetime) else current_bar.name,
                        end_time=pd.to_datetime(str(current_bar.name)) if not isinstance(current_bar.name, datetime) else current_bar.name,
                        trigger_level=current_bar['High'],
                        reversal_level=current_bar['Close'],
                        strength=min(upper_wick_ratio * 0.2, 1.0),
                        volume_surge=ManipulationDetector._check_volume_surge(recent_data, i),
                        session=session
                    ))
                
                # Lower wick rejection (bullish)
                elif (lower_wick_ratio >= 3.0 and 
                      current_bar['Close'] > current_bar['Open'] and
                      lower_wick / current_bar['Low'] >= 0.4):
                    
                    patterns.append(ManipulationPattern(
                        pattern_type='stop_hunt',
                        start_time=pd.to_datetime(str(current_bar.name)) if not isinstance(current_bar.name, datetime) else current_bar.name,
                        end_time=pd.to_datetime(str(current_bar.name)) if not isinstance(current_bar.name, datetime) else current_bar.name,
                        trigger_level=current_bar['Low'],
                        reversal_level=current_bar['Close'],
                        strength=min(lower_wick_ratio * 0.2, 1.0),
                        volume_surge=ManipulationDetector._check_volume_surge(recent_data, i),
                        session=session
                    ))
        
        return patterns
    
    @staticmethod
    def _check_volume_surge(df: pd.DataFrame, index: int) -> bool:
        """Check if there's a volume surge at the given index"""
        if 'Volume' not in df.columns or len(df) < 5:
            return False
        
        current_volume = df['Volume'].iloc[index]
        avg_volume = df['Volume'].tail(10).mean()
        
        return current_volume > avg_volume * 1.5  # 50% above average