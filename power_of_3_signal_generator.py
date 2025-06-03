"""
Power of 3 Signal Generator - Advanced ICT Implementation
========================================================

This module implements the core Power of 3 methodology for generating
high-probability trading signals with institutional flow analysis.

Key Components:
1. Session-based analysis (London Open, NY Open, London Close)
2. Three-phase market structure (Accumulation → Manipulation → Direction)
3. Liquidity zone analysis and manipulation detection
4. Signal generation with 1:5+ risk-reward ratios
5. Signal quality scoring and filtering
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# POWER OF 3 CORE DEFINITIONS
# =============================================================================

class PowerOf3Phase(Enum):
    """Three phases of Power of 3 methodology"""
    ACCUMULATION = "accumulation"      # Smart money builds positions quietly
    MANIPULATION = "manipulation"      # Price moves to trigger retail stops
    DIRECTION = "direction"            # True institutional move begins

class SignalType(Enum):
    """Types of Power of 3 signals"""
    LONG = "long"
    SHORT = "short"
    NO_SIGNAL = "no_signal"

class SignalQuality(Enum):
    """Signal quality levels"""
    EXCELLENT = "excellent"    # 9-10 score
    GOOD = "good"             # 7-8 score
    AVERAGE = "average"       # 5-6 score
    POOR = "poor"            # 1-4 score

class TradingSession(Enum):
    """Power of 3 trading sessions"""
    LONDON_OPEN = "london_open"        # 2:00-5:00 AM EST
    NEW_YORK_OPEN = "new_york_open"    # 7:00-10:00 AM EST
    LONDON_CLOSE = "london_close"      # 10:00 AM-12:00 PM EST

@dataclass
class LiquidityZone:
    """Represents a liquidity zone where stops are clustered"""
    level: float
    zone_type: str  # 'resistance', 'support'
    strength: float  # 0-1, based on number of touches and volume
    session_created: TradingSession
    touches: int
    last_test: datetime
    broken: bool = False

@dataclass
class ManipulationPattern:
    """Detected manipulation pattern"""
    pattern_type: str  # 'liquidity_sweep', 'fake_breakout', 'stop_hunt'
    start_time: datetime
    end_time: datetime
    trigger_level: float
    reversal_level: float
    strength: float  # 0-1 confidence
    volume_surge: bool
    session: TradingSession

@dataclass
class PowerOf3Signal:
    """Complete Power of 3 trading signal"""
    symbol: str
    signal_type: SignalType
    session: TradingSession
    phase: PowerOf3Phase
    
    # Entry details
    entry_price: float
    entry_time: datetime
    entry_reasoning: str
    
    # Risk management
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    risk_reward_ratio: float
    position_size: float
    risk_amount: float
    
    # Signal quality
    quality_score: float  # 0-10
    quality_level: SignalQuality
    confidence: float  # 0-1
    
    # Supporting analysis
    manipulation_detected: Optional[ManipulationPattern]
    liquidity_zones: List[LiquidityZone]
    market_structure: str  # 'bullish', 'bearish', 'neutral'
    
    # Metadata
    signal_id: str
    generated_at: datetime
    expires_at: datetime
    
    def to_dict(self) -> Dict:
        """Convert signal to dictionary for logging/storage"""
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'type': self.signal_type.value,
            'session': self.session.value,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit_1': self.take_profit_1,
            'risk_reward': self.risk_reward_ratio,
            'quality_score': self.quality_score,
            'confidence': self.confidence,
            'generated_at': self.generated_at.isoformat(),
            'expires_at': self.expires_at.isoformat()
        }

# =============================================================================
# SESSION MANAGER
# =============================================================================

class SessionManager:
    """Manages Power of 3 trading sessions and timing"""
    
    @staticmethod
    def get_current_session() -> Optional[TradingSession]:
        """Get current Power of 3 session based on EST time"""
        now = datetime.now(pytz.timezone('US/Eastern'))
        hour = now.hour
        minute = now.minute
        current_time = hour + minute / 60.0
        
        # London Open: 2:00-5:00 AM EST
        if 2.0 <= current_time < 5.0:
            return TradingSession.LONDON_OPEN
        
        # New York Open: 7:00-10:00 AM EST
        elif 7.0 <= current_time < 10.0:
            return TradingSession.NEW_YORK_OPEN
        
        # London Close: 10:00 AM-12:00 PM EST
        elif 10.0 <= current_time < 12.0:
            return TradingSession.LONDON_CLOSE
        
        return None
    
    @staticmethod
    def get_session_phase(session_start: datetime) -> PowerOf3Phase:
        """Determine which phase of Power of 3 we're in"""
        now = datetime.now(pytz.timezone('US/Eastern'))
        session_duration = (now - session_start).total_seconds() / 60  # minutes
        
        # First 60 minutes: Accumulation phase
        if session_duration <= 60:
            return PowerOf3Phase.ACCUMULATION
        
        # Next 60 minutes: Manipulation phase
        elif session_duration <= 120:
            return PowerOf3Phase.MANIPULATION
        
        # Remaining time: Direction phase
        else:
            return PowerOf3Phase.DIRECTION
    
    @staticmethod
    def is_high_impact_time() -> bool:
        """Check if we're near high-impact news times"""
        now = datetime.now(pytz.timezone('US/Eastern'))
        
        # Avoid signals 30 minutes before/after major news times
        high_impact_times = [
            (8, 30),   # US Core CPI, Non-Farm Payrolls
            (10, 0),   # US FOMC announcements
            (14, 0),   # FOMC press conferences
        ]
        
        for hour, minute in high_impact_times:
            news_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            time_diff = abs((now - news_time).total_seconds() / 60)
            
            if time_diff <= 30:  # Within 30 minutes
                return True
        
        return False

# =============================================================================
# LIQUIDITY ANALYSIS
# =============================================================================

class LiquidityAnalyzer:
    """Analyzes liquidity zones and stop clustering"""
    
    @staticmethod
    def identify_liquidity_zones(df: pd.DataFrame, session: TradingSession, 
                                lookback_periods: int = 50) -> List[LiquidityZone]:
        """Identify key liquidity zones where retail stops are clustered"""
        zones = []
        
        if len(df) < lookback_periods:
            return zones
        
        recent_data = df.tail(lookback_periods)
        
        # 1. Previous session high/low
        session_high = recent_data['High'].max()
        session_low = recent_data['Low'].min()
        
        zones.extend([
            LiquidityZone(
                level=session_high,
                zone_type='resistance',
                strength=0.8,
                session_created=session,
                touches=1,
                last_test=recent_data.index[-1]
            ),
            LiquidityZone(
                level=session_low,
                zone_type='support',
                strength=0.8,
                session_created=session,
                touches=1,
                last_test=recent_data.index[-1]
            )
        ])
        
        # 2. Swing highs and lows (areas where price reversed)
        swing_highs = LiquidityAnalyzer._find_swing_points(recent_data, 'high')
        swing_lows = LiquidityAnalyzer._find_swing_points(recent_data, 'low')
        
        for swing_high in swing_highs:
            strength = LiquidityAnalyzer._calculate_zone_strength(
                recent_data, swing_high['price'], 'resistance'
            )
            if strength >= 0.3:  # Only include significant levels
                zones.append(LiquidityZone(
                    level=swing_high['price'],
                    zone_type='resistance',
                    strength=strength,
                    session_created=session,
                    touches=swing_high['touches'],
                    last_test=swing_high['timestamp']
                ))
        
        for swing_low in swing_lows:
            strength = LiquidityAnalyzer._calculate_zone_strength(
                recent_data, swing_low['price'], 'support'
            )
            if strength >= 0.3:
                zones.append(LiquidityZone(
                    level=swing_low['price'],
                    zone_type='support',
                    strength=strength,
                    session_created=session,
                    touches=swing_low['touches'],
                    last_test=swing_low['timestamp']
                ))
        
        # 3. Psychological levels (round numbers)
        current_price = recent_data['Close'].iloc[-1]
        psychological_levels = LiquidityAnalyzer._find_psychological_levels(current_price)
        
        for level in psychological_levels:
            if recent_data['Low'].min() <= level <= recent_data['High'].max():
                zone_type = 'resistance' if level > current_price else 'support'
                strength = 0.4  # Moderate strength for psychological levels
                
                zones.append(LiquidityZone(
                    level=level,
                    zone_type=zone_type,
                    strength=strength,
                    session_created=session,
                    touches=0,
                    last_test=recent_data.index[-1]
                ))
        
        # Sort by strength and return top zones
        zones.sort(key=lambda x: x.strength, reverse=True)
        return zones[:10]  # Return top 10 zones
    
    @staticmethod
    def _find_swing_points(df: pd.DataFrame, point_type: str, 
                          window: int = 5) -> List[Dict]:
        """Find swing highs or lows"""
        points = []
        
        if point_type == 'high':
            column = 'High'
            condition = lambda i: all(df[column].iloc[i] >= df[column].iloc[i-j] 
                                    for j in range(1, window+1)) and \
                                  all(df[column].iloc[i] >= df[column].iloc[i+j] 
                                    for j in range(1, min(window+1, len(df)-i)))
        else:
            column = 'Low'
            condition = lambda i: all(df[column].iloc[i] <= df[column].iloc[i-j] 
                                    for j in range(1, window+1)) and \
                                  all(df[column].iloc[i] <= df[column].iloc[i+j] 
                                    for j in range(1, min(window+1, len(df)-i)))
        
        for i in range(window, len(df) - window):
            if condition(i):
                # Count how many times this level was tested
                level = df[column].iloc[i]
                touches = LiquidityAnalyzer._count_level_touches(df, level, tolerance=0.001)
                
                points.append({
                    'price': level,
                    'timestamp': df.index[i],
                    'touches': touches
                })
        
        return points
    
    @staticmethod
    def _count_level_touches(df: pd.DataFrame, level: float, tolerance: float = 0.001) -> int:
        """Count how many times price touched a level"""
        touches = 0
        for _, row in df.iterrows():
            if abs(row['High'] - level) / level <= tolerance or \
               abs(row['Low'] - level) / level <= tolerance:
                touches += 1
        return touches
    
    @staticmethod
    def _calculate_zone_strength(df: pd.DataFrame, level: float, zone_type: str) -> float:
        """Calculate strength of a liquidity zone"""
        touches = LiquidityAnalyzer._count_level_touches(df, level)
        
        # Base strength from number of touches
        touch_strength = min(touches * 0.2, 1.0)
        
        # Volume analysis (if available)
        volume_strength = 0.5
        if 'Volume' in df.columns:
            # Find bars near this level
            tolerance = 0.002  # 0.2%
            near_level = df[
                (abs(df['High'] - level) / level <= tolerance) |
                (abs(df['Low'] - level) / level <= tolerance)
            ]
            
            if len(near_level) > 0:
                avg_volume = df['Volume'].mean()
                level_volume = near_level['Volume'].mean()
                volume_strength = min(level_volume / avg_volume, 2.0) * 0.5
        
        # Recency factor (more recent = stronger)
        current_price = df['Close'].iloc[-1]
        distance = abs(current_price - level) / current_price
        recency_strength = max(0.2, 1.0 - distance * 2)  # Closer = stronger
        
        # Combine factors
        total_strength = (touch_strength * 0.4 + volume_strength * 0.3 + recency_strength * 0.3)
        return min(total_strength, 1.0)
    
    @staticmethod
    def _find_psychological_levels(price: float) -> List[float]:
        """Find psychological levels (round numbers) near current price"""
        levels = []
        
        # Determine the appropriate round number intervals based on price
        if price >= 10000:
            intervals = [100, 250, 500, 1000]
        elif price >= 1000:
            intervals = [10, 25, 50, 100]
        elif price >= 100:
            intervals = [1, 2.5, 5, 10]
        else:
            intervals = [0.1, 0.25, 0.5, 1.0]
        
        for interval in intervals:
            # Find nearest round numbers above and below current price
            lower = int(price / interval) * interval
            upper = lower + interval
            
            # Include levels within reasonable range (±5%)
            if abs(lower - price) / price <= 0.05:
                levels.append(lower)
            if abs(upper - price) / price <= 0.05:
                levels.append(upper)
        
        return sorted(list(set(levels)))

# =============================================================================
# MANIPULATION DETECTOR
# =============================================================================

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

# =============================================================================
# SIGNAL GENERATOR
# =============================================================================

class PowerOf3SignalGenerator:
    """Main Power of 3 signal generation engine"""
    
    def __init__(self, min_risk_reward: float = 5.0, max_risk_percent: float = 2.0):
        self.min_risk_reward = min_risk_reward
        self.max_risk_percent = max_risk_percent
        self.session_manager = SessionManager()
        self.liquidity_analyzer = LiquidityAnalyzer()
        self.manipulation_detector = ManipulationDetector()
    
    def generate_signals(self, symbol: str, df: pd.DataFrame, 
                        account_balance: float = 10000.0) -> List[PowerOf3Signal]:
        """Generate Power of 3 trading signals"""
        signals = []
        
        # Check if we're in a Power of 3 session
        current_session = self.session_manager.get_current_session()
        if not current_session:
            logger.info("Not in Power of 3 session - no signals generated")
            return signals
        
        # Avoid high-impact news times
        if self.session_manager.is_high_impact_time():
            logger.info("High-impact news time detected - avoiding signals")
            return signals
        
        # Ensure we have enough data
        if len(df) < 100:
            logger.warning(f"Insufficient data for {symbol} - need at least 100 bars")
            return signals
        
        # Step 1: Identify liquidity zones
        liquidity_zones = self.liquidity_analyzer.identify_liquidity_zones(
            df, current_session
        )
        
        # Step 2: Detect manipulation patterns
        manipulation_patterns = self.manipulation_detector.detect_manipulation_patterns(
            df, current_session, liquidity_zones
        )
        
        # Step 3: Analyze market structure
        market_structure = self._analyze_market_structure(df)
        
        # Step 4: Generate signals based on Power of 3 methodology
        for pattern in manipulation_patterns:
            signal = self._create_signal_from_pattern(
                symbol, df, pattern, liquidity_zones, market_structure, 
                current_session, account_balance
            )
            
            if signal and signal.quality_score >= 5.0:  # Only include decent signals
                signals.append(signal)
        
        # Step 5: Look for continuation signals if no manipulation detected
        if not manipulation_patterns:
            continuation_signals = self._generate_continuation_signals(
                symbol, df, liquidity_zones, market_structure, 
                current_session, account_balance
            )
            signals.extend(continuation_signals)
        
        # Sort by quality score
        signals.sort(key=lambda x: x.quality_score, reverse=True)
        
        return signals[:3]  # Return top 3 signals maximum
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> str:
        """Analyze overall market structure"""
        if len(df) < 50:
            return 'neutral'
        
        recent_data = df.tail(50)
        
        # Simple trend analysis using swing highs/lows
        highs = recent_data['High'].rolling(5).max()
        lows = recent_data['Low'].rolling(5).min()
        
        recent_highs = highs.tail(10)
        recent_lows = lows.tail(10)
        
        # Check for higher highs and higher lows (bullish)
        higher_highs = recent_highs.iloc[-1] > recent_highs.iloc[-5]
        higher_lows = recent_lows.iloc[-1] > recent_lows.iloc[-5]
        
        # Check for lower highs and lower lows (bearish)
        lower_highs = recent_highs.iloc[-1] < recent_highs.iloc[-5]
        lower_lows = recent_lows.iloc[-1] < recent_lows.iloc[-5]
        
        if higher_highs and higher_lows:
            return 'bullish'
        elif lower_highs and lower_lows:
            return 'bearish'
        else:
            return 'neutral'
    
    def _create_signal_from_pattern(self, symbol: str, df: pd.DataFrame,
                                  pattern: ManipulationPattern,
                                  liquidity_zones: List[LiquidityZone],
                                  market_structure: str,
                                  session: TradingSession,
                                  account_balance: float) -> Optional[PowerOf3Signal]:
        """Create a trading signal from a manipulation pattern"""
        
        current_price = df['Close'].iloc[-1]
        current_time = df.index[-1]
        
        # Determine signal direction based on pattern and market structure
        if pattern.pattern_type == 'liquidity_sweep':
            if pattern.reversal_level > pattern.trigger_level:
                signal_type = SignalType.LONG
                entry_price = current_price
                stop_loss = pattern.trigger_level - (pattern.trigger_level * 0.001)  # Just below sweep level
            else:
                signal_type = SignalType.SHORT
                entry_price = current_price
                stop_loss = pattern.trigger_level + (pattern.trigger_level * 0.001)  # Just above sweep level
                
        elif pattern.pattern_type == 'fake_breakout':
            if pattern.reversal_level > pattern.trigger_level:
                signal_type = SignalType.LONG
                entry_price = current_price
                stop_loss = pattern.trigger_level - (pattern.trigger_level * 0.002)
            else:
                signal_type = SignalType.SHORT
                entry_price = current_price
                stop_loss = pattern.trigger_level + (pattern.trigger_level * 0.002)
                
        elif pattern.pattern_type == 'stop_hunt':
            if pattern.reversal_level > pattern.trigger_level:
                signal_type = SignalType.LONG
                entry_price = current_price
                stop_loss = pattern.trigger_level - (pattern.trigger_level * 0.0015)
            else:
                signal_type = SignalType.SHORT
                entry_price = current_price
                stop_loss = pattern.trigger_level + (pattern.trigger_level * 0.0015)
        else:
            return None
        
        # Calculate risk amount
        risk_per_trade = account_balance * (self.max_risk_percent / 100)
        risk_distance = abs(entry_price - stop_loss)
        
        if risk_distance == 0:
            return None
        
        # Calculate targets using liquidity zones and fibonacci levels
        targets = self._calculate_profit_targets(
            entry_price, stop_loss, signal_type, liquidity_zones, risk_distance
        )
        
        if not targets or len(targets) < 3:
            return None
        
        # Calculate risk-reward ratio
        primary_target_distance = abs(targets[0] - entry_price)
        risk_reward_ratio = primary_target_distance / risk_distance
        
        # Only proceed if risk-reward meets minimum requirement
        if risk_reward_ratio < self.min_risk_reward:
            return None
        
        # Calculate position size
        position_size = risk_per_trade / risk_distance
        
        # Calculate signal quality score
        quality_score = self._calculate_quality_score(
            pattern, market_structure, signal_type, risk_reward_ratio,
            session, len(liquidity_zones)
        )
        
        # Determine quality level
        if quality_score >= 9:
            quality_level = SignalQuality.EXCELLENT
        elif quality_score >= 7:
            quality_level = SignalQuality.GOOD
        elif quality_score >= 5:
            quality_level = SignalQuality.AVERAGE
        else:
            quality_level = SignalQuality.POOR
        
        # Generate unique signal ID
        signal_id = f"{symbol}_{session.value}_{int(current_time.timestamp())}"
        
        # Create signal
        signal = PowerOf3Signal(
            symbol=symbol,
            signal_type=signal_type,
            session=session,
            phase=PowerOf3Phase.DIRECTION,  # Signals generated in direction phase
            
            entry_price=entry_price,
            entry_time=current_time,
            entry_reasoning=f"{pattern.pattern_type} detected with {pattern.strength:.2f} strength",
            
            stop_loss=stop_loss,
            take_profit_1=targets[0],
            take_profit_2=targets[1],
            take_profit_3=targets[2],
            risk_reward_ratio=risk_reward_ratio,
            position_size=position_size,
            risk_amount=risk_per_trade,
            
            quality_score=quality_score,
            quality_level=quality_level,
            confidence=pattern.strength,
            
            manipulation_detected=pattern,
            liquidity_zones=liquidity_zones,
            market_structure=market_structure,
            
            signal_id=signal_id,
            generated_at=datetime.now(pytz.timezone('US/Eastern')),
            expires_at=datetime.now(pytz.timezone('US/Eastern')) + timedelta(hours=2)
        )
        
        return signal
    
    def _calculate_profit_targets(self, entry_price: float, stop_loss: float,
                                signal_type: SignalType, liquidity_zones: List[LiquidityZone],
                                risk_distance: float) -> List[float]:
        """Calculate profit targets using liquidity zones and fibonacci ratios"""
        targets = []
        
        # Fibonacci extension ratios for targets
        fib_ratios = [1.618, 2.618, 4.236]  # For 1:1.6, 1:2.6, 1:4.2 RR
        
        if signal_type == SignalType.LONG:
            # Base targets on fibonacci extensions
            for ratio in fib_ratios:
                target = entry_price + (risk_distance * ratio)
                targets.append(target)
            
            # Adjust targets based on liquidity zones (resistance levels)
            relevant_zones = [z for z in liquidity_zones 
                            if z.zone_type == 'resistance' and z.level > entry_price]
            
            if relevant_zones:
                # Sort by distance from entry
                relevant_zones.sort(key=lambda x: abs(x.level - entry_price))
                
                # Use nearby resistance levels as targets if they're close to fibonacci levels
                for i, zone in enumerate(relevant_zones[:3]):
                    if i < len(targets):
                        target_distance = abs(targets[i] - entry_price)
                        zone_distance = abs(zone.level - entry_price)
                        
                        # If zone is within 20% of fibonacci target, use zone level
                        if abs(target_distance - zone_distance) / target_distance <= 0.2:
                            targets[i] = zone.level - (zone.level * 0.001)  # Just before resistance
        
        else:  # SHORT signal
            for ratio in fib_ratios:
                target = entry_price - (risk_distance * ratio)
                targets.append(target)
            
            # Adjust for support levels
            relevant_zones = [z for z in liquidity_zones 
                            if z.zone_type == 'support' and z.level < entry_price]
            
            if relevant_zones:
                relevant_zones.sort(key=lambda x: abs(x.level - entry_price))
                
                for i, zone in enumerate(relevant_zones[:3]):
                    if i < len(targets):
                        target_distance = abs(targets[i] - entry_price)
                        zone_distance = abs(zone.level - entry_price)
                        
                        if abs(target_distance - zone_distance) / target_distance <= 0.2:
                            targets[i] = zone.level + (zone.level * 0.001)  # Just above support
        
        return targets
    
    def _calculate_quality_score(self, pattern: ManipulationPattern, market_structure: str,
                               signal_type: SignalType, risk_reward: float,
                               session: TradingSession, num_liquidity_zones: int) -> float:
        """Calculate signal quality score (0-10)"""
        score = 0.0
        
        # 1. Pattern strength (0-2 points)
        score += pattern.strength * 2.0
        
        # 2. Market structure alignment (0-2 points)
        if ((signal_type == SignalType.LONG and market_structure == 'bullish') or
            (signal_type == SignalType.SHORT and market_structure == 'bearish')):
            score += 2.0
        elif market_structure == 'neutral':
            score += 1.0
        
        # 3. Risk-reward ratio (0-2 points)
        if risk_reward >= 5.0:
            score += 2.0
        elif risk_reward >= 3.0:
            score += 1.5
        elif risk_reward >= 2.0:
            score += 1.0
        else:
            score += 0.5
        
        # 4. Session timing (0-2 points)
        if session == TradingSession.NEW_YORK_OPEN:
            score += 2.0  # Best session
        elif session == TradingSession.LONDON_OPEN:
            score += 1.5
        else:
            score += 1.0
        
        # 5. Volume surge (0-1 point)
        if pattern.volume_surge:
            score += 1.0
        else:
            score += 0.5
        
        # 6. Liquidity zone density (0-1 point)
        if num_liquidity_zones >= 5:
            score += 1.0
        elif num_liquidity_zones >= 3:
            score += 0.7
        else:
            score += 0.3
        
        return min(score, 10.0)
    
    def _generate_continuation_signals(self, symbol: str, df: pd.DataFrame,
                                     liquidity_zones: List[LiquidityZone],
                                     market_structure: str,
                                     session: TradingSession,
                                     account_balance: float) -> List[PowerOf3Signal]:
        """Generate continuation signals when no manipulation is detected"""
        signals = []
        
        # Only generate continuation signals in strong trending markets
        if market_structure == 'neutral':
            return signals
        
        current_price = df['Close'].iloc[-1]
        current_time = df.index[-1]
        
        # Look for pullbacks to key levels in trending markets
        if market_structure == 'bullish':
            # Look for pullbacks to support levels
            support_zones = [z for z in liquidity_zones 
                           if z.zone_type == 'support' and z.level < current_price]
            
            if support_zones:
                # Find the strongest support near current price
                nearest_support = min(support_zones, 
                                    key=lambda x: abs(x.level - current_price))
                
                # Check if we're near this support level (within 1%)
                if abs(current_price - nearest_support.level) / current_price <= 0.01:
                    
                    entry_price = current_price
                    stop_loss = nearest_support.level - (nearest_support.level * 0.002)
                    risk_distance = abs(entry_price - stop_loss)
                    
                    # Calculate targets
                    targets = self._calculate_profit_targets(
                        entry_price, stop_loss, SignalType.LONG, 
                        liquidity_zones, risk_distance
                    )
                    
                    if targets and len(targets) >= 3:
                        rr_ratio = abs(targets[0] - entry_price) / risk_distance
                        
                        if rr_ratio >= self.min_risk_reward:
                            quality_score = 6.0 + (nearest_support.strength * 2.0)  # Base score for continuation
                            
                            signal = PowerOf3Signal(
                                symbol=symbol,
                                signal_type=SignalType.LONG,
                                session=session,
                                phase=PowerOf3Phase.DIRECTION,
                                
                                entry_price=entry_price,
                                entry_time=current_time,
                                entry_reasoning=f"Bullish continuation at support level {nearest_support.level:.2f}",
                                
                                stop_loss=stop_loss,
                                take_profit_1=targets[0],
                                take_profit_2=targets[1],
                                take_profit_3=targets[2],
                                risk_reward_ratio=rr_ratio,
                                position_size=(account_balance * self.max_risk_percent / 100) / risk_distance,
                                risk_amount=account_balance * self.max_risk_percent / 100,
                                
                                quality_score=quality_score,
                                quality_level=SignalQuality.GOOD if quality_score >= 7 else SignalQuality.AVERAGE,
                                confidence=nearest_support.strength,
                                
                                manipulation_detected=None,
                                liquidity_zones=liquidity_zones,
                                market_structure=market_structure,
                                
                                signal_id=f"{symbol}_continuation_{int(current_time.timestamp())}",
                                generated_at=datetime.now(pytz.timezone('US/Eastern')),
                                expires_at=datetime.now(pytz.timezone('US/Eastern')) + timedelta(hours=1)
                            )
                            
                            signals.append(signal)
        
        elif market_structure == 'bearish':
            # Similar logic for bearish continuation signals
            resistance_zones = [z for z in liquidity_zones 
                              if z.zone_type == 'resistance' and z.level > current_price]
            
            if resistance_zones:
                nearest_resistance = min(resistance_zones, 
                                       key=lambda x: abs(x.level - current_price))
                
                if abs(current_price - nearest_resistance.level) / current_price <= 0.01:
                    
                    entry_price = current_price
                    stop_loss = nearest_resistance.level + (nearest_resistance.level * 0.002)
                    risk_distance = abs(entry_price - stop_loss)
                    
                    targets = self._calculate_profit_targets(
                        entry_price, stop_loss, SignalType.SHORT, 
                        liquidity_zones, risk_distance
                    )
                    
                    if targets and len(targets) >= 3:
                        rr_ratio = abs(targets[0] - entry_price) / risk_distance
                        
                        if rr_ratio >= self.min_risk_reward:
                            quality_score = 6.0 + (nearest_resistance.strength * 2.0)
                            
                            signal = PowerOf3Signal(
                                symbol=symbol,
                                signal_type=SignalType.SHORT,
                                session=session,
                                phase=PowerOf3Phase.DIRECTION,
                                
                                entry_price=entry_price,
                                entry_time=current_time,
                                entry_reasoning=f"Bearish continuation at resistance level {nearest_resistance.level:.2f}",
                                
                                stop_loss=stop_loss,
                                take_profit_1=targets[0],
                                take_profit_2=targets[1],
                                take_profit_3=targets[2],
                                risk_reward_ratio=rr_ratio,
                                position_size=(account_balance * self.max_risk_percent / 100) / risk_distance,
                                risk_amount=account_balance * self.max_risk_percent / 100,
                                
                                quality_score=quality_score,
                                quality_level=SignalQuality.GOOD if quality_score >= 7 else SignalQuality.AVERAGE,
                                confidence=nearest_resistance.strength,
                                
                                manipulation_detected=None,
                                liquidity_zones=liquidity_zones,
                                market_structure=market_structure,
                                
                                signal_id=f"{symbol}_continuation_{int(current_time.timestamp())}",
                                generated_at=datetime.now(pytz.timezone('US/Eastern')),
                                expires_at=datetime.now(pytz.timezone('US/Eastern')) + timedelta(hours=1)
                            )
                            
                            signals.append(signal)
        
        return signals

# =============================================================================
# SIGNAL FORMATTER
# =============================================================================

class SignalFormatter:
    """Formats signals for display and logging"""
    
    @staticmethod
    def format_signal_for_display(signal: PowerOf3Signal) -> str:
        """Format signal for human-readable display"""
        
        direction_emoji = "🟢" if signal.signal_type == SignalType.LONG else "🔴"
        quality_emoji = {
            SignalQuality.EXCELLENT: "⭐⭐⭐",
            SignalQuality.GOOD: "⭐⭐",
            SignalQuality.AVERAGE: "⭐",
            SignalQuality.POOR: "⚠️"
        }
        
        return f"""
{direction_emoji} **{signal.signal_type.value.upper()} SIGNAL** {quality_emoji[signal.quality_level]}

**Symbol:** {signal.symbol}
**Session:** {signal.session.value.replace('_', ' ').title()}
**Quality Score:** {signal.quality_score:.1f}/10
**Confidence:** {signal.confidence:.0%}

**ENTRY DETAILS:**
📍 Entry Price: {signal.entry_price:.2f}
🛑 Stop Loss: {signal.stop_loss:.2f}
🎯 Target 1: {signal.take_profit_1:.2f}
🎯 Target 2: {signal.take_profit_2:.2f}
🎯 Target 3: {signal.take_profit_3:.2f}

**RISK MANAGEMENT:**
📊 Risk/Reward: 1:{signal.risk_reward_ratio:.1f}
💰 Position Size: {signal.position_size:.4f}
📉 Risk Amount: ${signal.risk_amount:.2f}

**ANALYSIS:**
📈 Market Structure: {signal.market_structure.title()}
🎭 Manipulation: {signal.manipulation_detected.pattern_type if signal.manipulation_detected else 'None detected'}
💡 Reasoning: {signal.entry_reasoning}

**TIMING:**
⏰ Generated: {signal.generated_at.strftime('%Y-%m-%d %H:%M:%S EST')}
⏳ Expires: {signal.expires_at.strftime('%Y-%m-%d %H:%M:%S EST')}

**Signal ID:** {signal.signal_id}
        """.strip()
    
    @staticmethod
    def format_signal_summary(signals: List[PowerOf3Signal]) -> str:
        """Format a summary of multiple signals"""
        if not signals:
            return "🔍 **No Power of 3 signals detected at this time.**\n\nWaiting for manipulation patterns during active sessions..."
        
        summary = f"📊 **Power of 3 Signal Summary** ({len(signals)} signals)\n\n"
        
        for i, signal in enumerate(signals, 1):
            direction = "🟢 LONG" if signal.signal_type == SignalType.LONG else "🔴 SHORT"
            summary += f"{i}. **{signal.symbol}** {direction} - Quality: {signal.quality_score:.1f}/10 - RR: 1:{signal.risk_reward_ratio:.1f}\n"
        
        return summary

# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

def test_signal_generator():
    """Test the Power of 3 signal generator with sample data"""
    
    # Create sample data that might show manipulation patterns
    dates = pd.date_range('2024-01-01 09:00:00', periods=100, freq='5min')
    
    # Create realistic price data with manipulation patterns
    np.random.seed(42)
    base_price = 2100.0
    prices = []
    
    for i in range(100):
        # Add some trend and noise
        trend = i * 0.1
        noise = np.random.normal(0, 2)
        
        # Add manipulation pattern around bar 50 (liquidity sweep)
        if i == 50:
            price = base_price + trend + 15  # Spike up
        elif i == 51:
            price = base_price + trend - 5   # Reverse down
        else:
            price = base_price + trend + noise
        
        prices.append(price)
    
    # Create OHLC data
    df_data = []
    for i, price in enumerate(prices):
        open_price = price + np.random.uniform(-1, 1)
        high_price = max(open_price, price) + abs(np.random.uniform(0, 2))
        low_price = min(open_price, price) - abs(np.random.uniform(0, 2))
        close_price = price
        volume = np.random.randint(1000, 5000)
        
        df_data.append({
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume
        })
    
    df = pd.DataFrame(df_data, index=dates)
    
    # Test signal generation
    signal_generator = PowerOf3SignalGenerator(min_risk_reward=3.0)
    signals = signal_generator.generate_signals('US30', df, account_balance=10000)
    
    print("=== POWER OF 3 SIGNAL GENERATOR TEST ===\n")
    
    if signals:
        print(f"Generated {len(signals)} signals:\n")
        for signal in signals:
            print(SignalFormatter.format_signal_for_display(signal))
            print("\n" + "="*60 + "\n")
    else:
        print("No signals generated (expected if not in Power of 3 session)")
        
        # Test liquidity analysis separately
        liquidity_analyzer = LiquidityAnalyzer()
        zones = liquidity_analyzer.identify_liquidity_zones(df, TradingSession.NEW_YORK_OPEN)
        print(f"\nLiquidity zones found: {len(zones)}")
        for zone in zones[:3]:
            print(f"- {zone.zone_type.title()}: {zone.level:.2f} (strength: {zone.strength:.2f})")
        
        # Test manipulation detection
        manipulation_detector = ManipulationDetector()
        patterns = manipulation_detector.detect_manipulation_patterns(
            df, TradingSession.NEW_YORK_OPEN, zones
        )
        print(f"\nManipulation patterns found: {len(patterns)}")
        for pattern in patterns:
            print(f"- {pattern.pattern_type}: {pattern.trigger_level:.2f} -> {pattern.reversal_level:.2f}")

if __name__ == "__main__":
    test_signal_generator()