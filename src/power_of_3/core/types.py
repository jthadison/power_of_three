from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

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