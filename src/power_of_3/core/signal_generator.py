# =============================================================================
# SIGNAL GENERATOR
# =============================================================================

from asyncio.log import logger
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import pytz

#from power_of_3_signal_generator import PowerOf3Signal
from src.power_of_3.core.types import LiquidityZone, ManipulationPattern, PowerOf3Phase, PowerOf3Signal, SignalQuality, SignalType, TradingSession
from src.power_of_3.core.liquidity_zones import LiquidityZoneDetector  # Import LiquidityAnalyzer
from src.power_of_3.core.session_detector import SessionDetector
from src.power_of_3.core.manipulation_detector import ManipulationDetector

class PowerOf3SignalGenerator:
    """Main Power of 3 signal generation engine"""
    
    def __init__(self, min_risk_reward: float = 5.0, max_risk_percent: float = 2.0):
        self.min_risk_reward = min_risk_reward
        self.max_risk_percent = max_risk_percent
        self.session_manager = SessionDetector()
        self.liquidity_analyzer = LiquidityZoneDetector()
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
        liquidity_zones_raw = self.liquidity_analyzer.identify_liquidity_zones(
            df
        )
        liquidity_zones = [LiquidityZone(**zone) for zone in liquidity_zones_raw]
        
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
                                quality_level=SignalQuality(SignalQuality.GOOD if quality_score >= 7 else SignalQuality.AVERAGE),
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