# =============================================================================
# SIGNAL FORMATTER
# =============================================================================

from typing import List
from src.power_of_3.core.types import PowerOf3Signal, SignalQuality, SignalType


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