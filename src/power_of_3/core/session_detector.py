"""
Session Detection Module for Power of 3 Trading System
=====================================================
Handles London Kill Zone, NY Kill Zone, and session timing logic.
"""

from datetime import datetime, time
from typing import Dict, Optional

import pytz

from src.power_of_3.core.types import PowerOf3Phase, TradingSession


class SessionDetector:
    """Detects trading sessions and timing for Power of 3 methodology"""
    
    def __init__(self):
        self.london_tz = pytz.timezone('Europe/London')
        self.ny_tz = pytz.timezone('America/New_York')
        
        # Session definitions
        self.sessions = {
            'london_kill': {
                'start': time(8, 30),   # 08:30 London time
                'end': time(11, 30),    # 11:30 London time
                'timezone': self.london_tz
            },
            'ny_kill': {
                'start': time(13, 30),  # 13:30 NY time
                'end': time(16, 30),    # 16:30 NY time  
                'timezone': self.ny_tz
            },
            'london_close': {
                'start': time(15, 30),  # 15:30 London time
                'end': time(17, 0),     # 17:00 London time
                'timezone': self.london_tz
            }
        }
    
    # def get_current_session(self) -> Optional[str]:
    #     """Get the current active session"""
    #     now = datetime.now(pytz.UTC)
        
    #     for session_name, session_info in self.sessions.items():
    #         if self._is_session_active(now, session_info):
    #             return session_name
                
    #     return None
    
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
    
    def _is_session_active(self, current_time: datetime, session_info: Dict) -> bool:
        """Check if a session is currently active"""
        session_tz = session_info['timezone']
        local_time = current_time.astimezone(session_tz).time()
        
        return session_info['start'] <= local_time <= session_info['end']
    
    def get_session_phase(self) -> str:
        """Determine Power of 3 phase: accumulation, manipulation, distribution"""
        current_session = self.get_current_session()
        
        if not current_session:
            return 'inactive'
            
        # Power of 3 timing logic
        now = datetime.now(pytz.UTC)
        session_info = self.sessions[current_session.value]
        session_tz = session_info['timezone']
        local_time = now.astimezone(session_tz).time()
        
        session_start = session_info['start']
        session_end = session_info['end']
        
        # Calculate session progress
        total_minutes = (session_end.hour * 60 + session_end.minute) - (session_start.hour * 60 + session_start.minute)
        current_minutes = (local_time.hour * 60 + local_time.minute) - (session_start.hour * 60 + session_start.minute)
        progress = current_minutes / total_minutes
        
        if progress < 0.33:
            return 'accumulation'
        elif progress < 0.66:
            return 'manipulation'
        else:
            return 'distribution'
        
    
    @staticmethod
    def get_session_phase2(session_start: datetime) -> PowerOf3Phase:
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