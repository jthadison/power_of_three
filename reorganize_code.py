#!/usr/bin/env python3
"""
Power of 3 Trading System - Code Reorganization Script
=====================================================
This script reorganizes your existing monolithic files into a professional
project structure with proper separation of concerns.

Usage: python reorganize_code.py
"""

import os
import shutil
from pathlib import Path

def create_folder_structure():
    """Create the complete folder structure"""
    folders = [
        # Source code structure
        'src/power_of_3/core',
        'src/power_of_3/agents', 
        'src/power_of_3/data/providers',
        'src/power_of_3/database',
        'src/power_of_3/strategies',
        'src/power_of_3/notifications',
        'src/power_of_3/utils',
        'src/power_of_3/config',
        
        # Project structure
        'scripts',
        'database',
        'config', 
        'tests/unit',
        'tests/integration',
        'tests/fixtures',
        'docs',
        'logs',
        'backtest/data',
        'backtest/results', 
        'backtest/reports',
        'monitoring'
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {folder}/")

def create_init_files():
    """Create __init__.py files for proper Python packages"""
    init_files = [
        'src/__init__.py',
        'src/power_of_3/__init__.py',
        'src/power_of_3/core/__init__.py',
        'src/power_of_3/agents/__init__.py',
        'src/power_of_3/data/__init__.py',
        'src/power_of_3/data/providers/__init__.py',
        'src/power_of_3/database/__init__.py',
        'src/power_of_3/strategies/__init__.py',
        'src/power_of_3/notifications/__init__.py',
        'src/power_of_3/utils/__init__.py',
        'src/power_of_3/config/__init__.py',
        'tests/__init__.py',
        'tests/unit/__init__.py',
        'tests/integration/__init__.py'
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✅ Created: {init_file}")

def create_session_detector():
    """Extract session detection logic"""
    session_detector_code = '''"""
Session Detection Module for Power of 3 Trading System
=====================================================
Handles London Kill Zone, NY Kill Zone, and session timing logic.
"""

import pytz
from datetime import datetime, time
from typing import Dict, Optional

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
    
    def get_current_session(self) -> Optional[str]:
        """Get the current active session"""
        now = datetime.now(pytz.UTC)
        
        for session_name, session_info in self.sessions.items():
            if self._is_session_active(now, session_info):
                return session_name
                
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
        session_info = self.sessions[current_session]
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
'''
    
    with open('src/power_of_3/core/session_detector.py', 'w') as f:
        f.write(session_detector_code.strip())
    print("✅ Created: src/power_of_3/core/session_detector.py")

def create_liquidity_zones():
    """Extract liquidity zone detection logic"""
    liquidity_code = '''"""
Liquidity Zone Detection for Power of 3 Trading System
=====================================================
Identifies key support/resistance levels and liquidity pools.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

class LiquidityZoneDetector:
    """Detects liquidity zones and support/resistance levels"""
    
    def __init__(self, lookback_periods: int = 50):
        self.lookback_periods = lookback_periods
    
    def identify_liquidity_zones(self, df: pd.DataFrame) -> List[Dict]:
        """Identify key liquidity zones from price data"""
        zones = []
        
        # Find swing highs and lows
        swing_highs = self._find_swing_highs(df)
        swing_lows = self._find_swing_lows(df)
        
        # Convert to liquidity zones
        for level in swing_highs:
            zones.append({
                'type': 'resistance',
                'price': level,
                'strength': self._calculate_zone_strength(df, level),
                'touches': self._count_touches(df, level)
            })
            
        for level in swing_lows:
            zones.append({
                'type': 'support', 
                'price': level,
                'strength': self._calculate_zone_strength(df, level),
                'touches': self._count_touches(df, level)
            })
        
        # Sort by strength
        return sorted(zones, key=lambda x: x['strength'], reverse=True)
    
    def _find_swing_highs(self, df: pd.DataFrame, window: int = 5) -> List[float]:
        """Find swing high points"""
        highs = []
        for i in range(window, len(df) - window):
            if all(df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, window+1)):
                highs.append(df['high'].iloc[i])
        return highs
    
    def _find_swing_lows(self, df: pd.DataFrame, window: int = 5) -> List[float]:
        """Find swing low points"""
        lows = []
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
        
        return touches * (volume_at_level / len(df))
    
    def _count_touches(self, df: pd.DataFrame, level: float, tolerance: float = 0.001) -> int:
        """Count how many times price touched this level"""
        touches = 0
        for _, row in df.iterrows():
            if abs(row['high'] - level) / level <= tolerance or abs(row['low'] - level) / level <= tolerance:
                touches += 1
        return touches
    
    def detect_liquidity_sweeps(self, df: pd.DataFrame) -> List[Dict]:
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
'''
    
    with open('src/power_of_3/core/liquidity_zones.py', 'w') as f:
        f.write(liquidity_code.strip())
    print("✅ Created: src/power_of_3/core/liquidity_zones.py")

def create_main_setup_py():
    """Create setup.py for package installation"""
    setup_code = '''from setuptools import setup, find_packages

setup(
    name="power-of-3-trading",
    version="1.0.0",
    description="Professional Power of 3 ICT Trading System with CrewAI",
    author="Your Name",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "crewai>=0.1.0",
        "pandas>=1.5.0", 
        "numpy>=1.20.0",
        "yfinance>=0.2.0",
        "psycopg2-binary>=2.9.0",
        "sqlalchemy>=2.0.0",
        "python-dotenv>=1.0.0",
        "pytz>=2023.3",
        "requests>=2.28.0",
        "pydantic>=2.0.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0", 
            "flake8>=6.0.0",
            "mypy>=1.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "power-of-3=power_of_3.main:main",
        ],
    },
)'''
    
    with open('setup.py', 'w') as f:
        f.write(setup_code)
    print("✅ Created: setup.py")

def create_main_runner():
    """Create main system runner"""
    runner_code = '''#!/usr/bin/env python3
"""
Main Trading System Runner
=========================
Entry point for the Power of 3 trading system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from power_of_3.agents.crew_coordinator import TradingCrew
from power_of_3.config.settings import load_config

def main():
    """Main entry point for the trading system"""
    print("🚀 Starting Power of 3 Trading System...")
    
    # Load configuration
    config = load_config()
    
    # Initialize trading crew
    crew = TradingCrew(config)
    
    # Start trading
    crew.start_trading()

if __name__ == "__main__":
    main()
'''
    
    with open('scripts/run_trading_system.py', 'w') as f:
        f.write(runner_code.strip())
    print("✅ Created: scripts/run_trading_system.py")

def create_config_files():
    """Create configuration files"""
    
    # Main settings
    settings_code = '''"""
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
'''
    
    with open('src/power_of_3/config/settings.py', 'w') as f:
        f.write(settings_code.strip())
    
    # Environment template
    env_template = '''# Power of 3 Trading System Configuration
# ==========================================

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/trading_db
DB_POOL_SIZE=5

# Trading Configuration
MAX_RISK_PERCENT=2.0
ACCOUNT_SIZE=10000
SYMBOLS=US30,NAS100,SPX500,XAUUSD

# API Keys
OPENAI_API_KEY=your-openai-api-key
TWELVE_DATA_API_KEY=your-twelve-data-key
OANDA_API_KEY=your-oanda-key

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trading.log
'''
    
    with open('.env.example', 'w') as f:
        f.write(env_template.strip())
    
    print("✅ Created: src/power_of_3/config/settings.py")
    print("✅ Created: .env.example")

def move_existing_files():
    """Instructions for moving existing files"""
    print("\n📁 FILE MOVEMENT INSTRUCTIONS:")
    print("=" * 50)
    
    movements = [
        ("power_of_3_signal_generator.py", "Break into core modules (session_detector.py, liquidity_zones.py, etc.)"),
        ("power_of_3_crewai_integration.py", "src/power_of_3/agents/crew_coordinator.py"),
        ("postgres_trading_schema.sql", "database/schema.sql"),
        ("database_integration.py", "src/power_of_3/database/repository.py"),
        ("database_setup_script.py", "scripts/setup_database.py"),
        ("power_of_3_database_guide.md", "docs/DATABASE.md")
    ]
    
    for old, new in movements:
        print(f"📄 {old} → {new}")

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv
env/
venv/

# Logs
logs/
*.log

# Database
*.db
*.sqlite3

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Trading specific
backtest/results/
monitoring/alerts/
config/production.env
'''
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content.strip())
    print("✅ Created: .gitignore")

def main():
    """Run the complete reorganization process"""
    print("🚀 Power of 3 Trading System - Code Reorganization")
    print("=" * 55)
    
    # Create structure
    print("\n📁 Creating folder structure...")
    create_folder_structure()
    
    print("\n📦 Creating package files...")
    create_init_files()
    
    print("\n🔧 Creating core modules...")
    create_session_detector()
    create_liquidity_zones()
    
    print("\n⚙️ Creating configuration...")
    create_config_files()
    
    print("\n🎯 Creating entry points...")
    create_main_runner()
    create_main_setup_py()
    
    print("\n📝 Creating project files...")
    create_gitignore()
    
    # Instructions
    move_existing_files()
    
    print("\n✅ REORGANIZATION COMPLETE!")
    print("=" * 30)
    print("\n🎯 NEXT STEPS:")
    print("1. Move your existing files as shown above")
    print("2. Copy content from monolithic files into new modules")
    print("3. Update import statements") 
    print("4. Run: pip install -e .")
    print("5. Test: python scripts/run_trading_system.py")
    print("\n🚀 Your code is now professionally organized!")

if __name__ == "__main__":
    main()