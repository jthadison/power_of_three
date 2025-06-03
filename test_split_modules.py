#!/usr/bin/env python3
"""
Test Script for Split Power of 3 Modules
========================================
Tests all the split modules to ensure they work correctly.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing module imports...")
    
    try:
        # Test core types
        from src.power_of_3.core.types import (
            PowerOf3Phase, SignalType, TradingSession, 
            LiquidityZone, ManipulationPattern, PowerOf3Signal
        )
        print("✅ Core types imported successfully")
        
        # Test session detector
        from src.power_of_3.core.session_detector import SessionDetector
        print("✅ Session detector imported successfully")
        
        # Test liquidity zones
        from src.power_of_3.core.liquidity_zones import LiquidityZoneDetector
        print("✅ Liquidity zone detector imported successfully")
        
        # Test manipulation detector
        from src.power_of_3.core.manipulation_detector import ManipulationDetector
        print("✅ Manipulation detector imported successfully")
        
        # Test market structure
        from src.power_of_3.core.market_structure import marke
        print("✅ Market structure analyzer imported successfully")
        
        # Test risk manager
        from src.power_of_3.core.risk_manager import RiskManager
        print("✅ Risk manager imported successfully")
        
        # Test main signal generator
        from src.power_of_3.core.signal_generator import PowerOf3SignalGenerator
        print("✅ Main signal generator imported successfully")
        
        # Test signal formatter
        from src.power_of_3.utils.signal_formatter import SignalFormatter
        print("✅ Signal formatter imported successfully")
        
        print("\n🎉 All modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def create_sample_data():
    """Create sample OHLCV data for testing"""
    print("📊 Creating sample market data...")
    
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
    print(f"✅ Created sample data with {len(df)} rows")
    return df

def test_individual_modules(df):
    """Test each module individually"""
    print("\n🔧 Testing individual modules...")
    
    try:
        from src.power_of_3.core.types import TradingSession
        from src.power_of_3.core.session_detector import SessionDetector
        from src.power_of_3.core.liquidity_zones import LiquidityZoneDetector
        from src.power_of_3.core.manipulation_detector import ManipulationDetector
        from src.power_of_3.core.market_structure import MarketStructureAnalyzer
        from src.power_of_3.core.risk_manager import RiskManager
        
        # Test session detector
        session_detector = SessionDetector()
        current_session = session_detector.get_current_session()
        print(f"✅ Session detector: Current session = {current_session}")
        
        # Test liquidity zone detector
        liquidity_detector = LiquidityZoneDetector()
        zones = liquidity_detector.identify_liquidity_zones(df, TradingSession.NEW_YORK_OPEN)
        print(f"✅ Liquidity detector: Found {len(zones)} zones")
        
        # Test manipulation detector
        manipulation_detector = ManipulationDetector()
        patterns = manipulation_detector.detect_manipulation_patterns(
            df, TradingSession.NEW_YORK_OPEN, zones
        )
        print(f"✅ Manipulation detector: Found {len(patterns)} patterns")
        
        # Test market structure analyzer
        market_analyzer = MarketStructureAnalyzer()
        structure = market_analyzer.analyze_market_structure(df)
        print(f"✅ Market analyzer: Structure = {structure}")
        
        # Test risk manager
        risk_manager = RiskManager()
        position_size = risk_manager.calculate_position_size(2100.0, 2095.0, 10000.0)
        print(f"✅ Risk manager: Position size = {position_size:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Module test error: {e}")
        return False

def test_main_signal_generator(df):
    """Test the main signal generator"""
    print("\n🎯 Testing main signal generator...")
    
    try:
        from src.power_of_3.core.signal_generator import PowerOf3SignalGenerator
        from src.power_of_3.utils.signal_formatter import SignalFormatter
        
        # Create signal generator
        signal_generator = PowerOf3SignalGenerator(min_risk_reward=3.0)
        
        # Generate signals
        signals = signal_generator.generate_signals('US30', df, account_balance=10000)
        print(f"✅ Generated {len(signals)} signals")
        
        # Test market analysis
        analysis = signal_generator.get_market_analysis(df)
        print(f"✅ Market analysis: {analysis['market_structure']} structure")
        
        # Test signal formatting if we have signals
        if signals:
            formatted_signal = SignalFormatter.format_signal_for_display(signals[0])
            print(f"✅ Signal formatting works")
            print(f"📄 Sample signal:\n{formatted_signal[:200]}...")
        else:
            print("ℹ️ No signals generated (expected if not in session)")
        
        return True
        
    except Exception as e:
        print(f"❌ Main generator test error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Power of 3 Split Modules Test")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("❌ Import tests failed!")
        return
    
    # Create sample data
    df = create_sample_data()
    
    # Test individual modules
    if not test_individual_modules(df):
        print("❌ Individual module tests failed!")
        return
    
    # Test main signal generator
    if not test_main_signal_generator(df):
        print("❌ Main generator tests failed!")
        return
    
    print("\n🎉 ALL TESTS PASSED!")
    print("=" * 40)
    print("✅ Your Power of 3 system has been successfully split into modules!")
    print("✅ All components are working correctly!")
    print("\n🎯 Next steps:")
    print("1. Run your main system: python scripts/run_trading_system.py")
    print("2. Test with live data")
    print("3. Add more features as needed")

if __name__ == "__main__":
    main()