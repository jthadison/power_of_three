#!/usr/bin/env python3
"""
Minimal Live Trading Script
===========================
Simple version to test the trading system without complex dependencies.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

print("🚀 Power of 3 Live Trading System")
print("=" * 40)

def main():
    """Main function to run live trading."""
    
    try:
        # Test import
        print("📦 Importing LiveTradingEngine...")
        from power_of_3.live.trading_engine import LiveTradingEngine
        print("✅ Import successful!")
        
        # Create engine
        print("🔧 Creating trading engine...")
        engine = LiveTradingEngine({
            'symbols': ['US30', 'NAS100', 'SPX500', 'XAUUSD'],
            'account_balance': 10000.0,
            'max_risk_percent': 2.0,
            'min_risk_reward': 5.0
        })
        print("✅ Engine created!")
        
        # Get status
        status = engine.get_status()
        print(f"📊 Engine status: {status}")
        
        # Start engine
        print("🚀 Starting trading engine...")
        engine.start()
        
        print("\n💡 Trading system is running!")
        print("📝 This is a minimal test version.")
        print("⚠️ No real trading will occur yet.")
        
        print("\n🔄 To implement full live trading:")
        print("1. Add data feed connections")
        print("2. Add broker integration") 
        print("3. Add signal generation")
        print("4. Add risk management")
        
        # Stop engine
        print("\n⏹️ Stopping engine...")
        engine.stop()
        
        print("✅ Test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n🔧 To fix this:")
        print("1. Run: python fix_import_error.py")
        print("2. Then try again")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 Your Power of 3 system is working!")
        print("Ready to add full live trading features.")
    else:
        print("\n⚠️ Issues found. Check error messages above.")