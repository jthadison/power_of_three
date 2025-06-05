#!/usr/bin/env python3
"""
Power of 3 Trading System - Backtest Runner
==========================================

Command-line interface for running Power of 3 backtests.
Integrates seamlessly with your existing architecture.

Usage Examples:
    # Single symbol backtest
    python run_backtest.py single --symbol US30 --start 2024-01-01 --end 2024-03-31 --plot
    
    # Multi-symbol comparison
    python run_backtest.py multi --symbols US30 NAS100 SPX500 --start 2024-01-01 --end 2024-03-31 --plot
    
    # Parameter optimization
    python run_backtest.py optimize --symbol US30 --start 2024-01-01 --end 2024-03-31 --plot
    
    # Quick test with default parameters
    python run_backtest.py quick --symbol US30
"""

import sys
import os
import argparse
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging

# Add src directory to Python path
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required components are available"""
    missing_components = []
    
    try:
        from src.power_of_3.config.settings import load_config
        logger.info("✅ Configuration system found")
    except ImportError:
        missing_components.append("Configuration system (src/power_of_3/config/settings.py)")
    
    try:
        from src.power_of_3.core.signal_generator import PowerOf3SignalGenerator
        logger.info("✅ Signal generator found")
    except ImportError:
        missing_components.append("Signal generator (src/power_of_3/core/signal_generator.py)")
    
    try:
        from src.power_of_3.backtesting.engine import PowerOf3BacktestEngine
        logger.info("✅ Backtesting engine found")
    except ImportError:
        missing_components.append("Backtesting engine (src/power_of_3/backtesting/engine.py)")
    
    try:
        from src.power_of_3.backtesting.interface import BacktestInterface
        logger.info("✅ Backtesting interface found")
    except ImportError:
        missing_components.append("Backtesting interface (src/power_of_3/backtesting/interface.py)")
    
    if missing_components:
        logger.error("❌ Missing required components:")
        for component in missing_components:
            logger.error(f"   - {component}")
        logger.error("\n💡 Please run setup_backtesting.py first to set up the backtesting system")
        return False
    
    return True

async def run_single_backtest(args):
    """Run backtest for a single symbol"""
    logger.info(f"🎯 Running single symbol backtest: {args.symbol}")
    
    try:
        from src.power_of_3.backtesting.interface import BacktestInterface
        
        # Initialize backtesting interface (it loads config internally)
        interface = BacktestInterface(results_dir="backtest_results")
        
        # Run backtest (saves results automatically)
        results = await interface.run_single_symbol_backtest(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            initial_balance=getattr(args, 'balance', 10000.0),
            timeframe=getattr(args, 'timeframe', '5min'),
            save_results=True
        )
        
        if results:
            # Display summary
            print("\n" + "="*60)
            print(f"📊 BACKTEST RESULTS - {args.symbol}")
            print("="*60)
            print(f"Period: {args.start} to {args.end}")
            print(f"Total Trades: {results.total_trades}")
            print(f"Win Rate: {results.win_rate:.1f}%")
            print(f"Total Return: {results.total_pnl_pct:.2f}%")
            print(f"Net P&L: ${results.total_pnl:.2f}")
            print(f"Max Drawdown: {results.max_drawdown_pct:.2f}%")
            print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
            print(f"Profit Factor: {results.profit_factor:.2f}")
            print(f"Avg Signal Quality: {results.avg_signal_quality:.1f}/10")
            
            # Generate visualization if requested
            if getattr(args, 'plot', False):
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_filename = f"single_{args.symbol}_{timestamp}"
                    chart_paths = await interface.create_visualizations(
                        results, 
                        str(Path("backtest_results") / base_filename)
                    )
                    if chart_paths:
                        print(f"\n📈 Charts created:")
                        for chart_path in chart_paths:
                            print(f"   {chart_path}")
                except Exception as e:
                    logger.warning(f"Could not generate charts: {e}")
            
            print(f"\n📁 All results saved to: backtest_results/")
            return True
        else:
            logger.error("❌ Backtest failed to produce results")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running single backtest: {e}")
        return False

async def run_multi_backtest(args):
    """Run backtest for multiple symbols"""
    logger.info(f"🎯 Running multi-symbol backtest: {', '.join(args.symbols)}")
    
    try:
        from src.power_of_3.backtesting.interface import BacktestInterface
        
        # Initialize backtesting interface
        interface = BacktestInterface(results_dir="backtest_results")
        
        # Run multi-symbol backtest (saves results automatically)
        results = await interface.run_multi_symbol_backtest(
            symbols=args.symbols,
            start_date=args.start,
            end_date=args.end,
            initial_balance=getattr(args, 'balance', 10000.0),
            timeframe=getattr(args, 'timeframe', '5min'),
            save_results=True
        )
        
        if results:
            # Display comparison summary
            print("\n" + "="*80)
            print("📊 MULTI-SYMBOL BACKTEST COMPARISON")
            print("="*80)
            print(f"Period: {args.start} to {args.end}")
            print(f"Symbols: {', '.join(args.symbols)}")
            print("\nPerformance Summary:")
            print("-" * 80)
            print(f"{'Symbol':<10} {'Trades':<8} {'Win Rate':<10} {'Return':<10} {'Sharpe':<8} {'Max DD':<10}")
            print("-" * 80)
            
            for symbol, result in results.items():
                print(f"{symbol:<10} {result.total_trades:<8} {result.win_rate:<10.1f}% "
                      f"{result.total_pnl_pct:<9.2f}% {result.sharpe_ratio:<8.2f} {result.max_drawdown_pct:<9.2f}%")
            
            # Generate comparison chart if requested
            if getattr(args, 'plot', False):
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_filename = f"multi_symbol_{timestamp}"
                    chart_paths = await interface.create_visualizations(
                        results, 
                        str(Path("backtest_results") / base_filename)
                    )
                    if chart_paths:
                        print(f"\n📈 Comparison charts created:")
                        for chart_path in chart_paths:
                            print(f"   {chart_path}")
                except Exception as e:
                    logger.warning(f"Could not generate comparison charts: {e}")
            
            print(f"\n📁 All results saved to: backtest_results/")
            return True
        else:
            logger.error("❌ Multi-symbol backtest failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running multi-symbol backtest: {e}")
        return False

async def run_optimization(args):
    """Run parameter optimization"""
    logger.info(f"🎯 Running parameter optimization: {args.symbol}")
    
    try:
        from src.power_of_3.backtesting.interface import BacktestInterface
        
        # Initialize backtesting interface
        interface = BacktestInterface(results_dir="backtest_results")
        
        # Define optimization parameters
        parameter_ranges = {
            'min_risk_reward': getattr(args, 'rr_values', [3.0, 4.0, 5.0, 6.0, 7.0]),
            'max_risk_percent': getattr(args, 'risk_values', [1.0, 1.5, 2.0, 2.5])
        }
        
        # Run optimization (saves results automatically)
        optimization_results = await interface.run_parameter_optimization(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            parameter_ranges=parameter_ranges,
            initial_balance=getattr(args, 'balance', 10000.0),
            timeframe=getattr(args, 'timeframe', '5min')
        )
        
        if optimization_results and optimization_results.get('all_results'):
            # Display optimization summary
            print("\n" + "="*80)
            print(f"🔧 PARAMETER OPTIMIZATION RESULTS - {args.symbol}")
            print("="*80)
            print(f"Period: {args.start} to {args.end}")
            print(f"Total combinations tested: {optimization_results['total_tested']}")
            
            if optimization_results['best_parameters']:
                best = optimization_results['best_parameters']
                print(f"\n🏆 BEST PARAMETERS:")
                print(f"   Min Risk/Reward Ratio: {best['min_risk_reward']}")
                print(f"   Max Risk Percentage: {best['max_risk_percent']}%")
                print(f"   Total Return: {best['total_return']:.2f}%")
                print(f"   Win Rate: {best['win_rate']:.1f}%")
                print(f"   Sharpe Ratio: {best['sharpe_ratio']:.2f}")
                print(f"   Max Drawdown: {best['max_drawdown']:.2f}%")
                print(f"   Optimization Score: {best['score']:.3f}")
            
            print("\nTop 5 Parameter Combinations:")
            print("-" * 80)
            print(f"{'RR Ratio':<10} {'Risk %':<8} {'Return %':<10} {'Win Rate':<10} {'Sharpe':<8} {'Score':<8}")
            print("-" * 80)
            
            # Show top 5 results
            all_results = optimization_results['all_results']
            for i, result in enumerate(all_results[:5]):
                print(f"{result['min_risk_reward']:<10.1f} {result['max_risk_percent']:<8.1f}% "
                      f"{result['total_return']:<9.2f}% {result['win_rate']:<9.1f}% "
                      f"{result['sharpe_ratio']:<8.2f} {result['score']:<8.3f}")
            
            print(f"\n📁 Full optimization results saved to: backtest_results/")
            return True
        else:
            logger.error("❌ Parameter optimization failed or produced no results")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running optimization: {e}")
        return False

async def run_quick_test(args):
    """Run a quick test with default parameters"""
    logger.info(f"🚀 Running quick test: {args.symbol}")
    
    # Set default parameters for quick test
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    
    # Create args object with defaults
    class QuickArgs:
        def __init__(self, symbol):
            self.symbol = symbol
            self.start = start_date
            self.end = end_date
            self.timeframe = '1H'
            self.balance = 10000.0
            self.plot = True
    
    quick_args = QuickArgs(args.symbol)
    
    # Run single backtest with quick parameters
    return await run_single_backtest(quick_args)

def create_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="Power of 3 Trading System Backtest Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (last 3 months)
  python run_backtest.py quick --symbol US30
  
  # Single symbol backtest
  python run_backtest.py single --symbol US30 --start 2024-01-01 --end 2024-03-31 --plot
  
  # Multi-symbol comparison  
  python run_backtest.py multi --symbols US30 NAS100 SPX500 --start 2024-01-01 --end 2024-03-31 --plot
  
  # Parameter optimization
  python run_backtest.py optimize --symbol US30 --start 2024-01-01 --end 2024-03-31 --plot
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Backtest type')
    
    # Quick test command
    quick_parser = subparsers.add_parser('quick', help='Quick test (last 3 months)')
    quick_parser.add_argument('--symbol', required=True, 
                             choices=['US30', 'NAS100', 'SPX500', 'XAUUSD'],
                             help='Symbol to test')
    
    # Single symbol command
    single_parser = subparsers.add_parser('single', help='Single symbol backtest')
    single_parser.add_argument('--symbol', required=True,
                              choices=['US30', 'NAS100', 'SPX500', 'XAUUSD'],
                              help='Symbol to backtest')
    single_parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    single_parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    single_parser.add_argument('--timeframe', default='5min', 
                              choices=['1min', '5min', '15min', '1h'],
                              help='Timeframe (default: 5min)')
    single_parser.add_argument('--balance', type=float, default=10000.0,
                              help='Initial account balance (default: 10000)')
    single_parser.add_argument('--plot', action='store_true',
                              help='Generate charts')
    
    # Multi-symbol command
    multi_parser = subparsers.add_parser('multi', help='Multi-symbol comparison')
    multi_parser.add_argument('--symbols', nargs='+', required=True,
                             choices=['US30', 'NAS100', 'SPX500', 'XAUUSD'],
                             help='Symbols to compare')
    multi_parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    multi_parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    multi_parser.add_argument('--timeframe', default='5min',
                             choices=['1min', '5min', '15min', '1h'],
                             help='Timeframe (default: 5min)')
    multi_parser.add_argument('--balance', type=float, default=10000.0,
                             help='Initial account balance (default: 10000)')
    multi_parser.add_argument('--plot', action='store_true',
                             help='Generate comparison charts')
    
    # Optimization command
    opt_parser = subparsers.add_parser('optimize', help='Parameter optimization')
    opt_parser.add_argument('--symbol', required=True,
                           choices=['US30', 'NAS100', 'SPX500', 'XAUUSD'],
                           help='Symbol to optimize')
    opt_parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    opt_parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    opt_parser.add_argument('--timeframe', default='5min',
                           choices=['1min', '5min', '15min', '1h'],
                           help='Timeframe (default: 5min)')
    opt_parser.add_argument('--balance', type=float, default=10000.0,
                           help='Initial account balance (default: 10000)')
    opt_parser.add_argument('--rr-values', nargs='+', type=float,
                           default=[3.0, 4.0, 5.0, 6.0, 7.0],
                           help='Risk-reward ratios to test (default: 3 4 5 6 7)')
    opt_parser.add_argument('--risk-values', nargs='+', type=float,
                           default=[1.0, 1.5, 2.0, 2.5],
                           help='Risk percentages to test (default: 1 1.5 2 2.5)')
    opt_parser.add_argument('--plot', action='store_true',
                           help='Generate optimization visualizations')
    
    return parser

async def main():
    """Main entry point"""
    print("🚀 Power of 3 Trading System - Backtest Runner")
    print("=" * 50)
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Missing required components. Please run setup first.")
        return
    
    # Create output directory
    output_dir = Path("backtest_results")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Route to appropriate function
        success = False
        
        if args.command == 'quick':
            success = await run_quick_test(args)
        elif args.command == 'single':
            success = await run_single_backtest(args)
        elif args.command == 'multi':
            success = await run_multi_backtest(args)
        elif args.command == 'optimize':
            success = await run_optimization(args)
        
        if success:
            print("\n✅ Backtest completed successfully!")
            print(f"📁 Results saved in: {output_dir.absolute()}")
        else:
            print("\n❌ Backtest failed. Check logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Backtest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())