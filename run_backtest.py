#!/usr/bin/env python3
"""
Power of 3 Trading System - Backtest Runner
==========================================

Command-line interface for running Power of 3 backtests.
Integrates seamlessly with your existing architecture.

Usage Examples:
    # Single symbol backtest
    python run_backtest.py single --symbol US30 --start 2024-01-01 --end 2024-03-31
    
    # Multi-symbol comparison
    python run_backtest.py multi --symbols US30 NAS100 SPX500 --start 2024-01-01 --end 2024-03-31
    
    # Parameter optimization
    python run_backtest.py optimize --symbol US30 --start 2024-01-01 --end 2024-03-31
    
    # Quick test with default parameters
    python run_backtest.py quick --symbol US30
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from src.power_of_3.data.providers.data_provider import DataProvider
from src.power_of_3.backtesting.engine import FallbackDataProvider

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

def setup_data_provider():
    """Setup data provider with fallback"""
    try:
        # Try to use existing DataProvider
        #from src.power_of_3.data.providers.data_provider import DataProvider
        return DataProvider()
    except ImportError:
            logger.error("❌ No data provider available")
            return None

async def run_single_backtest(args):
    """Run backtest for a single symbol"""
    logger.info(f"🎯 Running single symbol backtest: {args.symbol}")
    
    try:
        from src.power_of_3.backtesting.interface import BacktestInterface
        from src.power_of_3.config.settings import load_config
        
        # Load configuration
        config = load_config()
        
        # Setup data provider
        data_provider = setup_data_provider()
        if not data_provider:
            return False
        
        # Initialize backtesting interface
        results_dir = "backtest_results"  # Define a valid results directory
        interface = BacktestInterface(config)
        
        # Run backtest
        results = await interface.run_single_symbol_backtest(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            initial_balance= getattr(args, 'initial_balance', 10000.0),
            timeframe=getattr(args, 'timeframe', '5min')
            #min_rr=getattr(args, 'min_rr', 5.0),
            #max_risk_percent=getattr(args, 'max_risk', 2.0)
        )
        
        if results:
            # Save results
            output_dir = Path("backtest_results")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{args.symbol}_{args.start}_{args.end}_{timestamp}"
            
            # Save JSON report
            json_file = output_dir / f"{base_filename}.json"
            interface.save_results_json(results, json_file)
            
            # Save CSV data
            csv_file = output_dir / f"{base_filename}.csv"
            interface.save_results_csv(results, csv_file)
            
            # Generate markdown report
            md_file = output_dir / f"{base_filename}.md"
            interface.generate_markdown_report(results, md_file)
            
            # Display summary
            print("\n" + "="*60)
            print(f"📊 BACKTEST RESULTS - {args.symbol}")
            print("="*60)
            print(f"Period: {args.start} to {args.end}")
            print(f"Total Trades: {results.total_trades}")
            print(f"Win Rate: {results.win_rate:.1f}%")
            print(f"Net Profit: ${results.net_profit:.2f}")
            print(f"Max Drawdown: ${results.max_drawdown:.2f}")
            print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
            print(f"Profit Factor: {results.profit_factor:.2f}")
            print(f"\n📁 Results saved to:")
            print(f"   JSON: {json_file}")
            print(f"   CSV: {csv_file}")
            print(f"   Report: {md_file}")
            
            # Generate visualization if requested
            if getattr(args, 'plot', False):
                try:
                    plot_file = output_dir / f"{base_filename}_chart.png"
                    interface.create_equity_curve(results, plot_file)
                    print(f"   Chart: {plot_file}")
                except Exception as e:
                    logger.warning(f"Could not generate chart: {e}")
            
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
        from src.power_of_3.config.settings import load_config
        
        # Load configuration
        config = load_config()
        
        # Setup data provider
        data_provider = setup_data_provider()
        if not data_provider:
            return False
        
        # Initialize backtesting interface
        interface = BacktestInterface(config, data_provider)
        
        # Run multi-symbol backtest
        results = await interface.run_multi_symbol_backtest(
            symbols=args.symbols,
            start_date=args.start,
            end_date=args.end,
            timeframe=getattr(args, 'timeframe', '5min'),
            min_risk_reward=getattr(args, 'min_rr', 5.0),
            max_risk_percent=getattr(args, 'max_risk', 2.0)
        )
        
        if results:
            # Save comparison results
            output_dir = Path("backtest_results")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"multi_symbol_{args.start}_{args.end}_{timestamp}"
            
            # Save comparison report
            comparison_file = output_dir / f"{base_filename}_comparison.json"
            interface.save_comparison_results(results, comparison_file)
            
            # Display comparison summary
            print("\n" + "="*80)
            print("📊 MULTI-SYMBOL BACKTEST COMPARISON")
            print("="*80)
            print(f"Period: {args.start} to {args.end}")
            print(f"Symbols: {', '.join(args.symbols)}")
            print("\nPerformance Summary:")
            print("-" * 80)
            print(f"{'Symbol':<10} {'Trades':<8} {'Win Rate':<10} {'Net P&L':<12} {'Sharpe':<8} {'Max DD':<10}")
            print("-" * 80)
            
            for symbol, result in results.items():
                print(f"{symbol:<10} {result.total_trades:<8} {result.win_rate:<10.1f}% "
                      f"${result.net_profit:<11.2f} {result.sharpe_ratio:<8.2f} ${result.max_drawdown:<10.2f}")
            
            print(f"\n📁 Comparison results saved to: {comparison_file}")
            
            # Generate comparison chart if requested
            if getattr(args, 'plot', False):
                try:
                    chart_file = output_dir / f"{base_filename}_comparison.png"
                    interface.create_comparison_chart(results, chart_file)
                    print(f"📈 Comparison chart: {chart_file}")
                except Exception as e:
                    logger.warning(f"Could not generate comparison chart: {e}")
            
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
        from src.power_of_3.config.settings import load_config
        
        # Load configuration
        config = load_config()
        
        # Setup data provider
        data_provider = setup_data_provider()
        if not data_provider:
            return False
        
        # Initialize backtesting interface
        interface = BacktestInterface(config, data_provider)
        
        # Define optimization parameters
        risk_reward_values = getattr(args, 'rr_values', [3.0, 4.0, 5.0, 6.0, 7.0])
        risk_percent_values = getattr(args, 'risk_values', [1.0, 1.5, 2.0, 2.5])
        
        # Run optimization
        optimization_results = await interface.run_parameter_optimization(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            risk_reward_values=risk_reward_values,
            risk_percent_values=risk_percent_values,
            timeframe=getattr(args, 'timeframe', '5min')
        )
        
        if optimization_results:
            # Save optimization results
            output_dir = Path("backtest_results")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            opt_file = output_dir / f"{args.symbol}_optimization_{timestamp}.json"
            
            interface.save_optimization_results(optimization_results, opt_file)
            
            # Display optimization summary
            print("\n" + "="*80)
            print(f"🔧 PARAMETER OPTIMIZATION RESULTS - {args.symbol}")
            print("="*80)
            print(f"Period: {args.start} to {args.end}")
            print("\nTop 5 Parameter Combinations:")
            print("-" * 80)
            print(f"{'RR Ratio':<10} {'Risk %':<8} {'Net P&L':<12} {'Win Rate':<10} {'Sharpe':<8} {'Trades':<8}")
            print("-" * 80)
            
            # Sort by net profit and show top 5
            sorted_results = sorted(optimization_results.items(), 
                                  key=lambda x: x[1].net_profit, reverse=True)
            
            for i, (params, result) in enumerate(sorted_results[:5]):
                rr, risk = params
                print(f"{rr:<10.1f} {risk:<8.1f}% ${result.net_profit:<11.2f} "
                      f"{result.win_rate:<10.1f}% {result.sharpe_ratio:<8.2f} {result.total_trades:<8}")
            
            print(f"\n📁 Full optimization results saved to: {opt_file}")
            
            # Generate optimization heatmap if requested
            if getattr(args, 'plot', False):
                try:
                    heatmap_file = output_dir / f"{args.symbol}_optimization_heatmap_{timestamp}.png"
                    interface.create_optimization_heatmap(optimization_results, heatmap_file)
                    print(f"🔥 Optimization heatmap: {heatmap_file}")
                except Exception as e:
                    logger.warning(f"Could not generate heatmap: {e}")
            
            return True
        else:
            logger.error("❌ Parameter optimization failed")
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
            self.timeframe = '5min'
            self.min_rr = 5.0
            self.max_risk = 2.0
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
  python run_backtest.py single --symbol US30 --start 2024-01-01 --end 2024-03-31
  
  # Multi-symbol comparison  
  python run_backtest.py multi --symbols US30 NAS100 SPX500 --start 2024-01-01 --end 2024-03-31
  
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
    single_parser.add_argument('--min-rr', type=float, default=5.0,
                              help='Minimum risk-reward ratio (default: 5.0)')
    single_parser.add_argument('--max-risk', type=float, default=2.0,
                              help='Maximum risk percentage (default: 2.0)')
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
    multi_parser.add_argument('--min-rr', type=float, default=5.0,
                             help='Minimum risk-reward ratio (default: 5.0)')
    multi_parser.add_argument('--max-risk', type=float, default=2.0,
                             help='Maximum risk percentage (default: 2.0)')
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
    opt_parser.add_argument('--rr-values', nargs='+', type=float,
                           default=[3.0, 4.0, 5.0, 6.0, 7.0],
                           help='Risk-reward ratios to test (default: 3 4 5 6 7)')
    opt_parser.add_argument('--risk-values', nargs='+', type=float,
                           default=[1.0, 1.5, 2.0, 2.5],
                           help='Risk percentages to test (default: 1 1.5 2 2.5)')
    opt_parser.add_argument('--plot', action='store_true',
                           help='Generate optimization heatmap')
    
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