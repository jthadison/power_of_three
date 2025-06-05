# Power of 3 Backtesting Interface  
# Save the 'Integrated Backtesting Interface' artifact as this file
"""
Power of 3 Integrated Backtesting Interface
===========================================

User interface for backtesting that integrates with your existing architecture.

File Location: src/power_of_3/backtesting/interface.py
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Try to import plotting libraries
try:   
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠ Matplotlib not available. Install with: pip install matplotlib")

# Import from your existing architecture
from .engine import PowerOf3BacktestEngine, BacktestResults
from ..config.settings import load_config

logger = logging.getLogger(__name__)

class BacktestInterface:
    """
    Professional interface for Power of 3 backtesting.
    Provides both programmatic API and command-line interface.
    """
    
    def __init__(self, config: Optional[Dict] = None, results_dir: str = "backtest/results"):
        """
        Initialize backtesting interface
        
        Args:
            config: Configuration dictionary
            results_dir: Directory to save results
        """
        self.config = config or load_config()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize engine
        self.engine = PowerOf3BacktestEngine(config=self.config, use_database=False)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for backtesting"""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.results_dir / 'backtest.log')
            ]
        )
    
    # =========================================================================
    # PROGRAMMATIC API
    # =========================================================================
    
    async def run_single_symbol_backtest(self, symbol: str, start_date: str, 
                                        end_date: str, initial_balance: float = 10000.0,
                                        timeframe: str = '5min',
                                        save_results: bool = True) -> BacktestResults:
        """
        Run backtest for a single symbol
        
        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_balance: Starting balance
            timeframe: Data timeframe
            save_results: Whether to save results to file
            
        Returns:
            BacktestResults object
        """
        logger.info(f"🚀 Running single symbol backtest: {symbol}")
        
        try:
            results = await self.engine.run_backtest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_balance=initial_balance,
                timeframe=timeframe
            )
            
            if save_results:
                await self._save_results(results, f"single_{symbol}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in single symbol backtest: {e}")
            raise
    
    async def run_multi_symbol_backtest(self, symbols: List[str], start_date: str,
                                       end_date: str, initial_balance: float = 10000.0,
                                       timeframe: str = '5min',
                                       save_results: bool = True) -> Dict[str, BacktestResults]:
        """
        Run backtest across multiple symbols
        
        Args:
            symbols: List of trading symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_balance: Starting balance
            timeframe: Data timeframe
            save_results: Whether to save results
            
        Returns:
            Dictionary of symbol -> BacktestResults
        """
        logger.info(f"🚀 Running multi-symbol backtest: {', '.join(symbols)}")
        
        try:
            results = await self.engine.run_multi_symbol_backtest(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                initial_balance=initial_balance,
                timeframe=timeframe
            )
            
            if save_results:
                await self._save_multi_results(results, "multi_symbol")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in multi-symbol backtest: {e}")
            raise
    
    async def run_parameter_optimization(self, symbol: str, start_date: str,
                                        end_date: str, parameter_ranges: Dict,
                                        initial_balance: float = 10000.0,
                                        timeframe: str = '5min') -> Dict:
        """
        Run parameter optimization backtest
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            parameter_ranges: Dictionary of parameter ranges to test
            initial_balance: Starting balance
            timeframe: Data timeframe
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"🔧 Running parameter optimization for {symbol}")
        
        # This is a simplified version - you could expand with grid search, etc.
        optimization_results = []
        
        # Example parameter ranges (expand as needed)
        min_rr_values = parameter_ranges.get('min_risk_reward', [3.0, 4.0, 5.0, 6.0])
        max_risk_values = parameter_ranges.get('max_risk_percent', [1.0, 1.5, 2.0, 2.5])
        
        total_combinations = len(min_rr_values) * len(max_risk_values)
        logger.info(f"Testing {total_combinations} parameter combinations")
        
        combination = 0
        for min_rr in min_rr_values:
            for max_risk in max_risk_values:
                combination += 1
                logger.info(f"Testing combination {combination}/{total_combinations}: "
                           f"min_rr={min_rr}, max_risk={max_risk}")
                
                # Create new engine with specific parameters
                test_config = self.config.copy()
                test_config['trading'] = test_config.get('trading', {})
                test_config['trading']['min_risk_reward'] = min_rr
                test_config['trading']['max_risk_percent'] = max_risk
                
                test_engine = PowerOf3BacktestEngine(config=test_config)
                
                try:
                    results = await test_engine.run_backtest(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        initial_balance=initial_balance,
                        timeframe=timeframe
                    )
                    
                    optimization_results.append({
                        'min_risk_reward': min_rr,
                        'max_risk_percent': max_risk,
                        'total_trades': results.total_trades,
                        'win_rate': results.win_rate,
                        'total_return': results.total_pnl_pct,
                        'max_drawdown': results.max_drawdown_pct,
                        'sharpe_ratio': results.sharpe_ratio,
                        'profit_factor': results.profit_factor,
                        'score': self._calculate_optimization_score(results)
                    })
                    
                except Exception as e:
                    logger.warning(f"Error with parameters min_rr={min_rr}, max_risk={max_risk}: {e}")
                    continue
        
        # Sort by optimization score
        optimization_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Save optimization results
        await self._save_optimization_results(optimization_results, symbol)
        
        return {
            'symbol': symbol,
            'best_parameters': optimization_results[0] if optimization_results else None,
            'all_results': optimization_results,
            'total_tested': len(optimization_results)
        }
    
    def _calculate_optimization_score(self, results: BacktestResults) -> float:
        """Calculate optimization score for parameter ranking"""
        # Composite score considering multiple factors
        if results.total_trades == 0:
            return 0.0
        
        # Factors (you can adjust weights)
        return_weight = 0.4
        sharpe_weight = 0.3
        drawdown_weight = 0.2
        trade_count_weight = 0.1
        
        # Normalize metrics
        return_score = min(results.total_pnl_pct / 100, 1.0)  # Cap at 100%
        sharpe_score = min(max(results.sharpe_ratio, 0) / 3.0, 1.0)  # Cap at 3.0
        drawdown_score = max(1.0 - (results.max_drawdown_pct / 20), 0.0)  # Penalty for >20% DD
        trade_score = min(results.total_trades / 50, 1.0)  # Prefer more trades up to 50
        
        total_score = (
            return_score * return_weight +
            sharpe_score * sharpe_weight +
            drawdown_score * drawdown_weight +
            trade_score * trade_count_weight
        )
        
        return total_score
    
    # =========================================================================
    # REPORTING AND VISUALIZATION
    # =========================================================================
    
    async def generate_report(self, results: Union[BacktestResults, Dict[str, BacktestResults]],
                             report_type: str = 'detailed') -> str:
        """
        Generate comprehensive backtest report
        
        Args:
            results: Backtest results (single or multi-symbol)
            report_type: 'summary' or 'detailed'
            
        Returns:
            Report content as string
        """
        if isinstance(results, BacktestResults):
            return self._generate_single_report(results, report_type)
        else:
            return self._generate_multi_report(results, report_type)
    
    def _generate_single_report(self, results: BacktestResults, report_type: str) -> str:
        """Generate report for single symbol results"""
        
        report = f"""
# Power of 3 Backtesting Report

## Summary
- **Symbol**: {results.symbol}
- **Period**: {results.start_date.strftime('%Y-%m-%d')} to {results.end_date.strftime('%Y-%m-%d')}
- **Initial Balance**: ${results.initial_balance:,.2f}
- **Final Balance**: ${results.final_balance:,.2f}

## Performance Metrics
- **Total Return**: {results.total_pnl_pct:.2f}%
- **Total P&L**: ${results.total_pnl:,.2f}
- **Total Trades**: {results.total_trades}
- **Win Rate**: {results.win_rate:.1f}%
- **Profit Factor**: {results.profit_factor:.2f}
- **Sharpe Ratio**: {results.sharpe_ratio:.2f}
- **Max Drawdown**: {results.max_drawdown_pct:.2f}%

## Trade Analysis
- **Winning Trades**: {results.winning_trades}
- **Losing Trades**: {results.losing_trades}
- **Average Win**: ${results.avg_win:.2f}
- **Average Loss**: ${results.avg_loss:.2f}
- **Largest Win**: ${results.largest_win:.2f}
- **Largest Loss**: ${results.largest_loss:.2f}

## Power of 3 Metrics
- **Average Signal Quality**: {results.avg_signal_quality:.1f}/10
- **Average Risk/Reward**: 1:{results.avg_risk_reward:.1f}
- **Average Trade Duration**: {results.avg_trade_duration}

## Session Performance
"""
        
        if results.session_performance:
            for session, stats in results.session_performance.items():
                report += f"""
### {session.replace('_', ' ').title()}
- Trades: {stats['trades']}
- Win Rate: {stats['win_rate']:.1f}%
- Total P&L: ${stats['total_pnl']:.2f}
- Avg P&L per Trade: ${stats['avg_pnl']:.2f}
"""
        
        if report_type == 'detailed' and results.trades:
            report += "\n## Trade Details\n"
            report += "| Entry Time | Type | Entry | Exit | P&L | Quality |\n"
            report += "|------------|------|-------|------|-----|----------|\n"
            
            for trade in results.trades[-20:]:  # Last 20 trades
                entry_time = trade.entry_time.strftime('%Y-%m-%d %H:%M') if trade.entry_time else 'N/A'
                exit_price = f"{trade.exit_price:.2f}" if trade.exit_price else 'Open'
                report += f"| {entry_time} | {trade.signal_type.upper()} | {trade.entry_price:.2f} | {exit_price} | ${trade.pnl:.2f} | {trade.quality_score:.1f} |\n"
        
        return report
    
    def _generate_multi_report(self, results: Dict[str, BacktestResults], report_type: str) -> str:
        """Generate report for multi-symbol results"""
        
        report = f"""
# Multi-Symbol Power of 3 Backtesting Report

## Summary
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Symbol Comparison
"""
        
        # Create comparison table
        report += "| Symbol | Total Return | Win Rate | Trades | Sharpe | Max DD | Profit Factor |\n"
        report += "|--------|--------------|----------|--------|--------|--------|---------------|\n"
        
        for symbol, result in results.items():
            report += f"| {symbol} | {result.total_pnl_pct:.2f}% | {result.win_rate:.1f}% | {result.total_trades} | {result.sharpe_ratio:.2f} | {result.max_drawdown_pct:.2f}% | {result.profit_factor:.2f} |\n"
        
        # Best performers
        if results:
            best_return = max(results.items(), key=lambda x: x[1].total_pnl_pct)
            best_sharpe = max(results.items(), key=lambda x: x[1].sharpe_ratio)
            best_winrate = max(results.items(), key=lambda x: x[1].win_rate)
            
            report += f"""
## Best Performers
- **Highest Return**: {best_return[0]} ({best_return[1].total_pnl_pct:.2f}%)
- **Best Sharpe Ratio**: {best_sharpe[0]} ({best_sharpe[1].sharpe_ratio:.2f})
- **Highest Win Rate**: {best_winrate[0]} ({best_winrate[1].win_rate:.1f}%)
"""
        
        if report_type == 'detailed':
            report += "\n## Detailed Results by Symbol\n"
            for symbol, result in results.items():
                report += f"\n### {symbol}\n"
                report += self._generate_single_report(result, 'summary')
        
        return report
    
    async def create_visualizations(self, results: Union[BacktestResults, Dict[str, BacktestResults]],
                                   save_path: Optional[str] = None) -> List[str]:
        """
        Create visualization charts for backtest results
        
        Args:
            results: Backtest results
            save_path: Path to save charts (optional)
            
        Returns:
            List of created chart file paths
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available. Skipping visualizations.")
            return []
        
        chart_paths = []
        
        if isinstance(results, BacktestResults):
            chart_paths.extend(await self._create_single_charts(results, save_path))
        else:
            chart_paths.extend(await self._create_multi_charts(results, save_path))
        
        return chart_paths
    
    async def _create_single_charts(self, results: BacktestResults, save_path: Optional[str]) -> List[str]:
        """Create charts for single symbol results"""
        chart_paths = []
        
        # 1. Equity Curve
        if results.equity_curve:
            if PLOTTING_AVAILABLE:
                fig, ax = plt.subplots(figsize=(12, 6))
            else:
                raise ImportError("Matplotlib is not available. Install it with: pip install matplotlib")
            
            timestamps = [datetime.fromisoformat(point['timestamp']) for point in results.equity_curve]
            equity_values = [point['total_equity'] for point in results.equity_curve]
            
            ax.plot(np.array(timestamps), equity_values, linewidth=2, color='blue')
            ax.axhline(y=results.initial_balance, color='gray', linestyle='--', alpha=0.7)
            
            ax.set_title(f'{results.symbol} - Equity Curve', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Account Value ($)')
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            if PLOTTING_AVAILABLE:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            else:
                raise ImportError("Matplotlib is not available. Install it with: pip install matplotlib")
            ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                chart_path = f"{save_path}_{results.symbol}_equity_curve.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                chart_paths.append(chart_path)
            
            plt.close()
        
        # 2. P&L Distribution
        if results.trades:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # P&L histogram
            pnl_values = [trade.pnl for trade in results.trades]
            ax1.hist(pnl_values, bins=20, alpha=0.7, color='green' if sum(pnl_values) > 0 else 'red')
            ax1.axvline(x=0, color='black', linestyle='--', alpha=0.7)
            ax1.set_title('P&L Distribution', fontweight='bold')
            ax1.set_xlabel('P&L ($)')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            
            # Cumulative P&L
            cumulative_pnl = np.cumsum(pnl_values)
            ax2.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7)
            ax2.set_title('Cumulative P&L', fontweight='bold')
            ax2.set_xlabel('Trade Number')
            ax2.set_ylabel('Cumulative P&L ($)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                chart_path = f"{save_path}_{results.symbol}_pnl_analysis.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                chart_paths.append(chart_path)
            
            plt.close()
        
        return chart_paths
    
    async def _create_multi_charts(self, results: Dict[str, BacktestResults], save_path: Optional[str]) -> List[str]:
        """Create charts for multi-symbol results"""
        chart_paths = []
        
        if not results:
            return chart_paths
        
        # Symbol comparison chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        symbols = list(results.keys())
        returns = [results[symbol].total_pnl_pct for symbol in symbols]
        win_rates = [results[symbol].win_rate for symbol in symbols]
        sharpe_ratios = [results[symbol].sharpe_ratio for symbol in symbols]
        max_drawdowns = [results[symbol].max_drawdown_pct for symbol in symbols]
        
        # Returns comparison
        colors = ['green' if r > 0 else 'red' for r in returns]
        ax1.bar(symbols, returns, color=colors, alpha=0.7)
        ax1.set_title('Total Returns by Symbol', fontweight='bold')
        ax1.set_ylabel('Return (%)')
        ax1.grid(True, alpha=0.3)
        
        # Win rates
        ax2.bar(symbols, win_rates, color='blue', alpha=0.7)
        ax2.set_title('Win Rates by Symbol', fontweight='bold')
        ax2.set_ylabel('Win Rate (%)')
        ax2.grid(True, alpha=0.3)
        
        # Sharpe ratios
        ax3.bar(symbols, sharpe_ratios, color='orange', alpha=0.7)
        ax3.set_title('Sharpe Ratios by Symbol', fontweight='bold')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.grid(True, alpha=0.3)
        
        # Max drawdowns
        ax4.bar(symbols, max_drawdowns, color='red', alpha=0.7)
        ax4.set_title('Maximum Drawdowns by Symbol', fontweight='bold')
        ax4.set_ylabel('Max Drawdown (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            chart_path = f"{save_path}_multi_symbol_comparison.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            chart_paths.append(chart_path)
        
        plt.close()
        
        return chart_paths
    
    # =========================================================================
    # FILE I/O
    # =========================================================================
    
    async def _save_results(self, results: BacktestResults, filename_prefix: str):
        """Save backtest results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"{filename_prefix}_{results.symbol}_{timestamp}"
        
        # Save JSON
        json_path = self.results_dir / f"{base_filename}.json"
        with open(json_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        
        # Save CSV (trades)
        csv_path = None
        if results.trades:
            csv_path = self.results_dir / f"{base_filename}_trades.csv"
            trades_df = pd.DataFrame([trade.to_dict() for trade in results.trades])
            trades_df.to_csv(csv_path, index=False)
        
        # Save report
        report = await self.generate_report(results, 'detailed')
        report_path = self.results_dir / f"{base_filename}_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Create visualizations
        chart_paths = await self.create_visualizations(results, str(self.results_dir / base_filename))
        
        logger.info(f"📁 Results saved:")
        logger.info(f"  📄 JSON: {json_path}")
        logger.info(f"  📊 CSV: {csv_path if results.trades else 'No trades to save'}")
        logger.info(f"  📋 Report: {report_path}")
        for chart_path in chart_paths:
            logger.info(f"  📈 Chart: {chart_path}")
    
    async def _save_multi_results(self, results: Dict[str, BacktestResults], filename_prefix: str):
        """Save multi-symbol results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"{filename_prefix}_{timestamp}"
        
        # Save combined JSON
        combined_results = {symbol: result.to_dict() for symbol, result in results.items()}
        json_path = self.results_dir / f"{base_filename}.json"
        with open(json_path, 'w') as f:
            json.dump(combined_results, f, indent=2, default=str)
        
        # Save summary CSV
        summary_data = []
        for symbol, result in results.items():
            summary_data.append({
                'symbol': symbol,
                'total_return_pct': result.total_pnl_pct,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown_pct': result.max_drawdown_pct,
                'avg_signal_quality': result.avg_signal_quality
            })
        
        summary_df = pd.DataFrame(summary_data)
        csv_path = self.results_dir / f"{base_filename}_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        
        # Save report
        report = await self.generate_report(results, 'detailed')
        report_path = self.results_dir / f"{base_filename}_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Create visualizations
        chart_paths = await self.create_visualizations(results, str(self.results_dir / base_filename))
        
        logger.info(f"📁 Multi-symbol results saved:")
        logger.info(f"  📄 JSON: {json_path}")
        logger.info(f"  📊 Summary CSV: {csv_path}")
        logger.info(f"  📋 Report: {report_path}")
        for chart_path in chart_paths:
            logger.info(f"  📈 Chart: {chart_path}")
    
    async def _save_optimization_results(self, results: List[Dict], symbol: str):
        """Save parameter optimization results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"optimization_{symbol}_{timestamp}"
        
        # Save CSV
        csv_path = self.results_dir / f"{filename}.csv"
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        
        # Save JSON
        json_path = self.results_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📁 Optimization results saved:")
        logger.info(f"  📊 CSV: {csv_path}")
        logger.info(f"  📄 JSON: {json_path}")

# =========================================================================
# COMMAND LINE INTERFACE
# =========================================================================

class CLI:
    """Command-line interface for backtesting"""
    
    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        """Create command-line argument parser"""
        parser = argparse.ArgumentParser(
            description='Power of 3 Backtesting System',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Single symbol backtest
  python -m power_of_3.backtesting.interface single --symbol US30 --start 2024-01-01 --end 2024-03-31

  # Multi-symbol backtest
  python -m power_of_3.backtesting.interface multi --symbols US30 NAS100 SPX500 --start 2024-01-01 --end 2024-03-31

  # Parameter optimization
  python -m power_of_3.backtesting.interface optimize --symbol US30 --start 2024-01-01 --end 2024-03-31
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Backtest type')
        
        # Single symbol command
        single_parser = subparsers.add_parser('single', help='Single symbol backtest')
        single_parser.add_argument('--symbol', required=True, help='Trading symbol')
        single_parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
        single_parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
        single_parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')
        single_parser.add_argument('--timeframe', default='5min', help='Data timeframe')
        
        # Multi-symbol command
        multi_parser = subparsers.add_parser('multi', help='Multi-symbol backtest')
        multi_parser.add_argument('--symbols', nargs='+', required=True, help='Trading symbols')
        multi_parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
        multi_parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
        multi_parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')
        multi_parser.add_argument('--timeframe', default='5min', help='Data timeframe')
        
        # Optimization command
        opt_parser = subparsers.add_parser('optimize', help='Parameter optimization')
        opt_parser.add_argument('--symbol', required=True, help='Trading symbol')
        opt_parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
        opt_parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
        opt_parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')
        opt_parser.add_argument('--timeframe', default='5min', help='Data timeframe')
        
        return parser
    
    @staticmethod
    async def run_cli():
        """Run command-line interface"""
        parser = CLI.create_parser()
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return
        
        # Initialize interface
        interface = BacktestInterface()
        
        try:
            if args.command == 'single':
                print(f"🚀 Running single symbol backtest: {args.symbol}")
                results = await interface.run_single_symbol_backtest(
                    symbol=args.symbol,
                    start_date=args.start,
                    end_date=args.end,
                    initial_balance=args.balance,
                    timeframe=args.timeframe
                )
                
                print("\n" + "="*60)
                print("📊 BACKTEST RESULTS")
                print("="*60)
                print(f"Symbol: {results.symbol}")
                print(f"Total Return: {results.total_pnl_pct:.2f}%")
                print(f"Total Trades: {results.total_trades}")
                print(f"Win Rate: {results.win_rate:.1f}%")
                print(f"Profit Factor: {results.profit_factor:.2f}")
                print(f"Max Drawdown: {results.max_drawdown_pct:.2f}%")
                print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
                
            elif args.command == 'multi':
                print(f"🚀 Running multi-symbol backtest: {', '.join(args.symbols)}")
                results = await interface.run_multi_symbol_backtest(
                    symbols=args.symbols,
                    start_date=args.start,
                    end_date=args.end,
                    initial_balance=args.balance,
                    timeframe=args.timeframe
                )
                
                print("\n" + "="*80)
                print("📊 MULTI-SYMBOL BACKTEST RESULTS")
                print("="*80)
                
                for symbol, result in results.items():
                    print(f"\n{symbol}:")
                    print(f"  Return: {result.total_pnl_pct:.2f}% | "
                          f"Trades: {result.total_trades} | "
                          f"Win Rate: {result.win_rate:.1f}% | "
                          f"Sharpe: {result.sharpe_ratio:.2f}")
                
            elif args.command == 'optimize':
                print(f"🔧 Running parameter optimization: {args.symbol}")
                
                # Define parameter ranges
                parameter_ranges = {
                    'min_risk_reward': [3.0, 4.0, 5.0, 6.0],
                    'max_risk_percent': [1.0, 1.5, 2.0, 2.5]
                }
                
                results = await interface.run_parameter_optimization(
                    symbol=args.symbol,
                    start_date=args.start,
                    end_date=args.end,
                    parameter_ranges=parameter_ranges,
                    initial_balance=args.balance,
                    timeframe=args.timeframe
                )
                
                print("\n" + "="*60)
                print("🔧 OPTIMIZATION RESULTS")
                print("="*60)
                
                if results['best_parameters']:
                    best = results['best_parameters']
                    print(f"Best Parameters:")
                    print(f"  Min Risk/Reward: {best['min_risk_reward']}")
                    print(f"  Max Risk %: {best['max_risk_percent']}%")
                    print(f"  Total Return: {best['total_return']:.2f}%")
                    print(f"  Win Rate: {best['win_rate']:.1f}%")
                    print(f"  Sharpe Ratio: {best['sharpe_ratio']:.2f}")
                    print(f"  Optimization Score: {best['score']:.3f}")
                
                print(f"\nTested {results['total_tested']} parameter combinations")
        
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"CLI error: {e}")
            sys.exit(1)
            
    def _standardize_data_columns(self, df):
        """Standardize column names for compatibility"""
        if df.empty:
            return df
            
        column_mapping = {
            'Open': 'open', 
            'High': 'high', 
            'Low': 'low', 
            'Close': 'close', 
            'Volume': 'volume'
        }
        
        columns_to_rename = {old: new for old, new in column_mapping.items() if old in df.columns}
        if columns_to_rename:
            df = df.rename(columns=columns_to_rename)
            logger.info(f"Standardized columns: {list(columns_to_rename.keys())}")
        
        return df

# Entry point for module execution
if __name__ == "__main__":
    asyncio.run(CLI.run_cli())