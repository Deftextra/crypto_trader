"""
Backtesting Framework for Cryptocurrency Trading Strategies

This module provides comprehensive backtesting capabilities with detailed
performance analysis, visualization, and risk metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Analyze trading strategy performance"""
    
    def __init__(self):
        pass
    
    def calculate_returns(self, portfolio_values: pd.Series) -> pd.Series:
        """Calculate portfolio returns"""
        return portfolio_values.pct_change().fillna(0)
    
    def calculate_cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """Calculate cumulative returns"""
        return (1 + returns).cumprod() - 1
    
    def calculate_drawdown(self, portfolio_values: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        return drawdown
    
    def calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        drawdown = self.calculate_drawdown(portfolio_values)
        return drawdown.min()
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Portfolio returns
            risk_free_rate: Annual risk-free rate
        
        Returns:
            Annualized Sharpe ratio
        """
        if returns.std() == 0:
            return 0.0
        
        # Convert to appropriate frequency (assuming hourly data)
        annual_return = returns.mean() * 24 * 365
        annual_volatility = returns.std() * np.sqrt(24 * 365)
        annual_risk_free = risk_free_rate
        
        return (annual_return - annual_risk_free) / annual_volatility
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if returns.mean() > 0 else 0.0
        
        annual_return = returns.mean() * 24 * 365
        annual_downside_volatility = downside_returns.std() * np.sqrt(24 * 365)
        
        return (annual_return - risk_free_rate) / annual_downside_volatility
    
    def calculate_calmar_ratio(self, returns: pd.Series, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        if max_drawdown == 0:
            return float('inf') if returns.mean() > 0 else 0.0
        
        annual_return = returns.mean() * 24 * 365
        return annual_return / abs(max_drawdown)
    
    def calculate_win_rate(self, trades_pnl: List[float]) -> float:
        """Calculate win rate from trade PnLs"""
        if not trades_pnl:
            return 0.0
        
        winning_trades = [pnl for pnl in trades_pnl if pnl > 0]
        return len(winning_trades) / len(trades_pnl)
    
    def calculate_profit_factor(self, trades_pnl: List[float]) -> float:
        """Calculate profit factor"""
        winning_trades = [pnl for pnl in trades_pnl if pnl > 0]
        losing_trades = [pnl for pnl in trades_pnl if pnl < 0]
        
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility"""
        return returns.std() * np.sqrt(24 * 365)
    
    def calculate_var(self, returns: pd.Series, confidence: float = 0.05) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, confidence * 100)
    
    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self.calculate_var(returns, confidence)
        return returns[returns <= var].mean()


class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 commission: float = 0.001,
                 slippage: float = 0.001):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital: Starting capital
            commission: Trading commission (fraction)
            slippage: Price slippage (fraction)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.analyzer = PerformanceAnalyzer()
    
    def run_backtest(self, 
                    strategy, 
                    data: pd.DataFrame, 
                    features: pd.DataFrame,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a complete backtest
        
        Args:
            strategy: Trading strategy instance
            data: Price data
            features: Feature data
            start_date: Start date for backtest (optional)
            end_date: End date for backtest (optional)
        
        Returns:
            Dictionary containing all backtest results
        """
        logger.info("Starting backtest...")
        
        # Filter data by date range if specified
        if start_date or end_date:
            data, features = self._filter_by_date(data, features, start_date, end_date)
        
        # Run strategy
        results = strategy.run_strategy(data, features, self.initial_capital)
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(results, strategy)
        
        # Create comprehensive results
        backtest_results = {
            'results': results,
            'performance': performance,
            'strategy': strategy,
            'data': data,
            'features': features,
            'config': {
                'initial_capital': self.initial_capital,
                'commission': self.commission,
                'slippage': self.slippage
            }
        }
        
        logger.info("Backtest completed successfully")
        return backtest_results
    
    def _filter_by_date(self, 
                       data: pd.DataFrame, 
                       features: pd.DataFrame,
                       start_date: Optional[str],
                       end_date: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter data by date range"""
        if 'timestamp' in data.columns:
            mask = pd.Series([True] * len(data))
            
            if start_date:
                mask &= data['timestamp'] >= pd.to_datetime(start_date)
            
            if end_date:
                mask &= data['timestamp'] <= pd.to_datetime(end_date)
            
            filtered_data = data[mask].reset_index(drop=True)
            filtered_features = features[mask].reset_index(drop=True)
            
            return filtered_data, filtered_features
        
        return data, features
    
    def _calculate_performance_metrics(self, 
                                     results: pd.DataFrame, 
                                     strategy) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        portfolio_values = results['portfolio_value']
        returns = self.analyzer.calculate_returns(portfolio_values)
        
        # Basic metrics
        total_return = (portfolio_values.iloc[-1] - self.initial_capital) / self.initial_capital
        annual_return = returns.mean() * 24 * 365
        
        # Risk metrics
        volatility = self.analyzer.calculate_volatility(returns)
        max_drawdown = self.analyzer.calculate_max_drawdown(portfolio_values)
        sharpe_ratio = self.analyzer.calculate_sharpe_ratio(returns)
        sortino_ratio = self.analyzer.calculate_sortino_ratio(returns)
        calmar_ratio = self.analyzer.calculate_calmar_ratio(returns, max_drawdown)
        var_5 = self.analyzer.calculate_var(returns, 0.05)
        cvar_5 = self.analyzer.calculate_cvar(returns, 0.05)
        
        # Trading metrics
        strategy_stats = strategy.get_strategy_stats()
        
        performance = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'var_5': var_5,
            'cvar_5': cvar_5,
            'final_portfolio_value': portfolio_values.iloc[-1],
            **strategy_stats
        }
        
        return performance
    
    def compare_strategies(self, 
                          strategies: List,
                          strategy_names: List[str],
                          data: pd.DataFrame,
                          features: pd.DataFrame) -> pd.DataFrame:
        """
        Compare multiple strategies
        
        Args:
            strategies: List of strategy instances
            strategy_names: Names for each strategy
            data: Price data
            features: Feature data
        
        Returns:
            DataFrame comparing strategy performance
        """
        comparison_results = []
        
        for strategy, name in zip(strategies, strategy_names):
            logger.info(f"Running backtest for {name}")
            
            backtest_result = self.run_backtest(strategy, data, features)
            performance = backtest_result['performance']
            performance['strategy_name'] = name
            comparison_results.append(performance)
        
        comparison_df = pd.DataFrame(comparison_results)
        return comparison_df
    
    def monte_carlo_simulation(self,
                             strategy,
                             data: pd.DataFrame,
                             features: pd.DataFrame,
                             n_simulations: int = 1000,
                             noise_std: float = 0.01) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation on strategy
        
        Args:
            strategy: Trading strategy
            data: Price data
            features: Feature data
            n_simulations: Number of simulations
            noise_std: Standard deviation of noise to add
        
        Returns:
            Monte Carlo results
        """
        logger.info(f"Running Monte Carlo simulation with {n_simulations} iterations")
        
        results = []
        
        for i in range(n_simulations):
            if i % 100 == 0:
                logger.info(f"Simulation {i+1}/{n_simulations}")
            
            # Add noise to prices
            noisy_data = data.copy()
            price_noise = np.random.normal(0, noise_std, len(data))
            noisy_data['close'] *= (1 + price_noise)
            noisy_data['high'] *= (1 + price_noise)
            noisy_data['low'] *= (1 + price_noise)
            noisy_data['open'] *= (1 + price_noise)
            
            # Run backtest
            backtest_result = self.run_backtest(strategy, noisy_data, features)
            results.append(backtest_result['performance'])
        
        # Aggregate results
        metrics_df = pd.DataFrame(results)
        
        monte_carlo_results = {
            'mean_metrics': metrics_df.mean(),
            'std_metrics': metrics_df.std(),
            'percentiles': metrics_df.quantile([0.05, 0.25, 0.5, 0.75, 0.95]),
            'all_results': results
        }
        
        logger.info("Monte Carlo simulation completed")
        return monte_carlo_results


class BacktestVisualizer:
    """Create visualizations for backtest results"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
    
    def plot_portfolio_value(self, results: pd.DataFrame, save_path: Optional[str] = None):
        """Plot portfolio value over time"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(results.index, results['portfolio_value'], label='Portfolio Value', linewidth=2)
        ax.axhline(y=results['portfolio_value'].iloc[0], color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
        
        ax.set_title('Portfolio Value Over Time', fontsize=16)
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_returns_distribution(self, results: pd.DataFrame, save_path: Optional[str] = None):
        """Plot distribution of returns"""
        returns = results['portfolio_value'].pct_change().fillna(0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Histogram
        ax1.hist(returns, bins=50, alpha=0.7, edgecolor='black')
        ax1.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
        ax1.set_title('Returns Distribution')
        ax1.set_xlabel('Returns')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal Distribution)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_drawdown(self, results: pd.DataFrame, save_path: Optional[str] = None):
        """Plot drawdown over time"""
        portfolio_values = results['portfolio_value']
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.fill_between(results.index, drawdown, 0, alpha=0.3, color='red')
        ax.plot(results.index, drawdown, color='red', linewidth=1)
        
        ax.set_title('Portfolio Drawdown', fontsize=16)
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
        # Highlight maximum drawdown
        max_dd_idx = drawdown.idxmin()
        ax.scatter(max_dd_idx, drawdown[max_dd_idx], color='darkred', s=100, zorder=5)
        ax.annotate(f'Max DD: {drawdown.min():.2%}', 
                   xy=(max_dd_idx, drawdown[max_dd_idx]),
                   xytext=(10, 10), textcoords='offset points')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_trades(self, results: pd.DataFrame, strategy, save_path: Optional[str] = None):
        """Plot trades on price chart"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot price
        ax.plot(results.index, results['price'], label='Price', linewidth=1, alpha=0.7)
        
        # Plot buy/sell signals
        buy_signals = results[results['signal'] == 1]
        sell_signals = results[results['signal'] == -1]
        
        ax.scatter(buy_signals.index, buy_signals['price'], color='green', marker='^', s=100, label='Buy Signal')
        ax.scatter(sell_signals.index, sell_signals['price'], color='red', marker='v', s=100, label='Sell Signal')
        
        ax.set_title('Price Chart with Trading Signals', fontsize=16)
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_comparison(self, comparison_df: pd.DataFrame, save_path: Optional[str] = None):
        """Plot comparison of multiple strategies"""
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                ax = axes[i]
                comparison_df.plot(x='strategy_name', y=metric, kind='bar', ax=ax, legend=False)
                ax.set_title(metric.replace('_', ' ').title())
                ax.set_xlabel('')
                plt.setp(ax.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_performance_report(self, backtest_results: Dict[str, Any], save_path: Optional[str] = None):
        """Create comprehensive performance report"""
        results = backtest_results['results']
        performance = backtest_results['performance']
        
        fig = plt.figure(figsize=(20, 15))
        
        # Portfolio value
        ax1 = plt.subplot(3, 3, (1, 2))
        ax1.plot(results['portfolio_value'], linewidth=2)
        ax1.set_title('Portfolio Value')
        ax1.grid(True, alpha=0.3)
        
        # Returns distribution
        ax2 = plt.subplot(3, 3, 3)
        returns = results['portfolio_value'].pct_change().fillna(0)
        ax2.hist(returns, bins=30, alpha=0.7)
        ax2.set_title('Returns Distribution')
        
        # Drawdown
        ax3 = plt.subplot(3, 3, (4, 5))
        portfolio_values = results['portfolio_value']
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        ax3.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        ax3.set_title('Drawdown')
        
        # Performance metrics table
        ax4 = plt.subplot(3, 3, 6)
        ax4.axis('off')
        
        metrics_text = f"""
        Performance Metrics:
        
        Total Return: {performance['total_return']:.2%}
        Annual Return: {performance['annual_return']:.2%}
        Volatility: {performance['volatility']:.2%}
        Sharpe Ratio: {performance['sharpe_ratio']:.2f}
        Max Drawdown: {performance['max_drawdown']:.2%}
        
        Trading Metrics:
        Total Trades: {performance.get('total_trades', 0)}
        Win Rate: {performance.get('win_rate', 0):.2%}
        Profit Factor: {performance.get('profit_factor', 0):.2f}
        """
        
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        # Price with signals
        ax5 = plt.subplot(3, 3, (7, 8))
        ax5.plot(results['price'], alpha=0.7, linewidth=1)
        buy_signals = results[results['signal'] == 1]
        sell_signals = results[results['signal'] == -1]
        ax5.scatter(buy_signals.index, buy_signals['price'], color='green', marker='^', s=50)
        ax5.scatter(sell_signals.index, sell_signals['price'], color='red', marker='v', s=50)
        ax5.set_title('Price with Signals')
        
        # Rolling Sharpe ratio
        ax6 = plt.subplot(3, 3, 9)
        rolling_returns = returns.rolling(window=24*7).mean()  # Weekly rolling
        rolling_std = returns.rolling(window=24*7).std()
        rolling_sharpe = rolling_returns / rolling_std * np.sqrt(24*365)
        ax6.plot(rolling_sharpe, linewidth=1)
        ax6.set_title('Rolling Sharpe Ratio (7 days)')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    from data.collector import CoinbaseDataCollector
    from utils.features import FeatureEngineer
    from models.linear_model import CryptoLinearModel
    from strategies.linear_strategy import LinearRegressionStrategy, RiskManager
    
    # Collect data
    collector = CoinbaseDataCollector()
    feature_engineer = FeatureEngineer()
    
    btc_data = collector.get_recent_data('BTC-USD', days=30)
    
    if not btc_data.empty:
        # Engineer features
        df_features = feature_engineer.engineer_all_features(btc_data)
        X, y = feature_engineer.prepare_ml_data(df_features)
        
        # Train model
        model = CryptoLinearModel(model_type='ridge', feature_selection='kbest', n_features=15)
        model.fit(X, y)
        
        # Create strategy
        risk_manager = RiskManager(
            max_position_size=0.1,
            stop_loss_pct=0.02,
            take_profit_pct=0.04
        )
        
        strategy = LinearRegressionStrategy(
            model=model,
            risk_manager=risk_manager,
            prediction_threshold=0.5
        )
        
        # Run backtest
        backtest_engine = BacktestEngine(initial_capital=10000)
        backtest_results = backtest_engine.run_backtest(
            strategy, 
            btc_data, 
            X.reset_index(drop=True)
        )
        
        # Print results
        performance = backtest_results['performance']
        print("Backtest Results:")
        print(f"Total Return: {performance['total_return']:.2%}")
        print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
        print(f"Total Trades: {performance.get('total_trades', 0)}")
        
        # Create visualizations
        visualizer = BacktestVisualizer()
        visualizer.create_performance_report(backtest_results)
