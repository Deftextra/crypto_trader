#!/usr/bin/env python3
"""
Cryptocurrency Linear Regression Trading System

This is the main application that integrates data collection, feature engineering,
model training, strategy execution, and backtesting for cryptocurrency trading
using linear regression.

Usage:
    python main.py --mode backtest --symbol BTC-USD --days 60
    python main.py --mode train --symbol ETH-USD --days 90
    python main.py --mode predict --symbol BTC-USD
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import project modules
from data.collector import CoinbaseDataCollector, get_popular_crypto_pairs
from utils.features import FeatureEngineer
from models.linear_model import CryptoLinearModel, ModelEnsemble
from strategies.linear_strategy import LinearRegressionStrategy, RiskManager
from backtesting.backtest import BacktestEngine, BacktestVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CryptoTradingSystem:
    """Main cryptocurrency trading system"""
    
    def __init__(self, config: dict = None):
        """
        Initialize the trading system
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.data_collector = CoinbaseDataCollector()
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.strategy = None
        
        # Create models directory if it doesn't exist
        self.models_dir = project_root / 'models' / 'saved'
        self.models_dir.mkdir(exist_ok=True)
        
        logger.info("Crypto Trading System initialized")
    
    def _get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            # Data parameters
            'symbol': 'BTC-USD',
            'days': 60,
            'granularity': 3600,  # 1 hour
            
            # Model parameters
            'model_type': 'ridge',
            'feature_selection': 'kbest',
            'n_features': 15,
            'target_hours': 1,
            
            # Strategy parameters
            'prediction_threshold': 0.5,
            'lookback_periods': 5,
            
            # Risk management
            'initial_capital': 10000,
            'max_position_size': 0.1,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'max_drawdown_pct': 0.1,
            'min_prediction_confidence': 0.01,
            
            # Backtesting
            'commission': 0.001,
            'slippage': 0.001
        }
    
    def collect_data(self, symbol: str = None, days: int = None) -> pd.DataFrame:
        """
        Collect historical data
        
        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            days: Number of days of historical data
        
        Returns:
            DataFrame with OHLCV data
        """
        symbol = symbol or self.config['symbol']
        days = days or self.config['days']
        
        logger.info(f"Collecting {days} days of data for {symbol}")
        
        data = self.data_collector.get_recent_data(
            symbol, 
            days=days, 
            granularity=self.config['granularity']
        )
        
        if data.empty:
            raise ValueError(f"No data collected for {symbol}")
        
        logger.info(f"Collected {len(data)} data points")
        return data
    
    def prepare_features(self, data: pd.DataFrame) -> tuple:
        """
        Engineer features from raw data
        
        Args:
            data: Raw OHLCV data
        
        Returns:
            Tuple of (features_df, target_series)
        """
        logger.info("Engineering features...")
        
        # Engineer all features
        df_features = self.feature_engineer.engineer_all_features(
            data, 
            target_hours=self.config['target_hours']
        )
        
        # Prepare ML data
        X, y = self.feature_engineer.prepare_ml_data(df_features)
        
        logger.info(f"Features prepared: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, save_model: bool = True) -> CryptoLinearModel:
        """
        Train the linear regression model
        
        Args:
            X: Feature matrix
            y: Target vector
            save_model: Whether to save the trained model
        
        Returns:
            Trained model
        """
        logger.info("Training linear regression model...")
        
        # Initialize model
        model = CryptoLinearModel(
            model_type=self.config['model_type'],
            feature_selection=self.config['feature_selection'],
            n_features=self.config['n_features']
        )
        
        # Train model
        metrics = model.fit(X, y)
        
        # Log training results
        logger.info("Training completed:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save model
        if save_model:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.models_dir / f"{self.config['symbol']}_{timestamp}.joblib"
            model.save_model(str(model_path))
            logger.info(f"Model saved to {model_path}")
        
        self.model = model
        return model
    
    def load_model(self, model_path: str) -> CryptoLinearModel:
        """Load a trained model"""
        logger.info(f"Loading model from {model_path}")
        
        model = CryptoLinearModel()
        model.load_model(model_path)
        
        self.model = model
        return model
    
    def create_strategy(self, model: CryptoLinearModel = None) -> LinearRegressionStrategy:
        """
        Create trading strategy
        
        Args:
            model: Trained model (uses self.model if None)
        
        Returns:
            Trading strategy instance
        """
        model = model or self.model
        if model is None:
            raise ValueError("No trained model available")
        
        # Create risk manager
        risk_manager = RiskManager(
            max_position_size=self.config['max_position_size'],
            stop_loss_pct=self.config['stop_loss_pct'],
            take_profit_pct=self.config['take_profit_pct'],
            max_drawdown_pct=self.config['max_drawdown_pct'],
            min_prediction_confidence=self.config['min_prediction_confidence']
        )
        
        # Create strategy
        strategy = LinearRegressionStrategy(
            model=model,
            risk_manager=risk_manager,
            prediction_threshold=self.config['prediction_threshold'],
            lookback_periods=self.config['lookback_periods']
        )
        
        self.strategy = strategy
        return strategy
    
    def run_backtest(self, 
                    data: pd.DataFrame, 
                    features: pd.DataFrame,
                    strategy: LinearRegressionStrategy = None) -> dict:
        """
        Run backtest on historical data
        
        Args:
            data: Price data
            features: Feature data
            strategy: Trading strategy (uses self.strategy if None)
        
        Returns:
            Backtest results
        """
        strategy = strategy or self.strategy
        if strategy is None:
            raise ValueError("No strategy available")
        
        logger.info("Running backtest...")
        
        # Create backtest engine
        backtest_engine = BacktestEngine(
            initial_capital=self.config['initial_capital'],
            commission=self.config['commission'],
            slippage=self.config['slippage']
        )
        
        # Run backtest
        results = backtest_engine.run_backtest(strategy, data, features)
        
        # Log results
        performance = results['performance']
        logger.info("Backtest Results:")
        logger.info(f"  Total Return: {performance['total_return']:.2%}")
        logger.info(f"  Annual Return: {performance['annual_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {performance['max_drawdown']:.2%}")
        logger.info(f"  Total Trades: {performance.get('total_trades', 0)}")
        logger.info(f"  Win Rate: {performance.get('win_rate', 0):.2%}")
        
        return results
    
    def generate_prediction(self, symbol: str = None) -> dict:
        """
        Generate price prediction for current market conditions
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Prediction results
        """
        if self.model is None:
            raise ValueError("No trained model available")
        
        symbol = symbol or self.config['symbol']
        
        logger.info(f"Generating prediction for {symbol}")
        
        # Get recent data
        data = self.collect_data(symbol, days=7)  # Get 7 days for features
        X, _ = self.prepare_features(data)
        
        # Generate prediction
        latest_features = X.tail(1)
        prediction = self.model.predict(latest_features)[0]
        
        # Get current price
        current_price = data.iloc[-1]['close']
        
        # Calculate predicted price
        predicted_return = prediction
        predicted_price = current_price * (1 + predicted_return / 100)
        
        prediction_results = {
            'symbol': symbol,
            'current_price': current_price,
            'predicted_return': predicted_return,
            'predicted_price': predicted_price,
            'timestamp': data.iloc[-1]['timestamp'],
            'confidence': abs(predicted_return)
        }
        
        logger.info(f"Prediction Results:")
        logger.info(f"  Current Price: ${current_price:.2f}")
        logger.info(f"  Predicted Return: {predicted_return:.2f}%")
        logger.info(f"  Predicted Price: ${predicted_price:.2f}")
        logger.info(f"  Confidence: {abs(predicted_return):.2f}%")
        
        return prediction_results
    
    def create_visualizations(self, backtest_results: dict, output_dir: str = None):
        """Create and save visualizations"""
        output_dir = Path(output_dir) if output_dir else project_root / 'output'
        output_dir.mkdir(exist_ok=True)
        
        visualizer = BacktestVisualizer()
        
        # Create comprehensive report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"backtest_report_{self.config['symbol']}_{timestamp}.png"
        
        try:
            visualizer.create_performance_report(backtest_results, str(report_path))
            logger.info(f"Visualization saved to {report_path}")
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def compare_models(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Compare different model configurations"""
        logger.info("Comparing different model configurations...")
        
        model_configs = [
            {'model_type': 'linear', 'feature_selection': 'none'},
            {'model_type': 'ridge', 'feature_selection': 'kbest', 'n_features': 10},
            {'model_type': 'ridge', 'feature_selection': 'kbest', 'n_features': 20},
            {'model_type': 'lasso', 'feature_selection': 'none'},
            {'model_type': 'elastic', 'feature_selection': 'kbest', 'n_features': 15}
        ]
        
        strategies = []
        strategy_names = []
        
        for config in model_configs:
            # Train model with this configuration
            model = CryptoLinearModel(**config)
            model.fit(features, features['target_return'] if 'target_return' in features else features.iloc[:, -1])
            
            # Create strategy
            strategy = self.create_strategy(model)
            strategies.append(strategy)
            
            name = f"{config['model_type']}"
            if config.get('feature_selection') != 'none':
                name += f"_{config.get('n_features', 'all')}"
            strategy_names.append(name)
        
        # Compare strategies
        backtest_engine = BacktestEngine(
            initial_capital=self.config['initial_capital'],
            commission=self.config['commission'],
            slippage=self.config['slippage']
        )
        
        comparison_results = backtest_engine.compare_strategies(
            strategies, strategy_names, data, features
        )
        
        return comparison_results


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='Cryptocurrency Linear Regression Trading System')
    parser.add_argument('--mode', choices=['train', 'backtest', 'predict', 'compare'], 
                       required=True, help='Operation mode')
    parser.add_argument('--symbol', default='BTC-USD', help='Trading symbol (default: BTC-USD)')
    parser.add_argument('--days', type=int, default=60, help='Number of days of historical data (default: 60)')
    parser.add_argument('--model-path', help='Path to saved model (for predict mode)')
    parser.add_argument('--save-viz', action='store_true', help='Save visualizations')
    parser.add_argument('--output-dir', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize system with default config and override specific values
    system = CryptoTradingSystem()
    system.config['symbol'] = args.symbol
    system.config['days'] = args.days
    
    try:
        if args.mode == 'train':
            # Training mode
            logger.info(f"Training mode: {args.symbol} with {args.days} days of data")
            
            # Collect data and prepare features
            data = system.collect_data()
            X, y = system.prepare_features(data)
            
            # Train model
            model = system.train_model(X, y)
            
            # Show feature importance
            importance = model.get_feature_importance()
            print("\nTop 10 Most Important Features:")
            print(importance.head(10))
        
        elif args.mode == 'backtest':
            # Backtesting mode
            logger.info(f"Backtest mode: {args.symbol} with {args.days} days of data")
            
            # Collect data and prepare features
            data = system.collect_data()
            X, y = system.prepare_features(data)
            
            # Train model
            model = system.train_model(X, y, save_model=False)
            
            # Create strategy and run backtest
            strategy = system.create_strategy(model)
            results = system.run_backtest(data, X, strategy)
            
            # Create visualizations if requested
            if args.save_viz:
                system.create_visualizations(results, args.output_dir)
            
            # Print detailed results
            performance = results['performance']
            print(f"\n{'='*50}")
            print("BACKTEST RESULTS")
            print(f"{'='*50}")
            print(f"Symbol: {args.symbol}")
            print(f"Period: {args.days} days")
            print(f"Initial Capital: ${system.config['initial_capital']:,.2f}")
            print(f"Final Value: ${performance['final_portfolio_value']:,.2f}")
            print(f"Total Return: {performance['total_return']:.2%}")
            print(f"Annual Return: {performance['annual_return']:.2%}")
            print(f"Volatility: {performance['volatility']:.2%}")
            print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
            print(f"Total Trades: {performance.get('total_trades', 0)}")
            print(f"Win Rate: {performance.get('win_rate', 0):.2%}")
            print(f"Profit Factor: {performance.get('profit_factor', 0):.2f}")
        
        elif args.mode == 'predict':
            # Prediction mode
            if args.model_path:
                system.load_model(args.model_path)
            else:
                # Train a new model quickly
                data = system.collect_data()
                X, y = system.prepare_features(data)
                system.train_model(X, y, save_model=False)
            
            # Generate prediction
            prediction = system.generate_prediction()
            
            print(f"\n{'='*50}")
            print("PRICE PREDICTION")
            print(f"{'='*50}")
            print(f"Symbol: {prediction['symbol']}")
            print(f"Current Price: ${prediction['current_price']:.2f}")
            print(f"Predicted Return: {prediction['predicted_return']:.2f}%")
            print(f"Predicted Price: ${prediction['predicted_price']:.2f}")
            print(f"Confidence: {prediction['confidence']:.2f}%")
            print(f"Timestamp: {prediction['timestamp']}")
        
        elif args.mode == 'compare':
            # Model comparison mode
            logger.info("Comparing different model configurations...")
            
            # Collect data and prepare features
            data = system.collect_data()
            X, y = system.prepare_features(data)
            
            # Compare models
            comparison = system.compare_models(data, X)
            
            print(f"\n{'='*50}")
            print("MODEL COMPARISON")
            print(f"{'='*50}")
            print(comparison[['strategy_name', 'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']].round(4))
    
    except Exception as e:
        logger.error(f"Error in main application: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
