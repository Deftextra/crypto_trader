"""
Linear Regression Trading Strategy

This module implements a trading strategy based on linear regression price predictions
with risk management and position sizing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signals"""
    BUY = 1
    SELL = -1
    HOLD = 0


class Position:
    """Represents a trading position"""
    
    def __init__(self, side: str, size: float, entry_price: float, timestamp: pd.Timestamp):
        """
        Initialize a position
        
        Args:
            side: 'long' or 'short'
            size: Position size (amount of cryptocurrency)
            entry_price: Entry price
            timestamp: Entry timestamp
        """
        self.side = side
        self.size = size
        self.entry_price = entry_price
        self.timestamp = timestamp
        self.exit_price : float | None = None 
        self.exit_timestamp = None
        self.pnl = 0.0
        self.is_open = True
    
    def close(self, exit_price: float, exit_timestamp: pd.Timestamp):
        """Close the position"""
        self.exit_price = exit_price
        self.exit_timestamp = exit_timestamp
        self.is_open = False
        
        # Calculate PnL
        if self.side == 'long':
            self.pnl = (exit_price - self.entry_price) * self.size
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.size
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL"""
        if not self.is_open:
            return self.pnl
        
        if self.side == 'long':
            return (current_price - self.entry_price) * self.size
        else:  # short
            return (self.entry_price - current_price) * self.size
    
    def __repr__(self):
        status = "OPEN" if self.is_open else "CLOSED"
        return f"Position({self.side}, {self.size:.4f}, {self.entry_price:.2f}, {status})"


class RiskManager:
    """Risk management for trading strategy"""
    
    def __init__(self,
                 max_position_size: float = 0.1,
                 stop_loss_pct: float = 0.02,
                 take_profit_pct: float = 0.04,
                 max_drawdown_pct: float = 0.1,
                 min_prediction_confidence: float = 0.01):
        """
        Initialize risk manager
        
        Args:
            max_position_size: Maximum position size as fraction of portfolio
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            max_drawdown_pct: Maximum drawdown before stopping trading
            min_prediction_confidence: Minimum prediction confidence to trade
        """
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.min_prediction_confidence = min_prediction_confidence
    
    def calculate_position_size(self, 
                              portfolio_value: float, 
                              current_price: float,
                              prediction_confidence: float) -> float:
        """
        Calculate position size based on portfolio value and confidence
        
        Args:
            portfolio_value: Current portfolio value
            current_price: Current asset price
            prediction_confidence: Prediction confidence (0-1)
        
        Returns:
            Position size in cryptocurrency units
        """
        # Base position size
        max_dollar_amount = portfolio_value * self.max_position_size
        
        # Adjust based on confidence
        confidence_multiplier = min(prediction_confidence / self.min_prediction_confidence, 1.0)
        adjusted_dollar_amount = max_dollar_amount * confidence_multiplier
        
        # Convert to cryptocurrency units
        position_size = adjusted_dollar_amount / current_price
        
        return position_size
    
    def should_stop_loss(self, position: Position, current_price: float) -> bool:
        """Check if position should be stopped out"""
        if not position.is_open:
            return False
        
        if position.side == 'long':
            loss_pct = (position.entry_price - current_price) / position.entry_price
            return loss_pct >= self.stop_loss_pct
        else:  # short
            loss_pct = (current_price - position.entry_price) / position.entry_price
            return loss_pct >= self.stop_loss_pct
    
    def should_take_profit(self, position: Position, current_price: float) -> bool:
        """Check if position should take profit"""
        if not position.is_open:
            return False
        
        if position.side == 'long':
            profit_pct = (current_price - position.entry_price) / position.entry_price
            return profit_pct >= self.take_profit_pct
        else:  # short
            profit_pct = (position.entry_price - current_price) / position.entry_price
            return profit_pct >= self.take_profit_pct
    
    def should_stop_trading(self, total_pnl: float, initial_capital: float) -> bool:
        """Check if trading should be stopped due to max drawdown"""
        drawdown_pct = abs(total_pnl) / initial_capital
        return total_pnl < 0 and drawdown_pct >= self.max_drawdown_pct


class LinearRegressionStrategy:
    """Trading strategy based on linear regression predictions"""
    
    def __init__(self,
                 model,
                 risk_manager: RiskManager = None,
                 prediction_threshold: float = 0.5,
                 lookback_periods: int = 5):
        """
        Initialize the trading strategy
        
        Args:
            model: Trained linear regression model
            risk_manager: Risk management instance
            prediction_threshold: Minimum prediction change to trigger signal (%)
            lookback_periods: Number of periods to look back for signal confirmation
        """
        self.model = model
        self.risk_manager = risk_manager or RiskManager()
        self.prediction_threshold = prediction_threshold
        self.lookback_periods = lookback_periods
        
        # Strategy state
        self.positions = []
        self.portfolio_value = 0.0
        self.initial_capital = 0.0
        self.cash = 0.0
        self.total_pnl = 0.0
        self.trade_history = []
    
    def initialize_portfolio(self, initial_capital: float):
        """Initialize portfolio with starting capital"""
        self.initial_capital = initial_capital
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.total_pnl = 0.0
        logger.info(f"Portfolio initialized with ${initial_capital:,.2f}")
    
    def generate_signal(self, features: pd.DataFrame, current_price: float) -> Tuple[Signal, float]:
        """
        Generate trading signal based on model predictions
        
        Args:
            features: Current features for prediction
            current_price: Current price
        
        Returns:
            Tuple of (signal, confidence)
        """
        if len(features) < self.lookback_periods:
            return Signal.HOLD, 0.0
        
        try:
            # Get prediction
            prediction = self.model.predict(features.tail(1))[0]
            
            # Calculate predicted return
            predicted_return = prediction  # Model predicts return percentage
            
            # Generate signal based on prediction
            confidence = abs(predicted_return)
            
            if predicted_return > self.prediction_threshold:
                signal = Signal.BUY
            elif predicted_return < -self.prediction_threshold:
                signal = Signal.SELL
            else:
                signal = Signal.HOLD
            
            # Additional confirmation: look at recent predictions
            if len(features) >= self.lookback_periods:
                recent_predictions = []
                # Get last few rows for consistency check
                lookback_window = min(self.lookback_periods, len(features))
                
                for i in range(1, lookback_window + 1):
                    # Handle both DataFrame and numpy array inputs
                    if hasattr(features, 'iloc'):
                        # It's a DataFrame
                        row_features = features.iloc[-i:-i+1] if i < len(features) else features.tail(1)
                    else:
                        # It's already a numpy array or similar
                        row_features = features[-i:] if i <= len(features) else features[-1:]
                    
                    try:
                        pred = self.model.predict(row_features)[0]
                        recent_predictions.append(pred)
                    except Exception as pred_error:
                        logger.debug(f"Prediction error for lookback {i}: {pred_error}")
                        # Skip this prediction if it fails
                        continue
                
                # Check consistency only if we have enough predictions
                if len(recent_predictions) >= 2:
                    if signal == Signal.BUY:
                        consistent = sum(1 for p in recent_predictions if p > 0) >= len(recent_predictions) * 0.6
                    elif signal == Signal.SELL:
                        consistent = sum(1 for p in recent_predictions if p < 0) >= len(recent_predictions) * 0.6
                    else:
                        consistent = True
                    
                    if not consistent:
                        confidence *= 0.5  # Reduce confidence if not consistent
                else:
                    # Not enough predictions for consistency check, keep original confidence
                    pass
            
            return signal, confidence
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return Signal.HOLD, 0.0
    
    def execute_trade(self, 
                     signal: Signal, 
                     confidence: float, 
                     current_price: float, 
                     timestamp: pd.Timestamp) -> bool:
        """
        Execute a trade based on signal
        
        Args:
            signal: Trading signal
            confidence: Signal confidence
            current_price: Current price
            timestamp: Current timestamp
        
        Returns:
            True if trade was executed, False otherwise
        """
        if confidence < self.risk_manager.min_prediction_confidence:
            return False
        
        # Close existing positions if signal changes
        open_positions = [p for p in self.positions if p.is_open]
        
        for position in open_positions:
            # Check stop loss and take profit
            if (self.risk_manager.should_stop_loss(position, current_price) or 
                self.risk_manager.should_take_profit(position, current_price)):
                self._close_position(position, current_price, timestamp, "Risk Management")
        
        # Check if we should stop trading
        if self.risk_manager.should_stop_trading(self.total_pnl, self.initial_capital):
            logger.warning("Maximum drawdown reached. Stopping trading.")
            return False
        
        # Open new position based on signal
        if signal in [Signal.BUY, Signal.SELL]:
            position_size = self.risk_manager.calculate_position_size(
                self.portfolio_value, current_price, confidence
            )
            
            # Check if we have enough cash
            required_cash = position_size * current_price
            if required_cash > self.cash:
                logger.warning(f"Insufficient cash for trade. Required: ${required_cash:.2f}, Available: ${self.cash:.2f}")
                return False
            
            # Create position
            side = 'long' if signal == Signal.BUY else 'short'
            position = Position(side, position_size, current_price, timestamp)
            self.positions.append(position)
            
            # Update cash
            self.cash -= required_cash
            
            # Log trade
            trade_info = {
                'timestamp': timestamp,
                'action': 'OPEN',
                'side': side,
                'size': position_size,
                'price': current_price,
                'confidence': confidence,
                'reason': 'Signal'
            }
            self.trade_history.append(trade_info)
            
            logger.info(f"Opened {side} position: {position_size:.4f} @ ${current_price:.2f}")
            return True
        
        return False
    
    def _close_position(self, 
                      position: Position, 
                      exit_price: float, 
                      timestamp: pd.Timestamp, 
                      reason: str = "Manual"):
        """Close a position"""
        position.close(exit_price, timestamp)
        
        # Update cash and PnL
        if position.side == 'long':
            self.cash += position.size * exit_price
        else:  # short
            # For short positions, we return the original borrowed amount plus/minus profit/loss
            self.cash += position.size * position.entry_price + position.pnl
        
        self.total_pnl += position.pnl
        
        # Log trade
        trade_info = {
            'timestamp': timestamp,
            'action': 'CLOSE',
            'side': position.side,
            'size': position.size,
            'price': exit_price,
            'pnl': position.pnl,
            'reason': reason
        }
        self.trade_history.append(trade_info)
        
        logger.info(f"Closed {position.side} position: {position.size:.4f} @ ${exit_price:.2f}, PnL: ${position.pnl:.2f}")
    
    def update_portfolio_value(self, current_price: float):
        """Update portfolio value based on current positions"""
        # Calculate value from cash
        portfolio_value = self.cash
        
        # Add value from open positions
        for position in self.positions:
            if position.is_open:
                if position.side == 'long':
                    portfolio_value += position.size * current_price
                else:  # short
                    # For short positions, we have the original borrowed value plus unrealized PnL
                    portfolio_value += position.size * position.entry_price + position.unrealized_pnl(current_price)
        
        self.portfolio_value = portfolio_value
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get strategy performance statistics"""
        if not self.trade_history:
            return {}
        
        # Calculate trade statistics
        closed_trades = [t for t in self.trade_history if t['action'] == 'CLOSE']
        
        if not closed_trades:
            return {'total_trades': 0}
        
        pnls = [t['pnl'] for t in closed_trades]
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        stats = {
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(closed_trades) if closed_trades else 0,
            'total_pnl': sum(pnls),
            'avg_win': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'max_win': max(pnls) if pnls else 0,
            'max_loss': min(pnls) if pnls else 0,
            'profit_factor': sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf'),
            'total_return': (self.portfolio_value - self.initial_capital) / self.initial_capital if self.initial_capital > 0 else 0
        }
        
        return stats
    
    def run_strategy(self, 
                    data: pd.DataFrame, 
                    features: pd.DataFrame,
                    initial_capital: float = 10000) -> pd.DataFrame:
        """
        Run the complete trading strategy
        
        Args:
            data: Price data with OHLCV columns
            features: Feature data for predictions
            initial_capital: Starting capital
        
        Returns:
            DataFrame with trading results
        """
        self.initialize_portfolio(initial_capital)
        
        results = []
        
        logger.info(f"Running strategy on {len(data)} data points")
        
        for i, (idx, row) in enumerate(data.iterrows()):
            current_price = row['close']
            timestamp = row['timestamp'] if 'timestamp' in row else idx
            
            # Get features up to current point
            current_features = features.iloc[:i+1]
            
            if len(current_features) < self.lookback_periods:
                continue
            
            # Generate signal
            signal, confidence = self.generate_signal(current_features, current_price)
            
            # Execute trade
            trade_executed = self.execute_trade(signal, confidence, current_price, timestamp)
            
            # Update portfolio value
            self.update_portfolio_value(current_price)
            
            # Record results
            result = {
                'timestamp': timestamp,
                'price': current_price,
                'signal': signal.value,
                'confidence': confidence,
                'trade_executed': trade_executed,
                'portfolio_value': self.portfolio_value,
                'cash': self.cash,
                'total_pnl': self.total_pnl,
                'open_positions': len([p for p in self.positions if p.is_open])
            }
            results.append(result)
        
        # Close all open positions at the end
        final_price = data.iloc[-1]['close']
        final_timestamp = data.iloc[-1]['timestamp'] if 'timestamp' in data.columns else data.index[-1]
        
        for position in self.positions:
            if position.is_open:
                self._close_position(position, final_price, final_timestamp, "End of backtest")
        
        # Final portfolio update
        self.update_portfolio_value(final_price)
        
        results_df = pd.DataFrame(results)
        
        # Log strategy statistics
        stats = self.get_strategy_stats()
        logger.info("Strategy Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        return results_df


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    from data.collector import CoinbaseDataCollector
    from utils.features import FeatureEngineer
    from models.linear_model import CryptoLinearModel
    
    # Collect data
    collector = CoinbaseDataCollector()
    feature_engineer = FeatureEngineer()
    
    btc_data = collector.get_recent_data('BTC-USD', days=60)
    
    if not btc_data.empty:
        # Engineer features
        df_features = feature_engineer.engineer_all_features(btc_data)
        X, y = feature_engineer.prepare_ml_data(df_features)
        
        # Train model
        model = CryptoLinearModel(model_type='ridge', feature_selection='kbest', n_features=15)
        model.fit(X, y)
        
        # Run strategy
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
        results = strategy.run_strategy(
            btc_data, 
            X.reset_index(drop=True), 
            initial_capital=10000
        )
        
        print(f"\nBacktest completed!")
        print(f"Final portfolio value: ${strategy.portfolio_value:.2f}")
        print(f"Total return: {((strategy.portfolio_value - 10000) / 10000) * 100:.2f}%")
        
        # Show last few results
        print("\nLast 5 trading periods:")
        print(results.tail()[['timestamp', 'price', 'signal', 'portfolio_value', 'total_pnl']])
