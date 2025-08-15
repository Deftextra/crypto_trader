# Cryptocurrency Linear Regression Trading System

A comprehensive cryptocurrency trading system that uses linear regression models to predict price movements and execute trades based on technical indicators and market data from Coinbase.

## Features

- **Data Collection**: Fetches real-time and historical cryptocurrency data from Coinbase Pro API
- **Feature Engineering**: Creates 100+ technical indicators and features from raw price data
- **Machine Learning**: Implements multiple linear regression models with feature selection
- **Trading Strategy**: Executes trades based on model predictions with risk management
- **Backtesting**: Comprehensive backtesting framework with performance analytics
- **Visualization**: Creates detailed performance reports and charts
- **Risk Management**: Position sizing, stop-loss, take-profit, and drawdown controls

## Project Structure

```
crypto_trader/
├── data/
│   ├── __init__.py
│   └── collector.py          # Data collection from Coinbase API
├── utils/
│   ├── __init__.py
│   └── features.py           # Feature engineering and technical indicators
├── models/
│   ├── __init__.py
│   ├── linear_model.py       # Linear regression models
│   └── saved/                # Directory for saved models
├── strategies/
│   ├── __init__.py
│   └── linear_strategy.py    # Trading strategy implementation
├── backtesting/
│   ├── __init__.py
│   └── backtest.py           # Backtesting framework
├── output/                   # Generated visualizations and reports
├── main.py                   # Main application entry point
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

1. Clone or create the project directory:
```bash
mkdir crypto_trader
cd crypto_trader
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The system supports four main modes of operation:

### 1. Training Mode
Train a model on historical data and save it for later use:

```bash
python main.py --mode train --symbol BTC-USD --days 90
```

### 2. Backtesting Mode
Run a complete backtest on historical data:

```bash
python main.py --mode backtest --symbol BTC-USD --days 60 --save-viz
```

Options:
- `--save-viz`: Save performance visualizations
- `--output-dir`: Specify output directory for results

### 3. Prediction Mode
Generate price predictions for current market conditions:

```bash
python main.py --mode predict --symbol BTC-USD
```

Or use a previously saved model:
```bash
python main.py --mode predict --symbol BTC-USD --model-path models/saved/BTC-USD_20240115_143022.joblib
```

### 4. Model Comparison Mode
Compare different model configurations:

```bash
python main.py --mode compare --symbol BTC-USD --days 60
```

## Supported Cryptocurrencies

The system works with any cryptocurrency pair available on Coinbase Pro. Popular pairs include:
- BTC-USD (Bitcoin)
- ETH-USD (Ethereum)
- ADA-USD (Cardano)
- DOT-USD (Polkadot)
- LINK-USD (Chainlink)
- LTC-USD (Litecoin)
- XLM-USD (Stellar)
- ALGO-USD (Algorand)

## Configuration

The system uses default configuration parameters that can be customized by modifying the `_get_default_config()` method in `main.py`:

```python
config = {
    # Data parameters
    'symbol': 'BTC-USD',
    'days': 60,
    'granularity': 3600,  # 1 hour intervals
    
    # Model parameters
    'model_type': 'ridge',        # 'linear', 'ridge', 'lasso', 'elastic'
    'feature_selection': 'kbest', # 'none', 'kbest', 'rfe'
    'n_features': 15,
    'target_hours': 1,            # Prediction horizon
    
    # Strategy parameters
    'prediction_threshold': 0.5,   # Minimum prediction confidence (%)
    'lookback_periods': 5,
    
    # Risk management
    'initial_capital': 10000,
    'max_position_size': 0.1,     # 10% of portfolio
    'stop_loss_pct': 0.02,        # 2% stop loss
    'take_profit_pct': 0.04,      # 4% take profit
    'max_drawdown_pct': 0.1,      # 10% max drawdown
    'min_prediction_confidence': 0.01,
    
    # Backtesting
    'commission': 0.001,          # 0.1% commission
    'slippage': 0.001             # 0.1% slippage
}
```

## Features Generated

The system automatically generates 100+ features from raw OHLCV data:

### Technical Indicators
- Moving Averages (SMA, EMA) with multiple periods
- Bollinger Bands
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- Average True Range (ATR)
- Williams %R

### Price Features
- Price changes and returns
- High-low percentage ranges
- Volume-weighted average price (VWAP)
- Price position within daily range

### Statistical Features
- Rolling means, standard deviations, min/max
- Volatility measures
- Skewness of price movements

### Lagged Features
- Historical values of key indicators
- Multiple lookback periods

### Time Features
- Hour of day, day of week, month
- Cyclical encoding of time features

## Model Types

The system supports multiple linear regression variants:

1. **Linear Regression**: Basic linear regression
2. **Ridge Regression**: L2 regularization for feature stability
3. **Lasso Regression**: L1 regularization for feature selection
4. **Elastic Net**: Combined L1/L2 regularization

## Risk Management

The system includes comprehensive risk management:

- **Position Sizing**: Based on portfolio value and prediction confidence
- **Stop Loss**: Automatic loss cutting at specified percentage
- **Take Profit**: Profit taking at target levels
- **Maximum Drawdown**: Trading halt if losses exceed threshold
- **Confidence Threshold**: Only trade on high-confidence predictions

## Performance Metrics

Backtesting provides detailed performance analytics:

### Returns
- Total return, annual return
- Risk-adjusted returns (Sharpe, Sortino, Calmar ratios)

### Risk Metrics
- Maximum drawdown
- Volatility
- Value at Risk (VaR)
- Conditional VaR

### Trading Metrics
- Total trades, win rate
- Average win/loss
- Profit factor
- Trade duration statistics

## Visualization

The system generates comprehensive performance reports including:
- Portfolio value over time
- Drawdown charts
- Returns distribution
- Price charts with trading signals
- Rolling performance metrics

## Example Output

```
==================================================
BACKTEST RESULTS
==================================================
Symbol: BTC-USD
Period: 60 days
Initial Capital: $10,000.00
Final Value: $10,847.32
Total Return: 8.47%
Annual Return: 51.82%
Volatility: 45.23%
Sharpe Ratio: 1.15
Max Drawdown: -12.34%
Total Trades: 23
Win Rate: 60.87%
Profit Factor: 1.67
```

## Limitations and Disclaimers

⚠️ **Important Disclaimers:**

1. **Educational Purpose**: This system is for educational and research purposes only
2. **No Financial Advice**: Not intended as financial advice or recommendations
3. **Past Performance**: Past performance does not guarantee future results
4. **Market Risk**: Cryptocurrency trading involves substantial risk of loss
5. **No Warranty**: The system is provided "as-is" without warranties

## API Rate Limits

The system respects Coinbase Pro API rate limits by:
- Adding delays between requests
- Using reasonable request frequencies
- Caching data when possible

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is provided for educational purposes. Use at your own risk.

## Support

For issues or questions:
1. Check the logs in `crypto_trader.log`
2. Review the error messages
3. Ensure all dependencies are installed
4. Verify API connectivity

## Changelog

### Version 1.0.0
- Initial release
- Basic linear regression models
- Coinbase data integration
- Comprehensive backtesting
- Risk management system
- Performance visualization
