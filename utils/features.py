"""
Feature Engineering Module for Cryptocurrency Trading

This module creates technical indicators and features from raw price data
for use in machine learning models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators for price data"""
    
    @staticmethod
    def moving_average(data: pd.Series, window: int) -> pd.Series:
        """Calculate simple moving average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def exponential_moving_average(data: pd.Series, window: int) -> pd.Series:
        """Calculate exponential moving average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Returns:
            Tuple of (middle_band, upper_band, lower_band)
        """
        middle_band = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        return middle_band, upper_band, lower_band
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        """
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator
        
        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=window).min()
        highest_high = high.rolling(window=window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()
        
        return k_percent, d_percent
    
    @staticmethod
    def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)"""
        high_low = high - low
        high_close_prev = np.abs(high - close.shift(1))
        low_close_prev = np.abs(low - close.shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r


class FeatureEngineer:
    """Feature engineering for cryptocurrency price data"""
    
    def __init__(self):
        self.technical_indicators = TechnicalIndicators()
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic price-based features"""
        df = df.copy()
        
        # Price changes and returns
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100
        df['high_low_pct'] = (df['high'] - df['low']) / df['low'] * 100
        
        # Price position within range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Volume-weighted average price
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        return df
    
    def create_moving_averages(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Create moving average features"""
        df = df.copy()
        
        for window in windows:
            # Simple moving averages
            df[f'sma_{window}'] = self.technical_indicators.moving_average(df['close'], window)
            df[f'ema_{window}'] = self.technical_indicators.exponential_moving_average(df['close'], window)
            
            # Price relative to moving average
            df[f'close_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
            df[f'close_ema_{window}_ratio'] = df['close'] / df[f'ema_{window}']
            
            # Volume moving averages
            df[f'volume_sma_{window}'] = self.technical_indicators.moving_average(df['volume'], window)
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']
        
        return df
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features"""
        df = df.copy()
        
        # Bollinger Bands
        bb_middle, bb_upper, bb_lower = self.technical_indicators.bollinger_bands(df['close'])
        df['bb_middle'] = bb_middle
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # RSI
        df['rsi'] = self.technical_indicators.rsi(df['close'])
        
        # MACD
        macd_line, signal_line, histogram = self.technical_indicators.macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # Stochastic Oscillator
        k_percent, d_percent = self.technical_indicators.stochastic_oscillator(
            df['high'], df['low'], df['close']
        )
        df['stoch_k'] = k_percent
        df['stoch_d'] = d_percent
        
        # Average True Range
        df['atr'] = self.technical_indicators.average_true_range(
            df['high'], df['low'], df['close']
        )
        
        # Williams %R
        df['williams_r'] = self.technical_indicators.williams_r(
            df['high'], df['low'], df['close']
        )
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Create lagged features for time series analysis"""
        df = df.copy()
        
        features_to_lag = ['close', 'volume', 'price_change_pct', 'rsi']
        
        for feature in features_to_lag:
            if feature in df.columns:
                for lag in lags:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        return df
    
    def create_rolling_statistics(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Create rolling statistical features"""
        df = df.copy()
        
        for window in windows:
            # Rolling statistics for close price
            df[f'close_mean_{window}'] = df['close'].rolling(window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window).std()
            df[f'close_min_{window}'] = df['close'].rolling(window).min()
            df[f'close_max_{window}'] = df['close'].rolling(window).max()
            df[f'close_skew_{window}'] = df['close'].rolling(window).skew()
            
            # Rolling statistics for volume
            df[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()
            df[f'volume_std_{window}'] = df['volume'].rolling(window).std()
            
            # Price volatility
            df[f'volatility_{window}'] = df['close'].pct_change().rolling(window).std() * np.sqrt(24)  # Hourly volatility
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()
        
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            
            # Cyclical encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def create_target_variable(self, df: pd.DataFrame, target_hours: int = 1) -> pd.DataFrame:
        """
        Create target variable for prediction
        
        Args:
            df: Input DataFrame
            target_hours: Number of hours ahead to predict
        
        Returns:
            DataFrame with target variables
        """
        df = df.copy()
        
        # Future price (target for regression)
        df['target_price'] = df['close'].shift(-target_hours)
        
        # Future price change (target for classification)
        df['target_return'] = (df['target_price'] - df['close']) / df['close'] * 100
        
        # Binary classification target (price goes up or down)
        df['target_direction'] = (df['target_return'] > 0).astype(int)
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame, target_hours: int = 1) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        logger.info("Starting feature engineering...")
        
        # Create all features
        df = self.create_price_features(df)
        df = self.create_moving_averages(df)
        df = self.create_technical_indicators(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_statistics(df)
        df = self.create_time_features(df)
        df = self.create_target_variable(df, target_hours)
        
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame, exclude_target: bool = True) -> List[str]:
        """Get list of feature columns (excluding target variables)"""
        exclude_cols = ['timestamp', 'target_price', 'target_return', 'target_direction']
        
        if not exclude_target:
            exclude_cols = ['timestamp']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return feature_cols
    
    def prepare_ml_data(self, df: pd.DataFrame, target_col: str = 'target_return') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for machine learning
        
        Returns:
            Tuple of (features_df, target_series)
        """
        df_clean = df.dropna().copy()
        
        feature_cols = self.get_feature_columns(df_clean)
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        logger.info(f"ML data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y


if __name__ == "__main__":
    # Example usage
    from data.collector import CoinbaseDataCollector
    
    # Collect sample data
    collector = CoinbaseDataCollector()
    btc_data = collector.get_recent_data('BTC-USD', days=30)
    
    if not btc_data.empty:
        # Engineer features
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.engineer_all_features(btc_data)
        
        print(f"Original data shape: {btc_data.shape}")
        print(f"Features data shape: {df_features.shape}")
        print(f"Feature columns: {len(feature_engineer.get_feature_columns(df_features))}")
        
        # Show sample of engineered features
        feature_cols = feature_engineer.get_feature_columns(df_features)[:10]  # First 10 features
        print("\nSample features:")
        print(df_features[feature_cols].head())
