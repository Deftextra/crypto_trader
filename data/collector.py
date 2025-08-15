"""
Cryptocurrency Data Collector Module

This module fetches historical and real-time cryptocurrency data from Coinbase's public API.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import List, Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoinbaseDataCollector:
    """Collect cryptocurrency data from Coinbase Pro API"""
    
    BASE_URL = "https://api.exchange.coinbase.com"
    
    def __init__(self):
        """Initialize the data collector"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoTrader/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
    
    def get_products(self) -> List[Dict]:
        """Get all available trading pairs"""
        try:
            response = self.session.get(f"{self.BASE_URL}/products")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching products: {e}")
            return []
    
    def get_product_ticker(self, product_id: str) -> Dict:
        """Get current ticker data for a specific product"""
        try:
            response = self.session.get(f"{self.BASE_URL}/products/{product_id}/ticker")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching ticker for {product_id}: {e}")
            return {}
    
    def get_historical_data(self, 
                           product_id: str, 
                           start_date: datetime, 
                           end_date: datetime, 
                           granularity: int = 3600) -> pd.DataFrame:
        """
        Get historical candle data for a specific product
        
        Args:
            product_id: Trading pair (e.g., 'BTC-USD')
            start_date: Start date for historical data
            end_date: End date for historical data  
            granularity: Time interval in seconds (60, 300, 900, 3600, 21600, 86400)
        
        Returns:
            DataFrame with columns: timestamp, low, high, open, close, volume
        """
        try:
            # Convert dates to Unix timestamps
            start_timestamp = int(start_date.timestamp())
            end_timestamp = int(end_date.timestamp())
            
            params = {
                'start': start_timestamp,
                'end': end_timestamp,
                'granularity': granularity
            }
            
            response = self.session.get(
                f"{self.BASE_URL}/products/{product_id}/candles", 
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            if not data:
                logger.warning(f"No data returned for {product_id}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Convert price columns to float
            price_cols = ['low', 'high', 'open', 'close']
            df[price_cols] = df[price_cols].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            logger.info(f"Fetched {len(df)} candles for {product_id}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching historical data for {product_id}: {e}")
            return pd.DataFrame()
    
    def get_recent_data(self, product_id: str, days: int = 30, granularity: int = 3600) -> pd.DataFrame:
        """Get recent historical data for a product"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.get_historical_data(product_id, start_date, end_date, granularity)
    
    def get_multiple_products_data(self, 
                                  product_ids: List[str], 
                                  days: int = 30, 
                                  granularity: int = 3600,
                                  delay: float = 0.1) -> Dict[str, pd.DataFrame]:
        """
        Get recent data for multiple products with rate limiting
        
        Args:
            product_ids: List of product IDs
            days: Number of days of historical data
            granularity: Time interval in seconds
            delay: Delay between API calls to avoid rate limiting
        
        Returns:
            Dictionary mapping product_id to DataFrame
        """
        data = {}
        
        for product_id in product_ids:
            logger.info(f"Fetching data for {product_id}")
            df = self.get_recent_data(product_id, days, granularity)
            if not df.empty:
                data[product_id] = df
            
            # Rate limiting
            if delay > 0:
                time.sleep(delay)
        
        return data
    
    def get_order_book(self, product_id: str, level: int = 2) -> Dict:
        """
        Get order book for a product
        
        Args:
            product_id: Trading pair
            level: Level of detail (1, 2, or 3)
        
        Returns:
            Order book data
        """
        try:
            params = {'level': level}
            response = self.session.get(
                f"{self.BASE_URL}/products/{product_id}/book",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching order book for {product_id}: {e}")
            return {}


def get_popular_crypto_pairs() -> List[str]:
    """Get a list of popular cryptocurrency trading pairs"""
    return [
        'BTC-USD',
        'ETH-USD', 
        'ADA-USD',
        'DOT-USD',
        'LINK-USD',
        'LTC-USD',
        'XLM-USD',
        'ALGO-USD'
    ]


if __name__ == "__main__":
    # Example usage
    collector = CoinbaseDataCollector()
    
    # Get recent BTC data
    btc_data = collector.get_recent_data('BTC-USD', days=7)
    print(f"BTC Data Shape: {btc_data.shape}")
    if not btc_data.empty:
        print(btc_data.head())
        print(f"Price range: ${btc_data['low'].min():.2f} - ${btc_data['high'].max():.2f}")
