#!/usr/bin/env python3
"""
Test script for the crypto trading system
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data.collector import CoinbaseDataCollector
from utils.features import FeatureEngineer
from models.linear_model import CryptoLinearModel

def test_data_collection():
    print("Testing data collection...")
    collector = CoinbaseDataCollector()
    data = collector.get_recent_data('BTC-USD', days=7)
    print(f"✓ Collected {len(data)} data points")
    return data

def test_feature_engineering(data):
    print("Testing feature engineering...")
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.engineer_all_features(data)
    X, y = feature_engineer.prepare_ml_data(df_features)
    print(f"✓ Generated {X.shape[1]} features from {X.shape[0]} samples")
    return X, y

def test_model_training(X, y):
    print("Testing model training...")
    model = CryptoLinearModel(model_type='ridge', feature_selection='kbest', n_features=10)
    metrics = model.fit(X, y)
    print(f"✓ Model trained successfully. Validation R²: {metrics['val_r2']:.3f}")
    return model

def test_prediction(model, X):
    print("Testing prediction...")
    prediction = model.predict(X.tail(1))
    print(f"✓ Generated prediction: {prediction[0]:.2f}%")
    return prediction

if __name__ == "__main__":
    try:
        # Test each component
        data = test_data_collection()
        X, y = test_feature_engineering(data)
        model = test_model_training(X, y)
        prediction = test_prediction(model, X)
        
        print(f"\n{'='*50}")
        print("ALL TESTS PASSED! 🎉")
        print(f"{'='*50}")
        print(f"Data points: {len(data)}")
        print(f"Features: {X.shape[1]}")
        print(f"Latest prediction: {prediction[0]:.2f}% return")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
