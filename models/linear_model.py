"""
Linear Regression Model for Cryptocurrency Price Prediction

This module implements linear regression models with feature selection,
validation, and prediction functionality for cryptocurrency trading.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import joblib
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class CryptoLinearModel:
    """Linear regression model for cryptocurrency price prediction"""
    
    def __init__(self, 
                 model_type: str = 'linear',
                 scaler_type: str = 'standard',
                 feature_selection: str = 'none',
                 n_features: int = 20):
        """
        Initialize the linear model
        
        Args:
            model_type: Type of linear model ('linear', 'ridge', 'lasso', 'elastic')
            scaler_type: Type of scaler ('standard', 'robust', 'none')
            feature_selection: Feature selection method ('none', 'kbest', 'rfe')
            n_features: Number of features to select (if using feature selection)
        """
        self.model_type = model_type
        self.scaler_type = scaler_type
        self.feature_selection = feature_selection
        self.n_features = n_features
        
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.feature_names = None
        self.selected_features = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize model, scaler, and feature selector components"""
        # Initialize model
        if self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif self.model_type == 'lasso':
            self.model = Lasso(alpha=1.0)
        elif self.model_type == 'elastic':
            self.model = ElasticNet(alpha=1.0, l1_ratio=0.5)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Initialize scaler
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif self.scaler_type == 'none':
            self.scaler = None
        else:
            raise ValueError(f"Unsupported scaler type: {self.scaler_type}")
        
        # Initialize feature selector
        if self.feature_selection == 'kbest':
            self.feature_selector = SelectKBest(score_func=f_regression, k=self.n_features)
        elif self.feature_selection == 'rfe':
            self.feature_selector = RFE(estimator=LinearRegression(), n_features_to_select=self.n_features)
        elif self.feature_selection == 'none':
            self.feature_selector = None
        else:
            raise ValueError(f"Unsupported feature selection: {self.feature_selection}")
    
    def preprocess_features(self, X: pd.DataFrame, y: pd.Series = None, fit: bool = False) -> np.ndarray:
        """
        Preprocess features with scaling and feature selection
        
        Args:
            X: Input features
            y: Target variable (required for fitting feature selection)
            fit: Whether to fit the preprocessors
        
        Returns:
            Preprocessed features
        """
        X_processed = X.copy()
        
        # Handle infinite values and NaN
        X_processed = X_processed.replace([np.inf, -np.inf], np.nan)
        X_processed = X_processed.fillna(X_processed.median())
        
        if fit:
            self.feature_names = list(X_processed.columns)
        
        # Feature selection
        if self.feature_selector is not None:
            if fit:
                if y is None:
                    raise ValueError("Target variable y is required for fitting feature selection")
                X_selected = self.feature_selector.fit_transform(X_processed, y)
                if hasattr(self.feature_selector, 'get_support'):
                    self.selected_features = [self.feature_names[i] for i, selected in enumerate(self.feature_selector.get_support()) if selected]
                else:
                    self.selected_features = self.feature_names
            else:
                X_selected = self.feature_selector.transform(X_processed)
        else:
            X_selected = X_processed.values
            if fit:
                self.selected_features = self.feature_names
        
        # Scaling
        if self.scaler is not None:
            if fit:
                X_scaled = self.scaler.fit_transform(X_selected)
            else:
                X_scaled = self.scaler.transform(X_selected)
        else:
            X_scaled = X_selected
        
        return X_scaled
    
    def fit(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> Dict[str, float]:
        """
        Fit the linear regression model
        
        Args:
            X: Input features
            y: Target variable
            validation_split: Fraction of data to use for validation
        
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training {self.model_type} model with {X.shape[1]} features")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, shuffle=False
        )
        
        # Preprocess features
        X_train_processed = self.preprocess_features(X_train, y=y_train, fit=True)
        X_val_processed = self.preprocess_features(X_val, fit=False)
        
        # Fit model
        self.model.fit(X_train_processed, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_processed)
        y_val_pred = self.model.predict(X_val_processed)
        
        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'val_mse': mean_squared_error(y_val, y_val_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'val_r2': r2_score(y_val, y_val_pred)
        }
        
        logger.info(f"Training complete. Validation R²: {metrics['val_r2']:.4f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        X_processed = self.preprocess_features(X, fit=False)
        predictions = self.model.predict(X_processed)
        
        return predictions
    
    def predict_with_confidence(self, X: pd.DataFrame, n_bootstrap: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals using bootstrap
        
        Args:
            X: Input features
            n_bootstrap: Number of bootstrap samples
        
        Returns:
            Tuple of (predictions, confidence_intervals)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        X_processed = self.preprocess_features(X, fit=False)
        
        # Get base predictions
        base_predictions = self.model.predict(X_processed)
        
        # Bootstrap predictions (simplified approach)
        # In practice, you'd want to retrain on bootstrap samples
        residuals = np.random.normal(0, np.std(base_predictions), (n_bootstrap, len(base_predictions)))
        bootstrap_predictions = base_predictions[np.newaxis, :] + residuals
        
        confidence_intervals = np.percentile(bootstrap_predictions, [2.5, 97.5], axis=0)
        
        return base_predictions, confidence_intervals
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation
        
        Args:
            X: Input features
            y: Target variable
            cv_folds: Number of cross-validation folds
        
        Returns:
            Dictionary of cross-validation scores
        """
        logger.info(f"Performing {cv_folds}-fold time series cross-validation")
        
        # Preprocess features
        X_processed = self.preprocess_features(X, fit=True)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Cross-validate
        cv_scores = {
            'mse': [],
            'mae': [],
            'r2': []
        }
        
        for train_idx, val_idx in tscv.split(X_processed):
            X_train_cv = X_processed[train_idx]
            X_val_cv = X_processed[val_idx]
            y_train_cv = y.iloc[train_idx]
            y_val_cv = y.iloc[val_idx]
            
            # Fit model
            self.model.fit(X_train_cv, y_train_cv)
            
            # Predict
            y_pred_cv = self.model.predict(X_val_cv)
            
            # Calculate metrics
            cv_scores['mse'].append(mean_squared_error(y_val_cv, y_pred_cv))
            cv_scores['mae'].append(mean_absolute_error(y_val_cv, y_pred_cv))
            cv_scores['r2'].append(r2_score(y_val_cv, y_pred_cv))
        
        # Log results
        for metric, scores in cv_scores.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            logger.info(f"CV {metric.upper()}: {mean_score:.4f} ± {std_score:.4f}")
        
        return cv_scores
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Optimize hyperparameters using grid search
        
        Args:
            X: Input features
            y: Target variable
        
        Returns:
            Dictionary with best parameters and scores
        """
        logger.info("Starting hyperparameter optimization")
        
        # Preprocess features
        X_processed = self.preprocess_features(X, fit=True)
        
        # Define parameter grids for different models
        param_grids = {
            'ridge': {'alpha': [0.1, 1.0, 10.0, 100.0]},
            'lasso': {'alpha': [0.01, 0.1, 1.0, 10.0]},
            'elastic': {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.7, 0.9]}
        }
        
        if self.model_type not in param_grids:
            logger.warning(f"No hyperparameters to optimize for {self.model_type}")
            return {}
        
        # Grid search with time series split
        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(
            self.model,
            param_grids[self.model_type],
            cv=tscv,
            scoring='r2',
            n_jobs=-1
        )
        
        grid_search.fit(X_processed, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best CV score: {results['best_score']:.4f}")
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model does not have feature coefficients")
        
        # Get coefficients
        coefficients = self.model.coef_
        
        # Get feature names (selected features if feature selection was used)
        feature_names = self.selected_features if self.selected_features else self.feature_names
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })
        
        # Sort by absolute coefficient
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str):
        """Save the trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'model_type': self.model_type,
            'scaler_type': self.scaler_type,
            'feature_selection': self.feature_selection,
            'n_features': self.n_features
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and preprocessors"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.feature_names = model_data['feature_names']
        self.selected_features = model_data['selected_features']
        self.model_type = model_data['model_type']
        self.scaler_type = model_data['scaler_type']
        self.feature_selection = model_data['feature_selection']
        self.n_features = model_data['n_features']
        
        logger.info(f"Model loaded from {filepath}")


class ModelEnsemble:
    """Ensemble of linear models for improved predictions"""
    
    def __init__(self, models: List[CryptoLinearModel]):
        """Initialize ensemble with list of models"""
        self.models = models
        self.weights = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Fit all models in the ensemble"""
        logger.info(f"Training ensemble of {len(self.models)} models")
        
        all_metrics = {}
        predictions = []
        
        # Train each model
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}")
            metrics = model.fit(X, y)
            all_metrics[f'model_{i+1}'] = metrics
            
            # Get predictions for weight calculation
            pred = model.predict(X)
            predictions.append(pred)
        
        # Calculate ensemble weights based on validation R²
        val_r2_scores = [all_metrics[f'model_{i+1}']['val_r2'] for i in range(len(self.models))]
        
        # Convert negative R² to small positive values
        val_r2_scores = [max(score, 0.001) for score in val_r2_scores]
        
        # Calculate weights (higher R² gets higher weight)
        total_score = sum(val_r2_scores)
        self.weights = [score / total_score for score in val_r2_scores]
        
        logger.info(f"Ensemble weights: {[f'{w:.3f}' for w in self.weights]}")
        
        return all_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        if self.weights is None:
            raise ValueError("Ensemble has not been trained yet")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        weighted_predictions = np.average(predictions, axis=0, weights=self.weights)
        
        return weighted_predictions


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    from data.collector import CoinbaseDataCollector
    from utils.features import FeatureEngineer
    
    # Collect and prepare data
    collector = CoinbaseDataCollector()
    feature_engineer = FeatureEngineer()
    
    btc_data = collector.get_recent_data('BTC-USD', days=60)
    
    if not btc_data.empty:
        # Engineer features
        df_features = feature_engineer.engineer_all_features(btc_data)
        X, y = feature_engineer.prepare_ml_data(df_features)
        
        print(f"Data shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Train model
        model = CryptoLinearModel(model_type='ridge', feature_selection='kbest', n_features=15)
        metrics = model.fit(X, y)
        
        print("\nTraining Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Feature importance
        importance = model.get_feature_importance()
        print(f"\nTop 10 Most Important Features:")
        print(importance.head(10))
