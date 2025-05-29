import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from config import *
from lightgbm import early_stopping, log_evaluation

class Model:
    """
    LightGBM model implementation with feature selection and cross-validation.
    """
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.models = []
        self.feature_importance = None
        self.selected_features = None
        
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Selects features based on importance and correlation.
        """
        try:
            # Drop datetime and timedelta columns
            X = X.select_dtypes(exclude=["datetime", "timedelta", "datetime64[ns]", "timedelta64[ns]"])
            # Train a model to get feature importance
            model = lgb.LGBMRegressor(**LGBM_PARAMS)
            model.fit(X, y)
            
            # Get feature importance
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Select features based on importance threshold
            important_features = importance[importance['importance'] > FEATURE_SELECTION['importance_threshold']]['feature'].tolist()
            
            if not important_features:
                logging.warning("No features met importance threshold, using all features")
                important_features = X.columns.tolist()
            
            # Calculate correlation matrix for selected features
            corr_matrix = X[important_features].corr().abs()
            
            # Select features with correlation below threshold
            selected_features = []
            for feature in important_features:
                if not any(corr_matrix[feature][selected_features] > FEATURE_SELECTION['correlation_threshold']):
                    selected_features.append(feature)
            
            if not selected_features:
                logging.warning("No features met correlation threshold, using important features")
                selected_features = important_features
            
            self.selected_features = selected_features
            self.feature_importance = importance
            
            return X[selected_features]
        except Exception as e:
            logging.error(f"Error in feature selection: {str(e)}")
            raise
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Trains the model using cross-validation and returns metrics.
        """
        try:
            # Drop datetime and timedelta columns
            X = X.select_dtypes(exclude=["datetime", "timedelta", "datetime64[ns]", "timedelta64[ns]"])
            # Select features
            X = self.select_features(X, y)
            
            # Initialize cross-validation
            kf = KFold(n_splits=CV_PARAMS['n_splits'], shuffle=CV_PARAMS['shuffle'], random_state=CV_PARAMS['random_state'])
            
            # Initialize metrics
            fold_metrics = []
            oof_predictions = np.zeros(len(X))
            
            # Train models for each fold
            for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Create and train model
                model = lgb.LGBMRegressor(**LGBM_PARAMS)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='rmse',
                    callbacks=[
                        early_stopping(TRAIN_PARAMS['early_stopping_rounds']),
                        log_evaluation(TRAIN_PARAMS['verbose_eval'])
                    ]
                )
                
                # Make predictions
                val_preds = model.predict(X_val)
                oof_predictions[val_idx] = val_preds
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_val, val_preds))
                r2 = r2_score(y_val, val_preds)
                fold_metrics.append({'fold': fold, 'rmse': rmse, 'r2': r2})
                
                # Save model
                self.models.append(model)
                
                logging.info(f"Fold {fold} - RMSE: {rmse:.4f}, R2: {r2:.4f}")
            
            # Calculate overall metrics
            overall_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
            overall_r2 = r2_score(y, oof_predictions)
            
            metrics = {
                'fold_metrics': fold_metrics,
                'overall_rmse': overall_rmse,
                'overall_r2': overall_r2
            }
            
            logging.info(f"Overall - RMSE: {overall_rmse:.4f}, R2: {overall_r2:.4f}")
            
            return metrics
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the trained models.
        """
        try:
            if not self.models:
                raise ValueError("No trained models available")
            # Drop datetime and timedelta columns
            X = X.select_dtypes(exclude=["datetime", "timedelta", "datetime64[ns]", "timedelta64[ns]"])
            if self.selected_features is not None:
                X = X[self.selected_features]
            
            # Make predictions with each model
            predictions = np.array([model.predict(X) for model in self.models])
            
            # Return average predictions
            return np.mean(predictions, axis=0)
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise
    
    def save(self, filename: str = "model.joblib") -> None:
        """
        Saves the model and feature information.
        """
        try:
            if not self.models:
                raise ValueError("No trained models to save")
            
            model_data = {
                'models': self.models,
                'feature_importance': self.feature_importance,
                'selected_features': self.selected_features
            }
            
            joblib.dump(model_data, self.model_dir / filename)
            logging.info(f"Model saved to {self.model_dir / filename}")
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise
    
    def load(self, filename: str = "model.joblib") -> None:
        """
        Loads a saved model and feature information.
        """
        try:
            model_data = joblib.load(self.model_dir / filename)
            
            self.models = model_data['models']
            self.feature_importance = model_data['feature_importance']
            self.selected_features = model_data['selected_features']
            
            logging.info(f"Model loaded from {self.model_dir / filename}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise 