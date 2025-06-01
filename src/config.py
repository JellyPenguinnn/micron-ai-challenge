import os
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR  # Data is in the root directory
MODEL_DIR = PROJECT_DIR / "trained_models"
RESULTS_DIR = PROJECT_DIR / "prediction_results"
LOG_DIR = PROJECT_DIR / "logs"

# Data parameters
TARGET_COL = "Measurement"
SENSOR_AGG_FEATURES = ['mean', 'std', 'min', 'max', 'first', 'last']

# Feature selection parameters
FEATURE_SELECTION = {
    'importance_threshold': 0.01,
    'correlation_threshold': 0.95
}

# Cross-validation parameters
CV_PARAMS = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': 42
}

# Training parameters
TRAIN_PARAMS = {
    'num_boost_round': 2000,
    'early_stopping_rounds': 100,
    'verbose_eval': 200
}

# LightGBM parameters
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

# Random seed for reproducibility
RANDOM_SEED = 42

# Model parameters
N_SPLITS = 5

# Training parameters
TRAINING_PARAMS = {
    "num_boost_round": 2000,
    "early_stopping_rounds": 100,
    "verbose_eval": 200
}

# Feature engineering parameters
TIME_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "day_of_month"
]

# Sensor aggregation features
SENSOR_AGG_FEATURES = [
    "mean",
    "std",
    "min",
    "max",
    "last"
] 