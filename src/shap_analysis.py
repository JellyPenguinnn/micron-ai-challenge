import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from data_processor import DataProcessor
from model import LGBMModel
from config import *
import os

# This script computes and saves SHAP summary plots for the best LGBM model.
# It helps with model interpretability and reporting for the competition.

def main():
    # Load the trained LGBM model (assumes the latest model is used)
    model_dir = sorted((ROOT_DIR / 'models').glob('*'), reverse=True)[0]
    model_path = model_dir / 'stacking_ensemble.joblib'
    lgbm_model = joblib.load(model_path).base_models[0]  # LGBM is the first base model

    # Prepare the data (use training data for SHAP)
    processor = DataProcessor(DATA_DIR)
    X_train, _ = processor.prepare_train_data()
    # Use the first fold's model for SHAP (for speed)
    booster = lgbm_model.models[0]
    explainer = shap.TreeExplainer(booster)
    print('Calculating SHAP values...')
    shap_values = explainer.shap_values(X_train)

    # Save SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_train, show=False)
    plot_path = model_dir / 'shap_summary_plot.png'
    plt.savefig(plot_path, bbox_inches='tight')
    print(f'SHAP summary plot saved to {plot_path}')

if __name__ == '__main__':
    main() 