import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
from data_processor import DataProcessor
from model import Model
from config import *

def setup_logging():
    """Set up logging configuration."""
    log_file = LOG_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def setup_directories():
    """Create necessary directories if they don't exist."""
    for directory in [MODEL_DIR, RESULTS_DIR, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def save_predictions(predictions: pd.Series, submission_template: pd.DataFrame):
    """Save predictions to a parquet file."""
    filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    submission = submission_template.copy()
    submission[TARGET_COL] = predictions
    output_path = RESULTS_DIR / filename
    submission.to_parquet(output_path)
    logging.info(f"Predictions saved to {output_path}")

def main():
    """Main training and prediction pipeline."""
    try:
        # Setup
        setup_directories()
        setup_logging()
        logging.info("Starting training pipeline")
        
        # Load and process data
        data_processor = DataProcessor(DATA_DIR)
        X_train, y_train = data_processor.prepare_train_data()
        
        # Train model
        model = Model(MODEL_DIR)
        metrics = model.train(X_train, y_train)
        model.save()
        
        # Generate predictions
        X_test = data_processor.prepare_test_data()
        predictions = model.predict(X_test)
        
        # Save predictions
        submission_template = pd.read_parquet(DATA_DIR / "submission" / "metrology_data.parquet")
        save_predictions(predictions, submission_template)
        
        logging.info("Training pipeline completed successfully")
        
    except Exception as e:
        logging.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 