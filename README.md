# Micron AI Challenge

This repository contains the implementation for the Micron AI Challenge, focusing on predicting measurements based on sensor data.

## Project Structure

```
.
├── src/                    # Source code
│   ├── config.py          # Configuration parameters
│   ├── data_processor.py  # Data loading and preprocessing
│   ├── model.py           # Model implementation
│   └── main.py            # Main training pipeline
├── train/                 # Training data directory
│   ├── run_data_*.parquet        # Run data files
│   ├── incoming_run_data_*.parquet # Incoming run data files
│   └── metrology_data*.parquet   # Metrology data files
├── test/                  # Test data directory
│   ├── run_data.parquet          # Test run data
│   └── incoming_run_data.parquet # Test incoming run data
├── submission/            # Submission data directory
│   └── metrology_data.parquet    # Submission template
├── trained_models/        # Saved trained models
├── prediction_results/    # Generated predictions
├── logs/                 # Training and execution logs
├── docs/                 # Documentation and additional resources
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Data Setup

The training and test data are too large to be included in this repository. You can download them from:
[Google Drive Link](https://drive.google.com/drive/folders/1xRzD47m2XcOYYEQBe9Cq_-w1jHJaz8Y4?usp=share_link)

After downloading:
1. Extract the data files
2. Place them in their respective directories:
   - Training data in the `train/` directory
   - Test data in the `test/` directory
   - Submission data in the `submission/` directory

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Pipeline

To run the complete training and prediction pipeline:

```bash
python src/main.py
```

This will:
1. Load and preprocess the training data
2. Train the model using cross-validation
3. Generate predictions for the test set
4. Save the predictions in the `prediction_results` directory

## Model Details

- The model uses LightGBM for regression
- Features include:
  - Sensor statistics (mean, std, min, max, first, last)
  - Time-based features (hour, day, month, etc.)
  - Spatial features (distance from center, angle, quadrant)
  - Process duration features
- Cross-validation with 5 folds for robust evaluation
- Feature selection based on importance and correlation
- Early stopping to prevent overfitting

## Output

- Trained models are saved in the `trained_models` directory
- Predictions are saved in the `prediction_results` directory
- Training logs are saved in the `logs` directory

## File Formats

- Input data: Parquet files
- Model files: Joblib format
- Predictions: Parquet files matching submission template
- Logs: Text files with timestamp

## Performance

The model achieves:
- RMSE: ~0.0329
- R2 Score: ~0.9625

## Development

- Python 3.12+ is required
- All dependencies are listed in `requirements.txt`
- Code follows PEP 8 style guide
- Logging is implemented for debugging and monitoring

## License

This project is licensed under the MIT License - see the LICENSE file for details.
