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
├── test/                  # Test data directory
├── submission/            # Submission data directory
├── requirements.txt       # Python dependencies
└── README.md             # This file
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

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
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
4. Save the predictions in the `results` directory

## Model Details

- The model uses LightGBM for regression
- Features include sensor statistics, time-based features, and spatial features
- Cross-validation is used for robust evaluation
- Feature selection is performed based on importance and correlation

## Output

- Trained models are saved in the `models` directory
- Predictions are saved in the `results` directory
- Training logs are saved in the `logs` directory

## License

This project is licensed under the MIT License - see the LICENSE file for details.
