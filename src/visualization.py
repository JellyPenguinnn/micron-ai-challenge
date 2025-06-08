import argparse
import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import DATA_DIR, MODEL_DIR, TARGET_COL  # existing constants
from data_processor import DataProcessor            # existing pipeline
from model import Model                             # for diagnostics

# ---------------------------------------------------------------------------
# Global paths – absolute to avoid Path.relative_to() issues
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path.cwd()  # assume you run `python src/visualization.py` from repo root
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
_pfx_counter = {"val": 0}

def _next(name: str) -> Path:
    """Return a unique figures/NN_name.png Path."""
    _pfx_counter["val"] += 1
    return FIGURES_DIR / f"{_pfx_counter['val']:02d}_{name}.png"


def _save(fig: plt.Figure, path: Path):
    """Save figure and emit a friendly log message."""
    path = path.resolve()  # make absolute to avoid .relative_to issues
    fig.tight_layout()
    fig.savefig(path, dpi=300)

    try:
        # Try to display a nice relative path (if within CWD)
        rel = path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = path
    logging.info(f"Saved {rel}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Exploratory-data-analysis (EDA) figures
# ---------------------------------------------------------------------------

def measurement_distribution(metrology_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(metrology_df[TARGET_COL], kde=True, ax=ax)
    ax.set_xlabel("Measurement value")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of target measurement values")
    _save(fig, _next("measurement_dist"))


def wafer_map(metrology_df: pd.DataFrame):
    """Scatter of measurement across wafer X-Y to visualise non-uniformity."""
    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(metrology_df["X"], metrology_df["Y"], c=metrology_df[TARGET_COL],
                    s=15, cmap="viridis")
    ax.set_aspect("equal")
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_title("Spatial distribution of measurements (train)")
    fig.colorbar(sc, label="Measurement")
    _save(fig, _next("wafer_map"))


def run_duration_distribution(run_df: pd.DataFrame):
    if "RunStartTime" in run_df.columns and "RunEndTime" in run_df.columns:
        dur = (pd.to_datetime(run_df["RunEndTime"]) - pd.to_datetime(run_df["RunStartTime"]))
        dur_min = dur.dt.total_seconds() / 60
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(dur_min, bins=50, ax=ax)
        ax.set_xlabel("Run duration (minutes)")
        ax.set_title("Distribution of run durations")
        _save(fig, _next("run_duration"))


def top_sensor_histograms(run_df: pd.DataFrame, top_n: int = 4):
    top_sensors = run_df["SensorName"].value_counts().head(top_n).index
    for sensor in top_sensors:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(run_df.loc[run_df["SensorName"] == sensor, "SensorValue"], bins=60, ax=ax)
        ax.set_xlabel(f"{sensor} value")
        ax.set_title(f"Distribution of {sensor} readings")
        _save(fig, _next(f"sensor_{sensor}"))

# ---------------------------------------------------------------------------
#  Diagnostics after model training
# ---------------------------------------------------------------------------

def feature_importance_bar(model: Model):
    if model.feature_importance is None:
        logging.warning("feature_importance not found – did you train the model?")
        return
    imp = model.feature_importance.sort_values("importance", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(y="feature", x="importance", data=imp, ax=ax)
    ax.set_title("Top 30 feature importances (LightGBM)")
    _save(fig, _next("feature_importance"))


def prediction_vs_actual(y_true: pd.Series, y_pred: np.ndarray):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.3, edgecolors="none")
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], linestyle="--")
    ax.set_xlabel("Actual measurement")
    ax.set_ylabel("Predicted measurement")
    ax.set_title("Prediction vs actual on validation folds")
    _save(fig, _next("pred_vs_actual"))

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_eda(dp: DataProcessor):
    logging.info("Generating EDA figures …")
    metrology = dp.load_metrology_data(is_train=True).sample(frac=0.3, random_state=42)
    run_data = dp.load_run_data(is_train=True).sample(frac=0.1, random_state=42)
    measurement_distribution(metrology)
    wafer_map(metrology)
    run_duration_distribution(run_data)
    top_sensor_histograms(run_data)


def generate_diagnostics(model: Model, dp: DataProcessor):
    logging.info("Generating diagnostics figures …")
    feature_importance_bar(model)
    X_train, y_train = dp.prepare_train_data()
    y_pred = model.predict(X_train.iloc[:len(y_train)])
    prediction_vs_actual(y_train, y_pred)

# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(stage: str):
    dp = DataProcessor(DATA_DIR)

    if stage in {"eda", "all"}:
        generate_eda(dp)

    if stage in {"diagnostics", "all"}:
        mdl = Model(MODEL_DIR)
        try:
            mdl.load()
        except FileNotFoundError:
            logging.error("No trained model found – train model first or set stage=eda")
            return
        generate_diagnostics(mdl, dp)

    logging.info("✔ All requested figures have been generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for Micron AI Challenge report")
    parser.add_argument("--stage", choices=["eda", "diagnostics", "all"], default="eda",
                        help="Which set of figures to generate (default: eda)")
    args = parser.parse_args()
    main(args.stage)
