"""Generate SHAP explanations for the churn prediction model."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "config"))

import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from settings import (
    RAW_DATA_FILE,
    MODEL_FILE,
    FEATURE_COLS_FILE,
    SHAP_EXPLAINER_FILE,
    SHAP_VALUES_FILE,
    SHAP_SUMMARY_PLOT,
    REPORTS_DIR,
)
from preprocess import clean_data, encode_categoricals, get_feature_columns
from feature_engineering import engineer_features


def create_shap_explainer():
    """Create and save SHAP explainer + summary plot."""

    print("Loading model and data...")
    model = joblib.load(MODEL_FILE)
    feature_cols = joblib.load(FEATURE_COLS_FILE)

    df = pd.read_csv(RAW_DATA_FILE)
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_categoricals(df)

    X = df[feature_cols].head(200)  # Sample for speed

    print("Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Save artifacts
    joblib.dump(explainer, SHAP_EXPLAINER_FILE)
    joblib.dump(shap_values, SHAP_VALUES_FILE)

    # Generate summary plot
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PLOT, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"SHAP explainer saved: {SHAP_EXPLAINER_FILE}")
    print(f"Summary plot saved: {SHAP_SUMMARY_PLOT}")

    return explainer


if __name__ == "__main__":
    create_shap_explainer()
