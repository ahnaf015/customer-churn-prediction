"""Train the XGBoost churn prediction model."""

import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "config"))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier
import joblib

from settings import (
    RAW_DATA_FILE,
    MODELS_DIR,
    MODEL_FILE,
    FEATURE_COLS_FILE,
    FEATURE_IMPORTANCE_FILE,
    MODEL_PARAMS,
)
from preprocess import clean_data, encode_categoricals, get_feature_columns
from feature_engineering import engineer_features


def train_model():
    """Train XGBoost churn prediction model and save artifacts."""

    # --- Load ---
    print("Loading data...")
    df = pd.read_csv(RAW_DATA_FILE)
    print(f"Dataset shape: {df.shape}")
    print(f"Churn rate: {df['Churn'].value_counts(normalize=True).to_dict()}")

    # --- Pipeline: clean -> feature engineer -> encode ---
    print("\nCleaning data...")
    df = clean_data(df)

    print("Engineering features...")
    df = engineer_features(df)

    print("Encoding categoricals...")
    df = encode_categoricals(df)

    # Dynamic feature selection (picks up one-hot columns)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["Churn"]

    print(f"Features: {len(feature_cols)} columns")

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    # --- Train ---
    print("\nTraining XGBoost model...")
    model = XGBClassifier(**MODEL_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # --- Evaluate ---
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"\nAUC Score: {auc:.3f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Probability distribution check
    print(f"\nProbability distribution:")
    print(f"  Min: {y_pred_proba.min():.3f}  Max: {y_pred_proba.max():.3f}  Mean: {y_pred_proba.mean():.3f}")
    print(f"  > 0.7 (High Risk): {(y_pred_proba > 0.7).sum()}")
    print(f"  0.3-0.7 (Medium):  {((y_pred_proba >= 0.3) & (y_pred_proba <= 0.7)).sum()}")
    print(f"  < 0.3 (Low Risk):  {(y_pred_proba < 0.3).sum()}")

    # Feature importance
    feature_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"\nTop 10 Important Features:")
    print(feature_importance.head(10).to_string(index=False))

    # Cross-validation
    print("\n5-Fold Cross-Validation AUC:")
    cv_params = {k: v for k, v in MODEL_PARAMS.items() if k != "early_stopping_rounds"}
    cv_model = XGBClassifier(**cv_params)
    cv_scores = cross_val_score(cv_model, X, y, cv=5, scoring="roc_auc")
    print(f"Mean AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # --- Save artifacts ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(feature_cols, FEATURE_COLS_FILE)
    joblib.dump(feature_importance, FEATURE_IMPORTANCE_FILE)

    print(f"\nModel saved: {MODEL_FILE}")
    print(f"Features saved: {FEATURE_COLS_FILE}")
    print(f"Final AUC: {auc:.3f}")

    return model, auc


if __name__ == "__main__":
    train_model()
