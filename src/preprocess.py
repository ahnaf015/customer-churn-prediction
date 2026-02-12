import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 1: Clean raw data (before feature engineering)."""

    df = df.copy()

    # TotalCharges has whitespace strings for new customers — coerce to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Encode binary target
    if df["Churn"].dtype == object:
        df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 2: Encode categoricals (after feature engineering).

    Uses binary encoding for Yes/No cols, and one-hot for multi-class cols.
    """

    df = df.copy()

    # Binary columns: simple 0/1 mapping
    binary_cols = [
        "gender", "Partner", "Dependents", "PhoneService",
        "PaperlessBilling",
    ]
    le = LabelEncoder()
    for col in binary_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # Multi-class columns: one-hot encode
    multi_class_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod",
    ]
    existing = [c for c in multi_class_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing, drop_first=True, dtype=int)

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: clean -> encode (without feature engineering)."""
    df = clean_data(df)
    return encode_categoricals(df)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all feature columns from the processed DataFrame.

    Dynamically picks up one-hot encoded columns.
    """

    exclude = {"customerID", "Churn", "tenure_group", "charges_bin"}
    return [c for c in df.columns if c not in exclude]
