import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Directory paths
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
CHROMA_DIR = ROOT_DIR / "chroma_db"

# Dataset
RAW_DATA_FILE = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CUSTOMER_HISTORIES_FILE = DATA_DIR / "customer_histories.csv"

# Model artifacts
MODEL_FILE = MODELS_DIR / "churn_model.pkl"
FEATURE_COLS_FILE = MODELS_DIR / "feature_cols.pkl"
FEATURE_IMPORTANCE_FILE = MODELS_DIR / "feature_importance.pkl"
SHAP_EXPLAINER_FILE = MODELS_DIR / "shap_explainer.pkl"
SHAP_VALUES_FILE = MODELS_DIR / "shap_values.pkl"

# Reports
SHAP_SUMMARY_PLOT = REPORTS_DIR / "shap_summary.png"

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model hyperparameters
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "eval_metric": "auc",
    "early_stopping_rounds": 50,
}

# RAG settings
RAG_COLLECTION_NAME = "customer_histories"
RAG_BATCH_SIZE = 100
HISTORY_SAMPLE_SIZE = 500

# Gemini settings
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_TEMPERATURE = 0.7
