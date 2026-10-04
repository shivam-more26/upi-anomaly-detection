"""
Global configuration parameters and constants for UPI Anomaly Detection Pipeline.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "upi_transactions_raw.csv"
FEATURE_DATA_PATH = PROCESSED_DATA_DIR / "upi_transactions_features.csv"
SCORED_DATA_PATH = PROCESSED_DATA_DIR / "upi_transactions_scored.csv"
USER_PROFILES_PATH = RAW_DATA_DIR / "user_profiles.json"

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Data Generation Settings
NUM_USERS = 500
TOTAL_TRANSACTIONS = 15000
START_DATE = "2026-01-01 00:00:00"
END_DATE = "2026-01-31 23:59:59"
ANOMALY_RATIO = 0.025  # ~2.5% injected synthetic anomalies

# Categories & Options
MERCHANT_CATEGORIES = [
    "grocery",
    "fuel",
    "p2p_transfer",
    "bill_payment",
    "e_commerce",
    "dining",
    "travel",
    "entertainment",
]

LOCATIONS = [
    "Mumbai",
    "Delhi",
    "Bangalore",
    "Hyderabad",
    "Chennai",
    "Kolkata",
    "Pune",
    "Ahmedabad",
]

DEVICE_TYPES = [
    "Android_Samsung",
    "Android_Xiaomi",
    "Android_Realme",
    "iOS_iPhone",
    "Android_OnePlus",
]

PAYMENT_STATUSES = ["SUCCESS", "FAILED"]  # ~96% SUCCESS, 4% FAILED
TRANSACTION_TYPES = ["P2P", "P2M"]

# Thresholds for Anomaly Risk Categorization (Percentile-based)
HIGH_RISK_PERCENTILE = 98.0  # Top 2%
SUSPICIOUS_PERCENTILE = 95.0  # Top 5%

# Model Configurations
IFOREST_PARAMS = {
    "n_estimators": 100,
    "contamination": ANOMALY_RATIO,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

LOF_PARAMS = {
    "n_neighbors": 20,
    "contamination": ANOMALY_RATIO,
    "novelty": False,
    "n_jobs": -1,
}

DBSCAN_PARAMS = {
    "eps": 2.5,
    "min_samples": 15,
}

PYOD_IFOREST_PARAMS = {
    "n_estimators": 100,
    "contamination": ANOMALY_RATIO,
    "random_state": RANDOM_SEED,
}
