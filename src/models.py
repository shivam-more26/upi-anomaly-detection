"""
Preprocessing, Unsupervised Anomaly Detection Modeling, Thresholding, Model Overlap,
and Deterministic Risk Explanation Engine.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, RobustScaler

import pyod.models.ecod as pyod_ecod
import config
from src.utils import load_featured_data, set_seed


def preprocess_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Prepare numeric scaling and one-hot encoding for the feature matrix.
    Explicitly checks that ground truth anomaly columns are excluded to prevent data leakage.
    """
    forbidden_cols = ["synthetic_anomaly_truth", "anomaly_type", "transaction_id", "sender_vpa", "receiver_vpa", "timestamp"]

    # Select numerical engineered features
    numeric_feature_cols = [
        "txns_last_1h", "txns_last_24h", "txns_last_7d", "time_since_prev_txn_sec",
        "velocity_spike_ratio", "amount_zscore", "amount_to_user_avg_ratio",
        "rolling_amount_std_5", "is_new_merchant_category", "is_new_receiver",
        "merchant_category_entropy", "hour_of_day", "day_of_week", "is_weekend",
        "active_hour_deviation", "is_new_device", "is_new_location",
        "distance_from_home_km", "unique_devices_last_24h"
    ]

    categorical_cols = ["merchant_category", "transaction_type", "payment_status"]

    # Data Leakage Safeguard Check
    for col in forbidden_cols:
        assert col not in numeric_feature_cols, f"DATA LEAKAGE WARNING: {col} found in feature matrix!"

    # 1. Robust Scaling for Numerical Features
    scaler = RobustScaler()
    scaled_numeric = scaler.fit_transform(df[numeric_feature_cols])

    # 2. One-Hot Encoding for Categorical Features
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_cat = ohe.fit_transform(df[categorical_cols])
    cat_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()

    # Combine numerical & categorical matrices
    X = np.hstack([scaled_numeric, encoded_cat])
    feature_names = numeric_feature_cols + cat_feature_names

    return X, feature_names


def normalize_scores(raw_scores: np.ndarray) -> np.ndarray:
    """Normalize raw anomaly scores into a standard [0, 1] range."""
    min_s, max_s = raw_scores.min(), raw_scores.max()
    if max_s - min_s == 0:
        return np.zeros_like(raw_scores)
    return np.round((raw_scores - min_s) / (max_s - min_s), 4)


def train_isolation_forest(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Train Isolation Forest model. Higher score = more anomalous."""
    model = IsolationForest(**config.IFOREST_PARAMS)
    model.fit(X)
    # Raw decision function: negative values indicate outliers
    raw_scores = -model.score_samples(X)
    norm_scores = normalize_scores(raw_scores)
    predictions = (norm_scores >= np.percentile(norm_scores, config.SUSPICIOUS_PERCENTILE)).astype(int)
    return norm_scores, predictions


def train_local_outlier_factor(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Train Local Outlier Factor (LOF) model."""
    lof = LocalOutlierFactor(**config.LOF_PARAMS)
    lof.fit_predict(X)
    raw_scores = -lof.negative_outlier_factor_
    norm_scores = normalize_scores(raw_scores)
    predictions = (norm_scores >= np.percentile(norm_scores, config.SUSPICIOUS_PERCENTILE)).astype(int)
    return norm_scores, predictions


def train_dbscan(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Train DBSCAN clustering model.
    Outliers (noise points) receive cluster label -1.
    Anomaly score is derived from k-distance to nearest cluster core point.
    """
    dbscan = DBSCAN(**config.DBSCAN_PARAMS)
    labels = dbscan.fit_predict(X)

    # Compute distance to nearest non-noise point as continuous anomaly score
    nbrs = NearestNeighbors(n_neighbors=5, n_jobs=-1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    mean_k_dist = distances.mean(axis=1)

    # Multiply distance for DBSCAN noise points (-1)
    raw_scores = np.where(labels == -1, mean_k_dist * 2.5, mean_k_dist)
    norm_scores = normalize_scores(raw_scores)
    predictions = (labels == -1).astype(int)
    return norm_scores, predictions


def train_pyod_ecod(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Train PyOD ECOD (Empirical Cumulative Distribution Functions for Anomaly Detection) detector."""
    detector = pyod_ecod.ECOD(contamination=config.ANOMALY_RATIO)
    detector.fit(X)
    raw_scores = detector.decision_scores_
    norm_scores = normalize_scores(raw_scores)
    predictions = (norm_scores >= np.percentile(norm_scores, config.SUSPICIOUS_PERCENTILE)).astype(int)
    return norm_scores, predictions


def generate_anomaly_reasons(row: pd.Series) -> str:
    """
    Deterministic Explanation Engine: Converts feature anomalies into human-readable risk signals.
    """
    reasons = []

    # Amount & Ratio Signals
    if row["amount_to_user_avg_ratio"] >= 4.0:
        reasons.append(f"Amount spike ({row['amount_to_user_avg_ratio']:.1f}x higher than user average)")
    elif row["amount_zscore"] >= 3.0:
        reasons.append(f"High Z-score amount deviation (+{row['amount_zscore']:.1f} std dev)")

    # Velocity Signals
    if row["velocity_spike_ratio"] >= 4.0:
        reasons.append(f"High transaction velocity ({row['txns_last_1h']} txns in last 1hr)")
    elif row["time_since_prev_txn_sec"] <= 30 and row["time_since_prev_txn_sec"] > 0:
        reasons.append(f"Rapid consecutive transaction ({int(row['time_since_prev_txn_sec'])}s delta)")

    # Novelty Flags
    if row["is_new_receiver"] == 1:
        reasons.append("First payment to unseen receiver VPA")
    if row["is_new_device"] == 1:
        reasons.append("Transaction initiated from new device fingerprint")
    if row["is_new_merchant_category"] == 1 and row["amount_to_user_avg_ratio"] >= 2.0:
        reasons.append("Unusual high-value category transition")

    # Temporal & Spatial Signals
    if row["active_hour_deviation"] >= 2.0:
        reasons.append(f"Off-hours transaction ({int(row['hour_of_day'])}:00 outside active window)")
    if row["distance_from_home_km"] >= 300:
        reasons.append(f"Geographic jump ({int(row['distance_from_home_km'])} km from home location)")

    if not reasons:
        return "Normal behavioral pattern"
    return " | ".join(reasons)


def run_modeling_pipeline() -> pd.DataFrame:
    """Full execution of preprocessing, training all detectors, thresholding, and PCA projection."""
    print("=" * 60)
    print("PHASE 4 & 5: Model Training, Standardization & Anomaly Scoring...")
    print("=" * 60)
    set_seed(config.RANDOM_SEED)

    df = load_featured_data()
    print(f"Loaded featured dataset: {len(df)} rows.")

    print("1. Preprocessing and Scaling Feature Matrix...")
    X, feature_names = preprocess_features(df)
    print(f"Feature matrix X shape: {X.shape}")

    print("2. Training Isolation Forest...")
    if_scores, if_preds = train_isolation_forest(X)

    print("3. Training Local Outlier Factor (LOF)...")
    lof_scores, lof_preds = train_local_outlier_factor(X)

    print("4. Training DBSCAN Clustering Detector...")
    dbscan_scores, dbscan_preds = train_dbscan(X)

    print("5. Training PyOD ECOD Detector...")
    pyod_scores, pyod_preds = train_pyod_ecod(X)

    # Assign individual model scores & predictions
    df["score_isolation_forest"] = if_scores
    df["pred_isolation_forest"] = if_preds

    df["score_lof"] = lof_scores
    df["pred_lof"] = lof_preds

    df["score_dbscan"] = dbscan_scores
    df["pred_dbscan"] = dbscan_preds

    df["score_pyod_ecod"] = pyod_scores
    df["pred_pyod_ecod"] = pyod_preds

    # Ensemble Primary Anomaly Score (Weighted Average prioritizing Isolation Forest)
    df["anomaly_score"] = np.round(
        (0.40 * if_scores + 0.25 * lof_scores + 0.20 * pyod_scores + 0.15 * dbscan_scores), 4
    )

    # Percentile-based Risk Categorization (No Ground Truth Leakage!)
    high_threshold = np.percentile(df["anomaly_score"], config.HIGH_RISK_PERCENTILE)
    suspicious_threshold = np.percentile(df["anomaly_score"], config.SUSPICIOUS_PERCENTILE)

    def assign_risk_level(score: float) -> str:
        if score >= high_threshold:
            return "HIGH_RISK"
        elif score >= suspicious_threshold:
            return "SUSPICIOUS"
        return "NORMAL"

    df["risk_level"] = df["anomaly_score"].apply(assign_risk_level)
    df["is_anomaly"] = (df["risk_level"] != "NORMAL").astype(int)

    print("\n6. Generating Deterministic Anomaly Reasons...")
    df["anomaly_reasons"] = df.apply(generate_anomaly_reasons, axis=1)

    print("7. Computing 2D PCA Projections for Visualization...")
    pca = PCA(n_components=2, random_state=config.RANDOM_SEED)
    pca_coords = pca.fit_transform(X)
    df["pca_x"] = np.round(pca_coords[:, 0], 4)
    df["pca_y"] = np.round(pca_coords[:, 1], 4)

    # Save final scored dataset
    df.to_csv(config.SCORED_DATA_PATH, index=False)
    print(f"\nScored dataset successfully saved to: {config.SCORED_DATA_PATH}")

    # Model Summary & Overlap Statistics
    print("\nModel Flagging & Overlap Summary:")
    print(f"- Total High Risk (Top 2%): {(df['risk_level'] == 'HIGH_RISK').sum()}")
    print(f"- Total Suspicious (Top 5%): {(df['risk_level'] == 'SUSPICIOUS').sum()}")
    print(f"- Total Normal: {(df['risk_level'] == 'NORMAL').sum()}")

    # Model overlap check
    if_flagged = set(df[df["pred_isolation_forest"] == 1].index)
    lof_flagged = set(df[df["pred_lof"] == 1].index)
    dbscan_flagged = set(df[df["pred_dbscan"] == 1].index)

    if_lof_overlap = len(if_flagged.intersection(lof_flagged)) / max(1, len(if_flagged)) * 100
    if_dbscan_overlap = len(if_flagged.intersection(dbscan_flagged)) / max(1, len(if_flagged)) * 100

    print(f"- Isolation Forest & LOF Overlap: {if_lof_overlap:.1f}%")
    print(f"- Isolation Forest & DBSCAN Overlap: {if_dbscan_overlap:.1f}%")

    # Post-hoc synthetic truth sanity check
    syn_truth = df["synthetic_anomaly_truth"]
    top_2pct_indices = set(df[df["risk_level"] == "HIGH_RISK"].index)
    syn_indices = set(df[df["synthetic_anomaly_truth"] == 1].index)
    captured = len(top_2pct_indices.intersection(syn_indices))
    print(f"\nSynthetic Anomaly Sanity Check (Post-Hoc Verification Only):")
    print(f"- Top 2% High-Risk flags captured {captured} / {len(syn_indices)} injected synthetic anomalies.")
    print("=" * 60)

    return df


if __name__ == "__main__":
    run_modeling_pipeline()
