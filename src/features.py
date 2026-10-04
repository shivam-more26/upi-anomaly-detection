"""
Feature Engineering Engine for UPI Anomaly Detection.
Constructs velocity, amount z-score, behavioral entropy, temporal, and device/location
features without data leakage (using past historical data strictly up to the current transaction).
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import math
import numpy as np
import pandas as pd
from collections import Counter

import config
from src.utils import load_raw_data, haversine_distance, set_seed


def calculate_entropy(categories: list[str]) -> float:
    """Calculate Shannon entropy for a list of spending categories."""
    if not categories:
        return 0.0
    counts = Counter(categories)
    total = len(categories)
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / total
        entropy -= p * math.log2(p)
    return float(np.round(entropy, 4))


def compute_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute velocity features per sender:
    - txns_last_1h, txns_last_24h, txns_last_7d
    - time_since_prev_txn_sec
    - velocity_spike_ratio
    """
    df = df.sort_values(["sender_vpa", "timestamp"]).reset_index(drop=True)

    # Time since previous transaction per sender
    df["prev_timestamp"] = df.groupby("sender_vpa")["timestamp"].shift(1)
    df["time_since_prev_txn_sec"] = (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds()
    df["time_since_prev_txn_sec"] = df["time_since_prev_txn_sec"].fillna(86400.0)  # Default 24h for first txn
    df.drop(columns=["prev_timestamp"], inplace=True)

    # Rolling time windows per sender using past timestamps search
    txns_1h = []
    txns_24h = []
    txns_7d = []

    for sender, group in df.groupby("sender_vpa", sort=False):
        group_ts = group["timestamp"].values
        for ts in group_ts:
            ts_ns = ts.astype("datetime64[ns]")
            ts_1h_ago = ts_ns - np.timedelta64(1, "h")
            ts_24h_ago = ts_ns - np.timedelta64(24, "h")
            ts_7d_ago = ts_ns - np.timedelta64(7, "D")

            cnt_1h = np.searchsorted(group_ts, ts_ns, side="right") - np.searchsorted(group_ts, ts_1h_ago, side="left") - 1
            cnt_24h = np.searchsorted(group_ts, ts_ns, side="right") - np.searchsorted(group_ts, ts_24h_ago, side="left") - 1
            cnt_7d = np.searchsorted(group_ts, ts_ns, side="right") - np.searchsorted(group_ts, ts_7d_ago, side="left") - 1

            txns_1h.append(max(0, cnt_1h))
            txns_24h.append(max(0, cnt_24h))
            txns_7d.append(max(0, cnt_7d))

    df["txns_last_1h"] = txns_1h
    df["txns_last_24h"] = txns_24h
    df["txns_last_7d"] = txns_7d

    # Spike ratio: 1-hour count vs average hourly count over last 7 days
    expected_hourly = (df["txns_last_7d"] / 168.0) + 0.05
    df["velocity_spike_ratio"] = np.round(df["txns_last_1h"] / expected_hourly, 4)

    return df


def compute_amount_features(df: pd.DataFrame, user_profiles_dict: dict) -> pd.DataFrame:
    """
    Compute amount features relative to user's historical profile and rolling window:
    - amount_zscore
    - amount_to_user_avg_ratio
    - rolling_amount_std_5
    """
    df["user_mean_amount"] = df["sender_vpa"].map(lambda x: user_profiles_dict.get(x, {}).get("mean_amount", 500.0))
    df["user_std_amount"] = df["sender_vpa"].map(lambda x: user_profiles_dict.get(x, {}).get("std_amount", 150.0))

    # Amount Z-score
    df["amount_zscore"] = np.round((df["amount"] - df["user_mean_amount"]) / (df["user_std_amount"] + 1e-5), 4)

    # Ratio of transaction amount to typical mean amount
    df["amount_to_user_avg_ratio"] = np.round(df["amount"] / (df["user_mean_amount"] + 1e-5), 4)

    # Rolling std dev over last 5 transactions per sender
    df["rolling_amount_std_5"] = (
        df.groupby("sender_vpa")["amount"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).std())
        .fillna(df["user_std_amount"])
        .round(4)
    )

    return df


def compute_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute behavioral novelty flags and category entropy per sender up to prior transaction:
    - is_new_merchant_category
    - is_new_receiver
    - merchant_category_entropy
    """
    is_new_cat = []
    is_new_rec = []
    cat_entropy = []

    seen_categories = {}
    seen_receivers = {}
    history_categories = {}

    for idx, row in df.iterrows():
        sender = row["sender_vpa"]
        cat = row["merchant_category"]
        rec = row["receiver_vpa"]

        if sender not in seen_categories:
            seen_categories[sender] = set()
            seen_receivers[sender] = set()
            history_categories[sender] = []

        new_c = 1 if (len(seen_categories[sender]) > 0 and cat not in seen_categories[sender]) else 0
        new_r = 1 if (len(seen_receivers[sender]) > 0 and rec not in seen_receivers[sender]) else 0

        ent = calculate_entropy(history_categories[sender])

        is_new_cat.append(new_c)
        is_new_rec.append(new_r)
        cat_entropy.append(ent)

        seen_categories[sender].add(cat)
        seen_receivers[sender].add(rec)
        history_categories[sender].append(cat)

    df["is_new_merchant_category"] = is_new_cat
    df["is_new_receiver"] = is_new_rec
    df["merchant_category_entropy"] = cat_entropy

    return df


def compute_temporal_features(df: pd.DataFrame, user_profiles_dict: dict) -> pd.DataFrame:
    """
    Compute temporal features and deviation from active hours:
    - hour_of_day, day_of_week, is_weekend
    - active_hour_deviation
    """
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(lambda d: 1 if d >= 5 else 0)

    # Active hours bounds
    start_hours = df["sender_vpa"].map(lambda x: user_profiles_dict.get(x, {}).get("active_start_hour", 7))
    end_hours = df["sender_vpa"].map(lambda x: user_profiles_dict.get(x, {}).get("active_end_hour", 22))

    deviations = []
    for h, start_h, end_h in zip(df["hour_of_day"], start_hours, end_hours):
        if start_h <= h <= end_h:
            deviations.append(0.0)
        elif h < start_h:
            deviations.append(float(start_h - h))
        else:
            deviations.append(float(h - end_h))

    df["active_hour_deviation"] = deviations
    return df


def compute_device_location_features(df: pd.DataFrame, user_profiles_dict: dict) -> pd.DataFrame:
    """
    Compute device and location anomaly indicators:
    - is_new_device
    - is_new_location
    - distance_from_home_km
    - unique_devices_last_24h
    """
    is_new_dev = []
    is_new_loc = []

    seen_devices = {}
    seen_locations = {}

    for idx, row in df.iterrows():
        sender = row["sender_vpa"]
        dev = row["device_id"]
        loc = row["location"]

        if sender not in seen_devices:
            seen_devices[sender] = set()
            seen_locations[sender] = set()

        new_d = 1 if (len(seen_devices[sender]) > 0 and dev not in seen_devices[sender]) else 0
        new_l = 1 if (len(seen_locations[sender]) > 0 and loc not in seen_locations[sender]) else 0

        is_new_dev.append(new_d)
        is_new_loc.append(new_l)

        seen_devices[sender].add(dev)
        seen_locations[sender].add(loc)

    df["is_new_device"] = is_new_dev
    df["is_new_location"] = is_new_loc

    # Distance from home location in KM
    home_locations = df["sender_vpa"].map(lambda x: user_profiles_dict.get(x, {}).get("home_location", "Mumbai"))
    distances = [haversine_distance(loc, home_loc) for loc, home_loc in zip(df["location"], home_locations)]
    df["distance_from_home_km"] = np.round(distances, 2)

    # Unique devices used in last 24h per sender
    unique_dev_24h = []
    for sender, group in df.groupby("sender_vpa", sort=False):
        ts_list = group["timestamp"].values
        dev_list = group["device_id"].values
        for i, ts in enumerate(ts_list):
            ts_ns = ts.astype("datetime64[ns]")
            ts_24h_ago = ts_ns - np.timedelta64(24, "h")
            idx_start = np.searchsorted(ts_list, ts_24h_ago, side="left")
            unique_dev_cnt = len(set(dev_list[idx_start : i + 1]))
            unique_dev_24h.append(unique_dev_cnt)

    df["unique_devices_last_24h"] = unique_dev_24h
    return df


def engineer_all_features() -> pd.DataFrame:
    """Full feature engineering pipeline loading raw data and applying feature extractors."""
    print("=" * 60)
    print("PHASE 3: Feature Engineering Pipeline...")
    print("=" * 60)

    df = load_raw_data()
    print(f"Loaded raw transaction dataset: {len(df)} rows.")

    # Load user profiles dictionary
    if config.USER_PROFILES_PATH.exists():
        with open(config.USER_PROFILES_PATH, "r") as f:
            profiles = json.load(f)
        user_profiles_dict = {p["sender_vpa"]: p for p in profiles}
    else:
        user_profiles_dict = {}

    print("1. Computing Velocity Features...")
    df = compute_velocity_features(df)

    print("2. Computing Amount & Z-Score Features...")
    df = compute_amount_features(df, user_profiles_dict)

    print("3. Computing Behavioral & Entropy Features...")
    df = compute_behavioral_features(df)

    print("4. Computing Temporal & Active Hour Features...")
    df = compute_temporal_features(df, user_profiles_dict)

    print("5. Computing Device & Geographic Distance Features...")
    df = compute_device_location_features(df, user_profiles_dict)

    # Re-sort strictly by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Save engineered feature dataset
    df.to_csv(config.FEATURE_DATA_PATH, index=False)
    print(f"\nFeature engineering complete! Saved to {config.FEATURE_DATA_PATH}")

    # Feature Summary
    feature_cols = [
        "txns_last_1h", "txns_last_24h", "txns_last_7d", "time_since_prev_txn_sec",
        "velocity_spike_ratio", "amount_zscore", "amount_to_user_avg_ratio",
        "rolling_amount_std_5", "is_new_merchant_category", "is_new_receiver",
        "merchant_category_entropy", "hour_of_day", "day_of_week", "is_weekend",
        "active_hour_deviation", "is_new_device", "is_new_location",
        "distance_from_home_km", "unique_devices_last_24h"
    ]
    print(f"Total Engineered Features: {len(feature_cols)}")
    print("Engineered Feature List:")
    for f in feature_cols:
        print(f" - {f}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    engineer_all_features()
