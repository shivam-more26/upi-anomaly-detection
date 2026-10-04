"""
Synthetic UPI Transaction Generator with realistic user behavioral profiles
and injected anomaly patterns for post-hoc evaluation.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from faker import Faker

import config
from src.utils import set_seed

fake = Faker("en_IN")


def generate_user_profiles(num_users: int = config.NUM_USERS) -> list[dict]:
    """
    Generate realistic user behavioral profiles with typical spend ranges,
    favorite merchant categories, home locations, active hours, and frequent receivers.
    """
    bank_handles = ["@upi", "@okaxis", "@ybl", "@paytm", "@icici", "@hdfc"]
    profiles = []

    for i in range(num_users):
        name = fake.first_name().lower() + str(random.randint(10, 99))
        vpa = f"{name}{random.choice(bank_handles)}"

        # Generate typical transaction amount range using Log-Normal distribution logic
        base_amount = np.random.choice([150, 450, 1200, 3500, 8500], p=[0.35, 0.35, 0.18, 0.09, 0.03])
        mean_amount = float(np.round(base_amount * np.random.uniform(0.8, 1.3), 2))
        std_amount = float(np.round(mean_amount * np.random.uniform(0.25, 0.45), 2))

        # Preferred categories & frequent receivers
        pref_cats = random.sample(config.MERCHANT_CATEGORIES, k=random.randint(2, 4))
        receivers = [f"merchant_{fake.domain_word()}{random.choice(bank_handles)}" for _ in range(random.randint(3, 7))]

        # Active hours range (e.g., 7 AM to 10 PM)
        start_hour = random.randint(6, 9)
        end_hour = random.randint(21, 23)

        profiles.append(
            {
                "user_id": f"USR_{i+1:04d}",
                "sender_vpa": vpa,
                "home_location": random.choice(config.LOCATIONS),
                "primary_device": random.choice(config.DEVICE_TYPES),
                "mean_amount": mean_amount,
                "std_amount": std_amount,
                "favorite_categories": pref_cats,
                "frequent_receivers": receivers,
                "active_start_hour": start_hour,
                "active_end_hour": end_hour,
            }
        )

    return profiles


def generate_synthetic_transactions(
    profiles: list[dict],
    total_txns: int = config.TOTAL_TRANSACTIONS,
    anomaly_ratio: float = config.ANOMALY_RATIO,
) -> pd.DataFrame:
    """
    Generate clean, chronological UPI transaction records based on user profiles,
    and inject ~2.5% realistic behavioral anomalies.
    """
    start_dt = pd.to_datetime(config.START_DATE)
    end_dt = pd.to_datetime(config.END_DATE)
    total_seconds = int((end_dt - start_dt).total_seconds())

    txns = []

    # Assign target normal vs anomaly counts
    num_anomalies = int(total_txns * anomaly_ratio)
    num_normal = total_txns - num_anomalies

    # 1. Generate Normal Transactions
    for i in range(num_normal):
        user = random.choice(profiles)

        # Timestamp within active hours
        rand_sec = random.randint(0, total_seconds)
        txn_dt = start_dt + timedelta(seconds=rand_sec)

        # Force normal timestamps to align mostly with active hours
        if random.random() < 0.85:
            hour = random.randint(user["active_start_hour"], user["active_end_hour"])
            txn_dt = txn_dt.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))

        # Amount around mean
        amount = float(np.round(np.clip(np.random.normal(user["mean_amount"], user["std_amount"]), 10, user["mean_amount"] * 3.5), 2))

        # Category and Receiver
        category = random.choice(user["favorite_categories"]) if random.random() < 0.8 else random.choice(config.MERCHANT_CATEGORIES)
        receiver = random.choice(user["frequent_receivers"]) if random.random() < 0.85 else f"user_{fake.word()}@upi"

        # Type, Device, Location
        txn_type = "P2M" if category in ["grocery", "fuel", "e_commerce", "dining", "bill_payment"] else "P2P"
        device = user["primary_device"] if random.random() < 0.95 else random.choice(config.DEVICE_TYPES)
        location = user["home_location"] if random.random() < 0.92 else random.choice(config.LOCATIONS)
        status = "SUCCESS" if random.random() < 0.96 else "FAILED"

        txns.append(
            {
                "sender_vpa": user["sender_vpa"],
                "receiver_vpa": receiver,
                "amount": amount,
                "timestamp": txn_dt,
                "merchant_category": category,
                "transaction_type": txn_type,
                "device_id": device,
                "location": location,
                "payment_status": status,
                "synthetic_anomaly_truth": 0,
                "anomaly_type": "NONE",
            }
        )

    # 2. Inject Realistic Anomaly Scenarios
    anomaly_patterns = [
        "AMOUNT_SPIKE",
        "VELOCITY_BURST",
        "NEW_RECEIVER",
        "NEW_DEVICE",
        "LOCATION_JUMP",
        "OFF_HOURS",
        "COMBINED_SUSPICIOUS",
    ]

    for i in range(num_anomalies):
        user = random.choice(profiles)
        pattern = random.choice(anomaly_patterns)

        rand_sec = random.randint(0, total_seconds)
        txn_dt = start_dt + timedelta(seconds=rand_sec)

        amount = float(np.round(user["mean_amount"] * np.random.uniform(1.1, 2.5), 2))
        category = random.choice(user["favorite_categories"])
        receiver = random.choice(user["frequent_receivers"])
        device = user["primary_device"]
        location = user["home_location"]
        status = "SUCCESS"

        if pattern == "AMOUNT_SPIKE":
            # 8x to 22x user's mean transaction amount
            amount = float(np.round(user["mean_amount"] * np.random.uniform(8.0, 22.0), 2))

        elif pattern == "VELOCITY_BURST":
            # Rapid transaction sequence in midnight or sudden burst
            amount = float(np.round(user["mean_amount"] * np.random.uniform(2.0, 5.0), 2))

        elif pattern == "NEW_RECEIVER":
            # High amount to completely unknown receiver VPA
            receiver = f"suspicious_unknown_{random.randint(10000, 99999)}@okfraud"
            amount = float(np.round(user["mean_amount"] * np.random.uniform(5.0, 12.0), 2))

        elif pattern == "NEW_DEVICE":
            # Unseen rogue device
            device = f"RogueDevice_{random.randint(100, 999)}"

        elif pattern == "LOCATION_JUMP":
            # Location suddenly jumps across country
            other_locations = [loc for loc in config.LOCATIONS if loc != user["home_location"]]
            location = random.choice(other_locations)

        elif pattern == "OFF_HOURS":
            # Transaction at 3:15 AM for daytime user
            txn_dt = txn_dt.replace(hour=random.randint(2, 4), minute=random.randint(10, 50))

        elif pattern == "COMBINED_SUSPICIOUS":
            # Multiple red flags simultaneously
            amount = float(np.round(user["mean_amount"] * np.random.uniform(10.0, 25.0), 2))
            txn_dt = txn_dt.replace(hour=random.randint(2, 4), minute=random.randint(10, 50))
            device = f"RogueDevice_{random.randint(100, 999)}"
            receiver = f"scam_account_{random.randint(1000, 9999)}@ybl"
            other_locations = [loc for loc in config.LOCATIONS if loc != user["home_location"]]
            location = random.choice(other_locations)

        txn_type = "P2M" if category in ["grocery", "fuel", "e_commerce", "dining", "bill_payment"] else "P2P"

        txns.append(
            {
                "sender_vpa": user["sender_vpa"],
                "receiver_vpa": receiver,
                "amount": amount,
                "timestamp": txn_dt,
                "merchant_category": category,
                "transaction_type": txn_type,
                "device_id": device,
                "location": location,
                "payment_status": status,
                "synthetic_anomaly_truth": 1,
                "anomaly_type": pattern,
            }
        )

    df = pd.DataFrame(txns)

    # Sort strictly by timestamp to maintain chronological order
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Assign sequential transaction IDs
    df.insert(0, "transaction_id", [f"TXN{i+1:08d}" for i in range(len(df))])

    return df


def main():
    print("=" * 60)
    print("PHASE 2: Generating Synthetic UPI Dataset...")
    print("=" * 60)
    set_seed(config.RANDOM_SEED)

    print(f"Generating profiles for {config.NUM_USERS} users...")
    profiles = generate_user_profiles(config.NUM_USERS)

    # Save profiles JSON for reference
    with open(config.USER_PROFILES_PATH, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"User profiles saved to {config.USER_PROFILES_PATH}")

    print(f"Generating {config.TOTAL_TRANSACTIONS} transactions...")
    df = generate_synthetic_transactions(profiles, config.TOTAL_TRANSACTIONS, config.ANOMALY_RATIO)

    # Save raw CSV
    df.to_csv(config.RAW_DATA_PATH, index=False)
    print(f"Raw dataset successfully saved to: {config.RAW_DATA_PATH}")

    # Dataset Summary
    anomaly_cnt = df["synthetic_anomaly_truth"].sum()
    print("\nDataset Generation Summary:")
    print(f"- Total Records: {len(df)}")
    print(f"- Unique Senders: {df['sender_vpa'].nunique()}")
    print(f"- Total Anomalies Injected: {anomaly_cnt} ({anomaly_cnt / len(df) * 100:.2f}%)")
    print("\nInjected Anomaly Type Distribution:")
    print(df[df["synthetic_anomaly_truth"] == 1]["anomaly_type"].value_counts().to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()
