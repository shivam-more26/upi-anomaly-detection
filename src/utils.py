"""
Utility functions for logging, random seed control, location distance calculations,
data loading, and saving artifacts.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import random
import numpy as np
import pandas as pd
import config


def set_seed(seed: int = config.RANDOM_SEED):
    """Set random seed for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)


def load_raw_data() -> pd.DataFrame:
    """Load raw transaction dataset from CSV."""
    if not config.RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {config.RAW_DATA_PATH}. Run src/simulate_data.py first."
        )
    df = pd.read_csv(config.RAW_DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_featured_data() -> pd.DataFrame:
    """Load feature-engineered dataset from CSV."""
    if not config.FEATURE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Featured dataset not found at {config.FEATURE_DATA_PATH}. Run src/features.py first."
        )
    df = pd.read_csv(config.FEATURE_DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_scored_data() -> pd.DataFrame:
    """Load anomaly-scored dataset from CSV."""
    if not config.SCORED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Scored dataset not found at {config.SCORED_DATA_PATH}. Run src/models.py first."
        )
    df = pd.read_csv(config.SCORED_DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# Coordinates for city location distance calculation (in KM)
CITY_COORDINATES = {
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.7041, 77.1025),
    "Bangalore": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
}


def haversine_distance(city1: str, city2: str) -> float:
    """
    Calculate the great-circle distance (in kilometers) between two Indian cities.
    Returns 0.0 if cities are the same or coordinates unavailable.
    """
    if city1 == city2:
        return 0.0
    if city1 not in CITY_COORDINATES or city2 not in CITY_COORDINATES:
        return 500.0  # Default fallback distance for unmapped cities

    lat1, lon1 = np.radians(CITY_COORDINATES[city1])
    lat2, lon2 = np.radians(CITY_COORDINATES[city2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371.0  # Radius of Earth in kilometers
    return float(r * c)
