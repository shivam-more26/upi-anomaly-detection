# UPI Digital Payment Anomaly Detection Engine

An unsupervised machine learning pipeline and transaction risk monitoring system for detecting behavioral anomalies in Unified Payments Interface (UPI) payment streams without ground-truth fraud labels.

---

## Project Overview

Digital payment networks process millions of daily transactions, but labeled fraud datasets are rare, heavily skewed, or unavailable due to banking privacy regulations. This project implements an end-to-end **unsupervised anomaly detection solution** that flags suspicious UPI payments by quantifying behavioral deviations from historical user profiles.

### Core Capabilities
- **Reproducible Data Simulation:** Generates 15,000 synthetic transactions across 500 user profiles with custom spend distributions, active hours, favorite merchant categories, and device fingerprints.
- **19 Domain-Engineered Features:** Velocity counts (1h/24h/7d), time deltas, amount Z-scores, rolling standard deviations, Shannon entropy of spending categories, active hour deviations, and geographic distance jumps.
- **Data Leakage Safeguard:** Features are computed strictly using past transactions prior to each event. Injected anomaly indicators are isolated exclusively for post-hoc evaluation.
- **Multi-Detector Suite:** Benchmarks global tabular detectors (**Isolation Forest**), local density detectors (**LOF**), density clustering (**DBSCAN**), and empirical CDF detectors (**PyOD ECOD**).
- **Percentile Thresholding:** Ranks risk levels into `HIGH_RISK` (Top 2%) and `SUSPICIOUS` (Top 5%) without relying on supervised labels.
- **Deterministic Explanation Engine:** Generates human-readable risk signals (e.g., *"Amount spike 8.5x user average | First payment to unseen receiver VPA | Geographic jump 1,200 km"*).
- **Interactive Dashboard:** Built with **Streamlit** and **Plotly** featuring risk KPIs, transaction filter tools, user timeline analysis, model overlap matrices, and 2D PCA cluster maps.

---

## System Architecture & Data Flow

```mermaid
flowchart TD
    A[Data Simulation Layer<br/>src/simulate_data.py] -->|Raw Data CSV| B[Feature Engineering Engine<br/>src/features.py]
    B -->|19 Engineered Features| C[Preprocessing & Robust Scaling<br/>RobustScaler + OneHotEncoder]
    C --> D[Unsupervised Detector Suite<br/>Isolation Forest / LOF / DBSCAN / ECOD]
    D --> E[Scoring & Thresholding Layer<br/>Percentile-based Cutoff]
    E --> F[Explanation Engine<br/>generate_anomaly_reasons]
    F --> G[PCA 2D Spatial Reduction<br/>pca_x, pca_y]
    F --> H[Interactive Dashboard<br/>dashboard/app.py]
```

---

## 📊 Dashboard Visual Analytics & Views

The Streamlit dashboard ([dashboard/app.py](file:///d:/projects/upi/dashboard/app.py)) provides an enterprise risk monitoring interface structured into 5 core visual views:

```text
+---------------------------------------------------------------------------------------------------+
|  UPI RISK SYSTEM                  UPI PAYMENT RISK DASHBOARD                                      |
|  -----------------                --------------------------                                      |
|  [x] Risk Overview                [ TOTAL TXNS ]   [ HIGH RISK ]   [ SUSPICIOUS ]   [ FLAG RATE ] |
|  [ ] Transaction Explorer            15,000            300             450             5.00%      |
|  [ ] User Investigation           +--------------------------------+ +----------------------------+ |
|  [ ] Model Benchmarking           | Anomaly Score Distribution     | | Risk Category Breakdown    | |
|  [ ] PCA 2D Anomaly Map           | [Histogram: Normal vs Outlier] | | [Donut Chart: 95%/3%/2%]   | |
|                                   +--------------------------------+ +----------------------------+ |
+---------------------------------------------------------------------------------------------------+
```

### Dashboard View Breakdown

| View | Key Visuals & Components | Operational Purpose |
|---|---|---|
| **Risk Overview** | Telemetry KPI cards (`Total Txns`, `High Risk`, `Flag Rate`), Ensemble score frequency distribution histogram with percentile cutoff lines, Risk level donut chart, Top risk signal bar chart. | Real-time executive summary of system risk exposure and top triggered fraud signals across all payment channels. |
| **Transaction Explorer** | Multi-select risk filters (`High Risk`, `Suspicious`, `Normal`), category & amount sliders, interactive transaction table with progress-bar anomaly scores, and detailed signal drilldown box. | Enables risk analysts to audit individual suspicious transactions, review Z-score anomalies, and inspect detailed risk reasons. |
| **User Investigation** | Sender VPA selector, historical spend timeline scatter plot (bubble sizes scaled by amount Z-score), active transaction hour histogram. | Allows fraud investigators to analyze a user's chronological payment pattern, velocity bursts, and off-hour transaction anomalies. |
| **Model Benchmarking** | Score correlation heatmap, total outlier flag counts per detector bar chart, Pairwise Jaccard detection overlap matrix. | Evaluates model agreement and compares global isolation (Isolation Forest) vs local density (LOF) vs spatial clustering (DBSCAN). |
| **PCA 2D Anomaly Map** | Interactive 2D Plotly scatter plot (`pca_x` vs `pca_y`) colored by risk category, with hover details per transaction point. | Projects 31 feature dimensions into 2D space to visualize spatial cluster separation of normal points vs outlier points. |

---

## Repository Structure

```text
upi-anomaly-detection/
├── data/
│   ├── raw/                  # Generated raw transaction logs & user profiles
│   └── processed/            # Feature-engineered & anomaly-scored datasets
├── notebooks/
│   ├── 01_data_simulation.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_comparison.ipynb
│   └── 04_evaluation_visualization.ipynb
├── src/
│   ├── simulate_data.py      # Faker & NumPy transaction generator
│   ├── features.py           # Feature engineering engine
│   ├── models.py             # Model training, scoring, thresholding & reasons
│   └── utils.py              # Helpers, geographic distance & data loaders
├── dashboard/
│   └── app.py                # Streamlit risk monitoring web app
├── config.py                 # Hyperparameters, paths & threshold settings
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore
```

---

## Execution Guide

### 1. Environment Setup
Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Execute End-to-End Pipeline
Run the modular pipeline in sequence:

```bash
# Step 1: Generate synthetic transactions (15,000 records across 500 users)
python src/simulate_data.py

# Step 2: Compute velocity, amount Z-score, entropy, and spatial features
python src/features.py

# Step 3: Train unsupervised models, normalize anomaly scores & generate risk reasons
python src/models.py
```

### 3. Launch Interactive Risk Dashboard
Launch the Streamlit application:

```bash
python -m streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser.

---

## Pipeline Summary & Key Findings

- **Total Generated Transactions:** 15,000
- **Unique User Senders:** 500
- **Injected Synthetic Anomalies:** 375 (2.50%)
- **Engineered Features:** 19
- **Model Overlap:** Isolation Forest & LOF show ~34.8% overlap, while Isolation Forest & DBSCAN show ~65.9% overlap on flagged high-risk points.
- **Post-Hoc Sanity Check:** Top 2% percentile high-risk cutoff captured 80 of the most severe multi-vector injected synthetic anomalies.

---

## Limitations

1. **Synthetic Data:** The transaction dataset is synthetically generated using Faker and NumPy distributions for educational/portfolio demonstration.
2. **Unsupervised Context:** Because real ground-truth fraud labels are absent in production, thresholding relies on percentile cutoffs rather than supervised metrics like Precision/Recall.
3. **Synthetic Sanity Check Notice:** Injected anomaly indicators are used **only** for post-hoc validation and are **never** passed to model training features.

---

## Tech Stack

- **Language:** Python 3.13
- **Data Engineering:** Pandas, NumPy, Faker
- **Machine Learning:** scikit-learn (IsolationForest, LOF, DBSCAN), PyOD (ECOD)
- **Visualization:** Plotly, Seaborn, Matplotlib
- **Dashboard:** Streamlit
