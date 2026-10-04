# Beginner Glossary: UPI Anomaly Detection Terminology

A quick-reference dictionary for key terms used in this project.

---

### 1. Payment & Banking Domain Terms
- **UPI (Unified Payments Interface):** Instant real-time payment system developed by NPCI (National Payments Corporation of India) for peer-to-peer and person-to-merchant transactions.
- **VPA (Virtual Payment Address):** Unique identifier for a bank account (e.g. `user@upi`, `shivam@okaxis`).
- **P2P (Person-to-Person):** Direct money transfer between two individual users.
- **P2M (Person-to-Merchant):** Payment made by a user to a business or merchant.
- **Merchant Category:** Classification of business type (e.g., `grocery`, `fuel`, `dining`, `e_commerce`).

---

### 2. Machine Learning & Anomaly Detection Terms
- **Anomaly / Outlier:** A data point that deviates significantly from the rest of the dataset.
- **Unsupervised Learning:** Machine learning technique that discovers patterns in data without pre-existing labels.
- **Isolation Forest:** An algorithm that isolates anomalies by randomly partitioning feature space using decision trees.
- **Local Outlier Factor (LOF):** An algorithm that measures the local density deviation of a data point relative to its $k$-nearest neighbors.
- **DBSCAN:** Density-based clustering algorithm that groups dense points and identifies sparse noise points (`-1`).
- **PyOD ECOD:** Empirical Cumulative Distribution Function detector that scores tail probabilities across feature distributions.
- **PCA (Principal Component Analysis):** Dimensionality reduction method that projects multi-dimensional features into 2 principal axes while maximizing variance.

---

### 3. Feature Engineering & Mathematics Terms
- **Feature:** An individual measurable property or characteristic of a phenomenon being observed.
- **Velocity Feature:** A metric measuring frequency or rate of transactions over time (e.g., transactions in last 1 hour).
- **Z-Score:** Standard score representing how many standard deviations a value lies above or below the historical mean ($Z = (X - \mu) / \sigma$).
- **Shannon Entropy:** Mathematical measure of uncertainty or diversity in categorical spending patterns.
- **Haversine Distance:** Formula calculating the shortest distance over the Earth's surface between two geographic coordinates.
- **Data Leakage:** Inadvertent use of future or target data during feature engineering or model training.
- **RobustScaler:** Feature scaling method using Median and Interquartile Range (IQR) to normalize features without being distorted by outliers.
