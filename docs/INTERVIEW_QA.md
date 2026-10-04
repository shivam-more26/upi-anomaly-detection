# Interview & Viva Q&A Guide: UPI Anomaly Detection

This guide contains interview and viva questions, conceptual drilldowns, and expert responses for presenting this project.

---

### Q1: What is the main objective of this project?
> **Answer:** To build an unsupervised machine learning pipeline that flags suspicious or anomalous UPI digital payment transactions by scoring behavioral deviations from historical user profiles without relying on ground-truth fraud labels.

---

### Q2: Why is Unsupervised Anomaly Detection necessary for digital payments?
> **Answer:** In real-world fintech environments, ground-truth fraud labels are rarely available in real time. Fraud reports arrive weeks later after customer dispute filings. Furthermore, fraud patterns evolve constantly. Unsupervised anomaly detection identifies novel, unknown fraud vectors based purely on statistical and behavioral deviation.

---

### Q3: How did you handle Data Leakage in time-window feature calculations?
> **Answer:** All velocity counts (`txns_last_1h`), rolling standard deviations, and Z-scores are computed strictly using historical timestamps prior to the current transaction. We use `np.searchsorted` to perform binary searches on past timestamps only. Additionally, injected synthetic anomaly indicators are excluded from preprocessing and model training.

---

### Q4: Explain the 19 behavioral features engineered in your pipeline.
> **Answer:** 
> 1. **Velocity Features:** `txns_last_1h`, `txns_last_24h`, `txns_last_7d`, `time_since_prev_txn_sec`, `velocity_spike_ratio`.
> 2. **Amount Features:** `user_mean_amount`, `user_std_amount`, `amount_zscore`, `amount_to_user_avg_ratio`, `rolling_amount_std_5`.
> 3. **Behavioral Features:** `is_new_merchant_category`, `is_new_receiver`, `merchant_category_entropy`.
> 4. **Temporal Features:** `hour_of_day`, `day_of_week`, `is_weekend`, `active_hour_deviation`.
> 5. **Spatial Features:** `is_new_device`, `is_new_location`, `distance_from_home_km`, `unique_devices_last_24h`.

---

### Q5: Why choose `RobustScaler` over `StandardScaler`?
> **Answer:** `StandardScaler` uses mean and standard deviation, which are strongly distorted by extreme outliers in anomaly datasets. `RobustScaler` uses Median and Interquartile Range (IQR), preserving outlier magnitude while keeping normal distributions scaled properly without collapsing variance.

---

### Q6: How does Isolation Forest work intuitively?
> **Answer:** Isolation Forest isolates points by randomly partitioning feature space with decision trees. Because anomalous points sit far from normal dense clusters, they require significantly fewer splits (shorter path length) to isolate.

---

### Q7: How did you convert DBSCAN output into continuous anomaly scores?
> **Answer:** DBSCAN assigns discrete cluster IDs (`-1` for noise). To produce a continuous score, we compute the mean $k$-nearest neighbor distance across features and multiply distance by 2.5 for noise points (`-1`).

---

### Q8: How are threshold cutoffs established without fraud labels?
> **Answer:** We compute a weighted ensemble anomaly score ($0.40 \cdot \text{IF} + 0.25 \cdot \text{LOF} + 0.20 \cdot \text{ECOD} + 0.15 \cdot \text{DBSCAN}$) and rank transactions using percentile cutoffs: Top 2% (`HIGH_RISK`), Top 5% (`SUSPICIOUS`), and remaining 95% (`NORMAL`).

---

### Q9: How does the Deterministic Explanation Engine work?
> **Answer:** For every flagged transaction, a rule engine evaluates feature thresholds (e.g., $Z \ge 3.0$, velocity spike ratio $\ge 4.0$, new device flag $= 1$, active hour deviation $\ge 2.0$) and generates human-readable risk text explaining *why* the transaction was flagged.

---

### Q10: How do you visualize 31 feature dimensions in Streamlit?
> **Answer:** We apply Principal Component Analysis (PCA) to compress 31 features into 2 Principal Components (`pca_x`, `pca_y`), plotting them on a 2D Plotly scatter plot where normal points cluster near the origin and high-risk points project outward into extreme spatial vectors.
