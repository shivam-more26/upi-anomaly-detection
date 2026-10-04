"""
UPI Transaction Risk Monitoring Dashboard (Streamlit).
Enterprise interface for transaction anomaly scoring, risk filtering, user timeline analysis,
model overlap benchmarking, and PCA projection mapping.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from src.utils import load_scored_data

# Page Setup
st.set_page_config(
    page_title="UPI Transaction Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Clean CSS Styling (No Overlaps, Clean White Card Layout)
st.markdown(
    """
    <style>
    .main-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 0.9rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 14px 18px;
        text-align: left;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .metric-title {
        font-size: 0.75rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1E293B;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_data() -> pd.DataFrame:
    """Load scored dataset with caching."""
    return load_scored_data()


try:
    df = get_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.info("Please run `python src/simulate_data.py`, `python src/features.py`, and `python src/models.py` first.")
    st.stop()

# Sidebar Navigation
st.sidebar.title("UPI Risk System")
st.sidebar.caption("Transaction Risk Monitoring & Analytics")

page = st.sidebar.radio(
    "Views",
    [
        "Risk Overview",
        "Transaction Explorer",
        "User Investigation",
        "Model Benchmarking",
        "PCA 2D Anomaly Map",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Threshold Configuration")
high_pct = st.sidebar.slider("High Risk Percentile", 90.0, 99.5, config.HIGH_RISK_PERCENTILE, 0.5)
susp_pct = st.sidebar.slider("Suspicious Percentile", 85.0, 97.0, config.SUSPICIOUS_PERCENTILE, 0.5)

# Dynamic Risk Recalculation based on Sidebar Thresholds
h_thresh = np.percentile(df["anomaly_score"], high_pct)
s_thresh = np.percentile(df["anomaly_score"], susp_pct)


def dynamic_risk(score):
    if score >= h_thresh:
        return "High Risk"
    elif score >= s_thresh:
        return "Suspicious"
    return "Normal"


df["dynamic_risk_level"] = df["anomaly_score"].apply(dynamic_risk)


# Common Plotly Theme Styling Helper
def apply_clean_plotly_theme(fig, height=380):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        font=dict(family="Segoe UI, sans-serif", color="#334155", size=12),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text="",
        ),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#F1F5F9")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#F1F5F9")
    return fig


# ==========================================
# VIEW 1: RISK OVERVIEW
# ==========================================
if page == "Risk Overview":
    st.markdown("<div class='main-header'>UPI Payment Risk Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Unsupervised transaction anomaly scoring and behavioral risk metrics</div>", unsafe_allow_html=True)

    # Top KPI Metrics
    total_txns = len(df)
    high_cnt = (df["dynamic_risk_level"] == "High Risk").sum()
    susp_cnt = (df["dynamic_risk_level"] == "Suspicious").sum()
    flag_rate = ((high_cnt + susp_cnt) / total_txns) * 100
    unique_senders = df["sender_vpa"].nunique()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Total Transactions</div><div class='metric-value'>{total_txns:,}</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>High Risk Flags</div><div class='metric-value' style='color:#DC2626;'>{high_cnt:,}</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Suspicious Flags</div><div class='metric-value' style='color:#D97706;'>{susp_cnt:,}</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Flag Rate</div><div class='metric-value'>{flag_rate:.2f}%</div></div>", unsafe_allow_html=True)
    with col5:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Unique Senders</div><div class='metric-value'>{unique_senders:,}</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns([55, 45])

    with col_left:
        st.markdown("<div class='section-title'>Anomaly Score Distribution</div>", unsafe_allow_html=True)
        fig_score = px.histogram(
            df,
            x="anomaly_score",
            color="dynamic_risk_level",
            nbins=50,
            color_discrete_map={"High Risk": "#DC2626", "Suspicious": "#D97706", "Normal": "#059669"},
            labels={"anomaly_score": "Anomaly Score", "count": "Transaction Count", "dynamic_risk_level": "Risk Level"},
            opacity=0.85,
        )
        fig_score.add_vline(x=h_thresh, line_dash="dash", line_color="#DC2626", annotation_text=f"High Risk ({high_pct}th)")
        fig_score.add_vline(x=s_thresh, line_dash="dot", line_color="#D97706", annotation_text=f"Suspicious ({susp_pct}th)")
        fig_score = apply_clean_plotly_theme(fig_score, height=360)
        st.plotly_chart(fig_score, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-title'>Risk Category Breakdown</div>", unsafe_allow_html=True)
        risk_counts = df["dynamic_risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        fig_pie = px.pie(
            risk_counts,
            values="Count",
            names="Risk Level",
            color="Risk Level",
            color_discrete_map={"High Risk": "#DC2626", "Suspicious": "#D97706", "Normal": "#059669"},
            hole=0.45,
        )
        fig_pie = apply_clean_plotly_theme(fig_pie, height=360)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Top Triggered Risk Signals
    st.markdown("<div class='section-title'>Top Risk Signals in Flagged Transactions</div>", unsafe_allow_html=True)
    high_risk_df = df[df["dynamic_risk_level"] == "High Risk"]
    all_reasons = []
    for reasons in high_risk_df["anomaly_reasons"]:
        for r in reasons.split(" | "):
            if r != "Normal behavioral pattern":
                all_reasons.append(r)

    if all_reasons:
        reason_counts = pd.Series(all_reasons).value_counts().head(7).reset_index()
        reason_counts.columns = ["Risk Signal", "Frequency"]
        fig_bar = px.bar(
            reason_counts,
            x="Frequency",
            y="Risk Signal",
            orientation="h",
            color="Frequency",
            color_continuous_scale="Blues",
        )
        fig_bar = apply_clean_plotly_theme(fig_bar, height=300)
        fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_bar, use_container_width=True)


# ==========================================
# VIEW 2: TRANSACTION EXPLORER
# ==========================================
elif page == "Transaction Explorer":
    st.markdown("<div class='main-header'>Transaction Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Filter and inspect transaction anomaly records</div>", unsafe_allow_html=True)

    # Filters Row
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        selected_risk = st.multiselect("Risk Level", ["High Risk", "Suspicious", "Normal"], default=["High Risk", "Suspicious"])
    with f_col2:
        selected_cat = st.multiselect("Merchant Category", df["merchant_category"].unique(), default=df["merchant_category"].unique())
    with f_col3:
        selected_type = st.multiselect("Transaction Type", df["transaction_type"].unique(), default=df["transaction_type"].unique())
    with f_col4:
        min_amt, max_amt = float(df["amount"].min()), float(df["amount"].max())
        amt_range = st.slider("Amount Range (INR)", min_amt, max_amt, (min_amt, max_amt))

    # Apply Filters
    filtered_df = df[
        (df["dynamic_risk_level"].isin(selected_risk))
        & (df["merchant_category"].isin(selected_cat))
        & (df["transaction_type"].isin(selected_type))
        & (df["amount"] >= amt_range[0])
        & (df["amount"] <= amt_range[1])
    ]

    st.write(f"Displaying **{len(filtered_df):,}** matching transactions of {len(df):,}")

    display_cols = [
        "transaction_id",
        "timestamp",
        "sender_vpa",
        "receiver_vpa",
        "amount",
        "merchant_category",
        "location",
        "dynamic_risk_level",
        "anomaly_score",
        "anomaly_reasons",
    ]

    st.dataframe(
        filtered_df[display_cols].sort_values("anomaly_score", ascending=False),
        column_config={
            "amount": st.column_config.NumberColumn("Amount (INR)", format="INR %.2f"),
            "anomaly_score": st.column_config.ProgressColumn("Anomaly Score", min_value=0.0, max_value=1.0, format="%.3f"),
        },
        use_container_width=True,
        height=420,
    )

    # Transaction Deep-Dive Drilldown
    st.markdown("<div class='section-title'>Transaction Detail Inspection</div>", unsafe_allow_html=True)
    txn_id_list = filtered_df.sort_values("anomaly_score", ascending=False)["transaction_id"].tolist()
    if txn_id_list:
        selected_txn_id = st.selectbox("Select Transaction ID to Inspect", txn_id_list[:100])
        txn_row = df[df["transaction_id"] == selected_txn_id].iloc[0]

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.markdown(f"**Transaction ID:** `{txn_row['transaction_id']}`")
            st.markdown(f"**Sender VPA:** `{txn_row['sender_vpa']}`")
            st.markdown(f"**Receiver VPA:** `{txn_row['receiver_vpa']}`")
            st.markdown(f"**Amount:** INR {txn_row['amount']:,.2f}")
            st.markdown(f"**Timestamp:** {txn_row['timestamp']}")
            st.markdown(f"**Location:** {txn_row['location']}")
        with d_col2:
            st.markdown(f"**Ensemble Anomaly Score:** `{txn_row['anomaly_score']:.4f}`")
            st.markdown(f"**Isolation Forest Score:** `{txn_row['score_isolation_forest']:.4f}`")
            st.markdown(f"**LOF Score:** `{txn_row['score_lof']:.4f}`")
            st.markdown(f"**DBSCAN Score:** `{txn_row['score_dbscan']:.4f}`")
            st.markdown(f"**PyOD ECOD Score:** `{txn_row['score_pyod_ecod']:.4f}`")

        st.info(f"**Triggered Risk Signals:** {txn_row['anomaly_reasons']}")


# ==========================================
# VIEW 3: USER INVESTIGATION
# ==========================================
elif page == "User Investigation":
    st.markdown("<div class='main-header'>User Timeline & Behavioral Profile</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Analyze user transaction history, velocity, and active hour patterns</div>", unsafe_allow_html=True)

    user_list = sorted(df["sender_vpa"].unique())
    selected_user = st.selectbox("Select User VPA", user_list)

    user_df = df[df["sender_vpa"] == selected_user].sort_values("timestamp")

    u_total = len(user_df)
    u_high = (user_df["dynamic_risk_level"] == "High Risk").sum()
    u_avg_amt = user_df["amount"].mean()
    u_max_amt = user_df["amount"].max()

    uc1, uc2, uc3, uc4 = st.columns(4)
    with uc1:
        st.metric("Total Transactions", u_total)
    with uc2:
        st.metric("High Risk Flags", u_high)
    with uc3:
        st.metric("Average Amount", f"INR {u_avg_amt:,.2f}")
    with uc4:
        st.metric("Max Amount", f"INR {u_max_amt:,.2f}")

    st.markdown("---")

    # Time series of amounts and anomaly scores over time (ensuring non-negative marker size for Plotly)
    user_df_copy = user_df.copy()
    user_df_copy["display_marker_size"] = np.clip(user_df_copy["amount_zscore"].clip(lower=0.1) * 3 + 4, 4, 30)

    fig_timeline = px.scatter(
        user_df_copy,
        x="timestamp",
        y="amount",
        color="dynamic_risk_level",
        size="display_marker_size",
        hover_data=["transaction_id", "receiver_vpa", "merchant_category", "amount_zscore", "anomaly_reasons"],
        color_discrete_map={"High Risk": "#DC2626", "Suspicious": "#D97706", "Normal": "#059669"},
        title=f"Transaction History for {selected_user}",
        labels={"timestamp": "Timestamp", "amount": "Amount (INR)"},
    )
    fig_timeline = apply_clean_plotly_theme(fig_timeline, height=380)
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Active Hours Histogram
    fig_hours = px.histogram(
        user_df,
        x="hour_of_day",
        color="dynamic_risk_level",
        nbins=24,
        color_discrete_map={"High Risk": "#DC2626", "Suspicious": "#D97706", "Normal": "#059669"},
        title=f"Transaction Hour Distribution for {selected_user}",
        labels={"hour_of_day": "Hour of Day (0-23)"},
    )
    fig_hours = apply_clean_plotly_theme(fig_hours, height=320)
    st.plotly_chart(fig_hours, use_container_width=True)


# ==========================================
# VIEW 4: MODEL BENCHMARKING
# ==========================================
elif page == "Model Benchmarking":
    st.markdown("<div class='main-header'>Model Benchmarking & Overlap</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Comparative analysis of Isolation Forest, LOF, DBSCAN, and PyOD ECOD</div>", unsafe_allow_html=True)

    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown("<div class='section-title'>Model Score Correlation Matrix</div>", unsafe_allow_html=True)
        score_cols = ["score_isolation_forest", "score_lof", "score_dbscan", "score_pyod_ecod", "anomaly_score"]
        corr_matrix = df[score_cols].corr().round(3)
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(color="Correlation"),
        )
        fig_corr = apply_clean_plotly_theme(fig_corr, height=360)
        st.plotly_chart(fig_corr, use_container_width=True)

    with m_col2:
        st.markdown("<div class='section-title'>Outliers Flagged per Detector</div>", unsafe_allow_html=True)
        pred_counts = {
            "Isolation Forest": df["pred_isolation_forest"].sum(),
            "Local Outlier Factor": df["pred_lof"].sum(),
            "DBSCAN Noise": df["pred_dbscan"].sum(),
            "PyOD ECOD": df["pred_pyod_ecod"].sum(),
        }
        pred_df = pd.DataFrame(list(pred_counts.items()), columns=["Model", "Flagged Count"])
        fig_models = px.bar(
            pred_df,
            x="Model",
            y="Flagged Count",
            color="Model",
        )
        fig_models = apply_clean_plotly_theme(fig_models, height=360)
        st.plotly_chart(fig_models, use_container_width=True)

    st.markdown("<div class='section-title'>Detection Overlap Matrix (Jaccard Similarity %)</div>", unsafe_allow_html=True)
    if_set = set(df[df["pred_isolation_forest"] == 1].index)
    lof_set = set(df[df["pred_lof"] == 1].index)
    dbscan_set = set(df[df["pred_dbscan"] == 1].index)
    pyod_set = set(df[df["pred_pyod_ecod"] == 1].index)

    def jaccard(s1, s2):
        if not s1 or not s2:
            return 0.0
        return len(s1.intersection(s2)) / len(s1.union(s2)) * 100

    j_matrix = pd.DataFrame(
        [
            [100.0, jaccard(if_set, lof_set), jaccard(if_set, dbscan_set), jaccard(if_set, pyod_set)],
            [jaccard(lof_set, if_set), 100.0, jaccard(lof_set, dbscan_set), jaccard(lof_set, pyod_set)],
            [jaccard(dbscan_set, if_set), jaccard(dbscan_set, lof_set), 100.0, jaccard(dbscan_set, pyod_set)],
            [jaccard(pyod_set, if_set), jaccard(pyod_set, lof_set), jaccard(pyod_set, dbscan_set), 100.0],
        ],
        columns=["Isolation Forest", "LOF", "DBSCAN", "PyOD ECOD"],
        index=["Isolation Forest", "LOF", "DBSCAN", "PyOD ECOD"],
    ).round(1)

    st.dataframe(j_matrix, use_container_width=True)


# ==========================================
# VIEW 5: PCA 2D ANOMALY MAP
# ==========================================
elif page == "PCA 2D Anomaly Map":
    st.markdown("<div class='main-header'>PCA 2D Feature Projection</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Principal Component Analysis (PCA) projection of feature space</div>", unsafe_allow_html=True)

    fig_pca = px.scatter(
        df,
        x="pca_x",
        y="pca_y",
        color="dynamic_risk_level",
        size="anomaly_score",
        hover_data=["transaction_id", "sender_vpa", "amount", "merchant_category", "anomaly_reasons"],
        color_discrete_map={"High Risk": "#DC2626", "Suspicious": "#D97706", "Normal": "#059669"},
        labels={"pca_x": "Principal Component 1", "pca_y": "Principal Component 2", "dynamic_risk_level": "Risk Level"},
        opacity=0.75,
    )
    fig_pca = apply_clean_plotly_theme(fig_pca, height=580)
    st.plotly_chart(fig_pca, use_container_width=True)
