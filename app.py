from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Fraud Ring Detection", layout="wide")

PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports")


@st.cache_data
def load_data():
    scores = pd.read_csv(PROCESSED_DIR / "user_scores.csv")
    features = pd.read_csv(PROCESSED_DIR / "user_features.csv")
    metrics = pd.read_csv(REPORT_DIR / "metrics.csv")
    top = pd.read_csv(REPORT_DIR / "top_suspicious_users.csv")
    return scores, features, metrics, top


st.title("Fraud Ring Detection")

if not (PROCESSED_DIR / "user_scores.csv").exists():
    st.warning("Run `python3 generate_data.py` and `python3 train_model.py` first.")
    st.stop()

scores, features, metrics, top = load_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Users", f"{len(scores):,}")
col2.metric("Fraud-ring users", f"{int(scores['is_fraud_ring'].sum()):,}")
col3.metric("Top 5% precision", f"{metrics.iloc[0]['precision_at_top_5pct']:.2f}")
col4.metric("Top 5% recall", f"{metrics.iloc[0]['recall_at_top_5pct']:.2f}")

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Model comparison")
    st.dataframe(metrics, use_container_width=True)

with right:
    st.subheader("Risk score distribution")
    chart_data = scores[["risk_score", "is_fraud_ring"]].copy()
    chart_data["bucket"] = pd.cut(chart_data["risk_score"], bins=20).astype(str)
    bucketed = chart_data.groupby(["bucket", "is_fraud_ring"]).size().reset_index(name="users")
    pivot = bucketed.pivot(index="bucket", columns="is_fraud_ring", values="users").fillna(0)
    pivot.columns = ["Normal", "Fraud ring"] if len(pivot.columns) == 2 else pivot.columns
    st.bar_chart(pivot)

st.subheader("Highest-risk users")
threshold = st.slider("Minimum risk score", 0.0, 1.0, 0.5, 0.01)
filtered = scores[scores["risk_score"] >= threshold].head(200)
st.dataframe(filtered, use_container_width=True)

st.subheader("Suspicious shared-infrastructure component")
svg_path = REPORT_DIR / "suspicious_ring.svg"
if svg_path.exists():
    st.image(str(svg_path), use_column_width=True)

selected_user = st.selectbox("Inspect user", scores["user_id"].head(200).tolist())
row = features[features["user_id"] == selected_user].iloc[0]
detail_cols = [
    "ring_id",
    "is_fraud_ring",
    "component_size",
    "device_id_degree_max",
    "ip_id_degree_max",
    "card_id_degree_max",
    "merchant_degree_max",
    "mule_merchant_rate",
    "graph_risk_score",
]
st.dataframe(row[detail_cols].to_frame("value"), use_container_width=True)
