from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Fraud Investigation Console", layout="wide")

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
REPORT_DIR = Path("reports")


@st.cache_data
def load_data():
    scores = pd.read_csv(PROCESSED_DIR / "user_scores.csv")
    features = pd.read_csv(PROCESSED_DIR / "user_features.csv")
    metrics = pd.read_csv(REPORT_DIR / "metrics.csv")
    transactions = pd.read_csv(PROCESSED_DIR / "transactions_enriched.csv", parse_dates=["timestamp"])
    users = pd.read_csv(RAW_DIR / "users.csv")
    user_view = scores.merge(
        features.drop(
            columns=[
                col
                for col in [
                    "ring_id",
                    "ring_type",
                    "is_fraud_ring",
                    "component_id",
                    "component_size",
                    "shared_device_count",
                    "shared_ip_count",
                    "mule_merchant_exposure",
                    "top_linked_users",
                    "top_linked_entities",
                ]
                if col in features.columns
            ]
        ),
        on="user_id",
        how="left",
    )
    user_view = user_view.merge(
        users[["user_id", "age", "account_age_days", "home_region", "income_band"]],
        on="user_id",
        how="left",
    )
    return scores, features, metrics, transactions, users, user_view


def format_metric(value, precision=2):
    if pd.isna(value):
        return "-"
    return f"{value:.{precision}f}"


def explain_user(row):
    reasons = []
    risk_score = float(row.get("risk_score", 0))
    if risk_score >= 0.8:
        reasons.append("Very high model score")
    elif risk_score >= 0.5:
        reasons.append("Elevated model score")

    component_size = int(row.get("component_size", 0))
    if component_size >= 25:
        reasons.append(f"Large shared-identifier component ({component_size:,} users)")
    elif component_size >= 5:
        reasons.append(f"Multi-user shared-identifier component ({component_size:,} users)")

    shared_devices = int(row.get("shared_device_count", 0))
    shared_ips = int(row.get("shared_ip_count", 0))
    if shared_devices:
        reasons.append(f"{shared_devices} shared device links")
    if shared_ips:
        reasons.append(f"{shared_ips} shared IP links")

    if int(row.get("card_id_degree_max", 0)) > 1:
        reasons.append(f"Card reused by up to {int(row['card_id_degree_max'])} users")
    if int(row.get("mule_merchant_exposure", 0)):
        reasons.append(f"{int(row['mule_merchant_exposure'])} mule merchant exposures")
    if float(row.get("night_rate", 0)) >= 0.35:
        reasons.append(f"Night activity rate {row['night_rate']:.0%}")
    if int(row.get("merchant_degree_max", 0)) >= 25:
        reasons.append(f"Merchant overlap up to {int(row['merchant_degree_max'])} users")
    if row.get("ring_id", "none") != "none":
        reasons.append(f"Known synthetic ring label: {row.get('ring_id')}")

    return reasons or ["No single dominant reason; review linked entities and transactions."]


def component_summary(users_df):
    if users_df.empty:
        return pd.DataFrame()
    return (
        users_df.groupby("component_id")
        .agg(
            matched_users=("user_id", "nunique"),
            max_score=("risk_score", "max"),
            avg_score=("risk_score", "mean"),
            ring_users=("is_fraud_ring", "sum"),
            shared_devices=("shared_device_count", "sum"),
            shared_ips=("shared_ip_count", "sum"),
            mule_exposures=("mule_merchant_exposure", "sum"),
            ring_ids=("ring_id", lambda values: ", ".join(sorted(v for v in values.unique() if v != "none")) or "none"),
        )
        .reset_index()
        .sort_values(["max_score", "ring_users", "matched_users"], ascending=False)
    )


def entity_table(component_tx, col, label, extra_cols=None):
    extra_cols = extra_cols or {}
    grouped = component_tx.groupby(col).agg(
        users=("user_id", "nunique"),
        transactions=("transaction_id", "count"),
        amount=("amount", "sum"),
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
    )
    for output_col, spec in extra_cols.items():
        grouped[output_col] = component_tx.groupby(col).agg(**{output_col: spec})[output_col]

    table = grouped.reset_index().rename(columns={col: label})
    table = table[table["users"] > 1].copy()
    table["amount"] = table["amount"].round(2)
    return table.sort_values(["users", "transactions", "amount"], ascending=False)


def linked_users_for_entity(component_tx, entity_col, entity_id):
    users = component_tx.loc[component_tx[entity_col] == entity_id, "user_id"].drop_duplicates()
    return ", ".join(users.head(12).tolist()) + (" ..." if len(users) > 12 else "")


if not (PROCESSED_DIR / "user_scores.csv").exists():
    st.warning("Run `python3 generate_data.py` and `python3 train_model.py` first.")
    st.stop()

scores, features, metrics, transactions, users, user_view = load_data()

st.title("Fraud Investigation Console")

with st.sidebar:
    st.header("Filters")
    min_score = st.slider("Minimum risk score", 0.0, 1.0, 0.2, 0.01)
    size_min, size_max = int(scores["component_size"].min()), int(scores["component_size"].max())
    component_size = st.slider("Component size", size_min, size_max, (size_min, size_max))
    ring_options = ["All", "Fraud rings only", "No known ring"] + sorted(
        ring for ring in scores["ring_id"].dropna().unique().tolist() if ring != "none"
    )
    ring_filter = st.selectbox("Ring", ring_options)
    show_count = st.slider("Review queue size", 25, 500, 150, 25)

filtered_users = user_view[
    (user_view["risk_score"] >= min_score)
    & (user_view["component_size"].between(component_size[0], component_size[1]))
].copy()
if ring_filter == "Fraud rings only":
    filtered_users = filtered_users[filtered_users["is_fraud_ring"] == 1]
elif ring_filter == "No known ring":
    filtered_users = filtered_users[filtered_users["ring_id"] == "none"]
elif ring_filter != "All":
    filtered_users = filtered_users[filtered_users["ring_id"] == ring_filter]

summary = component_summary(filtered_users)

metric_cols = st.columns(5)
metric_cols[0].metric("Users", f"{len(scores):,}")
metric_cols[1].metric("Filtered queue", f"{len(filtered_users):,}")
metric_cols[2].metric("Components", f"{summary['component_id'].nunique() if not summary.empty else 0:,}")
metric_cols[3].metric("Fraud-ring users", f"{int(filtered_users['is_fraud_ring'].sum()):,}")
metric_cols[4].metric("Best AP", format_metric(metrics["average_precision"].max(), 3))

model_tab, queue_tab, component_tab = st.tabs(["Model comparison", "Review queue", "Component investigation"])

with model_tab:
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Model leaderboard")
        metric_view = metrics.copy()
        for col in ["roc_auc", "average_precision", "precision_at_top_5pct", "recall_at_top_5pct"]:
            metric_view[col] = metric_view[col].round(3)
        st.dataframe(metric_view, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Ranking quality")
        chart = metrics.set_index("model")[["average_precision", "roc_auc"]]
        st.bar_chart(chart)

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.subheader("Top 5% queue performance")
        st.bar_chart(metrics.set_index("model")[["precision_at_top_5pct", "recall_at_top_5pct"]])
    with bottom_right:
        st.subheader("Risk score distribution")
        chart_data = scores[["risk_score", "is_fraud_ring"]].copy()
        chart_data["bucket"] = pd.cut(chart_data["risk_score"], bins=20).astype(str)
        bucketed = chart_data.groupby(["bucket", "is_fraud_ring"]).size().reset_index(name="users")
        pivot = bucketed.pivot(index="bucket", columns="is_fraud_ring", values="users").fillna(0)
        pivot = pivot.rename(columns={0: "Normal", 1: "Fraud ring"})
        st.bar_chart(pivot)

with queue_tab:
    st.subheader("Prioritized users")
    queue_cols = [
        "user_id",
        "risk_score",
        "ring_id",
        "ring_type",
        "component_id",
        "component_size",
        "shared_device_count",
        "shared_ip_count",
        "mule_merchant_exposure",
        "top_linked_users",
        "top_linked_entities",
    ]
    queue_cols = [col for col in queue_cols if col in filtered_users.columns]
    review_queue = filtered_users.sort_values("risk_score", ascending=False).head(show_count)
    st.dataframe(review_queue[queue_cols], use_container_width=True, hide_index=True)

    st.subheader("Suspicious components")
    st.dataframe(summary.head(100), use_container_width=True, hide_index=True)

with component_tab:
    if summary.empty:
        st.info("No components match the current filters.")
        st.stop()

    labels = [
        f"{row.component_id} | matched users {row.matched_users:,} | max score {row.max_score:.3f} | rings {row.ring_ids}"
        for row in summary.itertuples()
    ]
    selected_label = st.selectbox("Select suspicious component", labels, key="component-selector")
    selected_component = selected_label.split(" | ", 1)[0]

    component_users = user_view[user_view["component_id"] == selected_component].copy()
    component_users = component_users.sort_values("risk_score", ascending=False)
    component_tx = transactions[transactions["user_id"].isin(component_users["user_id"])]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Component users", f"{component_users['user_id'].nunique():,}")
    c2.metric("Max risk", format_metric(component_users["risk_score"].max(), 3))
    c3.metric("Ring users", f"{int(component_users['is_fraud_ring'].sum()):,}")
    c4.metric("Transactions", f"{len(component_tx):,}")

    component_left, component_right = st.columns([1.25, 1])
    with component_left:
        st.subheader("Users in component")
        component_cols = [
            "user_id",
            "risk_score",
            "ring_id",
            "ring_type",
            "shared_device_count",
            "shared_ip_count",
            "mule_merchant_exposure",
            "device_id_degree_max",
            "ip_id_degree_max",
            "card_id_degree_max",
            "merchant_degree_max",
            "top_linked_users",
        ]
        component_cols = [col for col in component_cols if col in component_users.columns]
        st.dataframe(component_users[component_cols].head(show_count), use_container_width=True, hide_index=True)

    with component_right:
        selected_user = st.selectbox("Inspect user", component_users["user_id"].tolist())
        selected_row = component_users[component_users["user_id"] == selected_user].iloc[0]
        st.subheader("Risk reasons")
        for reason in explain_user(selected_row):
            st.write(f"- {reason}")

        detail_cols = [
            "risk_score",
            "ring_id",
            "ring_type",
            "age",
            "account_age_days",
            "home_region",
            "income_band",
            "tx_count",
            "amount_mean",
            "online_rate",
            "night_rate",
            "graph_risk_score",
        ]
        detail_cols = [col for col in detail_cols if col in selected_row.index]
        st.dataframe(selected_row[detail_cols].astype(str).to_frame("value"), use_container_width=True)

    st.subheader("Linked entities")
    device_tab, ip_tab, card_tab, merchant_tab = st.tabs(["Devices", "IPs", "Cards", "Merchants"])
    entity_specs = [
        (device_tab, "device_id", "device_id", {}),
        (ip_tab, "ip_id", "ip_id", {}),
        (card_tab, "card_id", "card_id", {}),
        (
            merchant_tab,
            "merchant_id",
            "merchant_id",
            {
                "mule_merchant": ("is_mule_merchant", "max"),
                "category": ("merchant_category", "first"),
            },
        ),
    ]
    for tab, col, label, extra in entity_specs:
        with tab:
            table = entity_table(component_tx, col, label, extra)
            if table.empty:
                st.info(f"No shared {label.replace('_', ' ')} found in this component.")
                continue
            selected_entity = st.selectbox(
                f"Inspect {label.replace('_', ' ')}",
                table[label].head(200).tolist(),
                key=f"entity-{col}",
            )
            table["linked_users"] = table[label].apply(lambda entity_id: linked_users_for_entity(component_tx, col, entity_id))
            st.dataframe(table.head(200), use_container_width=True, hide_index=True)

            entity_tx = component_tx[component_tx[col] == selected_entity].sort_values("timestamp", ascending=False)
            tx_cols = [
                "timestamp",
                "transaction_id",
                "user_id",
                "amount",
                "merchant_id",
                "merchant_category",
                "device_id",
                "ip_id",
                "card_id",
                "is_mule_merchant",
                "is_fraud",
            ]
            st.dataframe(entity_tx[tx_cols].head(50), use_container_width=True, hide_index=True)

    st.subheader("Static network snapshot")
    svg_path = REPORT_DIR / "suspicious_ring.svg"
    if svg_path.exists():
        components.html(svg_path.read_text(), height=620, scrolling=False)
