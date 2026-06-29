from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_user_features
from .visualization import write_suspicious_ring_svg


PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")


TABULAR_FEATURES = [
    "tx_count",
    "amount_mean",
    "amount_std",
    "amount_max",
    "amount_log_mean",
    "online_rate",
    "night_rate",
    "merchant_count",
    "category_count",
    "tx_per_merchant",
]

GRAPH_FEATURES = [
    "device_id_degree_max",
    "device_id_degree_mean",
    "ip_id_degree_max",
    "ip_id_degree_mean",
    "card_id_degree_max",
    "card_id_degree_mean",
    "merchant_degree_max",
    "merchant_degree_mean",
    "mule_merchant_rate",
    "mule_merchant_count",
    "component_size",
    "shared_device_ip_pressure",
    "graph_risk_score",
]


def train_and_evaluate():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    features = build_user_features()
    train_df, test_df = train_test_split(
        features,
        test_size=0.3,
        random_state=42,
        stratify=features["is_fraud_ring"],
    )

    models = {
        "tabular_logistic": _logistic_pipeline(),
        "graph_logistic": _logistic_pipeline(),
        "graph_random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }
    feature_sets = {
        "tabular_logistic": TABULAR_FEATURES,
        "graph_logistic": TABULAR_FEATURES + GRAPH_FEATURES,
        "graph_random_forest": TABULAR_FEATURES + GRAPH_FEATURES,
    }

    rows = []
    fitted = {}
    for name, model in models.items():
        cols = feature_sets[name]
        model.fit(train_df[cols], train_df["is_fraud_ring"])
        scores = model.predict_proba(test_df[cols])[:, 1]
        rows.append(_metrics(name, test_df["is_fraud_ring"], scores))
        fitted[name] = (model, cols)

    metrics = pd.DataFrame(rows).sort_values("average_precision", ascending=False)
    metrics.to_csv(REPORT_DIR / "metrics.csv", index=False)
    best_name = metrics.iloc[0]["model"]
    best_model, best_cols = fitted[best_name]

    all_scores = best_model.predict_proba(features[best_cols])[:, 1]
    score_cols = ["user_id", "ring_id", "is_fraud_ring", "component_id", "component_size"]
    if "ring_type" in features.columns:
        score_cols.insert(2, "ring_type")
    scored = features[score_cols].copy()
    scored["risk_score"] = all_scores
    scored = scored.sort_values("risk_score", ascending=False)
    scored.to_csv(PROCESSED_DIR / "user_scores.csv", index=False)
    scored.head(100).to_csv(REPORT_DIR / "top_suspicious_users.csv", index=False)

    joblib.dump({"model": best_model, "features": best_cols, "model_name": best_name}, MODEL_DIR / "fraud_ring_model.joblib")
    write_suspicious_ring_svg(scored, features, REPORT_DIR / "suspicious_ring.svg")

    print(metrics.to_string(index=False))
    print("Best model: %s" % best_name)
    print("Wrote scores, model, and suspicious ring report.")


def _logistic_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")),
        ]
    )


def _metrics(name, y_true, scores):
    threshold = pd.Series(scores).quantile(0.95)
    pred = (scores >= threshold).astype(int)
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_true, scores),
        "average_precision": average_precision_score(y_true, scores),
        "precision_at_top_5pct": precision_score(y_true, pred, zero_division=0),
        "recall_at_top_5pct": recall_score(y_true, pred, zero_division=0),
    }
