# Synthetic Fraud Ring Detection With Graph ML

This project simulates a payments ecosystem with hidden fraud rings, then compares a tabular fraud model with a graph-enhanced model that uses shared devices, IPs, cards, and merchants.

## What It Builds

- Synthetic users, cards, devices, IPs, merchants, and transactions.
- Injected fraud rings that share infrastructure and route payments through mule merchants.
- User-level graph features such as shared-device degree, shared-IP degree, connected component size, and risky merchant overlap.
- A baseline tabular model versus a graph-enhanced model.
- A Streamlit dashboard for risk scores and suspicious ring inspection.

## Quick Start

```bash
python3 generate_data.py
python3 train_model.py
python3 -m streamlit run app.py
```

The dashboard will read generated artifacts from `data/processed/`, `models/`, and `reports/`.

For a larger Colab-style run:

```bash
python3 generate_data.py --users 20000 --transactions 150000 --rings 50
python3 train_model.py
```

## Project Structure

```text
.
├── app.py
├── generate_data.py
├── train_model.py
├── src/fraud_ring_detection/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   └── visualization.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
└── reports/
```

## Modelling Approach

The target is user-level fraud-ring membership. The baseline model uses only transaction aggregates, such as transaction count, average amount, online share, night activity, and merchant diversity. The graph model adds relationship features that expose shared infrastructure:

- maximum users sharing any device used by the user
- maximum users sharing any IP used by the user
- maximum users sharing any card used by the user
- connected component size across shared identifiers
- high-risk merchant concentration
- graph risk score from shared identifiers and merchant overlap

This mirrors a common fraud analytics pattern: individual behavior can look plausible, but the relationship network reveals coordinated abuse.

## Outputs

After training:

- `data/processed/user_features.csv`
- `data/processed/user_scores.csv`
- `models/fraud_ring_model.joblib`
- `reports/metrics.csv`
- `reports/top_suspicious_users.csv`
- `reports/suspicious_ring.svg`

## Next Upgrades

- Add Node2Vec embeddings once graph packages are available.
- Convert the graph to PyTorch Geometric for GraphSAGE node classification.
- Add temporal graph features, such as shared infrastructure within rolling windows.
- Add cost-sensitive threshold tuning for analyst review capacity.
