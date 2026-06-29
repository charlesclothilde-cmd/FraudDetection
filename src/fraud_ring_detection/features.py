from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}
        self.size = {item: 1 for item in items}

    def find(self, item):
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def build_user_features(raw_dir=RAW_DIR, output_dir=PROCESSED_DIR):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    users = pd.read_csv(raw_dir / "users.csv")
    transactions = pd.read_csv(raw_dir / "transactions.csv", parse_dates=["timestamp"])
    merchants = pd.read_csv(raw_dir / "merchants.csv")

    transactions["hour"] = transactions["timestamp"].dt.hour
    transactions["is_night"] = transactions["hour"].between(0, 5).astype(int)
    transactions["amount_log"] = np.log1p(transactions["amount"])

    merchant_meta = merchants[["merchant_id", "merchant_category", "is_mule_merchant"]]
    tx = transactions.merge(merchant_meta, on="merchant_id", how="left")

    tabular = tx.groupby("user_id").agg(
        tx_count=("transaction_id", "count"),
        amount_mean=("amount", "mean"),
        amount_std=("amount", "std"),
        amount_max=("amount", "max"),
        amount_log_mean=("amount_log", "mean"),
        online_rate=("is_online", "mean"),
        night_rate=("is_night", "mean"),
        merchant_count=("merchant_id", "nunique"),
        category_count=("merchant_category", "nunique"),
    )
    tabular["amount_std"] = tabular["amount_std"].fillna(0.0)
    tabular["tx_per_merchant"] = tabular["tx_count"] / tabular["merchant_count"].clip(lower=1)

    user_ids = users["user_id"].tolist()
    graph = _graph_features(tx, user_ids)
    embeddings = build_node2vec_embeddings(tx, user_ids)
    user_cols = ["user_id", "ring_id", "is_fraud_ring"]
    if "ring_type" in users.columns:
        user_cols.insert(2, "ring_type")
    features = users[user_cols].merge(tabular, on="user_id", how="left")
    features = features.merge(graph, on="user_id", how="left").fillna(0)
    features = features.merge(embeddings, on="user_id", how="left").fillna(0)

    features.to_csv(output_dir / "user_features.csv", index=False)
    tx.to_csv(output_dir / "transactions_enriched.csv", index=False)
    return features


def build_node2vec_embeddings(
    tx,
    user_ids,
    dimensions=16,
    walk_length=12,
    walks_per_node=3,
    window_size=4,
    p=1.0,
    q=0.5,
    random_state=42,
):
    """Create Node2Vec-style user embeddings from shared identifiers.

    The implementation avoids optional compiled dependencies so the project
    remains easy to run in Colab: it generates biased random walks, builds a
    skip-gram co-occurrence matrix, and compresses it with SVD.
    """
    rng = np.random.RandomState(random_state)
    user_ids = list(user_ids)
    index = {user_id: ix for ix, user_id in enumerate(user_ids)}
    adjacency = _build_user_user_graph(tx, user_ids)
    walks = _node2vec_walks(adjacency, user_ids, walk_length, walks_per_node, p, q, rng)
    cooccurrence = _walk_cooccurrence(walks, index, window_size)

    n_users = len(user_ids)
    max_components = max(1, min(dimensions, n_users - 1, cooccurrence.shape[1] - 1))
    if max_components < 1 or cooccurrence.sum() == 0:
        values = np.zeros((n_users, dimensions))
    else:
        svd = TruncatedSVD(n_components=max_components, random_state=random_state)
        values = svd.fit_transform(cooccurrence)
        values = normalize(values)
        if max_components < dimensions:
            values = np.pad(values, ((0, 0), (0, dimensions - max_components)))

    cols = ["n2v_%02d" % ix for ix in range(dimensions)]
    return pd.DataFrame(values, columns=cols).assign(user_id=user_ids)[["user_id"] + cols]


def _build_user_user_graph(tx, user_ids):
    adjacency = {user_id: defaultdict(float) for user_id in user_ids}
    relation_weights = {
        "device_id": 3.0,
        "ip_id": 2.0,
        "card_id": 3.0,
        "merchant_id": 0.6,
    }
    max_group_sizes = {
        "device_id": 80,
        "ip_id": 120,
        "card_id": 60,
        "merchant_id": 35,
    }

    for col, weight in relation_weights.items():
        pairs = tx[["user_id", col]].drop_duplicates()
        for _, group in pairs.groupby(col):
            users = group["user_id"].tolist()
            if len(users) < 2 or len(users) > max_group_sizes[col]:
                continue
            edge_weight = weight / np.log1p(len(users))
            for i, left in enumerate(users):
                if left not in adjacency:
                    continue
                for right in users[i + 1 :]:
                    if right not in adjacency:
                        continue
                    adjacency[left][right] += edge_weight
                    adjacency[right][left] += edge_weight

    return {user: dict(neighbors) for user, neighbors in adjacency.items()}


def _node2vec_walks(adjacency, user_ids, walk_length, walks_per_node, p, q, rng):
    walks = []
    for _ in range(walks_per_node):
        shuffled = list(user_ids)
        rng.shuffle(shuffled)
        for start in shuffled:
            walks.append(_node2vec_walk(adjacency, start, walk_length, p, q, rng))
    return walks


def _node2vec_walk(adjacency, start, walk_length, p, q, rng):
    walk = [start]
    while len(walk) < walk_length:
        current = walk[-1]
        neighbors = list(adjacency.get(current, {}))
        if not neighbors:
            break
        weights = np.array([adjacency[current][neighbor] for neighbor in neighbors], dtype=float)
        if len(walk) > 1:
            previous = walk[-2]
            adjusted = []
            previous_neighbors = adjacency.get(previous, {})
            for neighbor, weight in zip(neighbors, weights):
                if neighbor == previous:
                    adjusted.append(weight / p)
                elif neighbor in previous_neighbors:
                    adjusted.append(weight)
                else:
                    adjusted.append(weight / q)
            weights = np.array(adjusted, dtype=float)
        probabilities = weights / weights.sum()
        walk.append(rng.choice(neighbors, p=probabilities))
    return walk


def _walk_cooccurrence(walks, index, window_size):
    rows = []
    cols = []
    data = []
    for walk in walks:
        positions = [index[node] for node in walk if node in index]
        for pos, center in enumerate(positions):
            left = max(0, pos - window_size)
            right = min(len(positions), pos + window_size + 1)
            for ctx_pos in range(left, right):
                if ctx_pos == pos:
                    continue
                context = positions[ctx_pos]
                distance = abs(ctx_pos - pos)
                rows.append(center)
                cols.append(context)
                data.append(1.0 / distance)
    return sparse.coo_matrix((data, (rows, cols)), shape=(len(index), len(index)), dtype=np.float32).tocsr()


def _graph_features(tx, user_ids):
    identifier_cols = ["device_id", "ip_id", "card_id"]
    frames = []
    for col in identifier_cols:
        degree = tx.groupby(col)["user_id"].nunique().rename("%s_user_degree" % col)
        user_degree = tx[["user_id", col]].drop_duplicates().merge(degree, on=col, how="left")
        agg = user_degree.groupby("user_id").agg(
            **{
                "%s_degree_max" % col: ("%s_user_degree" % col, "max"),
                "%s_degree_mean" % col: ("%s_user_degree" % col, "mean"),
            }
        )
        frames.append(agg)

    merchant_degree = tx.groupby("merchant_id")["user_id"].nunique().rename("merchant_user_degree")
    user_merchant_degree = tx[["user_id", "merchant_id"]].drop_duplicates().merge(
        merchant_degree, on="merchant_id", how="left"
    )
    merchant_agg = user_merchant_degree.groupby("user_id").agg(
        merchant_degree_max=("merchant_user_degree", "max"),
        merchant_degree_mean=("merchant_user_degree", "mean"),
    )
    frames.append(merchant_agg)

    mule_agg = tx.groupby("user_id").agg(
        mule_merchant_rate=("is_mule_merchant", "mean"),
        mule_merchant_count=("is_mule_merchant", "sum"),
    )
    frames.append(mule_agg)

    component = _shared_identifier_components(tx, user_ids, identifier_cols)
    frames.append(component.set_index("user_id"))

    graph = pd.concat(frames, axis=1).reset_index().rename(columns={"index": "user_id"}).fillna(0)
    graph["shared_device_ip_pressure"] = (
        graph["device_id_degree_max"] + graph["ip_id_degree_max"] + graph["card_id_degree_max"]
    )
    graph["graph_risk_score"] = (
        0.35 * np.log1p(graph["component_size"])
        + 0.25 * np.log1p(graph["shared_device_ip_pressure"])
        + 0.20 * graph["mule_merchant_rate"]
        + 0.20 * np.log1p(graph["merchant_degree_max"])
    )
    return graph


def _shared_identifier_components(tx, user_ids, identifier_cols):
    uf = UnionFind(user_ids)
    for col in identifier_cols:
        pairs = tx[["user_id", col]].drop_duplicates()
        for _, group in pairs.groupby(col):
            users = group["user_id"].tolist()
            if len(users) > 1:
                first = users[0]
                for user in users[1:]:
                    uf.union(first, user)

    roots = {user: uf.find(user) for user in user_ids}
    sizes = pd.Series(list(roots.values())).value_counts().to_dict()
    return pd.DataFrame(
        {
            "user_id": user_ids,
            "component_id": [roots[user] for user in user_ids],
            "component_size": [sizes[roots[user]] for user in user_ids],
        }
    )
