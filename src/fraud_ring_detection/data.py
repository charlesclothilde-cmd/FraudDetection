from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path("data/raw")


def _choice(rng, values, size=None, p=None):
    return rng.choice(np.asarray(values), size=size, replace=True, p=p)


def generate_synthetic_data(
    n_users=8000,
    n_transactions=30000,
    n_rings=20,
    random_state=42,
    output_dir=RAW_DIR,
):
    """Generate synthetic fraud-ring data with shared infrastructure."""
    rng = np.random.RandomState(random_state)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    users = pd.DataFrame(
        {
            "user_id": ["u_%05d" % i for i in range(n_users)],
            "age": rng.randint(18, 75, n_users),
            "account_age_days": rng.gamma(shape=2.5, scale=180, size=n_users).astype(int) + 1,
            "home_region": _choice(rng, ["north", "south", "east", "west", "central"], n_users),
            "income_band": _choice(rng, ["low", "medium", "high"], n_users, p=[0.35, 0.5, 0.15]),
        }
    )
    users["ring_id"] = "none"
    users["is_fraud_ring"] = 0

    available = list(users["user_id"])
    ring_members = {}
    for ring_ix in range(n_rings):
        size = int(rng.randint(6, 18))
        selected = list(rng.choice(available, size=size, replace=False))
        available = [u for u in available if u not in selected]
        ring_id = "ring_%02d" % ring_ix
        ring_members[ring_id] = selected
        users.loc[users["user_id"].isin(selected), ["ring_id", "is_fraud_ring"]] = [ring_id, 1]

    normal_devices = ["d_%06d" % i for i in range(n_users + 2000)]
    normal_ips = ["ip_%06d" % i for i in range(n_users + 2500)]
    normal_cards = ["c_%06d" % i for i in range(n_users + 1000)]
    merchants = pd.DataFrame(
        {
            "merchant_id": ["m_%04d" % i for i in range(1200)],
            "merchant_category": _choice(
                rng,
                ["grocery", "fuel", "electronics", "travel", "gaming", "crypto", "marketplace"],
                1200,
                p=[0.22, 0.12, 0.14, 0.12, 0.12, 0.08, 0.20],
            ),
            "merchant_region": _choice(rng, ["north", "south", "east", "west", "central"], 1200),
        }
    )
    merchants["is_mule_merchant"] = 0
    mule_merchants = list(rng.choice(merchants["merchant_id"], size=n_rings * 3, replace=False))
    merchants.loc[merchants["merchant_id"].isin(mule_merchants), "is_mule_merchant"] = 1

    identity = pd.DataFrame(
        {
            "user_id": users["user_id"],
            "primary_device": rng.choice(normal_devices, size=n_users, replace=True),
            "primary_ip": rng.choice(normal_ips, size=n_users, replace=True),
            "primary_card": rng.choice(normal_cards, size=n_users, replace=True),
        }
    )

    ring_identity_rows = []
    for ring_ix, (ring_id, members) in enumerate(ring_members.items()):
        shared_devices = ["rd_%02d_%02d" % (ring_ix, i) for i in range(rng.randint(1, 4))]
        shared_ips = ["rip_%02d_%02d" % (ring_ix, i) for i in range(rng.randint(1, 5))]
        shared_cards = ["rc_%02d_%02d" % (ring_ix, i) for i in range(rng.randint(1, 3))]
        ring_mules = mule_merchants[ring_ix * 3 : ring_ix * 3 + 3]
        for user_id in members:
            identity.loc[identity["user_id"] == user_id, "primary_device"] = rng.choice(shared_devices)
            identity.loc[identity["user_id"] == user_id, "primary_ip"] = rng.choice(shared_ips)
            if rng.rand() < 0.35:
                identity.loc[identity["user_id"] == user_id, "primary_card"] = rng.choice(shared_cards)
            ring_identity_rows.append(
                {
                    "ring_id": ring_id,
                    "user_id": user_id,
                    "ring_devices": ",".join(shared_devices),
                    "ring_ips": ",".join(shared_ips),
                    "ring_cards": ",".join(shared_cards),
                    "ring_merchants": ",".join(ring_mules),
                }
            )

    is_ring_lookup = dict(zip(users["user_id"], users["is_fraud_ring"]))
    ring_id_lookup = dict(zip(users["user_id"], users["ring_id"]))
    device_lookup = dict(zip(identity["user_id"], identity["primary_device"]))
    ip_lookup = dict(zip(identity["user_id"], identity["primary_ip"]))
    card_lookup = dict(zip(identity["user_id"], identity["primary_card"]))
    merchant_ids = merchants["merchant_id"].values
    mule_lookup = dict(zip(merchants["merchant_id"], merchants["is_mule_merchant"]))
    ring_merchant_lookup = {
        row["user_id"]: row["ring_merchants"].split(",") for row in ring_identity_rows
    }

    tx_rows = []
    user_weights = np.ones(n_users)
    user_weights[users["is_fraud_ring"].values == 1] = 4.0
    user_weights = user_weights / user_weights.sum()
    tx_users = rng.choice(users["user_id"].values, size=n_transactions, replace=True, p=user_weights)

    start = pd.Timestamp("2026-01-01")
    for tx_ix, user_id in enumerate(tx_users):
        is_ring = int(is_ring_lookup[user_id])
        ring_id = ring_id_lookup[user_id]
        amount_base = rng.lognormal(mean=3.3, sigma=0.75)
        if is_ring:
            amount = amount_base * rng.uniform(1.4, 4.5)
            is_online = int(rng.rand() < 0.88)
            hour = int(rng.choice(list(range(24)), p=_hour_probs(fraud=True)))
            merchant_id = rng.choice(ring_merchant_lookup[user_id]) if rng.rand() < 0.7 else rng.choice(merchant_ids)
            device_id = device_lookup[user_id] if rng.rand() < 0.82 else rng.choice(normal_devices)
            ip_id = ip_lookup[user_id] if rng.rand() < 0.86 else rng.choice(normal_ips)
            card_id = card_lookup[user_id] if rng.rand() < 0.9 else rng.choice(normal_cards)
        else:
            amount = amount_base * rng.uniform(0.6, 1.8)
            is_online = int(rng.rand() < 0.48)
            hour = int(rng.choice(list(range(24)), p=_hour_probs(fraud=False)))
            merchant_id = rng.choice(merchant_ids)
            device_id = device_lookup[user_id] if rng.rand() < 0.94 else rng.choice(normal_devices)
            ip_id = ip_lookup[user_id] if rng.rand() < 0.88 else rng.choice(normal_ips)
            card_id = card_lookup[user_id] if rng.rand() < 0.96 else rng.choice(normal_cards)

        mule_merchant = int(mule_lookup[merchant_id])
        fraud_probability = 0.006 + 0.52 * is_ring + 0.18 * mule_merchant
        fraud_probability += 0.08 * int(hour <= 5) + 0.04 * int(amount > 180)
        is_fraud = int(rng.rand() < min(fraud_probability, 0.95))
        ts = start + pd.Timedelta(days=int(rng.randint(0, 120)), hours=hour, minutes=int(rng.randint(0, 60)))

        tx_rows.append(
            {
                "transaction_id": "tx_%07d" % tx_ix,
                "user_id": user_id,
                "timestamp": ts,
                "amount": round(float(amount), 2),
                "merchant_id": merchant_id,
                "device_id": device_id,
                "ip_id": ip_id,
                "card_id": card_id,
                "is_online": is_online,
                "is_fraud": is_fraud,
                "ring_id": ring_id,
            }
        )

    transactions = pd.DataFrame(tx_rows)
    ring_infrastructure = pd.DataFrame(ring_identity_rows)

    users.to_csv(output_dir / "users.csv", index=False)
    identity.to_csv(output_dir / "user_identity.csv", index=False)
    merchants.to_csv(output_dir / "merchants.csv", index=False)
    transactions.to_csv(output_dir / "transactions.csv", index=False)
    ring_infrastructure.to_csv(output_dir / "ring_infrastructure.csv", index=False)

    print("Generated %d users and %d transactions in %s" % (len(users), len(transactions), output_dir))


def _hour_probs(fraud=False):
    probs = np.ones(24, dtype=float)
    if fraud:
        probs[:6] = 4.0
        probs[20:] = 2.2
    else:
        probs[:6] = 0.25
        probs[9:20] = 1.8
    return probs / probs.sum()
