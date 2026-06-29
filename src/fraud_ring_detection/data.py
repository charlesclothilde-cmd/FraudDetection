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
    """Generate synthetic fraud-ring data with varied, noisy shared behavior."""
    rng = np.random.RandomState(random_state)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp("2026-01-01")
    ring_type_names = ["account_farm", "mule_merchants", "shared_devices", "promo_abuse"]
    ring_profiles = {
        "account_farm": {
            "device_share": 0.72,
            "ip_share": 0.88,
            "card_share": 0.18,
            "mule_rate": 0.35,
            "online_rate": 0.84,
            "amount_low": 0.55,
            "amount_high": 1.30,
            "fraud_lift": 0.30,
            "window_days": (3, 8),
            "merchant_count": 1,
        },
        "mule_merchants": {
            "device_share": 0.28,
            "ip_share": 0.34,
            "card_share": 0.22,
            "mule_rate": 0.76,
            "online_rate": 0.55,
            "amount_low": 0.90,
            "amount_high": 2.35,
            "fraud_lift": 0.34,
            "window_days": (5, 15),
            "merchant_count": 4,
        },
        "shared_devices": {
            "device_share": 0.86,
            "ip_share": 0.45,
            "card_share": 0.10,
            "mule_rate": 0.30,
            "online_rate": 0.66,
            "amount_low": 0.70,
            "amount_high": 1.75,
            "fraud_lift": 0.26,
            "window_days": (2, 6),
            "merchant_count": 2,
        },
        "promo_abuse": {
            "device_share": 0.38,
            "ip_share": 0.65,
            "card_share": 0.08,
            "mule_rate": 0.42,
            "online_rate": 0.93,
            "amount_low": 0.35,
            "amount_high": 0.95,
            "fraud_lift": 0.22,
            "window_days": (1, 5),
            "merchant_count": 2,
        },
    }

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
    users["ring_type"] = "none"
    users["is_fraud_ring"] = 0

    available = list(users["user_id"])
    ring_members = {}
    ring_metadata = {}
    for ring_ix in range(n_rings):
        ring_type = ring_type_names[ring_ix % len(ring_type_names)]
        if ring_type == "account_farm":
            size = int(rng.randint(14, 34))
        elif ring_type == "mule_merchants":
            size = int(rng.randint(5, 14))
        elif ring_type == "shared_devices":
            size = int(rng.randint(6, 18))
        else:
            size = int(rng.randint(10, 28))
        selected = list(rng.choice(available, size=size, replace=False))
        available = [u for u in available if u not in selected]
        ring_id = "ring_%02d" % ring_ix
        ring_members[ring_id] = selected
        active_start = start + pd.Timedelta(days=int(rng.randint(0, 105)))
        window_min, window_max = ring_profiles[ring_type]["window_days"]
        active_end = active_start + pd.Timedelta(days=int(rng.randint(window_min, window_max + 1)))
        ring_metadata[ring_id] = {
            "ring_type": ring_type,
            "active_start": active_start,
            "active_end": active_end,
        }
        users.loc[users["user_id"].isin(selected), ["ring_id", "ring_type", "is_fraud_ring"]] = [
            ring_id,
            ring_type,
            1,
        ]

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
            "benign_share_group": "none",
            "benign_share_type": "none",
        }
    )

    normal_available = users.loc[users["is_fraud_ring"] == 0, "user_id"].tolist()
    benign_groups = []
    for group_ix in range(max(8, n_users // 260)):
        share_type = rng.choice(["family", "office", "coworking", "student_house"], p=[0.34, 0.34, 0.18, 0.14])
        if share_type == "family":
            group_size = int(rng.randint(2, 6))
            share_device_probability = 0.42
        elif share_type == "office":
            group_size = int(rng.randint(8, 35))
            share_device_probability = 0.05
        elif share_type == "coworking":
            group_size = int(rng.randint(12, 50))
            share_device_probability = 0.02
        else:
            group_size = int(rng.randint(3, 9))
            share_device_probability = 0.18
        if len(normal_available) < group_size:
            break
        members = list(rng.choice(normal_available, size=group_size, replace=False))
        normal_available = [u for u in normal_available if u not in members]
        group_id = "benign_%03d" % group_ix
        shared_ip = "bip_%03d" % group_ix
        shared_device = "bd_%03d" % group_ix
        mask = identity["user_id"].isin(members)
        identity.loc[mask, ["benign_share_group", "benign_share_type", "primary_ip"]] = [
            group_id,
            share_type,
            shared_ip,
        ]
        device_mask = mask & (rng.rand(len(identity)) < share_device_probability)
        identity.loc[device_mask, "primary_device"] = shared_device
        benign_groups.append(
            {
                "benign_share_group": group_id,
                "benign_share_type": share_type,
                "shared_ip": shared_ip,
                "shared_device": shared_device,
                "member_count": len(members),
            }
        )

    ring_identity_rows = []
    for ring_ix, (ring_id, members) in enumerate(ring_members.items()):
        ring_type = ring_metadata[ring_id]["ring_type"]
        profile = ring_profiles[ring_type]
        shared_devices = ["rd_%02d_%02d" % (ring_ix, i) for i in range(rng.randint(1, 3))]
        shared_ips = ["rip_%02d_%02d" % (ring_ix, i) for i in range(rng.randint(1, 4))]
        shared_cards = ["rc_%02d_%02d" % (ring_ix, i) for i in range(rng.randint(1, 3))]
        merchant_count = int(profile["merchant_count"])
        ring_mules = mule_merchants[ring_ix * 3 : ring_ix * 3 + max(3, merchant_count)]
        if len(ring_mules) < merchant_count:
            ring_mules = list(rng.choice(mule_merchants, size=merchant_count, replace=True))
        else:
            ring_mules = ring_mules[:merchant_count]
        for user_id in members:
            if rng.rand() < profile["device_share"]:
                identity.loc[identity["user_id"] == user_id, "primary_device"] = rng.choice(shared_devices)
            if rng.rand() < profile["ip_share"]:
                identity.loc[identity["user_id"] == user_id, "primary_ip"] = rng.choice(shared_ips)
            if rng.rand() < profile["card_share"]:
                identity.loc[identity["user_id"] == user_id, "primary_card"] = rng.choice(shared_cards)
            ring_identity_rows.append(
                {
                    "ring_id": ring_id,
                    "ring_type": ring_type,
                    "user_id": user_id,
                    "ring_devices": ",".join(shared_devices),
                    "ring_ips": ",".join(shared_ips),
                    "ring_cards": ",".join(shared_cards),
                    "ring_merchants": ",".join(ring_mules),
                    "active_start": ring_metadata[ring_id]["active_start"].date(),
                    "active_end": ring_metadata[ring_id]["active_end"].date(),
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
    ring_type_lookup = dict(zip(users["user_id"], users["ring_type"]))

    noisy_user_count = max(20, int(0.012 * n_users))
    noisy_users = set(rng.choice(users.loc[users["is_fraud_ring"] == 0, "user_id"], size=noisy_user_count, replace=False))

    tx_rows = []
    user_weights = np.ones(n_users)
    user_weights[users["is_fraud_ring"].values == 1] = 1.18
    user_weights[users["user_id"].isin(noisy_users).values] = 1.26
    user_weights = user_weights / user_weights.sum()
    tx_users = rng.choice(users["user_id"].values, size=n_transactions, replace=True, p=user_weights)

    for tx_ix, user_id in enumerate(tx_users):
        is_ring = int(is_ring_lookup[user_id])
        ring_id = ring_id_lookup[user_id]
        ring_type = ring_type_lookup[user_id]
        amount_base = rng.lognormal(mean=3.3, sigma=0.75)
        if is_ring:
            profile = ring_profiles[ring_type]
            amount = amount_base * rng.uniform(profile["amount_low"], profile["amount_high"])
            is_online = int(rng.rand() < profile["online_rate"])
            hour = int(rng.choice(list(range(24)), p=_hour_probs(fraud=True)))
            merchant_id = rng.choice(ring_merchant_lookup[user_id]) if rng.rand() < profile["mule_rate"] else rng.choice(merchant_ids)
            device_id = device_lookup[user_id] if rng.rand() < (0.56 + 0.30 * profile["device_share"]) else rng.choice(normal_devices)
            ip_id = ip_lookup[user_id] if rng.rand() < (0.54 + 0.32 * profile["ip_share"]) else rng.choice(normal_ips)
            card_id = card_lookup[user_id] if rng.rand() < (0.82 + 0.10 * profile["card_share"]) else rng.choice(normal_cards)
            ts = _ring_timestamp(rng, start, ring_metadata[ring_id], hour)
        else:
            amount = amount_base * rng.uniform(0.6, 1.8)
            is_online = int(rng.rand() < 0.48)
            hour = int(rng.choice(list(range(24)), p=_hour_probs(fraud=False)))
            if user_id in noisy_users and rng.rand() < 0.20:
                merchant_id = rng.choice(mule_merchants)
            else:
                merchant_id = rng.choice(merchant_ids)
            device_id = device_lookup[user_id] if rng.rand() < 0.94 else rng.choice(normal_devices)
            ip_id = ip_lookup[user_id] if rng.rand() < 0.88 else rng.choice(normal_ips)
            card_id = card_lookup[user_id] if rng.rand() < 0.96 else rng.choice(normal_cards)
            ts = start + pd.Timedelta(days=int(rng.randint(0, 120)), hours=hour, minutes=int(rng.randint(0, 60)))

        mule_merchant = int(mule_lookup[merchant_id])
        fraud_probability = 0.006 + 0.18 * mule_merchant
        if is_ring:
            fraud_probability += ring_profiles[ring_type]["fraud_lift"]
            if not _within_ring_window(ts, ring_metadata[ring_id]):
                fraud_probability *= 0.45
        if user_id in noisy_users:
            fraud_probability += 0.055
        fraud_probability += 0.055 * int(hour <= 5) + 0.035 * int(amount > 180)
        is_fraud = int(rng.rand() < min(fraud_probability, 0.88))

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
                "ring_type": ring_type,
            }
        )

    transactions = pd.DataFrame(tx_rows)
    ring_infrastructure = pd.DataFrame(ring_identity_rows)
    benign_infrastructure = pd.DataFrame(benign_groups)

    users.to_csv(output_dir / "users.csv", index=False)
    identity.to_csv(output_dir / "user_identity.csv", index=False)
    merchants.to_csv(output_dir / "merchants.csv", index=False)
    transactions.to_csv(output_dir / "transactions.csv", index=False)
    ring_infrastructure.to_csv(output_dir / "ring_infrastructure.csv", index=False)
    benign_infrastructure.to_csv(output_dir / "benign_shared_infrastructure.csv", index=False)

    print("Generated %d users and %d transactions in %s" % (len(users), len(transactions), output_dir))


def _hour_probs(fraud=False):
    probs = np.ones(24, dtype=float)
    if fraud:
        probs[:6] = 1.8
        probs[20:] = 1.5
    else:
        probs[:6] = 0.25
        probs[9:20] = 1.8
    return probs / probs.sum()


def _ring_timestamp(rng, start, metadata, hour):
    if rng.rand() < 0.74:
        active_start = metadata["active_start"]
        active_end = metadata["active_end"]
        window_days = max(1, (active_end - active_start).days + 1)
        day_offset = int(rng.randint(0, window_days))
        return active_start + pd.Timedelta(days=day_offset, hours=hour, minutes=int(rng.randint(0, 60)))
    return start + pd.Timedelta(days=int(rng.randint(0, 120)), hours=hour, minutes=int(rng.randint(0, 60)))


def _within_ring_window(timestamp, metadata):
    return metadata["active_start"] <= timestamp <= metadata["active_end"] + pd.Timedelta(days=1)
