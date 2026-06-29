import argparse

from src.fraud_ring_detection.data import generate_synthetic_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic fraud-ring data.")
    parser.add_argument("--users", type=int, default=8000)
    parser.add_argument("--transactions", type=int, default=30000)
    parser.add_argument("--rings", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_synthetic_data(
        n_users=args.users,
        n_transactions=args.transactions,
        n_rings=args.rings,
        random_state=args.seed,
    )
