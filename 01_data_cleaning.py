"""
Phase 2: Data Cleaning
-----------------------
Loads raw order-level data and produces an analysis-ready DataFrame.

Input : data/orders.csv
Output: data/orders_clean.csv
"""

import pandas as pd

TIMESTAMP_COLS = ["order_ts", "dispatch_ts", "delivered_ts", "promised_ts"]


def load_raw_orders(path: str = "data/orders.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=TIMESTAMP_COLS)


def clean_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate, drop incomplete rows, and derive core fields."""
    orders = orders.drop_duplicates(subset="order_id").copy()
    orders = orders.dropna(subset=["order_id", "order_ts", "delivered_ts", "promised_ts"])

    # Core derived fields used across every downstream step
    orders["delivery_hours"] = (
        orders["delivered_ts"] - orders["order_ts"]
    ).dt.total_seconds() / 3600
    orders["on_time"] = orders["delivered_ts"] <= orders["promised_ts"]

    # Drop implausible delivery times (negative or > 3 days = likely bad data)
    orders = orders[(orders["delivery_hours"] > 0) & (orders["delivery_hours"] < 72)]

    return orders.reset_index(drop=True)


def main():
    raw = load_raw_orders()
    clean = clean_orders(raw)
    clean.to_csv("data/orders_clean.csv", index=False)
    print(f"Cleaned {len(raw)} raw rows -> {len(clean)} usable orders "
          f"({len(raw) - len(clean)} dropped).")


if __name__ == "__main__":
    main()
