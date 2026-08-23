"""
Phase 3: Exploratory Analysis — KPI Baseline
----------------------------------------------
Computes the five headline KPIs and a per-hub breakdown so underperforming
hubs/zones can be identified before modeling.

Input : data/orders_clean.csv
Output: printed summary + data/kpi_by_hub.csv
"""

import pandas as pd


def compute_headline_kpis(orders: pd.DataFrame) -> dict:
    return {
        "on_time_delivery_rate_pct": round(orders["on_time"].mean() * 100, 1),
        "avg_fulfillment_cycle_time_hrs": round(orders["delivery_hours"].mean(), 2),
        "orders_analyzed": len(orders),
    }


def kpi_by_hub(orders: pd.DataFrame) -> pd.DataFrame:
    summary = orders.groupby("hub_id").agg(
        otdr_pct=("on_time", lambda s: round(s.mean() * 100, 1)),
        avg_cycle_hours=("delivery_hours", "mean"),
        order_count=("order_id", "count"),
    ).sort_values("otdr_pct")
    return summary


def main():
    orders = pd.read_csv("data/orders_clean.csv", parse_dates=["order_ts", "delivered_ts", "promised_ts"])

    headline = compute_headline_kpis(orders)
    print("Headline KPIs")
    print("-------------")
    for k, v in headline.items():
        print(f"{k}: {v}")

    by_hub = kpi_by_hub(orders)
    print("\nOTDR by hub (lowest first — investigate these):")
    print(by_hub)

    by_hub.to_csv("data/kpi_by_hub.csv")


if __name__ == "__main__":
    main()
