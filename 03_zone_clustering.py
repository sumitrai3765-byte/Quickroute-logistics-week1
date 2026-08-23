"""
Phase 4: Predictive Modeling — Delivery-Zone Clustering
----------------------------------------------------------
Groups delivery points into geographically dense micro-zones with K-Means.
Zone assignments feed both route optimization (05) and demand-pattern
analysis for inventory rebalancing.

Input : data/orders_clean.csv
Output: data/orders_with_zones.csv, data/zone_demand.csv
"""

import pandas as pd
from sklearn.cluster import KMeans

N_ZONES = 12  # tune based on hub coverage / vehicle capacity


def assign_zones(orders: pd.DataFrame, n_zones: int = N_ZONES) -> pd.DataFrame:
    coords = orders[["lat", "lon"]].dropna()
    kmeans = KMeans(n_clusters=n_zones, random_state=42, n_init=10)

    orders = orders.copy()
    orders.loc[coords.index, "zone_id"] = kmeans.fit_predict(coords)
    return orders


def zone_demand_profile(orders: pd.DataFrame) -> pd.DataFrame:
    """SKU demand per zone — used to spot which zones/hubs have similar
    demand patterns, informing inventory rebalancing decisions."""
    return orders.groupby(["zone_id", "sku_id"]).size().unstack(fill_value=0)


def main():
    orders = pd.read_csv("data/orders_clean.csv")
    zoned = assign_zones(orders)
    zoned.to_csv("data/orders_with_zones.csv", index=False)

    demand = zone_demand_profile(zoned)
    demand.to_csv("data/zone_demand.csv")

    print(f"Assigned {zoned['zone_id'].nunique()} zones across {len(zoned)} orders.")
    print(zoned.groupby("zone_id").size().sort_values(ascending=False))


if __name__ == "__main__":
    main()
