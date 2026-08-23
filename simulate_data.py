"""
Week 3: Data Simulation
------------------------
Generates a hypothetical QuickRoute Logistics shipment dataset with
realistic structure and correlations, used for the EDA and visualization
task (continuing the Week 1/2 QuickRoute scenario).
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 3000
HUBS = ["Delhi Hub", "Gurugram Hub", "Noida Hub"]
ZONES = [f"Zone {i}" for i in range(1, 13)]
START_DATE = pd.Timestamp("2026-05-01")

# --- Base structural variables ---
hub_id = np.random.choice(HUBS, size=N, p=[0.40, 0.35, 0.25])
zone_id = np.random.choice(ZONES, size=N)
order_date = START_DATE + pd.to_timedelta(np.random.randint(0, 90, size=N), unit="D")
distance_km = np.round(np.random.gamma(shape=3.0, scale=3.2, size=N) + 1, 2)  # right-skewed, realistic
distance_km = np.clip(distance_km, 1, 45)

# Hub load: simulates congestion, higher for Delhi Hub (largest share)
hub_load_map = {"Delhi Hub": 1.25, "Gurugram Hub": 1.05, "Noida Hub": 0.9}
hub_load_factor = np.array([hub_load_map[h] for h in hub_id])

# --- Shipment volume (units per shipment) ---
shipment_volume = np.round(np.random.lognormal(mean=1.6, sigma=0.55, size=N)).astype(int)
shipment_volume = np.clip(shipment_volume, 1, 60)

# --- Delivery time (hours): driven by distance + hub congestion + noise ---
delivery_time_hours = (
    2.0
    + 0.55 * distance_km * hub_load_factor
    + np.random.normal(0, 1.8, size=N)
)
delivery_time_hours = np.clip(delivery_time_hours, 0.5, 48)

# --- Transportation cost (INR): driven by distance + volume + noise ---
transportation_cost = (
    60
    + 18.5 * distance_km
    + 9.0 * shipment_volume
    + np.random.normal(0, 40, size=N)
)
transportation_cost = np.round(np.clip(transportation_cost, 80, None), 2)

# --- SLA promise: 6 hours same-zone, scaled with distance ---
promised_hours = 4 + 0.35 * distance_km
on_time = delivery_time_hours <= promised_hours

df = pd.DataFrame({
    "shipment_id": [f"QR{100000+i}" for i in range(N)],
    "order_date": order_date,
    "hub_id": hub_id,
    "zone_id": zone_id,
    "distance_km": distance_km,
    "shipment_volume": shipment_volume,
    "delivery_time_hours": np.round(delivery_time_hours, 2),
    "promised_hours": np.round(promised_hours, 2),
    "on_time": on_time,
    "transportation_cost": transportation_cost,
})

df = df.sort_values("order_date").reset_index(drop=True)
df.to_csv("quickroute_shipments_simulated.csv", index=False)
print(df.shape)
print(df.head())
print(df.describe(numeric_only=True))
