"""
Phase 4: Predictive Modeling — Delivery-Time Regression
------------------------------------------------------------
Predicts expected delivery time from distance, hub load, and time features,
so at-risk orders can be flagged before dispatch (supports OTDR and
Fulfillment Cycle Time KPIs).

Input : data/orders_with_zones.csv
Output: printed MAE + data/delivery_time_model.pkl
"""

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

FEATURES = ["distance_km", "hub_load", "hour_of_day", "is_weekend"]
TARGET = "delivery_hours"


def train_model(orders: pd.DataFrame):
    X = orders[FEATURES]
    y = orders[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    return model, mae


def flag_at_risk_orders(model, orders: pd.DataFrame, threshold_hrs: float = 24.0) -> pd.DataFrame:
    """Orders where predicted delivery time already exceeds the SLA threshold
    should be re-sequenced or reassigned at dispatch."""
    orders = orders.copy()
    orders["predicted_delivery_hours"] = model.predict(orders[FEATURES])
    return orders[orders["predicted_delivery_hours"] > threshold_hrs]


def main():
    orders = pd.read_csv("data/orders_with_zones.csv")
    model, mae = train_model(orders)
    print(f"Delivery-time model MAE: {mae:.2f} hrs")

    at_risk = flag_at_risk_orders(model, orders)
    print(f"{len(at_risk)} orders flagged as SLA-risk at dispatch.")

    joblib.dump(model, "data/delivery_time_model.pkl")


if __name__ == "__main__":
    main()
