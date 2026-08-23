"""
Phase 4: Predictive Modeling — Stockout Risk Classification
------------------------------------------------------------
Flags SKU-hub combinations at high risk of stockout in the next 7 days,
enabling proactive stock transfers between hubs (supports Inventory Turnover
and Stockout Rate KPIs).

Input : data/sku_hub_daily.csv  (see data/README.md for schema)
Output: printed high-risk SKU-hub pairs + data/stockout_risk.csv
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

FEATURES = ["avg_daily_sales", "current_stock", "lead_time_days", "sales_volatility"]
TARGET = "stockout_next_7d"
RISK_THRESHOLD = 0.7


def train_classifier(sku_hub: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        sku_hub[FEATURES], sku_hub[TARGET], test_size=0.2, random_state=42, stratify=sku_hub[TARGET]
    )
    clf = LogisticRegression(class_weight="balanced")
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    return clf, accuracy


def score_current_risk(clf, sku_hub_current: pd.DataFrame) -> pd.DataFrame:
    sku_hub_current = sku_hub_current.copy()
    sku_hub_current["stockout_risk"] = clf.predict_proba(sku_hub_current[FEATURES])[:, 1]
    return sku_hub_current


def main():
    sku_hub = pd.read_csv("data/sku_hub_daily.csv")

    clf, accuracy = train_classifier(sku_hub)
    print(f"Stockout classifier holdout accuracy: {accuracy:.2%}")

    scored = score_current_risk(clf, sku_hub)
    high_risk = scored[scored["stockout_risk"] > RISK_THRESHOLD].sort_values(
        "stockout_risk", ascending=False
    )

    print(f"{len(high_risk)} SKU-hub pairs above the {RISK_THRESHOLD:.0%} risk threshold:")
    print(high_risk[["sku_id", "hub_id", "stockout_risk"]].head(20))

    scored.to_csv("data/stockout_risk.csv", index=False)


if __name__ == "__main__":
    main()
