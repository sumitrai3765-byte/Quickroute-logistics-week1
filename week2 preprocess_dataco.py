"""
Week 2: Data Collection, Cleaning, and Preprocessing
------------------------------------------------------
Simulates collection from the DataCo Smart Supply Chain dataset (Kaggle) and
runs it through a cleaning/preprocessing pipeline for QuickRoute Logistics.

Input : data/DataCoSupplyChainDataset.csv  (download from Kaggle, not committed)
Output: data/dataco_clean.csv
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_raw(path: str = "data/DataCoSupplyChainDataset.csv") -> pd.DataFrame:
    # Source file uses Latin-1 encoding, not UTF-8
    return pd.read_csv(path, encoding="latin-1")


def drop_near_empty_columns(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    empty_share = df.isna().mean()
    cols_to_drop = empty_share[empty_share > threshold].index
    return df.drop(columns=cols_to_drop)


def impute_zipcode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Customer Zipcode"] = df.groupby("Customer City")["Customer Zipcode"].transform(
        lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else s)
    )
    df["zipcode_was_missing"] = df["Customer Zipcode"].isna()
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["Order Id", "Product Card Id"])
    print(f"Removed {before - len(df)} duplicate order-item rows")
    return df


def cap_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return series.clip(lower=lower, upper=upper)


def standardize_text_and_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Customer City"] = df["Customer City"].str.strip().str.title()
    df["order date (DateOrders)"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
    df["shipping date (DateOrders)"] = pd.to_datetime(df["shipping date (DateOrders)"], errors="coerce")
    return df


def scale_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scale_cols = ["days_for_shipping_capped", "order_qty_capped", "Order Item Product Price"]
    scaler = MinMaxScaler()
    df[[c + "_scaled" for c in scale_cols]] = scaler.fit_transform(df[scale_cols])
    return df


def run_pipeline(path: str = "data/DataCoSupplyChainDataset.csv") -> pd.DataFrame:
    df = load_raw(path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    df = drop_near_empty_columns(df)
    df = impute_zipcode(df)
    df = drop_duplicates(df)

    df["days_for_shipping_capped"] = cap_outliers_iqr(df["Days for shipping (real)"])
    df["order_qty_capped"] = cap_outliers_iqr(df["Order Item Quantity"])

    df = standardize_text_and_dates(df)
    df = scale_numeric_features(df)

    print(f"Final cleaned shape: {df.shape}")
    return df


if __name__ == "__main__":
    cleaned = run_pipeline()
    cleaned.to_csv("data/dataco_clean.csv", index=False)
