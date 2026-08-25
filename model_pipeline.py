"""
Week 4: Predictive Modeling and Optimization
-----------------------------------------------
Problem: forecast delivery_time_hours for a QuickRoute shipment at the
point of dispatch, using shipment-level features available before delivery.

This reuses the Week 3 simulated dataset (quickroute_shipments_simulated.csv).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load + feature engineering
# ---------------------------------------------------------------
df = pd.read_csv("quickroute_shipments_simulated.csv", parse_dates=["order_date"])
df["day_of_week"] = df["order_date"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Target: delivery_time_hours
# Features available BEFORE delivery (no leakage from transportation_cost,
# which is itself a downstream/parallel outcome of distance, not a predictor
# available at dispatch time in a real system)
FEATURES_NUM = ["distance_km", "shipment_volume", "is_weekend"]
FEATURES_CAT = ["hub_id"]
TARGET = "delivery_time_hours"

X = df[FEATURES_NUM + FEATURES_CAT]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), FEATURES_CAT),
], remainder="passthrough")

# ---------------------------------------------------------------
# 2. Candidate models
# ---------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=RANDOM_STATE),
}

results = []
kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    cv_rmse = -cross_val_score(pipe, X_train, y_train, cv=kfold,
                                scoring="neg_root_mean_squared_error")

    results.append({
        "model": name,
        "test_rmse": round(rmse, 3),
        "test_mae": round(mae, 3),
        "test_r2": round(r2, 3),
        "cv_rmse_mean": round(cv_rmse.mean(), 3),
        "cv_rmse_std": round(cv_rmse.std(), 3),
    })

results_df = pd.DataFrame(results).sort_values("test_rmse")
print(results_df.to_string(index=False))
results_df.to_csv("model_comparison_results.csv", index=False)

# ---------------------------------------------------------------
# 3. Hyperparameter tuning on the best-performing model family (Random Forest)
# ---------------------------------------------------------------
rf_pipe = Pipeline([
    ("prep", preprocess),
    ("model", RandomForestRegressor(random_state=RANDOM_STATE)),
])

param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [4, 6, 8, None],
    "model__min_samples_leaf": [1, 5, 10],
}

grid = GridSearchCV(
    rf_pipe, param_grid, cv=kfold,
    scoring="neg_root_mean_squared_error", n_jobs=-1
)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
best_preds = best_model.predict(X_test)

tuned_rmse = np.sqrt(mean_squared_error(y_test, best_preds))
tuned_mae = mean_absolute_error(y_test, best_preds)
tuned_r2 = r2_score(y_test, best_preds)

print("\nBest params:", grid.best_params_)
print(f"Tuned RF -> RMSE: {tuned_rmse:.3f} | MAE: {tuned_mae:.3f} | R2: {tuned_r2:.3f}")

with open("best_model_summary.txt", "w") as f:
    f.write(f"Best params: {grid.best_params_}\n")
    f.write(f"Tuned RF Test RMSE: {tuned_rmse:.3f}\n")
    f.write(f"Tuned RF Test MAE: {tuned_mae:.3f}\n")
    f.write(f"Tuned RF Test R2: {tuned_r2:.3f}\n")

# ---------------------------------------------------------------
# 4. Feature importance (from tuned Random Forest)
# ---------------------------------------------------------------
ohe_cols = list(best_model.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(FEATURES_CAT))
all_feature_names = ohe_cols + FEATURES_NUM
importances = best_model.named_steps["model"].feature_importances_

fi_df = pd.DataFrame({"feature": all_feature_names, "importance": importances}) \
    .sort_values("importance", ascending=False)
fi_df.to_csv("feature_importance.csv", index=False)
print("\nFeature importances:\n", fi_df.to_string(index=False))

# Save predictions for plotting
pred_df = pd.DataFrame({"actual": y_test.values, "predicted": best_preds})
pred_df.to_csv("test_predictions.csv", index=False)

# Also save the Linear Regression predictions for a baseline comparison plot
lr_pipe = Pipeline([("prep", preprocess), ("model", LinearRegression())])
lr_pipe.fit(X_train, y_train)
lr_preds = lr_pipe.predict(X_test)
pd.DataFrame({"actual": y_test.values, "predicted": lr_preds}).to_csv("lr_test_predictions.csv", index=False)
