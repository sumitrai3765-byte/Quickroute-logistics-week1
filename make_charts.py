import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid", font_scale=1.0)
ACCENT = "#C0511A"
PALETTE = ["#C0511A", "#2E5F6E", "#8C9A6B"]

# --- Chart 1: Model comparison (test RMSE) ---
results = pd.read_csv("model_comparison_results.csv")
fig, ax = plt.subplots(figsize=(7, 4.3))
bars = ax.bar(results["model"], results["test_rmse"], color=PALETTE)
for b, v in zip(bars, results["test_rmse"]):
    ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
ax.set_title("Model Comparison: Test RMSE (Delivery Time, hrs)", fontsize=13, weight="bold")
ax.set_ylabel("RMSE (hours)")
ax.set_ylim(0, max(results["test_rmse"]) * 1.25)
fig.tight_layout()
fig.savefig("charts/01_model_comparison_rmse.png", dpi=160)
plt.close(fig)

# --- Chart 2: Actual vs Predicted (tuned Random Forest) ---
pred = pd.read_csv("test_predictions.csv")
fig, ax = plt.subplots(figsize=(6.2, 6))
ax.scatter(pred["actual"], pred["predicted"], alpha=0.3, s=16, color=ACCENT)
lims = [min(pred["actual"].min(), pred["predicted"].min()),
        max(pred["actual"].max(), pred["predicted"].max())]
ax.plot(lims, lims, color="#1F2937", linestyle="--", linewidth=1.5, label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_title("Actual vs. Predicted Delivery Time\n(Tuned Random Forest)", fontsize=13, weight="bold")
ax.set_xlabel("Actual Delivery Time (hrs)")
ax.set_ylabel("Predicted Delivery Time (hrs)")
ax.legend()
fig.tight_layout()
fig.savefig("charts/02_actual_vs_predicted.png", dpi=160)
plt.close(fig)

# --- Chart 3: Feature importance ---
fi = pd.read_csv("feature_importance.csv")
fig, ax = plt.subplots(figsize=(7.5, 4.3))
sns.barplot(data=fi, x="importance", y="feature", color=ACCENT, ax=ax)
ax.set_title("Feature Importance (Tuned Random Forest)", fontsize=13, weight="bold")
ax.set_xlabel("Importance")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig("charts/03_feature_importance.png", dpi=160)
plt.close(fig)

# --- Chart 4: Residual plot ---
pred["residual"] = pred["actual"] - pred["predicted"]
fig, ax = plt.subplots(figsize=(7.5, 4.3))
ax.scatter(pred["predicted"], pred["residual"], alpha=0.3, s=16, color=ACCENT)
ax.axhline(0, color="#1F2937", linestyle="--", linewidth=1.5)
ax.set_title("Residual Plot (Tuned Random Forest)", fontsize=13, weight="bold")
ax.set_xlabel("Predicted Delivery Time (hrs)")
ax.set_ylabel("Residual (Actual − Predicted)")
fig.tight_layout()
fig.savefig("charts/04_residual_plot.png", dpi=160)
plt.close(fig)

print("All Week 4 charts saved.")
