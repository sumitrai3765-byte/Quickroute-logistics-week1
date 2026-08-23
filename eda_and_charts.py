"""
Week 3: Exploratory Data Analysis & Visualization
---------------------------------------------------
Loads the simulated QuickRoute shipment dataset, computes EDA summaries,
and produces the visualization set for the Week 3 report.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.0)
ACCENT = "#C0511A"
PALETTE = ["#C0511A", "#2E5F6E", "#8C9A6B", "#D9A441", "#6B4E71"]

df = pd.read_csv("quickroute_shipments_simulated.csv", parse_dates=["order_date"])

# ---------------------------------------------------------------
# EDA: central tendency, dispersion, correlation
# ---------------------------------------------------------------
numeric_cols = ["distance_km", "shipment_volume", "delivery_time_hours",
                 "transportation_cost"]

summary = df[numeric_cols].agg(["mean", "median", "std", "min", "max"]).T
summary["skew"] = df[numeric_cols].skew()
summary.to_csv("eda_summary_stats.csv")
print("Summary stats:\n", summary, "\n")

corr = df[numeric_cols].corr()
corr.to_csv("eda_correlation_matrix.csv")
print("Correlation matrix:\n", corr, "\n")

otdr_overall = df["on_time"].mean() * 100
otdr_by_hub = df.groupby("hub_id")["on_time"].mean().sort_values() * 100
print(f"Overall OTDR: {otdr_overall:.1f}%")
print("OTDR by hub:\n", otdr_by_hub, "\n")

# ---------------------------------------------------------------
# Chart 1: Distribution of delivery times (histogram + KDE)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
sns.histplot(df["delivery_time_hours"], bins=35, kde=True, color=ACCENT, ax=ax)
ax.axvline(df["delivery_time_hours"].mean(), color="#2E5F6E", linestyle="--",
           label=f"Mean = {df['delivery_time_hours'].mean():.1f} hrs")
ax.axvline(df["delivery_time_hours"].median(), color="#1F2937", linestyle=":",
           label=f"Median = {df['delivery_time_hours'].median():.1f} hrs")
ax.set_title("Distribution of Delivery Times", fontsize=13, weight="bold")
ax.set_xlabel("Delivery Time (hours)")
ax.set_ylabel("Number of Shipments")
ax.legend()
fig.tight_layout()
fig.savefig("charts/01_delivery_time_distribution.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------
# Chart 2: Delivery time by hub (boxplot)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
order = df.groupby("hub_id")["delivery_time_hours"].median().sort_values().index
sns.boxplot(data=df, x="hub_id", y="delivery_time_hours", order=order,
            palette=PALETTE, ax=ax)
ax.set_title("Delivery Time Spread by Hub", fontsize=13, weight="bold")
ax.set_xlabel("Hub")
ax.set_ylabel("Delivery Time (hours)")
fig.tight_layout()
fig.savefig("charts/02_delivery_time_by_hub.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------
# Chart 3: Distance vs. Transportation Cost (scatter + trend)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
sns.regplot(data=df, x="distance_km", y="transportation_cost",
            scatter_kws={"alpha": 0.25, "s": 14, "color": ACCENT},
            line_kws={"color": "#1F2937"}, ax=ax)
ax.set_title("Transportation Cost vs. Delivery Distance", fontsize=13, weight="bold")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Transportation Cost (₹)")
fig.tight_layout()
fig.savefig("charts/03_distance_vs_cost.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------
# Chart 4: Correlation heatmap
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.8, 5.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="OrRd", vmin=-1, vmax=1,
            linewidths=0.5, ax=ax, cbar_kws={"label": "Correlation"})
ax.set_title("Correlation Between Key Logistics Variables", fontsize=13, weight="bold")
fig.tight_layout()
fig.savefig("charts/04_correlation_heatmap.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------
# Chart 5: Average shipment volume by zone (bar)
# ---------------------------------------------------------------
zone_volume = df.groupby("zone_id")["shipment_volume"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8.5, 4.5))
sns.barplot(x=zone_volume.index, y=zone_volume.values, color=ACCENT, ax=ax)
ax.set_title("Average Shipment Volume by Delivery Zone", fontsize=13, weight="bold")
ax.set_xlabel("Zone")
ax.set_ylabel("Avg. Shipment Volume (units)")
plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
fig.tight_layout()
fig.savefig("charts/05_volume_by_zone.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------
# Chart 6: Daily transportation cost trend + 7-day rolling average
# ---------------------------------------------------------------
daily = df.groupby("order_date")["transportation_cost"].sum().reset_index()
daily["rolling_7d"] = daily["transportation_cost"].rolling(7).mean()

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.plot(daily["order_date"], daily["transportation_cost"], color="#C9C4BB",
        linewidth=1, label="Daily total cost")
ax.plot(daily["order_date"], daily["rolling_7d"], color=ACCENT, linewidth=2.2,
        label="7-day rolling average")
ax.set_title("Daily Transportation Cost Trend (90 Days)", fontsize=13, weight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Total Transportation Cost (₹)")
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("₹{x:,.0f}"))
fig.autofmt_xdate()
ax.legend()
fig.tight_layout()
fig.savefig("charts/06_daily_cost_trend.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------
# Chart 7: On-time delivery rate by hub (bar, ties back to KPI)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
otdr_sorted = otdr_by_hub.sort_values()
bars = ax.bar(otdr_sorted.index, otdr_sorted.values, color=PALETTE[:len(otdr_sorted)])
ax.axhline(95, color="#1F2937", linestyle="--", linewidth=1, label="Industry benchmark (95%)")
ax.set_title("On-Time Delivery Rate by Hub", fontsize=13, weight="bold")
ax.set_ylabel("OTDR (%)")
ax.set_ylim(0, 100)
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
            f"{b.get_height():.1f}%", ha="center", fontsize=10)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("charts/07_otdr_by_hub.png", dpi=160)
plt.close(fig)

print("All charts saved to charts/")
