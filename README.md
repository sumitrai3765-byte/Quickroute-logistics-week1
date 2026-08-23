# QuickRoute Logistics — Week 1: Strategic Planning & Data Exploration

**Yuva Intern (NSDC) — Data Analytics Internship, Week 1 Task**
Author: Sumeet Rai

## Objective

This project simulates the strategic planning and exploration phase of a logistics
data analytics engagement. It defines a realistic last-mile delivery scenario,
identifies key performance indicators (KPIs), and outlines how core data science
techniques — implemented in Python — can be applied to address common logistics
challenges: inefficient routing, poor inventory visibility, and unpredictable
delivery performance.

The full strategic planning report (background research, roadmap, and
conclusions) is in `Week1_Strategic_Planning_Logistics_SumeetRai.docx`. This
repo contains the accompanying Python code referenced in that report.

## Scenario

**QuickRoute Logistics** (illustrative) runs a last-mile delivery network for an
e-commerce/quick-commerce client base across Delhi NCR (Delhi, Gurugram, Noida,
Faridabad, Ghaziabad), operating three regional fulfillment hubs and serving
8,000–12,000 shipments/day. The business faces rising delivery costs,
inconsistent on-time performance during peak hours, and inventory imbalance
across hubs.

## KPIs Tracked

| KPI | Definition |
|---|---|
| On-Time Delivery Rate (OTDR) | % of orders delivered within the promised window |
| Average Cost per Delivery | Total last-mile operating cost ÷ orders delivered |
| Inventory Turnover Ratio | Cost of goods dispatched ÷ average inventory value, per hub |
| Stockout Rate | % of SKU-hub-days with zero available stock against demand |
| Order Fulfillment Cycle Time | Hours from order placement to hub dispatch |

## Repository Structure

```
quickroute-logistics-week1/
├── README.md
├── requirements.txt
├── data/
│   └── README.md          # notes on expected input schema / data sources
└── src/
    ├── 01_data_cleaning.py
    ├── 02_kpi_baseline.py
    ├── 03_zone_clustering.py
    ├── 04_delivery_time_regression.py
    ├── 05_route_optimization.py
    └── 06_stockout_classification.py
```

Each script in `src/` corresponds to one phase of the strategic roadmap and one
technique from the report (regression, clustering, optimization,
classification). Scripts are written to run against a cleaned orders dataset
(see `data/README.md` for the expected schema) and are structured to run
independently or be imported as a small pipeline.

## Roadmap (5 Phases)

1. **Data Collection** — order, delivery, inventory, and geolocation data
2. **Data Cleaning** — `src/01_data_cleaning.py`
3. **Exploratory Analysis (EDA)** — `src/02_kpi_baseline.py`
4. **Predictive Modeling** — `src/03_zone_clustering.py`, `04_delivery_time_regression.py`, `05_route_optimization.py`, `06_stockout_classification.py`
5. **Deployment / Recommendation** — hub-level and dispatch-level recommendations (see report, Section 6)

## Data Sources (Public, for Prototyping)

- Kaggle — Amazon Delivery Dataset / Logistics & Supply Chain datasets
- Kaggle — DataCo Smart Supply Chain dataset
- Open Government Data (OGD) Platform India
- OpenStreetMap (via OSMnx) for Delhi NCR road-network data

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Status

Week 1 deliverable: strategic plan + illustrative code. Data acquisition and a
working EDA notebook against a chosen public dataset are planned for Week 2.
