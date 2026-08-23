# Expected Data Schema

Place a cleaned orders export here as `orders.csv` (not committed — see
`.gitignore`). Scripts in `src/` expect the following columns:

| Column | Type | Description |
|---|---|---|
| `order_id` | str | Unique order identifier |
| `hub_id` | str | Fulfillment hub that served the order |
| `sku_id` | str | Product SKU |
| `order_ts` | datetime | Timestamp the order was placed |
| `dispatch_ts` | datetime | Timestamp the order left the hub |
| `delivered_ts` | datetime | Timestamp the order was delivered |
| `promised_ts` | datetime | SLA-promised delivery timestamp |
| `distance_km` | float | Straight-line or road distance, hub to customer |
| `lat` / `lon` | float | Delivery point coordinates |
| `hub_load` | float | Concurrent order load at the hub at dispatch time |
| `hour_of_day` | int | Hour the order was dispatched (0–23) |
| `is_weekend` | bool | Whether dispatch occurred on a weekend |

For inventory/stockout modeling (`06_stockout_classification.py`), a second
table `sku_hub_daily.csv` is expected with: `sku_id`, `hub_id`, `date`,
`avg_daily_sales`, `current_stock`, `lead_time_days`, `sales_volatility`,
`stockout_next_7d`.

## Suggested Public Sources for Prototyping

- Kaggle — Amazon Delivery Dataset
- Kaggle — DataCo Smart Supply Chain Dataset
- OpenStreetMap (via OSMnx) — Delhi NCR road network, for `distance_km`
