# StockWise static dashboard

Generate data: `stockwise export-dashboard-data artifacts frontend/public/data`.

This also writes `public/data/planner.json`: a compact, deterministic sample of 30 CA_1 items
(the ten highest positive fold-3 forecast-volume items in each category). It includes only each
selected item's 28 held-out recursive forecasts and pre-fold demand variability, never raw M5
files, processed Parquet data, or the full prediction-level output.

Then run `npm install` and `npm run dev`; use `npm run build` for a static production build. No backend, database, or API is required.
