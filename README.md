# StockWise

StockWise is an end-to-end retail demand forecasting and inventory decision-support
platform. It forecasts item-level demand for the next 28 days and translates forecasts
into transparent reorder recommendations.

## Frozen MVP

- Dataset: M5 Forecasting Accuracy data, initially scoped to store `CA_1`
- Grain: one row per item and day
- Forecast horizon: 28 days
- Models: seasonal-naive baseline and one global LightGBM model
- Validation: rolling-origin time-series validation
- Primary forecast metric: WAPE; secondary metrics: MAE and RMSE
- Inventory output: safety stock, reorder point, and recommended order quantity
- Delivery: FastAPI service, Streamlit dashboard, Docker image, tests, CI, and deployment

Inventory levels and lead times are not present in M5. StockWise therefore treats them as
explicit scenario inputs and clearly labels inventory results as simulations rather than
historical ground truth.

## Current status

The foundation and M5 ingestion pipeline are runnable. The repository contains evaluated
metric functions, a weekly seasonal baseline, rolling-origin split generation, an inventory
policy engine, chunked store-level transformation, Parquet output, and automated data-quality
reporting. Running the pipeline on the complete M5 files is the next milestone.

## Quick start

```bash
cd stockwise
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,data]'
pytest
stockwise demo
```

## Expected raw files

After downloading the M5 dataset, place these files in `data/raw/`:

- `calendar.csv`
- `sell_prices.csv`
- `sales_train_evaluation.csv`

Raw data is intentionally excluded from Git.

Validate and transform the files with:

```bash
stockwise validate-m5 data/raw --store-id CA_1
stockwise build-m5 data/raw data/processed/ca_1 --store-id CA_1
```

The build command intentionally refuses to write into a non-empty output directory. This
protects existing processed data and makes every run explicit.

## Repository map

```text
src/stockwise/       Reusable forecasting, evaluation, and inventory code
tests/               Unit tests
docs/                Frozen scope and technical decisions
data/                 Local data stages (contents ignored by Git)
artifacts/            Generated models and reports (contents ignored by Git)
PROJECT_CONTEXT.md   Compact handoff context for future Codex sessions
DECISIONS.md         Architecture decision log
NEXT_TASKS.md        Ordered implementation queue
```

## Definition of done

The final release must be reproducible from a fresh clone, outperform the seasonal baseline
on held-out dates, expose documented predictions, show business-facing inventory outputs,
pass automated tests, and include a deployed demo plus measured results in this README.
