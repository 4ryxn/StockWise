# Next Tasks

## Milestone 1 - Data foundation (Days 1-3)

- [x] Freeze MVP, metrics, assumptions, and architecture decisions.
- [x] Create package scaffold and core unit tests.
- [x] Add an M5 raw-file validator with helpful error messages.
- [x] Transform `CA_1` sales from wide to long format in memory-safe chunks.
- [x] Join calendar and selling-price features.
- [x] Write processed Parquet partitions and a compact data-quality report.
- [x] Add synthetic fixtures and integration tests for the data pipeline.
- [x] Run the pipeline on the complete M5 files and review the quality report.

## Milestone 2 - Analysis and forecasting (Days 4-12)

- [x] Create reproducible EDA outputs and business KPIs.
- [x] Evaluate the seasonal-naive baseline on rolling origins.
- [x] Build leakage-safe lag and rolling-window features.
- [x] Train one global LightGBM model and track experiments.
- [x] Compare models overall and by department/category.

## Milestone 3 - Inventory decisions (Days 13-16)

- [x] Convert forecast distributions into scenario inputs.
- [x] Backtest the reorder policy against a fixed-rule policy.
- [x] Report stockout, holding-cost, and service-level tradeoffs.

## Milestone 4 - Product and deployment (Days 17-25)

- [ ] Build FastAPI endpoints and contract tests.
- [ ] Build a business-facing Streamlit dashboard.
- [ ] Add PostgreSQL persistence where it creates user value.
- [ ] Add Docker, CI, deployment, monitoring, and smoke tests.
- [ ] Finish README results, screenshots, demo video, and resume bullets.
