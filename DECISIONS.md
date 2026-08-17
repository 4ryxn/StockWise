# Architecture Decisions

## ADR-001: Use M5 and begin with one store

**Decision:** Start with all item series for `CA_1`, not all ten stores.

**Reason:** This retains thousands of related time series while keeping feature generation,
training, and deployment feasible within 25 days. The pipeline should retain a `store_id`
column so expansion remains possible.

## ADR-002: Forecast 28 days at item-day grain

**Decision:** Predict unit demand for each item for the next 28 days.

**Reason:** This matches the dataset's natural evaluation horizon and supports actionable
inventory planning.

## ADR-003: Use a global tabular model

**Decision:** Compare a 7-day seasonal-naive baseline against one global LightGBM model.

**Reason:** A global model can learn across item series and is faster to train and explain
than thousands of separate models. Additional models are out of scope until the full MVP is
deployed.

## ADR-004: Use WAPE as the headline metric

**Decision:** Report WAPE first, with MAE and RMSE as supporting metrics.

**Reason:** WAPE is easy to explain to business users and remains aggregate-volume weighted.
Zero-demand slices must be handled explicitly.

## ADR-005: Inventory outputs are scenario simulations

**Decision:** Accept lead time, service level, on-hand stock, on-order stock, and backorders as
scenario inputs.

**Reason:** M5 contains sales rather than observed inventory positions. Simulated assumptions
must remain visible in the UI and documentation.

## ADR-006: Use familiar deployment tools

**Decision:** Use FastAPI, Streamlit, PostgreSQL/Neon, Docker, GitHub Actions, and Render unless
a concrete deployment blocker appears.

**Reason:** These match existing skills and minimize delivery risk. React, Kafka, Airflow,
Kubernetes, deep learning, and LLM features are excluded from v1.

## ADR-007: Use only known-in-advance features for demand modeling

**Decision:** Build lags and rolling demand statistics from values strictly before the target day,
retain known calendar fields, and exclude `sell_price`.

**Reason:** Shifting before rolling prevents target leakage. Future M5 selling prices are not
known when a forecast is made, so including them would make validation unrealistically optimistic.

## ADR-008: Evaluate LightGBM recursively with a fixed recent-history window

**Decision:** Train one global LightGBM model on the 730 most recent pre-cutoff days per fold and
recursively feed predictions back into the 28-day forecast horizon.

**Reason:** This matches forecast-time information availability and caps training-memory use.

## ADR-009: Use a lost-sales inventory scenario backtest

**Decision:** Simulate 7-day lead/review cycles with 95% target service, initialized from pre-fold
demand history, and compare fixed historical versus forecast-driven order-up-to policies.

**Reason:** M5 has sales rather than inventory positions, receipts, or unmet demand; results must
be labelled as scenario simulations.

## ADR-010: Explore inventory service and safety-stock trade-offs separately

**Decision:** Produce a Pareto frontier over fixed service-level and safety-stock scenarios without
selecting a configuration from outer validation results.

**Reason:** These are business assumptions, not forecasting hyperparameters, and must be chosen
with stakeholders rather than optimized on the held-out folds.
