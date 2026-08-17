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

