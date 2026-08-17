# StockWise Project Context

## Goal

Build a deployed final-year project in 25 days that strengthens Aryan Singhal's portfolio for
data analyst, data scientist, and ML engineer roles.

## Existing strengths to reuse

Python, SQL, PostgreSQL, Pandas, NumPy, scikit-learn, Streamlit, Plotly, Power BI,
FastAPI, SQLAlchemy, Docker, GitHub Actions, Render, Vercel, and Neon.

## Portfolio gap this project addresses

Rigorous non-LLM modeling, time-series validation, experiment comparison, business
optimization, and model monitoring. Do not turn this project into another RAG application.

## Hard constraints

- Final deployed version in 25 days.
- ChatGPT Work and Codex in VS Code share a limited weekly allowance.
- Prefer bundled, testable tasks and durable written handoffs.
- Favor a reliable MVP over additional infrastructure.

## Frozen v1 scope

- M5 dataset, store `CA_1`
- 28-day item-level demand forecasts
- Seasonal-naive baseline plus one global LightGBM model
- Rolling-origin validation; WAPE is the primary metric
- Scenario-based safety stock, reorder point, and order recommendation
- FastAPI, Streamlit, PostgreSQL, Docker, GitHub Actions, and deployment

## Rules for future coding sessions

1. Read this file, `DECISIONS.md`, and `NEXT_TASKS.md` first.
2. Work on the first unchecked task unless explicitly redirected.
3. Keep raw M5 files out of Git.
4. Add or update tests with every behavior change.
5. Run the relevant tests and lint checks before handing off.
6. Update `NEXT_TASKS.md` and this context when scope or status changes.
7. Never claim inventory results are observed; M5 inventory inputs are simulated.

## Current status

The complete M5 `CA_1` pipeline has run successfully, producing 5,918,109 item-day rows and a
quality report. Reproducible EDA artifacts, the fixed three-fold 7-day seasonal-naive,
28-day-horizon baseline backtest, leakage-safe demand features, and the fixed recursive global
LightGBM backtest and a lost-sales inventory scenario simulation are complete. Inventory results
are simulations, not observed M5 inventory outcomes. Next, report service/holding trade-offs.
