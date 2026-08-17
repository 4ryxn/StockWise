"""Validation and memory-conscious transformation for the M5 retail dataset."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

SALES_FILE = "sales_train_evaluation.csv"
CALENDAR_FILE = "calendar.csv"
PRICES_FILE = "sell_prices.csv"

SALES_ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CALENDAR_COLUMNS = [
    "date",
    "wm_yr_wk",
    "weekday",
    "wday",
    "month",
    "year",
    "d",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
]
PRICE_COLUMNS = ["store_id", "item_id", "wm_yr_wk", "sell_price"]


@dataclass(frozen=True)
class M5ValidationReport:
    raw_directory: str
    store_id: str
    day_column_count: int
    first_day: str
    last_day: str
    store_series_count: int


@dataclass(frozen=True)
class M5BuildReport:
    store_id: str
    output_directory: str
    partition_count: int
    row_count: int
    item_count: int
    minimum_date: str
    maximum_date: str
    missing_price_rows: int
    missing_price_rate: float


def _required_paths(raw_directory: Path) -> dict[str, Path]:
    return {
        SALES_FILE: raw_directory / SALES_FILE,
        CALENDAR_FILE: raw_directory / CALENDAR_FILE,
        PRICES_FILE: raw_directory / PRICES_FILE,
    }


def _require_columns(actual: list[str], required: list[str], filename: str) -> None:
    missing = sorted(set(required) - set(actual))
    if missing:
        raise ValueError(f"{filename} is missing required columns: {', '.join(missing)}")


def _ordered_day_columns(columns: list[str]) -> list[str]:
    day_columns = [column for column in columns if column.startswith("d_")]
    try:
        ordered = sorted(day_columns, key=lambda column: int(column.removeprefix("d_")))
    except ValueError as exc:
        raise ValueError("sales day columns must use the form d_<integer>") from exc
    if not ordered:
        raise ValueError("sales file does not contain any d_<integer> demand columns")

    day_numbers = [int(column.removeprefix("d_")) for column in ordered]
    if day_numbers != list(range(day_numbers[0], day_numbers[-1] + 1)):
        raise ValueError("sales demand columns must form a consecutive day sequence")
    return ordered


def validate_m5_raw_files(
    raw_directory: str | Path,
    store_id: str = "CA_1",
) -> M5ValidationReport:
    """Validate required files, schemas, day coverage, and target-store availability."""
    raw_path = Path(raw_directory)
    paths = _required_paths(raw_path)
    missing_files = [name for name, path in paths.items() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing M5 files in {raw_path}: {', '.join(sorted(missing_files))}"
        )

    sales_columns = pd.read_csv(paths[SALES_FILE], nrows=0).columns.tolist()
    calendar_columns = pd.read_csv(paths[CALENDAR_FILE], nrows=0).columns.tolist()
    price_columns = pd.read_csv(paths[PRICES_FILE], nrows=0).columns.tolist()
    _require_columns(sales_columns, SALES_ID_COLUMNS, SALES_FILE)
    _require_columns(calendar_columns, CALENDAR_COLUMNS, CALENDAR_FILE)
    _require_columns(price_columns, PRICE_COLUMNS, PRICES_FILE)

    day_columns = _ordered_day_columns(sales_columns)
    available_calendar_days = set(pd.read_csv(paths[CALENDAR_FILE], usecols=["d"])["d"])
    missing_calendar_days = [day for day in day_columns if day not in available_calendar_days]
    if missing_calendar_days:
        preview = ", ".join(missing_calendar_days[:5])
        raise ValueError(f"calendar.csv does not cover sales days: {preview}")

    stores = pd.read_csv(paths[SALES_FILE], usecols=["store_id"])["store_id"]
    store_series_count = int((stores == store_id).sum())
    if store_series_count == 0:
        available = ", ".join(sorted(stores.dropna().astype(str).unique()))
        raise ValueError(f"store_id {store_id!r} not found; available stores: {available}")

    return M5ValidationReport(
        raw_directory=str(raw_path.resolve()),
        store_id=store_id,
        day_column_count=len(day_columns),
        first_day=day_columns[0],
        last_day=day_columns[-1],
        store_series_count=store_series_count,
    )


def _load_store_prices(path: Path, store_id: str, chunk_size: int = 500_000) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=PRICE_COLUMNS, chunksize=chunk_size):
        selected = chunk.loc[chunk["store_id"] == store_id]
        if not selected.empty:
            chunks.append(selected)
    if not chunks:
        raise ValueError(f"No selling-price records found for store_id {store_id!r}")
    return pd.concat(chunks, ignore_index=True)


def transform_sales_chunk(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Convert a sales chunk to item-day grain and join calendar and price features."""
    _require_columns(sales.columns.tolist(), SALES_ID_COLUMNS, SALES_FILE)
    _require_columns(calendar.columns.tolist(), CALENDAR_COLUMNS, CALENDAR_FILE)
    _require_columns(prices.columns.tolist(), PRICE_COLUMNS, PRICES_FILE)
    day_columns = _ordered_day_columns(sales.columns.tolist())

    long_sales = sales.melt(
        id_vars=SALES_ID_COLUMNS,
        value_vars=day_columns,
        var_name="d",
        value_name="units_sold",
    )
    enriched = long_sales.merge(calendar[CALENDAR_COLUMNS], on="d", how="left", validate="m:1")
    enriched = enriched.merge(
        prices[PRICE_COLUMNS],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="m:1",
    )
    enriched["date"] = pd.to_datetime(enriched["date"], errors="raise")
    enriched["day_index"] = enriched["d"].str.removeprefix("d_").astype("int16")
    enriched["units_sold"] = pd.to_numeric(enriched["units_sold"], errors="raise").astype(
        "int32"
    )
    enriched["sell_price"] = enriched["sell_price"].astype("float32")
    enriched["is_event"] = enriched[["event_name_1", "event_name_2"]].notna().any(axis=1)
    enriched = enriched.sort_values(["item_id", "date"], kind="stable").reset_index(drop=True)
    return enriched


def build_store_dataset(
    raw_directory: str | Path,
    output_directory: str | Path,
    store_id: str = "CA_1",
    item_chunk_size: int = 250,
) -> M5BuildReport:
    """Build partitioned Parquet files for one store without loading all sales at once."""
    if item_chunk_size <= 0:
        raise ValueError("item_chunk_size must be positive")

    raw_path = Path(raw_directory)
    output_path = Path(output_directory)
    validate_m5_raw_files(raw_path, store_id)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    calendar = pd.read_csv(raw_path / CALENDAR_FILE, usecols=CALENDAR_COLUMNS)
    prices = _load_store_prices(raw_path / PRICES_FILE, store_id)

    row_count = 0
    missing_price_rows = 0
    item_ids: set[str] = set()
    minimum_date: pd.Timestamp | None = None
    maximum_date: pd.Timestamp | None = None
    partition_count = 0

    sales_reader = pd.read_csv(raw_path / SALES_FILE, chunksize=item_chunk_size)
    for sales_chunk in sales_reader:
        store_chunk = sales_chunk.loc[sales_chunk["store_id"] == store_id]
        if store_chunk.empty:
            continue
        transformed = transform_sales_chunk(store_chunk, calendar, prices)
        partition_file = output_path / f"part-{partition_count:04d}.parquet"
        try:
            transformed.to_parquet(partition_file, index=False)
        except ImportError as exc:
            raise RuntimeError(
                "Parquet support is missing; install StockWise with the data extra: "
                "python -m pip install -e '.[data]'"
            ) from exc

        partition_count += 1
        row_count += len(transformed)
        missing_price_rows += int(transformed["sell_price"].isna().sum())
        item_ids.update(transformed["item_id"].astype(str).unique())
        chunk_minimum = transformed["date"].min()
        chunk_maximum = transformed["date"].max()
        minimum_date = chunk_minimum if minimum_date is None else min(minimum_date, chunk_minimum)
        maximum_date = chunk_maximum if maximum_date is None else max(maximum_date, chunk_maximum)

    if partition_count == 0 or minimum_date is None or maximum_date is None:
        raise RuntimeError(f"No partitions were created for store_id {store_id!r}")

    report = M5BuildReport(
        store_id=store_id,
        output_directory=str(output_path.resolve()),
        partition_count=partition_count,
        row_count=row_count,
        item_count=len(item_ids),
        minimum_date=minimum_date.date().isoformat(),
        maximum_date=maximum_date.date().isoformat(),
        missing_price_rows=missing_price_rows,
        missing_price_rate=missing_price_rows / row_count,
    )
    (output_path / "data_quality_report.json").write_text(
        json.dumps(asdict(report), indent=2) + "\n",
        encoding="utf-8",
    )
    return report

